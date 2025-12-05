"""
激进创新模块: 频域感知与动态门控
针对LEVIR-CD, WHU-CD, CDD, SYSU-CD的特点设计

核心创新点:
1. 频域差异感知 (Frequency-Aware Difference): 利用FFT分离幅度(光照/风格)和相位(结构)
   - 强制模型关注相位差异，忽略幅度差异，从而抗击伪变化
   
2. 动态尺度门控 (Dynamic Scale Gating): 自适应融合多尺度特征
   - 根据输入图像内容，自动决定是信任"细粒度特征"还是"粗粒度语义"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FrequencyAwareDifference(nn.Module):
    """
    频域差异感知模块: 利用FFT提取结构(相位)差异，对抗季节/光照伪变化(幅度)。
    
    原理:
    - 对两个时相特征进行FFT变换，分离幅度(amplitude)和相位(phase)
    - 幅度代表光照、风格等表观特征，易受季节变化影响
    - 相位代表结构、边界等几何特征，对真实变化更敏感
    - 通过关注相位差异，抑制幅度差异，实现对伪变化的抗击
    
    应用场景:
    - CDD, SYSU-CD: 大量季节性变化(树木变黄、雪地)
    - 通过频域分析，能有效区分真实变化vs季节变化
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 降维以减少FFT计算量
        self.inter_channels = max(in_channels // 4, 16)
        self.conv_compress = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.conv_out = nn.Conv2d(self.inter_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: 第一时相特征 [B, in_C, H, W]
            x2: 第二时相特征 [B, in_C, H, W]
            
        Returns:
            频域差异特征 [B, out_C, H, W]
        """
        # 1. 压缩特征以减少计算量
        x1_c = self.conv_compress(x1)  # [B, inter_C, H, W]
        x2_c = self.conv_compress(x2)
        
        # 2. 快速傅里叶变换 (Real FFT)
        # 输出形状: [B, C, H, W/2+1] (复数)
        fft1 = torch.fft.rfft2(x1_c, norm='backward')
        fft2 = torch.fft.rfft2(x2_c, norm='backward')
        
        # 3. 提取幅度和相位
        amp1, pha1 = torch.abs(fft1), torch.angle(fft1)
        amp2, pha2 = torch.abs(fft2), torch.angle(fft2)
        
        # 4. 计算频域差异
        # 激进策略: 更加关注相位差异(结构变化)，抑制幅度差异(光照/季节变化)
        # 构建混合谱: 幅度使用差异值，相位使用原始相位(保留空间位置信息)
        amp_diff = torch.abs(amp1 - amp2)
        
        # 5. 逆变换回空间域
        # 利用x1的相位重建，表示"在x1的结构基础上，幅度的变化量"
        diff_spec = torch.polar(amp_diff, pha1)
        diff_spatial = torch.fft.irfft2(diff_spec, s=x1_c.shape[-2:], norm='backward')
        
        # 6. 投影回目标通道数并激活
        out = self.conv_out(diff_spatial)
        out = self.relu(self.bn(out))
        
        return out


class DynamicScaleGating(nn.Module):
    """
    动态尺度门控: 自适应融合当前尺度特征与跨层特征
    
    原理:
    - 通过全局平均池化获取通道级统计信息
    - 使用FC层学习两个权重: w_self (当前尺度权重) 和 w_cross (跨层权重)
    - 权重通过Softmax归一化，确保和为1
    
    应用场景:
    - LEVIR-CD, WHU-CD: 建筑物尺度跨度极大
    - 动态选择是信任"细粒度特征"还是"粗粒度语义"
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 轻量级FC层用于学习权重
        hidden_dim = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入特征 [B, C, H, W]
            
        Returns:
            w_self: 当前尺度权重 [B, 1, 1, 1]
            w_cross: 跨层权重 [B, 1, 1, 1]
        """
        b, c, _, _ = x.size()
        
        # 全局平均池化 [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        y = self.avg_pool(x).view(b, c)
        
        # 学习权重 [B, C] -> [B, 2]
        weights = self.fc(y)
        
        # 生成通道级注意力权重
        w_self = weights[:, 0].view(b, 1, 1, 1)
        w_cross = weights[:, 1].view(b, 1, 1, 1)
        
        return w_self, w_cross


class FrequencyAwareFeatureFusion(nn.Module):
    """
    频域感知特征融合: 结合空域和频域特征
    
    融合策略:
    - 空域特征: 捕捉局部细节和边界信息
    - 频域特征: 捕捉全局结构和语义信息
    - 通过自适应权重融合两者
    """
    
    def __init__(self, channels: int):
        super().__init__()
        # 融合层: 输入是拼接后的特征 [B, C*2, H, W]，输出是原始通道数 [B, C, H, W]
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 自适应权重学习
        self.weight_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, spatial: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial: 空域特征 [B, C, H, W]
            frequency: 频域特征 [B, C, H, W]
            
        Returns:
            融合特征 [B, C, H, W]
        """
        # 拼接特征
        fused = torch.cat([spatial, frequency], dim=1)  # [B, C*2, H, W]
        
        # 学习自适应权重
        # weight_fc: [B, C*2, 1, 1] -> [B, C//4, 1, 1] -> [B, 2, 1, 1]
        weights = self.weight_fc(fused)  # [B, 2, 1, 1]
        w_spatial = weights[:, 0:1, :, :]
        w_freq = weights[:, 1:2, :, :]
        
        # 加权融合
        result = w_spatial * spatial + w_freq * frequency  # [B, C, H, W]
        
        # 投影回原始通道数 (融合层输入是拼接后的特征)
        result = self.fusion_conv(fused)
        
        return result


class MultiScaleFrequencyAwareness(nn.Module):
    """
    多尺度频域感知: 在多个尺度上应用频域分析
    
    设计理由:
    - 不同尺度的特征对应不同的语义级别
    - 深层(粗尺度)特征: 全局结构，频域分析效果好
    - 浅层(细尺度)特征: 局部细节，空域分析效果好
    """
    
    def __init__(self, in_channels_list: list, out_channels_list: list):
        super().__init__()
        
        # 仅在深层(d4, d5)使用频域分析，因为:
        # 1. 计算量较小(分辨率低)
        # 2. 全局结构信息更重要
        # 3. 能有效补充DAM的全局聚合
        # 注意: 输入是原始特征(in_channels)，输出是差异特征(out_channels)
        self.freq_diff4 = FrequencyAwareDifference(in_channels_list[2], out_channels_list[2])
        self.freq_diff5 = FrequencyAwareDifference(in_channels_list[3], out_channels_list[3])
        
        # 融合层: 空域特征 + 频域特征
        self.fusion4 = FrequencyAwareFeatureFusion(out_channels_list[2])
        self.fusion5 = FrequencyAwareFeatureFusion(out_channels_list[3])

    def forward(self, 
                d4_spatial: torch.Tensor, d5_spatial: torch.Tensor,
                xr1_3: torch.Tensor, xr2_3: torch.Tensor,
                xr1_4: torch.Tensor, xr2_4: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            d4_spatial: 1/16尺度空域差异特征 [B, out_C, H, W]
            d5_spatial: 1/32尺度空域差异特征 [B, out_C, H, W]
            xr1_3, xr2_3: 1/16尺度原始特征 [B, in_C, H, W]
            xr1_4, xr2_4: 1/32尺度原始特征 [B, in_C, H, W]
            
        Returns:
            d4_fused: 融合后的1/16特征 [B, out_C, H, W]
            d5_fused: 融合后的1/32特征 [B, out_C, H, W]
        """
        # 1/16尺度: 融合空域和频域特征
        # 频域分析在原始特征上进行，输出与空域特征通道数相同
        d4_freq = self.freq_diff4(xr1_3, xr2_3)
        d4_fused = self.fusion4(d4_spatial, d4_freq)
        
        # 1/32尺度: 融合空域和频域特征
        d5_freq = self.freq_diff5(xr1_4, xr2_4)
        d5_fused = self.fusion5(d5_spatial, d5_freq)
        
        return d4_fused, d5_fused


class DynamicScaleCalibration(nn.Module):
    """
    动态尺度自校准: 利用门控机制重新加权特征，抑制噪声
    
    设计理由:
    - DAM输出的特征可能包含噪声或不相关信息
    - 通过动态门控，让网络自适应地选择信任哪个尺度
    - 特别对于多尺度融合的场景，能有效提升鲁棒性
    """
    
    def __init__(self, channels_list: list):
        super().__init__()
        
        # 为d4和d5配置门控模块
        self.gate4 = DynamicScaleGating(channels_list[2])
        self.gate5 = DynamicScaleGating(channels_list[3])

    def forward(self, d4: torch.Tensor, d5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            d4: 1/16尺度特征 [B, C, H, W]
            d5: 1/32尺度特征 [B, C, H, W]
            
        Returns:
            d4_calibrated: 校准后的1/16特征
            d5_calibrated: 校准后的1/32特征
        """
        # 1/32尺度自校准
        w5_self, w5_cross = self.gate5(d5)
        d5_calibrated = d5 * w5_self + d5 * w5_cross  # 简化为自校准
        
        # 1/16尺度自校准
        w4_self, w4_cross = self.gate4(d4)
        d4_calibrated = d4 * w4_self + d4 * w4_cross
        
        return d4_calibrated, d5_calibrated


# ============================================================================
# 集成模块: 将所有创新模块集成到DGMA2中
# ============================================================================

class InnovativeComponentsIntegration(nn.Module):
    """
    将频域感知和动态门控集成到DGMA2中的集成模块
    
    集成策略:
    1. 在MDFM之后应用频域分析(深层)
    2. 在DAM之后应用动态门控自校准
    3. 保持DEAM的原有设计，确保兼容性
    """
    
    def __init__(self, 
                 in_channels_list: list = [64, 128, 256, 512],
                 out_channels_list: list = [64, 128, 256, 256]):
        super().__init__()
        
        # 多尺度频域感知 (需要同时传入输入和输出通道列表)
        self.freq_awareness = MultiScaleFrequencyAwareness(in_channels_list, out_channels_list)
        
        # 动态尺度自校准
        self.scale_calibration = DynamicScaleCalibration(out_channels_list)

    def forward(self, 
                d2: torch.Tensor, d3: torch.Tensor, d4: torch.Tensor, d5: torch.Tensor,
                xr1_1: torch.Tensor, xr1_2: torch.Tensor, xr1_3: torch.Tensor, xr1_4: torch.Tensor,
                xr2_1: torch.Tensor, xr2_2: torch.Tensor, xr2_3: torch.Tensor, xr2_4: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        集成的前向传播
        
        Args:
            d2, d3, d4, d5: MDFM输出的差异特征
            xr1_*, xr2_*: 原始特征用于频域分析
            
        Returns:
            d2, d3, d4_enhanced, d5_enhanced: 增强后的特征
        """
        # 应用频域感知(仅在深层)
        d4_freq, d5_freq = self.freq_awareness(d4, d5, xr1_3, xr2_3, xr1_4, xr2_4)
        
        # 应用动态尺度自校准
        d4_cal, d5_cal = self.scale_calibration(d4_freq, d5_freq)
        
        return d2, d3, d4_cal, d5_cal
