"""
创新模块2: 频域-空域联合学习
- 频域分解捕获周期性纹理变化
- 高频分量增强边缘检测
- 低频分量提供语义一致性
"""
import torch
import torch.nn as nn
import torch.nn. functional as F
import math


class FrequencySpatialFusion(nn.Module):
    """
    频域-空域联合学习模块
    
    核心创新:
    1. DCT频域分解，分离高低频信息
    2. 频域注意力，自适应加权不同频率成分
    3.  空域-频域交叉融合
    """
    def __init__(self, channels, freq_groups=4):
        super().__init__()
        self.channels = channels
        self.freq_groups = freq_groups
        
        # 空域到频域投影
        self. spatial_to_freq = nn. Sequential(
            nn.Conv2d(channels, channels, 1),
            nn. GroupNorm(8, channels),
            nn. GELU()
        )
        
        # 频域注意力（按频率分组）
        self.freq_attention = nn.ModuleList([
            nn.Sequential(
                nn. AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, 1),
                nn.Sigmoid()
            ) for _ in range(freq_groups)
        ])
        
        # 高频增强模块
        self. high_freq_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn. Conv2d(channels, channels, 1),
            nn. GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.Conv2d(channels, channels, 1)
        )
        
        # 低频语义模块
        self. low_freq_semantic = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),  # 降采样提取语义
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        
        # 频域-空域融合
        self. fusion = nn.Sequential(
            nn. Conv2d(channels * 3, channels, 1),  # 空域+高频+低频
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        
        # 可学习的频率权重
        self.freq_weights = nn.Parameter(torch.ones(freq_groups) / freq_groups)
        
        # 预计算DCT基
        self.register_buffer('dct_basis', None)
    
    def _get_dct_basis(self, H, W, device):
        """生成2D DCT基"""
        if self.dct_basis is not None and self.dct_basis.shape[-2:] == (H, W):
            return self.dct_basis
        
        # 1D DCT基
        n = torch.arange(H, dtype=torch.float32, device=device)
        k = torch.arange(H, dtype=torch. float32, device=device)
        dct_h = torch.cos(math.pi * (2 * n. unsqueeze(0) + 1) * k.unsqueeze(1) / (2 * H))
        dct_h[0] *= 1 / math.sqrt(2)
        dct_h *= math.sqrt(2 / H)
        
        m = torch.arange(W, dtype=torch.float32, device=device)
        l = torch.arange(W, dtype=torch.float32, device=device)
        dct_w = torch.cos(math.pi * (2 * m. unsqueeze(0) + 1) * l.unsqueeze(1) / (2 * W))
        dct_w[0] *= 1 / math.sqrt(2)
        dct_w *= math. sqrt(2 / W)
        
        self.dct_basis = (dct_h, dct_w)
        return self. dct_basis
    
    def dct2d(self, x):
        """2D DCT变换"""
        B, C, H, W = x.shape
        dct_h, dct_w = self._get_dct_basis(H, W, x.device)
        
        # 先对行做DCT
        x = torch.matmul(dct_h, x)
        # 再对列做DCT
        x = torch.matmul(x, dct_w. T)
        return x
    
    def idct2d(self, x):
        """2D IDCT变换"""
        B, C, H, W = x.shape
        dct_h, dct_w = self._get_dct_basis(H, W, x.device)
        
        # IDCT = DCT^T
        x = torch. matmul(dct_h.T, x)
        x = torch.matmul(x, dct_w)
        return x
    
    def _split_freq_groups(self, freq_map, H, W):
        """将频谱按频率分组"""
        groups = []
        step_h = H // self. freq_groups
        step_w = W // self.freq_groups
        
        for i in range(self.freq_groups):
            mask = torch.zeros(1, 1, H, W, device=freq_map.device)
            # 对角带状分组
            for h in range(H):
                for w in range(W):
                    dist = (h + w) / (H + W - 2)  # 归一化距离
                    if i / self.freq_groups <= dist < (i + 1) / self.freq_groups:
                        mask[0, 0, h, w] = 1
            groups.append(freq_map * mask)
        
        return groups
    
    def forward(self, x1, x2):
        """
        Args:
            x1: 时相1特征 [B, C, H, W]
            x2: 时相2特征 [B, C, H, W]
        Returns:
            变化增强特征 [B, C, H, W]
        """
        B, C, H, W = x1. shape
        
        # 计算差异
        diff = x1 - x2
        
        # 空域特征
        spatial_feat = self.spatial_to_freq(diff)
        
        # 频域分析
        freq_map = self. dct2d(diff)
        
        # 频率分组加权
        freq_weights = F.softmax(self.freq_weights, dim=0)
        freq_groups = self._split_freq_groups(freq_map, H, W)
        
        weighted_freq = torch.zeros_like(freq_map)
        for i, (fg, attn) in enumerate(zip(freq_groups, self.freq_attention)):
            # 对每个频率组应用注意力
            fg_spatial = self.idct2d(fg)  # 转回空域计算注意力
            weight = attn(fg_spatial)
            weighted_freq = weighted_freq + freq_weights[i] * fg * weight. mean(dim=(2, 3), keepdim=True)
        
        # 高频增强（边缘和纹理）
        high_freq_mask = torch.zeros(1, 1, H, W, device=x1.device)
        high_freq_mask[:, :, H//2:, :] = 1
        high_freq_mask[:, :, :, W//2:] = 1
        high_freq = self.idct2d(freq_map * high_freq_mask)
        high_freq = self. high_freq_enhance(high_freq)
        
        # 低频语义（全局结构）
        low_freq_mask = 1 - high_freq_mask
        low_freq = self.idct2d(freq_map * low_freq_mask)
        low_freq = self.low_freq_semantic(low_freq)
        low_freq = F.interpolate(low_freq, (H, W), mode='bilinear', align_corners=True)
        
        # 融合
        fused = self.fusion(torch.cat([spatial_feat, high_freq, low_freq], dim=1))
        
        return fused + diff  # 残差连接


class AdaptiveFrequencyFilter(nn.Module):
    """
    自适应频率滤波器
    
    创新点：
    1. 学习场景自适应的频率滤波器
    2. 根据变化类型（建筑/植被/水体）调整滤波策略
    """
    def __init__(self, channels, n_filters=8):
        super().__init__()
        self.n_filters = n_filters
        
        # 滤波器库（可学习）
        self.filter_bank = nn. Parameter(torch.randn(n_filters, 1, 7, 7) * 0.02)
        
        # 滤波器选择网络
        self. filter_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn. Flatten(),
            nn.Linear(channels, n_filters),
            nn. Softmax(dim=-1)
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn. Conv2d(channels, channels, 1),
            nn. GroupNorm(8, channels),
            nn.GELU()
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 选择滤波器权重
        weights = self.filter_selector(x)  # [B, n_filters]
        
        # 组合滤波器
        combined_filter = torch.zeros(1, 1, 7, 7, device=x.device)
        for i in range(self. n_filters):
            combined_filter = combined_filter + weights[:, i:i+1, None, None] * self.filter_bank[i:i+1]
        
        # 应用滤波（分组卷积）
        x_reshaped = x. view(B * C, 1, H, W)
        filtered = F.conv2d(x_reshaped, combined_filter. expand(B, -1, -1, -1). reshape(-1, 1, 7, 7), 
                           padding=3, groups=B)
        filtered = filtered.view(B, C, H, W)
        
        return self.output_proj(filtered) + x