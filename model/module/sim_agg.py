import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence

# ==========================================
# 1. 基础注意力模块 (保持不变)
# ==========================================

class Spatial_Attention(nn.Module):
    def __init__(self, spatial_kernel=7):
        super(Spatial_Attention, self).__init__()
        self.mlp = nn.Sequential(nn.Conv2d(3, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False),
                                 nn.Sigmoid())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        merge = avg_out + max_out
        return x * self.mlp(torch.concat([merge, avg_out, max_out], dim=1))


class Channel_Attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Channel_Attention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(channel // reduction, channel, 1, bias=False),
                                 nn.Sigmoid())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        p = self.pool(x)
        return x * self.mlp(p)

# ==========================================
# 2. 新增核心组件: PKI & ConvModule
# ==========================================

def autopad(kernel_size: int, padding: Optional[int] = None, dilation: int = 1) -> int:
    """根据卷积核大小和扩张率计算填充大小。"""
    if padding is None:
        padding = (kernel_size - 1) * dilation // 2
    return padding

def make_divisible(value: int, divisor: int = 8) -> int:
    """将值调整为可被指定除数整除。"""
    return int((value + divisor // 2) // divisor * divisor)

class ConvModule(nn.Module):
    """标准卷积模块封装"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None):
        super().__init__()
        layers = []
        # 卷积层
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=(norm_cfg is None)))
        # 归一化层
        if norm_cfg:
            layers.append(self._get_norm_layer(out_channels, norm_cfg))
        # 激活层
        if act_cfg:
            layers.append(self._get_act_layer(act_cfg))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        return nn.BatchNorm2d(num_features)

    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        return nn.ReLU(inplace=True)

class Poly_Kernel_Inception_Block(nn.Module):
    """
    多核 Inception 模块 (PKI Block)
    结合了: 点卷积(1x1) + 多尺度深度卷积(DW-Conv) + 残差连接
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 1.0,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        # 1. 预处理投影 (Pointwise Conv)
        self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # 2. 多尺度深度可分离卷积 (Depthwise Conv with different kernels)
        self.dw_convs = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            self.dw_convs.append(
                ConvModule(hidden_channels, hidden_channels, k, 1,
                           autopad(k, None, d),
                           dilation=d, groups=hidden_channels,
                           norm_cfg=None, act_cfg=None) 
            )

        # 3. 点卷积融合 (Pointwise Conv)
        self.pw_conv = ConvModule(hidden_channels, hidden_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        # 4. 后处理投影 (Pointwise Conv)
        self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.pre_conv(x)
        
        # 多核特征累加
        x_sum = x 
        for conv in self.dw_convs:
            x_sum = x_sum + conv(x)
            
        x = self.pw_conv(x_sum)
        x = self.post_conv(x)
        return x

# ==========================================
# 3. 重构后的特征融合模块
# ==========================================

class FeatureFusionModule(nn.Module):
    def __init__(self, dim):
        super(FeatureFusionModule, self).__init__()

        # 1. 降维聚合 (4*dim -> dim)
        self.proj_in = ConvModule(
            dim * 4, dim, 1, 
            norm_cfg=dict(type='BN'), 
            act_cfg=dict(type='SiLU') 
        )

        # 2. 核心创新：多核 Inception 模块 (PKI)
        self.pki_block = Poly_Kernel_Inception_Block(
            in_channels=dim,
            out_channels=dim,
            kernel_sizes=(3, 5, 7, 9, 11),
            dilations=(1, 1, 1, 1, 1),
            expansion=1.0,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='SiLU')
        )

        # 3. 注意力机制
        self.ca = Channel_Attention(dim)
        self.sa = Spatial_Attention()
        
        # 4. 最终融合投影
        self.proj_out = ConvModule(dim, dim, 1, norm_cfg=dict(type='BN'), act_cfg=None)

    def forward(self, x):
        # x: [B, 4*dim, H, W]
        x_in = self.proj_in(x)
        
        # 使用 PKI 提取多尺度上下文
        x_context = self.pki_block(x_in)
        
        # 注意力增强
        x_ca = self.ca(x_context)
        x_sa = self.sa(x_context)
        
        # 融合策略：原始输入残差 + PKI上下文 + 通道注意 + 空间注意
        out = x_in + x_context + x_ca + x_sa
        
        return self.proj_out(out)

# ==========================================
# 4. 特征增强模块 (保持原有逻辑，调用新模块)
# ==========================================

class Feature_Enhancement_Module(nn.Module):
    def __init__(self, in_d=None, out_d=64):
        super(Feature_Enhancement_Module, self).__init__()
        if in_d is None:
            in_d = [64, 128, 256, 512]
        self.in_d = in_d
        self.mid_d = in_d[0]
        self.out_d = out_d

        # Scale 1
        self.conv_scale1_c1 = nn.Sequential(nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True))
        self.conv_scale2_c1 = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),
                                            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True))
        self.conv_scale3_c1 = nn.Sequential(nn.AvgPool2d(kernel_size=4, stride=4),
                                            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True))
        self.conv_scale4_c1 = nn.Sequential(nn.AvgPool2d(kernel_size=8, stride=8),
                                            nn.Conv2d(self.in_d[0], self.mid_d, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True))

        # Scale 2
        self.conv_scale1_c2 = nn.Sequential(nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True))
        self.conv_scale2_c2 = nn.Sequential(nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True))
        self.conv_scale3_c2 = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),
                                            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True))
        self.conv_scale4_c2 = nn.Sequential(nn.AvgPool2d(kernel_size=4, stride=4),
                                            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(self.mid_d),
                                            nn.ReLU(inplace=True))

        # Scale 3
        self.conv_scale1_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale2_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale3_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale4_c3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )

        # Scale 4
        self.conv_scale1_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale2_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale3_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale4_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

        # Fusion (使用新的 FeatureFusionModule)
        self.conv_aggregation_s1 = FeatureFusionModule(self.mid_d)
        self.conv_aggregation_s2 = FeatureFusionModule(self.mid_d)
        self.conv_aggregation_s3 = FeatureFusionModule(self.mid_d)
        self.conv_aggregation_s4 = FeatureFusionModule(self.mid_d)

    def forward(self, c1, c2, c3, c4):
        # scale 1
        c1_s1 = self.conv_scale1_c1(c1)
        c1_s2 = self.conv_scale2_c1(c1)
        c1_s3 = self.conv_scale3_c1(c1)
        c1_s4 = self.conv_scale4_c1(c1)

        # scale 2
        c2_s1 = F.interpolate(self.conv_scale1_c2(c2), scale_factor=(2, 2), mode='bilinear')
        c2_s2 = self.conv_scale2_c2(c2)
        c2_s3 = self.conv_scale3_c2(c2)
        c2_s4 = self.conv_scale4_c2(c2)

        # scale 3
        c3_s1 = F.interpolate(self.conv_scale1_c3(c3), scale_factor=(4, 4), mode='bilinear')
        c3_s2 = F.interpolate(self.conv_scale2_c3(c3), scale_factor=(2, 2), mode='bilinear')
        c3_s3 = self.conv_scale3_c3(c3)
        c3_s4 = self.conv_scale4_c3(c3)

        # scale 4
        c4_s1 = F.interpolate(self.conv_scale1_c4(c4), scale_factor=(8, 8), mode='bilinear')
        c4_s2 = F.interpolate(self.conv_scale2_c4(c4), scale_factor=(4, 4), mode='bilinear')
        c4_s3 = F.interpolate(self.conv_scale3_c4(c4), scale_factor=(2, 2), mode='bilinear')
        c4_s4 = self.conv_scale4_c4(c4)

        s1 = self.conv_aggregation_s1(torch.cat([c1_s1, c2_s1, c3_s1, c4_s1], dim=1))
        s2 = self.conv_aggregation_s2(torch.cat([c1_s2, c2_s2, c3_s2, c4_s2], dim=1))
        s3 = self.conv_aggregation_s3(torch.cat([c1_s3, c2_s3, c3_s3, c4_s3], dim=1))
        s4 = self.conv_aggregation_s4(torch.cat([c1_s4, c2_s4, c3_s4, c4_s4], dim=1))

        return s1, s2, s3, s4


if __name__ == '__main__':
    x1 = torch.randn((1, 64, 64, 64))
    x2 = torch.randn((1, 128, 32, 32))
    x3 = torch.randn((1, 256, 16, 16))
    x4 = torch.randn((1, 512, 8, 8))
    model = Feature_Enhancement_Module()
    out = model(x1, x2, x3, x4)
    for i in out:
        print(i.shape)
