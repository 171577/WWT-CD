"""
DGMA2 Core Components (Optimized)
Integrates Poly Kernel Inception Block, Coordinate Attention, and Enhanced DEAM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence
from model.module.depthwise_separable import DepthwiseSeparableConvWithReLU

# ==================== 1. 新增: Poly Kernel Inception Block 组件 ====================

def autopad(kernel_size: int, padding: Optional[int] = None, dilation: int = 1) -> int:
    if padding is None:
        padding = (kernel_size - 1) * dilation // 2
    return padding

def make_divisible(value: int, divisor: int = 8) -> int:
    return int((value + divisor // 2) // divisor * divisor)

class ConvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, groups: int = 1,
                 norm_cfg: Optional[dict] = None, act_cfg: Optional[dict] = None):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                dilation=dilation, groups=groups, bias=(norm_cfg is None)))
        if norm_cfg:
            layers.append(nn.BatchNorm2d(out_channels, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5)))
        if act_cfg:
            if act_cfg['type'] == 'ReLU': layers.append(nn.ReLU(inplace=True))
            elif act_cfg['type'] == 'SiLU': layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Poly_Kernel_Inception_Block(nn.Module):
    """
    替代原有的多分支卷积，使用多尺度大核深度卷积提取特征
    """
    def __init__(self, in_channels: int, out_channels: Optional[int] = None,
                 kernel_sizes: Sequence[int] = (3, 5, 7, 9), # 覆盖原有的3,5,7并扩展
                 dilations: Sequence[int] = (1, 1, 1, 1),
                 expansion: float = 1.0,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = dict(type='ReLU')): # 保持与原项目一致使用ReLU
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # 动态创建深度卷积层，根据kernel_sizes的长度
        self.dw_convs = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            self.dw_convs.append(
                ConvModule(hidden_channels, hidden_channels, k, 1, autopad(k, None, d),
                           dilation=d, groups=hidden_channels, norm_cfg=None, act_cfg=None)
            )

        self.pw_conv = ConvModule(hidden_channels, hidden_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.pre_conv(x)
        # 多尺度特征相加融合
        x_sum = x
        for conv in self.dw_convs:
            x_sum = x_sum + conv(x)

        x = self.pw_conv(x_sum)
        x = self.post_conv(x)
        return x

# ==================== 2. 新增: 坐标注意力 (Coordinate Attention) ====================
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out

# ==================== 3. 改造: MSFF (使用 PKIB + SE) ====================
class MSFF(nn.Module):
    """Multi-Scale Feature Fusion Module (Optimized with PKIB)"""
    def __init__(self, inchannel, mid_channel=None, use_depthwise=True):
        super(MSFF, self).__init__()

        # 1. 使用 Poly Kernel Inception Block 替代原本的4分支结构
        # 这将自动处理 3x3, 5x5, 7x7, 9x9 的多尺度特征提取
        self.pkib = Poly_Kernel_Inception_Block(
            in_channels=inchannel,
            out_channels=inchannel,
            kernel_sizes=[3, 5, 7, 9], # 覆盖并扩展原有的尺度
            expansion=1.0
        )

        # 2. 保留 SE Attention (通道注意力)
        # PKIB 负责空间多尺度，SE 负责通道重要性，两者互补
        self.se_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inchannel, inchannel // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel // 4, inchannel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 提取多尺度空间特征
        x_feat = self.pkib(x)

        # 应用通道注意力重校准
        att = self.se_attn(x_feat)
        out = x_feat * att

        # 残差连接 (可选，视网络深度而定，这里建议加上以保持梯度)
        return out + x

# ==================== 4. 改造: MDFM Block (使用 CoordAtt + 动态融合) ====================
class MDFMBlock(nn.Module):
    """Multi-scale Difference Feature Module (Optimized)"""
    def __init__(self, in_d, out_d, dw=False, use_depthwise=True):
        super(MDFMBlock, self).__init__()
        self.in_d = in_d
        self.out_d = out_d

        # 使用新版 MSFF
        self.MPFL = MSFF(inchannel=in_d, mid_channel=64, use_depthwise=use_depthwise)

        # 特征增强卷积
        if use_depthwise:
            self.conv_diff_enh = DepthwiseSeparableConvWithReLU(in_d, in_d, kernel_size=3, padding=1)
            self.conv_dr = DepthwiseSeparableConvWithReLU(in_d, out_d, kernel_size=3, padding=1)
            self.conv_sub = DepthwiseSeparableConvWithReLU(in_d, in_d, kernel_size=3, padding=1)
        else:
            self.conv_diff_enh = nn.Sequential(nn.Conv2d(in_d, in_d, 3, 1, 1, bias=False), nn.BatchNorm2d(in_d), nn.ReLU(inplace=True))
            self.conv_dr = nn.Sequential(nn.Conv2d(in_d, out_d, 3, 1, 1, bias=False), nn.BatchNorm2d(out_d), nn.ReLU(inplace=True))
            self.conv_sub = nn.Sequential(nn.Conv2d(in_d, in_d, 3, 1, 1, bias=False), nn.BatchNorm2d(in_d), nn.ReLU(inplace=True))

        self.convmix = nn.Sequential(
            nn.Conv2d(2 * in_d, in_d, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_d),
            nn.ReLU(inplace=True)
        )

        # 创新点: 坐标注意力 (替代简单的Sigmoid加权)
        self.coord_att = CoordAtt(in_d, in_d)

        # 创新点: 动态融合层 (替代硬编码的 0.2/0.05 权重)
        # 输入通道 = 增强特征(in_d) + 原始差值(in_d) + 绝对差值(in_d)
        self.diff_fusion = nn.Conv2d(in_d * 3, in_d, 1, bias=False)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x_diff = x1 - x2
        x_sub = torch.abs(x_diff)

        # 生成注意力图
        x_att = torch.sigmoid(self.conv_sub(x_sub))

        # 特征增强
        x1_enh = (x1 * x_att) + self.MPFL(self.conv_diff_enh(x1))
        x2_enh = (x2 * x_att) + self.MPFL(self.conv_diff_enh(x2))

        # 融合 T1 和 T2
        x_f = torch.stack((x1_enh, x2_enh), dim=2)
        x_f = torch.reshape(x_f, (b, -1, h, w))
        x_f = self.convmix(x_f)

        # 应用坐标注意力增强融合特征
        x_f = self.coord_att(x_f)

        # 动态融合: 将增强特征、原始差值、绝对差值拼接融合
        # 替代了原代码: x_f * x_att + x_diff * 0.2 + x_sub * 0.05
        fusion_in = torch.cat([x_f * x_att, x_diff, x_sub], dim=1)
        x_final = self.diff_fusion(fusion_in)

        out = self.conv_dr(x_final)
        return out

# ==================== DAM Components (保持不变或微调) ====================
# (此处 Refine, CIEM, GRM, DAMBlock 代码保持原样即可，
#  因为它们主要负责流程控制，核心算子已在上面优化)
class Refine(nn.Module):
    def __init__(self, inchannel, outchannel, use_depthwise=True):
        super(Refine, self).__init__()
        if use_depthwise:
            self.conv1 = DepthwiseSeparableConvWithReLU(inchannel, inchannel, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(inchannel),
                nn.ReLU(inplace=True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inchannel + outchannel, outchannel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
    def forward(self, x1, x2):
        x1 = F.interpolate(x1, x2.size()[2:], mode='bilinear', align_corners=True)
        x1 = self.conv1(x1)
        x_f = torch.cat([x1, x2], dim=1)
        x_f = self.conv2(x_f)
        return x_f

class CIEM(nn.Module):
    def __init__(self, base_dim=64, out_d=[64, 128, 256, 256], use_depthwise=True):
        super(CIEM, self).__init__()
        self.base_dim = base_dim
        self.refine1 = Refine(out_d[3], out_d[2], use_depthwise=use_depthwise)
        self.refine2 = Refine(out_d[2], out_d[1], use_depthwise=use_depthwise)
        self.refine3 = Refine(out_d[1], out_d[0], use_depthwise=use_depthwise)
        self.conv_dr = nn.Sequential(
            nn.Conv2d(out_d[0], base_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )
        self.pools_sizes = [2, 4, 8]
        self.conv_pool1 = nn.Sequential(nn.AvgPool2d(self.pools_sizes[0]), nn.Conv2d(base_dim, out_d[1], 3, 1, 1, bias=False))
        self.conv_pool2 = nn.Sequential(nn.AvgPool2d(self.pools_sizes[1]), nn.Conv2d(base_dim, out_d[2], 3, 1, 1, bias=False))
        self.conv_pool3 = nn.Sequential(nn.AvgPool2d(self.pools_sizes[2]), nn.Conv2d(base_dim, out_d[3], 3, 1, 1, bias=False))

    def forward(self, d5, d4, d3, d2):
        r1 = self.refine1(d5, d4)
        r2 = self.refine2(r1, d3)
        x = self.refine3(r2, d2)
        x = self.conv_dr(x)
        return self.conv_pool3(x), self.conv_pool2(x), self.conv_pool1(x), x

class GRM(nn.Module):
    def __init__(self, out_d=[64, 128, 256, 256], use_depthwise=True):
        super(GRM, self).__init__()
        if use_depthwise:
            self.conv_d5 = DepthwiseSeparableConvWithReLU(out_d[3] * 2, out_d[3], 3, padding=1)
            self.conv_d4 = DepthwiseSeparableConvWithReLU(out_d[2] * 2, out_d[2], 3, padding=1)
            self.conv_d3 = DepthwiseSeparableConvWithReLU(out_d[1] * 2, out_d[1], 3, padding=1)
            self.conv_d2 = DepthwiseSeparableConvWithReLU(out_d[0] * 2, out_d[0], 3, padding=1)
        else:
            # 简化写法，实际使用时可补全
            self.conv_d5 = nn.Sequential(nn.Conv2d(out_d[3]*2, out_d[3], 3, 1, 1), nn.BatchNorm2d(out_d[3]), nn.ReLU(True))
            self.conv_d4 = nn.Sequential(nn.Conv2d(out_d[2]*2, out_d[2], 3, 1, 1), nn.BatchNorm2d(out_d[2]), nn.ReLU(True))
            self.conv_d3 = nn.Sequential(nn.Conv2d(out_d[1]*2, out_d[1], 3, 1, 1), nn.BatchNorm2d(out_d[1]), nn.ReLU(True))
            self.conv_d2 = nn.Sequential(nn.Conv2d(out_d[0]*2, out_d[0], 3, 1, 1), nn.BatchNorm2d(out_d[0]), nn.ReLU(True))

    def stack(self, x1, x2):
        b, c, h, w = x1.size()
        x_f = torch.stack((x1, x2), dim=2)
        return torch.reshape(x_f, (b, -1, h, w))

    def forward(self, d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p):
        return self.conv_d5(self.stack(d5_p, d5)), self.conv_d4(self.stack(d4_p, d4)), \
               self.conv_d3(self.stack(d3_p, d3)), self.conv_d2(self.stack(d2_p, d2))

class DAMBlock(nn.Module):
    def __init__(self, base_dim=64, out_d=[64, 128, 256, 256], use_depthwise=True):
        super(DAMBlock, self).__init__()
        self.ciem = CIEM(base_dim, out_d, use_depthwise)
        self.grm = GRM(out_d, use_depthwise)
    def forward(self, d5, d4, d3, d2):
        d5_p, d4_p, d3_p, d2_p = self.ciem(d5, d4, d3, d2)
        return self.grm(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

# ==================== 5. 改造: DEAM (使用 Similarity Bias) ====================
class DEAM_with_Swsi(nn.Module):
    """Dual-temporal Enhancement Attention Module with Similarity Bias"""
    def __init__(self, input_dim, diff_dim, ds=8, beta_init=0.3):
        super(DEAM_with_Swsi, self).__init__()
        self.input_dim = input_dim
        self.diff_dim = diff_dim
        self.key_channel = self.diff_dim // 8
        self.ds = ds

        self.pool = nn.AvgPool2d(self.ds)
        self.query_conv = nn.Conv2d(diff_dim, diff_dim // 8, 1)
        self.key_conv = nn.Conv2d(diff_dim, diff_dim // 8, 1)
        self.value_conv = nn.Conv2d(input_dim, input_dim, 1)

        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Conv2d(input_dim, diff_dim, 1) if input_dim != diff_dim else nn.Identity()

    def forward(self, input, diff, s_wsi=None):
        x = self.pool(input)
        diff = self.pool(diff)
        b, c, h, w = diff.size()

        proj_query = self.query_conv(diff).view(b, -1, h * w).permute(0, 2, 1)
        proj_key = self.key_conv(diff).view(b, -1, h * w)

        # 计算原始能量
        energy = torch.bmm(proj_query, proj_key)
        energy = (self.key_channel ** -0.5) * energy

        # 创新点: 将相似度作为 Attention Bias 注入
        # 如果某个区域相似度高，我们增加其在注意力矩阵中的权重
        if s_wsi is not None:
            s_wsi_resized = F.interpolate(s_wsi, size=(h, w), mode='bilinear', align_corners=True)
            sim_flat = s_wsi_resized.view(b, 1, h * w)
            # 广播加法: 增强相似区域的注意力响应
            energy = energy + torch.sigmoid(sim_flat) * self.beta

        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(b, -1, h * w)

        # 双重增强: Value 调制 (保留)
        if s_wsi is not None:
             s_wsi_resized = F.interpolate(s_wsi, size=(h, w), mode='bilinear', align_corners=True)
             sim_gate = torch.sigmoid(s_wsi_resized).view(b, -1, h * w)
             proj_value = proj_value * (1.0 + 0.1 * sim_gate)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, self.input_dim, h, w)
        out = F.interpolate(out, [h * self.ds, w * self.ds], mode='bilinear', align_corners=True)
        out = self.gamma * out + input

        out = self.proj(out)
        return out


# ==================== Wrapper classes for backward compatibility ====================
class MDFMBlock_InceptionMultiScale(MDFMBlock):
    """Wrapper for MDFMBlock that accepts mid_channel parameter"""
    def __init__(self, in_d, out_d, mid_channel=None, dw=False, use_depthwise=True):
        # mid_channel is accepted but not used in the new PKIB-based design
        super().__init__(in_d, out_d, dw=dw, use_depthwise=use_depthwise)


class MDFMBlock_InceptionMultiScale_Attention(MDFMBlock):
    """Wrapper for MDFMBlock with attention (same as base MDFMBlock for now)"""
    def __init__(self, in_d, out_d, mid_channel=None, dw=False, use_depthwise=True):
        # mid_channel is accepted but not used in the new PKIB-based design
        super().__init__(in_d, out_d, dw=dw, use_depthwise=use_depthwise)
