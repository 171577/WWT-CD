"""
Cross-Scale Attention Bridge
跨尺度注意力桥接模块

实现不同尺度间的信息交互:
- 细尺度提供边界细节
- 粗尺度提供语义指导
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossScaleAttentionBridge(nn.Module):
    """
    跨尺度注意力桥接模块

    设计理念:
    - 粗尺度(d4, d5) → 细尺度(d2, d3): 语义指导
    - 细尺度(d2, d3) → 粗尺度(d4, d5): 边界细节

    注意力机制:
    - 使用轻量级的自注意力实现跨尺度交互
    - 避免全连接层，使用卷积保持空间信息
    """

    def __init__(self, channels_list, reduction=4):
        """
        Args:
            channels_list: 各尺度通道数 [C2, C3, C4, C5]
            reduction: 注意力压缩比例
        """
        super().__init__()
        self.channels_list = channels_list

        # === 粗到细的语义指导 ===
        # d5 → d4: 全局语义
        self.coarse_to_fine_54 = self._build_attention_module(
            channels_list[3], channels_list[2], reduction
        )

        # d4 → d3: 中层语义
        self.coarse_to_fine_43 = self._build_attention_module(
            channels_list[2], channels_list[1], reduction
        )

        # d3 → d2: 局部语义
        self.coarse_to_fine_32 = self._build_attention_module(
            channels_list[1], channels_list[0], reduction
        )

        # === 细到粗的边界细节 ===
        # d2 → d3: 细节信息
        self.fine_to_coarse_23 = self._build_attention_module(
            channels_list[0], channels_list[1], reduction
        )

        # d3 → d4: 边界信息
        self.fine_to_coarse_34 = self._build_attention_module(
            channels_list[1], channels_list[2], reduction
        )

        # d4 → d5: 结构信息
        self.fine_to_coarse_45 = self._build_attention_module(
            channels_list[2], channels_list[3], reduction
        )

        # 融合权重 (可学习)
        self.alpha_d2 = nn.Parameter(torch.tensor(0.3))  # 细尺度更依赖语义指导
        self.alpha_d3 = nn.Parameter(torch.tensor(0.25))
        self.alpha_d4 = nn.Parameter(torch.tensor(0.2))
        self.alpha_d5 = nn.Parameter(torch.tensor(0.15))  # 粗尺度更依赖边界细节

    def _build_attention_module(self, from_channels, to_channels, reduction):
        """
        构建跨尺度注意力模块

        Args:
            from_channels: 源尺度通道数
            to_channels: 目标尺度通道数
            reduction: 压缩比例

        Returns:
            nn.Module: 注意力模块
        """
        mid_channels = max(to_channels // reduction, 16)

        return nn.Sequential(
            # 通道对齐
            nn.Conv2d(from_channels, to_channels, 1, bias=False),
            nn.BatchNorm2d(to_channels),
            nn.ReLU(inplace=True),

            # 自注意力
            nn.Conv2d(to_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, to_channels, 1, bias=False),
            nn.Sigmoid()  # 生成注意力权重
        )

    def forward(self, d2, d3, d4, d5):
        """
        跨尺度双向注意力

        Args:
            d2, d3, d4, d5: 各尺度特征 [B, C, H, W]

        Returns:
            d2_out, d3_out, d4_out, d5_out: 增强后的特征
        """
        # === 阶段1: 粗到细的语义指导 ===
        # d5 → d4
        d5_to_d4 = F.interpolate(d5, d4.shape[2:], mode='bilinear', align_corners=True)
        attn_54 = self.coarse_to_fine_54(d5_to_d4)
        d4_semantic = d4 * attn_54

        # d4 → d3
        d4_to_d3 = F.interpolate(d4, d3.shape[2:], mode='bilinear', align_corners=True)
        attn_43 = self.coarse_to_fine_43(d4_to_d3)
        d3_semantic = d3 * attn_43

        # d3 → d2
        d3_to_d2 = F.interpolate(d3, d2.shape[2:], mode='bilinear', align_corners=True)
        attn_32 = self.coarse_to_fine_32(d3_to_d2)
        d2_semantic = d2 * attn_32

        # === 阶段2: 细到粗的边界细节 ===
        # d2 → d3
        d2_to_d3 = F.adaptive_avg_pool2d(d2, d3.shape[2:])
        attn_23 = self.fine_to_coarse_23(d2_to_d3)
        d3_detail = d3 * attn_23

        # d3 → d4
        d3_to_d4 = F.adaptive_avg_pool2d(d3, d4.shape[2:])
        attn_34 = self.fine_to_coarse_34(d3_to_d4)
        d4_detail = d4 * attn_34

        # d4 → d5
        d4_to_d5 = F.adaptive_avg_pool2d(d4, d5.shape[2:])
        attn_45 = self.fine_to_coarse_45(d4_to_d5)
        d5_detail = d5 * attn_45

        # === 阶段3: 双向融合 ===
        d2_out = d2 + d2_semantic * self.alpha_d2
        d3_out = d3 + (d3_semantic + d3_detail) * self.alpha_d3
        d4_out = d4 + (d4_semantic + d4_detail) * self.alpha_d4
        d5_out = d5 + d5_detail * self.alpha_d5

        return d2_out, d3_out, d4_out, d5_out


class GatedResidualFusion(nn.Module):
    """
    门控残差融合模块

    替代简单的stack操作，使用可学习的门控机制融合多尺度特征
    """

    def __init__(self, channels, n_features=2):
        """
        Args:
            channels: 特征通道数
            n_features: 要融合的特征数量
        """
        super().__init__()
        self.n_features = n_features

        # 门控网络
        self.gate = nn.Sequential(
            nn.Conv2d(channels * n_features, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, n_features, 1),
            nn.Softmax(dim=1)  # 归一化权重
        )

        # 融合卷积
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * n_features, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 残差权重
        self.residual_weight = nn.Parameter(torch.tensor(0.2))

    def forward(self, *features):
        """
        门控融合多个特征

        Args:
            *features: 多个特征 [B, C, H, W] x n_features

        Returns:
            fused: 融合后的特征 [B, C, H, W]
        """
        if len(features) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(features)}")

        # 拼接特征
        concat_feat = torch.cat(features, dim=1)  # [B, C*n, H, W]

        # 计算门控权重
        gate_weights = self.gate(concat_feat)  # [B, n, 1, 1] after adaptive pooling

        # 加权融合
        weighted_features = []
        for i, feat in enumerate(features):
            weight = gate_weights[:, i:i + 1, :, :]
            # 广播权重到空间维度
            weight = weight.expand_as(feat)
            weighted_features.append(feat * weight)

        # 求和
        fused_weighted = sum(weighted_features)

        # 融合卷积
        fused = self.fusion(concat_feat)

        # 残差连接
        return fused_weighted + fused * self.residual_weight


class UncertaintyModule(nn.Module):
    """
    不确定性估计模块

    使用MC Dropout估计模型预测的不确定性:
    - 训练时: 正常Dropout
    - 推理时: 多次前向传播+Dropout，计算方差
    """

    def __init__(self, channels, dropout_rate=0.1, n_samples=5):
        """
        Args:
            channels: 特征通道数
            dropout_rate: Dropout概率
            n_samples: MC采样次数 (仅推理时使用)
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples

        # 空间Dropout (2D)
        self.mc_dropout = nn.Dropout2d(dropout_rate)

        # 不确定性量化网络
        self.uncertainty_estimator = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()  # 输出不确定性分数 [0, 1]
        )

    def forward(self, x, return_uncertainty=False):
        """
        前向传播 + 不确定性估计

        Args:
            x: 输入特征 [B, C, H, W]
            return_uncertainty: 是否返回不确定性图

        Returns:
            如果return_uncertainty=False: x_dropped
            如果return_uncertainty=True: (x_mean, uncertainty_map)
        """
        if self.training or not return_uncertainty:
            # 训练时或不需要不确定性时，直接dropout
            return self.mc_dropout(x)

        else:
            # 推理时: MC Dropout采样
            samples = []
            for _ in range(self.n_samples):
                samples.append(self.mc_dropout(x))

            # 计算均值和方差
            samples_stacked = torch.stack(samples, dim=0)  # [n_samples, B, C, H, W]
            x_mean = samples_stacked.mean(dim=0)  # [B, C, H, W]
            x_var = samples_stacked.var(dim=0)  # [B, C, H, W]

            # 通道维度平均，得到空间不确定性图
            uncertainty_map = x_var.mean(dim=1, keepdim=True)  # [B, 1, H, W]

            # 归一化不确定性
            uncertainty_map = self.uncertainty_estimator(x_mean)

            return x_mean, uncertainty_map


class LightweightCrossScaleAttention(nn.Module):
    """
    轻量级跨尺度注意力 (简化版)

    更少的参数，更快的速度
    仅实现单向的粗到细语义指导
    """

    def __init__(self, channels_list):
        super().__init__()

        # 仅保留粗到细的语义指导
        self.guide_54 = nn.Conv2d(channels_list[3], channels_list[2], 1)
        self.guide_43 = nn.Conv2d(channels_list[2], channels_list[1], 1)
        self.guide_32 = nn.Conv2d(channels_list[1], channels_list[0], 1)

        # 门控权重
        self.alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, d2, d3, d4, d5):
        """简化的跨尺度注意力"""
        # 粗到细传播
        d5_to_d4 = F.interpolate(d5, d4.shape[2:], mode='bilinear', align_corners=True)
        d4_out = d4 + self.guide_54(d5_to_d4) * self.alpha

        d4_to_d3 = F.interpolate(d4_out, d3.shape[2:], mode='bilinear', align_corners=True)
        d3_out = d3 + self.guide_43(d4_to_d3) * self.alpha

        d3_to_d2 = F.interpolate(d3_out, d2.shape[2:], mode='bilinear', align_corners=True)
        d2_out = d2 + self.guide_32(d3_to_d2) * self.alpha

        return d2_out, d3_out, d4_out, d5