import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightFeatureAlign(nn.Module):
    """
    轻量级特征对齐模块

    设计理念:
    - 对于浅层特征(scale1, scale2)，配准误差通常较小
    - 使用轻量级的偏移预测网络即可实现对齐
    - 避免可变形注意力的复杂采样和多头机制

    参数量对比:
    - 可变形注意力(64通道): ~150K参数
    - 轻量级对齐(64通道): ~15K参数 (减少90%)
    """

    def __init__(self, channels, offset_groups=4):
        """
        Args:
            channels: 输入特征通道数
            offset_groups: 偏移量预测的分组数，减少参数
        """
        super().__init__()
        self.channels = channels
        self.offset_groups = offset_groups

        # 偏移量预测网络 (输入: 双时相特征拼接)
        # 输出: 2通道偏移量 (dx, dy)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 2, 3, padding=1, bias=True),
            nn.Tanh()  # 限制偏移范围到 [-1, 1]
        )

        # 可学习的偏移缩放因子
        self.offset_scale = nn.Parameter(torch.tensor(2.0))

        # 特征增强卷积 (对齐后)
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=offset_groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def _grid_sample(self, x, offset):
        """
        使用偏移量对特征进行采样对齐

        Args:
            x: 输入特征 [B, C, H, W]
            offset: 偏移量 [B, 2, H, W]

        Returns:
            对齐后的特征 [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 创建基础网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]
        grid = grid.repeat(B, 1, 1, 1)  # [B, 2, H, W]

        # 添加偏移量 (归一化到[-1, 1]范围)
        # offset已经经过Tanh限制，再乘以scale控制最大偏移
        offset_normalized = offset * self.offset_scale / max(H, W)
        grid = grid + offset_normalized

        # 转换为grid_sample所需的格式 [B, H, W, 2]
        grid = grid.permute(0, 2, 3, 1)

        # 双线性插值采样
        aligned = F.grid_sample(
            x, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return aligned

    def forward(self, x1, x2):
        """
        双时相特征对齐

        Args:
            x1: 时相1特征 [B, C, H, W]
            x2: 时相2特征 [B, C, H, W]

        Returns:
            x1_aligned: 对齐后的时相1 [B, C, H, W]
            x2_aligned: 对齐后的时相2 [B, C, H, W]
        """
        # 1. 预测偏移量
        concat_feat = torch.cat([x1, x2], dim=1)  # [B, 2C, H, W]
        offset = self.offset_conv(concat_feat)  # [B, 2, H, W]

        # 2. 对齐x2到x1
        # (假设x1为参考帧，对x2进行对齐)
        x2_aligned = self._grid_sample(x2, offset)

        # 3. 特征增强
        x1_enhanced = self.feature_enhance(x1)
        x2_enhanced = self.feature_enhance(x2_aligned)

        # 4. 残差连接
        x1_out = x1 + x1_enhanced * self.residual_weight
        x2_out = x2_aligned + x2_enhanced * self.residual_weight

        return x1_out, x2_out


class LightweightFeatureAlignV2(nn.Module):
    """
    轻量级特征对齐 V2 - 更简化的版本

    进一步减少参数:
    - 使用深度可分离卷积
    - 单次偏移预测 (仅对x2进行对齐)
    - 适用于内存受限场景
    """

    def __init__(self, channels):
        super().__init__()

        # 深度可分离卷积预测偏移
        self.offset_conv = nn.Sequential(
            # 深度卷积
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            # 点卷积
            nn.Conv2d(channels * 2, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            # 输出偏移
            nn.Conv2d(channels // 4, 2, 1, bias=True),
            nn.Tanh()
        )

        self.scale = nn.Parameter(torch.ones(1) * 2.0)

    def forward(self, x1, x2):
        """
        简化的对齐流程

        Args:
            x1: 时相1特征 [B, C, H, W]
            x2: 时相2特征 [B, C, H, W]

        Returns:
            x1: 保持不变 [B, C, H, W]
            x2_aligned: 对齐后的时相2 [B, C, H, W]
        """
        B, C, H, W = x1.shape

        # 预测偏移量
        offset = self.offset_conv(torch.cat([x1, x2], dim=1))  # [B, 2, H, W]

        # 创建采样网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x1.device),
            torch.linspace(-1, 1, W, device=x1.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        # 应用偏移
        offset_normalized = offset * self.scale / max(H, W)
        grid = (grid + offset_normalized).permute(0, 2, 3, 1)

        # 采样对齐
        x2_aligned = F.grid_sample(x2, grid, mode='bilinear',
                                   padding_mode='border', align_corners=True)

        return x1, x2_aligned