"""
Decoder with Low-Dimensional Feedback Mechanism
解码器低维度反馈优化实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.sim_agg import Feature_Enhancement_Module
from model.module.depthwise_separable import DepthwiseSeparableConvWithReLU
class Decoder_MultiScale_WithFeedback(nn.Module):
    """
    带反馈机制的多尺度解码器

    核心改进:
    1. Bottom-Up反馈路径: d2→d3→d4→d5 (细节信息反馈)
    2. 多尺度深度监督: d3, d4输出辅助监督信号
    3. 多尺度特征聚合: 融合所有尺度特征进行最终分类

    优化说明:
    - 残差权重尺度自适应: 粗尺度0.4, 中间0.3, 细尺度0.2
    - 反馈门控权重初始化为0.2，训练后自动调整

    预期收益:
    - F1 Score: +2-4%
    - 边界精度: +5-8%
    - 小目标检测: +10-15%
    """
    def __init__(self, in_d_list=[64, 128, 256, 256], out_d=2, use_depthwise=True, use_feedback=True):
        super(Decoder_MultiScale_WithFeedback, self).__init__()
        self.in_d_list = in_d_list  # [d2, d3, d4, d5]
        self.out_d = out_d
        self.use_feedback = use_feedback

        # ============= 现有Top-Down模块 =============
        # Process each scale with its own channel count
        if use_depthwise:
            self.conv5 = DepthwiseSeparableConvWithReLU(in_d_list[3], in_d_list[3], kernel_size=3, padding=1)
            self.conv4 = DepthwiseSeparableConvWithReLU(in_d_list[2], in_d_list[2], kernel_size=3, padding=1)
            self.conv3 = DepthwiseSeparableConvWithReLU(in_d_list[1], in_d_list[1], kernel_size=3, padding=1)
            self.conv2 = DepthwiseSeparableConvWithReLU(in_d_list[0], in_d_list[0], kernel_size=3, padding=1)
        else:
            self.conv5 = nn.Sequential(
                nn.Conv2d(in_d_list[3], in_d_list[3], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_d_list[3]),
                nn.ReLU(inplace=True)
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(in_d_list[2], in_d_list[2], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_d_list[2]),
                nn.ReLU(inplace=True)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_d_list[1], in_d_list[1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_d_list[1]),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_d_list[0], in_d_list[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_d_list[0]),
                nn.ReLU(inplace=True)
            )

        # Top-Down融合层
        self.fuse4 = nn.Sequential(
            nn.Conv2d(in_d_list[3] + in_d_list[2], in_d_list[2]*2, kernel_size=1),
            nn.BatchNorm2d(in_d_list[2]*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_d_list[2]*2, in_d_list[2], kernel_size=1)
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(in_d_list[2] + in_d_list[1], in_d_list[1]*2, kernel_size=1),
            nn.BatchNorm2d(in_d_list[1]*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_d_list[1]*2, in_d_list[1], kernel_size=1)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(in_d_list[1] + in_d_list[0], in_d_list[0]*2, kernel_size=1),
            nn.BatchNorm2d(in_d_list[0]*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_d_list[0]*2, in_d_list[0], kernel_size=1)
        )

        # Similarity fusion modules
        from model.module.decoder import Similarity_Fusion_Module
        self.SFM5 = Similarity_Fusion_Module(in_d_list[3])
        self.SFM4 = Similarity_Fusion_Module(in_d_list[2])
        self.SFM3 = Similarity_Fusion_Module(in_d_list[1])
        self.SFM2 = Similarity_Fusion_Module(in_d_list[0])

        # 优化: 残差连接权重尺度自适应初始化
        # 粗尺度使用更大的残差权重（保留更多语义信息）
        # 细尺度使用较小的残差权重（更依赖融合和反馈信息）
        self.residual_alpha_d4 = nn.Parameter(torch.tensor(0.4))  # 1/16尺度
        self.residual_alpha_d3 = nn.Parameter(torch.tensor(0.3))  # 1/8尺度
        self.residual_alpha_d2 = nn.Parameter(torch.tensor(0.2))  # 1/4尺度

        # ============= 新增: Bottom-Up反馈机制 =============
        if use_feedback:
            feedback_dim = 32  # 低维度反馈，避免参数爆炸

            # 1. 反馈压缩模块（将高分辨率特征压缩到低维）
            self.feedback_compress_d2 = nn.Sequential(
                nn.Conv2d(in_d_list[0], feedback_dim, 1, bias=False),
                nn.BatchNorm2d(feedback_dim),
                nn.ReLU(inplace=True)
            )
            self.feedback_compress_d3 = nn.Sequential(
                nn.Conv2d(in_d_list[1], feedback_dim, 1, bias=False),
                nn.BatchNorm2d(feedback_dim),
                nn.ReLU(inplace=True)
            )
            self.feedback_compress_d4 = nn.Sequential(
                nn.Conv2d(in_d_list[2], feedback_dim, 1, bias=False),
                nn.BatchNorm2d(feedback_dim),
                nn.ReLU(inplace=True)
            )

            # 2. 反馈融合模块（注入到粗尺度特征）
            self.feedback_fusion_d3 = nn.Sequential(
                nn.Conv2d(in_d_list[1] + feedback_dim, in_d_list[1], 1, bias=False),
                nn.BatchNorm2d(in_d_list[1]),
                nn.ReLU(inplace=True)
            )
            self.feedback_fusion_d4 = nn.Sequential(
                nn.Conv2d(in_d_list[2] + feedback_dim, in_d_list[2], 1, bias=False),
                nn.BatchNorm2d(in_d_list[2]),
                nn.ReLU(inplace=True)
            )
            self.feedback_fusion_d5 = nn.Sequential(
                nn.Conv2d(in_d_list[3] + feedback_dim, in_d_list[3], 1, bias=False),
                nn.BatchNorm2d(in_d_list[3]),
                nn.ReLU(inplace=True)
            )

            # 3. 可学习的反馈门控权重（初始值较小，避免破坏预训练权重）
            self.feedback_gate_d3 = nn.Parameter(torch.tensor(0.2))
            self.feedback_gate_d4 = nn.Parameter(torch.tensor(0.2))
            self.feedback_gate_d5 = nn.Parameter(torch.tensor(0.2))

            # ============= 新增: 多尺度深度监督 =============
            self.aux_cls_d3 = nn.Conv2d(in_d_list[1], out_d, 1)  # 1/8分辨率辅助分类
            self.aux_cls_d4 = nn.Conv2d(in_d_list[2], out_d, 1)  # 1/16分辨率辅助分类

            # ============= 新增: 多尺度特征聚合分类器 =============
            # 将所有尺度特征聚合后进行分类
            self.multi_scale_aggregator = nn.Sequential(
                nn.Conv2d(sum(in_d_list), in_d_list[0], 1, bias=False),
                nn.BatchNorm2d(in_d_list[0]),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConvWithReLU(in_d_list[0], in_d_list[0], 3, padding=1) if use_depthwise else
                nn.Sequential(
                    nn.Conv2d(in_d_list[0], in_d_list[0], 3, padding=1, bias=False),
                    nn.BatchNorm2d(in_d_list[0]),
                    nn.ReLU(inplace=True)
                )
            )

        # 最终分类器
        self.cls = nn.Conv2d(in_d_list[0], out_d, 1)

    def forward(self, d5, d4, d3, d2, sim5, sim4, sim3, sim2):
        """
        两阶段解码:
        阶段1: Top-Down Pass (语义传播, 从粗到细)
        阶段2: Bottom-Up Feedback (细节反馈, 从细到粗)
        阶段3: 多尺度聚合 + 深度监督

        Args:
            d5, d4, d3, d2: DGMA2输出的多尺度差异特征
            sim5, sim4, sim3, sim2: 多尺度相似度图

        Returns:
            training模式: (mask, aux_mask_d3, aux_mask_d4)
            eval模式: mask
        """
        # ========== 阶段1: Top-Down Pass (现有流程) ==========
        # d5 (1/32) → d4 (1/16) → d3 (1/8) → d2 (1/4)

        # Process d5
        d5 = self.conv5(d5)
        d5 = self.SFM5(d5, sim5)
        d5_up = F.interpolate(d5, d4.size()[2:], mode='bilinear', align_corners=True)

        # Process d4 with d5
        d4_input = d4  # 保存原始特征用于残差连接
        d4 = self.conv4(d4)
        d4 = self.fuse4(torch.cat([d4, d5_up], dim=1))
        d4 = self.SFM4(d4, sim4)
        d4 = d4 + d4_input * self.residual_alpha_d4
        d4_up = F.interpolate(d4, d3.size()[2:], mode='bilinear', align_corners=True)

        # Process d3 with d4
        d3_input = d3
        d3 = self.conv3(d3)
        d3 = self.fuse3(torch.cat([d3, d4_up], dim=1))
        d3 = self.SFM3(d3, sim3)
        d3 = d3 + d3_input * self.residual_alpha_d3
        d3_up = F.interpolate(d3, d2.size()[2:], mode='bilinear', align_corners=True)

        # Process d2 with d3
        d2_input = d2
        d2 = self.conv2(d2)
        d2 = self.fuse2(torch.cat([d2, d3_up], dim=1))
        d2 = self.SFM2(d2, sim2)
        d2 = d2 + d2_input * self.residual_alpha_d2

        # ========== 阶段2: Bottom-Up Feedback (新增) ==========
        if self.use_feedback:
            # d2 (1/4) → d3 (1/8) → d4 (1/16) → d5 (1/32)
            # 细节信息反馈到粗尺度

            # 1. d2反馈到d3
            fb_d2 = self.feedback_compress_d2(d2)  # [B, 32, H/4, W/4]
            fb_d2_down = F.adaptive_avg_pool2d(fb_d2, d3.size()[2:])  # [B, 32, H/8, W/8]
            d3_with_fb = torch.cat([d3, fb_d2_down], dim=1)  # [B, 128+32, H/8, W/8]
            d3_refined = self.feedback_fusion_d3(d3_with_fb)  # [B, 128, H/8, W/8]
            d3 = d3 + d3_refined * self.feedback_gate_d3  # 门控融合

            # 2. d3反馈到d4
            fb_d3 = self.feedback_compress_d3(d3)  # [B, 32, H/8, W/8]
            fb_d3_down = F.adaptive_avg_pool2d(fb_d3, d4.size()[2:])  # [B, 32, H/16, W/16]
            d4_with_fb = torch.cat([d4, fb_d3_down], dim=1)  # [B, 256+32, H/16, W/16]
            d4_refined = self.feedback_fusion_d4(d4_with_fb)  # [B, 256, H/16, W/16]
            d4 = d4 + d4_refined * self.feedback_gate_d4

            # 3. d4反馈到d5
            fb_d4 = self.feedback_compress_d4(d4)  # [B, 32, H/16, W/16]
            fb_d4_down = F.adaptive_avg_pool2d(fb_d4, d5.size()[2:])  # [B, 32, H/32, W/32]
            d5_with_fb = torch.cat([d5, fb_d4_down], dim=1)  # [B, 256+32, H/32, W/32]
            d5_refined = self.feedback_fusion_d5(d5_with_fb)  # [B, 256, H/32, W/32]
            d5 = d5 + d5_refined * self.feedback_gate_d5

        # ========== 阶段3: 多尺度聚合 + 深度监督 ==========
        if self.use_feedback:
            # 多尺度聚合: 所有尺度上采样到d2分辨率
            d5_to_d2 = F.interpolate(d5, d2.size()[2:], mode='bilinear', align_corners=True)
            d4_to_d2 = F.interpolate(d4, d2.size()[2:], mode='bilinear', align_corners=True)
            d3_to_d2 = F.interpolate(d3, d2.size()[2:], mode='bilinear', align_corners=True)

            # 拼接所有尺度特征
            multi_scale_feat = torch.cat([d2, d3_to_d2, d4_to_d2, d5_to_d2], dim=1)
            # [B, 64+128+256+256, H/4, W/4] -> [B, 64, H/4, W/4]
            aggregated_feat = self.multi_scale_aggregator(multi_scale_feat)

            # 最终分类
            mask = self.cls(aggregated_feat)

            # 辅助监督（训练时使用）
            if self.training:
                aux_mask_d3 = self.aux_cls_d3(d3)
                aux_mask_d4 = self.aux_cls_d4(d4)
                return mask, aux_mask_d3, aux_mask_d4
            else:
                return mask
        else:
            # 不使用反馈时，只用d2分类（与原始行为一致）
            mask = self.cls(d2)
            return mask


# 别名，保持向后兼容
Decoder_MultiScale_V2 = Decoder_MultiScale_WithFeedback

