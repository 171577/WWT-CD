import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.sim_agg import Feature_Enhancement_Module
from model.module.depthwise_separable import DepthwiseSeparableConvWithReLU
class Similarity_Fusion_Module(nn.Module):
    def __init__(self, channel, reduction=16, use_depthwise=True):
        super(Similarity_Fusion_Module, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 修复: 合并max和avg结果后再调用MLP，避免重复计算
        self.mlp1 = nn.Sequential(nn.Conv2d(channel * 2, channel // reduction, 1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(channel // reduction, channel, 1, bias=False))

        self.mlp2 = nn.Sequential(nn.Conv2d(channel * 2, channel // reduction, 1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(channel // reduction, channel, 1, bias=False))

        # 修复5: 优化融合层，避免通道翻倍
        # 使用深度可分离卷积替换3×3卷积
        if use_depthwise:
            self.conv = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1, bias=False),  # 输入通道改为channel
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConvWithReLU(channel, channel, kernel_size=3, padding=1)
            )
        else:
            self.conv = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False), # 输入通道改为channel
                                      nn.BatchNorm2d(channel),
                                      nn.ReLU(inplace=True))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, sim):
        # 修复: 合并max和avg结果后再调用MLP，减少冗余计算
        max_self = self.max_pool(x)
        avg_self = self.avg_pool(x)
        channel_self = self.sigmoid(self.mlp1(torch.cat([max_self, avg_self], dim=1)))
        
        # 修复8: 使用注意力加权融合而非简单相乘
        # 相似度图归一化后作为权重，增强融合效果
        sim_weight = torch.sigmoid(sim)  # 归一化相似度到(0,1)
        
        # 修复5: 使用加权融合而非拼接，减少计算量
        # x_self = channel_self * x
        # x_sim = sim_weight * x
        # out = self.conv(torch.cat([x_self, x_sim], dim=1))
        
        # 融合: 通道注意力加权 + 相似度加权
        # 0.5是平衡因子
        fused = x * channel_self + x * sim_weight * 0.5
        out = self.conv(fused)
        
        max_out = self.max_pool(out)
        avg_out = self.avg_pool(out)
        channel_out = self.sigmoid(self.mlp2(torch.cat([max_out, avg_out], dim=1)))
        out = channel_out * out

        return out


class Interaction_Module(nn.Module):
    def __init__(self, channels, num_paths=2):
        super(Interaction_Module, self).__init__()
        self.num_paths = num_paths
        attn_channels = channels // 16
        attn_channels = max(attn_channels, 8)

        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x1, x2):

        x = torch.stack([x1, x2], dim=1)
        attn = x.sum(1).mean((2, 3), keepdim=True)
        attn = self.fc_reduce(attn)
        attn = self.act(attn)
        attn = self.fc_select(attn)
        B, C, H, W = attn.shape
        attn1, attn2 = attn.reshape(B, self.num_paths, C // self.num_paths, H, W).transpose(0, 1)
        attn1 = torch.sigmoid(attn1)
        attn2 = torch.sigmoid(attn2)

        return x1 * attn1, x2 * attn2



class GlobalLowDimensionalFeature(nn.Module):
    """
    全局低维度特征提取模块 (方案A)
    从多尺度特征中提取全局上下文信息，用于增强解码器性能
    
    设计思想:
    1. 全局池化各尺度特征
    2. 拼接后压缩到低维度 (32维)
    3. 在每个解码层融合全局信息
    """
    def __init__(self, in_channels_list=[64, 128, 256, 256], low_dim=32):
        super(GlobalLowDimensionalFeature, self).__init__()
        self.low_dim = low_dim
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 多尺度特征压缩到低维度
        total_channels = sum(in_channels_list)
        self.compress = nn.Sequential(
            nn.Conv2d(total_channels, low_dim * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_dim * 2, low_dim, kernel_size=1, bias=False)
        )
    
    def forward(self, d2, d3, d4, d5):
        """
        Args:
            d2: [B, 64, H/4, W/4]
            d3: [B, 128, H/8, W/8]
            d4: [B, 256, H/16, W/16]
            d5: [B, 256, H/32, W/32]
        
        Returns:
            global_feat: [B, low_dim, 1, 1] 全局低维度特征
        """
        # 全局池化各尺度特征
        d2_pool = self.global_pool(d2)  # [B, 64, 1, 1]
        d3_pool = self.global_pool(d3)  # [B, 128, 1, 1]
        d4_pool = self.global_pool(d4)  # [B, 256, 1, 1]
        d5_pool = self.global_pool(d5)  # [B, 256, 1, 1]
        
        # 拼接所有全局特征
        global_feat = torch.cat([d2_pool, d3_pool, d4_pool, d5_pool], dim=1)
        
        # 压缩到低维度
        global_feat = self.compress(global_feat)  # [B, low_dim, 1, 1]
        
        return global_feat


class Decoder(nn.Module):
    def __init__(self, in_d, out_d):
        super(Decoder, self).__init__()
        self.in_d = in_d
        self.out_d = out_d

        self.conv5 = nn.Sequential(nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_d),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_d),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_d),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_d),
                                   nn.ReLU(inplace=True))

        self.cls = nn.Conv2d(self.in_d, self.out_d, kernel_size=1)

        self.SFM5 = Similarity_Fusion_Module(in_d)
        self.SFM4 = Similarity_Fusion_Module(in_d)
        self.SFM3 = Similarity_Fusion_Module(in_d)
        self.SFM2 = Similarity_Fusion_Module(in_d)

    def forward(self, d5, d4, d3, d2, sim5, sim4, sim3, sim2):

        d5 = self.conv5(d5)
        d5 = self.SFM5(d5, sim5)
        d5 = F.interpolate(d5, d4.size()[2:], mode='bilinear')

        d4 = self.conv4(d4 + d5)
        d4 = self.SFM4(d4, sim4)
        d4 = F.interpolate(d4, d3.size()[2:], mode='bilinear')

        d3 = self.conv3(d3 + d4)
        d3 = self.SFM3(d3, sim3)
        d3 = F.interpolate(d3, d2.size()[2:], mode='bilinear')

        d2 = self.conv2(d2 + d3)
        d2 = self.SFM2(d2, sim2)

        mask = self.cls(d2)

        return mask


class Decoder_MultiScale(nn.Module):
    """Decoder with multi-scale channel support for DGMA2"""
    def __init__(self, in_d_list=[64, 128, 256, 256], out_d=2, use_depthwise=True):
        super(Decoder_MultiScale, self).__init__()
        self.in_d_list = in_d_list  # [d2, d3, d4, d5]
        self.out_d = out_d
        
        # Process each scale with its own channel count
        # 使用深度可分离卷积替换3×3卷积
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
        
        # 修复方案: 融合层使用1.5倍中间通道数（保持与预训练权重兼容）
        # 这样既能增强特征交互，又能兼容现有的预训练权重
        self.fuse4 = nn.Sequential(
            nn.Conv2d(in_d_list[3] + in_d_list[2], int(in_d_list[2] * 1.5), kernel_size=1),
            nn.BatchNorm2d(int(in_d_list[2] * 1.5)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_d_list[2] * 1.5), in_d_list[2], kernel_size=1)
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(in_d_list[2] + in_d_list[1], int(in_d_list[1] * 1.5), kernel_size=1),
            nn.BatchNorm2d(int(in_d_list[1] * 1.5)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_d_list[1] * 1.5), in_d_list[1], kernel_size=1)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(in_d_list[1] + in_d_list[0], int(in_d_list[0] * 1.5), kernel_size=1),
            nn.BatchNorm2d(int(in_d_list[0] * 1.5)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_d_list[0] * 1.5), in_d_list[0], kernel_size=1)
        )
        
        # Final classifier
        self.cls = nn.Conv2d(in_d_list[0], out_d, kernel_size=1)
        
        # Similarity fusion modules
        self.SFM5 = Similarity_Fusion_Module(in_d_list[3])
        self.SFM4 = Similarity_Fusion_Module(in_d_list[2])
        self.SFM3 = Similarity_Fusion_Module(in_d_list[1])
        self.SFM2 = Similarity_Fusion_Module(in_d_list[0])
        
        # === 新增: 方案A - 全局低维度特征 ===
        self.global_low_dim = GlobalLowDimensionalFeature(in_d_list, low_dim=32)
        
        # 全局特征融合层 (在每个尺度)
        self.global_fusion_d5 = nn.Sequential(
            nn.Conv2d(in_d_list[3] + 32, in_d_list[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(in_d_list[3]),
            nn.ReLU(inplace=True)
        )
        self.global_fusion_d4 = nn.Sequential(
            nn.Conv2d(in_d_list[2] + 32, in_d_list[2], kernel_size=1, bias=False),
            nn.BatchNorm2d(in_d_list[2]),
            nn.ReLU(inplace=True)
        )
        self.global_fusion_d3 = nn.Sequential(
            nn.Conv2d(in_d_list[1] + 32, in_d_list[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(in_d_list[1]),
            nn.ReLU(inplace=True)
        )
        self.global_fusion_d2 = nn.Sequential(
            nn.Conv2d(in_d_list[0] + 32, in_d_list[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(in_d_list[0]),
            nn.ReLU(inplace=True)
        )
        
        # 全局特征权重 (可学习)
        self.global_weight = nn.Parameter(torch.tensor(0.1))
        
        # 优化: 残差连接权重尺度自适应初始化
        # 粗尺度使用更大的残差权重（保留更多原始特征）
        # 细尺度使用较小的残差权重（更依赖融合后的特征）
        self.residual_alpha_d4 = nn.Parameter(torch.tensor(0.4))  # 1/16尺度，粗尺度
        self.residual_alpha_d3 = nn.Parameter(torch.tensor(0.3))  # 1/8尺度，中间尺度
        self.residual_alpha_d2 = nn.Parameter(torch.tensor(0.2))  # 1/4尺度，细尺度

        # === 新增: 反馈路径（Bottom-Up） ===
        # 1. 反馈压缩模块（降维避免参数爆炸）
        feedback_dim = 32  # 低维度反馈
        
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
        
        # 2. 反馈融合模块（注入到下一尺度）
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
        
        # 3. 反馈权重（可学习的门控）
        self.feedback_gate_d3 = nn.Parameter(torch.tensor(0.1))
        self.feedback_gate_d4 = nn.Parameter(torch.tensor(0.1))
        self.feedback_gate_d5 = nn.Parameter(torch.tensor(0.1))
        
        # === 新增: 多尺度深度监督 ===
        self.aux_cls_d3 = nn.Conv2d(in_d_list[1], out_d, 1)  # 辅助监督1/8
        self.aux_cls_d4 = nn.Conv2d(in_d_list[2], out_d, 1)  # 辅助监督1/16
        
        # === 新增: 多尺度特征聚合分类器 ===
        # 聚合所有尺度特征到d2分辨率
        self.multi_scale_aggregator = nn.Sequential(
            nn.Conv2d(sum(in_d_list), in_d_list[0], 1, bias=False),
            nn.BatchNorm2d(in_d_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_d_list[0], in_d_list[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_d_list[0]),
            nn.ReLU(inplace=True)
        )

    def forward(self, d5, d4, d3, d2, sim5, sim4, sim3, sim2):
        # ====== 阶段0: 提取全局低维度特征 (方案A) ======
        global_feat = self.global_low_dim(d2, d3, d4, d5)  # [B, 32, 1, 1]
        
        # ====== 阶段1: Top-Down Fusion (从粗到细) ======
        # Process d5
        d5 = self.conv5(d5)
        
        # 融合全局特征
        global_feat_d5 = F.interpolate(global_feat, d5.size()[2:], mode='bilinear', align_corners=True)
        d5 = torch.cat([d5, global_feat_d5], dim=1)
        d5 = self.global_fusion_d5(d5)
        
        d5 = self.SFM5(d5, sim5)
        d5_up = F.interpolate(d5, d4.size()[2:], mode='bilinear', align_corners=True)
        
        # Process d4 with d5 + 残差连接
        d4_input = d4  # 保存原始特征用于残差连接
        d4 = self.conv4(d4)
        
        # 融合全局特征
        global_feat_d4 = F.interpolate(global_feat, d4.size()[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, global_feat_d4], dim=1)
        d4 = self.global_fusion_d4(d4)
        
        d4 = self.fuse4(torch.cat([d4, d5_up], dim=1))
        d4 = self.SFM4(d4, sim4)
        d4 = d4 + d4_input * self.residual_alpha_d4  # 方案1: 残差连接
        d4_up = F.interpolate(d4, d3.size()[2:], mode='bilinear', align_corners=True)
        
        # Process d3 with d4 + 残差连接
        d3_input = d3  # 保存原始特征用于残差连接
        d3 = self.conv3(d3)
        
        # 融合全局特征
        global_feat_d3 = F.interpolate(global_feat, d3.size()[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, global_feat_d3], dim=1)
        d3 = self.global_fusion_d3(d3)
        
        d3 = self.fuse3(torch.cat([d3, d4_up], dim=1))
        d3 = self.SFM3(d3, sim3)
        d3 = d3 + d3_input * self.residual_alpha_d3  # 方案1: 残差连接
        d3_up = F.interpolate(d3, d2.size()[2:], mode='bilinear', align_corners=True)
        
        # Process d2 with d3 + 残差连接
        d2_input = d2  # 保存原始特征用于残差连接
        d2 = self.conv2(d2)
        
        # 融合全局特征
        global_feat_d2 = F.interpolate(global_feat, d2.size()[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, global_feat_d2], dim=1)
        d2 = self.global_fusion_d2(d2)
        
        d2 = self.fuse2(torch.cat([d2, d3_up], dim=1))
        d2 = self.SFM2(d2, sim2)
        d2 = d2 + d2_input * self.residual_alpha_d2  # 方案1: 残差连接
        
        # ====== 阶段2: Bottom-Up Feedback (新增) ======
        # d2 → d3 → d4 → d5 (反馈细节信息)
        # 修复: 改为上采样而非下采样，保留反馈信息
        
        # 1. d2反馈到d3
        fb_d2 = self.feedback_compress_d2(d2)
        fb_d2_up = F.interpolate(fb_d2, d3.size()[2:], mode='bilinear', align_corners=True)
        d3_with_fb = torch.cat([d3, fb_d2_up], dim=1)
        d3_refined = self.feedback_fusion_d3(d3_with_fb)
        d3 = d3 + d3_refined * self.feedback_gate_d3
        
        # 2. d3反馈到d4
        fb_d3 = self.feedback_compress_d3(d3)
        fb_d3_up = F.interpolate(fb_d3, d4.size()[2:], mode='bilinear', align_corners=True)
        d4_with_fb = torch.cat([d4, fb_d3_up], dim=1)
        d4_refined = self.feedback_fusion_d4(d4_with_fb)
        d4 = d4 + d4_refined * self.feedback_gate_d4
        
        # 3. d4反馈到d5
        fb_d4 = self.feedback_compress_d4(d4)
        fb_d4_up = F.interpolate(fb_d4, d5.size()[2:], mode='bilinear', align_corners=True)
        d5_with_fb = torch.cat([d5, fb_d4_up], dim=1)
        d5_refined = self.feedback_fusion_d5(d5_with_fb)
        d5 = d5 + d5_refined * self.feedback_gate_d5
        
        # ====== 阶段3: 多尺度聚合 + 深度监督 ======
        # 修复: 改为金字塔融合而非简单拼接，减少过度上采样
        
        # 金字塔融合：从粗到细逐步融合，需要先调整通道
        d5_up = F.interpolate(d5, d4.size()[2:], mode='bilinear', align_corners=True)
        # d5和d4通道可能不同，需要融合后再处理
        if d5_up.shape[1] != d4.shape[1]:
            d5_up = F.interpolate(d5_up, size=(d4.shape[2], d4.shape[3]), mode='bilinear', align_corners=True)
            # 通过融合层调整通道
            d54 = self.fuse4(torch.cat([d5_up, d4], dim=1))
        else:
            d54 = d4 + d5_up * 0.5
        
        d54_up = F.interpolate(d54, d3.size()[2:], mode='bilinear', align_corners=True)
        if d54_up.shape[1] != d3.shape[1]:
            d543 = self.fuse3(torch.cat([d54_up, d3], dim=1))
        else:
            d543 = d3 + d54_up * 0.5
        
        d543_up = F.interpolate(d543, d2.size()[2:], mode='bilinear', align_corners=True)
        if d543_up.shape[1] != d2.shape[1]:
            aggregated_feat = self.fuse2(torch.cat([d543_up, d2], dim=1))
        else:
            aggregated_feat = d2 + d543_up * 0.5
        
        # 最终分类
        mask = self.cls(aggregated_feat)
        
        # 辅助监督（训练时使用）
        if self.training:
            aux_mask_d3 = self.aux_cls_d3(d3)
            aux_mask_d4 = self.aux_cls_d4(d4)
            return mask, aux_mask_d3, aux_mask_d4
        else:
            return mask


class Decoder_sim(nn.Module):
    """
    相似度解码器：生成多尺度相似度图

    修复说明:
    - 支持多通道输入 in_d=[64, 128, 256, 512] 或单一通道 in_d=64
    - FEM模块统一特征到out_d通道（默认64）
    - 后续处理统一在out_d通道上进行
    """
    def __init__(self, in_d, out_d=64, use_depthwise=True):
        super(Decoder_sim, self).__init__()

        # 修复1: 支持多通道输入配置
        if isinstance(in_d, list):
            self.in_d_list = in_d  # [64, 128, 256, 512]
            self.use_multi_channel = True
        else:
            self.in_d_list = [in_d] * 4  # 向后兼容：统一通道
            self.use_multi_channel = False

        self.out_d = out_d

        # Feature Enhancement Module: 将多尺度特征统一到out_d通道
        self.FEM = Feature_Enhancement_Module(in_d=self.in_d_list, out_d=out_d)

        # 修复2: 后续卷积层使用统一的out_d通道
        # 使用深度可分离卷积替换3×3卷积
        if use_depthwise:
            self.conv4 = DepthwiseSeparableConvWithReLU(out_d, out_d, kernel_size=3, padding=1)
            self.conv3 = DepthwiseSeparableConvWithReLU(out_d, out_d, kernel_size=3, padding=1)
            self.conv2 = DepthwiseSeparableConvWithReLU(out_d, out_d, kernel_size=3, padding=1)
            self.conv1 = DepthwiseSeparableConvWithReLU(out_d, out_d, kernel_size=3, padding=1)
        else:
            self.conv4 = nn.Sequential(nn.Conv2d(out_d, out_d, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(out_d),
                                       nn.ReLU(inplace=True))
            self.conv3 = nn.Sequential(nn.Conv2d(out_d, out_d, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(out_d),
                                       nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(out_d, out_d, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(out_d),
                                       nn.ReLU(inplace=True))
            self.conv1 = nn.Sequential(nn.Conv2d(out_d, out_d, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(out_d),
                                       nn.ReLU(inplace=True))

        # Interaction Modules: 使用统一的out_d通道
        self.IM4 = Interaction_Module(channels=out_d)
        self.IM3 = Interaction_Module(channels=out_d)
        self.IM2 = Interaction_Module(channels=out_d)
        self.IM1 = Interaction_Module(channels=out_d)

    def cal_sim(self, x1, x2):
        # Calculate cosine similarity between features
        sim = F.cosine_similarity(x1, x2, dim=1)
        # Apply sigmoid to map to [0, 1]
        sim = torch.sigmoid(sim)
        # Return similarity directly without inversion
        return sim.unsqueeze(dim=1)

    def forward(self, x1_1, x1_2, x1_3, x1_4, x2_1, x2_2, x2_3, x2_4):

        x1_1, x1_2, x1_3, x1_4 = self.FEM(x1_1, x1_2, x1_3, x1_4)
        x2_1, x2_2, x2_3, x2_4 = self.FEM(x2_1, x2_2, x2_3, x2_4)

        x1_4 = self.conv4(x1_4)
        x2_4 = self.conv4(x2_4)
        x1_4, x2_4 = self.IM4(x1_4, x2_4)
        sim4 = self.cal_sim(x1_4, x2_4)
        x1_4 = F.interpolate(x1_4, x1_3.size()[2:], mode='bilinear')
        x2_4 = F.interpolate(x2_4, x2_3.size()[2:], mode='bilinear')

        x1_3 = self.conv3(x1_4 + x1_3)
        x2_3 = self.conv3(x2_4 + x2_3)
        x1_3, x2_3 = self.IM3(x1_3, x2_3)
        sim3 = self.cal_sim(x1_3, x2_3)
        x1_3 = F.interpolate(x1_3, x1_2.size()[2:], mode='bilinear')
        x2_3 = F.interpolate(x2_3, x2_2.size()[2:], mode='bilinear')

        x1_2 = self.conv2(x1_3 + x1_2)
        x2_2 = self.conv2(x2_3 + x2_2)
        x1_2, x2_2 = self.IM2(x1_2, x2_2)
        sim2 = self.cal_sim(x1_2, x2_2)
        x1_2 = F.interpolate(x1_2, x1_1.size()[2:], mode='bilinear')
        x2_2 = F.interpolate(x2_2, x2_1.size()[2:], mode='bilinear')

        x1_1 = self.conv1(x1_2 + x1_1)
        x2_1 = self.conv1(x2_2 + x2_1)
        x1_1, x2_1 = self.IM1(x1_1, x2_1)
        sim1 = self.cal_sim(x1_1, x2_1)

        return sim4, sim3, sim2, sim1


if __name__ == '__main__':
    # 测试 Decoder_sim
    model = Decoder_sim(in_d=[64, 128, 256, 512], out_d=64)
    x1_1 = torch.randn((2, 64, 64, 64))
    x1_2 = torch.randn((2, 128, 32, 32))
    x1_3 = torch.randn((2, 256, 16, 16))
    x1_4 = torch.randn((2, 512, 8, 8))
    x2_1 = torch.randn((2, 64, 64, 64))
    x2_2 = torch.randn((2, 128, 32, 32))
    x2_3 = torch.randn((2, 256, 16, 16))
    x2_4 = torch.randn((2, 512, 8, 8))

    model.eval()
    with torch.no_grad():
        sim4, sim3, sim2, sim1 = model(x1_1, x1_2, x1_3, x1_4, x2_1, x2_2, x2_3, x2_4)
    print(f"sim4: {sim4.shape}, sim3: {sim3.shape}, sim2: {sim2.shape}, sim1: {sim1.shape}")
