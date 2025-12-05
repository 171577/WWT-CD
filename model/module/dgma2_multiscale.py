import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.module.dgma2_components import DAMBlock, DEAM_with_Swsi, MDFMBlock, DAMBlockV2, DEAM_with_Swsi_V2
    from model.module.innovative_modules import InnovativeComponentsIntegration
except ImportError:
    from dgma2_components import DAMBlock, DEAM_with_Swsi, MDFMBlock, DAMBlockV2, DEAM_with_Swsi_V2
    from innovative_modules import InnovativeComponentsIntegration

# 尝试导入跨尺度注意力模块
try:
    from model.module.cross_scale_attention import LightweightCrossScaleAttention
except ImportError:
    try:
        from cross_scale_attention import LightweightCrossScaleAttention
    except ImportError:
        LightweightCrossScaleAttention = None


class DGMA2Supervised_MultiScale(nn.Module):
    def __init__(
            self,
            in_channels_list=[64, 128, 256, 512],
            out_channels_list=[64, 128, 256, 256],
            dam_per_scale=2,
            beta_init=0.3,
            use_depthwise=True,
            use_cross_scale_attn=False,  # 新增: 是否使用跨尺度注意力
            use_improved_dam=False  # 新增: 是否使用改进的DAMBlock
    ):
        super(DGMA2Supervised_MultiScale, self).__init__()
        self.use_cross_scale_attn = use_cross_scale_attn

        # 优化: 直接使用MDFMBlock，移除mid_channel参数（已在内部优化）
        self.mdfm2 = MDFMBlock(in_channels_list[0], out_channels_list[0], use_depthwise=use_depthwise)  # 1/4
        self.mdfm3 = MDFMBlock(in_channels_list[1], out_channels_list[1], use_depthwise=use_depthwise)  # 1/8
        self.mdfm4 = MDFMBlock(in_channels_list[2], out_channels_list[2], use_depthwise=use_depthwise)  # 1/16
        self.mdfm5 = MDFMBlock(in_channels_list[3], out_channels_list[3], use_depthwise=use_depthwise)  # 1/32

        # DAM blocks (stackable) - 传递 use_depthwise 参数
        # 可选使用改进版DAMBlock
        dam_block_class = DAMBlockV2 if use_improved_dam else DAMBlock
        self.dam_blocks = nn.ModuleList([
            dam_block_class(base_dim=64, out_d=out_channels_list, use_depthwise=use_depthwise)
            for _ in range(dam_per_scale)
        ])

        # 跨尺度注意力桥接 (可选)
        if use_cross_scale_attn and LightweightCrossScaleAttention is not None:
            self.cross_scale_bridge = LightweightCrossScaleAttention(out_channels_list)

        # [创新] 集成频域感知和动态门控模块
        self.innovative_components = InnovativeComponentsIntegration(
            in_channels_list=in_channels_list,
            out_channels_list=out_channels_list
        )

        self.deam2 = DEAM_with_Swsi(
            in_channels_list[0], out_channels_list[0],
            ds=4,  # 1/4尺度，细节丰富
            beta_init=0.2  # 细节尺度，弱相似度引导
        )
        self.deam3 = DEAM_with_Swsi(
            in_channels_list[1], out_channels_list[1],
            ds=4,  # 1/8尺度
            beta_init=0.3  # 中间尺度
        )
        self.deam4 = DEAM_with_Swsi(
            in_channels_list[2], out_channels_list[2],
            ds=8,  # 1/16尺度
            beta_init=0.4  # 粗尺度
        )
        self.deam5 = DEAM_with_Swsi(
            in_channels_list[3], out_channels_list[3],
            ds=8,  # 1/32尺度，最粗
            beta_init=0.5  # 最强相似度引导
        )

    def forward(self, xr1_1, xr1_2, xr1_3, xr1_4, xr2_1, xr2_2, xr2_3, xr2_4, s_wsi=None, s_wsi_list=None):
        # Stage 1: MDFM with InceptionMultiScale - Extract multi-scale difference features
        d2 = self.mdfm2(xr1_1, xr2_1)  # 1/4
        d3 = self.mdfm3(xr1_2, xr2_2)  # 1/8
        d4 = self.mdfm4(xr1_3, xr2_3)  # 1/16
        d5 = self.mdfm5(xr1_4, xr2_4)  # 1/32

        # Stage 2: DAM - Aggregate and refine difference features
        for dam in self.dam_blocks:
            d5, d4, d3, d2 = dam(d5, d4, d3, d2)

        # Stage 2.5: [创新] 应用频域感知和动态门控
        # 在DAM之后、DEAM之前应用，增强特征表达
        d2, d3, d4, d5 = self.innovative_components(
            d2, d3, d4, d5,
            xr1_1, xr1_2, xr1_3, xr1_4,
            xr2_1, xr2_2, xr2_3, xr2_4
        )

        # Stage 2.6: [新增] 跨尺度注意力桥接
        if self.use_cross_scale_attn:
            d2, d3, d4, d5 = self.cross_scale_bridge(d2, d3, d4, d5)

        # Stage 3: DEAM - Cross-temporal attention with similarity guidance
        # 优化调用顺序: 先粗后细 (从d5到d2)
        # Use multi-scale similarity masks if provided, otherwise fall back to single mask
        if s_wsi_list is not None:
            sim_mask_4, sim_mask_3, sim_mask_2, sim_mask_1 = s_wsi_list
            dr4 = self.deam5(xr1_4, d5, sim_mask_4)  # 1/32 (粗尺度优先)
            dr3 = self.deam4(xr1_3, d4, sim_mask_3)  # 1/16
            dr2 = self.deam3(xr1_2, d3, sim_mask_2)  # 1/8
            dr1 = self.deam2(xr1_1, d2, sim_mask_1)  # 1/4 (细尺度最后)
        else:
            dr4 = self.deam5(xr1_4, d5, s_wsi)  # 1/32
            dr3 = self.deam4(xr1_3, d4, s_wsi)  # 1/16
            dr2 = self.deam3(xr1_2, d3, s_wsi)  # 1/8
            dr1 = self.deam2(xr1_1, d2, s_wsi)  # 1/4

        return dr4, dr3, dr2, dr1


class DGMA2Supervised_MultiScale_Attention(nn.Module):
    def __init__(
            self,
            in_channels_list=[64, 128, 256, 512],
            out_channels_list=[64, 128, 256, 256],
            dam_per_scale=2,
            beta_init=0.3,
            use_depthwise=True,
            use_cross_scale_attn=False,  # 新增: 是否使用跨尺度注意力
            use_improved_dam=False  # 新增: 是否使用改进的DAMBlock
    ):
        super(DGMA2Supervised_MultiScale_Attention, self).__init__()
        self.use_cross_scale_attn = use_cross_scale_attn

        # MDFM with InceptionMultiScale + Attention for each scale (adjusted mid_channel for better feature extraction)
        # 传递 use_depthwise 参数到 MDFMBlock
        # 优化: 直接使用MDFMBlock，移除mid_channel参数
        self.mdfm2 = MDFMBlock(in_channels_list[0], out_channels_list[0], use_depthwise=use_depthwise)  # 1/4
        self.mdfm3 = MDFMBlock(in_channels_list[1], out_channels_list[1], use_depthwise=use_depthwise)  # 1/8
        self.mdfm4 = MDFMBlock(in_channels_list[2], out_channels_list[2], use_depthwise=use_depthwise)  # 1/16
        self.mdfm5 = MDFMBlock(in_channels_list[3], out_channels_list[3], use_depthwise=use_depthwise)  # 1/32

        # DAM blocks (stackable) - 传递 use_depthwise 参数
        dam_block_class = DAMBlockV2 if use_improved_dam else DAMBlock
        self.dam_blocks = nn.ModuleList([
            dam_block_class(base_dim=64, out_d=out_channels_list, use_depthwise=use_depthwise)
            for _ in range(dam_per_scale)
        ])

        # [创新] 集成频域感知和动态门控模块
        self.innovative_components = InnovativeComponentsIntegration(
            in_channels_list=in_channels_list,
            out_channels_list=out_channels_list
        )

        # 跨尺度注意力桥接 (可选)
        if use_cross_scale_attn and LightweightCrossScaleAttention is not None:
            self.cross_scale_bridge = LightweightCrossScaleAttention(out_channels_list)

        # 优化: DEAM使用尺度自适应的下采样因子和Beta参数
        # 细尺度(d2, d3): ds=4保留更多空间细节，beta较小(0.2-0.3)
        # 粗尺度(d4, d5): ds=8减少计算量，beta较大(0.4-0.5)增强相似度引导
        self.deam2 = DEAM_with_Swsi(in_channels_list[0], out_channels_list[0], ds=4, beta_init=0.2)  # 1/4，细节尺度
        self.deam3 = DEAM_with_Swsi(in_channels_list[1], out_channels_list[1], ds=4, beta_init=0.3)  # 1/8，中间尺度
        self.deam4 = DEAM_with_Swsi(in_channels_list[2], out_channels_list[2], ds=8, beta_init=0.4)  # 1/16，语义尺度
        self.deam5 = DEAM_with_Swsi(in_channels_list[3], out_channels_list[3], ds=8, beta_init=0.5)  # 1/32，全局尺度

    def forward(self, xr1_1, xr1_2, xr1_3, xr1_4, xr2_1, xr2_2, xr2_3, xr2_4, s_wsi=None, s_wsi_list=None):
        # Stage 1: MDFM with InceptionMultiScale + Attention
        d2 = self.mdfm2(xr1_1, xr2_1)  # 1/4
        d3 = self.mdfm3(xr1_2, xr2_2)  # 1/8
        d4 = self.mdfm4(xr1_3, xr2_3)  # 1/16
        d5 = self.mdfm5(xr1_4, xr2_4)  # 1/32

        # Stage 2: DAM - Aggregate and refine difference features
        for dam in self.dam_blocks:
            d5, d4, d3, d2 = dam(d5, d4, d3, d2)

        # Stage 2.5: [创新] 应用频域感知和动态门控
        # 在DAM之后、DEAM之前应用，增强特征表达
        d2, d3, d4, d5 = self.innovative_components(
            d2, d3, d4, d5,
            xr1_1, xr1_2, xr1_3, xr1_4,
            xr2_1, xr2_2, xr2_3, xr2_4
        )

        # Stage 2.6: [新增] 跨尺度注意力桥接
        if self.use_cross_scale_attn:
            d2, d3, d4, d5 = self.cross_scale_bridge(d2, d3, d4, d5)

        # Stage 3: DEAM - Cross-temporal attention with similarity guidance
        # 优化调用顺序: 先粗后细 (从d5到d2)
        # Use multi-scale similarity masks if provided, otherwise fall back to single mask
        if s_wsi_list is not None:
            sim_mask_4, sim_mask_3, sim_mask_2, sim_mask_1 = s_wsi_list
            dr4 = self.deam5(xr1_4, d5, sim_mask_4)  # 1/32 (粗尺度优先)
            dr3 = self.deam4(xr1_3, d4, sim_mask_3)  # 1/16
            dr2 = self.deam3(xr1_2, d3, sim_mask_2)  # 1/8
            dr1 = self.deam2(xr1_1, d2, sim_mask_1)  # 1/4 (细尺度最后)
        else:
            dr4 = self.deam5(xr1_4, d5, s_wsi)  # 1/32
            dr3 = self.deam4(xr1_3, d4, s_wsi)  # 1/16
            dr2 = self.deam3(xr1_2, d3, s_wsi)  # 1/8
            dr1 = self.deam2(xr1_1, d2, s_wsi)  # 1/4
        return dr4, dr3, dr2, dr1