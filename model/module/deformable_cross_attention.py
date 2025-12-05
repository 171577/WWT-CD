"""
创新模块1: 可变形交叉注意力
- 自适应学习时相间的对应关系
- 处理非刚性变形和配准误差
"""
import torch
import torch. nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


class DeformableCrossAttention(nn.Module):
    """
    可变形交叉时相注意力模块
    
    核心创新:
    1. 学习时相间的空间偏移量，处理配准误差
    2. 多头注意力捕获不同类型的变化模式
    3.  动态采样点权重，聚焦于变化区域
    """
    def __init__(self, d_model, n_heads=8, n_points=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads
        
        # 偏移量预测网络
        self. offset_net = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, 3, 1, 1),
            nn.GroupNorm(8, d_model),
            nn. GELU(),
            nn.Conv2d(d_model, n_heads * n_points * 2, 3, 1, 1)  # 2 for (dx, dy)
        )
        
        # 采样点权重
        self.attention_weights = nn. Sequential(
            nn. Conv2d(d_model * 2, d_model, 3, 1, 1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            nn.Conv2d(d_model, n_heads * n_points, 1)
        )
        
        # 投影层
        self.query_proj = nn. Conv2d(d_model, d_model, 1)
        self.value_proj = nn. Conv2d(d_model, d_model, 1)
        self.output_proj = nn. Conv2d(d_model, d_model, 1)
        
        # 门控机制
        self.gate = nn.Sequential(
            nn. Conv2d(d_model * 2, d_model, 1),
            nn. Sigmoid()
        )
        
        self.dropout = nn. Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        # 初始化偏移量为0（无偏移）
        constant_(self.offset_net[-1].weight, 0.)
        constant_(self.offset_net[-1].bias, 0.)
        
        # 初始化注意力权重均匀
        constant_(self.attention_weights[-1].weight, 0.)
        constant_(self. attention_weights[-1].bias, 0.)
    
    def forward(self, x1, x2, return_offsets=False):
        """
        Args:
            x1: 时相1特征 [B, C, H, W]
            x2: 时相2特征 [B, C, H, W]
        Returns:
            x1_enhanced, x2_enhanced: 增强后的特征
        """
        B, C, H, W = x1. shape
        
        # 拼接特征用于预测偏移和权重
        x_cat = torch.cat([x1, x2], dim=1)
        
        # 预测偏移量 [B, n_heads * n_points * 2, H, W]
        offsets = self.offset_net(x_cat)
        offsets = offsets. view(B, self.n_heads, self.n_points, 2, H, W)
        offsets = offsets. permute(0, 1, 2, 4, 5, 3)  # [B, n_heads, n_points, H, W, 2]
        
        # 限制偏移范围
        offsets = offsets.tanh() * 3  # 最大偏移3个像素
        
        # 预测注意力权重 [B, n_heads, n_points, H, W]
        attn_weights = self.attention_weights(x_cat)
        attn_weights = attn_weights.view(B, self.n_heads, self.n_points, H, W)
        attn_weights = F.softmax(attn_weights, dim=2)
        
        # 生成采样网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x1.device),
            torch. linspace(-1, 1, W, device=x1. device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).unsqueeze(0). unsqueeze(0)  # [1, 1, 1, H, W, 2]
        
        # Value投影
        v2 = self.value_proj(x2). view(B, self.n_heads, self.head_dim, H, W)
        v1 = self.value_proj(x1).view(B, self.n_heads, self.head_dim, H, W)
        
        # 可变形采样
        x1_sampled = self._deformable_sample(v2, grid, offsets, attn_weights)
        x2_sampled = self._deformable_sample(v1, grid, -offsets, attn_weights)  # 反向偏移
        
        # 重塑
        x1_sampled = x1_sampled.view(B, C, H, W)
        x2_sampled = x2_sampled.view(B, C, H, W)
        
        # 门控融合
        gate1 = self.gate(torch.cat([x1, x1_sampled], dim=1))
        gate2 = self.gate(torch. cat([x2, x2_sampled], dim=1))
        
        x1_enhanced = x1 + self.dropout(self.output_proj(x1_sampled)) * gate1
        x2_enhanced = x2 + self.dropout(self. output_proj(x2_sampled)) * gate2
        
        if return_offsets:
            return x1_enhanced, x2_enhanced, offsets
        return x1_enhanced, x2_enhanced
    
    def _deformable_sample(self, value, grid, offsets, weights):
        """可变形采样"""
        B, n_heads, head_dim, H, W = value.shape
        n_points = offsets.shape[2]
        
        # 归一化偏移量
        offsets_normalized = offsets. clone()
        offsets_normalized[..., 0] = offsets[..., 0] / (W / 2)
        offsets_normalized[..., 1] = offsets[.. ., 1] / (H / 2)
        
        # 对每个采样点进行采样
        sampled_values = []
        for p in range(n_points):
            sample_grid = grid + offsets_normalized[:, :, p:p+1]
            sample_grid = sample_grid. squeeze(2)  # [B, n_heads, H, W, 2]
            
            # 对每个head单独采样
            sampled = []
            for h in range(n_heads):
                v_h = value[:, h]  # [B, head_dim, H, W]
                g_h = sample_grid[:, h]  # [B, H, W, 2]
                s_h = F.grid_sample(v_h, g_h, mode='bilinear', padding_mode='border', align_corners=True)
                sampled.append(s_h)
            sampled = torch.stack(sampled, dim=1)  # [B, n_heads, head_dim, H, W]
            sampled_values.append(sampled)
        
        sampled_values = torch.stack(sampled_values, dim=2)  # [B, n_heads, n_points, head_dim, H, W]
        
        # 加权求和
        weights = weights.unsqueeze(3)  # [B, n_heads, n_points, 1, H, W]
        output = (sampled_values * weights).sum(dim=2)  # [B, n_heads, head_dim, H, W]
        
        return output


class TemporalDifferenceTransformer(nn. Module):
    """
    时序差异Transformer
    
    创新点:
    1. 将变化检测视为序列到序列的变换问题
    2.  使用可学习的变化查询向量
    3.  分层交叉注意力捕获多粒度变化
    """
    def __init__(self, d_model, n_heads=8, n_layers=2, n_change_queries=16):
        super().__init__()
        self.d_model = d_model
        self.n_change_queries = n_change_queries
        
        # 可学习的变化查询
        self.change_queries = nn.Parameter(torch.randn(1, n_change_queries, d_model))
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        # 变化预测头
        self.change_head = nn.Sequential(
            nn. Linear(d_model, d_model),
            nn. GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # 空间重建
        self.spatial_decoder = nn.Sequential(
            nn.Conv2d(n_change_queries, d_model // 2, 1),
            nn.GroupNorm(8, d_model // 2),
            nn.GELU(),
            nn.Conv2d(d_model // 2, d_model, 1)
        )
    
    def forward(self, x1, x2):
        B, C, H, W = x1. shape
        
        # 展平特征
        x1_flat = x1.flatten(2). permute(0, 2, 1)  # [B, HW, C]
        x2_flat = x2.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        
        # 添加位置编码
        pos = self.pos_embed[:, :H*W, :]
        x1_flat = x1_flat + pos
        x2_flat = x2_flat + pos
        
        # 时序差异特征
        diff_feat = x1_flat - x2_flat  # [B, HW, C]
        
        # 扩展查询
        queries = self.change_queries.expand(B, -1, -1)  # [B, n_queries, C]
        
        # Transformer解码
        for layer in self.layers:
            queries = layer(queries, diff_feat, x1_flat, x2_flat)
        
        # 变化预测
        change_feat = self.change_head(queries)  # [B, n_queries, C]
        
        # 计算查询与空间位置的相似度
        similarity = torch.bmm(change_feat, diff_feat.permute(0, 2, 1))  # [B, n_queries, HW]
        similarity = similarity.view(B, self.n_change_queries, H, W)
        
        # 空间重建
        output = self.spatial_decoder(similarity)  # [B, C, H, W]
        
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=None, dropout=0. 1):
        super().__init__()
        dim_feedforward = dim_feedforward or d_model * 4
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # 交叉注意力（对差异特征）
        self.cross_attn_diff = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # 双时相交叉注意力
        self.cross_attn_t1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn_t2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # FFN
        self. ffn = nn.Sequential(
            nn. Linear(d_model, dim_feedforward),
            nn. GELU(),
            nn. Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn. Dropout(dropout)
        )
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self. norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn. LayerNorm(d_model)
        
        # 融合门控
        self.fusion_gate = nn. Sequential(
            nn.Linear(d_model * 2, d_model),
            nn. Sigmoid()
        )
    
    def forward(self, queries, diff_feat, x1_feat, x2_feat):
        # 自注意力
        q = self.norm1(queries)
        queries = queries + self.self_attn(q, q, q)[0]
        
        # 差异交叉注意力
        q = self.norm2(queries)
        queries = queries + self. cross_attn_diff(q, diff_feat, diff_feat)[0]
        
        # 双时相交叉注意力
        q = self.norm3(queries)
        attn_t1 = self. cross_attn_t1(q, x1_feat, x1_feat)[0]
        attn_t2 = self.cross_attn_t2(q, x2_feat, x2_feat)[0]
        
        # 门控融合
        gate = self.fusion_gate(torch. cat([attn_t1, attn_t2], dim=-1))
        queries = queries + gate * attn_t1 + (1 - gate) * attn_t2
        
        # FFN
        queries = queries + self.ffn(self.norm4(queries))
        
        return queries