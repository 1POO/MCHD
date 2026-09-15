import torch
import torch.nn as nn

class FeatureProjection(nn.Module):
    def __init__(self, obj_dim, txt_dim, mask_dim, hidden_dim):
        super().__init__()
        self.obj_proj = nn.Linear(obj_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)
        self.mask_proj = nn.Linear(mask_dim, hidden_dim)  # 若mask是二值掩码，可替换为其他操作
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, obj_feat, txt_feat, mask_feat):
        obj = self.norm(self.obj_proj(obj_feat))  # [B, N_obj, D]
        txt = self.norm(self.txt_proj(txt_feat))  # [B, N_txt, D]
        mask = self.norm(self.mask_proj(mask_feat))  # [B, N_obj, D]
        return obj, txt, mask
class CrossTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.cross_attn1 = nn.MultiheadAttention(hidden_dim, num_heads)
        self.cross_attn2 = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.GELU(),
            nn.Linear(4*hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, obj, txt, mask):
        # 交叉注意力1：文本Query，Object Key/Value + mask
        txt_attn, _ = self.cross_attn1(
            query=txt.transpose(0, 1),
            key=obj.transpose(0, 1),
            value=obj.transpose(0, 1),
            key_padding_mask=mask.squeeze(-1)  # 若mask是二值掩码
        )
        txt_out = self.norm1(txt + txt_attn.transpose(0, 1))

        # 交叉注意力2：Object Query，文本Key/Value + 反向mask
        obj_attn, _ = self.cross_attn2(
            query=obj.transpose(0, 1),
            key=txt.transpose(0, 1),
            value=txt.transpose(0, 1),
            key_padding_mask=(1 - mask.squeeze(-1))  # 反向mask（可选）
        )
        obj_out = self.norm2(obj + obj_attn.transpose(0, 1))

        # FFN融合
        obj_out = self.ffn(obj_out)
        txt_out = self.ffn(txt_out)
        
        return obj_out, txt_out
class CrossTransformer(nn.Module):
    def __init__(self, obj_dim, txt_dim, mask_dim, hidden_dim=256, num_layers=3, num_heads=4):
        super().__init__()
        self.projection = FeatureProjection(obj_dim, txt_dim, mask_dim, hidden_dim)
        self.layers = nn.ModuleList([
            CrossTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, obj_feat, txt_feat, mask_feat):
        obj, txt, mask = self.projection(obj_feat, txt_feat, mask_feat)
        
        for layer in self.layers:
            obj, txt = layer(obj, txt, mask)
        
        # 特征聚合（示例：均值池化）
        fused_obj = self.pool(obj.transpose(1, 2)).squeeze(-1)  # [B, D]
        fused_txt = self.pool(txt.transpose(1, 2)).squeeze(-1)  # [B, D]
        fused_feat = torch.cat([fused_obj, fused_txt], dim=-1)  # [B, 2D]
        
        return fused_feat