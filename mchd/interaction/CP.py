from torch import nn
import torch.nn.functional as F
import torch


class CascadePyramid(nn.Module):
    def __init__(self):
        super().__init__()
        # 空洞卷积层（调整输出维度）
        self.conv_stage = nn.Sequential(
            nn.Conv1d(256, 256, 3, dilation=2, padding=2),  # 保持输出维度256
            nn.GELU(),
            nn.BatchNorm1d(256)
        )

        # 多尺度池化分支
        self.pool_stage = nn.ModuleDict({
            'avg': nn.AvgPool1d(3, stride=1, padding=1),
            'max': nn.MaxPool1d(5, stride=1, padding=2)
        })

        # 注意力机制（维度匹配关键）
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)  # 输入输出dim=256

        # 投影层（新增独立线性层）
        self.proj = nn.Linear(256, 256)  # 确保维度一致

    def forward(self, x):
        # 输入维度转换 (seq_len, batch, dim) -> (batch, dim, seq_len)
        x_t = x.permute(1, 2, 0)  # [32, 256, 75]

        # 空洞卷积处理
        conv_out = self.conv_stage(x_t)  # [32, 256, 75]

        # 双路池化处理
        pool_avg = self.pool_stage['avg'](conv_out)  # [32, 256, 75]
        pool_max = self.pool_stage['max'](conv_out)  # [32, 256, 75]

        # 调整维度顺序用于注意力 (seq_len, batch, dim)
        pool_avg = pool_avg.permute(2, 0, 1)  # [75, 32, 256]
        pool_max = pool_max.permute(2, 0, 1)  # [75, 32, 256]

        # 交叉注意力计算
        attn_out, _ = self.attn(
            query=pool_avg,
            key=pool_max,
            value=pool_max
        )  # [75, 32, 256]

        # 恢复原始维度顺序 (batch, seq_len, dim)
        attn_out = attn_out.permute(1, 0, 2)  # [32, 75, 256]

        # 投影层处理
        projected = self.proj(attn_out)  # [32, 75, 256]

        # 残差连接（对齐原始输入维度）
        return projected + x.permute(1, 0, 2)  # [32, 75, 256]

# x = torch.randn(75, 32, 256)
# model = CascadePyramid()
# out = model(x)
# print(out.shape)