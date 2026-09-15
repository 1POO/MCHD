import torch
import torch.nn as nn
import torch.nn.functional as F

class UniformTriCrossAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"

        # 为每个模态定义独立的Q/K/V投影层
        self.proj_q = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.proj_k = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.proj_v = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])

        # 输出层与权重融合
        self.fuse_weights = nn.Parameter(torch.ones(3))  # 可学习的模态融合权重
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2, x3,dim=256):
        """
        输入形状:
        x1, x2, x3 均为 (batch_size=32, seq_len=75, feat_dim=256)
        """
        bsz = x1.size(0)
        seq_len = x1.size(1)
        modalities = [x1, x2, x3]
        outputs = []

        # 为每个模态生成Q/K/V
        q_list = [proj(modality) for proj, modality in zip(self.proj_q, modalities)]
        k_list = [proj(modality) for proj, modality in zip(self.proj_k, modalities)]
        v_list = [proj(modality) for proj, modality in zip(self.proj_v, modalities)]

        # 对每个模态计算作为Query时的交叉注意力
        for i in range(3):
            # 当前模态的Query
            q = q_list[i]  # (32, 75, 256)
            q = q.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (32, h, 75, d_k)

            # 其他两个模态的Key和Value拼接
            other_k = torch.cat([k_list[j] for j in range(3) if j != i], dim=1)  # (32, 150, 256)
            other_v = torch.cat([v_list[j] for j in range(3) if j != i], dim=1)

            # 投影到多头空间
            k = other_k.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (32, h, 150, d_k)
            v = other_v.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

            # 计算注意力分数
            # 正确无隐藏字符的注意力计算
            attn_scores = torch.matmul(q, k.transpose(-2, -1))/(self.head_dim ** 0.5)  # (32, h, 75, 150)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # 加权求和
            attn_output = torch.matmul(attn_weights, v)  # (32, h, 75, d_k)
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
            outputs.append(attn_output)

        # 自适应加权融合
        weights = F.softmax(self.fuse_weights, dim=0)
        fused_output = sum(w * out for w, out in zip(weights, outputs))  # (32, 75, 256)

        # 最终投影
        return fused_output

# 测试用例
if __name__ == "__main__":
    # 输入均为 (32, 75, 256)
    x1 = torch.randn(32, 75, 256)
    x2 = torch.randn(32, 75, 256)
    x3 = torch.randn(32, 75, 256)

    attn = UniformTriCrossAttention()
    output = attn(x1, x2, x3)
    print(output.shape)  # 应输出 torch.Size([32, 75, 256])