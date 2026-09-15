from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# mask_logits 函数用于将输入 logits 中的某些部分通过 mask 进行屏蔽，常用于避免对无效区域进行计算
def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)  # 将 mask 转换为 float32 类型
    return inputs + (1.0 - mask) * mask_value  # 通过 mask 对无效区域进行屏蔽，赋予一个极小的值，避免计算影响

class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)

# VSL的吗？对
class CQAttention(nn.Module):
    def __init__(self, dim, drop_rate=0.1):
        super(CQAttention, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.cqa_linear= nn.LSTM(4 * dim,
        #                         dim ,
        #                         num_layers=1,
        #                         bidirectional=False,
        #                         dropout=drop_rate,
        #                         batch_first=True)

    def forward(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(context, query)  # (batch_size, c_seq_len, q_seq_len)
        score_ = nn.Softmax(dim=2)(mask_logits(score, q_mask.unsqueeze(1)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = score_t.transpose(1, 2)  # (batch_size, q_seq_len, c_seq_len)
        c2q = torch.matmul(score_, query)  # (batch_size, c_seq_len, dim)
        q2c = torch.matmul(torch.matmul(score_, score_t), context)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, c2q, torch.mul(context, c2q), torch.mul(context, q2c)], dim=2)
        output = self.cqa_linear(output) # (batch_size, c_seq_len, dim)
        # output = self.cqa_linear(output)[0] 使用LSTM
        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand([-1, -1, q_seq_len])  # (batch_size, c_seq_len, q_seq_len)
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand([-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)
        return res


class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(x, self.weight, dims=1)  # shape = (batch_size, seq_length, 1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)  # (batch_size, dim, 1)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x


class CQConcatenate(nn.Module):
    def __init__(self, dim):
        super(CQConcatenate, self).__init__()
        self.weighted_pool = WeightedPool(dim=dim)
        self.conv1d = Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, q_mask):
        pooled_query = self.weighted_pool(query, q_mask)  # (batch_size, dim)
        _, c_seq_len, _ = context.shape
        pooled_query = pooled_query.unsqueeze(1).repeat(1, c_seq_len, 1)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, pooled_query], dim=2)  # (batch_size, c_seq_len, 2*dim)
        output = self.conv1d(output)
        return output


class TemporalDeconv(nn.Module):
    def __init__(self, in_dim, out_dim, expansion=2):
        super().__init__()
        # 增加通道扩展层
        self.expand_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1
        )
        self.deconv = nn.ConvTranspose1d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

    def forward(self, x, target_len):
        """ 输入x形状: (B, seq_len, in_dim) """
        x = x.permute(0, 2, 1)  # (B, in_dim, seq_len)
        x = self.expand_conv(x)  # 通道扩展 (B, out_dim, seq_len)
        x = self.deconv(x)  # (B, out_dim, upsampled_len)
        x = x.permute(0, 2, 1)  # (B, upsampled_len, out_dim)

        # 动态调整长度
        if x.size(1) > target_len:
            x = x[:, :target_len, :]
        elif x.size(1) < target_len:
            pad = torch.zeros(x.size(0), target_len - x.size(1), x.size(2)).to(x.device)
            x = torch.cat([x, pad], dim=1)

        return self.act(self.norm(x))

class HierarchicalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 短语级注意力
        self.phrase_attn = CQAttention(dim)

        # 片段级投影
        self.video_proj = Conv1D(dim, dim, 1)
        self.text_proj = Conv1D(dim, dim, 1)

        # 新增相似度投影层
        self.sim_proj = Conv1D(dim, dim, 1)  # 根据分片维度调整
        # 修改后的反卷积模块
        self.deconv = TemporalDeconv(
            in_dim=1,  # 输入是相似度分数（标量）
            out_dim=dim  # 输出与原始特征维度一致
        )
        self.sim_proj = Conv1D(dim, dim, 1)

        # 全局池化
        self.global_pool = WeightedPool(dim)

    def _create_segments(self, x, mask, window_size, stride):
        """生成池化后的片段向量"""
        # 输入形状: (B, L, D)
        segments = x.unfold(1, window_size, stride)  # (B, num_seg, D, window_size)
        segments = segments.permute(0, 1, 3, 2)  # (B, num_seg, window_size, D)
        segments = segments.mean(dim=2)  # (B, num_seg, D)

        # 生成有效掩码
        mask_seg = mask.unfold(1, window_size, stride).float().sum(dim=2) > 0
        return segments * mask_seg.unsqueeze(-1)  # (B, num_seg, D)

    def forward(self, video, text, vmask, qmask):
        # 短语级交互
        phrase_out = self.phrase_attn(video, text, vmask, qmask)  # (B,75,256)

        # 视频片段处理 ---------------------------------------------------------
        v_proj = self.video_proj(video)  # (B,75,256)
        v_seg = self._create_segments(v_proj, vmask, 15, 15)  # (B,5,256)
        v_len = video.size(1)
        # 文本片段处理 ---------------------------------------------------------
        text_len = text.size(1)
        q_window = max(1, text_len // 5)  # 动态窗口计算
        q_proj = self.text_proj(text)  # (B,n,256)
        q_seg = self._create_segments(q_proj, qmask, q_window, q_window)  # (B,m,256)

        # 片段相似度计算 -------------------------------------------------------
        segment_sim = torch.einsum('bsd,btd->bst', v_seg, q_seg)  # (B,5,m)

        # 修正相似度矩阵处理
        B, num_vseg, num_qseg = segment_sim.shape
        # 添加通道维度 (B, num_vseg, num_qseg, 1)
        segment_out = segment_sim.unsqueeze(-1)
        # 调整形状 (B, num_vseg*num_qseg, 1)
        segment_out = segment_out.view(B, -1, 1)

        # 反卷积处理
        segment_out = self.deconv(segment_out, target_len=75)  # (B,75,dim)
        segment_out = self.sim_proj(segment_out)

        # 投影到特征空间 -------------------------------------------------------
        segment_out = self.sim_proj(segment_out)  # (B,75,256)

        # 全局特征融合 --------------------------------------------------------
        global_v = self.global_pool(video, vmask).unsqueeze(1)  # (B,1,256)
        global_t = self.global_pool(text, qmask).unsqueeze(1)
        global_out = (global_v + global_t).expand_as(video)  # (B,75,256)

        # 最终输出 ------------------------------------------------------------
        return F.relu(
            phrase_out +
            0.5 * segment_out +
            0.2 * global_out
        )

# 输入维度验证
# vfeats = torch.randn(32, 75, 256)  # 视频特征
# qfeats = torch.randn(32, 21, 256)  # 文本特征（假设n=20）
# vmask = torch.ones(32, 75)         # 视频掩码
# qmask = torch.ones(32, 21)         # 文本掩码
#
# model = HierarchicalFusion(dim=256)
# output = model(vfeats, qfeats, vmask, qmask)
# print(output.shape)  # torch.Size([32, 75, 256])

