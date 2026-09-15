# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
# 导入必要的库
import math
import torch
from torch import nn


class TrainablePositionalEncoding(nn.Module):
    """可训练的位置编码,从词嵌入、位置嵌入和token类型嵌入构建嵌入层
    """
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        # 创建位置嵌入层,大小为max_position_embeddings x hidden_size
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        # 创建层归一化
        self.LayerNorm = nn.LayerNorm(hidden_size)
        # 创建dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        """前向传播函数
        Args:
            input_feat: 输入特征 (N, L, D), N为batch size, L为序列长度, D为特征维度
        """
        # 获取batch size和序列长度
        bsz, seq_length = input_feat.shape[:2]
        # 创建位置id序列 [0,1,2,...,seq_length-1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        # 扩展position_ids到batch维度 (N, L)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)

        # 获取位置嵌入
        position_embeddings = self.position_embeddings(position_ids)

        # 将输入特征与位置嵌入相加并进行层归一化
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        # 应用dropout
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionEmbeddingSine(nn.Module):
    """正弦位置编码
    这是位置编码的标准版本,与Attention is all you need论文中使用的非常相似,
    已经推广到可以处理图像(1D序列)
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        # 位置特征的维度
        self.num_pos_feats = num_pos_feats
        # 用于缩放位置编码的温度参数
        self.temperature = temperature
        # 是否归一化
        self.normalize = normalize
        # 检查scale参数的合法性
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        """前向传播函数
        Args:
            x: 输入张量, shape为(batch_size, L, d)
            mask: 掩码张量, shape为(batch_size, L), 1表示有效位置
        """
        assert mask is not None
        # 计算累积和作为位置编码的基础 (bsz, L)
        x_embed = mask.cumsum(1, dtype=torch.float32)
        # 计算输入 mask 的累积和，具体来说，
        # 它是对 mask 在第一个维度（通常是序列长度维度）进行累加。
        # 例如，如果 mask 的某一行是 [1, 1, 0, 1]，
        # 那么 x_embed 的对应行将是 [1, 2, 2, 3]，
        # 表示第一个位置有效，第二个位置有效（累计2），
        # 第三个位置无效（保持2），第四个位置有效（累计3）。

        # 如果需要归一化
        if self.normalize:
            eps = 1e-6
            # 将位置编码归一化到[0, scale]范围内
            # x_embed[:, -1:] 获取每个序列最后一个位置的值作为归一化因子
            # 加上eps避免除零错误
            # 最后乘以self.scale将值缩放到指定范围
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        # 创建不同频率的正弦波
        # 创建一个从0到num_pos_feats-1的张量,用于生成不同频率
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # 计算不同频率的指数,每两个位置共享相同的频率
        # 使用temperature参数来控制频率的变化范围
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 计算位置编码 (bsz, L, num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t
        # 交替使用sin和cos函数,并重塑维度 (bsz, L, num_pos_feats*2)
        # 将sin和cos交替应用到位置编码上:
        # 1. pos_x[:, :, 0::2].sin() 对偶数索引位置应用sin函数
        # 2. pos_x[:, :, 1::2].cos() 对奇数索引位置应用cos函数
        # 3. torch.stack(..., dim=3) 将sin和cos的结果在新维度上堆叠
        # 4. flatten(2) 将最后两个维度展平,得到最终的位置编码
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        return pos_x


class PositionEmbeddingLearned(nn.Module):
    """可学习的绝对位置编码"""
    def __init__(self, num_pos_feats=256):
        super().__init__()
        # 创建行嵌入和列嵌入
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        # 使用均匀分布初始化权重
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x, mask):
        # 获取特征图的高度和宽度
        h, w = x.shape[-2:]
        # 创建位置索引
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        # 获取列嵌入和行嵌入
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        # 组合行列位置编码并调整维度
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    """构建位置编码
    Args:
        args: 配置参数
    Returns:
        position_embedding: 视觉特征的位置编码
        txt_pos_embed: 文本特征的位置编码
    """
    # 设置位置编码的维度
    N_steps = args.hidden_dim
    # 根据配置选择位置编码类型
    if args.position_embedding in ('v2', 'sine'):
        position_vid_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        position_obj_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    # 创建文本的位置编码
    txt_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=args.max_q_l,
            hidden_size=args.hidden_dim, dropout=args.input_dropout)
    return position_vid_embedding, position_obj_embedding,txt_pos_embed
