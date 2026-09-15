import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=5000):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        # x: (Batch, Seq, Dim)
        seq_len = x.size(1)
        return self.encoding[:seq_len].unsqueeze(0)

class FactorizedAttention(nn.Module):
    def __init__(self, dim=256, heads=8, time_segments=16, batch_first=False):
        super().__init__()
        self.batch_first = batch_first  # 关键：添加兼容性属性
        self.time_segments = time_segments
        self.space_attn = nn.MultiheadAttention(dim, heads, batch_first=batch_first)
        self.time_attn = nn.MultiheadAttention(dim, heads, batch_first=batch_first)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # 维度转换保证兼容性
        if self.batch_first:
            # 输入应为 (Batch, Seq, Dim)
            x = x.permute(1, 0, 2)  # 转为 (Seq, Batch, Dim) 处理
            
        batch_size = x.size(1)
        seq_len = x.size(0)
        spatial_dim = seq_len // self.time_segments
        
        # 空间注意力
        x_space = x.view(self.time_segments, spatial_dim, batch_size, -1)
        x_space = x_space.permute(1, 0, 2, 3).reshape(spatial_dim, -1, x.size(-1))
        x_space, _ = self.space_attn(x_space, x_space, x_space)
        
        # 时间注意力
        x_time = x.view(spatial_dim, self.time_segments, batch_size, -1)
        x_time = x_time.permute(1, 0, 2, 3).reshape(self.time_segments, -1, x.size(-1))
        x_time, _ = self.time_attn(x_time, x_time, x_time)
        
        # 融合结果
        output = torch.sigmoid(self.alpha) * x_space + (1 - torch.sigmoid(self.alpha)) * x_time
        
        if self.batch_first:
            output = output.permute(1, 0, 2)  # 转回 (Batch, Seq, Dim)
            
        return output

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=512, 
                 time_segments=16, dropout=0.1, batch_first=False):
        super().__init__()
        self.self_attn = FactorizedAttention(d_model, nhead, time_segments, batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        # 自注意力分支
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈分支
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class VideoTransformer(nn.Module):
    def __init__(self, time_segments=16, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.pos_embed = PositionalEncoding()
        self.encoder = nn.TransformerEncoder(
            encoder_layer=CustomTransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=512,
                time_segments=time_segments,
                batch_first=batch_first
            ),
            num_layers=6
        )

    def forward(self, x):
        # 输入形状: (Batch=32, Seq, 256)
        x = x + self.pos_embed(x)
        
        if not self.batch_first:
            x = x.permute(1, 0, 2)
            
        x = self.encoder(x)
        
        if not self.batch_first:
            x = x.permute(1, 0, 2)
            
        return x


# x = torch.randn(32, 75, 256)
# model = VideoTransformer(time_segments=x.shape[1])
# out = model(x)
# print(out.shape)