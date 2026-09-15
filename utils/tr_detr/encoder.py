import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用cos
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]  # 自动广播到batch维度
        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 输入形状为 [batch, seq, feature]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        src = self.pos_encoder(src)  # 添加位置编码
        output = self.encoder(src)    # 通过Transformer Encoder
        return output

# # 实例化模型
# model = TransformerEncoder(
#     d_model=256,
#     nhead=4,
#     num_layers=4,
#     dim_feedforward=1024,
#     dropout=0.1
# )

# # 测试输入输出
# if __name__ == "__main__":
#     x = torch.randn(32, 75, 256)  # [batch_size, seq_len, d_model]
#     output = model(x)
#     print(output.shape)  # torch.Size([32, 75, 256])