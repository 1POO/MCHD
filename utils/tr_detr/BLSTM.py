import torch
from torch import nn

class BLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim, hidden_dim)

        # 定义多层双向 LSTM（每层输入自动继承前一层输出）
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=input_dim if i == 0 else 2 * hidden_dim,
                hidden_size=hidden_dim,
                bidirectional=True,
                batch_first=True
            ) for i in range(num_layers)
        ])

        # 每层参考点生成器（输入维度应为 2*hidden_dim）
        self.ref_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            ) for _ in range(num_layers)
        ])

        # 时间步下采样（75 → 10）
        self.pool = nn.AdaptiveAvgPool1d(10)

    def forward(self, x):
        batch_size = x.size(0)
        hs_layers = []
        references = []

        for i in range(self.num_layers):
            # 第 i 层双向 LSTM 前向传播
            x, _ = self.lstm_layers[i](x)  # x shape: (32, 75, 2 * 256=512)

            # 时间步下采样到 10
            x_pooled = self.pool(x.permute(0, 2, 1))  # (32, 512, 10)
            x_pooled = x_pooled.permute(0, 2, 1)      # (32, 10, 512)

            # 保留完整的双向特征（不再求和）
            layer_hs = x_pooled  # (32, 10, 512)
            hs_layers.append(layer_hs)

            # 生成参考点
            ref = self.ref_mlps[i](layer_hs)  # (32, 10, 2)
            references.append(ref)

        # 按层堆叠
        hs = torch.stack(hs_layers, dim=0)        # (2, 32, 10, 512)
        hs = self.linear(hs)
        references = torch.stack(references, dim=0)  # (2, 32, 10, 2)

        return hs, references

