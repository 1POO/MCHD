import torch
import torch.nn as nn

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=eps, max=1-eps)
    return torch.log(x/(1-x))

class DualLayerBiLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_queries=10):
        super().__init__()
        # 双向LSTM（2层，每层隐藏维度256 → 双向输出512）
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        # 投影层：双向输出512 → 256
        self.proj = nn.Linear(hidden_dim*2, hidden_dim)
        # 可学习查询向量（10个查询）
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        
    def forward(self, x):
        # 输入x: (32,75,512)
        out, _ = self.lstm(x)  # 输出形状: (32,75,512)前向后向的隐藏层拼接为512
        out = self.proj(out)   # 投影到 (32,75,256)
        
        # 时间步全局平均池化
        global_feat = out.mean(dim=1)  # (32,256)
        # 生成多层隐藏状态（模拟2层Decoder）
        layer1 = global_feat.unsqueeze(1) + self.queries  # (32,10,256)
        layer2 = layer1 + self.queries                    # 模拟第二层
        return torch.stack([layer1, layer2], dim=0)        # (2,32,10,256)
    
class RefPointGenerator(nn.Module):
    def __init__(self, hidden_dim=256, num_queries=10):
        super().__init__()
        # 初始化参考点（Sigmoid归一化到[0,1]）
        self.ref_points = nn.Parameter(torch.randn(num_queries, 2))
        # 偏移量预测网络
        self.offset_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, hs):
        # hs: (2,32,10,256)
        all_refs = []
        refs = torch.sigmoid(self.ref_points).unsqueeze(0)  # (1,10,2)
        
        for layer in range(hs.shape[0]):
            # 当前层隐藏状态
            h = hs[layer]  # (32,10,256)
            # 预测偏移量
            delta = self.offset_net(h)  # (32,10,2)
            # 逆Sigmoid更新
            refs = inverse_sigmoid(refs) + delta
            refs = torch.sigmoid(refs)
            all_refs.append(refs)
            
        return torch.stack(all_refs, dim=0)  # (2,32,10,2)
    
class BLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DualLayerBiLSTM()
        self.ref_decoder = RefPointGenerator()
    
    def forward(self, x):
        hs = self.encoder(x)           # (2,32,10,256)
        refs = self.ref_decoder(hs)    # (2,32,10,2)
        return hs, refs

# model = BLSTM()
# x = torch.randn(32,75,512)
# out = model(x)
# print(out.shape)