import torch
from torch import nn
class TemporalGatedFusion(nn.Module):
    def __init__(self, feat_dim=256, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=2*feat_dim,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size//2
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, video_feat, object_feat):
        B, T, D = video_feat.shape
        concat_feat = torch.cat([video_feat, object_feat], dim=-1)  # (B,T,2D)
        concat_feat = concat_feat.permute(0, 2, 1)                  # (B,2D,T)
        gate = self.conv(concat_feat)               # (B,1,T)
        gate = gate.permute(0, 2, 1)                # (B,T,1)
        gate = self.sigmoid(gate)
        
        fused_feat = gate * video_feat + (1 - gate) * object_feat
        return fused_feat

# x = torch.randn(32,75,256)
# y = torch.randn(32,75,256)
#
# gate = TemporalGatedFusion(feat_dim=256)
#
# out = gate(x,y)
# print(out.shape)
