import torch
import torch.nn as nn
import torch.nn.functional as F

# from detectron2.configs.common.models.retinanet import model
"""
    通道注意力+空间注意力
"""

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels)
        )

    def masked_pooling(self, x, mask, pool_type='avg'):
        # x: (BS, T, C), mask: (BS, T)
        mask = mask.unsqueeze(-1)  # (BS, T, 1)

        if pool_type == 'avg':
            # 有效长度计算
            valid_cnt = torch.sum(mask, dim=1)  # (BS, 1)
            x_masked = x * mask
            pooled = torch.sum(x_masked, dim=1) / torch.clamp(valid_cnt, min=1e-6)
        elif pool_type == 'max':
            # 用极小值填充被mask的位置
            x_masked = x.masked_fill(mask == 0, -1e6)
            pooled = torch.max(x_masked, dim=1)[0]
        return pooled.unsqueeze(1)  # (BS, 1, C)

    def forward(self, x, mask):
        # x: (BS, T, C), mask: (BS, T)
        avg_pool = self.masked_pooling(x, mask, 'avg')
        max_pool = self.masked_pooling(x, mask, 'max')

        # 共享MLP
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        # 合并结果
        channel_weights = torch.sigmoid(avg_out + max_out)
        return x * channel_weights  # 广播相乘


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, mask):
        # x: (BS, T, C), mask: (BS, T)
        # 沿通道维度池化
        avg_out = torch.mean(x, dim=2, keepdim=True)  # (BS, T, 1)
        max_out, _ = torch.max(x, dim=2, keepdim=True)

        # 拼接特征
        combined = torch.cat([avg_out, max_out], dim=2)  # (BS, T, 2)

        # 调整维度用于1D卷积
        combined = combined.permute(0, 2, 1)  # (BS, 2, T)

        # 生成空间权重
        spatial_weights = self.conv(combined)  # (BS, 1, T)
        spatial_weights = spatial_weights.permute(0, 2, 1)  # (BS, T, 1)
        spatial_weights = torch.sigmoid(spatial_weights)

        # 应用mask
        spatial_weights = spatial_weights * mask.unsqueeze(-1)
        return x * spatial_weights  # 广播相乘

class VideoCBAM(nn.Module):
    def __init__(self, channels=256, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x, video_mask):
        # x: (BS, T, C), video_mask: (BS, T)
        x = self.channel_att(x, video_mask)
        x = self.spatial_att(x, video_mask)
        return x

# model = VideoCBAM(channels=256, reduction_ratio=16, kernel_size=7)
# x = torch.randn(32 ,75,256)
# y = torch.randn(32 ,75)
# out = model(x, y)
# print(out.shape)