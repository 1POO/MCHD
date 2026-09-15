import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
"""
    local Multi-Modal Alignment LOSS
"""

# VCTC
class CTC_Loss(nn.Module):
    def __init__(self, temperature=0.07):
        super(CTC_Loss, self).__init__()
        self.temperature = temperature

    def forward(self, vid_feat, txt_feat, pos_mask, src_vid_mask=None, src_txt_mask=None):
        # vid_feat: (bs, t, d)
        # txt_feat: (bs, n, d)
        # pos_mask: (bs, t)
        # src_vid_mask: (bs, t) or None
        # src_txt_mask: (bs, n) or None
        bs = vid_feat.size(0)
        t = vid_feat.size(1)
        n = txt_feat.size(1)
        d = vid_feat.size(2)
        # normalize the feature vectors
        vid_feat = F.normalize(vid_feat, dim=2) # (bs, t, d)
        txt_feat = F.normalize(txt_feat, dim=2) # (bs, n, d)
        # compute the global text feature by mean pooling
        if src_txt_mask is not None:
            src_txt_mask = src_txt_mask.unsqueeze(-1) # (bs, n, 1)
            txt_feat = txt_feat * src_txt_mask # (bs, n, d)
            txt_global = torch.sum(txt_feat, dim=1) / torch.sum(src_txt_mask, dim=1) # (bs, d)
        else:
            txt_global = torch.mean(txt_feat, dim=1) # (bs, d)
        # compute the similarity matrix
        sim_mat = torch.bmm(vid_feat, txt_global.unsqueeze(-1)).squeeze(-1) # (bs, t)
        # apply the video mask if given
        if src_vid_mask is not None:
            sim_mat = sim_mat * src_vid_mask # (bs, t)
        # compute the logits and labels
        logits = sim_mat / self.temperature # (bs, t)
        labels = pos_mask.long() # (bs, t)
        # compute the binary cross entropy loss with logits
        loss = F.binary_cross_entropy_with_logits(logits, labels.float()) # scalar
        # return the loss
        return loss
if __name__ == "__main__":
    # 测试数据
    batch_size = 32
    time_steps = 75
    text_tokens = 22
    feat_dim = 256

    # 初始化模型
    model = CTC_Loss()

    # 模拟输入
    video_feat = torch.randn(batch_size, time_steps, feat_dim)
    text_feat = torch.randn(batch_size, text_tokens, feat_dim)
    pos_mask = torch.randint(0, 2, (batch_size, time_steps)).float()  # 二进制对齐标签

    # 前向计算
    loss = model(
        vid_feat=video_feat,
        txt_feat=text_feat,
        pos_mask=pos_mask,
        src_vid_mask=torch.ones(batch_size, time_steps).bool(),  # 有效帧mask
        src_txt_mask=torch.ones(batch_size, text_tokens).bool()  # 有效token mask
    )

    print(loss)
