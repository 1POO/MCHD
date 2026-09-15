import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalTextFrameAlign(nn.Module):
    def __init__(self, temperature=0.07, pos_weight=1.0):
        super().__init__()
        self.temp = temperature
        self.pos_weight = pos_weight

    def _get_global_text(self, txt_feat, txt_mask):
        """生成全局文本表示（带掩码的注意力池化）"""
        # txt_feat: [B, Lt, D], txt_mask: [B, Lt]
        attn = torch.einsum('btd,bd->bt', txt_feat, txt_feat.mean(dim=1))  # 自注意力
        attn = attn.masked_fill(~txt_mask, -1e9)
        attn_weights = F.softmax(attn, dim=1)  # [B, Lt]
        return torch.einsum('bt,btd->bd', attn_weights, txt_feat)  # [B, D]

    def forward(self, vid_feat, txt_feat, vid_mask, txt_mask, pos_mask):
        """
        输入维度：
        vid_feat: [B, Lv, D]  视频帧特征 (B=32, Lv=75)
        txt_feat: [B, Lt, D]  文本特征
        vid_mask: [B, Lv]     视频有效帧掩码
        txt_mask: [B, Lt]     文本有效token掩码
        pos_mask: [B, Lv]     帧对齐标签 (1=与文本相关)
        """
        # 1. 生成全局文本表示
        txt_global = self._get_global_text(txt_feat, txt_mask)  # [B, D]

        # 2. 计算帧级相似度
        vid_feat_norm = F.normalize(vid_feat, p=2, dim=-1)
        txt_global_norm = F.normalize(txt_global, p=2, dim=-1)
        sim = torch.einsum('bvd,bd->bv', vid_feat_norm, txt_global_norm)  # [B, Lv]

        # 3. 掩码处理
        sim = sim.masked_fill(~vid_mask, -1e9)  # 过滤无效帧

        # 4. 加权二元分类损失
        loss = F.binary_cross_entropy_with_logits(
            sim / self.temp,
            pos_mask.float(),
            weight=vid_mask.float(),
            pos_weight=torch.tensor([self.pos_weight], device=sim.device)
        )

        return loss

if __name__ == "__main__":
    # 测试数据
    batch_size = 32
    time_steps = 75
    text_tokens = 22
    feat_dim = 256

    # 初始化模型
    model = GlobalTextFrameAlign()

    # 模拟输入
    video_feat = torch.randn(batch_size, time_steps, feat_dim)
    text_feat = torch.randn(batch_size, text_tokens, feat_dim)
    pos_mask = torch.randint(0, 2, (batch_size, time_steps)).float()  # 二进制对齐标签

    # 前向计算
    loss = model(
        vid_feat=video_feat,
        txt_feat=text_feat,
        pos_mask=pos_mask,
        vid_mask=torch.ones(batch_size, time_steps).bool(),  # 有效帧mask
        txt_mask=torch.ones(batch_size, text_tokens).bool()  # 有效token mask
    )

    print(loss)