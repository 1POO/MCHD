# import torch
# from torch import nn
# from mamba_ssm import Mamba
#
# class MaskedQueryEncoder(nn.Module):
#     def __init__(self, d_model=256):
#         super().__init__()
#         self.mamba = Mamba(d_model=d_model)
#         self.pool = nn.AdaptiveMaxPool1d(1)
#
#     def forward(self, query, mask):
#         pooled_list = []
#         for i in range(query.size(0)):
#             valid_len = int(mask[i].sum().item())
#             if valid_len == 0:
#                 pooled_list.append(torch.zeros(256, device=query.device))
#                 continue
#
#             seq = query[i, :valid_len].unsqueeze(0)
#             mamba_out = self.mamba(seq)
#             pooled = self.pool(mamba_out.transpose(1,2)).squeeze()
#             pooled_list.append(pooled)
#
#         return torch.stack(pooled_list, dim=0)
#
# class CrossModalAttention(nn.Module):
#     def __init__(self, d_video=256, d_obj=256):
#         super().__init__()
#         self.vid2obj = nn.MultiheadAttention(d_video, num_heads=4, batch_first=True)
#         self.obj2vid = nn.MultiheadAttention(d_obj, num_heads=4, batch_first=True)
#
#     def forward(self, video_feat, obj_feat):
#         attn_vid, _ = self.vid2obj(video_feat, obj_feat, obj_feat)
#         attn_obj, _ = self.obj2vid(obj_feat, video_feat, video_feat)
#         return attn_vid, attn_obj
#
# class DynamicFusion(nn.Module):
#     def __init__(self, d_video=256, d_obj=256, d_out=256):
#         super().__init__()
#         self.gate = nn.Sequential(
#             nn.Linear(d_video + d_obj, 128),
#             nn.ReLU(),
#             nn.Linear(128, 2),
#             nn.Softmax(dim=-1)
#         )
#         self.proj = nn.Linear(d_video, d_out)
#
#     def forward(self, video_feat, obj_feat):
#         concat = torch.cat([video_feat, obj_feat], dim=-1)
#         gates = self.gate(concat)
#         fused = gates[:,:,0:1] * video_feat + gates[:,:,1:2] * obj_feat
#         return self.proj(fused)
#
# class QueryGuidedFusion(nn.Module):
#     def __init__(self, d_ctx=256, d_video=256, d_obj=256, d_out=256):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(
#             embed_dim=d_ctx,
#             kdim=d_video+d_obj,
#             vdim=d_video+d_obj,
#             num_heads=4,
#             batch_first=True
#         )
#         self.fusion_gate = nn.Sequential(
#             nn.Linear(d_ctx*2 + d_video + d_obj, 256),
#             nn.ReLU(),
#             nn.Linear(256, 3),
#             nn.Softmax(dim=-1)
#         )
#         self.final_proj = nn.Linear(512, d_out)
#
#     def forward(self, video_feat, obj_feat, ctx_feat):
#         combined = torch.cat([video_feat, obj_feat], dim=-1)
#         attn_output, _ = self.self_attn(
#             query=ctx_feat.unsqueeze(1),
#             key=combined,
#             value=combined
#         )
#         expanded_ctx = attn_output.expand(-1, video_feat.size(1), -1)
#         gate_input = torch.cat([
#             video_feat,
#             obj_feat,
#             expanded_ctx,
#             ctx_feat.unsqueeze(1).expand(-1,video_feat.size(1),-1)
#         ], dim=-1)
#         gates = self.fusion_gate(gate_input)
#         fused = (
#             gates[...,0:1] * video_feat +
#             gates[...,1:2] * obj_feat +
#             gates[...,2:3] * expanded_ctx
#         )
#         return self.final_proj(torch.cat([fused, expanded_ctx], dim=-1))
#
# class MultiModalEnhancer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.query_encoder = MaskedQueryEncoder().to('cuda')
#         self.base_mamba = Mamba(d_model=256).to('cuda')
#         self.cross_attn = CrossModalAttention().to('cuda')
#         self.stage1_fusion = DynamicFusion().to('cuda')
#         self.stage2_fusion = QueryGuidedFusion().to('cuda')
#
#     def forward(self, video_feat, obj_feat, query_feat, query_mask):
#         query_ctx = self.query_encoder(query_feat, query_mask)
#         base_vid = self.base_mamba(video_feat)
#         base_obj = self.base_mamba(obj_feat)
#         attn_vid, attn_obj = self.cross_attn(base_vid, base_obj)
#         final_vid = base_vid + attn_vid
#         final_obj = base_obj + attn_obj
#         stage1_output = self.stage1_fusion(final_vid, final_obj)
#         return self.stage2_fusion(stage1_output, final_obj, query_ctx)
#
# # # 初始化配置
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model = MultiModalEnhancer().to(device)
#
# # # 测试数据（直接创建在GPU上）
# # video = torch.randn(32, 75, 256, device=device)
# # obj = torch.randn(32, 75, 256, device=device)
# # query = torch.randn(32, 25, 256, device=device)
# # mask = torch.cat([torch.ones(32,15, device=device),
# #                 torch.zeros(32,10, device=device)], dim=1)
#
# # # 前向测试
# # with torch.no_grad():
# #     output = model(video, obj, query, mask)
# #     print("Output shape on device:", output.shape, output.device)