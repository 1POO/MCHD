# import torch
# import torch.nn as nn
# from mamba_ssm import Mamba
#
#
# class MambaDecoder(nn.Module):
#     def __init__(self,
#                  input_dim=512,
#                  hidden_dim=256,
#                  num_layers=2,
#                  output_seq_len=10,
#                  conv_dim=2):
#         super().__init__()
#         self.num_layers = num_layers
#         self.output_seq_len = output_seq_len
#
#         self.mamba_blocks = nn.ModuleList([
#             Mamba(d_model=hidden_dim,
#                   d_state=16,
#                   d_conv=4,
#                   expand=2)
#             for _ in range(num_layers)
#         ])
#
#         self.input_proj = nn.Linear(input_dim, hidden_dim)
#
#         self.ref_predictor = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim // 2),
#                 nn.GELU(),
#                 nn.Linear(hidden_dim // 2, conv_dim)
#             ) for _ in range(num_layers)
#         ])
#
#         self.sequence_adapter = nn.AdaptiveAvgPool1d(output_seq_len)
#
#         self.residual_proj = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(conv_dim * output_seq_len, hidden_dim),
#                 nn.LayerNorm(hidden_dim)
#             ) for _ in range(num_layers)
#         ])
#
#     def forward(self, x, init_ref=None):
#         x = self.input_proj(x)  # [32,75,512] => [32,75,256]
#
#         B = x.size(0)
#         if init_ref is None:
#             ref = torch.zeros(B, 2, self.output_seq_len, device=x.device)
#         else:
#             ref = init_ref
#
#         all_hs = []
#         all_ref = []
#
#         for i in range(self.num_layers):
#             x = self.mamba_blocks[i](x)  # [32,75,256]
#
#             hs = self.sequence_adapter(x.transpose(1, 2)).transpose(1, 2)  # [32,10,256]
#             all_hs.append(hs)
#
#             delta = self.ref_predictor[i](x)  # [32,75,2]
#             delta = self.sequence_adapter(delta.transpose(1, 2))  # [32,2,10]
#
#             ref = ref + delta
#             # 调整维度顺序为 [B, seq_len, conv_dim]
#             all_ref.append(ref.permute(0, 2, 1))  # [32,10,2]
#
#             ref_feat = ref.permute(0, 2, 1)  # [32,10,2] => [32,2,10]
#             ref_feat = ref_feat.reshape(B, -1)  # [32, 2 * 10]
#             ref_feat = ref_feat.unsqueeze(1).expand(-1, x.size(1), -1)  # [32,75,20]
#
#             x = x + self.residual_proj[i](ref_feat)  # [32,75,256]
#
#         return torch.stack(all_hs, dim=0), torch.stack(all_ref, dim=0)
#
# # x = torch.randn(32,75,512).to("cuda")
# # modle = MambaDecoder().to("cuda")
# # hs,refs = modle(x)
# # print(hs.size())