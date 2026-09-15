# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from paint.SEABORN import plot_2d_scatter_with_seaborn
from paint.paint_3D import plot_3d_scatter_with_unified_pca, plot_3d_scatter_with_tsne
from paint.paint_2D import paint_2D
from paint.paint_feat_matrix import paint_feat_matrix
from paint.paint_feat_matrix2 import plot_feature_as_heatmap
from paint.paint_mean_variance import paint_mean_variance
from paint.paint_scatterplot import paint_scatterplot
from paint.paint_mean_variance_over_time import paint_mean_variance_over_time
from paint.paint_scatterplot2 import plot_feature_relationships, plot_feature_scatter
from paint.paint_cos_similarity import plot_feature_cos_similarity2
from paint.paint_tsne import plot_tsne
from paint.plot_3D_surface import plot_3d_surface
from mchd.paint_feat import paint_feat
from mchd.span_utils import generalized_temporal_iou, span_cxw_to_xx

from mchd.matcher import build_matcher
from mchd.transformer import build_transformer
from mchd.position_encoding import build_position_encoding
from mchd.misc import accuracy
from mchd.interaction.test_CQA import VSLFuser
import torchvision

import numpy as np
def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class GRUFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=False):
        super(GRUFeatureExtractor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        output, hidden = self.gru(x)

        # If bidirectional, concatenate the forward and backward hidden states
        if self.gru.bidirectional:
            # Combine the hidden states of both directions (forward and backward)
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # If not bidirectional, use the hidden state of the last layer
            hidden = hidden[-1, :, :]

        # Output the hidden state(s) as the global feature(s)
        # For a single layer, hidden shape: (batch_size, hidden_dim)
        # For multiple layers, hidden shape: (batch_size, num_layers * hidden_dim)
        return hidden

class TRDETR(nn.Module):
    """ TR DETR. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False,
                 contrastive_align_loss=False, contrastive_hdim=64,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2, sub_dim=0, clip_len = 2):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         TR-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_align_loss: If true, perform span - tokens contrastive learning
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.clip_len = clip_len
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj

        self.query_embed = nn.Embedding(num_queries, 2)
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + sub_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)
        self.aux_loss = aux_loss

        self.hidden_dim = hidden_dim
        self.saliency_proj1 = nn.Linear(transformer.d_model, transformer.d_model)

        self.fuser = VSLFuser(transformer.d_model)

        self.gru_extractor = GRUFeatureExtractor(hidden_dim, hidden_dim, num_layers=1, bidirectional=False)

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, src_sub=None, src_sub_mask=None):

        if src_sub is not None:
            src_vid = torch.cat([src_vid, src_sub], dim=2) #3332


        src_vid = self.input_vid_proj(src_vid) # 1 75 256
        src_txt = self.input_txt_proj(src_txt) # 1 10 256


        src_vid_copy = src_vid.clone()
        src_txt_copy = src_txt.clone()
        # # plot_3d_surface(src_vid)
        # plot_2d_scatter_with_seaborn(src_vid, src_txt,title="src_vid with src_txt before fuser & after linear PCA")
        # plot_2d_scatter_with_seaborn(src_vid,src_txt,method="tsne",title="src_vid with src_txt before fuser & after linear TSNE")
        # plot_3d_scatter_with_unified_pca(src_vid,src_txt,title="src_vid with src_txt before fuser & after linear PCA")
        # plot_3d_scatter_with_tsne(src_vid,src_txt,title="src_vid with src_txt before fuser & after linear TSNE")
        #
        # paint_feat(src_txt,"src_txt before fuser & after linear")
        # paint_feat(src_vid,"src_vid before fuser & after linear")
        # # visualize_tsne(src_vid,src_txt)
        # plot_feature_cos_similarity2(src_vid,src_txt,title="cos_similarity Src_vid with src_txt before fuser & after linear")
        # plot_feature_as_heatmap(src_vid,title="src_vid before fuser")
        #
        # # plot_feature_relationships(src_vid,src_txt)
        # paint_feat_matrix(src_vid,target_dim=10,title="src_vid before fuser")
        # paint_feat_matrix(src_txt,title="src_txt before fuser")

        # CTC
        src_txt_ed = src_txt
        src_vid_ed = src_vid
        # VTC
        src_vid_cls_ed = src_vid.mean(1) # 1 256
        src_txt_cls_ed = src_txt.mean(1) # 1 256

        src_vid = self.fuser(src_vid, src_txt, src_vid_mask, src_txt_mask)


        # plot_2d_scatter_with_seaborn(src_vid, src_txt, title="src_vid with src_txt after fuser PCA")
        # plot_2d_scatter_with_seaborn(src_vid, src_txt, method="tsne",
        #                              title="src_vid with src_txt after fuser TSNE")
        # plot_3d_scatter_with_unified_pca(src_vid,src_txt,title="src_vid with src_txt after fuser PCA")
        # plot_3d_scatter_with_tsne(src_vid,src_txt,title="src_vid with src_txt after fuser TSNE")
        #
        #
        # # plot_feature_relationships(src_vid,src_txt_copy)
        # # plot_feature_relationships(src_vid,src_vid_copy)
        #
        # plot_feature_cos_similarity2(src_vid,src_txt,title="cos_similarity src_vid with src_txt after fuser")
        #
        #
        #
        # paint_feat(src_vid,title="stc_vid after fuser")
        # paint_scatterplot(src_vid,target_dim=10 , title="src_vid after fuser dim=75")
        # paint_scatterplot(src_vid,target_dim=1 , title="src_vid after fuser dim=1")
        # paint_feat_matrix(src_vid,target_dim=10,title="src_vid after fuser")
        # plot_feature_as_heatmap(src_vid,title="src_vid after fuser")

        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)

        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        video_length = src_vid.shape[1]

        hs, reference, memory = self.transformer(
            src,  # 输入特征
            ~mask,  # 掩码取反
            self.query_embed.weight,  # 查询嵌入权重
            pos,  # 位置编码
            self.saliency_proj1,  # 显著性投影
            video_length=video_length,  # 视频长度
        )

        # paint_feat(memory,title="memory_before_mr2hd")
        #
        # plot_2d_scatter_with_seaborn(memory, src_txt, title="memory with src_txt after transformer before MR2HD PCA")
        # plot_2d_scatter_with_seaborn(memory, src_txt, method="tsne",
        #                              title="memory with src_txt after transformer before MR2HD TSNE")
        # plot_3d_scatter_with_unified_pca(memory,src_txt,title="memory with src_txt after transformer before MR2HD PCA")
        # plot_3d_scatter_with_tsne(memory,src_txt,title="memory with src_txt after transformer before MR2HD TSNE")
        #
        # plot_2d_scatter_with_seaborn(memory, src_vid, title="memory with src_vid after transformer before MR2HD PCA")
        # plot_2d_scatter_with_seaborn(memory, src_vid, method="tsne",
        #                              title="memory with src_vid after transformer before MR2HD TSNE")
        # plot_3d_scatter_with_unified_pca(memory,src_vid,lable1="m",lable2="src_vid",title="memory with src_vid after transformer before MR2HD")
        # plot_3d_scatter_with_tsne(memory,src_vid,lable1="m",lable2="src_vid",title="memory with src_vid after transformer before MR2HD TSNE")
        # plot_feature_cos_similarity2(memory,src_txt,title="cos_similarity memory with src_txt after transformer before MR2HD")
        # paint_feat_matrix(memory,target_dim=10,title="memory after tf before MR2HD")
        # plot_feature_as_heatmap(memory,title="memory after tf before MR2HD")




        # 进行分类预测
        outputs_class = self.class_embed(hs)  # 进行分类预测 linear层 # (#layers, batch_size, #queries, #classes)

        # 计算边界框预测
        reference_before_sigmoid = inverse_sigmoid(reference)  # 反向sigmoid处理
        tmp = self.span_embed(hs)  # 计算跨度嵌入 MLP层 获得预测时刻
        outputs_coord = tmp + reference_before_sigmoid  # 计算边界框坐标
        # outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        # reference_before_sigmoid = inverse_sigmoid(reference)
        # tmp = self.span_embed(hs)
        # outputs_coord = tmp + reference_before_sigmoid
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}

        # *Refine the highlight score prediction process using the retrieved moments from moment retrieval
        saliency_scores_refined , memory_after_mr2hd = self.MR2HD(out, memory, video_length, src_vid_ed)

        txt_mem = memory[:, src_vid.shape[1]:] # (1, 0, 256)

        vid_mem = memory[:, :src_vid.shape[1]] # (1, 75, 256)
        # plot_2d_scatter_with_seaborn(memory_after_mr2hd, src_txt, title="memory with src_txt after MR2HD PCA by seaborn")
        # plot_2d_scatter_with_seaborn(memory_after_mr2hd, src_txt, method="tsne",
        #                              title="memory with src_txt after MR2HD TSNE by seaborn")
        # plot_3d_scatter_with_unified_pca(memory_after_mr2hd,src_txt,title="memory with src_txt after MR2HD PCA")
        # plot_3d_scatter_with_tsne(memory_after_mr2hd,src_txt,title="memory with src_txt after MR2HD TSNE")
        #
        # plot_2d_scatter_with_seaborn(memory_after_mr2hd, src_vid,
        #                              title="memory with src_vid after MR2HD PCA by seaborn")
        # plot_2d_scatter_with_seaborn(memory_after_mr2hd, src_vid, method="tsne",
        #                              title="memory with src_vid after MR2HD TSNE by seaborn")
        # plot_3d_scatter_with_unified_pca(memory_after_mr2hd, src_vid, lable1="mams",lable2="src_vid",title="memory with src_vid after MR2HD PCA")
        # plot_3d_scatter_with_tsne(memory_after_mr2hd, src_vid, lable1="mams",lable2="src_vid",title="memory with src_vid after MR2HD TSNE")
        # paint_feat_matrix(memory_after_mr2hd,target_dim=10,title="memory after MR2HD")
        #
        # paint_feat(memory_after_mr2hd,"memory_after_mr2hd")
        # plot_feature_cos_similarity2(memory_after_mr2hd,src_txt,title="cos_similarity memory with src_txt after MR2HD")
        #
        #
        # # paint_scatterplot(memory,title="memory_before_mr2hd")
        # # paint_scatterplot(memory_after_mr2hd,title="memory_after_mr2hd")
        # # paint_mean_variance(memory,title="memory_before_mr2hd")
        # # paint_mean_variance(memory_after_mr2hd,title="memory_after_mr2hd")
        # # paint_mean_variance_over_time(memory,title="memory_before_mr2hd")
        # # paint_mean_variance_over_time(memory_after_mr2hd,title="memory_after_mr2hd")
        # # paint_feat_matrix(memory_after_mr2hd,target_dim=13,title="memory after MR2HD")
        # plot_feature_as_heatmap(memory_after_mr2hd,title="memory after MR2HD")



        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(dict(
                proj_queries=proj_queries[-1],
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem
            ))

        out["saliency_scores"] = saliency_scores_refined
        out["video_mask"] = src_vid_mask
        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            if self.contrastive_align_loss:
                for idx, d in enumerate(proj_queries[:-1]):
                    out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem))

        out['src_txt_ed'] = src_txt_ed
        out['src_vid_ed'] = src_vid_ed
        out['src_vid_cls_ed'] = src_vid_cls_ed
        out['src_txt_cls_ed'] = src_txt_cls_ed

        return out
    
    def MR2HD(self, out, memory, video_length, src_vid_ed):
        """
        MR2HD模块的主要目标是通过Moment Retrieval任务检索到的时刻来优化Highlight Detection任务的高光分数预测。
        """
        prob = F.softmax(out["pred_logits"], -1)  #  是模型输出的未归一化的类别分数（logits），表示每个检测框属于各个类别的得分。
        scores = prob[..., 0]
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        sorted_indices_max  = sorted_indices[:,:1]


        pred_spans = out['pred_spans']
        spans = span_cxw_to_xx(pred_spans) * (video_length * self.clip_len)
        spans = torch.floor(spans / self.clip_len)

        selected_values_max = spans[torch.arange(spans.size(0)).unsqueeze(1), sorted_indices_max].squeeze(1)

        sliced_samples = []
        b = memory.size(0)
        max_time = memory.size(1)

        fixed_slice_size = max_time

        for i in range(b):
            start_time = int(selected_values_max[i, 0])
            end_time = int(selected_values_max[i, 1])
            sliced_sample = src_vid_ed[i, start_time:end_time + 1, :]

            padding_size = fixed_slice_size - sliced_sample.size(0)
            if padding_size > 0:
                padded_slice = F.pad(sliced_sample, (0, 0, 0, padding_size), value=0)
            else:
                padded_slice = sliced_sample[:fixed_slice_size, :]

            sliced_samples.append(padded_slice)


        sliced_features = torch.stack(sliced_samples, dim=0) # orch.Size([32, 75, 256])
        mask = (sliced_features != 0.0)
        # 使用门控循环单元（GRU）从检索到的时刻中捕获全局信息，得到检索时刻(xiang ying de qie pian shi ke )的全局特征向量
        sliced_features_global_expanded = self.gru_extractor(sliced_features).unsqueeze(1) # orch.Size([32, 256])

        # 计算检索时刻的全局特征与视频片段特征之间的相关性分数
        src_vid_ed=(src_vid_ed.transpose(1, 2))
        cosine_similarity_matrix = torch.matmul(sliced_features_global_expanded, src_vid_ed).squeeze(1)  # shape: (32, 1, 75)
        #
        weight = cosine_similarity_matrix
        #
        memory = memory*weight.unsqueeze(-1)  +  memory

        saliency_scores = (torch.sum(self.saliency_proj1(memory), dim=-1) / np.sqrt(self.hidden_dim))
        return saliency_scores,memory



class SetCriterion(nn.Module):
    """ 该类用于计算DETR的损失。
    过程分为两步:
        1) 使用匈牙利算法计算ground truth boxes和模型输出之间的匹配
        2) 监督每对匹配的ground-truth/prediction对(监督类别和边界框)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1, use_matcher=True):
        """ 创建损失函数标准。
        参数:
            matcher: 能够计算目标和预测之间匹配的模块
            weight_dict: 字典,包含损失名称作为key,相应权重作为value 
            eos_coef: 应用于no-object类别的相对分类权重
            losses: 要应用的所有损失列表。参见get_loss获取可用损失列表
            temperature: float类型,NCE损失的温度参数
            span_loss_type: 字符串类型,[l1, ce]
            max_v_l: 整数类型
            saliency_margin: float类型,显著性边界值
        """
        super().__init__()
        self.matcher = matcher  # 匹配器
        self.weight_dict = weight_dict  # 损失权重字典
        self.losses = losses  # 损失列表
        self.temperature = temperature  # 温度参数
        self.span_loss_type = span_loss_type  # span损失类型
        self.max_v_l = max_v_l  # 最大视频长度
        self.saliency_margin = saliency_margin  # 显著性边界值

        # 前景和背景分类
        self.foreground_label = 0  # 前景标签
        self.background_label = 1  # 背景标签
        self.eos_coef = eos_coef  # end-of-sequence系数
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # 降低背景权重(索引1,前景索引0)
        self.register_buffer('empty_weight', empty_weight)
        
        # 用于tvsum数据集
        self.use_matcher = use_matcher  # 是否使用匹配器

    def loss_spans(self, outputs, targets, indices):
        """计算与边界框相关的损失,包括L1回归损失和GIoU损失
           targets字典必须包含键"spans",其中包含维度为[nb_tgt_spans, 2]的张量
           目标spans应该是(center_x, w)格式,由图像大小归一化
        """
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            # 计算L1损失
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            # 计算GIoU损失
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce损失
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """分类损失(NLL)
        targets字典必须包含键"labels",包含维度为[nb_target_boxes]的张量
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx是两个1D张量的元组(batch_idx, src_idx),长度相同==批次中的对象数
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries) # 1
        target_classes[idx] = self.foreground_label # 0
        target_classes = F.one_hot(target_classes, num_classes=2).permute(0, 2, 1) # (32, 10, 2)
        src_logits = src_logits.to(torch.float32)
        target_classes = target_classes.to(torch.float32)
        # 使用Focal Loss计算分类损失
        loss_ce = torchvision.ops.focal_loss.sigmoid_focal_loss(src_logits.transpose(1, 2), target_classes, alpha=0.25, gamma=2.0, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses
    
    def loss_saliency(self, outputs, targets, indices, log=True):
        """对正样本片段给予更高的分数"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        vid_token_mask = outputs["video_mask"]
        saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
        saliency_contrast_label = targets["saliency_all_labels"]
        # 对无效位置使用大的负值掩码
        saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

        tau = 0.5  # 温度参数
        loss_rank_contrastive = 0.

        # 计算对比损失
        for rand_idx in range(1, 12):
            drop_mask = ~(saliency_contrast_label > 100)  # 不丢弃
            pos_mask = (saliency_contrast_label >= rand_idx)  # 当等于或高于rand_idx时为正样本

            if torch.sum(pos_mask) == 0:  # 没有正样本
                continue
            else:
                batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # 负样本指示器

            # 丢弃更高的排名
            cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3

            # 数值稳定性
            logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]

            # softmax计算
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)

            loss = - mean_log_prob_pos * batch_drop_mask

            loss_rank_contrastive = loss_rank_contrastive + loss.mean()

        loss_rank_contrastive = loss_rank_contrastive / 12
        
        # 边界损失: Moment-DETR
        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # 通常是2或4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        # 计算边界损失
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                        / (len(pos_scores) * num_pairs) * 2  # *2保持损失相同的尺度
        loss_saliency = loss_saliency + loss_rank_contrastive 
        return {"loss_saliency": loss_saliency}

    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """鼓励匹配的查询span和输入文本之间有更高的分数"""
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  文本标记
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        # 计算相似度矩阵
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        # 计算NCE损失
        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def loss_contrastive_align_vid_txt(self, outputs, targets, indices, log=True):
        """鼓励匹配的查询span和输入文本之间有更高的分数"""
        # TODO (1) 对齐vid_mem和txt_mem
        # TODO (2) 将L1损失更改为75个标签的CE损失,类似于MDETR中的软标记预测
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  文本标记
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def _get_src_permutation_idx(self, indices):
        """获取源序列的排列索引"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # 两个相同长度的1D张量

    def _get_tgt_permutation_idx(self, indices):
        """获取目标序列的排列索引"""
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        """获取指定类型的损失"""
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "contrastive_align": self.loss_contrastive_align,
            "saliency": self.loss_saliency,
        }
        assert loss in loss_map, f'你确定要计算{loss}损失吗?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ 执行损失计算。
        参数:
             outputs: 张量字典,查看模型输出规范了解格式
             targets: 字典列表,使得len(targets) == batch_size
                      每个字典中的预期键取决于应用的损失,查看每个损失的文档
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # 检索最后一层输出和目标之间的匹配
        # list(tuples),每个tuple是(pred_span_indices, tgt_span_indices)

        # 仅用于HL,不使用匹配器
        if self.use_matcher:
            indices = self.matcher(outputs_without_aux, targets)# 匈牙利算法

            losses_target = self.losses
        else:
            indices = None
            losses_target = ["saliency"]

        # 计算所有请求的损失
        losses = {}
        for loss in losses_target:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # 对于辅助损失,我们对每个中间层的输出重复此过程
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if self.use_matcher:
                    indices = self.matcher(aux_outputs, targets)
                    losses_target = self.losses
                else:
                    indices = None
                    losses_target = ["saliency"]    
                for loss in losses_target:
                    if "saliency" == loss:  # 跳过,因为它只在顶层
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/qd_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)
    # 根据是否使用音频来构建模型S
    if args.s_feat_dir is None:
        model = TRDETR(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            clip_len=args.clip_length
        )
    else:
        model = TRDETR(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            sub_dim=args.s_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            clip_len=args.clip_length
        )

    # 构建匹配器
    matcher = build_matcher(args)
    # 定义损失函数的权重字典
    weight_dict = {"loss_span": args.span_loss_coef,      # 时间跨度损失权重
                   "loss_giou": args.giou_loss_coef,      # GIoU损失权重
                   "loss_label": args.label_loss_coef,    # 标签损失权重
                   "loss_saliency": args.lw_saliency}     # 显著性损失权重
    
    # 如果使用对比对齐损失,添加相应的权重
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
        
    # 如果使用辅助损失
    if args.aux_loss:
        aux_weight_dict = {}
        # 为每一个解码器层(除最后一层)添加辅助损失权重
        for i in range(args.dec_layers - 1):
            # 更新辅助权重字典,复制所有损失权重(除显著性损失外)
            # 更新辅助权重字典:
            # 遍历weight_dict中除"loss_saliency"外的所有损失权重(k,v)
            # 为每个损失添加层索引后缀'_i'作为新的key
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        # 将辅助损失权重添加到主权重字典中
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency']
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
        
    # For tvsum dataset
    use_matcher = not (args.dset_name == 'tvsum')
        
    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin, use_matcher=use_matcher,
    )
    criterion.to(device)
    return model, criterion
