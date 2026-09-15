# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from mchd.span_utils import generalized_temporal_iou, span_cxw_to_xx


class HungarianMatcher(nn.Module):
    """该类计算网络预测结果和目标之间的匹配关系

    出于效率考虑,目标中不包含no_object(非目标)类别。因此通常情况下,预测结果的数量会多于目标数量。
    在这种情况下,我们对最佳预测结果进行一对一的匹配,而其他未匹配的预测结果则被视为非目标。
    """
    def __init__(self,  cost_class: float = 1, cost_span: float = 1, cost_giou: float = 1,
                 span_loss_type: str = "l1", max_v_l: int = 75):
        """初始化匹配器

        参数:
            cost_class: 分类损失的权重系数
            cost_span: span坐标L1误差在匹配代价中的相对权重
            cost_giou: span的giou损失在匹配代价中的相对权重
            span_loss_type: span损失的类型,可选"l1"
            max_v_l: 视频最大长度
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.foreground_label = 0
        assert cost_class != 0 or cost_span != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """执行匹配过程

        参数:
            outputs: 包含以下键值的字典:
                "pred_spans": 形状为[batch_size, num_queries, 2]的张量,包含预测的span坐标,
                            使用标准化的(cx, w)格式
                "pred_logits": 形状为[batch_size, num_queries, num_classes]的张量,包含分类logits

            targets: 目标列表(长度为batch_size),每个目标是包含以下内容的字典:
                "spans": 形状为[num_target_spans, 2]的张量,包含目标span坐标。
                        span使用标准化的(cx, w)格式

        返回值:
            大小为batch_size的列表,包含(index_i, index_j)元组,其中:
                - index_i是选定预测的索引(有序)
                - index_j是对应选定目标的索引(有序)
            对于每个batch元素:
                len(index_i) = len(index_j) = min(num_queries, num_target_spans)
        """
        bs, num_queries = outputs["pred_spans"].shape[:2]
        targets = targets["span_labels"]

        # 连接目标标签和spans
        # 计算预测的类别概率分布, 将输出的logits展平并应用softmax函数
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        
        # 将所有目标的span坐标连接成一个张量
        tgt_spans = torch.cat([v["spans"] for v in targets])  # [num_target_spans in batch, 2]
        
        # 创建一个张量, 用于存储目标的标签, 其值为foreground_label
        tgt_ids = torch.full([len(tgt_spans)], self.foreground_label)   # [total #spans in the batch]

        # 计算分类代价。与损失不同,我们不使用NLL,
        # 而是用1 - prob[target class]来近似。
        # 这里的1是一个不影响匹配的常数,可以省略。
        cost_class = -out_prob[:, tgt_ids]  # [batch_size * num_queries, total #spans in the batch]

        if self.span_loss_type == "l1":
            # 展平以批量计算代价矩阵
            out_spans = outputs["pred_spans"].flatten(0, 1)  # [batch_size * num_queries, 2]

            # 计算spans之间的L1代价
            cost_span = torch.cdist(out_spans, tgt_spans, p=1)  # [batch_size * num_queries, total #spans in the batch]

            # 计算spans之间的giou代价
            cost_giou = - generalized_temporal_iou(span_cxw_to_xx(out_spans), span_cxw_to_xx(tgt_spans))
        else:
            # 处理非l1类型的span损失
            pred_spans = outputs["pred_spans"]  # (bsz, #queries, max_v_l * 2)
            pred_spans = pred_spans.view(bs * num_queries, 2, self.max_v_l).softmax(-1)  # (bsz * #queries, 2, max_v_l)
            cost_span = - pred_spans[:, 0][:, tgt_spans[:, 0]] - \
                pred_spans[:, 1][:, tgt_spans[:, 1]]  # (bsz * #queries, #spans)
            cost_giou = 0

        # 最终代价矩阵
        C = self.cost_span * cost_span + self.cost_giou * cost_giou + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        # 使用匈牙利算法计算最优匹配
        sizes = [len(v["spans"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    """构建匈牙利匹配器
    
    Args:
        args: 配置参数,包含以下字段:
            set_cost_span: span损失的权重系数
            set_cost_giou: giou损失的权重系数 
            set_cost_class: 分类损失的权重系数
            span_loss_type: span损失的类型
            max_v_l: 视频最大长度
            
    Returns:
        HungarianMatcher: 匈牙利匹配器实例,用于将预测结果与目标进行匹配
    """
    return HungarianMatcher(
        cost_span=args.set_cost_span, cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_class, span_loss_type=args.span_loss_type, max_v_l=args.max_v_l
    )
