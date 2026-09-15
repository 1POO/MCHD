# 导入必要的库
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
import logging
from os.path import join, exists
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from mchd.span_utils import span_xx_to_cxw

logger = logging.getLogger(__name__)


class StartEndDataset(Dataset):
    # 定义支持的查询特征类型
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
    """数据集中每条数据的格式示例:
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dirs,q_feat_dir,
                 q_feat_type="last_hidden_state", # 查询特征类型,默认使用BERT的最后一层隐藏状态作为特征
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0,
                 dset_domain=None):
        """初始化数据集
        Args:
            dset_name: 数据集名称
            data_path: 数据文件路径
            v_feat_dirs: 视频特征目录
            q_feat_dir: 查询特征目录
            q_feat_type: 查询特征类型
            max_q_l: 最大查询长度
            max_v_l: 最大视频长度
            data_ratio: 使用数据的比例
            ctx_mode: 上下文模式
            normalize_v: 是否归一化视频特征
            normalize_t: 是否归一化文本特征
            load_labels: 是否加载标签
            clip_len: 视频片段长度
            max_windows: 最大窗口数
            span_loss_type: 跨度损失类型
            txt_drop_ratio: 文本丢弃比例
            dset_domain: 数据集域
        """
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dirs = q_feat_dirs
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.max_windows = max_windows  # 用作标签的最大窗口数
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0

        # 检查查询特征类型是否合法
        assert q_feat_type in self.Q_FEAT_TYPES

        # 加载数据
        self.data = self.load_data()
        
        # 为tvsum数据集加载特定域的数据
        if self.dset_name == 'tvsum':
            target_domain = dset_domain
            assert target_domain in ["BK", "BT", "DS", "FM", "GA", "MS", "PK", "PR", "VT", "VU"]

            new_data = []
            for d in self.data:
                if target_domain == d['domain']:
                    new_data.append(d)
            self.data = new_data
        

    def load_data(self):
        """加载数据文件
        Returns:
            datalist: 数据列表
        """
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, index):
        """获取单个数据样本
        Args:
            index: 数据索引
        Returns:
            dict: 包含元数据和模型输入的字典
        """
        # 获取当前索引的元数据
        meta = self.data[index]

        # 创建模型输入字典
        model_inputs = dict()
        # 获取查询特征
        model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)
        if self.use_video:
            # 如果使用视频,获取视频特征
            model_inputs["video_feat"],model_inputs["video_object_feat"] = self._get_video_feat_by_vid(meta["vid"])  # (Lv, Dv)
            ctx_vl = len(model_inputs["video_feat"])
            ctx_ol = len(model_inputs["video_object_feat"])
        else:
            # 否则使用最大视频长度
            ctx_vl = self.max_v_l
            ctx_ol = self.max_v_l

        if self.use_tef:
            # 如果使用时间编码特征(TEF)
            # 生成开始和结束时间编码
            tef_vst = torch.arange(0, ctx_vl, 1.0) / ctx_vl
            tef_ost = torch.arange(0, ctx_ol, 1.0) / ctx_ol
            tef_ved = tef_vst + 1.0 / ctx_vl
            tef_oed = tef_ost + 1.0 / ctx_ol

            tef_v = torch.stack([tef_vst, tef_ved], dim=1)  # (Lv, 2)
            tef_o = torch.stack([tef_ost, tef_oed], dim=1)
            if self.use_video:
                # 将TEF与视频特征拼接
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef_v], dim=1)  # (Lv, Dv+2)
                model_inputs["video_object_feat"] = torch.cat([model_inputs["video_object_feat"],tef_o],dim=1)
            else:
                # 仅使用TEF作为特征
                model_inputs["video_feat"] = tef_v
                model_inputs["video_object_feat"] = tef_o

        if self.load_labels:
            if self.dset_name == 'tvsum':
                # TVSum数据集的特殊处理
                max_l = ctx_l//2

                # 获取标签并计算聚合分数
                meta_label = meta['label']
                agg_scores = np.sum(meta_label - np.ones_like(meta_label), axis=-1)[:ctx_l] # 从1开始,所以减1
                sort_indices = np.argsort(agg_scores)  # 升序排序
                pos_idx = torch.tensor(sort_indices[max_l:])

                # 创建掩码张量
                mask = torch.zeros_like(torch.ones(ctx_l)) # ctx_l=75

                # 处理掩码长度
                if pos_idx.max() >= len(mask):
                    new_mask = torch.zeros_like(torch.ones(pos_idx.max()+1 ))
                    new_mask[pos_idx] = 1
                    new_mask[:len(mask)] = mask
                    mask = new_mask
                else:
                    mask[pos_idx] = 1

                model_inputs["pos_mask"] = mask

                # 获取负样本索引
                neg_idx = torch.tensor(list(set(range(ctx_l)) - set(pos_idx)))

                # 填充正样本索引张量
                pad_tensor = torch.ones(ctx_l) * -2
                pad_tensor[:len(pos_idx)] = pos_idx
                model_inputs["pos_idx"] = pad_tensor

                # 填充负样本索引张量
                pad_tensor = torch.ones(ctx_l) * -2
                pad_tensor[:len(neg_idx)] = neg_idx
                model_inputs["neg_idx"] = pad_tensor

                # 设置跨度标签和显著性标签
                model_inputs["span_labels"] = torch.tensor([[0., 0.]])
                meta_label = meta['label']
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                            self.get_saliency_labels_all_tvsum(meta_label, ctx_l)
            else:
                # 其他数据集的处理
                # 获取正样本索引并创建掩码
                pos_idx = torch.tensor(meta['relevant_clip_ids']) # 根据相关片段索引获得位置id
                mask_v = torch.zeros_like(torch.ones(ctx_vl)) # 创建和视频片段长度一样的全0向量
                mask_o = torch.zeros_like(torch.ones(ctx_ol))

                # 处理掩码长度
                if pos_idx.max() >= len(mask_v):
                    new_mask = torch.zeros_like(torch.ones(pos_idx.max()+1 ))
                    new_mask[pos_idx] = 1
                    new_mask[:len(mask_v)] = mask_v
                    mask_v = new_mask
                else:
                    mask_v[pos_idx] = 1 # 和位置id相同位置为1

                model_inputs["pos_mask_v"] = mask_v
                model_inputs["pos_mask_o"] = mask_v
                # 处理掩码长度
                if pos_idx.max() >= len(mask_o):
                    new_mask = torch.zeros_like(torch.ones(pos_idx.max() + 1))
                    new_mask[pos_idx] = 1
                    new_mask[:len(mask_o)] = mask_o
                    mask_o = new_mask
                else:
                    mask_o[pos_idx] = 1  # 和位置id相同位置为1

                model_inputs["pos_mask_o"] = mask_o

                # 获取跨度标签
                model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_vl)  # (#windows, 2)
                if "subs_train" not in self.data_path:
                    # 非字幕训练数据的显著性标签
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_all(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_vl)
                else:
                    # 字幕训练数据的显著性标签
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_vl)  # 只使用一个ground truth

        return dict(meta=meta, model_inputs=model_inputs)

    def get_saliency_labels_sub_as_query(self, gt_window, ctx_l, max_n=2):
        """获取以查询为基准的显著性标签
        Args:
            gt_window: ground truth窗口
            ctx_l: 上下文长度
            max_n: 最大样本数
        Returns:
            pos_clip_indices: 正样本片段索引
            neg_clip_indices: 负样本片段索引
            score_array: 分数数组
        """
        gt_st = int(gt_window[0] / self.clip_len)
        gt_ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed+1), k=max_n)
        else:
            pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l))
        neg_clip_indices = random.sample(neg_pool, k=max_n)
        
        score_array = np.zeros(ctx_l)
        score_array[gt_st:gt_ed+1] = 1

        return pos_clip_indices, neg_clip_indices, score_array
        

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """获取显著性标签。将三个标注的分数相加,然后取最高分的两个片段作为正样本,最低分的两个作为负样本。
        Args:
            rel_clip_ids: 相关片段ID列表
            scores: 分数列表[anno1_score, anno2_score, anno3_score]
            ctx_l: 上下文长度
            max_n: 用作正负样本的最大片段数
            add_easy_negative: 是否添加简单负样本
        Returns:
            pos_clip_indices: 正样本片段索引
            neg_clip_indices: 负样本片段索引
        """
        # 在rel_clip_ids内的索引
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # 升序

        # 在整个视频中的索引
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # 复制困难样本
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels_all(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """获取所有显著性标签。将三个标注的分数相加,然后取最高分的两个片段作为正样本,最低分的两个作为负样本。
        Args:
            rel_clip_ids: 相关片段ID列表
            scores: 分数列表[anno1_score, anno2_score, anno3_score]
            ctx_l: 上下文长度
            max_n: 用作正负样本的最大片段数
            add_easy_negative: 是否添加简单负样本
        Returns:
            pos_clip_indices: 正样本片段索引
            neg_clip_indices: 负样本片段索引
            score_array: 分数数组
        """
        # 在rel_clip_ids内的索引
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # 升序

        score_array = np.zeros(ctx_l)
        for idx in range(len(rel_clip_ids)):
            if rel_clip_ids[idx] >= ctx_l:
                score_array_new = np.zeros(ctx_l + 1)
                score_array_new[:ctx_l] = score_array
                score_array = score_array_new
            score_array[rel_clip_ids[idx]] = agg_scores[idx]

        # 在整个视频中的索引
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # 复制困难样本
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels_all_tvsum(self, labels, ctx_l, max_n=1, add_easy_negative=False):
        """获取TVSum数据集的所有显著性标签
        Args:
            labels: 标签数组
            ctx_l: 上下文长度
            max_n: 最大样本数
            add_easy_negative: 是否添加简单负样本
        Returns:
            pos_clip_indices: 正样本片段索引
            neg_clip_indices: 负样本片段索引
            score_array: 分数数组
        """
        agg_scores = np.sum(labels - np.ones_like(labels), axis=-1)[:ctx_l] # 从1开始,所以减1
        score_array = agg_scores / 80 * 12
        sort_indices = np.argsort(agg_scores)  # 升序

        hard_pos_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # 复制困难样本
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices, score_array

    def get_span_labels(self, windows, ctx_l):
        """获取跨度标签
        Args:
            windows: 窗口列表[st, ed],单位为秒。例如[[26, 36]],对应的st_ed片段索引为[[13, 17]](包含)
                注意最多使用self.max_windows个窗口
            ctx_l: 上下文长度
        Returns:
            windows: 标准化后的窗口张量(#windows, 2),每行为[center, width]
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # 标准化的xx格式窗口
            windows = span_xx_to_cxw(windows)  # 标准化的cxw格式窗口
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # 包含
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid):
        """根据查询ID获取查询特征
        Args:
            qid: 查询ID
        Returns:
            查询特征张量
        """
        # 根据数据集类型处理查询特征
        if self.dset_name == 'tvsum':
            # 对于TVSum数据集,直接加载token特征
            q_feat = np.load(join(self.q_feat_dir, "{}.npz".format(qid))) # 'token', 'text'
            return torch.from_numpy(q_feat['token'])
        else:
            # 对于QVHighlight数据集:
            # 1. 加载特征文件
            # 构建查询特征文件路径
            t_feat_list = []
            # print(qid)
            for _feat_dir in self.q_feat_dirs:
                if "clip" in _feat_dir:
                    q_feat_path = join(_feat_dir, f"qid{qid}.npz")
                    q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
                    if self.q_feat_type == "last_hidden_state":
                        q_feat = q_feat[:self.max_q_l]  # 保留前max_q_l个token的特征

                    # 数据预处理
                    if self.normalize_t:
                        q_feat = l2_normalize_np_array(q_feat)
                    if self.txt_drop_ratio > 0:
                        q_feat = self.random_drop_rows(q_feat)

                else:
                    q_feat_path = join(_feat_dir, f"qid{qid}.npz")
                    q_feat = np.load(q_feat_path)["features"].astype(np.float32)

                    # 数据预处理
                    if self.normalize_t:
                        q_feat = l2_normalize_np_array(q_feat)
                    if self.txt_drop_ratio > 0:
                        q_feat = self.random_drop_rows(q_feat)

                t_feat_list.append(q_feat)

                # 新增的维度补齐逻辑
            if not t_feat_list:
                return np.array([], dtype=np.float32)

                # 计算最大行数
            max_rows = max(feat.shape[0] for feat in t_feat_list)

            # 补齐维度
            padded_feats = []
            for feat in t_feat_list:
                current_rows = feat.shape[0]
                if current_rows < max_rows:
                    # 使用零填充行
                    pad = np.zeros((max_rows - current_rows, feat.shape[1]), dtype=np.float32)
                    padded_feat = np.vstack([feat, pad])
                else:
                    padded_feat = feat
                padded_feats.append(padded_feat)

            # 最终拼接
            q_feat = np.concatenate(padded_feats, axis=1)

        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def random_drop_rows(self, embeddings):
        """随机将embeddings中的num_drop行置为零
        Args:
            embeddings: np.ndarray (L, D)
        Returns:
            处理后的embeddings
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_video_feat_by_vid(self, vid):
        """根据视频ID获取视频特征
        Args:
            vid: 视频ID
        Returns:
            视频特征张量
        """
        if self.dset_name == 'tvsum':
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                _feat_path = join(_feat_dir, f"{vid}_rgb.npy")
                _feat_rgb = np.load(_feat_path)[:self.max_v_l].astype(np.float32)

                _feat_path = join(_feat_dir, f"{vid}_opt.npy")
                _feat_opt = np.load(_feat_path)[:self.max_v_l].astype(np.float32)
                
                _feat = np.concatenate([_feat_rgb, _feat_opt], axis=-1)
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            # 某些特征略长于其他特征
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)

        else:
            v_objectFeat_dict = {}
            v_feat_list = []
            # 改
            for _feat_dir in self.v_feat_dirs:
                # _feat_path = join(_feat_dir, f"{vid}.npz")
                # _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                # if self.normalize_v:
                #     _feat = l2_normalize_np_array(_feat)
                # v_feat_list.append(_feat)
                if "object" in _feat_dir:
                    _feat_path = _feat_dir + "/" + f"{vid}.npz"
                    _feat = (np.load(_feat_path)["frame_features"])  # (75,2560)
                    n_frames = _feat.shape[0]

                    # 初始化位置编码张量
                    position_encoding = np.zeros((n_frames, 5, 2))

                    for i in range(n_frames):
                        # 计算时间步t，范围在0到1之间
                        t = i / (n_frames - 1)

                        for j in range(5):
                            # 计算对象位置o，范围在0到1之间
                            o = j / (5 - 1)
                            position_encoding[i, j] = [t, o]
                    _feat = np.concatenate([_feat, position_encoding], axis=2)
                    _feat = _feat.reshape(n_frames, -1)
                    if self.normalize_v:
                        _feat = l2_normalize_np_array(_feat)
                    v_objectFeat_dict["v_object_feature"] = _feat

                    # print(_feat.shape)
                    # _feat_path = join(_feat_dir, f"{vid}.npz")
                else:
                    _feat_path = _feat_dir + "/" + f"{vid}.npz"
                    try:
                        _feat = np.load(_feat_path)["features"][:75].astype(np.float32)
                    except Exception as e:
                        print(f"Error loading feature for vid:{vid}, path:{_feat_path}")
                        raise e

                    if True:
                        _feat = l2_normalize_np_array(_feat)
                    v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)

        return torch.from_numpy(v_feat),torch.from_numpy(v_objectFeat_dict["v_object_feature"]) # (Lv, D)



def start_end_collate(batch):
    """批处理数据的整理函数
    Args:
        batch: 批数据
    Returns:
        batch_meta: 批元数据
        batched_data: 批处理后的数据
    """
    batch_meta = [e["meta"] for e in batch]  # 似乎不需要整理?

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        if k == "saliency_all_labels":
            pad_data, mask_data = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=np.float32, fixed_length=None)
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue

        batched_data[k] = pad_sequences_1d(
            [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    """准备批处理输入
    Args:
        batched_model_inputs: 批处理后的模型输入
        device: 设备
        non_blocking: 是否非阻塞
    Returns:
        model_inputs: 模型输入
        targets: 目标
    """
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
        src_obj_vid=batched_model_inputs["video_object_feat"][0].to(device, non_blocking=non_blocking),
        src_obj_vid_mask=batched_model_inputs["video_object_feat"][1].to(device, non_blocking=non_blocking)
    )
    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)
    
    if "pos_mask_v" in batched_model_inputs:
        targets['src_pos_mask_v']=batched_model_inputs["pos_mask_v"][0].to(device, non_blocking=non_blocking)
    targets = None if len(targets) == 0 else targets

    if "pos_mask_o" in batched_model_inputs:
        targets['src_pos_mask_o']=batched_model_inputs["pos_mask_o"][0].to(device, non_blocking=non_blocking)
    targets = None if len(targets) == 0 else targets
    return model_inputs, targets
