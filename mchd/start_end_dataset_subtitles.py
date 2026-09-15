import torch
from torch.utils.data import Dataset
import sys

sys.path.append(".")
import numpy as np
from tqdm import tqdm
import random
import logging
from os.path import join, exists
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from mchd.span_utils import span_xx_to_cxw

logger = logging.getLogger(__name__)


class StartEndDataset_subtitles(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir, q_feat_dirs, s_feat_dir=None,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0,
                 dset_domain=None):
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_dirs = q_feat_dirs
        self.s_feat_dir = s_feat_dir

        self.q_feat_type = q_feat_type
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.use_sub_tef = "tef____" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES

        # data
        self.data = self.load_data()

        # load specific domain data for tvsum dataset
        if self.dset_name == 'tvsum':
            target_domain = dset_domain
            assert target_domain in ["BK", "BT", "DS", "FM", "GA", "MS", "PK", "PR", "VT", "VU"]

            new_data = []
            for d in self.data:
                if target_domain == d['domain']:
                    new_data.append(d)
            self.data = new_data

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]

        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)
        if self.use_video:
            # 如果使用视频,获取视频特征
            model_inputs["video_feat"], model_inputs["object_feat"] = self._get_video_feat_by_vid(
                meta["vid"])  # (Lv, Dv)
            ctx_vl = len(model_inputs["video_feat"])
            ctx_ol = len(model_inputs["object_feat"])
        else:
            # 否则使用最大视频长度
            ctx_vl = self.max_v_l
            ctx_ol = self.max_v_l
        if self.s_feat_dir is not None:
            model_inputs["subtitles_feat"] = self._get_subtitles_feat_by_vid(meta["vid"])  # (Lv, Da) 75 2048
            ctx_sl = len(model_inputs["subtitles_feat"])

            ctx_vl = min(ctx_vl, ctx_sl, ctx_ol)
            model_inputs["video_feat"] = model_inputs["video_feat"][:ctx_vl, :]
            model_inputs["subtitles_feat"] = model_inputs["subtitles_feat"][:ctx_vl, :]
            model_inputs["object_feat"] = model_inputs["object_feat"][:ctx_vl, :]

        if self.use_tef:
            # 如果使用时间编码特征(TEF)
            # 生成开始和结束时间编码
            tef_vst = torch.arange(0, ctx_vl, 1.0) / ctx_vl
            tef_ost = torch.arange(0, ctx_vl, 1.0) / ctx_vl
            tef_sst = torch.arange(0, ctx_vl, 1.0) / ctx_vl
            tef_ved = tef_vst + 1.0 / ctx_vl
            tef_oed = tef_ost + 1.0 / ctx_vl
            tef_sed = tef_sst + 1.0 / ctx_vl
            tef_v = torch.stack([tef_vst, tef_ved], dim=1)  # (Lv, 2)
            tef_o = torch.stack([tef_ost, tef_oed], dim=1)
            tef_s = torch.stack([tef_sst, tef_sed], dim=1)
            # print(tef.shape, model_inputs['video_feat'].shape, model_inputs['audio_feat'].shape)
            if self.use_video:
                # 将TEF与视频特征拼接
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef_v], dim=1)  # (Lv, Dv+2)
                model_inputs["object_feat"] = torch.cat([model_inputs["object_feat"], tef_o], dim=1)
            else:
                # 仅使用TEF作为特征
                model_inputs["video_feat"] = tef_v
                model_inputs["object_feat"] = tef_o
            # 字幕特征先不加时间
            if self.use_sub_tef:
                model_inputs["subtitles_feat"] = torch.cat(
                    [model_inputs["subtitles_feat"], tef_s], dim=1)  # (Lv, Dv+2)

        if len(model_inputs["query_feat"].shape) == 3:
            # There is batch dimension, which I should have removed at the feature extraction time, but didn't.
            # Remove it online
            model_inputs["query_feat"] = model_inputs["query_feat"][0]

        if self.load_labels:
            if self.dset_name == 'tvsum':
                max_l = ctx_vl // 2

                meta_label = meta['label']
                agg_scores = np.sum(meta_label - np.ones_like(meta_label), axis=-1)[:ctx_vl]  # start from 1, so minus 1
                sort_indices = np.argsort(agg_scores)  # increasing
                pos_idx = torch.tensor(sort_indices[max_l:])
                neg_idx = torch.tensor(list(set(range(ctx_vl)) - set(pos_idx)))

                pad_tensor = torch.ones(ctx_vl) * -2
                pad_tensor[:len(pos_idx)] = pos_idx
                model_inputs["pos_idx"] = pad_tensor

                pad_tensor = torch.ones(ctx_vl) * -2
                pad_tensor[:len(neg_idx)] = neg_idx
                model_inputs["neg_idx"] = pad_tensor

                model_inputs["span_labels"] = torch.tensor([[0., 0.]])
                meta_label = meta['label']
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs[
                    "saliency_all_labels"] = \
                    self.get_saliency_labels_all_tvsum(meta_label, ctx_vl)
                ctx_vl = min(ctx_vl, ctx_sl, ctx_ol)
                mask_v = torch.zeros_like(torch.ones(ctx_vl))  # 创建和视频片段长度一样的全0向量
                mask_o = torch.zeros_like(torch.ones(ctx_vl))
                mask_s = torch.zeros_like(torch.ones(ctx_vl))

                # 处理掩码长度
                if pos_idx.max() >= len(mask_v):
                    new_mask = torch.zeros_like(torch.ones(pos_idx.max() + 1))
                    new_mask[pos_idx] = 1
                    new_mask[:len(mask_v)] = mask_v
                    mask_v = new_mask
                else:
                    mask_v[pos_idx] = 1  # 和位置id相同位置为1

                model_inputs["pos_mask_v"] = mask_v

                # 处理掩码长度
                if pos_idx.max() >= len(mask_o):
                    new_mask = torch.zeros_like(torch.ones(pos_idx.max() + 1))
                    new_mask[pos_idx] = 1
                    new_mask[:len(mask_o)] = mask_o
                    mask_o = new_mask
                else:
                    mask_o[pos_idx] = 1  # 和位置id相同位置为1

                    # 处理掩码长度
                if pos_idx.max() >= len(mask_s):
                    new_mask = torch.zeros_like(torch.ones(pos_idx.max() + 1))
                    new_mask[pos_idx] = 1
                    new_mask[:len(mask_s)] = mask_s
                    mask_s = new_mask
                else:
                    mask_s[pos_idx] = 1  # 和位置id相同位置为1

                model_inputs["pos_mask_o"] = mask_o

                model_inputs["pos_mask_s"] = mask_s
            elif self.dset_name == "youtube_uni":
                model_inputs["span_labels"] = torch.tensor([[0.0, 0.0]])
                meta_label = meta["label"]
                (
                    model_inputs["saliency_pos_labels"],
                    model_inputs["saliency_neg_labels"],
                    model_inputs["saliency_all_labels"],
                ) = self.get_saliency_labels_all_youtube(meta_label, ctx_vl)
            else:
                if "relevant_windows" in meta:  ## For Qvhighlights test set
                    model_inputs["span_labels"] = self.get_span_labels(
                        meta["relevant_windows"], ctx_vl
                    )  # (#windows, 2)
                    if self.dset_name in [
                        "charadesSTA",
                        "tacos",
                        "activitynet",
                    ]:  ## charades, tacos, nlq
                        (
                            model_inputs["saliency_pos_labels"],
                            model_inputs["saliency_neg_labels"],
                            model_inputs["saliency_all_labels"],
                        ) = self.get_saliency_labels_sub_as_query(
                            meta["relevant_windows"][0], meta["duration"], ctx_vl
                        )  # only one gt
                        # 参数设置
                        time_unit = 2.0  # 1秒为时间片粒度
                        duration = meta["duration"]
                        windows = [
                            [max(0, min(w[0], duration)), max(0, min(w[1], duration))]
                            for w in meta["relevant_windows"]
                        ]

                        # 计算总位置数
                        total_positions = ctx_vl

                        # 初始化全0掩码
                        location_mask = [0] * total_positions

                        # 填充积极位置
                        for w_start, w_end in windows:
                            start_idx = int(np.floor(w_start / time_unit))
                            end_idx = int(np.ceil(w_end / time_unit))
                            for i in range(start_idx, end_idx):
                                if i < total_positions:
                                    location_mask[i] = 1
                        location_mask = torch.tensor(location_mask)
                        model_inputs["pos_mask_v"] = location_mask
                        model_inputs["pos_mask_s"] = location_mask
                        model_inputs["pos_mask_o"] = location_mask

                    elif self.dset_name in ["nlq"]:
                        (
                            model_inputs["saliency_pos_labels"],
                            model_inputs["saliency_neg_labels"],
                            model_inputs["saliency_all_labels"],
                        ) = self.get_saliency_labels_sub_as_query(
                            meta["relevant_windows"][0], meta["duration"], ctx_vl, 2
                        )  # only one gt
                    elif "subs_train" not in self.data_path:
                        (
                            model_inputs["saliency_pos_labels"],
                            model_inputs["saliency_neg_labels"],
                            model_inputs["saliency_all_labels"],
                        ) = self.get_saliency_labels_all(
                            meta["relevant_clip_ids"], meta["saliency_scores"], ctx_vl
                        )
                    else:
                        (
                            model_inputs["saliency_pos_labels"],
                            model_inputs["saliency_neg_labels"],
                            model_inputs["saliency_all_labels"],
                        ) = self.get_saliency_labels_sub_as_query(
                            meta["relevant_windows"][0], meta["duration"], ctx_vl
                        )  # only one gt

            if "hl" in self.data_path:

                # 其他数据集的处理
                # 获取正样本索引并创建掩码
                pos_idx = torch.tensor(meta['relevant_clip_ids'])  # 根据相关片段索引获得位置id
                mask_v = torch.zeros_like(torch.ones(ctx_vl))  # 创建和视频片段长度一样的全0向量
                mask_o = torch.zeros_like(torch.ones(ctx_ol))
                mask_s = torch.zeros_like(torch.ones(ctx_sl))

                # 处理掩码长度
                if pos_idx.max() >= len(mask_v):
                    new_mask = torch.zeros_like(torch.ones(pos_idx.max() + 1))
                    new_mask[pos_idx] = 1
                    new_mask[:len(mask_v)] = mask_v
                    mask_v = new_mask
                else:
                    mask_v[pos_idx] = 1  # 和位置id相同位置为1

                model_inputs["pos_mask_v"] = mask_v

                # 处理掩码长度
                if pos_idx.max() >= len(mask_o):
                    new_mask = torch.zeros_like(torch.ones(pos_idx.max() + 1))
                    new_mask[pos_idx] = 1
                    new_mask[:len(mask_o)] = mask_o
                    mask_o = new_mask
                else:
                    mask_o[pos_idx] = 1  # 和位置id相同位置为1

                    # 处理掩码长度
                if pos_idx.max() >= len(mask_s):
                    new_mask = torch.zeros_like(torch.ones(pos_idx.max() + 1))
                    new_mask[pos_idx] = 1
                    new_mask[:len(mask_s)] = mask_s
                    mask_s = new_mask
                else:
                    mask_s[pos_idx] = 1  # 和位置id相同位置为1

                model_inputs["pos_mask_o"] = mask_o

                model_inputs["pos_mask_s"] = mask_s

                # 获取跨度标签
                model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_vl)  # (#windows, 2)
                if "subs_train" not in self.data_path:
                    # 非字幕训练数据的显著性标签
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs[
                        "saliency_all_labels"] = \
                        self.get_saliency_labels_all(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_vl)
                else:
                    # 字幕训练数据的显著性标签
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs[
                        "saliency_all_labels"] = \
                        self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_vl)  # 只使用一个ground truth
        model_inputs = model_inputs
        return dict(meta=meta, model_inputs=model_inputs)

    def get_saliency_labels_sub_as_query(self, gt_window, duration, ctx_l, max_n=2):
        clip_len = duration / ctx_l
        gt_st = int(gt_window[0] / clip_len)
        gt_ed = max(0, min(int(gt_window[1] / clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed + 1), k=max_n)
        else:
            if self.dset_name == "nlq":
                pos_clip_indices = [gt_st] * 2
            else:
                pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed + 1, ctx_l))
        try:
            neg_clip_indices = random.sample(neg_pool, k=max_n)
        except:
            neg_clip_indices = pos_clip_indices

        # For charades_sta
        score_array = np.zeros(ctx_l)
        score_array[gt_st: gt_ed + 1] = 1

        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels_all(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # score_array = [min(agg_scores[idx], ctx_l-1) for idx in range(ctx_l)]
        score_array = np.zeros(ctx_l)
        for idx in range(len(rel_clip_ids)):
            if rel_clip_ids[idx] >= ctx_l:
                score_array_new = np.zeros(ctx_l + 1)
                score_array_new[:ctx_l] = score_array
                score_array = score_array_new
            # if rel_clip_ids[idx] == ctx_l:
            #     print(rel_clip_ids[idx], ctx_l)
            score_array[rel_clip_ids[idx]] = agg_scores[idx]

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l - 1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels_all_tvsum(self, labels, ctx_l, max_n=1, add_easy_negative=False):

        agg_scores = np.sum(labels - np.ones_like(labels), axis=-1)[:ctx_l]  # start from 1, so minus 1
        score_array = agg_scores / 80 * 12
        sort_indices = np.argsort(agg_scores)  # increasing

        hard_pos_clip_indices = [min(idx, ctx_l - 1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(idx, ctx_l - 1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices, score_array

    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid):
        if self.dset_name == 'tvsum':
            q_feat = np.load(join(self.q_feat_dir, "{}.npz".format(qid)))  # 'token', 'text'
            return torch.from_numpy(q_feat['token'])
        elif self.dset_name == 'tacos':
            for _feat_dir in self.q_feat_dirs:
                q_feat_path = join(_feat_dir, f"qid{qid}.npz")
                q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
                if self.q_feat_type == "last_hidden_state":
                    q_feat = q_feat[: self.max_q_l]
                if self.normalize_t:
                    q_feat = l2_normalize_np_array(q_feat)
                if self.txt_drop_ratio > 0:
                    q_feat = self.random_drop_rows(q_feat)
        else:
            # QVhighlight dataset

            # only clip feat
            # q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")
            # q_feat = np.load(q_feat_path)["features"].astype(np.float32)
            # 原
            # q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
            # print(q_feat.shape)
            # if self.q_feat_type == "last_hidden_state":
            #     q_feat = q_feat[:self.max_q_l]

            # 改query特征
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
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
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
            v_objectFeat_dict = {}
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
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
                else:
                    _feat_path = _feat_dir + "/" + f"{vid}.npz"
                    _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                    if True:
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
                    _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                    if True:
                        _feat = l2_normalize_np_array(_feat)
                    v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)

        return torch.from_numpy(v_feat), torch.from_numpy(v_objectFeat_dict["v_object_feature"])  # (Lv, D)

    # def _get_subtitles_feat_by_vid(self, vid):
    #     s_feat_path = join(self.s_feat_dir, f"{vid}.npz")
    #     s_feat = np.load(s_feat_path)[:self.max_v_l].astype(np.float32)
    #     if self.normalize_v:
    #         s_feat = l2_normalize_np_array(s_feat)

    #     return torch.from_numpy(s_feat)  # (D, ) or (Lq, D)
    def _get_subtitles_feat_by_vid(self, vid):
        s_feat_path = join(self.s_feat_dir, f"{vid}.npz")
        s_feat = np.load(s_feat_path)
        # 从npz文件中获取第一个数组（假设文件中只有一个数组）
        s_feat = s_feat["features"][:self.max_v_l].astype(np.float32)
        if self.normalize_v:
            s_feat = l2_normalize_np_array(s_feat)

        return torch.from_numpy(s_feat)  # (D, ) or (Lq, D)


def start_end_collate_subtitles(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

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
            pad_data, mask_data = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=np.float32,
                                                   fixed_length=None)
            # print(pad_data, mask_data)
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue

        batched_data[k] = pad_sequences_1d(
            [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data


def prepare_batch_inputs_subtitles(opt, batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
        src_obj=batched_model_inputs["object_feat"][0].to(device, non_blocking=non_blocking),
        src_obj_mask=batched_model_inputs["object_feat"][1].to(device, non_blocking=non_blocking),
        src_sub=batched_model_inputs["subtitles_feat"][0].to(device, non_blocking=non_blocking),
        src_sub_mask=batched_model_inputs["subtitles_feat"][1].to(device, non_blocking=non_blocking)

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
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device,
                                                                                        non_blocking=non_blocking)

    if "query_pos_tag" in batched_model_inputs:
        targets["query_pos_tag"] = batched_model_inputs["query_pos_tag"][0].to(device, non_blocking=non_blocking),
    if "pos_idx" in batched_model_inputs:
        targets["src_pos_idx"] = batched_model_inputs["pos_idx"][0].to(device, non_blocking=non_blocking),
    if "neg_idx" in batched_model_inputs:
        targets["src_neg_idx"] = batched_model_inputs["neg_idx"][0].to(device, non_blocking=non_blocking),

    if "pos_mask_v" in batched_model_inputs:
        targets['src_pos_mask_v'] = batched_model_inputs["pos_mask_v"][0].to(device, non_blocking=non_blocking)
    targets = None if len(targets) == 0 else targets

    if "pos_mask_o" in batched_model_inputs:
        targets['src_pos_mask_o'] = batched_model_inputs["pos_mask_o"][0].to(device, non_blocking=non_blocking)
    targets = None if targets == None else targets

    if "pos_mask_s" in batched_model_inputs:
        targets['src_pos_mask_s'] = batched_model_inputs["pos_mask_s"][0].to(device, non_blocking=non_blocking)
    targets = None if targets == None else targets
    return model_inputs, targets
