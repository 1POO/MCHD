# 导入所需的库
import os
import json
import zipfile
import numpy as np
import pickle
from collections import OrderedDict, Counter
import pandas as pd


def load_pickle(filename):
    """从文件中加载pickle对象"""
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    """将数据以pickle格式保存到文件"""
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(filename):
    """从文件中加载JSON数据"""
    with open(filename, "r") as f:
        return json.load(f)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    """将数据保存为JSON格式
    Args:
        data: 要保存的数据
        filename: 保存的文件名
        save_pretty: 是否格式化保存
        sort_keys: 是否对键进行排序
    """

    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_jsonl(filename):
    """加载JSONL格式文件"""
    with open(filename, "r") as f:  # 打开文件用于读取
        return [json.loads(l.strip("\n")) for l in f.readlines()]  # 读取每一行,去除换行符,解析为JSON对象并返回列表


def save_jsonl(data, filename):
    """将数据保存为JSONL格式
    Args:
        data: 要保存的列表数据
        filename: 保存的文件名
    """
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def save_lines(list_of_str, filepath):
    """将字符串列表保存到文件,每行一个字符串"""
    with open(filepath, "w") as f:
        f.write("\n".join(list_of_str))


def read_lines(filepath):
    """从文件中读取行,返回字符串列表"""
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]


def mkdirp(p):
    """创建目录,如果目录不存在"""
    if not os.path.exists(p):
        os.makedirs(p)


def flat_list_of_lists(l):
    """将嵌套列表展平为一维列表
    例如: [[1,2], [3,4]] -> [1,2,3,4]
    """
    return [item for sublist in l for item in sublist]


def convert_to_seconds(hms_time):
    """将时分秒格式转换为秒数
    Args:
        hms_time: 时分秒字符串,如'00:01:12'
    Returns:
        转换后的秒数,如72
    """
    times = [float(t) for t in hms_time.split(":")]
    return times[0] * 3600 + times[1] * 60 + times[2]


def get_video_name_from_url(url):
    """从URL中提取视频名称"""
    return url.split("/")[-1][:-4]


def merge_dicts(list_dicts):
    """合并多个字典"""
    merged_dict = list_dicts[0].copy()
    for i in range(1, len(list_dicts)):
        merged_dict.update(list_dicts[i])
    return merged_dict


def l2_normalize_np_array(np_array, eps=1e-5):
    """对numpy数组进行L2归一化
    Args:
        np_array: numpy数组,形状为(*, D)
        eps: 避免除零的小量
    """
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)


def make_zipfile(src_dir, save_path, enclosing_dir="", exclude_dirs=None, exclude_extensions=None,
                 exclude_dirs_substring=None):
    """创建源目录的zip压缩文件
    Args:
        src_dir: 源目录
        save_path: 保存路径
        enclosing_dir: 外层目录名
        exclude_dirs: 要排除的目录列表
        exclude_extensions: 要排除的文件扩展名
        exclude_dirs_substring: 要排除的目录名子串
    """
    abs_src = os.path.abspath(src_dir)
    with zipfile.ZipFile(save_path, "w") as zf:
        for dirname, subdirs, files in os.walk(src_dir):
            if exclude_dirs is not None:
                for e_p in exclude_dirs:
                    if e_p in subdirs:
                        subdirs.remove(e_p)
            if exclude_dirs_substring is not None:
                to_rm = []
                for d in subdirs:
                    if exclude_dirs_substring in d:
                        to_rm.append(d)
                for e in to_rm:
                    subdirs.remove(e)
            arcname = os.path.join(enclosing_dir, dirname[len(abs_src) + 1:])
            zf.write(dirname, arcname)
            for filename in files:
                if exclude_extensions is not None:
                    if os.path.splitext(filename)[1] in exclude_extensions:
                        continue  # do not zip it
                absname = os.path.join(dirname, filename)
                arcname = os.path.join(enclosing_dir, absname[len(abs_src) + 1:])
                zf.write(absname, arcname)


class AverageMeter(object):
    """计算并存储平均值、当前值、最大值、最小值"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10
        self.reset()

    def reset(self):
        """重置所有统计值"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10

    def update(self, val, n=1):
        """更新统计值
        Args:
            val: 当前值
            n: 当前值的权重
        """
        self.max = max(val, self.max)
        self.min = min(val, self.min)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dissect_by_lengths(np_array, lengths, dim=0, assert_equal=True):
    """将数组按照给定长度切分成子数组列表
    Args:
        np_array: 输入数组
        lengths: 切分长度列表
        dim: 切分维度
        assert_equal: 是否检查总长度相等
    """
    if assert_equal:
        assert len(np_array) == sum(lengths)
    length_indices = [0, ]
    for i in range(len(lengths)):
        length_indices.append(length_indices[i] + lengths[i])
    if dim == 0:
        array_list = [np_array[length_indices[i]:length_indices[i+1]] for i in range(len(lengths))]
    elif dim == 1:
        array_list = [np_array[:, length_indices[i]:length_indices[i + 1]] for i in range(len(lengths))]
    elif dim == 2:
        array_list = [np_array[:, :, length_indices[i]:length_indices[i + 1]] for i in range(len(lengths))]
    else:
        raise NotImplementedError
    return array_list


def get_ratio_from_counter(counter_obj, threshold=200):
    """计算Counter对象中大于阈值的值的比例"""
    keys = counter_obj.keys()
    values = counter_obj.values()
    filtered_values = [counter_obj[k] for k in keys if k > threshold]
    return float(sum(filtered_values)) / sum(values)


def get_counter_dist(counter_object, sort_type="none"):
    """计算Counter对象的分布
    Args:
        counter_object: Counter对象
        sort_type: 排序方式
    """
    _sum = sum(counter_object.values())
    dist = {k: float(f"{100 * v / _sum:.2f}") for k, v in counter_object.items()}
    if sort_type == "value":
        dist = OrderedDict(sorted(dist.items(), reverse=True))
    return dist


def get_show_name(vid_name):
    """从视频名称获取电视节目名称
    Args:
        vid_name: 视频片段名称
    Returns:
        电视节目名称
    """
    show_list = ["friends", "met", "castle", "house", "grey"]
    vid_name_prefix = vid_name.split("_")[0]
    show_name = vid_name_prefix if vid_name_prefix in show_list else "bbt"
    return show_name


def get_abspaths_by_ext(dir_path, ext=(".jpg",)):
    """获取目录下指定扩展名的文件的绝对路径
    Args:
        dir_path: 目录路径
        ext: 文件扩展名元组
    """
    if isinstance(ext, list):
        ext = tuple(ext)
    if isinstance(ext, str):
        ext = tuple([ext, ])
    filepaths = [os.path.join(root, name)
                 for root, dirs, files in os.walk(dir_path)
                 for name in files
                 if name.endswith(tuple(ext))]
    return filepaths


def get_basename_no_ext(path):
    """获取路径中的文件名(不含扩展名)
    例如: '/data/movienet/240p_keyframe_feats/tt7672188.npz' --> 'tt7672188'
    """
    return os.path.splitext(os.path.split(path)[1])[0]


def dict_to_markdown(d, max_str_len=120):
    """将字典转换为Markdown格式
    Args:
        d: 输入字典
        max_str_len: 字符串最大长度
    """
    # 将列表转换为字符串表示
    d = {k: v.__repr__() if isinstance(v, list) else v for k, v in d.items()}
    # 截断超过最大长度的字符串
    if max_str_len is not None:
        d = {k: v[-max_str_len:] if isinstance(v, str) else v for k, v in d.items()}
    return pd.DataFrame(d, index=[0]).transpose().to_markdown()
