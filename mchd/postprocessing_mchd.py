import pprint
import numpy as np
import torch
from utils.basic_utils import load_jsonl
from standalone_eval.eval import eval_submission
from tqdm import tqdm


class PostProcessorDETR:
    """
    后处理类，处理DETR（Deformable DETR）模型的预测窗口输出。

    该类提供了几种操作来后处理预测的时间窗口：
    1. 剪裁窗口的时间戳，确保它们在指定的最小值和最大值范围内。
    2. 将窗口的时间戳调整为给定的 `clip_length` 的倍数。
    3. 调整窗口的长度，确保其在指定的最小和最大长度范围内。

    每个处理函数可以按顺序应用，修改预测窗口，确保它们符合预定的规则。
    """

    def __init__(self, clip_length=2, min_ts_val=0, max_ts_val=150,
                 min_w_l=2, max_w_l=70, move_window_method="center",
                 process_func_names=("clip_window_l", "clip_ts", "round_multiple")):
        """
        初始化后处理器配置项

        参数:
            clip_length (int): 每个窗口的时间戳应被调整为的最小单位长度（默认为2秒）
            min_ts_val (int): 允许的最小时间戳值，窗口的起始和结束时间戳不能低于这个值
            max_ts_val (int): 允许的最大时间戳值，窗口的起始和结束时间戳不能高于这个值
            min_w_l (int): 最小窗口长度（单位为时间，默认为2秒），窗口的持续时间必须大于或等于这个值
            max_w_l (int): 最大窗口长度，窗口的持续时间不能大于此值（默认为70秒）
            move_window_method (str): 移动窗口的方法，有三个选项：'left'、'center'、'right'，分别表示保持左边界不变、保持中心不变、保持右边界不变
            process_func_names (tuple): 处理函数的名称元组，指定后处理步骤的顺序。默认顺序是 `clip_window_l`（调整窗口长度）、`clip_ts`（剪裁时间戳）、`round_multiple`（调整时间戳为 `clip_length` 的倍数）
        """
        # 初始化处理所需的各种参数
        self.clip_length = clip_length
        self.min_ts_val = min_ts_val
        self.max_ts_val = max_ts_val
        self.min_w_l = min_w_l
        self.max_w_l = max_w_l
        self.move_window_method = move_window_method
        self.process_func_names = process_func_names
        self.name2func = dict(
            clip_ts=self.clip_min_max_timestamps,
            round_multiple=self.round_to_multiple_clip_lengths,
            clip_window_l=self.clip_window_lengths
        )

    def __call__(self, lines):
        """
        处理每一行预测结果，并返回后处理后的结果。

        参数:
            lines (list of dict): 预测结果列表，每个元素是一个字典，包含预测窗口的起始和结束时间戳，以及分数。

        返回:
            processed_lines (list of dict): 处理后的预测结果列表，每个元素是一个字典，包含后处理后的窗口和对应的分数。
        """
        processed_lines = []  # 存储处理后的所有行
        for line in tqdm(lines, desc=f"convert to multiples of clip_length={self.clip_length}"):
            # 从每一行中提取预测的时间窗口和对应的分数
            windows_and_scores = torch.tensor(line["pred_relevant_windows"])  # 转换为张量
            windows = windows_and_scores[:, :2]  # 获取窗口的时间戳（起始时间和结束时间）

            # 按照指定的处理函数依次处理窗口
            for func_name in self.process_func_names:
                windows = self.name2func[func_name](windows)

            # 将处理后的时间窗口与原来的分数结合，更新到字典中
            line["pred_relevant_windows"] = torch.cat(
                [windows, windows_and_scores[:, 2:3]], dim=1).tolist()  # 保留分数并将窗口更新到预测结果中
            line["pred_relevant_windows"] = [
                e[:2] + [float(f"{e[2]:.4f}")] for e in line["pred_relevant_windows"]
            ]  # 格式化为四舍五入的浮动数
            processed_lines.append(line)  # 将处理后的结果添加到最终列表中
        return processed_lines  # 返回处理后的所有行

    def clip_min_max_timestamps(self, windows):
        """
        剪裁时间戳，确保所有窗口的时间戳在[min_ts_val, max_ts_val]范围内。

        参数:
            windows (torch.Tensor): 形状为 (N, 2) 的张量，表示多个窗口的时间戳（每一行代表一个窗口，包含起始时间和结束时间）

        返回:
            torch.Tensor: 剪裁后的窗口时间戳，确保所有窗口的时间戳都在指定范围内
        """
        # 使用 torch.clamp 函数将时间戳限制在最小值和最大值之间
        return torch.clamp(windows, min=self.min_ts_val, max=self.max_ts_val)

    def round_to_multiple_clip_lengths(self, windows):
        """
        将窗口的时间戳调整为 `clip_length` 的倍数。

        参数:
            windows (torch.Tensor): 形状为 (N, 2) 的张量，表示多个窗口的时间戳

        返回:
            torch.Tensor: 时间戳已调整为 `clip_length` 的倍数
        """
        # 将每个时间戳除以 clip_length 后四舍五入，然后再乘以 clip_length
        return torch.round(windows / self.clip_length) * self.clip_length

    def clip_window_lengths(self, windows):
        """
        确保每个窗口的长度在 [min_w_l, max_w_l] 范围内。如果窗口的时长不符合要求，则调整它们的时间戳。

        参数:
            windows (torch.Tensor): 形状为 (N, 2) 的张量，表示多个窗口的时间戳

        返回:
            torch.Tensor: 调整后的时间窗口，满足时长要求
        """
        # 计算每个窗口的时长（结束时间 - 起始时间）
        window_lengths = windows[:, 1] - windows[:, 0]

        # 处理时长小于最小要求的窗口
        small_rows = window_lengths < self.min_w_l
        if torch.sum(small_rows) > 0:  # 如果有窗口时长过小
            # 使用指定的移动方法调整窗口，使其时长符合要求
            windows = self.move_windows(
                windows, small_rows, self.min_w_l, move_method=self.move_window_method)

        # 处理时长大于最大要求的窗口
        large_rows = window_lengths > self.max_w_l
        if torch.sum(large_rows) > 0:  # 如果有窗口时长过大
            # 使用指定的移动方法调整窗口，使其时长符合要求
            windows = self.move_windows(
                windows, large_rows, self.max_w_l, move_method=self.move_window_method)

        return windows  # 返回调整后的窗口

    @classmethod
    def move_windows(cls, windows, row_selector, new_length, move_method="left"):
        """
        移动窗口的时间戳，使得窗口的长度符合 `new_length`，并根据选择的移动方法调整时间戳。

        参数:
            windows (torch.Tensor): 形状为 (N, 2) 的张量，表示多个窗口的时间戳
            row_selector (torch.Tensor): 选择需要调整的窗口的布尔掩码
            new_length (int): 需要的窗口的新长度
            move_method (str): 移动方法，有三种选择：
                - 'left': 保持左边界不变
                - 'center': 保持中心点不变
                - 'right': 保持右边界不变

        返回:
            torch.Tensor: 调整后的时间窗口
        """
        # 根据选定的移动方法调整窗口
        if move_method == "left":
            # 保持左边界不变，调整右边界，使得窗口的长度为 new_length
            windows[row_selector, 1] = windows[row_selector, 0] + new_length
        elif move_method == "right":
            # 保持右边界不变，调整左边界，使得窗口的长度为 new_length
            windows[row_selector, 0] = windows[row_selector, 1] - new_length
        elif move_method == "center":
            # 保持窗口的中心不变，调整左右边界，使窗口长度为 new_length
            center = (windows[row_selector, 1] + windows[row_selector, 0]) / 2.
            windows[row_selector, 0] = center - new_length / 2.
            windows[row_selector, 1] = center + new_length / 2.

        return windows  # 返回调整后的
