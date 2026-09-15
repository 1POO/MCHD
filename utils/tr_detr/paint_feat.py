import numpy as np
import matplotlib.pyplot as plt
import torch


# def paint_feat(feat):
#     """
#     可视化特征图，并增强颜色对比度。
#     输入特征形状为 (1, 75, 256)，其中：
#     - 1 是批次大小，
#     - 75 是时间步长（帧数），
#     - 256 是特征维度。
#     """
#     # 将输入张量转换为 numpy 数组，并确保计算图和梯度计算被停止
#     feat = feat.detach().cpu().numpy()
#
#     # 去掉批次维度，得到形状为 (75, 256)
#     feat = feat.squeeze(0)
#
#     # 计算数据的均值和标准差
#     mean = feat.mean()
#     std = feat.std()
#
#     # 设置 vmin 和 vmax 为均值 ± 标准差，增强对比度
#     vmin = mean - std
#     vmax = mean + std
#
#     # 设置横坐标范围为 0 到 150
#     xmin, xmax = 0, 150  # 横坐标范围
#     ymin, ymax = 0, feat.shape[0]  # 纵坐标范围（帧数）
#
#     # 绘制热图
#     plt.figure(figsize=(12, 6))
#     plt.imshow(feat, cmap='Blues', aspect='auto', vmin=vmin, vmax=vmax,
#                extent=[xmin, xmax, ymax, ymin])  # 设置 extent 参数
#     plt.colorbar(label='Feature Value')  # 添加颜色条
#
#     # 设置横坐标刻度和标签（更加细致）
#     xticks = np.arange(xmin, xmax + 1, 10)  # 每 10 个单位一个刻度
#     plt.xticks(xticks, labels=[f"{x:.0f}" for x in xticks])  # 设置刻度标签
#
#     # 设置坐标轴标签
#     plt.xlabel('Time (Scaled to 150)')
#     plt.ylabel('Time Step (Frames)')
#     plt.title('Feature Map Visualization with Enhanced Contrast and Fine X-axis')
#
#     # 显示图像
#     plt.show()


import numpy as np
import matplotlib.pyplot as plt
import torch


# def paint_feat(feat):
#     """
#     可视化特征图，并在 150 秒的时间维度上表达特征信息。
#     输入特征形状可以是以下之一：
#     - (1, 75, 256)：去掉第一维，得到 (75, 256)。
#     - (75, 1, 256)：去掉第二维，得到 (75, 256)。
#     - (75, 256)：直接使用。
#     """
#     # 将输入张量转换为 numpy 数组，并确保计算图和梯度计算被停止
#     feat = feat.detach().cpu().numpy()
#
#     # 根据输入形状去掉多余的维度
#     if feat.ndim == 3:  # 如果输入是 3 维
#         if feat.shape[0] == 1:  # 形状为 (1, 75, 256)
#             feat = feat.squeeze(0)  # 去掉第一维，得到 (75, 256)
#         elif feat.shape[1] == 1:  # 形状为 (75, 1, 256)
#             feat = feat.squeeze(1)  # 去掉第二维，得到 (75, 256)
#         else:
#             raise ValueError("Input shape is not supported. Expected (1, 75, 256) or (75, 1, 256).")
#     elif feat.ndim == 2:  # 如果输入是 2 维
#         if feat.shape != (75, 256):
#             raise ValueError("Input shape must be (75, 256) if 2D.")
#     else:
#         raise ValueError("Input shape must be 2D or 3D.")
#
#     # 计算数据的均值和标准差
#     mean = feat.mean()
#     std = feat.std()
#
#     # 设置 vmin 和 vmax 为均值 ± 标准差，增强对比度
#     vmin = mean - std
#     vmax = mean + std
#
#     # 设置横坐标范围为 0 到 150 秒
#     xmin, xmax = 0, 150  # 横坐标范围（时间）
#     ymin, ymax = 0, feat.shape[1]  # 纵坐标范围（特征维度）
#
#     # 绘制热图
#     plt.figure(figsize=(15, 6))
#     plt.imshow(feat.T, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax,
#                extent=[xmin, xmax, ymin, ymax])  # 设置 extent 参数
#     plt.colorbar(label='Feature Value')  # 添加颜色条
#
#     # 设置横坐标刻度和标签（每 10 秒一个刻度）
#     xticks = np.arange(xmin, xmax + 1, 10)  # 每 10 秒一个刻度
#     plt.xticks(xticks, labels=[f"{x:.0f}" for x in xticks])  # 设置刻度标签
#
#     # 设置纵坐标刻度和标签（每 50 个特征维度一个刻度）
#     yticks = np.arange(ymin, ymax + 1, 50)  # 每 50 个特征维度一个刻度
#     plt.yticks(yticks, labels=[f"{y:.0f}" for y in yticks])  # 设置刻度标签
#
#     # 设置坐标轴标签
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Feature Dimension')
#     plt.title('Feature Map Visualization over 150 Seconds')
#
#     # 显示图像
#     plt.show()


import numpy as np
import matplotlib.pyplot as plt
import torch


import numpy as np
import matplotlib.pyplot as plt
import torch


# def paint_feat(feat):
#     """
#     可视化特征图，并在 150 秒的时间维度上表达特征信息。
#     输入特征形状可以是以下之一：
#     - (1, 75, 256)：去掉第一维，得到 (75, 256)。
#     - (75, 1, 256)：去掉第二维，得到 (75, 256)。
#     - (75, 256)：直接使用。
#     """
#     # 将输入张量转换为 numpy 数组，并确保计算图和梯度计算被停止
#     feat = feat.detach().cpu().numpy()
#
#     # 根据输入形状去掉多余的维度
#     if feat.ndim == 3:  # 如果输入是 3 维
#         if feat.shape[0] == 1:  # 形状为 (1, 75, 256)
#             feat = feat.squeeze(0)  # 去掉第一维，得到 (75, 256)
#         elif feat.shape[1] == 1:  # 形状为 (75, 1, 256)
#             feat = feat.squeeze(1)  # 去掉第二维，得到 (75, 256)
#         else:
#             raise ValueError("Input shape is not supported. Expected (1, 75, 256) or (75, 1, 256).")
#     elif feat.ndim == 2:  # 如果输入是 2 维
#         if feat.shape != (75, 256):
#             raise ValueError("Input shape must be (75, 256) if 2D.")
#     else:
#         raise ValueError("Input shape must be 2D or 3D.")
#
#     # 在 256 维上做均值，得到形状为 (75,)
#     feat_mean = feat.mean(axis=1)
#
#     # 将均值特征值扩展为二维矩阵，形状为 (75, 1)
#     feat_mean_2d = feat_mean[:, np.newaxis]
#
#     # 计算数据的均值和标准差
#     mean = feat_mean_2d.mean()
#     std = feat_mean_2d.std()
#
#     # 设置 vmin 和 vmax 为均值 ± 标准差，增强对比度
#     vmin = mean - std
#     vmax = mean + std
#
#     # 设置横坐标范围为 0 到 150 秒
#     xmin, xmax = 0, 150  # 横坐标范围（时间）
#     ymin, ymax = 0, 1  # 纵坐标范围（归一化）
#
#     # 绘制热图
#     plt.figure(figsize=(15, 2))  # 设置图像大小
#     plt.imshow(feat_mean_2d.T, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax,
#                extent=[xmin, xmax, ymin, ymax])  # 设置 extent 参数
#     plt.colorbar(label='Mean Feature Value')  # 添加颜色条
#
#     # 设置横坐标刻度和标签（每 10 秒一个刻度）
#     xticks = np.arange(xmin, xmax + 1, 10)  # 每 10 秒一个刻度
#     plt.xticks(xticks, labels=[f"{x:.0f}" for x in xticks])  # 设置刻度标签
#
#     # 隐藏纵坐标刻度（因为只有一行数据）
#     plt.yticks([])
#
#     # 设置坐标轴标签
#     plt.xlabel('Time (seconds)')
#     plt.title('Mean Feature Value over 150 Seconds')
#
#     # 显示图像
#     plt.show()

"""
取均值
"""
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from scipy.ndimage import gaussian_filter
#
# # 热图进行平滑处理
# def paint_feat(feat):
#     """
#     可视化特征图，并在 150 秒的时间维度上表达特征信息。
#     输入特征形状可以是以下之一：
#     - (1, 75, 256)：去掉第一维，得到 (75, 256)。
#     - (75, 1, 256)：去掉第二维，得到 (75, 256)。
#     - (75, 256)：直接使用。
#     """
#     # 将输入张量转换为 numpy 数组，并确保计算图和梯度计算被停止
#     feat = feat.detach().cpu().numpy()
#
#     # 根据输入形状去掉多余的维度
#     if feat.ndim == 3:  # 如果输入是 3 维
#         if feat.shape[0] == 1:  # 形状为 (1, 75, 256)
#             feat = feat.squeeze(0)  # 去掉第一维，得到 (75, 256)
#         elif feat.shape[1] == 1:  # 形状为 (75, 1, 256)
#             feat = feat.squeeze(1)  # 去掉第二维，得到 (75, 256)
#         else:
#             raise ValueError("Input shape is not supported. Expected (1, 75, 256) or (75, 1, 256).")
#     elif feat.ndim == 2:  # 如果输入是 2 维
#         if feat.shape != (75, 256):
#             raise ValueError("Input shape must be (75, 256) if 2D.")
#     else:
#         raise ValueError("Input shape must be 2D or 3D.")
#
#     # 在 256 维上做均值，得到形状为 (75,)
#     feat_mean = feat.mean(axis=1)
#
#     # 对均值特征值进行高斯平滑
#     feat_mean_smoothed = gaussian_filter(feat_mean, sigma=1)  # sigma 控制平滑程度
#
#     # 将平滑后的均值特征值扩展为二维矩阵，形状为 (75, 1)
#     feat_mean_2d = feat_mean_smoothed[:, np.newaxis]
#
#     # 计算数据的均值和标准差
#     mean = feat_mean_2d.mean()
#     std = feat_mean_2d.std()
#
#     # 设置 vmin 和 vmax 为均值 ± 标准差，增强对比度
#     vmin = mean - std
#     vmax = mean + std
#
#     # 设置横坐标范围为 0 到 150 秒
#     xmin, xmax = 0, 150  # 横坐标范围（时间）
#     ymin, ymax = 0, 1  # 纵坐标范围（归一化）
#
#     # 绘制热图
#     plt.figure(figsize=(15, 2))  # 设置图像大小
#     plt.imshow(feat_mean_2d.T, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax,
#                extent=[xmin, xmax, ymin, ymax], interpolation='bilinear')  # 使用双线性插值
#     plt.colorbar(label='Mean Feature Value')  # 添加颜色条
#
#     # 设置横坐标刻度和标签（每 10 秒一个刻度）
#     xticks = np.arange(xmin, xmax + 1, 10)  # 每 10 秒一个刻度
#     plt.xticks(xticks, labels=[f"{x:.0f}" for x in xticks])  # 设置刻度标签
#
#     # 隐藏纵坐标刻度（因为只有一行数据）
#     plt.yticks([])
#
#     # 设置坐标轴标签
#     plt.xlabel('Time (seconds)')
#     plt.title('Mean Feature Value over 150 Seconds (Smoothed)')
#
#     # 显示图像
#     plt.show()


"""
pca 未加高斯平滑
"""
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from sklearn.decomposition import PCA
#
# def paint_feat(feat):
#     """
#     使用PCA将特征图从[75, 256]降到[75, 1]并可视化。
#     输入特征形状可以是以下之一：
#     - (1, 75, 256)：去掉第一维，得到 (75, 256)。
#     - (75, 1, 256)：去掉第二维，得到 (75, 256)。
#     - (75, 256)：直接使用。
#     """
#     # 将输入张量转换为 numpy 数组，并确保计算图和梯度计算被停止
#     feat = feat.detach().cpu().numpy()
#
#     # 根据输入形状去掉多余的维度
#     if feat.ndim == 3:  # 如果输入是 3 维
#         if feat.shape[0] == 1:  # 形状为 (1, 75, 256)
#             feat = feat.squeeze(0)  # 去掉第一维，得到 (75, 256)
#         elif feat.shape[1] == 1:  # 形状为 (75, 1, 256)
#             feat = feat.squeeze(1)  # 去掉第二维，得到 (75, 256)
#         else:
#             raise ValueError("Input shape is not supported. Expected (1, 75, 256) or (75, 1, 256).")
#     elif feat.ndim == 2:  # 如果输入是 2 维
#         if feat.shape != (75, 256):
#             raise ValueError("Input shape must be (75, 256) if 2D.")
#     else:
#         raise ValueError("Input shape must be 2D or 3D.")
#
#     # 初始化PCA，选择要保留的主成分数目为1
#     pca = PCA(n_components=1)
#
#     # 执行PCA降维
#     feat_pca = pca.fit_transform(feat)
#
#     # 计算数据的均值和标准差
#     mean = feat_pca.mean()
#     std = feat_pca.std()
#
#     # 设置 vmin 和 vmax 为均值 ± 标准差，增强对比度
#     vmin = mean - std
#     vmax = mean + std
#
#     # 设置横坐标范围为 0 到 150 秒
#     xmin, xmax = 0, 150  # 横坐标范围（时间）
#     ymin, ymax = 0, 1  # 纵坐标范围（归一化）
#
#     # 将PCA结果扩展为二维矩阵，形状为 (75, 1)
#     feat_pca_2d = feat_pca.reshape(-1, 1)
#
#     # 绘制热图
#     plt.figure(figsize=(15, 2))  # 设置图像大小
#     plt.imshow(feat_pca_2d.T, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax,
#                extent=[xmin, xmax, ymin, ymax])
#     plt.colorbar(label='PCA Feature Value')  # 添加颜色条
#
#     # 设置横坐标刻度和标签（每 10 秒一个刻度）
#     xticks = np.arange(xmin, xmax + 1, 10)  # 每 10 秒一个刻度
#     plt.xticks(xticks, labels=[f"{x:.0f}" for x in xticks])  # 设置刻度标签
#
#     # 隐藏纵坐标刻度（因为只有一行数据）
#     plt.yticks([])
#
#     # 设置坐标轴标签
#     plt.xlabel('Time (seconds)')
#     plt.title('PCA Reduced Feature Value over 150 Seconds')
#
#     # 显示图像
#     plt.show()

# 示例调用
# 假设feat是一个形状为(1, 75, 256)的torch张量
# paint_feat(feat)


"""
高斯平滑+pca
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter

def paint_feat(feat,title=""):
    """
    使用PCA将特征图从[75, 256]降到[75, 1]并应用高斯平滑后可视化。
    输入特征形状可以是以下之一：
    - (1, 75, 256)：去掉第一维，得到 (75, 256)。
    - (75, 1, 256)：去掉第二维，得到 (75, 256)。
    - (75, 256)：直接使用。
    """
    # 将输入张量转换为 numpy 数组，并确保计算图和梯度计算被停止
    feat = feat.detach().cpu().numpy()

    # 根据输入形状去掉多余的维度
    if feat.ndim == 3:  # 如果输入是 3 维
        if feat.shape[0] == 1:  # 形状为 (1, 75, 256)
            feat = feat.squeeze(0)  # 去掉第一维，得到 (75, 256)
        elif feat.shape[1] == 1:  # 形状为 (75, 1, 256)
            feat = feat.squeeze(1)  # 去掉第二维，得到 (75, 256)

    # 初始化PCA，选择要保留的主成分数目为1
    pca = PCA(n_components=1)

    # 执行PCA降维
    feat_pca = pca.fit_transform(feat)

    # 对PCA降维后的结果进行高斯平滑
    feat_pca_smoothed = gaussian_filter(feat_pca.flatten(), sigma=1).reshape(-1, 1)  # 调整sigma值以控制平滑程度

    # 设置横坐标范围为 0 到 150 秒
    xmin, xmax = 0, 150  # 横坐标范围（时间）
    ymin, ymax = 0, 1  # 纵坐标范围（归一化）

    # 绘制热图
    plt.figure(figsize=(45, 6))  # 设置图像大小
    plt.imshow(feat_pca_smoothed.T, cmap='viridis', aspect='auto',
               extent=[xmin, xmax, ymin, ymax], interpolation='bilinear')  # 使用双线性插值
    plt.colorbar()  # 添加颜色条

    # 设置横坐标刻度和标签（每 10 秒一个刻度）
    xticks = np.arange(xmin, xmax + 1, 10)  # 每 10 秒一个刻度
    plt.xticks(xticks, labels=[f"{x:.0f}" for x in xticks],fontsize=30)  # 设置刻度标签

    # 隐藏纵坐标刻度（因为只有一行数据）
    plt.yticks([])

    # 设置坐标轴标签
    plt.xlabel('Time (seconds)',fontsize=30)
    plt.title(title,fontsize=30)

    # 显示图像
    plt.show()
