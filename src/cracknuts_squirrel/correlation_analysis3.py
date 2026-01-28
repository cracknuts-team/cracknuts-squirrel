import numpy as np
import zarr


def _correlation_analysis(traces: np.ndarray, hypothesis: np.ndarray) -> np.ndarray:
    """
    Pearson 相关性分析函数

    :param traces:
        功耗/电磁迹线矩阵，形状为 ``(num_traces, num_samples)``
    :type traces: numpy.ndarray

    :param hypothesis:
        泄漏模型假设值（如汉明重量 HW），形状为 ``(num_traces,)``
    :type hypothesis: numpy.ndarray

    :return:
        每个采样点对应的 Pearson 相关系数，形状为 ``(num_samples,)``
    :rtype: numpy.ndarray
    """

    # 转为 float，避免整数溢出
    traces = traces.astype(np.float64)
    hypothesis = hypothesis.astype(np.float64)

    # 去均值
    h_mean = hypothesis.mean()
    t_mean = traces.mean(axis=0)

    h_centered = hypothesis - h_mean
    t_centered = traces - t_mean

    # 分子：协方差
    numerator = np.dot(h_centered, t_centered) # 这里dot是计算内积计算了每一列的协方差

    # 分母：标准差乘积
    denominator = np.sqrt(
        np.sum(h_centered ** 2) * np.sum(t_centered ** 2, axis=0)
    )

    # 防止除零
    correlation = numerator / denominator
    return correlation


def correlation_analysis(traces: np.ndarray, original_data: np.ndarray) -> np.ndarray:
    ...