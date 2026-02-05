import numpy as np


def correlation_analysis(
    traces: np.ndarray,
    hypothesis: np.ndarray
) -> np.ndarray:
    """
    通用 Pearson 相关性分析函数（矩阵形式）。

    关键维度说明：

    - **N**：trace 数量（样本数），用于统计均值、协方差等计算
      （计算完成后不出现在输出中）
    - **T**：每条 trace 的采样点数（时间轴长度）
    - **H**：假设数量（例如 AES CPA 时 H=16*256 或 16*key_guess 数）

    本函数计算功耗/电磁迹线与泄漏模型假设之间的 Pearson 相关系数，
    适用于 CPA、DPA 以及更一般的侧信道统计分析场景。

    输出相关性矩阵的形状为 ``(H, T)``，
    每个元素表示第 h 个假设与第 t 个采样点之间的相关系数。

    对于 AES CPA 等高维 hypothesis（多字节、多猜测、多模型），
    应在调用前将其 reshape 为二维矩阵 ``(N, H)``，
    第 0 维对应 trace，确保每条 trace 的假设值只占一行。

    :param traces: 功耗/电磁迹线矩阵，形状为 `(num_traces, num_samples)`。
    :type traces: np.ndarray
    :param hypothesis: 泄漏模型假设矩阵，形状为 `(num_traces, num_hypotheses)`。
                       多维 hypothesis 在调用前应 flatten 为 `(num_traces, H)`。
    :type hypothesis: np.ndarray
    :return: Pearson 相关系数矩阵，形状为 `(num_hypotheses, num_samples)`。
    :rtype: np.ndarray
    """
    # 输入检查
    if traces.ndim != 2:
        raise ValueError("traces 必须是二维数组 (num_traces, num_samples)")

    if hypothesis.ndim != 2:
        raise ValueError("hypothesis 必须是二维数组 (num_traces, num_hypotheses)")

    if traces.shape[0] != hypothesis.shape[0]:
        raise ValueError("traces 与 hypothesis 的 trace 数量不一致")

    # 转为 float，避免整数溢出
    traces = traces.astype(np.float64)
    hypothesis = hypothesis.astype(np.float64)

    # 去均值（中心化）
    traces_centered = traces - np.mean(traces, axis=0, keepdims=True)
    hypothesis_centered = hypothesis - np.mean(hypothesis, axis=0, keepdims=True)

    # 分子：协方差（矩阵乘法）
    # (H, N) @ (N, T) -> (H, T)
    numerator = hypothesis_centered.T @ traces_centered

    # 分母：标准差乘积
    hypothesis_norm = np.sqrt(
        np.sum(hypothesis_centered ** 2, axis=0)
    )  # (H,)

    traces_norm = np.sqrt(
        np.sum(traces_centered ** 2, axis=0)
    )  # (T,)

    denominator = np.outer(hypothesis_norm, traces_norm)  # (H, T)

    # 防止除零
    denominator[denominator == 0] = np.nan

    correlation = numerator / denominator

    return correlation
