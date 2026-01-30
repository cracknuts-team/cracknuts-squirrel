# load base trace from a real trace file
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import truncnorm

from cracknuts_squirrel.correlation_analysis3 import correlation_analysis


def load_base_trace() -> np.ndarray:
    return np.load(
        "./base_trace.npy",
        allow_pickle=True
    )

def shift_trace_crop(trace, shift, max_shift):
    """
    trace: (T,)
    shift: int, 偏移量，可以为负
    max_shift: 最大可能偏移
    返回：裁剪后的 trace，长度 = T - 2*max_shift
    """
    T = len(trace)
    start = max_shift + shift
    end = T - max_shift + shift
    return trace[start:end]

def shift_trace_frac(trace, shift):
    """
    trace: (T,)
    shift: float，单位 = sample，正数表示向右偏移
    返回偏移后的 trace
    """
    T = len(trace)
    x = np.arange(T)
    f = interp1d(x, trace, kind='linear', fill_value=0, bounds_error=False)
    return f(x - shift)


def add_jitter(traces, max_shift, std=2.0):
    """
    traces: (N, T)
    max_shift: 最大绝对偏移，裁剪超出部分
    std: 偏移标准差
    """
    N, T = traces.shape
    out_len = T - 2*max_shift  # 保留有效长度
    out = np.zeros((N, out_len))

    # 正态分布偏移，float
    a = -max_shift / std
    b = max_shift / std
    shifts = truncnorm.rvs(a, b, loc=0.0, scale=std, size=N)
    for i in range(N):
        trace = traces[i]
        shift = shifts[i]
        # 裁剪前后各 max_shift
        start = max_shift
        end = T - max_shift
        out[i] = shift_trace_frac(trace, shift)[start:end]

    return out


def add_noise(traces, noise_std=1.0, max_sigma=4):
    """
    max_sigma: 最大噪声幅值 = max_sigma * noise_std
    """
    a = -max_sigma
    b =  max_sigma
    out = np.zeros(traces.shape)
    for i, trace in enumerate(traces):
        noise = truncnorm.rvs(
            a, b,
            loc=0.0,
            scale=noise_std,
            size=len(trace)
        )
        out[i] = trace + noise
    return out


def add_spikes(
    traces,
    spike_prob=0.3,          # 每条曲线出现毛刺的概率
    spike_rate=0.002,        # 曲线内部每个点成为毛刺的概率
    spike_amp=120.0,         # 毛刺幅值上限（正负）
    seed=None
):
    """
    给多条功耗曲线增加随机毛刺（spikes）

    Parameters
    ----------
    traces : ndarray, shape (N, T)
        N 条曲线，每条长度 T
    spike_prob : float
        一条曲线“是否包含毛刺”的概率
    spike_rate : float
        在已选中曲线中，单个采样点成为毛刺的概率
    spike_amp : float
        毛刺最大幅值（均匀分布在 [-spike_amp, spike_amp]）
    seed : int or None

    Returns
    -------
    traces_out : ndarray, shape (N, T)
    """
    rng = np.random.default_rng(seed)
    traces_out = traces.copy()

    N, T = traces.shape

    for i in range(N):
        # 1. 决定这一条曲线是否有毛刺
        if rng.random() > spike_prob:
            continue

        # 2. 决定毛刺位置（稀疏）
        spike_mask = rng.random(T) < spike_rate
        num_spikes = spike_mask.sum()
        if num_spikes == 0:
            continue

        # 3. 生成毛刺幅值
        spikes = rng.uniform(
            -spike_amp, spike_amp, size=num_spikes
        )

        traces_out[i, spike_mask] += spikes

    return traces_out


import numpy as np

def add_leakage(
    traces,
    leakage,
    poi,
    amplitude=5.0,
    width=2.0,
    jitter_std=0.0,
    seed=None,
):
    """
    在多条功耗曲线上叠加 leakage 信号

    Parameters
    ----------
    traces : ndarray, shape (N, T)
        原始曲线
    leakage : ndarray, shape (N,)
        泄漏变量（如 HW(key ⊕ plaintext)）
    poi : int
        理想的泄漏位置
    amplitude : float
        leakage 最大幅值
    width : float
        泄漏脉冲宽度（σ，单位：采样点）
    jitter_std : float
        POI 抖动标准差（采样点）
    seed : int or None

    Returns
    -------
    traces_out : ndarray, shape (N, T)
    """
    rng = np.random.default_rng(seed)
    traces_out = traces.copy().astype(np.float32)

    N, T = traces.shape
    x = np.arange(T)

    # 标准化 leakage（防止幅值随输入分布变化）
    leakage_norm = (leakage - leakage.mean()) / leakage.std()

    for i in range(N):
        # POI 抖动
        poi_i = poi + rng.normal(0, jitter_std)

        # 高斯型泄漏模板
        template = np.exp(-0.5 * ((x - poi_i) / width) ** 2)

        # 叠加 leakage
        traces_out[i] += amplitude * leakage_norm[i] * template

    return traces_out


def cpa(traces, leakage):
    traces_z = (traces - traces.mean(axis=0)) / traces.std(axis=0)
    leak_z = (leakage - leakage.mean()) / leakage.std()
    return np.abs(np.mean(traces_z * leak_z[:, None], axis=0))
######################################

def __main__():
    trace_count = 200
    sample_count = 20000 - 20 # 预留给 jitter 裁剪
    AES_SBOX = np.array([99,124,119,123,242,107,111,197,48,1,103,43,254,215,171,118,
                    202,130,201,125,250,89,71,240,173,212,162,175,156,164,114,192,
                    183,253,147,38,54,63,247,204,52,165,229,241,113,216,49,21,
                    4,199,35,195,24,150,5,154,7,18,128,226,235,39,178,117,
                    9,131,44,26,27,110,90,160,82,59,214,179,41,227,47,132,
                    83,209,0,237,32,252,177,91,106,203,190,57,74,76,88,207,
                    208,239,170,251,67,77,51,133,69,249,2,127,80,60,159,168,
                    81,163,64,143,146,157,56,245,188,182,218,33,16,255,243,210,
                    205,12,19,236,95,151,68,23,196,167,126,61,100,93,25,115,
                    96,129,79,220,34,42,144,136,70,238,184,20,222,94,11,219,
                    224,50,58,10,73,6,36,92,194,211,172,98,145,149,228,121,
                    231,200,55,109,141,213,78,169,108,86,244,234,101,122,174,8,
                    186,120,37,46,28,166,180,198,232,221,116,31,75,189,139,138,
                    112,62,181,102,72,3,246,14,97,53,87,185,134,193,29,158,
                    225,248,152,17,105,217,142,148,155,30,135,233,206,85,40,223,
                    140,161,137,13,191,230,66,104,65,153,45,15,176,84,187,22], dtype=np.uint8)

    def hw(x):
        return np.unpackbits(x[:, None], axis=1).sum(axis=1)

    plaintexts = np.random.randint(0, 256, size=trace_count, dtype=np.uint8)
    true_key = 0x2B

    intermediate = AES_SBOX[plaintexts ^ true_key]
    leakage = hw(intermediate)  # shape (N,)

    base_trace = load_base_trace()
    traces = np.tile(base_trace, (trace_count, 1))
    traces = add_leakage(
        traces,
        leakage,
        poi=4000,
        amplitude=16.0,
        width=0.5,
        jitter_std=1.5
    )
    # traces = add_jitter(traces, max_shift=1, std=0.1)  # 调小或者注释，可以模拟对齐后的曲线
    traces = add_noise(traces, max_sigma=50, noise_std=5.0) # 调小或者注释可以模拟删除噪声曲线的情况
    # traces = add_spikes(traces, seed=42) # 可以注释，模拟无毛刺情况
    print(traces.shape)
    plt.figure(figsize = (18,6))

    for i, trace in enumerate(traces):
        plt.plot(trace[:200], alpha=0.5, linewidth = 1, label = i)
    plt.legend()
    plt.show()

    corr = cpa(traces, leakage)
    poi_est = np.argmax(corr)
    print("估计 POI:", poi_est)
    print("最大相关性:", corr[poi_est])

    # print(leakage.shape)
    # corr = correlation_analysis(traces, leakage.reshape(200, 1))
    # for i, t in enumerate(corr):
    #     plt.plot(t, alpha=0.5, linewidth = 1, label = i)
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    __main__()