import typing

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import truncnorm


def load_base_trace() -> np.ndarray:
    """
    从磁盘加载一条基础功耗曲线。

    :return: 一维基础功耗曲线
    :rtype: numpy.ndarray，shape = (num_samples,)
    """
    return np.load(
        "./base_trace.npy",
        allow_pickle=True
    )


def shift_trace_frac(trace, shift):
    """
    对功耗曲线施加亚采样级（小数）时间偏移。

    使用线性插值实现分数偏移，超出原始范围的
    采样点以 0 填充。

    :param trace: 输入功耗曲线
    :type trace: numpy.ndarray，shape = (num_samples,)
    :param shift: 偏移量（单位：采样点），可为小数，
                  正值表示向右偏移
    :type shift: float
    :return: 偏移后的功耗曲线
    :rtype: numpy.ndarray，shape = (num_samples,)
    """
    num_samples = len(trace)
    time_idx = np.arange(num_samples)

    interp_func = interp1d(
        time_idx,
        trace,
        kind="linear",
        fill_value=0,
        bounds_error=False,
    )

    return interp_func(time_idx - shift)


def add_jitter(traces, max_shift=4, std=2.0):
    """
    为多条功耗曲线添加随机时间抖动（jitter）。

    抖动服从截断正态分布，采用分数偏移实现，
    最终通过裁剪保证所有曲线长度一致。

    :param traces: 输入功耗曲线集合
    :type traces: numpy.ndarray，shape = (num_traces, num_samples)
    :param max_shift: 最大绝对偏移量（采样点）
    :type max_shift: int
    :param std: 抖动的标准差（采样点）
    :type std: float
    :return: 添加抖动后的功耗曲线
    :rtype: numpy.ndarray，
            shape = (num_traces, num_samples - 2 * max_shift)
    """
    num_traces, num_samples = traces.shape
    out_len = num_samples - 2 * max_shift

    print(f"==== {num_samples} = {num_traces} - {out_len}")
    traces_out = np.zeros((num_traces, out_len))

    a = -max_shift / std
    b = max_shift / std
    shifts = truncnorm.rvs(a, b, loc=0.0, scale=std, size=num_traces)

    for trace_idx in range(num_traces):
        shifted = shift_trace_frac(traces[trace_idx], shifts[trace_idx])
        traces_out[trace_idx] = shifted[max_shift: num_samples - max_shift]

    return traces_out


def add_noise(traces, noise_std=1.0, max_sigma=4):
    """
    为功耗曲线添加受限幅值的白噪声。

    噪声服从截断正态分布，用于模拟测量噪声、
    ADC 噪声等随机干扰。

    :param traces: 输入功耗曲线集合
    :type traces: numpy.ndarray，shape = (num_traces, num_samples)
    :param noise_std: 噪声标准差
    :type noise_std: float
    :param max_sigma: 最大噪声幅值倍数
                      （最大幅值 = max_sigma * noise_std）
    :type max_sigma: float
    :return: 添加噪声后的功耗曲线
    :rtype: numpy.ndarray，shape = (num_traces, num_samples)
    """
    a = -max_sigma
    b = max_sigma

    traces_out = np.zeros_like(traces)

    for trace_idx, trace in enumerate(traces):
        noise = truncnorm.rvs(
            a,
            b,
            loc=0.0,
            scale=noise_std,
            size=len(trace),
        )
        traces_out[trace_idx] = trace + noise

    return traces_out


def add_spikes(
        traces,
        spike_prob=0.3,
        spike_rate=0.002,
        spike_amp=120.0,
        seed=None,
):
    """
    在功耗曲线中注入稀疏随机毛刺（spikes）。

    用于模拟电磁干扰、采集异常、系统调度抖动等
    非理想采集现象。

    :param traces: 输入功耗曲线集合
    :type traces: numpy.ndarray，shape = (num_traces, num_samples)
    :param spike_prob: 单条曲线包含毛刺的概率
    :type spike_prob: float
    :param spike_rate: 在已选中曲线中，
                       单个采样点成为毛刺的概率
    :type spike_rate: float
    :param spike_amp: 毛刺最大幅值（正负）
    :type spike_amp: float
    :param seed: 随机种子
    :type seed: int or None
    :return: 注入毛刺后的功耗曲线
    :rtype: numpy.ndarray，shape = (num_traces, num_samples)
    """
    rng = np.random.default_rng(seed)
    traces_out = traces.copy()

    num_traces, num_samples = traces.shape

    for trace_idx in range(num_traces):
        if rng.random() > spike_prob:
            continue

        spike_mask = rng.random(num_samples) < spike_rate
        num_spikes = spike_mask.sum()
        if num_spikes == 0:
            continue

        spikes = rng.uniform(
            -spike_amp,
            spike_amp,
            size=num_spikes,
        )

        traces_out[trace_idx, spike_mask] += spikes

    return traces_out


# def add_leakage(
#         traces,
#         leakage,
#         poi,
#         amplitude=5.0,
#         width=2.0,
#         jitter_std=0.0,
#         seed=None,
# ):
#     """
#     在功耗曲线上叠加数据相关的泄漏信号。
#
#     泄漏信号建模为以 POI 为中心的高斯脉冲，
#     幅值与泄漏变量（如 HW(key ⊕ plaintext)）成正比。
#
#     :param traces: 原始功耗曲线
#     :type traces: numpy.ndarray，shape = (num_traces, num_samples)
#     :param leakage: 泄漏变量
#     :type leakage: numpy.ndarray，shape = (num_traces,)
#     :param poi: 理想泄漏点（采样点索引）
#     :type poi: int
#     :param amplitude: 泄漏信号最大幅值
#     :type amplitude: float
#     :param width: 泄漏脉冲宽度（高斯 σ，单位：采样点）
#     :type width: float
#     :param jitter_std: POI 抖动标准差（采样点）
#     :type jitter_std: float
#     :param seed: 随机种子
#     :type seed: int or None
#     :return: 叠加泄漏后的功耗曲线
#     :rtype: numpy.ndarray，shape = (num_traces, num_samples)
#     """
#     rng = np.random.default_rng(seed)
#     traces_out = traces.copy().astype(np.float32)
#
#     num_traces, num_samples = traces.shape
#     time_idx = np.arange(num_samples)
#
#     leakage_norm = (leakage - leakage.mean()) / leakage.std()
#
#     for trace_idx in range(num_traces):
#         poi_jittered = poi + rng.normal(0, jitter_std)
#
#         template = np.exp(
#             -0.5 * ((time_idx - poi_jittered) / width) ** 2
#         )
#
#         traces_out[trace_idx] += (
#                 amplitude * leakage_norm[trace_idx] * template
#         )
#
#     return traces_out


def add_leakage(
        traces,
        leakage,
        poi_dict,
        amplitude=5.0,
        width=2.0,
        jitter_std=0.0,
        seed=None,
):
    """
    在功耗曲线上叠加数据相关的泄漏信号（多 POI）。

    每个泄漏源建模为以 POI 为中心的高斯脉冲，
    幅值与对应泄漏变量成正比，多个泄漏源线性叠加。

    :param traces: 原始功耗曲线
    :type traces: numpy.ndarray，shape = (num_traces, num_samples)
    :param leakage: 泄漏变量
    :type leakage: numpy.ndarray，shape = (num_traces, num_sources)
    :param poi_dict: 泄漏源索引到 POI 的映射
    :type poi_dict: dict[int, int]
    :param amplitude: 泄漏信号最大幅值
    :type amplitude: float
    :param width: 泄漏脉冲宽度（高斯 σ，单位：采样点）
    :type width: float
    :param jitter_std: POI 抖动标准差（采样点）
    :type jitter_std: float
    :param seed: 随机种子
    :type seed: int or None
    :return: 叠加泄漏后的功耗曲线
    :rtype: numpy.ndarray，shape = (num_traces, num_samples)
    """
    rng = np.random.default_rng(seed)
    traces_out = traces.copy().astype(np.float32)

    num_traces, num_samples = traces.shape
    time_idx = np.arange(num_samples)

    leakage = np.asarray(leakage, dtype=np.float32)

    if leakage.ndim != 2:
        raise ValueError("leakage 必须是二维数组 (num_traces, num_sources)")

    # 每个泄漏源独立标准化
    leakage_norm = (
            (leakage - leakage.mean(axis=0))
            / leakage.std(axis=0)
    )

    for src_idx, poi in poi_dict.items():
        for trace_idx in range(num_traces):
            poi_jittered = poi + rng.normal(0, jitter_std)

            template = np.exp(
                -0.5 * ((time_idx - poi_jittered) / width) ** 2
            )

            traces_out[trace_idx] += (
                    amplitude
                    * leakage_norm[trace_idx, src_idx]
                    * template
            )

    return traces_out


AES_SBOX = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
])

HW_TABLE = np.array([
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
], np.uint8)


def hamming_weight(x: int) -> int:
    return bin(x).count("1")


def generate_base_trace(
        num_samples: int,
        enc_start: int,
        enc_end: int,
        cycle_len: int = 40,
        scale: float = 50.0,
        noise_std: float = 0.5,
        seed: int | None = None,
) -> np.ndarray:
    """
    Generate a realistic base power trace with strong periodic structure
    inside an encryption window.

    :param num_samples: total trace length
    :param enc_start: encryption start index
    :param enc_end: encryption end index
    :param cycle_len: samples per execution cycle
    :param scale: overall amplitude
    :param noise_std: weak measurement noise
    :param seed: random seed
    :return: base_trace, shape (num_samples,)
    """
    rng = np.random.default_rng(seed)

    # ---------- 1. build one execution-cycle power template ----------
    t = np.linspace(0, 1, cycle_len, endpoint=False)

    template = (
            0.7 * np.exp(-((t - 0.18) / 0.06) ** 2) +  # load
            1.2 * np.exp(-((t - 0.50) / 0.10) ** 2) +  # compute
            0.5 * np.exp(-((t - 0.78) / 0.05) ** 2)  # store
    )

    # non-linear compression → no sharp edges
    template = np.tanh(template)

    # normalize template
    template -= template.mean()
    template /= template.std()

    # ---------- 2. repeat template in encryption window ----------
    base = np.zeros(num_samples, dtype=np.float32)

    enc_len = enc_end - enc_start
    num_cycles = enc_len // cycle_len

    pos = enc_start
    for _ in range(num_cycles):
        # small cycle-to-cycle variation (very weak)
        amp = 1.0 + rng.normal(0, 0.05)
        base[pos:pos + cycle_len] += template * amp
        pos += cycle_len

    # tail if window not perfectly divisible
    if pos < enc_end:
        remain = enc_end - pos
        base[pos:enc_end] += template[:remain]

    # ---------- 3. add weak bounded noise everywhere ----------
    noise = rng.normal(0, noise_std, size=num_samples)
    base += noise

    # ---------- 4. final scaling ----------
    base -= base.mean()
    base /= np.max(np.abs(base))
    base *= scale

    return base.astype(np.float32)


def gen_aes_golden_trace(
        num_traces: int,
        key: np.ndarray,
        base_trace: np.ndarray | None = None,
        num_samples: int = 5000,
        leakage_model="HW",
        prev_state="zero",
        seed=None,
        # for leakage parameters
        do_leakage: bool = True,
        leakage_poi_dict: dict[int, int] | None = None,
        leakage_amplitude=5.0,
        leakage_width=2.0,
        leakage_jitter_std=1.0,
        # for noise parameters
        do_noise: bool = True,
        noise_max_sigma: float = 50.0,
        noise_std: float = 5.0,
        # for spike parameters
        do_spikes: bool = True,
        spike_prob=0.3,
        spike_rate=0.002,
        spike_amp=120.0,
        # for jitter parameters
        do_jitter: bool = True,
        jitter_max_shift=5,
        jitter_std=2.0,
        save_to_zarr=None
):
    """
    生成 AES Golden Trace（用于侧信道分析的功耗曲线仿真）。

    本函数用于生成**具有真实统计特性和可控物理含义的 AES 功耗曲线**，
    适用于 CPA / DPA / Template Attack 等侧信道算法的调试、评估与教学。

    单条功耗曲线按如下顺序构成：

    1. 基础功耗曲线（base trace，表示与数据无关的整体功耗形态）
    2. 数据相关泄露（HW / ID / HD，高斯时间扩散）
    3. 加性白噪声（模拟测量与环境噪声）
    4. 随机功耗毛刺（spikes，模拟异常干扰）
    5. 时间轴抖动（jitter，模拟触发与时序不稳定）

    每一类效应均可通过 ``do_*`` 参数独立启用或关闭，
    便于构造从“理想模型”到“接近真实设备”的不同实验场景。

    --------------------
    基本参数
    --------------------

    :param num_traces:
        生成的功耗曲线数量。

        数量越大，统计攻击（如 CPA）的相关性估计越稳定。
    :type num_traces: int

    :param key:
        AES 密钥，长度为 16 字节。
    :type key: numpy.ndarray, shape=(16,)

    :param base_trace:
        基础功耗曲线模板。

        - 若提供：所有 trace 共享该整体功耗形态
        - 若为 ``None``：自动生成一条具有周期结构的基础曲线

        基础曲线决定**宏观功耗轮廓**，但不包含数据相关信息。
    :type base_trace: numpy.ndarray or None

    :param num_samples:
        单条功耗曲线的采样点数量。

        仅在 ``base_trace is None`` 时生效。
    :type num_samples: int

    :param seed:
        随机种子，用于控制：
        - HD 模型中的随机前置状态
        - 泄露位置抖动（若启用）
        - 各类随机过程的可复现性
    :type seed: int or None

    --------------------
    泄露模型参数
    --------------------

    :param leakage_model:
        泄露模型类型：

        - ``"HW"``：汉明重量模型
          适用于寄存器/总线静态泄露，是最常见的假设模型。
        - ``"ID"``：中间值直接泄露
          理想化模型，泄露与中间值大小直接相关。
        - ``"HD"``：汉明距离模型
          模拟寄存器翻转功耗，通常更贴近真实 CMOS 行为。

    :type leakage_model: str

    :param prev_state:
        仅在 ``leakage_model="HD"`` 时有效，用于定义前置状态：

        - ``"zero"``：前态全 0（上电或寄存器清零假设）
        - ``"plaintext"``：前态为明文字节
        - ``"random"``：前态为随机状态

        不同前置状态会显著影响 HD 泄露的统计分布。
    :type prev_state: str

    --------------------
    泄露信号形态参数
    --------------------

    :param do_leakage:
        是否叠加数据相关泄露信号。
    :type do_leakage: bool

    :param leakage_poi_dict:
        泄露源到采样点的映射 ``{byte_index: poi_sample}``。

        - key：AES 字节索引 (0–15)
        - value：该字节泄露发生的采样点位置

        若为 ``None``，默认在曲线中间三分之一区域内均匀分布。
    :type leakage_poi_dict: dict or None

    :param leakage_amplitude:
        泄露幅值缩放因子。

        值越大，数据相关成分在功耗中占比越高，
        CPA 相关性峰值越明显。
    :type leakage_amplitude: float

    :param leakage_width:
        泄露的时间扩散宽度（高斯 σ，单位：采样点）。

        - 小值：泄露集中、尖锐（理想触发）
        - 大值：泄露被时间扩散、平滑（更贴近真实设备）
    :type leakage_width: float

    :param leakage_jitter_std:
        泄露中心 POI 的随机抖动标准差。

        用于模拟流水线抖动、触发不稳定等现象。
    :type leakage_jitter_std: float

    --------------------
    噪声参数
    --------------------

    :param do_noise:
        是否叠加加性白噪声。
    :type do_noise: bool

    :param noise_std:
        高斯白噪声标准差。

        直接控制信噪比（SNR），值越大攻击难度越高。
    :type noise_std: float

    :param noise_max_sigma:
        噪声截断上限（倍数），用于限制极端噪声幅值。
    :type noise_max_sigma: float

    --------------------
    毛刺参数
    --------------------

    :param do_spikes:
        是否注入随机功耗毛刺。
    :type do_spikes: bool

    :param spike_prob:
        单条曲线出现毛刺的概率。
    :type spike_prob: float

    :param spike_rate:
        单个采样点成为毛刺的概率。
    :type spike_rate: float

    :param spike_amp:
        毛刺幅值（正负），模拟 EMI 或采集异常。
    :type spike_amp: float

    --------------------
    时间抖动参数
    --------------------

    :param do_jitter:
        是否对整条曲线施加时间轴抖动。
    :type do_jitter: bool

    :param jitter_max_shift:
        最大时间偏移（采样点）。
    :type jitter_max_shift: int

    :param jitter_std:
        时间偏移标准差。

        会显著降低未对齐 CPA 的攻击效果。
    :type jitter_std: float
    :param save_to_zarr:
        若设置则保存到对应的路径，格式为 zarr，名称为: 日期+aes_golden_trace.zarr。
    :type save_to_zarr: str or None

    --------------------
    返回值
    --------------------

    :return:
        ``(traces, leakage)``

        - ``traces``：生成的功耗曲线
          shape = ``(num_traces, num_samples)``
        - ``plaintext``：对应的明文
          shape = ``(num_traces, 16)``
        - ``ciphertexts``：对应的密文
          shape = ``(num_traces, 16)``
        - ``leakage``：真实泄露变量
          shape = ``(num_traces, 16)``
    :rtype: tuple[numpy.ndarray, numpy.ndarray]

    --------------------
    使用示例
    --------------------

    最小示例（HW 模型，默认参数）::

        key = np.random.randint(0, 256, size=16, dtype=np.uint8)

        traces, leakage = gen_aes_golden_trace(
            num_traces=2000,
            key=key
        )

    使用 HD 模型并关闭非理想效应::

        traces, leakage = gen_aes_golden_trace(
            num_traces=1000,
            key=key,
            leakage_model="HD",
            prev_state="plaintext",
            do_noise=False,
            do_spikes=False,
            do_jitter=False
        )
    """

    if base_trace is None:
        if num_samples is None:
            raise ValueError("当 base_trace 为 None 时，必须指定 num_samples")
        base_trace = generate_base_trace(num_samples, num_samples // 3, 2 * num_samples // 3, scale=300)

    plaintext = np.random.randint(0, 256, size=(num_traces, 16), dtype=np.uint8)
    traces = np.tile(base_trace, (num_traces, 1)).astype(np.float32)
    inter = AES_SBOX[plaintext ^ key]

    rng = np.random.default_rng(seed)

    if leakage_model == "HW":
        leakage = HW_TABLE[inter]
    elif leakage_model == "ID":
        leakage = inter.astype(np.float32)
    elif leakage_model == "HD":
        if prev_state == "zero":
            prev = np.zeros_like(inter, dtype=np.uint8)
        elif prev_state == "plaintext":
            prev = plaintext
        elif prev_state == "random":
            prev = rng.integers(0, 256, size=inter.shape, dtype=np.uint8)
        else:
            raise ValueError("未知前置状态模型")
        leakage = HW_TABLE[prev ^ inter]
    else:
        raise ValueError(f"未知泄露模型: {leakage_model}")

    if leakage_poi_dict is None:
        num_samples = base_trace.shape[0]
        start = num_samples // 3
        end = 2 * num_samples // 3
        poi_positions = np.linspace(start, end, 16, endpoint=False, dtype=int)
        leakage_poi_dict = {byte_idx: poi_positions[byte_idx] for byte_idx in range(16)}

    if do_leakage:
        traces = add_leakage(
            traces=traces,
            leakage=leakage,
            poi_dict=leakage_poi_dict,
            amplitude=leakage_amplitude,
            width=leakage_width,
            jitter_std=leakage_jitter_std,
            seed=seed,
        )
    if do_noise:
        traces = add_noise(
            traces=traces,
            max_sigma=noise_max_sigma,
            noise_std=noise_std,
        )
    if do_spikes:
        traces = add_spikes(
            traces=traces,
            spike_prob=spike_prob,
            spike_rate=spike_rate,
            spike_amp=spike_amp,
            seed=None,
        )
    if do_jitter:
        traces = add_jitter(
            traces=traces,
            max_shift=jitter_max_shift,
            std=jitter_std
        )

    ciphertext = np.zeros((num_traces, 16), dtype=np.uint8)
    if save_to_zarr is not None:
        import zarr
        import datetime
        from Crypto.Cipher import AES

        # 真实计算密文
        key_bytes = key.tobytes()
        cipher = AES.new(key_bytes, AES.MODE_ECB)
        for i in range(plaintext.shape[0]):
            ciphertext[i, :] = np.frombuffer(cipher.encrypt(plaintext[i].tobytes()), dtype=np.uint8)

        date_str = datetime.datetime.now().strftime("%Y%m%d")
        zarr_path = f"{save_to_zarr}/{date_str}_aes_golden_trace.zarr"
        zarr_root = zarr.open(zarr_path, mode='w')
        group_root = zarr_root.create_group('/0/0')
        group_root.create_dataset('traces', data=traces)
        group_root.create_dataset('plaintext', data=plaintext)
        group_root.create_dataset('ciphertexts', data=ciphertext)
        print(f"Saved traces and leakage to {zarr_path}")

    return traces, plaintext, ciphertext, leakage

