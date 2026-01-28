import numpy as np


def _gaussian_window(length: int, center: float, sigma: float) -> np.ndarray:
    t = np.arange(length)
    return np.exp(-0.5 * ((t - center) / sigma) ** 2)


def generate_traces(
    leakage: np.ndarray,
    *,
    trace_length: int,
    poi: np.ndarray | list,
    amplitude: float = 1.0,
    leakage_width: float = 8.0,
    white_noise_std: float = 0.5,
    jitter_std: float = 0.0,
    drift_std: float = 0.0,
    time_warp_strength: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate synthetic side-channel traces from abstract leakage values.

    This function implements a *generic, algorithm-agnostic side-channel
    trace generator*. It converts abstract leakage values (e.g. Hamming
    Weight, Hamming Distance, bit-level leakage, or any user-defined model)
    into time-domain power or EM traces that resemble real acquisition data.

    The generator is intentionally separated from any cryptographic
    algorithm assumptions. All algorithm-specific semantics (AES, DES,
    SM4, etc.) must be expressed *only* through the provided ``leakage``
    input. This design allows the same trace generator to be reused for
    different algorithms, leakage granularities, and attack scenarios.

    --------------------
    Leakage model
    --------------------
    The input ``leakage`` is interpreted as a matrix of abstract leakage
    values with shape ``(num_traces, num_units)``:

    - ``num_traces`` corresponds to the number of acquisitions.
    - ``num_units`` corresponds to the number of independent leakage units.

    A *leakage unit* is an abstract concept representing any independently
    leaking intermediate value, such as:

    - One byte of an AES intermediate state
    - One bit of a register (bit-level leakage)
    - A multi-byte word or bus value
    - Any user-defined grouping used for correlation analysis

    This function does not assume any specific meaning or size of a leakage
    unit. It only assumes that each unit produces a localized contribution
    on the time axis.

    --------------------
    Time-domain modeling
    --------------------
    Each leakage unit contributes to the trace through a localized temporal
    window centered around a Point Of Interest (POI). The POI specifies the
    approximate time at which the corresponding intermediate value leaks.

    The temporal contribution of one leakage unit is modeled as:

        contribution(t) = amplitude × leakage_value × window(t - poi)

    where ``window`` is a smooth windowing function (Gaussian by default).
    This reflects the physical reality that leakage is not confined to a
    single sample but spreads over a short time interval.

    When multiple leakage units are present, their contributions are
    linearly superposed. Leakage windows may overlap in time, allowing the
    simulation of parallel or pipelined implementations.

    --------------------
    Leakage width
    --------------------
    The parameter ``leakage_width`` controls the temporal spread of the
    leakage window. Larger values produce broader leakage regions, while
    smaller values approximate near-impulsive leakage.

    This parameter is critical for evaluating:
    - the robustness of POI selection
    - the effectiveness of alignment algorithms
    - the impact of temporal filtering

    --------------------
    Noise and non-ideal effects
    --------------------
    To better approximate real measurement conditions, the generator
    supports several non-ideal effects:

    - Additive white Gaussian noise to simulate measurement and environmental
      noise.
    - Random POI jitter, modeling small timing variations between traces.
    - Optional global drift or time warping effects (if enabled), which
      emulate trigger instability or clock variations.

    These effects can be independently tuned or disabled, allowing the
    generator to smoothly transition from an idealized model to a more
    realistic and challenging scenario.

    --------------------
    Reproducibility
    --------------------
    All random components (noise, jitter, drift) are generated using a
    pseudo-random number generator. When the ``seed`` parameter is provided,
    the entire trace generation process becomes deterministic and fully
    reproducible.

    This is essential for:
    - algorithm debugging
    - regression testing
    - fair comparison of analysis techniques
    - scientific reproducibility

    --------------------
    Intended usage
    --------------------
    This function is primarily intended as a *test and validation tool*
    for side-channel analysis pipelines. Typical use cases include:

    - Validating CPA/DPA implementations
    - Testing alignment and preprocessing algorithms
    - Evaluating the impact of noise and leakage dispersion
    - Generating controlled datasets for teaching and experimentation

    It is **not** intended to faithfully reproduce the electrical behavior
    of a specific device or technology node. Instead, it focuses on modeling
    the statistical properties that are most relevant for side-channel
    attacks.

    :param leakage:
        Abstract leakage values of shape ``(num_traces, num_units)``.
        Each column corresponds to one independent leakage unit.
    :type leakage: numpy.ndarray

    :param trace_length:
        Number of samples per generated trace.
    :type trace_length: int

    :param poi:
        Center sample index for each leakage unit. Must have length
        ``num_units``.
    :type poi: array-like

    :param leakage_width:
        Standard deviation of the temporal leakage window (in samples).
    :type leakage_width: float

    :param amplitude:
        Global scaling factor applied to all leakage contributions.
    :type amplitude: float

    :param white_noise_std:
        Standard deviation of additive white Gaussian noise.
    :type white_noise_std: float

    :param jitter_std:
        Standard deviation of per-trace, per-unit POI jitter (in samples).
    :type jitter_std: float

    :param drift_std:
        Standard deviation of global time drift applied per trace (in samples).
        This models trigger instability or acquisition start-time variations.
    :type drift_std: float

    :param time_warp_strength:
        Strength of non-linear time warping applied to each trace.
        This simulates clock instability or sampling rate variation.
        Set to 0 to disable time warping.
    :type time_warp_strength: float

    :param seed:
        Seed for the pseudo-random number generator. If ``None``, the output
        is non-deterministic.
    :type seed: int or None

    :return:
        Generated side-channel traces with shape
        ``(num_traces, trace_length)``.
    :rtype: numpy.ndarray
    """

    leakage = np.asarray(leakage, dtype=np.float64)
    poi = np.asarray(poi, dtype=np.float64)

    num_traces, num_units = leakage.shape

    if poi.shape[0] != num_units:
        raise ValueError(
            f"len(poi)={poi.shape[0]} must equal num_units={num_units}"
        )

    rng = np.random.default_rng(seed)

    traces = np.zeros((num_traces, trace_length), dtype=np.float64)

    # 漂移：每条 trace 一个全局偏移
    global_drift = rng.normal(0.0, drift_std, size=num_traces)

    for i in range(num_traces):
        # 白噪声
        trace = rng.normal(0.0, white_noise_std, size=trace_length)

        for u in range(num_units):
            # POI 抖动
            center = poi[u] + rng.normal(0.0, jitter_std) + global_drift[i]

            # 时间扭曲（简单线性近似）
            if time_warp_strength > 0:
                center *= 1.0 + rng.normal(0.0, time_warp_strength)

            window = _gaussian_window(
                trace_length, center, leakage_width
            )

            trace += amplitude * leakage[i, u] * window

        traces[i] = trace

    return traces


def generate_traces2(
        leakage: np.ndarray,
        *,
        trace_length: int,
        poi: np.ndarray | list,
        amplitude: float = 1.0,
        leakage_width: float = 8.0,
        white_noise_std: float = 0.5,
        jitter_std: float = 0.0,
        drift_std: float = 0.0,
        time_warp_strength: float = 0.0,
        seed: int | None = None,
):
    """
    生成基于抽象泄漏值的侧信道功耗/电磁迹线。

    本函数是一个**通用、算法无关的侧信道迹线生成器**。它将抽象泄漏值
    （例如汉明重量、汉明距离、比特级泄漏或自定义泄漏模型）转换为时间域
    的功耗或 EM 迹线，模拟真实采集数据的统计特性。

    函数设计为**与具体加密算法无关**。所有算法特定的语义（AES、DES、SM4 等）
    仅通过 ``leakage`` 输入体现。这样的设计允许同一生成器适用于不同算法、
    不同泄漏粒度和不同攻击场景。

    --------------------
    泄漏模型
    --------------------
    输入 ``leakage`` 是一个形状为 ``(num_traces, num_units)`` 的矩阵：

    - ``num_traces`` 表示迹线数量
    - ``num_units`` 表示独立泄漏单元数量

    **泄漏单元（leakage unit）** 是一个抽象概念，表示物理上相对独立的泄漏源，例如：

    - AES 中一个字节的中间状态
    - 一个寄存器的比特（bit-level）
    - 多字节总线的一个字
    - 任何用户定义的用于相关性分析的分组

    函数不假设泄漏单元的语义或大小，仅假设每个单元在时间轴上产生局部泄漏贡献。

    --------------------
    时间域建模
    --------------------
    每个泄漏单元在时间轴上的贡献由一个**局部时间窗口**控制，中心位于指定
    的 POI（Point of Interest）：

        contribution(t) = amplitude × leakage_value × window(t - poi)

    其中 ``window`` 是平滑窗口函数（默认为高斯函数）。
    这一模型反映了真实器件中泄漏**不会局限于单个采样点，而是跨越一段时间**。

    多个泄漏单元的贡献线性叠加，窗口可能在时间上重叠，可用于模拟
    并行或流水线实现。

    --------------------
    泄漏宽度
    --------------------
    参数 ``leakage_width`` 控制每个泄漏窗口的时间扩散程度（标准差 σ）：

    - 值小（1-2）：接近单采样点泄漏
    - 值中（5-10）：单次操作跨多个采样点
    - 值大（20+）：操作跨越多个周期或逻辑级

    该参数对 POI 选择、对齐算法和滤波策略影响显著。

    --------------------
    噪声与非理想因素
    --------------------
    为更接近真实测量条件，生成器支持多种非理想效应：

    - **白噪声（white_noise_std）**：模拟测量噪声和环境干扰
    - **POI 抖动（jitter_std）**：每条迹线每个泄漏单元的微小时间偏移
    - **全局漂移（drift_std）**：每条迹线整体时间偏移，模拟触发或采集起点漂移
    - **时间扭曲（time_warp_strength）**：非线性时间轴形变，模拟时钟不稳定或采样频率变化

    这些效应可独立调节或关闭，使生成器可在理想模型和真实复杂情况间平滑过渡。

    --------------------
    可复现性
    --------------------
    所有随机成分（噪声、抖动、漂移、时间扭曲）均使用伪随机数生成器产生。
    当提供 ``seed`` 参数时，生成过程完全确定，可复现。

    可复现性适用于：

    - 算法调试
    - 回归测试
    - 不同方法的公平比较
    - 科学实验复现

    --------------------
    使用场景
    --------------------
    本函数主要用于**侧信道分析算法测试与验证**，典型用例包括：

    - CPA / DPA 攻击算法验证
    - 对齐算法（SAD / DTW 等）测试
    - 滤波和噪声处理效果评估
    - 教学或实验数据生成

    注意：本函数不用于精确重现特定芯片或技术节点的电路行为，
    而是聚焦于侧信道攻击中**相关性产生与破坏的统计特性**。

    --------------------
    参数说明
    --------------------
    :param leakage: 形状为 ``(num_traces, num_units)`` 的泄漏矩阵，每列对应一个泄漏单元
    :type leakage: numpy.ndarray

    :param trace_length: 每条生成迹线的采样点数量
    :type trace_length: int

    :param poi: 每个泄漏单元在时间轴上的中心采样点，长度为 ``num_units``
    :type poi: array-like

    :param leakage_width: 泄漏窗口的标准差 σ（以采样点为单位）
    :type leakage_width: float

    :param amplitude: 全局泄漏幅度缩放因子
    :type amplitude: float

    :param white_noise_std: 添加白噪声的标准差
    :type white_noise_std: float

    :param jitter_std: 每条迹线、每个泄漏单元 POI 抖动标准差
    :type jitter_std: float

    :param drift_std: 每条迹线的整体时间漂移标准差
    :type drift_std: float

    :param time_warp_strength: 非线性时间扭曲强度，0 表示关闭
    :type time_warp_strength: float

    :param seed: 随机数生成器种子，用于保证结果可复现
    :type seed: int or None

    :return: 生成的迹线矩阵，形状为 ``(num_traces, trace_length)``
    :rtype: numpy.ndarray
    """
    ...