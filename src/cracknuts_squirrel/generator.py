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



def _generate_base_waveform(length: int) -> np.ndarray:
    """
    生成平滑主波形，类似真实能量轨迹。
    可以叠加多个低频正弦波或缓慢变化曲线。
    """
    t = np.linspace(0, 4*np.pi, length)
    waveform = 0.3 * np.sin(t) + 0.2 * np.sin(0.5*t)  # 叠加两个正弦波
    return waveform

def generate_traces2(
    leakage: np.ndarray,
    *,
    trace_length: int,
    poi: np.ndarray | list,
    amplitude: float = 1.0,
    leakage_width: float = 8.0,
    white_noise_std: float = 0.05,
    jitter_std: float = 0.5,
    drift_std: float = 0.2,
    seed: int | None = None,
) -> np.ndarray:
    """
    生成更贴近真实能量轨迹的侧信道 trace。
    """
    leakage = np.asarray(leakage, dtype=np.float64)
    poi = np.asarray(poi, dtype=np.float64)
    num_traces, num_units = leakage.shape

    if poi.shape[0] != num_units:
        raise ValueError(f"len(poi)={poi.shape[0]} must equal num_units={num_units}")

    rng = np.random.default_rng(seed)
    traces = np.zeros((num_traces, trace_length), dtype=np.float64)

    # -----------------------------
    # 1. 构造基础平滑波形
    # -----------------------------
    base_trace = _generate_base_waveform(trace_length)

    # 在基础波形上叠加第0条 trace 的 leakage unit
    for u in range(num_units):
        base_trace += amplitude * leakage[0, u] * _gaussian_window(trace_length, poi[u], leakage_width)

    # -----------------------------
    # 2. 生成每条 trace
    # -----------------------------
    global_drift = rng.normal(0.0, drift_std, size=num_traces)

    for i in range(num_traces):
        trace = base_trace.copy()
        # 白噪声
        trace += rng.normal(0.0, white_noise_std, size=trace_length)

        # 每个 unit 的 POI 抖动 + 时间扭曲
        for u in range(num_units):
            center = poi[u] + rng.normal(0.0, jitter_std) + global_drift[i]
            # 叠加单位差异（相对于 base_trace）
            trace += amplitude * (leakage[i, u] - leakage[0, u]) * _gaussian_window(trace_length, center, leakage_width)

        traces[i] = trace

    return traces

