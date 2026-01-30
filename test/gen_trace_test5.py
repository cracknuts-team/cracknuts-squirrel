import numpy as np
import matplotlib.pyplot as plt

from cracknuts_squirrel.correlation_analysis3 import correlation_analysis


# ============================================================
# Gaussian window
# ============================================================

def gaussian_window(length: int, center: float, sigma: float) -> np.ndarray:
    t = np.arange(length)
    return np.exp(-0.5 * ((t - center) / sigma) ** 2)


# ============================================================
# Base trace generator (realistic power-like waveform)
# ============================================================

def generate_base_trace(length: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)

    t = np.linspace(0, 10, length)

    trend = (
        np.sin(0.05 * t) * 50
        + np.sin(0.01 * t * (1 + 0.2 * rng.random())) * 25
    )

    noise = rng.normal(0, 30, size=length)

    spikes = np.zeros(length)
    spike_positions = rng.random(length) < 0.005
    spikes[spike_positions] = rng.uniform(
        -120, 120, size=spike_positions.sum()
    )

    return trend + noise + spikes


# ============================================================
# Trace generator (SEMANTICALLY CORRECT)
# ============================================================

def generate_traces(
    leakage: np.ndarray,
    *,
    trace_length: int,
    poi: np.ndarray,
    amplitude: float = 5.0,
    leakage_width: float = 12.0,
    white_noise_std: float = 10.0,
    jitter_std: float = 2.0,
    drift_std: float = 4.0,
    seed: int | None = None,
) -> np.ndarray:

    leakage = np.asarray(leakage, dtype=np.float64)
    poi = np.asarray(poi, dtype=np.float64)

    num_traces, num_units = leakage.shape
    assert poi.shape[0] == num_units

    rng = np.random.default_rng(seed)

    base_trace = generate_base_trace(trace_length, seed=seed)
    traces = np.zeros((num_traces, trace_length), dtype=np.float64)

    global_drift = rng.normal(0.0, drift_std, size=num_traces)

    for i in range(num_traces):
        trace = base_trace.copy()

        trace += rng.normal(0.0, white_noise_std, size=trace_length)

        for u in range(num_units):
            center = poi[u] + global_drift[i] + rng.normal(0.0, jitter_std)

            window = gaussian_window(
                trace_length, center, leakage_width
            )

            # ★ 关键点：泄漏幅度必须和噪声同量级，否则 CPA 必然失败
            trace += amplitude * leakage[i, u] * window

        traces[i] = trace

    return traces


# ============================================================
# CPA implementation
# ============================================================

def cpa(traces: np.ndarray, hypothesis: np.ndarray) -> np.ndarray:
    """
    Pearson correlation between hypothesis and each time sample
    """
    traces_c = traces - traces.mean(axis=0)
    hyp_c = hypothesis - hypothesis.mean()

    num = hyp_c @ traces_c
    den = np.sqrt(
        np.sum(hyp_c ** 2) * np.sum(traces_c ** 2, axis=0)
    )

    return num / den


# ============================================================
# TEST CASE
# ============================================================

if __name__ == "__main__":
    # -----------------------------
    # Parameters
    # -----------------------------
    num_traces = 3000
    num_units = 3
    trace_length = 2000

    rng = np.random.default_rng(123)

    # -----------------------------
    # Leakage model (Hamming Weight)
    # -----------------------------
    data = rng.integers(0, 256, size=(num_traces, num_units))
    leakage = np.unpackbits(
        data.astype(np.uint8), axis=1
    ).reshape(num_traces, num_units, 8).sum(axis=2)

    # -----------------------------
    # POIs
    # -----------------------------
    poi = np.array([1100, 1300, 1600])

    # -----------------------------
    # Generate traces
    # -----------------------------
    traces = generate_traces(
        leakage,
        trace_length=trace_length,
        poi=poi,
        amplitude=5.0,          # ★ 不要太小
        leakage_width=8,
        white_noise_std=12.0,
        jitter_std=2.0,
        drift_std=5.0,
        seed=42,
    )


    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.plot(traces[i], alpha=0.6)
    plt.title("Synthetic Side-Channel Traces")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))

    corrs = correlation_analysis(traces, leakage)
    for i in range(corrs.shape[0]):
        plt.plot(corrs[i], alpha=0.5, linewidth=1, label=f"poi-{i}")

    # for u in range(num_units):
    #     corr = cpa(traces, leakage[:, u])
    #     plt.plot(
    #         np.abs(corr),
    #         label=f"Unit {u} (POI≈{poi[u]})",
    #         alpha=0.9
    #     )
    #
    #     peak = np.argmax(np.abs(corr))
    #     print(f"Unit {u}: CPA peak at {peak}, corr={corr[peak]:.3f}")

    # for p in poi:
    #     plt.axvline(p, color="k", linestyle="--", alpha=0.3)

    plt.title("CPA Correlation Curves (Multiple Leakage Units)")
    plt.xlabel("Sample index")
    plt.ylabel("Correlation")
    plt.legend()
    plt.tight_layout()
    plt.show()
