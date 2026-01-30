import numpy as np
import matplotlib.pyplot as plt

from cracknuts_squirrel.correlation_analysis3 import correlation_analysis
from cracknuts_squirrel.generator import generate_traces, generate_traces2

# ============================================================
# TEST CASE
# ============================================================

if __name__ == "__main__":
    # -----------------------------
    # Parameters
    # -----------------------------
    num_traces = 3000
    num_units = 3
    trace_length = 8000

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
    poi = np.array([1800, 3900, 6200])

    # -----------------------------
    # Generate traces
    # -----------------------------
    traces = generate_traces2(
        leakage,
        trace_length=trace_length,
        poi=poi,
        amplitude=25.0,          # ★ 不要太小
        leakage_width=4.0,
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

    for corr in corrs:
        plt.plot(corr, alpha=0.5)

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
