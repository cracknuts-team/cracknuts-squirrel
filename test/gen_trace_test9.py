import numpy as np


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
        0.7 * np.exp(-((t - 0.18) / 0.06) ** 2) +   # load
        1.2 * np.exp(-((t - 0.50) / 0.10) ** 2) +   # compute
        0.5 * np.exp(-((t - 0.78) / 0.05) ** 2)     # store
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


# ---------------- example usage ----------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    trace = generate_base_trace(
        num_samples=4000,
        enc_start=800,
        enc_end=3200,
        cycle_len=48,
        scale=80.0,
        noise_std=0.6,
        seed=1,
    )

    plt.figure(figsize=(10, 3))
    plt.plot(trace, linewidth=1)
    plt.axvspan(800, 3200, color="red", alpha=0.1)
    plt.title("Realistic Base Trace")
    plt.tight_layout()
    plt.show()
