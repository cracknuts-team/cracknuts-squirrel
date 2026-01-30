from cracknuts_squirrel.generator import generate_traces, generate_traces2
import numpy as np
import matplotlib.pyplot as plt

num_traces = 200
trace_length = 50000

# 假设泄漏模型：随机 0~8 的汉明重量
rng = np.random.default_rng(1234)
leakage = rng.integers(0, 9, size=(num_traces, 16))

print(leakage.shape)

poi = [
    25000, 25200, 25400, 25600, 25800, 26000, 26200, 26400, 
    26800, 27000, 27200, 27400, 27600, 27800, 28000, 28200
]  # 泄漏发生在中间位置

traces = generate_traces2(
    leakage=leakage,
    trace_length=trace_length,
    poi=poi,
    leakage_width=4.0,
    amplitude=2.0,  # 极强泄漏
    white_noise_std=0.1,  # 极低噪声
    jitter_std=0.0,  # 无局部抖动
    drift_std=0.0,  # 无全局漂移
    time_warp_strength=0.0,
    seed=42,
)



# plt.figure(figsize=(10, 4))
# for i in range(20):
#     plt.plot(traces[i], alpha=0.6)
#
# plt.axvline(poi[0], linestyle="--")
# plt.title("Synthetic Side-Channel Traces")
# plt.xlabel("Sample index")
# plt.ylabel("Amplitude")
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(10, 4))

for i in range(2):
    plt.plot(traces[i,:10], linewidth=1, alpha=0.5)


# for i in range(16):
#     ct = _correlation_analysis(traces, leakage[:,i])
#     plt.plot(ct, linewidth=1)
#     # plt.axvline(poi[0], linestyle="--")


plt.title("Synthetic Side-Channel Traces")
plt.xlabel("Sample index")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
