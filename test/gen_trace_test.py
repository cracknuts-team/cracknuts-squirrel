from cracknuts_squirrel.correlation_analysis3 import correlation_analysis
from cracknuts_squirrel.generate_trace import gen_aes_golden_trace
import numpy as np
import matplotlib.pyplot as plt


traces, leakage = gen_aes_golden_trace(
    # base_trace=np.load('./base_trace.npy'),
    num_traces=2000,
    key=np.array([0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff], dtype=np.uint8),
    leakage_model='ID',
    do_jitter=False
)

print(f"the traces shape: {traces.shape}")

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.plot(traces[i, :1000], linewidth=1, label=f'Trace {i}')
plt.legend(
    loc="upper right",   # 位置
    fontsize=10,         # 字体大小
    frameon=True,        # 是否显示边框
)
plt.title('Generated AES Golden Traces')
plt.show()


corr = correlation_analysis(traces, leakage)
for i, t in enumerate(corr):
    plt.plot(t, alpha=0.5, linewidth = 1, label = i)
plt.legend()
plt.show()