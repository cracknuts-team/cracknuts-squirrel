import numpy as np
import matplotlib.pyplot as plt

# 参数
num_samples = 10240  # 波形长度
num_traces = 5  # 模拟多条曲线
trend_strength = 50  # 总体波动幅值
high_freq_std = 30  # 高频尖峰标准差
spike_prob = 0.01  # 尖峰出现概率
spike_height = 100  # 尖峰幅值

# 时间轴
t = np.linspace(0, 10, num_samples)

# 用于存储模拟波形
traces = np.zeros((num_traces, num_samples))

for i in range(num_traces):
    # 慢变化趋势
    trend = np.sin(0.05 * t) * trend_strength + np.sin(0.01 * t * (1 + 0.2 * np.random.rand())) * (trend_strength / 2)

    # 高频随机噪声
    noise = np.random.normal(0, high_freq_std, size=num_samples)

    # 随机尖峰
    spikes = np.zeros(num_samples)
    spike_positions = np.random.rand(num_samples) < spike_prob
    spikes[spike_positions] = np.random.uniform(-spike_height, spike_height, size=spike_positions.sum())

    # 叠加生成最终波形
    traces[i] = trend + noise + spikes

# 可视化第一条波形
plt.figure(figsize=(12, 4))
plt.plot(traces[0], color='blue')
plt.title("ca")
plt.xlabel("samples")
plt.ylabel("value")
plt.show()
