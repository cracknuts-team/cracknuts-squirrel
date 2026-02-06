import numpy as np
import zarr
from cracknuts_squirrel.correlation_analysis3 import correlation_analysis
import matplotlib.pyplot as plt

def cpa_analysis():
    zd = zarr.open(r"D:\work\01.testing\99.test_dataset\smt32f103_aes_power_10000.zarr", mode="r")
    traces = zd['/0/0/traces']
    hypothesis = np.tile(np.arange(256), (16, traces.shape[0], 1))  # 示例：生成 AES CPA 的假设索引
    print(hypothesis.shape)
    for i in range(1):
        print(hypothesis[i].shape)
        corr = correlation_analysis(traces, hypothesis[i])
        print(corr.shape)
        print(corr[1])
        for j in range(corr.shape[0]):
            plt.plot(corr[i])


if __name__ == "__main__":
    plt.figure(figsize=(16, 8))
    cpa_analysis()
    plt.show()