import zarr
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from cracknuts_squirrel.correlation_analysis3 import correlation_analysis
import time


_HW_TABLE = np.array([
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

d_path = r'D:\work\01.testing\99.test_dataset\smt32f103_aes_power_10000.zarr'
zd = zarr.open(d_path, mode='r')
traces = zd['/0/0/traces']
plaintext = zd['/0/0/plaintext']

plt.figure(figsize=(16, 6))
plt.xlabel("Sample Index")
plt.ylabel("Absolute Correlation")
plt.title("CPA Correlation Result")
plt.grid(True)


s = time.time()
## old
# cvs = []
# for i in tqdm(range(16)):
#     cvs.append(_correlation_analysis(traces[:], _HW_TABLE[plaintext[:, i]]))
# print("old: cost time:", time.time() - s)
## new
print(f"traces shape: {traces.shape}, plaintext shape: {plaintext.shape}")
cvs = correlation_analysis(traces, _HW_TABLE[plaintext])
print("new: cost time:", time.time() - s)
## oldest
# from cracknuts_squirrel.correlation_analysis import CorrelationAnalysis
# analyzer = CorrelationAnalysis(input_path=d_path)
# analyzer.auto_out_filename()
# analyzer.perform_analysis()
# print("oldest: cost time:", time.time() - s)
# r_zd = zarr.open(r'D:\work\01.testing\99.test_dataset\smt32f103_aes_power_10000_CorrelationAnalysis.zarr')
# cvs = r_zd['/0/0/correlation']
# print(f"xxxxx {cvs.shape}")
###################################
# for cv in cvs:
#     plt.plot(cv, linewidth=1)
for i in range(16):
    plt.plot(cvs[i], linewidth=1, label = f'plaintext byte {i}')

plt.legend(
    loc="upper right",   # 位置
    fontsize=10,         # 字体大小
    frameon=True,        # 是否显示边框
)
plt.show()