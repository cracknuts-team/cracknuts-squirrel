# 示例用法
import matplotlib.pyplot as plt
import numpy as np
from cracknuts_squirrel.correlation_analysis2 import CorrelationAnalysis, AnalysisParams
import zarr

d = zarr.open(r'D:\work\00.project\cracknuts-show\dataset\smt32f103_aes_power_10000.zarr')
print(d['/0/0/traces'].shape)

# analyzer = CorrelationAnalysis(input_path=r'D:\project\cracknuts\demo\jupyter\dataset\20250521110621(aes).zarr')
analyzer = CorrelationAnalysis(input_path=r'D:\work\00.project\cracknuts-show\dataset\smt32f103_aes_power_10000.zarr')
analyzer.auto_out_filename()
# analyzer.set_range(sample_range=(500, 10000))

result = analyzer.perform_analysis(AnalysisParams(data_type="plaintext"), AnalysisParams(data_type="ciphertext"))

print(result.shape)

plt.figure(figsize=(12, 6))
for x in range(result.shape[0]):
    plt.plot(result[x, :], linewidth=1)

plt.show()

print(f"分析完成，最大相关系数：{np.nanmax(np.abs(result)):.4f}")