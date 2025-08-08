# 示例用法
import matplotlib.pyplot as plt
import numpy as np
from cracknuts_squirrel.correlation_analysis2 import CorrelationAnalysis

analyzer = CorrelationAnalysis(input_path=r'D:\project\cracknuts\demo\jupyter\dataset\20250521110621(aes).zarr')
analyzer.auto_out_filename()
# analyzer.set_range(sample_range=(500, 10000))

result = analyzer.perform_analysis(data_type="ciphertext", bit_width=2)

print(result.shape)
for x in range(result.shape[0]):
    plt.plot(result[x, :])

plt.show()

print(f"分析完成，最大相关系数：{np.nanmax(np.abs(result)):.4f}")