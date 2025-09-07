import numpy as np
import pytest

from cracknuts_squirrel.correlation_analysis2 import CorrelationAnalysis

@pytest.mark.parametrize("data_length", [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], ids=lambda x: f"data_length:[{x}]")
@pytest.mark.parametrize("bit_width", [1, 2, 4, 8, 16, 32, 64], ids=lambda x: f"bit_width:[{x}]")
def test_hw2(data_length: int, bit_width: int):
    ...

def test_hw():
    d = np.frombuffer(bytes.fromhex("00 01 02 03 04 05 06 07"), dtype=np.uint8).reshape((-1, 4))
    print(CorrelationAnalysis.hamming_weight(data=d, bit_width=32))
    print(CorrelationAnalysis.hamming_weight(data=d, bit_width=8))

def test_analysis():
    import matplotlib.pyplot as plt

    analyzer = CorrelationAnalysis(input_path=r'D:\project\cracknuts\demo\jupyter\dataset\20250521110621(aes).zarr')
    analyzer.auto_out_filename()
    # analyzer.set_range(sample_range=(500, 10000))

    result = analyzer.perform_analysis(data_type="key")

    print(result.shape)
    for x in range(16):
        plt.plot(result[x,:1800])

    plt.show()

    print(f"分析完成，最大相关系数：{np.nanmax(np.abs(result)):.4f}")