import numpy as np
import zarr

from cracknuts_squirrel.preprocessing_basic import PPBasic


class CorrelationAnalysis(PPBasic):
    # Hamming weights of the values 0-255 used for model values
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

    """
    通用相关系数分析类，继承自预处理基类PPBasic
    """
    def __init__(self, input_path=None, output_path=None, sample_range=(0, None), **kwargs):
        super().__init__(input_path=input_path, output_path=output_path, **kwargs)
        self.sample_range = sample_range

    @classmethod
    def hamming_weight(cls, data: np.ndarray, bit_width: int) -> np.ndarray:

        if data.ndim != 2:
            raise ValueError(f"输入必须是二维数组，当前形状为 {data.shape}")

        if data.dtype != np.uint8:
            raise ValueError("输入数组必须是 uint8 类型")

        if bit_width == 8:
            return cls._HW_TABLE[data]

        n, m = data.shape

        if bit_width > 8:
            if bit_width not in (16, 32, 64):
                raise ValueError("合并仅支持 16, 32, 64 位")
            num_bytes = bit_width // 8
            if m % num_bytes != 0:
                raise ValueError(f"列数 {m} 不能被 {num_bytes} 整除")

            reshaped = data.reshape(n, m // num_bytes, num_bytes)
            tmp_hw = cls._HW_TABLE[reshaped]
            
            return tmp_hw.astype(f"uint{bit_width}").sum(axis=2)

        else:  # bit_length < 8
            if bit_width not in (1, 2, 4):
                raise ValueError("拆分仅支持 1, 2, 4 位")
            if 8 % bit_width != 0:
                raise ValueError(f"8 不能被 {bit_width} 整除")
            splits_per_byte = 8 // bit_width
            new_m = m * splits_per_byte
            result = np.zeros((n, new_m), dtype=np.uint8)

            for i in range(splits_per_byte):
                shift = 8 - bit_width * (i + 1)
                mask = (1 << bit_width) - 1
                result[:, i::splits_per_byte] = (data >> shift) & mask

            return cls._HW_TABLE[result]

    @staticmethod
    def calculate_correlation(traces, data_bytes):
        """
        计算轨迹(traces)与明文字节(plaintext_bytes)之间的相关系数
        
        参数:
        traces: numpy数组，形状为(n_traces, n_samples)，包含功率轨迹数据
        data_bytes: numpy数组，形状为(n_traces, n_bytes)，包含明文字节数据
        
        返回:
        correlations: numpy数组，形状为(n_bytes, n_samples)，包含每个字节位置与每个样本点的相关系数
        """
        # 确保输入是numpy数组
        traces = np.asarray(traces)
        data_bytes = np.asarray(data_bytes)
        
        # 检查输入维度是否匹配
        if traces.shape[0] != data_bytes.shape[0]:
            raise ValueError("轨迹数量与明文数量不匹配")
        
        n_traces, n_samples = traces.shape
        n_bytes = data_bytes.shape[1]
        
        # 初始化相关系数结果数组
        correlations = np.zeros((n_bytes, n_samples))
        
        # 对每个字节位置计算相关系数
        for byte_idx in range(n_bytes):
            # 提取当前字节数据
            byte_data = data_bytes[:, byte_idx]
            
            # 对每个样本点计算相关系数
            for sample_idx in range(n_samples):
                # 提取当前样本点的轨迹数据
                trace_data = traces[:, sample_idx]

                if np.std(byte_data) == 0:
                    correlations[byte_idx, sample_idx] = 0  # 或者 np.nan
                else:
                    correlations[byte_idx, sample_idx] = np.corrcoef(byte_data, trace_data)[0, 1]

        return correlations

    def perform_analysis(self, data_type="plaintext", bit_width=8):
        """
        执行相关系数分析（使用明文字节的汉明重量作为模型值）
        """
        store = zarr.DirectoryStore(self.output_path)
        root = zarr.group(store=store, overwrite=True)
    
        # 获取处理后的轨迹数据
        processed_traces = self.t[:, self.sample_range[0]:self.sample_range[1]]

        if data_type == "plaintext":
            # 获取明文字节并计算汉明重量
            plaintext = self.plaintext[:self.sel_num_traces, :]
            hw_matrix = self.hamming_weight(plaintext, bit_width)
        elif data_type == "ciphertext":
            ciphertext = self.ciphertext[:self.sel_num_traces, :]
            hw_matrix = self.hamming_weight(ciphertext, bit_width)
        elif data_type == "key":
            key = self.key[:self.sel_num_traces, :]
            hw_matrix = self.hamming_weight(key, bit_width)
        else:
            print(f"data_type error: [{data_type}].")
            return
        
        # 计算相关系数矩阵
        correlation_matrix = self.calculate_correlation(
            traces=processed_traces,
            data_bytes=hw_matrix
        )
    
        # 存储结果
        root.create_dataset(
            '/0/0/correlation',
            data=correlation_matrix,
            chunks=(16, 1000)
        )
    
        # 添加元数据
        root.attrs.update({
            "analysis_metadata": {
                "sample_range": self.sample_range,
                "trace_count": self.sel_num_traces,
                "model_type": "hamming_weight"
            }
        })
        return correlation_matrix

if __name__ == "__main__":

    # d = np.frombuffer(bytes.fromhex("00 01 02 03 04 05 06 07"), dtype=np.uint8).reshape((-1, 4))
    #
    # print(CorrelationAnalysis.hamming_weight(data=d, bit_width=4))

    # 示例用法
    import matplotlib.pyplot as plt

    analyzer = CorrelationAnalysis(input_path=r'D:\project\cracknuts\demo\jupyter\dataset\20250521110621(aes).zarr')
    analyzer.auto_out_filename()
    # analyzer.set_range(sample_range=(500, 10000))

    result = analyzer.perform_analysis(data_type="plaintext", bit_width=2)

    print(result.shape)
    for x in range(result.shape[0]):
        plt.plot(result[x,:])

    plt.show()

    print(f"分析完成，最大相关系数：{np.nanmax(np.abs(result)):.4f}")