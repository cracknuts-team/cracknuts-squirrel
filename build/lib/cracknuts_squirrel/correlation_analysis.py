import numpy as np
import zarr
from preprocessingBasic import PPBasic
import matplotlib.pyplot as plt

# Hamming weights of the values 0-255 used for model values
WEIGHTS = np.array([0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
                    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
                    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
                    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
                    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
                    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
                    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
                    4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8],
                    np.float32)

class CorrelationAnalysis(PPBasic):
    """
    通用相关系数分析类，继承自预处理基类PPBasic
    """
    def __init__(self, input_path=None, output_path=None, sample_range=(0, None), **kwargs):
        super().__init__(input_path=input_path, output_path=output_path, **kwargs)
        self.sample_range = sample_range

    def calculate_correlation(self, traces, plaintext_bytes):
        """
        计算轨迹(traces)与明文字节(plaintext_bytes)之间的相关系数
        
        参数:
        traces: numpy数组，形状为(n_traces, n_samples)，包含功率轨迹数据
        plaintext_bytes: numpy数组，形状为(n_traces, n_bytes)，包含明文字节数据
        
        返回:
        correlations: numpy数组，形状为(n_bytes, n_samples)，包含每个字节位置与每个样本点的相关系数
        """
        # 确保输入是numpy数组
        traces = np.asarray(traces)
        plaintext_bytes = np.asarray(plaintext_bytes)
        
        # 检查输入维度是否匹配
        if traces.shape[0] != plaintext_bytes.shape[0]:
            raise ValueError("轨迹数量与明文数量不匹配")
        
        n_traces, n_samples = traces.shape
        n_bytes = plaintext_bytes.shape[1]
        
        # 初始化相关系数结果数组
        correlations = np.zeros((n_bytes, n_samples))
        
        # 对每个字节位置计算相关系数
        for byte_idx in range(n_bytes):
            # 提取当前字节数据
            byte_data = plaintext_bytes[:, byte_idx]
            
            # 对每个样本点计算相关系数
            for sample_idx in range(n_samples):
                # 提取当前样本点的轨迹数据
                trace_data = traces[:, sample_idx]
                
                # 计算皮尔逊相关系数
                correlations[byte_idx, sample_idx] = np.corrcoef(byte_data, trace_data)[0, 1]
        
        return correlations

    def perform_analysis(self):
        """
        执行相关系数分析（使用明文字节的汉明重量作为模型值）
        """
        store = zarr.DirectoryStore(self.output_path)
        root = zarr.group(store=store, overwrite=True)
    
        # 获取处理后的轨迹数据
        processed_traces = self.t[:, self.sample_range[0]:self.sample_range[1]]
        
        # 获取明文字节并计算汉明重量
        plaintext = self.plaintext[:self.sel_num_traces, :16]
        hw_matrix = WEIGHTS[plaintext]
        
        # 计算相关系数矩阵
        correlation_matrix = self.calculate_correlation(
            traces=processed_traces,
            plaintext_bytes=hw_matrix
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
    # 示例用法
    analyzer = CorrelationAnalysis(input_path=r'D:\project\cracknuts-test\dataset\20250617104740.zarr')
    analyzer.auto_out_filename()
    analyzer.set_range(sample_range=(500, 10000))
    
    result = analyzer.perform_analysis()

    print(result.shape)
    for i in range(16):
        plt.plot(result[i,:])
    
    plt.show()
    
    print(f"分析完成，最大相关系数：{np.nanmax(np.abs(result)):.4f}")