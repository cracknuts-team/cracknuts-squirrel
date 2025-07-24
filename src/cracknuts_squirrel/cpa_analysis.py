import dask.array as da
import numpy as np
import numba as nb
from preprocessingBasic import PPBasic
from dask.diagnostics import ProgressBar
import zarr
from Crypto.Cipher import AES

# AES-128 sbox used to compute model values
AES_SBOX = np.array([99,124,119,123,242,107,111,197,48,1,103,43,254,215,171,118,
                    202,130,201,125,250,89,71,240,173,212,162,175,156,164,114,192,
                    183,253,147,38,54,63,247,204,52,165,229,241,113,216,49,21,
                    4,199,35,195,24,150,5,154,7,18,128,226,235,39,178,117,
                    9,131,44,26,27,110,90,160,82,59,214,179,41,227,47,132,
                    83,209,0,237,32,252,177,91,106,203,190,57,74,76,88,207,
                    208,239,170,251,67,77,51,133,69,249,2,127,80,60,159,168,
                    81,163,64,143,146,157,56,245,188,182,218,33,16,255,243,210,
                    205,12,19,236,95,151,68,23,196,167,126,61,100,93,25,115,
                    96,129,79,220,34,42,144,136,70,238,184,20,222,94,11,219,
                    224,50,58,10,73,6,36,92,194,211,172,98,145,149,228,121,
                    231,200,55,109,141,213,78,169,108,86,244,234,101,122,174,8,
                    186,120,37,46,28,166,180,198,232,221,116,31,75,189,139,138,
                    112,62,181,102,72,3,246,14,97,53,87,185,134,193,29,158,
                    225,248,152,17,105,217,142,148,155,30,135,233,206,85,40,223,
                    140,161,137,13,191,230,66,104,65,153,45,15,176,84,187,22])

AES_invSBOX = np.array([82,9,106,213,48,54,165,56,191,64,163,158,129,243,215,251,124,227,57,130,155,47,255,135,52,142,67,68,196,222,233,203,84,123,148,50,166,194,35,61,238,76,149,11,66,250,195,78,8,46,161,102,40,217,36,178,118,91,162,73,109,139,209,37,114,248,246,100,134,104,152,22,212,164,92,204,93,101,182,146,108,112,72,80,253,237,185,218,94,21,70,87,167,141,157,132,144,216,171,0,140,188,211,10,247,228,88,5,184,179,69,6,208,44,30,143,202,63,15,2,193,175,189,3,1,19,138,107,58,145,17,65,79,103,220,234,151,242,207,206,240,180,230,115,150,172,116,34,231,173,53,133,226,249,55,232,28,117,223,110,71,241,26,113,29,41,197,137,111,183,98,14,170,24,190,27,252,86,62,75,198,210,121,32,154,219,192,254,120,205,90,244,31,221,168,51,136,7,199,49,177,18,16,89,39,128,236,95,96,81,127,169,25,181,74,13,45,229,122,159,147,201,156,239,160,224,59,77,174,42,245,176,200,235,187,60,131,83,153,97,23,43,4,126,186,119,214,38,225,105,20,99,85,33,12,125])

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

@nb.njit(parallel=True)
def inv_shift_rows(state):
    """
    AES逆向行移位操作
    :param state: 16字节的状态矩阵(4x4)
    :return: 逆向行移位后的状态
    """
    # 将1D数组转换为4x4矩阵
    for i in nb.prange(state.shape[0]):
        state1 = state[i].reshape(4,4)
        
        # 对每一行进行不同的右移
        state1[1] = np.roll(state1[1], -1)
        state1[2] = np.roll(state1[2], -2)
        state1[3] = np.roll(state1[3], -3)

        state[i] = state1.flatten()
    
    return state

class CPAAnalysis(PPBasic):
    """
    AES-CPA分析类，继承自预处理基类PPBasic
    """
    def __init__(self, input_path=None, output_path=None, byte_pos=list(range(16)), sample_range=(0, None), dr='enc', **kwargs):
        super().__init__(input_path=input_path, output_path=output_path, **kwargs)
        self.byte_pos = byte_pos if isinstance(byte_pos, (list, tuple)) else [byte_pos] 
        self.sample_range = sample_range
        self.dr = dr



    # def update(self, traces: np.ndarray, data: np.ndarray):
    #     # Update the number of rows processed
    #     self.trace_count += traces.shape[0]
    #     # Update sample accumulator
    #     self.sample_sum += np.sum(traces, axis=0)
    #     # Update sample squared accumulator
    #     self.sample_sq_sum += np.sum(np.square(traces), axis=0)
    #     # Update model accumulator
    #     self.model_sum += np.sum(data, axis=0)
    #     # Update model squared accumulator
    #     self.model_sq_sum += np.sum(np.square(data), axis=0)
    #     data = data.reshape((data.shape[0], -1))
    #     # Update product accumulator
    #     self.prod_sum += np.matmul(data.T, traces)

    # def calculate(self):
    #     # Sample mean computation
    #     sample_mean = np.divide(self.sample_sum, self.trace_count)
    #     # Model mean computation
    #     model_mean = np.divide(self.model_sum, self.trace_count)

    #     prod_mean = np.divide(self.prod_sum, self.trace_count)
    #     # Calculate correlation coefficient numerator
    #     numerator = np.subtract(prod_mean, model_mean*sample_mean)
    #     # Calculate correlation coeefficient denominator sample part
    #     to_sqrt = np.subtract(np.divide(self.sample_sq_sum, self.trace_count), np.square(sample_mean))
    #     to_sqrt[to_sqrt < 0] = 0
    #     denom_sample = np.sqrt(to_sqrt)
    #     # Calculate correlation coefficient denominator model part
    #     to_sqrt = np.subtract(np.divide(self.model_sq_sum, self.trace_count), np.square(model_mean))
    #     to_sqrt = np.maximum(to_sqrt, 0)
    #     denom_model = np.sqrt(to_sqrt)

    #     denominator = denom_model*denom_sample

    #     denominator[denominator == 0] = 1

    #     return np.divide(numerator, denominator)

    def perform_cpa(self):
        """执行相关系数分析"""
        store = zarr.DirectoryStore(self.output_path)
        root = zarr.group(store=store, overwrite=True)
        
        # correlation = root.create_dataset(
        #     '/0/0/correlation',
        #     shape=(256, 16, self.sel_num_samples),
        #     chunks=(256, 16, 1000)
        # )
        correlation = np.zeros((256, 16, self.sel_num_samples), dtype=np.float32)
        

        traces = self.t[:self.sel_num_traces, self.sample_range[0]:self.sample_range[1]]
        
        # plaintext_bytes = np.array([pt[self.byte_pos] for pt in self.plaintext[:self.sel_num_traces]])
        if self.dr == 'enc':
            plaintext_bytes = self.plaintext[:self.sel_num_traces, :16]
        elif self.dr == 'dec':
            ciphertext_bytes = self.plaintext[:self.sel_num_traces, :16].compute()
            
        # key_bytes = np.arange(256)[:, np.newaxis]
        # xor_result = np.bitwise_xor(plaintext_bytes, key_bytes)
        # sbox_output = AES_SBOX[xor_result]
        # hw_matrix = WEIGHTS[sbox_output]
        
        # 创建Dask延迟计算任务
        def compute_correlation(key_byte):
            key_byte_array = np.full((self.sel_num_traces, 16), key_byte, dtype=np.uint8)
            if self.dr == 'enc':            
                xor_result = np.bitwise_xor(plaintext_bytes, key_byte_array)
                sbox_output = AES_SBOX[xor_result]
                hw_matrix = WEIGHTS[sbox_output]
            elif self.dr == 'dec':
                xor_result = np.bitwise_xor(ciphertext_bytes, key_byte_array)
                sbox_output = np.bitwise_xor(AES_invSBOX[xor_result], inv_shift_rows(ciphertext_bytes))
                hw_matrix = WEIGHTS[sbox_output]
        
            trace_count = self.sel_num_traces
            # 使用Dask进行延迟计算
            # trace_count = da.from_array([self.sel_num_traces], chunks=1)
            sample_sum = da.sum(traces, axis=0)
            model_sum = da.sum(hw_matrix, axis=0)
            prod_sum = da.dot(hw_matrix.T, traces)
            
            # 构建计算图
            model_mean = model_sum / trace_count
            sample_mean = sample_sum / trace_count
            
            # 调整维度以便广播
            numerator = (prod_sum / trace_count) - (model_mean.reshape(-1, 1) * sample_mean)
            
            denom = da.sqrt(
                da.maximum(0, (da.sum(hw_matrix**2)/trace_count - model_mean**2).reshape(-1, 1)) * 
                da.maximum(0, (da.sum(traces**2, axis=0)/trace_count - sample_mean**2))
            )
            denom = da.where(denom == 0, 1, denom)
            
            return da.divide(numerator, denom)

        # 并行计算所有密钥字节
        delayed_results = [compute_correlation(kb) for kb in range(256)]
        with ProgressBar():
            futures = da.compute(*delayed_results)
        
        # 收集结果
        # correlation = np.zeros((256, self.sel_num_samples), dtype=np.float32)
        for key_byte, result in enumerate(futures):
            correlation[key_byte] = result

        # 优化后的候选值分析（向量化操作）
        max_indices = np.argmax(np.abs(correlation), axis=2)
        candidates = np.take_along_axis(correlation, max_indices[:, :, np.newaxis], axis=2).squeeze()
        
        for j in self.byte_pos:
            # 获取当前字节的最优候选
            best_key = np.abs(candidates[:, j]).argmax()
            print(f'第{j+1}字节密钥: {hex(best_key)} 相关系数: {candidates[best_key, j]:.4f}')
            
            # 获取前5候选（向量化版本）
            top5_indices = np.argsort(np.abs(candidates[:, j]))[::-1][:5]
            for rank, idx in enumerate(top5_indices, 1):
                print(f'第{rank}候选值：{hex(idx)}，相关系数：{candidates[idx, j]:.4f}')
            print('\n')

        root.create_dataset(
            '/0/0/correlation',
            data = correlation[:, self.byte_pos, :]
        )
        # 添加元数据
        root.attrs.update({
            "cpa_metadata": {
                "analyzed_byte": self.byte_pos,
                "sample_range": self.sample_range,
                "trace_count": self.sel_num_traces
            }
        })

if __name__ == "__main__":
    # 示例用法
    cpa = CPAAnalysis(input_path=r'E:\\codes\\Acquisition\\dataset\\20250724071935.zarr', dr='dec')
    # cpa = CPAAnalysis(input_path=r'E:\\codes\\template\\dataset\\nut476_aes_random_data.zarr', dr='enc')
    
    cpa.auto_out_filename()
    # cpa.set_range(sample_range=(4000, 5200))  # 设置分析的采样点范围
    cpa.perform_cpa()