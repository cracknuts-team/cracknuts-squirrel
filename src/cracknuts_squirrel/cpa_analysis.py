# Copyright 2024 CrackNuts. All rights reserved.

import numba as nb
import numpy as np
import zarr
from dask import delayed  # 添加delayed导入
from dask.diagnostics import ProgressBar

from cracknuts_squirrel.preprocessing_basic import PPBasic

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

AES_invSBOX = np.array([0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38,
    0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87,
    0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D,
    0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2,
    0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16,
    0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA,
    0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A,
    0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02,
    0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA,
    0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85,
    0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89,
    0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20,
    0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31,
    0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D,
    0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0,
    0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26,
    0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D])
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
        
        correlation = np.zeros((256, 16, self.sel_num_samples), dtype=np.float32)
        
        traces = self.t[:self.sel_num_traces, self.sample_range[0]:self.sample_range[1]]
        
        if self.dr == 'enc':
            plaintext_bytes = self.plaintext[:self.sel_num_traces, :16]
        elif self.dr == 'dec':
            ciphertext_bytes = self.plaintext[:self.sel_num_traces, :16].compute()
            
        # 使用@dask.delayed装饰器包装NumPy计算
        @delayed
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
            
            # 使用NumPy直接计算，确保数学正确性
            sample_mean = np.mean(traces, axis=0)
            model_mean = np.mean(hw_matrix, axis=0)
            
            # 向量化计算协方差矩阵 - 大幅优化性能
            # 计算 hw_matrix 和 traces 的乘积和
            prod_sum = np.dot(hw_matrix.T, traces)
            
            # 计算协方差矩阵
            cov_matrix = (prod_sum / trace_count) - np.outer(model_mean, sample_mean)
            
            # 计算标准差
            model_var = np.maximum(0, np.var(hw_matrix, axis=0))
            sample_var = np.maximum(0, np.var(traces, axis=0))
            
            model_std = np.sqrt(model_var)
            sample_std = np.sqrt(sample_var)
            
            # 向量化计算相关系数
            denominator = np.outer(model_std, sample_std)
            denominator[denominator == 0] = 1  # 避免除零
            
            correlation_result = cov_matrix / denominator
            
            return correlation_result

        # 并行计算所有密钥字节
        delayed_results = [compute_correlation(kb) for kb in range(256)]
        with ProgressBar():
            futures = delayed_results
        
        # 收集结果
        for key_byte, result in enumerate(futures):
            correlation[key_byte] = result.compute()

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
    cpa = CPAAnalysis(input_path=r'E:\\codes\\Acquisition\\dataset\\20250909100916.zarr', dr='enc')
    # cpa = CPAAnalysis(input_path=r'E:\\codes\\template\\dataset\\nut476_aes_random_data.zarr', dr='enc')
    
    cpa.auto_out_filename()
    # cpa.set_range(sample_range=(4000, 5200))  # 设置分析的采样点范围
    cpa.perform_cpa()