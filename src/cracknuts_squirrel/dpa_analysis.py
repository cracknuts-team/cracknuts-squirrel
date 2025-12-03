# Copyright 2024 CrackNuts. All rights reserved.

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

import numba as nb
import numpy as np
import zarr

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

@nb.njit(parallel=True)
def compute_dpa_core(traces, hw_matrix, threshold, num_samples):
    """
    使用numba加速的DPA核心计算
    """
    num_traces = traces.shape[0]
    dpa_difference = np.zeros((16, num_samples), dtype=np.float32)
    
    for byte_idx in nb.prange(16):
        # 根据汉明权重阈值将轨迹分为两组
        high_group_mask = hw_matrix[:, byte_idx] > threshold
        low_group_mask = ~high_group_mask
        
        high_count = np.sum(high_group_mask)
        low_count = np.sum(low_group_mask)
        
        # 确保两组都有足够的轨迹
        if high_count > 0 and low_count > 0:
            # 计算高汉明权重组的平均轨迹
            high_sum = np.zeros(num_samples, dtype=np.float32)
            for i in range(num_traces):
                if high_group_mask[i]:
                    high_sum += traces[i]
            high_group_mean = high_sum / high_count
            
            # 计算低汉明权重组的平均轨迹
            low_sum = np.zeros(num_samples, dtype=np.float32)
            for i in range(num_traces):
                if low_group_mask[i]:
                    low_sum += traces[i]
            low_group_mean = low_sum / low_count
            
            # 计算差分轨迹
            dpa_difference[byte_idx] = high_group_mean - low_group_mean
        else:
            # 如果某组轨迹不足，设为0
            dpa_difference[byte_idx] = np.zeros(num_samples)
    
    return dpa_difference

class DPAAnalysis(PPBasic):
    """
    AES-DPA分析类，继承自预处理基类PPBasic
    DPA（差分功耗分析）使用差分均值分析而不是相关系数分析
    """
    def __init__(self, input_path=None, output_path=None, byte_pos=list(range(16)), sample_range=(0, None), dr='enc', threshold=4, **kwargs):
        super().__init__(input_path=input_path, output_path=output_path, **kwargs)
        self.byte_pos = byte_pos if isinstance(byte_pos, (list, tuple)) else [byte_pos] 
        self.sample_range = sample_range
        self.dr = dr
        self.threshold = threshold  # 汉明权重阈值，用于分组

    def perform_dpa(self):
        """执行差分功耗分析"""
        store = zarr.DirectoryStore(self.output_path)
        root = zarr.group(store=store, overwrite=True)
        
        # 初始化差分结果数组
        dpa_result = np.zeros((256, 16, self.sel_num_samples), dtype=np.float32)
        
        # 提前计算并缓存轨迹数据
        traces = self.t[:self.sel_num_traces, self.sample_range[0]:self.sample_range[1]].compute()
        
        if self.dr == 'enc':
            plaintext_bytes = self.plaintext[:self.sel_num_traces, :16].compute()
        elif self.dr == 'dec':
            ciphertext_bytes = self.plaintext[:self.sel_num_traces, :16].compute()
        
        def compute_dpa_for_key_byte(key_byte):
            """计算单个密钥字节的DPA结果"""
            key_byte_array = np.full((self.sel_num_traces, 16), key_byte, dtype=np.uint8)
            
            # 计算中间值
            if self.dr == 'enc':            
                xor_result = np.bitwise_xor(plaintext_bytes, key_byte_array)
                sbox_output = AES_SBOX[xor_result]
                hw_matrix = WEIGHTS[sbox_output]
            elif self.dr == 'dec':
                xor_result = np.bitwise_xor(ciphertext_bytes, key_byte_array)
                sbox_output = np.bitwise_xor(AES_invSBOX[xor_result], inv_shift_rows(ciphertext_bytes))
                hw_matrix = WEIGHTS[sbox_output]
            
            # 使用numba加速的DPA核心计算
            dpa_difference = compute_dpa_core(traces, hw_matrix, self.threshold, self.sel_num_samples)
            
            return key_byte, dpa_difference

        # 使用线程池并行计算
        print("开始DPA分析...")
        num_workers = min(mp.cpu_count(), 8)  # 限制最大线程数
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            futures = {executor.submit(compute_dpa_for_key_byte, kb): kb for kb in range(256)}
            
            # 收集结果
            completed = 0
            for future in as_completed(futures):
                key_byte, result = future.result()
                dpa_result[key_byte] = result
                completed += 1
                if completed % 10 == 0:
                    print(f"进度: {completed}/256 ({completed/256*100:.1f}%)")

        # 分析结果并输出最佳候选密钥
        for j in self.byte_pos:
            # 计算每个采样点的最大差分绝对值
            max_dpa_values = np.max(np.abs(dpa_result[:, j, :]), axis=1)
            
            # 找到最佳密钥候选
            best_key = np.argmax(max_dpa_values)
            best_dpa_value = max_dpa_values[best_key]
            
            print(f'第{j+1}字节密钥: {hex(best_key)} 最大DPA值: {best_dpa_value:.4f}')
            
            # 获取前5候选
            top5_indices = np.argsort(max_dpa_values)[::-1][:5]
            for rank, idx in enumerate(top5_indices, 1):
                print(f'第{rank}候选值：{hex(idx)}，DPA值：{max_dpa_values[idx]:.4f}')
            print('\n')

        # 保存结果到zarr文件
        root.create_dataset(
            '/0/0/dpa_result',
            data=dpa_result[:, self.byte_pos, :]
        )
        
        # 添加元数据
        root.attrs.update({
            "dpa_metadata": {
                "analyzed_byte": self.byte_pos,
                "sample_range": self.sample_range,
                "trace_count": self.sel_num_traces,
                "threshold": self.threshold,
                "direction": self.dr
            }
        })

    def perform_dpa_with_multiple_thresholds(self, thresholds=[3, 4, 5]):
        """
        使用多个阈值进行DPA分析，提高准确性
        """
        best_results = {}
        
        # 提前缓存数据，避免重复计算
        traces = self.t[:self.sel_num_traces, self.sample_range[0]:self.sample_range[1]].compute()
        
        if self.dr == 'enc':
            plaintext_bytes = self.plaintext[:self.sel_num_traces, :16].compute()
        elif self.dr == 'dec':
            ciphertext_bytes = self.plaintext[:self.sel_num_traces, :16].compute()
        
        for threshold in thresholds:
            print(f"使用阈值 {threshold} 进行DPA分析...")
            
            # 执行DPA分析
            store = zarr.DirectoryStore(f"{self.output_path}_threshold_{threshold}")
            root = zarr.group(store=store, overwrite=True)
            
            dpa_result = np.zeros((256, 16, self.sel_num_samples), dtype=np.float32)
            
            def compute_dpa_for_threshold(key_byte):
                key_byte_array = np.full((self.sel_num_traces, 16), key_byte, dtype=np.uint8)
                
                if self.dr == 'enc':            
                    xor_result = np.bitwise_xor(plaintext_bytes, key_byte_array)
                    sbox_output = AES_SBOX[xor_result]
                    hw_matrix = WEIGHTS[sbox_output]
                elif self.dr == 'dec':
                    xor_result = np.bitwise_xor(ciphertext_bytes, key_byte_array)
                    sbox_output = np.bitwise_xor(AES_invSBOX[xor_result], inv_shift_rows(ciphertext_bytes))
                    hw_matrix = WEIGHTS[sbox_output]
                
                # 使用numba加速的DPA核心计算
                dpa_difference = compute_dpa_core(traces, hw_matrix, threshold, self.sel_num_samples)
                
                return key_byte, dpa_difference
            
            # 使用线程池并行计算
            num_workers = min(mp.cpu_count(), 8)
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(compute_dpa_for_threshold, kb): kb for kb in range(256)}
                
                completed = 0
                for future in as_completed(futures):
                    key_byte, result = future.result()
                    dpa_result[key_byte] = result
                    completed += 1
                    if completed % 20 == 0:
                        print(f"阈值{threshold}进度: {completed}/256 ({completed/256*100:.1f}%)")
            
            # 分析每个字节的结果
            for j in self.byte_pos:
                max_dpa_values = np.max(np.abs(dpa_result[:, j, :]), axis=1)
                best_key = np.argmax(max_dpa_values)
                best_dpa_value = max_dpa_values[best_key]
                
                if j not in best_results or best_dpa_value > best_results[j]['value']:
                    best_results[j] = {
                        'key': best_key,
                        'value': best_dpa_value,
                        'threshold': threshold
                    }
            
            root.create_dataset('/0/0/dpa_result', data=dpa_result[:, self.byte_pos, :])
            root.attrs.update({
                "dpa_metadata": {
                    "analyzed_byte": self.byte_pos,
                    "sample_range": self.sample_range,
                    "trace_count": self.sel_num_traces,
                    "threshold": threshold,
                    "direction": self.dr
                }
            })
        
        # 输出最佳结果
        print("\n=== 多阈值DPA分析结果 ===")
        for j in self.byte_pos:
            result = best_results[j]
            print(f'第{j+1}字节最佳密钥: {hex(result["key"])} '
                  f'阈值: {result["threshold"]} '
                  f'DPA值: {result["value"]:.4f}')

if __name__ == "__main__":
    # 示例用法
    # dpa = DPAAnalysis(input_path=r'E:\\codes\\Acquisition\\dataset\\20250912072624.zarr', dr='dec')
    dpa = DPAAnalysis(input_path=r'E:\\codes\\template\\dataset\\nut476_aes_random_data.zarr', dr='enc')
    
    dpa.auto_out_filename()
    # dpa.set_range(sample_range=(4000, 5200))  # 设置分析的采样点范围
    
    # 执行基本DPA分析
    print("执行基本DPA分析...")
    dpa.perform_dpa()
    
    # 执行多阈值DPA分析
    print("\n执行多阈值DPA分析...")
    dpa.perform_dpa_with_multiple_thresholds(thresholds=[3, 4, 5])