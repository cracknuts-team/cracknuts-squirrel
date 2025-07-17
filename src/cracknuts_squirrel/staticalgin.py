from enum import auto
import dask.array as da
from cracknuts_squirrel.preprocessing_basic import PPBasic
import numba as nb
import numpy as np
import scipy as sp
from tqdm import tqdm
import zarr

class Staticalign(PPBasic):
    """
    用于侧信道曲线对齐的类，继承自曲线预处理的基类PPBasic
    """
    
    def __init__(self, input_path=None, output_path=None, ref_path=None, ref_index=0, tile='/0/0/', threshold=0, **kwargs):
        """
        初始化参数
        :param input_paths: 多个输入Zarr文件路径列表
        :param output_path: 输出Zarr文件路径
        :param reference_index: 参考曲线索引
        :param kwargs: 传递给父类的其他参数
        """
        super().__init__(input_path=input_path, output_path=output_path, tile='/0/0/',**kwargs)
        self.threshold = threshold
        self.ref_path = ref_path
        self.ref_index = ref_index
        
    def set_ref(self, ref_range=None, max_shift=0):
        """
        设置参考曲线
        :param ref_range: 参考曲线范围，默认为(0, num_samples)
        :param max_shift: 最大偏移量
        """
        if ref_range is None:
            ref_range = (0, self.num_samples)
        if self.ref_path is None:
            ref_path = self.input_path
        self.ref_range = ref_range
        self.max_shift = max_shift
        self.reference_trs = da.from_zarr(ref_path+'/0/0/traces')[self.ref_index, :].compute()
        self.reference = self.reference_trs[ref_range[0]:ref_range[1]]
        self.reference = self.reference[::-1]
        cross = sp.signal.fftconvolve(self.reference_trs, self.reference, mode='valid')
        self._refmaxloc = np.argmax(cross[ref_range[0]:ref_range[1]])
        self._refmaxsize = max(cross[ref_range[0]:ref_range[1]])
    

    # @nb.njit(parallel=True)
    def align_curves(self, method='correlation'):
        """
        :param method: 对齐方法，可选'correlation'(相关性)或'feature'(特征点)
        :return: 对齐后的Dask数组列表
        """
        store = zarr.DirectoryStore(self.output_path)
        root = zarr.group(store=store, overwrite=True)
        
        # 创建traces数据集
        traces = root.create_dataset(
            '/0/0/traces', 
            shape=(self.sel_num_traces, self.num_samples)
        )
        plaintext = root.create_dataset(
            '/0/0/plaintext', 
            shape=(self.plaintext.shape[0], self.plaintext.shape[1])
        )
        plaintext[self.trace_range[0]:self.trace_range[1]] = self.plaintext[self.trace_range[0]:self.trace_range[1]].compute()
        
        # 添加attrs元数据
        root.attrs.update({
            "metadata": {
                "channel_names": ["1"],
                "create_time": 1740140842,
                "data_length": self.plaintext.shape[1],
                "sample_count": self.sel_num_samples,
                "trace_count": self.sel_num_traces,
                "version": "((0, '0.0.1'), (0, '0.0.1'))"
            }
        })

        for trace in tqdm(range(self.trace_range[0], self.trace_range[1]), desc="对齐进度", unit="trace"):

            curr_trace = self.t[trace, :]
            # 归一化处理
            norm_trace = (curr_trace - np.mean(curr_trace)) / (np.std(curr_trace) * len(curr_trace))
            norm_ref = (self.reference - np.mean(self.reference)) / np.std(self.reference)
            # 计算互相关并归一化为相关系数
            cross = sp.signal.fftconvolve(norm_trace, norm_ref[::-1], mode='valid') 

            newmaxloc = np.argmax(cross[self.ref_range[0]:self.ref_range[1]])
                    
            diff = newmaxloc-self._refmaxloc
            if diff < 0:
                newtrace = np.append(np.zeros(-diff), curr_trace[:diff])
            elif diff > 0:
                newtrace = np.append(curr_trace[diff:], np.zeros(diff))
            traces[trace, :] = newtrace
            

if __name__ == "__main__":
    # 示例用法    
    aligner = Staticalign(input_path='E:\\codes\\template\\dataset\\20250221202721.zarr')
    aligner.auto_out_filename()
    aligner.set_ref(ref_range=(100, 1000))  # 设置参考曲线范围
    aligner.set_range(trace_range=(0, 100))  # 设置读取范围，可选
    aligner.align_curves(method='correlation')  # 使用相关性方法对齐曲线
