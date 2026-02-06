from enum import auto
import dask.array as da
from cracknuts_squirrel.preprocessing_basic import PPBasic
import numba as nb
import numpy as np
import scipy as sp
from tqdm import tqdm
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import zarr
import os
from datetime import datetime

# AES S盒（固定常量，用于侧信道泄漏仿真）
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
                    140,161,137,13,191,230,66,104,65,153,45,15,176,84,187,22],
                    dtype='uint8')

class Gendata(PPBasic):
    """
    用于侧信道曲线对齐的类，继承自曲线预处理的基类PPBasic
    功能：生成AES侧信道仿真数据（明文/密文/泄漏轨迹），并存储为Zarr格式
    """
    def __init__(self, input_path=None, num_traces=5000, sample_length=50, model_pos=[0],posset=[4,24], tile='/0/0/',  **kwargs):
        """
        初始化参数
        :param input_path: Zarr文件输出路径（同时作为父类input_path，需提前创建空数据集满足父类校验）
        :param num_traces: 生成轨迹数量，默认5000
        :param sample_length: 每条轨迹的采样点数量，默认50
        :param model_pos: 侧信道建模位置，默认[0]（AES字节位置）
        :param tile: Zarr数据集存储路径，默认'/0/0/'
        :param kwargs: 传递给父类的其他参数
        """
        # ========== 核心修复：父类初始化前，提前创建空Zarr数据集 ==========
        # 1. 确保输出路径存在，不存在则创建
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        
        # 泄漏位置：4+model_pos（明文泄漏）、24+model_pos（S盒输出泄漏）  
        if posset[1] + model_pos[-1]> sample_length:
            sample_length = posset[1] + model_pos[-1]
        self.posset=posset
        self.num_traces = num_traces    # 轨迹总数
        self.sample_length = sample_length  # 单轨迹采样点数量
        self.batch_size = 5000          # 批处理大小
        # 2. 创建空Zarr组和空traces/plaintext/ciphertext数据集，满足父类校验
        self._create_empty_zarr_datasets(input_path, tile, self.num_traces, self.sample_length)
        
        # 3. 调用父类构造函数（此时路径下已有空数据集，父类检查通过）
        super().__init__(input_path=input_path, output_path=input_path, tile=tile,** kwargs)
        
        # 实例属性初始化
        
        self.model_positions = model_pos # 建模位置
        self.key = None                 # AES密钥（16字节uint8）
        self.plaintext = None           # 明文数组（N×16 uint8）
        self.ciphertext = None          # 密文数组（N×16 uint8）
        self.traces = None              # 泄漏轨迹数组（N×sample_length int64）
        self.fetch_async = True
        self.zarr_output_path = input_path  # Zarr输出路径
        self.tile = tile                  # Zarr存储子路径

    def _create_empty_zarr_datasets(self, zarr_path, tile, num_traces, sample_length):
        """
        私有方法：在父类初始化前创建空的Zarr数据集，满足父类的文件存在性校验
        :param zarr_path: Zarr根路径
        :param tile: 子路径（如/0/0/）
        :param num_traces: 轨迹数量
        :param sample_length: 单轨迹采样点数量
        """
        # 去除tile首尾的/，拼接正确的Zarr子路径
        tile_clean = tile.strip('/')
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store=store, overwrite=False)  # 不覆盖，仅创建空数据集
        
        # 递归创建子组（如/0/0/）
        current_group = root
        for part in tile_clean.split('/'):
            if part not in current_group:
                current_group = current_group.create_group(part)
        
        # 创建空的traces数据集（父类核心校验的数据集）
        if 'traces' not in current_group:
            current_group.create_dataset(
                'traces',
                shape=(num_traces, sample_length),
                dtype=np.int64,
                chunks=(self.batch_size, sample_length)
            )
        # 创建空的plaintext数据集
        if 'plaintext' not in current_group:
            current_group.create_dataset(
                'plaintext',
                shape=(num_traces, 16),
                dtype=np.uint8,
                chunks=(self.batch_size, 16)
            )
        # 创建空的ciphertext数据集
        if 'ciphertext' not in current_group:
            current_group.create_dataset(
                'ciphertext',
                shape=(num_traces, 16),
                dtype=np.uint8,
                chunks=(self.batch_size, 16)
            )
        print(f"已提前创建空Zarr数据集，满足父类校验：{zarr_path + tile}")

    def configure(self, tile_x, tile_y, model_positions, convergence_step=None):
        self.model_positions = model_positions
        self.slabs = []
        batch_start_index = 0
        while batch_start_index < self.num_traces:
            entry_count = min(self.batch_size, self.num_traces - batch_start_index)
            self.slabs.append(slice(batch_start_index, batch_start_index+entry_count))
            batch_start_index += entry_count
        return 1

    def get_plaintext(self):
        return self.plaintext
    def get_ciphertext(self):
        return self.ciphertext
    def get_traces(self):
        return self.traces
    def get_key(self):
        return self.key
    
    def get_byte_batch(self, slab, model_pos):
        return [self.plaintext[slab, [model_pos]], self.key[[model_pos]], self.traces[slab,:]]
    
    def get_batches_by_byte(self, tile_x, tile_y, model_pos):
        for slab in self.slabs:
            yield self.get_byte_batch(slab, model_pos)
    
    def get_batch(self, slab):
        return [self.plaintext[slab,self.model_positions], self.key[self.model_positions], self.traces[slab,:]]
    
    def get_batches_all(self, tile_x, tile_y):
        for slab in self.slabs:
            yield self.get_batch(slab)

    def get_batch_index(self, index):
        if index >= len(self.slabs):
            return []
        return [self.plaintext[self.slabs[index], self.model_positions], self.key[self.model_positions], self.traces[self.slabs[index], :]]

    def generate_data(self, method='aes'):
        """
        核心功能：生成AES侧信道仿真数据，覆盖提前创建的空Zarr数据集
        :param method: 加密方法，固定为'aes'
        :return: 无返回值，数据直接写入Zarr文件
        """
        store = zarr.DirectoryStore(self.zarr_output_path)
        root = zarr.group(store=store, overwrite=False)  # 不覆盖组，仅覆盖数据集内容
        
        # 拼接子路径，获取已创建的空数据集
        tile_parts = self.tile.strip('/').split('/')
        current_group = root
        for part in tile_parts:
            current_group = current_group[part]
        zarr_traces = current_group['traces']
        zarr_plaintext = current_group['plaintext']
        zarr_ciphertext = current_group['ciphertext']

        # ********** 核心：生成AES明文/密钥/密文 **********
        N = self.num_traces       # 轨迹数量 = 明文块数量
        l = self.sample_length    # 单轨迹采样点数量
        # 生成随机明文：N×16 uint8，每个字节0-255
        plaintexts = np.random.randint(0, 256, (N, 16), dtype=np.uint8)
        # 生成随机AES-128密钥：16 uint8，全局唯一密钥
        self.key = np.random.randint(0, 256, 16, dtype=np.uint8)
        key_bytes = self.key.tobytes()  # 密钥转字节串，供AES加密使用

        # 初始化AES-ECB加密器（侧信道仿真常用ECB模式，无IV，独立块加密）
        cipher = AES.new(key_bytes, AES.MODE_ECB)
        # 初始化密文数组：N×16 uint8，存储所有明文块的加密结果
        ciphertexts = np.empty((N, 16), dtype=np.uint8)

        # 批量加密所有明文块（带进度条，直观查看生成进度）
        for i in tqdm(range(N), desc="Generating AES ciphertexts"):
            # 单明文块转字节串 → 加密 → 密文字节串转回uint8数组
            plain_block = plaintexts[i].tobytes()
            cipher_block = cipher.encrypt(plain_block)
            ciphertexts[i, :] = np.frombuffer(cipher_block, dtype=np.uint8)

        # ********** 核心：生成侧信道泄漏轨迹（基于AES S盒输出）**********
        # 初始化随机轨迹基底：N×l int64，范围-128~127（模拟原始采集噪声）
        traces = np.random.randint(-128, 128, (N, l), dtype=np.int64)

        # 注入AES侧信道泄漏：明文字节、S盒输出字节的数值泄漏（经典仿真方式）
        # 泄漏位置：4+model_pos（明文泄漏）、24+model_pos（S盒输出泄漏）
        for model_pos in self.model_positions:
            # 确保建模位置在0-15范围内（AES共16个字节位置）
            if 0 <= model_pos < 16:
                # 明文字节泄漏：trace[:,4+pos] = 明文字节 - 128（中心化，适配采集范围）
                leak_plain = plaintexts[:, model_pos]
                traces[:, self.posset[0] + model_pos] = np.subtract(leak_plain, 128, dtype=np.int16)
                # S盒输出泄漏：AES核心操作 S[plaintext ^ key]，注入到指定轨迹位置
                leak_sbox_in = plaintexts[:, model_pos] ^ self.key[model_pos]  # 明文与密钥异或
                leak_sbox_out = AES_SBOX[leak_sbox_in]  # S盒查表输出（AES核心非线性操作）
                traces[:, self.posset[1] + model_pos] = np.subtract(leak_sbox_out, 128, dtype=np.int16)

        # ********** 完善Zarr元数据（实时时间戳+关键侧信道信息）**********
        root.attrs.update({
            "metadata": {
                "channel_names": ["AES_SCA_Channel_1"],  # 信道名称，标识侧信道采集信道
                "create_time": int(datetime.now().timestamp()),  # 实时生成创建时间戳（秒）
                "data_length": 16,  # AES数据长度（固定16字节）
                "sample_count": self.sample_length,  # 单轨迹采样点数量
                "trace_count": self.num_traces,      # 轨迹总数
                "aes_key": self.key.tolist(),        # 关键：存储AES密钥（列表格式，可直接读取）
                "sbox_leakage_pos": [self.posset[1] +p for p in self.model_positions],  # S盒泄漏轨迹位置
                "plain_leakage_pos": [self.posset[0] +p for p in self.model_positions],   # 明文泄漏轨迹位置
                "version": "((0, '0.0.1'), (0, '0.0.1'))"
            }
        })

        # ********** 核心：将生成的真实数据覆盖提前创建的空Zarr数据集 **********
        zarr_traces[:] = traces           # 覆盖空轨迹数据集
        zarr_plaintext[:] = plaintexts    # 覆盖空明文数据集
        zarr_ciphertext[:] = ciphertexts  # 覆盖空密文数据集

        # ********** 实例属性赋值（供外部调用get_*方法获取数据）**********
        self.plaintext = zarr_plaintext   # Zarr明文数据集
        self.ciphertext = zarr_ciphertext # Zarr密文数据集
        self.traces = zarr_traces         # Zarr轨迹数据集

        print(f"\n✅ 数据生成完成！已存储至Zarr文件：{self.zarr_output_path}")
        print(f"📊 生成信息：{self.num_traces}条轨迹 | 每条{self.sample_length}个采样点 | AES-128密钥：{self.key.tolist()}")

if __name__ == "__main__":
    # 示例用法：生成100条轨迹，每条50个采样点，存储至指定Windows路径
    zarr_save_path = './aes_gen.zarr'
    # 初始化数据生成器：指定输出路径、轨迹数、采样点数量，建模位置为[0,1]（AES前2个字节）
    gder = Gendata(
        input_path=zarr_save_path,  # 该路径会提前创建空Zarr数据集，满足父类校验
        num_traces=1000,             # 生成100条轨迹
        sample_length= 60,           # 每条轨迹50个采样点
        model_pos=[0],             # 对AES第0、1字节进行侧信道建模
        posset=[4,24],
    )
    # 配置批处理（可选，默认batch_size=5000）
    gder.configure(tile_x=0, tile_y=0, model_positions=[0,1])
    # 核心调用：生成AES侧信道数据并覆盖空Zarr数据集
    gder.generate_data(method='aes')

    # 验证：读取生成的Zarr数据（示例）
    print("\n--- 📋 验证生成的数据 ---")
    print(f"AES密钥：{gder.get_key()}")
    print(f"明文形状：{gder.get_plaintext().shape}")
    print(f"密文形状：{gder.get_ciphertext().shape}")
    print(f"轨迹形状：{gder.get_traces().shape}")