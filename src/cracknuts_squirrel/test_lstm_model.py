# Copyright 2024 CrackNuts. All rights reserved.

from collections import Counter

import numpy as np
import torch
import zarr

from lstm_aes_hd import AESHDModel, Sbox, inv_sbox, calc_GE


def load_model(model_path, trace_length, units):
    """加载训练好的模型"""
    model = AESHDModel(trace_length, units)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_test_data(zarr_path, test_index, params):
    """加载测试数据"""
    zarr_store = zarr.DirectoryStore(zarr_path)
    zarr_group = zarr.open(zarr_store, mode='r')
    
    traces = np.array(zarr_group['0/0/traces'])
    plaintext = np.array(zarr_group['0/0/plaintext'])
    
    test_traces = []
    test_plaintexts = []
    
    for i in test_index:
        trace = traces[i][params['trace_offset']:params['trace_offset']+params['trace_length']]
        trace = trace.reshape(1, -1).astype(np.float32) / 64.0
        pt = plaintext[i]
        
        test_traces.append(torch.tensor(trace, dtype=torch.float32))
        test_plaintexts.append(pt)
    
    return test_traces, test_plaintexts

def perform_attack(model, test_traces, test_plaintexts, params):
    """执行攻击并猜测密钥"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    predictions = []
    with torch.no_grad():
        for trace in test_traces:
            trace = trace.to(device)
            output = model(trace.unsqueeze(0))  # Add batch dimension
            predictions.append(output.cpu().numpy()[0])
    
    # 猜测密钥
    key_list = []
    byte_index = params['byte_index']
    
    for i, (pred, pt) in enumerate(zip(predictions, test_plaintexts)):
        predicted_label = np.argmax(pred)
        p_value = pt[byte_index]
        # 使用S盒进行逆运算以恢复密钥猜测
        sin = Sbox[predicted_label ^ int(p_value)]
        key_guess = int(sin) ^ int(p_value)
        key_list.append(key_guess)
        
        if i % 500 == 0 and i > 0:
            print(f'Attack trace: {i}')
            key_counter = Counter(key_list)
            print(key_counter.most_common(10))
    
    # 计算最终的密钥猜测
    final_key_counter = Counter(key_list)
    print('Final key guesses:')
    print(final_key_counter.most_common(10))
    
    # 计算猜测熵
    inter_value_pro = np.zeros(256)
    
    for j, (pred, pt) in enumerate(zip(predictions, test_plaintexts)):
        p_value = pt[byte_index]
        
        for key in range(256):
            # 计算中间值
            inter_value = int(inv_sbox[int(int(p_value) ^ int(key))] ^ int(p_value))
            inter_value_pro[key] += np.log(pred[inter_value] + 1e-10)  # 添加小值避免log(0)
        
        if j % 500 == 0 and j > 0:
            print(f'Attack trace: {j}')
            entropy, sorted_pro, sorted_index = calc_GE(inter_value_pro, params['key_suppose'])
            print(f'Entropy: {entropy}')
            for k in range(255, 245, -1):
                print(f'{sorted_pro[k]}, {sorted_index[k]}')
    
    # 最终熵计算
    entropy, sorted_pro, sorted_index = calc_GE(inter_value_pro, params['key_suppose'])
    key_guess_ge = sorted_index[255]
    print(f'Final entropy: {entropy}')
    print('Top 10 key candidates:')
    for k in range(255, 245, -1):
        print(f'{sorted_pro[k]}, {sorted_index[k]}')
    
    return key_guess_ge, final_key_counter.most_common(1)[0][0] if final_key_counter else None

def main():
    # 配置参数
    trace_offset = 500
    trace_length = 2000
    units = 128
    byte_index = 3
    key_array = [0xd0, 0x14, 0xf9, 0xa8, 0xc9, 0xee, 0x25, 0x89, 0xe1, 0x3f, 0xc, 0xc8, 0xb6, 0x63, 0xc, 0xa6]
    key_suppose = key_array[byte_index]
    
    test_index = np.arange(0, 5000)
    
    params = {
        'trace_offset': trace_offset,
        'trace_length': trace_length,
        'byte_index': byte_index,
        'key_suppose': key_suppose
    }
    
    # 文件路径 (根据实际情况修改这些路径)
    model_path = 'E:\\models\\byte3.pt'  # 模型文件路径
    zarr_path = 'E:\\codes\\Acquisition\\dataset\\merged.zarr'  # Zarr数据文件路径
    
    # 加载模型
    print('Loading model...')
    model = load_model(model_path, trace_length, units)
    
    # 加载测试数据
    print('Loading test data...')
    test_traces, test_plaintexts = load_test_data(zarr_path, test_index, params)
    
    # 执行攻击
    print('Performing attack...')
    guessed_key_ge, guessed_key = perform_attack(model, test_traces, test_plaintexts, params)
    
    if guessed_key_ge is not None:
        print(f'\nMost likely key byte: 0x{guessed_key_ge:02x}')
        print(f'Most counted key byte: 0x{guessed_key:02x}')
        print(f'Actual key byte: 0x{key_suppose:02x}')
        print(f'Match: {guessed_key_ge == key_suppose}')
    else:
        print('No key guess could be made.')

if __name__ == '__main__':
    main()