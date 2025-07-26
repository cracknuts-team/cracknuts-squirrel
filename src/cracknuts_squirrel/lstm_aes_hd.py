import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import zarr
import os
from tqdm import tqdm

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 在文件开头添加CUDA检查
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA版本:", torch.version.cuda)
    print("可用GPU数量:", torch.cuda.device_count())
    print("当前GPU:", torch.cuda.get_device_name(0))


# 自定义数据集类
class TraceDataset(Dataset):
    def __init__(self, index_in, params):
        self.index_in = index_in
        self.params = params
        self.trace_struc = zarr.open(params['trs_file_path'], mode='r')
        
    def __len__(self):
        return len(self.index_in)
    
    def __getitem__(self, idx):
        i = self.index_in[idx]
        trace = self.trace_struc['0/0/traces'][i]
        label = self.trace_struc['0/0/plaintext'][i]
        
        trace = trace[self.params['trace_offset']:self.params['trace_offset']+self.params['trace_length']]
        trace = trace.reshape(1, -1).astype(np.float32) / 64.0  # 修改为(1, length)格式
        label_value = label[self.params['byte_index']]
        
        return torch.FloatTensor(trace), torch.LongTensor([label_value])
    


class InMemoryTraceDataset(Dataset):
    def __init__(self, index_list, params):
        super().__init__()
        self.index_list = index_list
        self.byte_index = params['byte_index']
        self.trace_length = params['trace_length']
        self.trace_offset = params['trace_offset']
        
        # 只打开一次Zarr文件，并将数据全部加载到内存
        zarr_store = zarr.DirectoryStore(params['trs_file_path'])
        zarr_group = zarr.open(zarr_store, mode='r')
        
        # 加载所有需要的traces和labels到内存
        # 注意：这会占用大量内存，请确保系统有足够的RAM
        self.traces = np.array(zarr_group['0/0/traces'])
        self.plaintext = np.array(zarr_group['0/0/plaintext'])
        
        # 可选：如果只需要部分数据，可以进行切片
        # self.traces = self.traces[:, self.trace_offset:self.trace_offset+self.trace_length]
        # self.labels = self.labels[:, self.byte_index]

    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self, idx):
        # 直接从内存中访问数据，无需再通过Zarr
        i = self.index_list[idx]
        trace = self.traces[i, self.trace_offset:self.trace_offset+self.trace_length]
        trace = trace.reshape(1, -1).astype(np.float32) / 64.0  # 修改为(1, length)格式
        # label = Sbox[self.plaintext[i, self.byte_index] ^ key_suppose]
        label = inv_sbox[self.plaintext[i, self.byte_index] ^ key_suppose] ^ self.plaintext[i, self.byte_index]
        
        # 转换为PyTorch张量
        return torch.tensor(trace, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_out):
        # lstm_out: [batch_size, seq_len, hidden_size]
        attn_weights = self.attn(lstm_out)  # [batch_size, seq_len, 1]
        attn_weights = self.softmax(attn_weights.squeeze(2))  # [batch_size, seq_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, seq_len]
        context = torch.bmm(attn_weights, lstm_out)  # [batch_size, 1, hidden_size]
        return context.squeeze(1)  # [batch_size, hidden_size]

class AESHDModel(nn.Module):
    def __init__(self, trace_length, units):
        super(AESHDModel, self).__init__()
        self.trace_length = trace_length
        self.units = units
        
        # LocallyConnected1D equivalent using Conv1d with groups
        self.local_conv = nn.Conv1d(1, 1, kernel_size=108, stride=54, padding=0, groups=1)
        
        # BatchNorm1d for sequence data
        self.bn_local = nn.BatchNorm1d(1)
        
        # BiLSTM
        self.fw_lstm = nn.LSTM(input_size=1, hidden_size=units, batch_first=True, bidirectional=False)
        self.bw_lstm = nn.LSTM(input_size=1, hidden_size=units, batch_first=True, bidirectional=False)
        
        # Attention layers
        self.fw_attention = Attention(units)
        self.bw_attention = Attention(units)
        
        # Output layers
        self.fc_out = nn.Linear(units * 2, 256)
        self.bn_out = nn.BatchNorm1d(256)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x shape: [batch_size, trace_length, 1]
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)
        
        # Local convolution
        x = x.transpose(1, 2)  # [batch_size, 1, trace_length]
        x = self.local_conv(x)  # Apply locally connected layer
        x = self.bn_local(x)
        x = x.transpose(1, 2)  # [batch_size, new_seq_len, 1]
        
        # Reshape to [batch_size, new_seq_len, 2]
        x = x.view(batch_size, -1, 1)
        
        # Forward LSTM
        fw_out, _ = self.fw_lstm(x)
        
        # Backward LSTM (reverse input)
        x_rev = torch.flip(x, dims=[1])
        bw_out, _ = self.bw_lstm(x_rev)
        bw_out = torch.flip(bw_out, dims=[1])  # Reverse output back
        
        # Apply attention
        fw_context = self.fw_attention(fw_out)
        bw_context = self.bw_attention(bw_out)
        
        # Concatenate forward and backward representations
        fb_represent = torch.cat([fw_context, bw_context], dim=1)
        
        # Output probabilities
        output = self.fc_out(fb_represent)
        output = self.bn_out(output)
        output = self.softmax(output)
        
        return output

# Usage example:
# model = KerasModelEquivalent(trace_length=your_trace_length, units=your_units)
# output = model(input_tensor)

# 模型定义
class LSTMModel(nn.Module):
    def __init__(self, trace_length, units):
        super().__init__()
        
        # 卷积编码器部分
        self.conv_encoder = nn.Sequential(
            nn.ConstantPad1d((40, 0), 0),
            nn.Conv1d(1, 4, kernel_size=6, stride=1),
            nn.BatchNorm1d(4),
            nn.ELU(),
            nn.MaxPool1d(2, stride=2),
            
            nn.Conv1d(4, 8, kernel_size=3, stride=1),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.MaxPool1d(2, stride=2),
            
            nn.Conv1d(8, 16, kernel_size=3, stride=1),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.MaxPool1d(2, stride=2),
            
            nn.Conv1d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(2, stride=2),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2, stride=2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(2, stride=2)
        )
        
        # 修改LSTM部分，添加dropout参数
        self.fw_lstm = nn.LSTM(input_size=128, hidden_size=units, batch_first=True, num_layers=2, dropout=0.2)
        self.bw_lstm = nn.LSTM(input_size=128, hidden_size=units, batch_first=True, num_layers=2, dropout=0.2)
        
        # 修改注意力机制部分
        # 重新计算序列长度，确保与卷积输出匹配
        # 原计算方式：seq_len = trace_length // 64
        # 改为手动计算卷积后的输出尺寸
        test_input = torch.randn(1, 1, trace_length)
        conv_out = self.conv_encoder(test_input)
        seq_len = conv_out.shape[2]  # 获取实际的卷积输出长度
        
        # 修改注意力机制部分
        self.fw_attention = nn.Sequential(
            nn.Linear(units, 1),
            nn.Flatten(),
            nn.BatchNorm1d(seq_len),  # 使用实际计算出的序列长度
            nn.Softmax(dim=1)
        )
        
        self.bw_attention = nn.Sequential(
            nn.Linear(units, 1),
            nn.Flatten(),
            nn.BatchNorm1d(seq_len),  # 使用实际计算出的序列长度
            nn.Softmax(dim=1)
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(units*2, 256),
            nn.BatchNorm1d(256),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # 卷积编码
        x = self.conv_encoder(x)
        
        # 调整维度
        x = x.permute(0, 2, 1)
        
        # LSTM部分
        fw_out, _ = self.fw_lstm(x)
        bw_out, _ = self.bw_lstm(torch.flip(x, dims=[1]))
        bw_out = torch.flip(bw_out, dims=[1])
        
        # # 添加dropout层 (新增部分)
        # dropout = nn.Dropout(p=0.05)  # 保留概率设为0.5
        # fw_out = dropout(fw_out)
        # bw_out = dropout(bw_out)
        
        # 注意力机制
        fw_att = self.fw_attention(fw_out)
        bw_att = self.bw_attention(bw_out)
        
        # 加权求和
        fw_rep = (fw_out * fw_att.unsqueeze(2)).sum(1)
        bw_rep = (bw_out * bw_att.unsqueeze(2)).sum(1)
        
        # 确保维度匹配
        assert fw_out.size(1) == self.fw_attention[2].num_features, \
            f"维度不匹配: 输入{fw_out.size(1)}, 期望{self.fw_attention[2].num_features}"
        
        # 合并特征
        features = torch.cat([fw_rep, bw_rep], dim=1)
        return self.output_layer(features)

def get_plaintext(test_index, dataset_name):
    zarr_store = zarr.DirectoryStore(dataset_name)
    zarr_group = zarr.open(zarr_store, mode='r')
    
    plaintexts = np.array(zarr_group['0/0/plaintext'])
    length = len(test_index)
    plain_text_need = np.zeros(length)
    
    count = 0
    for i in test_index:
        
        plain_text = plaintexts[i]
        plain_text_need[count] = plain_text[byte_index]
        
        count += 1
    return plain_text_need

# 计算Guessing Entropy
def calc_GE(inter_value_pro, key_suppose):
    sorted_pro = np.sort(inter_value_pro)
    sorted_index = np.argsort(inter_value_pro)
    posi_of_key = np.where(sorted_index==key_suppose)[0]
    entropy = 256 - posi_of_key
    return entropy, sorted_pro, sorted_index

# 自定义回调函数
class SaveModelCallback:
    def __init__(self, model, test_loader, params):
        self.model = model
        self.test_loader = test_loader
        self.params = params
        
    def on_epoch_end(self, epoch):
        if (epoch+1) % epochs_per_save == 0:
            print(f'saving model of epoch {epoch+1}')
            torch.save(self.model.state_dict(), f'E:\\models\\byte{byte_index}.pt')
            
            # 评估模型
            self.model.eval()
            with torch.no_grad():
                # 预测和评估逻辑
                predict_sout = []
                labels_list = []
                
                for traces, labels in self.test_loader:
                    traces, labels = traces.to(device), labels.to(device)
                    outputs = self.model(traces)
                    predict_sout.append(outputs.cpu().numpy())  # Add .cpu() before .numpy()
                    labels_list.append(labels.cpu().numpy())     # Also fix for labels
                
                predict_sout = np.concatenate(predict_sout)
                labels_list = np.concatenate(labels_list)
                sout_array = predict_sout.argmax(axis=-1)
                
                key_list = []
                p_array = get_plaintext(test_index, dataset_name)
            
                for m in range(len(sout_array)):
                    
                    if m>510:
                        attack_step = attack_step_large
                    else :
                        attack_step = attack_step_small
                        
                    # sin = inv_sbox[sout_array[m]]
                    sin = Sbox[sout_array[m] ^ int(p_array[m])]
                    key_list.append(int(sin) ^ int(p_array[m]))
                    if m%attack_step == 0:
                        print('attack_trace:',m)
                        key_counter = Counter(np.asarray(key_list))
                        print(key_counter.most_common(10))
                print('attack_trace:',m)
                key_counter= Counter(np.asarray(key_list))
                print(key_counter.most_common(10))
                
                inter_value_pro = np.zeros(256)
                pic_GE = np.zeros(pic_num)
                
                for j in range(len(sout_array)):
                    
                    if j>510:
                        attack_step = attack_step_large
                    else :
                        attack_step = attack_step_small
                        
                    for key in range(256):
                        # inter_value = int(Sbox[int(int(p_array[j])^int(key))])
                        inter_value =  int(inv_sbox[int(int(p_array[j])^int(key))]^int(p_array[j]))
                        inter_value_pro[key] += np.log(predict_sout[j][inter_value])
                    
                    if j<pic_num:
                        entropy, sorted_pro, sorted_index = calc_GE(inter_value_pro, key_suppose)
                        pic_GE[j] = entropy
                        if j%attack_step == 0:
                            print('attack_trace:',j)
                            print('entropy:', entropy)
                            for k in range(255,245,-1):
                                print(sorted_pro[k], sorted_index[k]) 
                        
                    elif j%attack_step == 0:
                        print('attack_trace:',j)
                        entropy, sorted_pro, sorted_index = calc_GE(inter_value_pro, key_suppose)
                        print('entropy:', entropy)
                        for k in range(255,245,-1):
                            print(sorted_pro[k], sorted_index[k]) 
                print('attack_trace:',j)            
                entropy, sorted_pro, sorted_index = calc_GE(inter_value_pro, key_suppose)
                print('entropy:', entropy)
                for k in range(255,245,-1):
                    print(sorted_pro[k], sorted_index[k])

                # print('GE_curve_100:')
                
                # # 绘制GE曲线
                # plt.figure(dpi=60)
                # plt.plot(pic_GE)
                # plt.show()

# 训练配置
trace_offset = 500
trace_length = 2000
# shift_scale = 50
# augment_shift_scale = 40
units = 128
batch_size = 200
epochs_per_save = 5
total_epoch = 5
epoch_offset = 0

byte_index = 3
# key_array = [0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0x00,0xaa,0xbb,0xcc,0xdd,0xee,0xff]
# key_array = [0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6, 0xAB, 0xF7, 0x15, 0x88, 0x09, 0xCF, 0x4F, 0x3C]
key_array = [0xd0, 0x14, 0xf9, 0xa8, 0xc9, 0xee, 0x25, 0x89, 0xe1, 0x3f, 0xc, 0xc8, 0xb6, 0x63, 0xc, 0xa6]
key_suppose = key_array[byte_index]

train_index = np.arange(0,200000)
test_index = np.arange(0,50000)
train_num = 200000
test_num = 50000
pic_num = 5000

attack_step_large = 5000
attack_step_small = 500

# dataset_name = '/mnt/20250718073027.zarr'
dataset_name = 'E:\\codes\\Acquisition\\dataset\\merged.zarr'
# S盒定义保持不变
inv_sbox = [82,9,106,213,48,54,165,56,191,64,163,158,129,243,215,251,124,227,57,130,155,47,255,135,52,142,67,68,196,222,233,203,84,123,148,50,166,194,35,61,238,76,149,11,66,250,195,78,8,46,161,102,40,217,36,178,118,91,162,73,109,139,209,37,114,248,246,100,134,104,152,22,212,164,92,204,93,101,182,146,108,112,72,80,253,237,185,218,94,21,70,87,167,141,157,132,144,216,171,0,140,188,211,10,247,228,88,5,184,179,69,6,208,44,30,143,202,63,15,2,193,175,189,3,1,19,138,107,58,145,17,65,79,103,220,234,151,242,207,206,240,180,230,115,150,172,116,34,231,173,53,133,226,249,55,232,28,117,223,110,71,241,26,113,29,41,197,137,111,183,98,14,170,24,190,27,252,86,62,75,198,210,121,32,154,219,192,254,120,205,90,244,31,221,168,51,136,7,199,49,177,18,16,89,39,128,236,95,96,81,127,169,25,181,74,13,45,229,122,159,147,201,156,239,160,224,59,77,174,42,245,176,200,235,187,60,131,83,153,97,23,43,4,126,186,119,214,38,225,105,20,99,85,33,12,125]
Sbox = [99,124,119,123,242,107,111,197,48,1,103,43,254,215,171,118,202,130,201,125,250,89,71,240,173,212,162,175,156,164,114,192,183,253,147,38,54,63,247,204,52,165,229,241,113,216,49,21,4,199,35,195,24,150,5,154,7,18,128,226,235,39, 178,117,9,131,44,26,27,110,90,160,82,59,214,179,41,227,47,132,83,209,0,237,32,252,177,91,106,203,190,57,74,76,88, 207,208,239,170,251,67,77,51,133,69,249,2,127,80,60,159,168,81,163,64,143,146,157,56,245,188,182,218,33,16,255, 243,210,205,12,19,236,95,151,68,23,196,167,126,61,100,93,25,115,96,129,79,220,34,42,144,136,70,238,184,20,222,94,11,219,224,50,58,10,73,6,36,92,194,211,172,98,145,149,228,121,231,200,55,109,141,213,78,169,108,86,244,234,101, 122,174,8,186,120,37,46,28,166,180,198,232,221,116,31,75,189,139,138,112,62,181,102,72,3,246,14,97,53,87,185,134, 193,29,158,225,248,152,17,105,217,142,148,155,30,135,233,206,85,40,223,140,161,137,13,191,230,66,104,65,153,45, 15,176,84,187,22]



# 初始化模型和优化器
# model = LSTMModel(trace_length, units)
model = AESHDModel(trace_length, units)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# 数据加载器
params_train = {
    'trs_file_path': dataset_name,
    'trace_offset': trace_offset,
    'trace_length': trace_length,
    'byte_index': byte_index
}

params_valid = {
    'trs_file_path': dataset_name,
    'trace_offset': trace_offset,
    'trace_length': trace_length,
    'byte_index': byte_index
}




if __name__ == "__main__":

    # train_dataset = TraceDataset(train_index, params_train)
    train_dataset = InMemoryTraceDataset(train_index, params_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # test_dataset = TraceDataset(test_index, params_valid)
    test_dataset = InMemoryTraceDataset(test_index, params_valid)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 训练循环
    save_callback = SaveModelCallback(model, test_loader, params_valid)
    for epoch in range(total_epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 在模型初始化后添加
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"当前使用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")  # 添加设备检测输出
        # if torch.cuda.is_available():
        #     print(f"GPU设备名称: {torch.cuda.get_device_name(0)}")
        model = model.to(device)
        
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epoch}', leave=True)
        
        # 在训练循环中修改数据加载部分
        for traces, labels in train_loader_tqdm:
            traces, labels = traces.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(traces)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()

            train_loader_tqdm.set_postfix({
                'Loss': f'{running_loss/(train_loader_tqdm.n+1):.4f}',
                'Acc': f'{100 * correct/total:.2f}%'
            })
        
        # 打印训练统计信息
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{total_epoch}, Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        
        # 回调函数
        save_callback.on_epoch_end(epoch)
