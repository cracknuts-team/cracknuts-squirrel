# CrackNuts Squirrel

## Installation

```shell
pip install cracknuts-squirrel
```

## For Developers

In the virtual environment, run the following command to install the dependencies:

```bash
pip install -e .
pip install -r requirements-dev.txt
```

## 文件功能说明

### lstm_aes_hd.py

该文件实现了基于LSTM的AES汉明距离（Hamming Distance）模型，用于侧信道分析攻击。主要功能包括：

- 定义了AESHDModel神经网络模型，结合了卷积层、双向LSTM层和注意力机制
- 实现了自定义数据集类（TraceDataset和InMemoryTraceDataset），用于加载和处理侧信道轨迹数据
- 包含模型训练逻辑，使用PyTorch框架进行训练
- 实现了Guessing Entropy（猜测熵）计算函数，用于评估攻击效果
- 提供了保存模型回调函数，在训练过程中定期保存模型并进行评估

### test_lstm_model.py

该文件用于测试训练好的LSTM模型，执行AES密钥恢复攻击。主要功能包括：

- 加载训练好的模型文件（.pt格式）
- 从zarr文件中加载测试数据（轨迹和明文）
- 执行攻击过程，使用模型预测结果进行密钥猜测
- 计算并显示猜测熵，评估攻击效果
- 显示最可能的密钥候选值，并与实际密钥进行比较

这两个文件共同构成了一个完整的AES侧信道分析工具链，从模型训练到攻击测试。
