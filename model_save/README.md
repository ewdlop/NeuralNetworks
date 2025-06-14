# spiking-neural-network-nmnist

這是一個使用 PyTorch 實現的脈衝神經網路模型，用於處理 NMNIST 數據集。

## 模型架構

- 輸入層：2312 個神經元
- 隱藏層：100 個神經元
- 輸出層：10 個神經元
- 時間窗口：20 個時間步

## 模型特點

- 使用積分發放神經元（Integrate-and-Fire Neurons）
- 實現了生物神經元的基本特性：膜電位積分、閾值觸發和脈衝發放
- 使用代理梯度方法進行訓練
- 能夠處理時序數據
- 具有生物學合理性
- 能量效率高（只在必要時發放脈衝）

## 使用方法

```python
import torch
from spiking_neural_net import SimpleSpikingNeuralNetwork

# 加載模型
model = SimpleSpikingNeuralNetwork(
    input_size=2312,
    hidden_size=100,
    output_size=10,
    time_window=20
)
model.load_state_dict(torch.load("model.pt"))

# 使用模型進行預測
model.eval()
with torch.no_grad():
    predictions = model(input_data)
```

## 數據集

本模型使用 NMNIST 數據集進行訓練，這是一個基於事件的神經形態數據集，由 MNIST 數字通過事件相機記錄而成。

## 訓練參數

- 優化器：Adam
- 學習率：1e-3
- 批次大小：64
- 訓練輪數：5
