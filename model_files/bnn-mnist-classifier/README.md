# bnn-mnist-classifier


    貝葉斯神經網絡模型用於MNIST手寫數字分類。
    該模型使用變分推斷來學習參數的不確定性，並能夠提供預測的不確定性估計。
    
    特點：
    - 使用貝葉斯神經網絡架構
    - 輸入：784維特徵（28x28 MNIST圖像）
    - 輸出：10個數字類別（0-9）
    - 支持不確定性估計
    

## 模型架構

- 輸入層：784 神經元（28x28 MNIST 圖像）
- 隱藏層1：400 神經元
- 隱藏層2：400 神經元
- 輸出層：10 神經元（數字 0-9）

## 貝葉斯特性

- 使用變分推斷進行參數估計
- 權重和偏置都服從高斯分佈
- 支持不確定性估計
- 使用蒙特卡羅採樣進行預測

## 使用方法

```python
from bayesian_neural_net import BayesianMLP
import torch

# 加載模型
model = BayesianMLP()
model.load_state_dict(torch.load("pytorch_model.bin"))

# 進行預測
x = torch.randn(1, 784)  # 示例輸入
probs, uncertainty = mc_predict(model, x, mc_runs=50)
```

## 訓練細節

- 優化器：Adam
- 學習率：1e-3
- 批次大小：128
- 訓練輪次：10
- 損失函數：證據下界（ELBO）
