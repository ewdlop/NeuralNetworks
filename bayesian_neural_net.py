#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
貝葉斯神經網絡 (貝葉斯反向傳播) - MNIST 手寫數字識別示例
-------------------------------------------------------
本程式實現了基於變分推斷的貝葉斯神經網絡，用於MNIST數據集的分類任務。

主要特點：
1. 使用變分推斷來近似後驗分佈
2. 實現了KL散度正則化的證據下界損失函數
3. 使用蒙特卡羅隨機採樣進行不確定性估計
4. 支持參數的隨機採樣和確定性預測

數學原理：
1. 變分推斷
   後驗分佈 p(θ|D) 通過變分分佈 q(θ) 近似：
   q(θ) = N(θ|μ, σ²)
   其中 μ 和 σ² 是可學習的參數

2. 證據下界 (ELBO)
   ELBO = E_q[log p(D|θ)] - KL(q(θ)||p(θ))
   其中：
   - E_q[log p(D|θ)] 是負對數似然
   - KL(q(θ)||p(θ)) 是變分分佈與先驗的KL散度

3. 重參數化技巧
   θ ~ q(θ) 等價於：
   θ = μ + σ * ε, 其中 ε ~ N(0,1)

4. 蒙特卡羅積分
   期望值通過採樣近似：
   E_q[f(θ)] ≈ 1/N Σᵢ f(θᵢ), θᵢ ~ q(θ)

依賴套件：
- PyTorch: 深度學習框架
- TorchVision: 用於數據加載和預處理
"""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torchvision import datasets, transforms


# -------------------------------------------------------
# 1. 貝葉斯線性層
# -------------------------------------------------------
class BayesianLinear(nn.Module):
    """
    貝葉斯線性層實現
    
    數學表示：
    1. 先驗分佈：p(w) = N(w|μ₀, σ₀²)
    2. 變分後驗：q(w) = N(w|μ, σ²)
    3. 重參數化：w = μ + σ * ε, ε ~ N(0,1)
    
    參數：
    - 輸入特徵數: 輸入特徵的維度
    - 輸出特徵數: 輸出特徵的維度
    - 先驗均值: 先驗分佈的均值 μ₀
    - 先驗標準差: 先驗分佈的標準差 σ₀
    """
    def __init__(self, 輸入特徵數, 輸出特徵數,
                 先驗均值: float = 0.0,
                 先驗標準差: float = 1.0):
        super().__init__()

        # 變分後驗參數：均值和對數標準差
        # μ 和 log(σ) 是可學習的參數
        self.權重均值 = nn.Parameter(torch.empty(輸出特徵數, 輸入特徵數))
        self.權重對數標準差 = nn.Parameter(torch.empty(輸出特徵數, 輸入特徵數))
        self.偏置均值 = nn.Parameter(torch.empty(輸出特徵數))
        self.偏置對數標準差 = nn.Parameter(torch.empty(輸出特徵數))

        # 初始化參數
        # 使用小的高斯分佈初始化均值，確保初始預測不確定性較大
        nn.init.normal_(self.權重均值, 0, 0.1)
        nn.init.normal_(self.偏置均值, 0, 0.1)
        # 初始化對數標準差為較小的負值，確保初始不確定性較小
        nn.init.constant_(self.權重對數標準差, -3.0)
        nn.init.constant_(self.偏置對數標準差, -3.0)

        # 設置先驗分佈：p(w) = N(w|μ₀, σ₀²)
        self.先驗分佈 = Normal(torch.tensor(先驗均值), torch.tensor(先驗標準差))

    @staticmethod
    def _計算標準差(對數標準差: torch.Tensor) -> torch.Tensor:
        """
        使用softplus函數將對數標準差轉換為標準差
        σ = log(1 + exp(ρ))
        確保標準差始終為正
        """
        return torch.log1p(torch.exp(對數標準差))

    def _採樣(self, 均值: torch.Tensor, 對數標準差: torch.Tensor) -> torch.Tensor:
        """
        使用重參數化技巧進行採樣
        w = μ + σ * ε, 其中 ε ~ N(0,1)
        """
        隨機噪聲 = torch.randn_like(均值)
        return 均值 + self._計算標準差(對數標準差) * 隨機噪聲

    def forward(self, 輸入: torch.Tensor, 是否採樣: bool = True) -> torch.Tensor:
        """
        前向傳播
        
        數學表示：
        1. 訓練時：y = (μ + σ * ε) * x + (μ_b + σ_b * ε_b)
        2. 推理時：y = μ * x + μ_b
        
        參數：
        - 輸入: 輸入張量 x
        - 是否採樣: 是否進行隨機採樣
        
        返回：
        - 線性變換後的結果 y
        """
        if self.training or 是否採樣:
            # 訓練時或需要採樣時，使用隨機權重
            權重 = self._採樣(self.權重均值, self.權重對數標準差)
            偏置 = self._採樣(self.偏置均值, self.偏置對數標準差)
        else:
            # 推理時使用均值
            權重, 偏置 = self.權重均值, self.偏置均值
        return F.linear(輸入, 權重, 偏置)

    def 計算KL散度(self) -> torch.Tensor:
        """
        計算與先驗分佈的KL散度
        
        數學表示：
        KL(q(w)||p(w)) = ∫ q(w) * log(q(w)/p(w)) dw
        對於高斯分佈，有解析解：
        KL(N(μ,σ²)||N(μ₀,σ₀²)) = 0.5 * [log(σ₀²/σ²) + (σ² + (μ-μ₀)²)/σ₀² - 1]
        
        返回：
        - 權重和偏置的KL散度之和
        """
        權重分佈 = Normal(self.權重均值, self._計算標準差(self.權重對數標準差))
        偏置分佈 = Normal(self.偏置均值, self._計算標準差(self.偏置對數標準差))
        return kl_divergence(權重分佈, self.先驗分佈).sum() + kl_divergence(偏置分佈, self.先驗分佈).sum()


# -------------------------------------------------------
# 2. 貝葉斯多層感知機
# -------------------------------------------------------
class BayesianMLP(nn.Module):
    """
    貝葉斯多層感知機
    
    網絡結構：
    1. 輸入層：784 (28x28)
    2. 隱藏層1：400
    3. 隱藏層2：400
    4. 輸出層：10 (數字0-9)
    
    數學表示：
    1. 輸入層：x ∈ ℝ^{784}
    2. 隱藏層1：h₁ = ReLU(W₁x + b₁), W₁ ∈ ℝ^{400×784}
    3. 隱藏層2：h₂ = ReLU(W₂h₁ + b₂), W₂ ∈ ℝ^{400×400}
    4. 輸出層：y = W₃h₂ + b₃, W₃ ∈ ℝ^{10×400}
    """
    def __init__(self):
        super().__init__()
        self.第一層 = BayesianLinear(28 * 28, 400)
        self.第二層 = BayesianLinear(400, 400)
        self.第三層 = BayesianLinear(400, 10)

    def forward(self, 輸入: torch.Tensor, 是否採樣: bool = True) -> torch.Tensor:
        """
        前向傳播
        
        數學表示：
        h₁ = ReLU(W₁x + b₁)
        h₂ = ReLU(W₂h₁ + b₂)
        y = W₃h₂ + b₃
        
        參數：
        - 輸入: 輸入圖像 x
        - 是否採樣: 是否進行隨機採樣
        
        返回：
        - 分類輸出 y
        """
        輸入 = 輸入.view(輸入.size(0), -1)  # 展平圖像
        輸入 = F.relu(self.第一層(輸入, 是否採樣))
        輸入 = F.relu(self.第二層(輸入, 是否採樣))
        return self.第三層(輸入, 是否採樣)

    def 計算KL散度(self) -> torch.Tensor:
        """
        計算整個網絡的KL散度
        
        數學表示：
        KL_total = KL₁ + KL₂ + KL₃
        其中 KLᵢ 是第i層的KL散度
        """
        return self.第一層.計算KL散度() + self.第二層.計算KL散度() + self.第三層.計算KL散度()


# -------------------------------------------------------
# 3. 證據下界損失函數
# -------------------------------------------------------
def 計算證據下界損失(模型: BayesianMLP, 輸出: torch.Tensor,
                 目標: torch.Tensor, 權重係數: float) -> tuple[torch.Tensor,
                                                           torch.Tensor,
                                                           torch.Tensor]:
    """
    計算證據下界 (Evidence Lower BOund) 損失
    
    數學表示：
    ELBO = E_q[log p(D|θ)] - β * KL(q(θ)||p(θ))
    其中：
    - E_q[log p(D|θ)] 是負對數似然
    - KL(q(θ)||p(θ)) 是變分分佈與先驗的KL散度
    - β 是KL散度的權重係數（用於KL退火）
    
    參數：
    - 模型: 貝葉斯模型
    - 輸出: 模型輸出 logits
    - 目標: 目標標籤 y
    - 權重係數: KL散度的權重係數 β
    
    返回：
    - 總損失: ELBO
    - 負對數似然: -log p(D|θ)
    - KL散度: KL(q(θ)||p(θ))
    """
    負對數似然 = F.cross_entropy(輸出, 目標, reduction='mean')  # 負對數似然
    KL散度 = 模型.計算KL散度() / 目標.size(0)   # 每個樣本的平均KL散度
    return 負對數似然 + 權重係數 * KL散度, 負對數似然, KL散度


# -------------------------------------------------------
# 4. 蒙特卡羅隨機採樣推論
# -------------------------------------------------------
@torch.no_grad()
def 蒙特卡羅預測(模型: BayesianMLP, 輸入: torch.Tensor,
              採樣次數: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    """
    使用蒙特卡羅隨機採樣進行預測
    
    數學表示：
    1. 採樣T次：p(y|x) ≈ 1/T Σᵗ p(y|x,θᵗ), θᵗ ~ q(θ)
    2. 計算均值：E[p(y|x)] ≈ 1/T Σᵗ p(y|x,θᵗ)
    3. 計算方差：Var[p(y|x)] ≈ 1/T Σᵗ (p(y|x,θᵗ) - E[p(y|x)])²
    
    參數：
    - 模型: 貝葉斯模型
    - 輸入: 輸入數據 x
    - 採樣次數: 蒙特卡羅採樣次數 T
    
    返回：
    - 預測概率的均值 E[p(y|x)]
    - 預測的不確定性 Var[p(y|x)]
    """
    模型.eval()
    預測結果 = torch.stack([模型(輸入, 是否採樣=True).softmax(-1)
                         for _ in range(採樣次數)])
    概率均值 = 預測結果.mean(dim=0)  # 平均預測概率
    不確定性 = 預測結果.var(dim=0)   # 預測的不確定性
    return 概率均值, 不確定性


# -------------------------------------------------------
# 5. 訓練與測試函數
# -------------------------------------------------------
def 訓練模型(模型: BayesianMLP, 數據加載器, 優化器, 設備, 當前輪次, 總輪次數):
    """
    訓練一個輪次
    
    數學表示：
    1. 對於每個批次 (x,y)：
       - 計算ELBO損失：L = -log p(y|x,θ) + β * KL(q(θ)||p(θ))
       - 更新參數：θ ← θ - η * ∇L
    2. KL退火：β = min(1.0, epoch/n_epochs)
    
    參數：
    - 模型: 貝葉斯模型
    - 數據加載器: 數據加載器
    - 優化器: 優化器
    - 設備: 計算設備
    - 當前輪次: 當前訓練輪次
    - 總輪次數: 總訓練輪次數
    """
    模型.train()
    for 步驟, (輸入, 標籤) in enumerate(數據加載器, 1):
        輸入, 標籤 = 輸入.to(設備), 標籤.to(設備)
        # KL散度退火：從0線性增加到1
        權重係數 = min(1.0, 當前輪次 / 總輪次數)
        輸出 = 模型(輸入, 是否採樣=True)
        總損失, 負對數似然, KL散度 = 計算證據下界損失(模型, 輸出, 標籤, 權重係數)

        優化器.zero_grad()
        總損失.backward()
        優化器.step()

        if 步驟 % 100 == 0 or 步驟 == len(數據加載器):
            print(f'  [訓練] 輪次{當前輪次:02d} {步驟:4d}/{len(數據加載器)} '
                  f'損失={總損失.item():.4f} '
                  f'負對數似然={負對數似然.item():.4f} KL散度={KL散度.item():.4f}')


def 測試模型(模型: BayesianMLP, 數據加載器, 設備):
    """
    測試模型性能
    
    數學表示：
    1. 對於每個批次 (x,y)：
       - 使用蒙特卡羅採樣計算預測：p(y|x) ≈ 1/T Σᵗ p(y|x,θᵗ)
       - 計算準確率：acc = 1/N Σᵢ I(argmax p(yᵢ|xᵢ) == yᵢ)
    
    參數：
    - 模型: 貝葉斯模型
    - 數據加載器: 測試數據加載器
    - 設備: 計算設備
    
    返回：
    - 測試準確率
    """
    模型.eval()
    總數, 正確數 = 0, 0
    with torch.no_grad():
        for 輸入, 標籤 in 數據加載器:
            輸入, 標籤 = 輸入.to(設備), 標籤.to(設備)
            輸出, _ = 蒙特卡羅預測(模型, 輸入, 採樣次數=20)  # 使用較少的採樣次數
            預測 = 輸出.argmax(dim=1)
            總數 += 標籤.size(0)
            正確數 += (預測 == 標籤).sum().item()
    準確率 = 正確數 / 總數
    print(f'  [測試] 準確率 = {準確率 * 100:.2f}%')
    return 準確率


# -------------------------------------------------------
# 6. 主程序
# -------------------------------------------------------
def main():
    """
    主程序：設置參數、加載數據、訓練模型、保存結果
    
    數學表示：
    1. 數據預處理：x ∈ [0,1]^{784}
    2. 模型訓練：最小化ELBO損失
    3. 模型評估：使用蒙特卡羅採樣計算預測和不確定性
    """
    parser = argparse.ArgumentParser(
        description='貝葉斯神經網絡 (貝葉斯反向傳播) MNIST手寫數字識別')
    parser.add_argument('--輪次數', type=int, default=10, help='訓練迭代次數')
    parser.add_argument('--批次大小', type=int, default=128, help='批次大小')
    parser.add_argument('--學習率', type=float, default=1e-3, help='學習率')
    parser.add_argument('--採樣次數', type=int, default=50,
                        help='推理時蒙特卡羅採樣次數')
    parser.add_argument('--保存路徑', type=str, default='bnn_mnist.pt',
                        help='模型保存路徑')
    args = parser.parse_args()

    設備 = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'使用設備：{設備}')

    # 加載MNIST數據集
    數據轉換 = transforms.ToTensor()
    Path('data').mkdir(exist_ok=True)
    訓練數據加載器 = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=數據轉換),
        batch_size=args.批次大小, shuffle=True)
    測試數據加載器 = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=數據轉換),
        batch_size=1024)

    # 創建模型和優化器
    模型 = BayesianMLP().to(設備)
    優化器 = torch.optim.Adam(模型.parameters(), lr=args.學習率)

    # 訓練模型
    for 輪次 in range(1, args.輪次數 + 1):
        訓練模型(模型, 訓練數據加載器, 優化器, 設備, 輪次, args.輪次數)
        測試模型(模型, 測試數據加載器, 設備)

    # 保存模型
    torch.save(模型.state_dict(), args.保存路徑)
    print(f'模型已保存至 {args.保存路徑}')

    # 推理示例：隨機選擇一批測試數據
    樣本輸入, _ = next(iter(測試數據加載器))
    樣本輸入 = 樣本輸入[:16].to(設備)
    預測概率, 不確定性 = 蒙特卡羅預測(模型, 樣本輸入, 採樣次數=args.採樣次數)

    print('\n前16個樣本的預測概率和不確定性（最大方差）：')
    for i, (概率, 方差) in enumerate(zip(預測概率, 不確定性)):
        print(f'  樣本{i:02d}  預測={概率.argmax().item()}  '
              f'最大概率={概率.max():.2f}  不確定性={方差.max():.4f}')


if __name__ == '__main__':
    main()
