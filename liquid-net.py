import torch
import torch.nn as nn
import torch.nn.functional as F

class LiquidNeuron(nn.Module):
    """
    液體神經元（Liquid Neuron）
    
    這是一種模仿生物神經元動態行為的神經元模型，
    其核心思想是引入可學習的時間常數（tau）與非線性調變參數（alpha, beta），
    使神經元能夠根據輸入與自身狀態動態調整。
    
    參數:
        input_size (int): 輸入特徵維度
        hidden_size (int): 隱藏狀態維度
    屬性:
        W_in: 輸入權重參數，形狀為 (hidden_size, input_size)
        W_rec: 循環（遞迴）權重參數，形狀為 (hidden_size, hidden_size)
        tau: 時間常數，控制狀態更新速度（可學習）
        alpha, beta: 非線性調變參數
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 輸入權重
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size))
        
        # 循環權重（遞迴連接）
        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size))
        
        # 時間常數（可學習），決定狀態變化的快慢
        self.tau = nn.Parameter(torch.ones(hidden_size) * 0.5)
        
        # 非線性調變參數
        self.alpha = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h):
        """
        前向傳播
        
        參數:
            x: (batch, input_size) 當前時間步的輸入
            h: (batch, hidden_size) 前一時間步的隱藏狀態
        流程:
            1. 輸入與遞迴加權求和
            2. 通過非線性激活（tanh）與調變
            3. 狀態更新方程：dh = (-h + alpha * tanh(u + beta)) / tau
            4. 新狀態 h_new = h + dh
        返回:
            h_new: (batch, hidden_size) 更新後的隱藏狀態
        """
        u = F.linear(x, self.W_in) + F.linear(h, self.W_rec)
        dh = (-h + self.alpha * torch.tanh(u + self.beta)) / self.tau
        h_new = h + dh
        return h_new

class LiquidLayer(nn.Module):
    """
    液體層（Liquid Layer）
    
    由多個 LiquidNeuron 組成，負責處理整個序列資料。
    每個時間步都會呼叫 LiquidNeuron 更新狀態。
    
    參數:
        input_size (int): 輸入特徵維度
        hidden_size (int): 隱藏狀態維度
        steps (int): 時間步長（序列長度）
    """
    def __init__(self, input_size, hidden_size, steps):
        super().__init__()
        self.cell = LiquidNeuron(input_size, hidden_size)
        self.steps = steps

    def forward(self, x):
        """
        前向傳播
        
        參數:
            x: (batch, steps, input_size) 輸入序列
        流程:
            1. 初始化隱藏狀態 h 為 0
            2. 依序處理每個時間步，更新 h
            3. 收集每個時間步的輸出
        返回:
            outputs: (batch, steps, hidden_size) 每個時間步的隱藏狀態序列
        """
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.cell.hidden_size, device=x.device) # 初始化隱藏狀態 h 為 0 
        outputs = []

        for t in range(seq_len):
            h = self.cell(x[:, t], h) # 更新隱藏狀態 h ，x[:, t] 是第 t 個時間步的輸入
            outputs.append(h.unsqueeze(1)) # unsqueeze(1) 是將隱藏狀態 h 添加到 outputs 列表中 ，dim=1 是將隱藏狀態 h 添加到 outputs 列表中 
        return torch.cat(outputs, dim=1)  # (batch, steps, hidden_size)

class LiquidNet(nn.Module):
    """
    液體神經網路（Liquid Neural Network）
    
    這是一個序列建模網路，前端為 LiquidLayer，後端為全連接層。
    適合處理時序資料、動態信號等。
    
    參數:
        input_size (int): 輸入特徵維度
        hidden_size (int): 隱藏層維度
        output_size (int): 輸出特徵維度（例如分類數）
        steps (int): 時間步長
    """
    def __init__(self, input_size, hidden_size, output_size, steps):
        super().__init__()
        self.liquid = LiquidLayer(input_size, hidden_size, steps)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向傳播
        
        參數:
            x: (batch, steps, input_size) 輸入序列
        流程:
            1. 通過 LiquidLayer 得到每個時間步的隱藏狀態
            2. 取最後一個時間步的隱藏狀態作為序列表示
            3. 通過全連接層得到最終輸出
        返回:
            out: (batch, output_size) 輸出結果
        """
        out = self.liquid(x)       # (batch, steps, hidden)
        last = out[:, -1, :]       # [:, -1, :] 是使用最後一個時間步的輸出
        return self.fc(last)
