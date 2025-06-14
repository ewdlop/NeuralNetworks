import torch
from torch.utils.data import DataLoader
import tonic
import tonic.transforms as transforms
from torch import nn
from huggingface_hub import HfApi, create_repo
import os
import getpass

class IntegrateAndFireNeuron(nn.Module):
    """
    積分發放神經元類別
    實現了生物神經元的基本特性：膜電位積分、閾值觸發和脈衝發放
    
    工作原理：
    1. 積分階段：神經元持續積累輸入信號到膜電位
    2. 發放階段：當膜電位超過閾值時，產生脈衝輸出
    3. 重置階段：發放脈衝後，膜電位重置
    4. 衰減階段：膜電位隨時間自然衰減
    """
    def __init__(self, threshold=1.0, decay=0.9):
        """
        初始化積分發放神經元
        參數：
            threshold: 脈衝發放閾值，當膜電位超過此值時神經元發放脈衝
            decay: 膜電位衰減率，控制膜電位隨時間的衰減速度
                  較大的衰減率意味著膜電位衰減較慢，神經元對歷史輸入的記憶更持久
        """
        super(IntegrateAndFireNeuron, self).__init__()
        self.threshold = threshold
        self.decay = decay
        self.membrane_potential = None  # 膜電位，記錄神經元的內部狀態

    def reset(self):
        """重置神經元狀態"""
        self.membrane_potential = None

    def forward(self, x):
        """
        前向傳播過程
        參數：
            x: 輸入信號，形狀為 [批次大小, 特徵數]
        返回：
            神經元的脈衝輸出，形狀與輸入相同
        """
        # 如果膜電位未初始化，則初始化為零
        if self.membrane_potential is None:
            self.membrane_potential = torch.zeros_like(x, requires_grad=True)

        # 更新膜電位
        new_membrane_potential = self.decay * self.membrane_potential + x
        
        # 使用代理梯度方法計算輸出
        surrogate_grad = torch.sigmoid(10 * (new_membrane_potential - self.threshold))
        
        # 更新膜電位（避免原地操作）
        self.membrane_potential = new_membrane_potential * (1 - surrogate_grad)
        
        return surrogate_grad

class SimpleSpikingNeuralNetwork(nn.Module):
    """
    簡單脈衝神經網路類別
    實現了一個具有輸入層、隱藏層和輸出層的脈衝神經網路
    
    網路結構：
    1. 輸入層：接收原始輸入數據
    2. 第一個全連接層：將輸入轉換為隱藏層表示
    3. 第一個積分發放層：處理時間相關特徵
    4. 第二個全連接層：將隱藏層表示轉換為輸出層表示
    5. 第二個積分發放層：產生最終的脈衝輸出
    
    特點：
    - 能夠處理時序數據
    - 具有生物學合理性
    - 能量效率高（只在必要時發放脈衝）
    """
    def __init__(self, input_size, hidden_size, output_size, time_window=20):
        """
        初始化脈衝神經網路
        參數：
            input_size: 輸入層神經元數量，決定了輸入特徵的維度
            hidden_size: 隱藏層神經元數量，決定了網路的容量
            output_size: 輸出層神經元數量，通常等於分類任務的類別數
            time_window: 時間窗口大小，決定了網路處理的時間步數
        """
        super(SimpleSpikingNeuralNetwork, self).__init__()
        # 輸入層到隱藏層的全連接層
        self.first_fully_connected = nn.Linear(input_size, hidden_size) # 1000, 100
        # 隱藏層的積分發放神經元
        self.first_integrate_and_fire = IntegrateAndFireNeuron() # 100
        # 隱藏層到輸出層的全連接層
        self.second_fully_connected = nn.Linear(hidden_size, output_size) # 100, 10
        # 輸出層的積分發放神經元
        self.second_integrate_and_fire = IntegrateAndFireNeuron() # 10
        self.time_window = time_window # 20

    def reset(self):
        """重置網路狀態"""
        self.first_integrate_and_fire.reset()
        self.second_integrate_and_fire.reset()

    def forward(self, x):
        """
        前向傳播過程
        參數：
            x: 輸入數據，形狀為 [批次大小, 時間窗口, 輸入特徵數]
        返回：
            所有時間步的脈衝輸出，形狀為 [批次大小, 時間窗口, 輸出特徵數]
        """
        # 重置網路狀態
        self.reset()
        
        # 用於記錄所有時間步的脈衝輸出
        spike_record = []

        # 對每個時間步進行處理
        for time_step in range(self.time_window):
            # 獲取當前時間步的輸入
            current_input = x[:, time_step, :]
            
            # 通過第一個全連接層
            hidden_layer_output = self.first_fully_connected(current_input)
            # 通過第一個積分發放神經元
            first_spike_output = self.first_integrate_and_fire(hidden_layer_output)
            
            # 通過第二個全連接層
            output_layer_input = self.second_fully_connected(first_spike_output)
            # 通過第二個積分發放神經元
            second_spike_output = self.second_integrate_and_fire(output_layer_input)
            
            # 記錄當前時間步的輸出
            spike_record.append(second_spike_output)

        # 將所有時間步的輸出堆疊成一個張量
        # dim=1 表示在時間維度上堆疊
        return torch.stack(spike_record, dim=1)

torch.autograd.set_detect_anomaly(True)

# 時間窗參數（微秒）
# 這個參數決定了每個時間步的持續時間
time_window_us = 1000

# 轉換事件資料為 frames tensor
# 這個轉換過程將原始的事件數據轉換為更容易處理的幀格式
frame_transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),  # 去除時間間隔大於 10000 微秒的噪聲
    transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size,  # 將事件轉換為幀
                      time_window=time_window_us)
])

# 載入 NMNIST 數據集
# NMNIST 是一個基於事件的神經形態數據集，由 MNIST 數字通過事件相機記錄而成
# 每個樣本包含一系列的事件，這些事件記錄了數字圖像的變化
train_ds = tonic.datasets.NMNIST(save_to='./data', train=True, transform=frame_transform)
test_ds  = tonic.datasets.NMNIST(save_to='./data', train=False, transform=frame_transform)

# 使用磁盤緩存加速數據載入
# 這可以顯著提高訓練速度，特別是對於大型數據集
# 緩存會保存處理後的數據，避免重複處理
from tonic import DiskCachedDataset
train_ds = DiskCachedDataset(train_ds, cache_path='./cache/train')
test_ds = DiskCachedDataset(test_ds, cache_path='./cache/test')

# 創建數據加載器
# PadTensors 確保批次中的所有樣本具有相同的形狀
# 這對於批處理是必要的，因為不同樣本可能具有不同數量的時間步
train_loader = DataLoader(train_ds, batch_size=64, collate_fn=tonic.collation.PadTensors())
test_loader  = DataLoader(test_ds, batch_size=64, collate_fn=tonic.collation.PadTensors())

# 設置模型參數
# 計算輸入特徵數：通道數 * 高度 * 寬度
input_size = 2 * tonic.datasets.NMNIST.sensor_size[0] * tonic.datasets.NMNIST.sensor_size[1]  # 2 * 34 * 34 = 2312
hidden_size = 100  # 隱藏層神經元數量
output_size = 10   # 輸出類別數（數字 0-9）
time_window = time_window_us // 100  # 時間窗口大小，與輸入轉換保持一致

# 初始化模型並移至適當的設備（GPU/CPU）
model = SimpleSpikingNeuralNetwork(input_size, hidden_size, output_size, time_window)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 打印模型結構和參數數量
print(f"模型結構：")
print(f"輸入層：{input_size} 個特徵")
print(f"隱藏層：{hidden_size} 個神經元")
print(f"輸出層：{output_size} 個類別")
print(f"時間窗口：{time_window} 個時間步")

# 設置優化器和損失函數
# Adam 優化器結合了動量和自適應學習率的優點
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 自定義損失函數：計算預測類別與真實類別的不匹配率
def spike_loss(predictions, targets):
    """
    計算脈衝神經網路的損失
    
    參數：
        predictions: 模型輸出，形狀為 [批次大小, 時間步, 輸出類別數]
        targets: 目標類別，形狀為 [批次大小]
    
    返回：
        損失值（標量）
    """
    # 對時間維度求和，得到每個類別的總脈衝數
    spike_counts = predictions.sum(dim=1)  # [批次大小, 輸出類別數]
    
    # 確保目標標籤是 Long 類型
    targets = targets.long()
    
    # 使用交叉熵損失
    return torch.nn.functional.cross_entropy(spike_counts, targets)

# 訓練循環
# 每個 epoch 遍歷整個訓練數據集一次
for epoch in range(1, 6):
    model.train()  # 設置模型為訓練模式
    total_loss = 0.0
    for frames, targets in train_loader:
        # 打印數據形狀以進行調試
        print(f"Frames shape: {frames.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Targets dtype: {targets.dtype}")  # 打印目標標籤的數據類型
        
        # 根據實際形狀處理數據
        if len(frames.shape) == 5:  # [批次大小, 時間步, 通道數, 高度, 寬度]
            B, T, C, H, W = frames.shape
            # 將通道、高度和寬度展平為一個特徵向量
            x = frames.view(B, T, -1).to(device)
        elif len(frames.shape) == 4:  # [批次大小, 時間步, 高度, 寬度]
            B, T, H, W = frames.shape
            x = frames.view(B, T, -1).to(device)  # 將高度和寬度展平
        elif len(frames.shape) == 3:  # [批次大小, 時間步, 特徵數]
            B, T, F = frames.shape
            x = frames.to(device)
        else:
            raise ValueError(f"意外的數據形狀: {frames.shape}")
            
        y = targets.to(device)

        # 前向傳播和反向傳播
        optimizer.zero_grad()  # 清除之前的梯度
        out_spikes = model(x)  # 輸出形狀：[批次大小, 時間步, 輸出類別數]
        loss = spike_loss(out_spikes, y)  # 使用新的損失函數
        loss.backward()  # 反向傳播
        optimizer.step()  # 更新參數

        total_loss += loss.item()
    print(f"Epoch {epoch}, Training error: {total_loss/len(train_loader):.4f}")

# 評估模型
model.eval()  # 設置模型為評估模式
correct = 0
total = 0
with torch.no_grad():  # 禁用梯度計算
    for frames, targets in test_loader:
        # 根據實際形狀處理數據
        if len(frames.shape) == 5:  # [批次大小, 時間步, 通道數, 高度, 寬度]
            B, T, C, H, W = frames.shape
            # 將通道、高度和寬度展平為一個特徵向量
            x = frames.view(B, T, -1).to(device)
        elif len(frames.shape) == 4:  # [批次大小, 時間步, 高度, 寬度]
            B, T, H, W = frames.shape
            x = frames.view(B, T, -1).to(device)  # 將高度和寬度展平
        elif len(frames.shape) == 3:  # [批次大小, 時間步, 特徵數]
            B, T, F = frames.shape
            x = frames.to(device)
        else:
            raise ValueError(f"意外的數據形狀: {frames.shape}")
            
        y = targets.to(device)
        out = model(x)
        # 使用與訓練時相同的邏輯進行預測
        spike_counts = out.sum(dim=1)  # [批次大小, 輸出類別數]
        preds = spike_counts.argmax(dim=1)  # [批次大小]
        correct += (preds == y).sum().item()
        total += B

print(f"Test accuracy: {correct/total:.4f}")

# 保存模型到本地
# 只保存模型參數，不保存整個模型結構
torch.save(model.state_dict(), 'spiking_neural_network.pth')

# 推送到 Hugging Face Hub
def push_to_huggingface(model, model_name, username, token):
    """
    將模型推送到 Hugging Face Hub
    
    參數：
        model: 要保存的模型
        model_name: 模型名稱
        username: Hugging Face 用戶名
    
    步驟：
    1. 創建新的模型倉庫
    2. 保存模型到本地臨時目錄
    3. 上傳模型文件到 Hugging Face
    """
    # 創建倉庫
    repo_name = f"{username}/{model_name}"
    try:
        create_repo(repo_name, private=False)
    except Exception as e:
        print(f"倉庫可能已存在: {e}")

    # 保存模型
    save_dir = "model_save"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)

    # 上傳到 Hugging Face
    api = HfApi()
    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_name,
        repo_type="model",
        token = token
    )
    print(f"模型已成功上傳到 https://huggingface.co/{repo_name}")

# 使用示例
# 請替換為您的 Hugging Face 用戶名
username = "ewdlop"
model_name = "spiking-neural-network-nmnist"

token = getpass.getpass()

if not token:
    print("錯誤：未提供訪問令牌")
    exit(1)

push_to_huggingface(model, model_name, username, token)
