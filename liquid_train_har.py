import torch
import torch.nn as nn
import torch.optim as optim
from liquid_net import LiquidNet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import numpy as np
import os

# 9 個原始訊號檔名（x/y/z = 3 軸 × 3 種訊號）
RAW_FILES = [
    "body_acc_x_", "body_acc_y_", "body_acc_z_",
    "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
    "total_acc_x_", "total_acc_y_", "total_acc_z_"
]

def load_har_data(root="datasets/UCI HAR Dataset"):
    """
    讀取 UCI HAR 9 個原始感測器訊號，回傳
        X_train, y_train, X_test, y_test
    其中  X  形狀 = (samples, 128, 9)
          y  為 0-based 類別標籤
    """
    def _load_split(split):  # split = "train" 或 "test"
        signals = []
        for fname in RAW_FILES:
            path = os.path.join(root, split, "Inertial Signals",
                                f"{fname}{split}.txt")
            # 每個檔案: (samples, 128) 文字矩陣
            signals.append(np.loadtxt(path))        # list of (N,128)
        # 堆疊成 (samples, 128, 9)
        X = np.stack(signals, axis=-1)              # axis=-1 得到 9 通道
        y = np.loadtxt(os.path.join(root, split, f"y_{split}.txt")).astype(int) - 1
        return X, y

    print("載入原始 Inertial Signals ...")
    X_train, y_train = _load_split("train")         # (7352,128,9)
    X_test,  y_test  = _load_split("test")          # (2947,128,9)

    print("完成！形狀如下：")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test : {X_test.shape},  y_test : {y_test.shape}")
    return X_train, y_train, X_test, y_test


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """訓練模型"""
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 驗證階段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_liquid_har_model.pth')
        
        # 每5個epoch顯示進度
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_losses, train_accs, val_accs

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """繪製訓練曲線"""
    plt.figure(figsize=(12, 5))
    
    # 繪製損失曲線
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    
    # 繪製準確率曲線
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('har_training_curves.png')
    plt.close()

def main():
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 載入數據
    X_train, y_train, X_test, y_test = load_har_data()
    
    # 轉換為 PyTorch 張量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # 創建數據加載器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=64)
    
    # 創建模型
    input_size = 9  # 每個時間步的9個特徵
    hidden_size = 128  # 隱藏層大小
    output_size = 6  # 動作類別數
    steps = 128  # 時間步長
    
    print(f"模型參數:")
    print(f"輸入特徵數: {input_size}")
    print(f"隱藏層大小: {hidden_size}")
    print(f"輸出類別數: {output_size}")
    print(f"時間步長: {steps}")
    
    model = LiquidNet(input_size, hidden_size, output_size, steps).to(device)
    
    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練模型
    num_epochs = 100
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    )
    
    # 繪製訓練曲線
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    print("訓練完成！最佳模型已保存為 'best_liquid_har_model.pth'")
    print("訓練曲線已保存為 'har_training_curves.png'")

if __name__ == "__main__":
    main() 