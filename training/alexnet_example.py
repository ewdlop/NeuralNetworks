import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from alexnet import AlexNet

def main():
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 數據預處理
    transform = transforms.Compose([
        transforms.Resize((227, 227)),  # 調整圖像大小為 227x227
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 數據集的均值和標準差
                           std=[0.229, 0.224, 0.225])
    ])

    # 假設我們有一個數據集目錄結構如下：
    # data/
    #   ├── train/
    #   │   ├── class1/
    #   │   ├── class2/
    #   │   └── ...
    #   └── val/
    #       ├── class1/
    #       ├── class2/
    #       └── ...
    
    # 創建數據加載器
    train_dataset = ImageFolder(root='data/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    val_dataset = ImageFolder(root='data/val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 獲取類別數量
    num_classes = len(train_dataset.classes)
    print(f"類別數量: {num_classes}")
    print(f"類別名稱: {train_dataset.classes}")

    # 創建模型
    model = AlexNet(num_classes=num_classes)
    model = model.to(device)

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 訓練模型
    num_epochs = 100
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向傳播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向傳播和優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 統計
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 計算訓練準確率
        train_acc = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f} '
              f'Train Acc: {train_acc:.2f}%')
        
        # 驗證階段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # 計算驗證準確率
        val_acc = 100. * correct / total
        print(f'Val Loss: {val_loss/len(val_loader):.4f} Val Acc: {val_acc:.2f}%')
        
        # 更新學習率
        scheduler.step()

    # 保存模型
    torch.save(model.state_dict(), 'alexnet_model.pth')
    print("模型已保存到 alexnet_model.pth")

if __name__ == '__main__':
    main() 