import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """
    AlexNet 神經網絡模型實現
    
    這是一個經典的卷積神經網絡架構，由 Alex Krizhevsky 等人在 2012 年提出。
    該模型在 ImageNet 競賽中取得了突破性的成果，標誌著深度學習在計算機視覺領域的崛起。
    
    主要特點：
    1. 使用 ReLU 激活函數代替傳統的 tanh
    2. 使用 Dropout 來防止過擬合
    3. 使用重疊的最大池化
    4. 使用數據增強技術

    特徵圖尺寸變化過程（輸入圖像：227 x 227）：
    1. 第一層卷積層 (11x11, stride=4, padding=2)：
       (227 + 2*2 - 11) / 4 + 1 = 56 x 56
    2. 第一層池化層 (3x3, stride=2)：
       (56 - 3) / 2 + 1 = 27 x 27
    3. 第二層卷積層 (5x5, stride=1, padding=2)：
       (27 + 2*2 - 5) / 1 + 1 = 27 x 27
    4. 第二層池化層 (3x3, stride=2)：
       (27 - 3) / 2 + 1 = 13 x 13
    5. 第三層卷積層 (3x3, stride=1, padding=1)：
       (13 + 2*1 - 3) / 1 + 1 = 13 x 13
    6. 第四層卷積層 (3x3, stride=1, padding=1)：
       (13 + 2*1 - 3) / 1 + 1 = 13 x 13
    7. 第五層卷積層 (3x3, stride=1, padding=1)：
       (13 + 2*1 - 3) / 1 + 1 = 13 x 13
    8. 最後池化層 (3x3, stride=2)：
       (13 - 3) / 2 + 1 = 6 x 6
    """
    def __init__(self, num_classes=1000):
        """
        初始化 AlexNet 模型
        
        參數:
            num_classes (int): 分類的類別數量，默認為 1000（ImageNet 的類別數）
        """
        super().__init__()
        # 特徵提取層
        self.features = nn.Sequential(
            # 第一層卷積層：輸入 3 通道，輸出 64 通道，使用 11x11 的卷積核, 步長為4，填充為2
            # 輸出尺寸：(227 + 2*2 - 11) / 4 + 1 = 56 x 56

            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),  # ReLU 激活函數，inplace=True 可以節省內存
            # 最大池化層，減少特徵圖大小
            # 輸出尺寸：(56 - 3) / 2 + 1 = 27 x 27
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 第二層卷積層：64 通道 -> 192 通道，使用 5x5 的卷積核,填充為2
            # 輸出尺寸：(27 + 2*2 - 5) / 1 + 1 = 27 x 27
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # 最大池化層，減少特徵圖大小
            # 輸出尺寸：(27 - 3) / 2 + 1 = 13 x 13
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 第三層卷積層：192 通道 -> 384 通道，使用 3x3 的卷積核,填充為1
            # 輸出尺寸：(13 + 2*1 - 3) / 1 + 1 = 13 x 13
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第四層卷積層：384 通道 -> 256 通道，使用 3x3 的卷積核,填充為1
            # 輸出尺寸：(13 + 2*1 - 3) / 1 + 1 = 13 x 13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第五層卷積層：256 通道 -> 256 通道，使用 3x3 的卷積核,填充為1
            # 輸出尺寸：(13 + 2*1 - 3) / 1 + 1 = 13 x 13
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 最大池化層，減少特徵圖大小
            # 輸出尺寸：(13 - 3) / 2 + 1 = 6 x 6
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 分類器層
        self.classifier = nn.Sequential(
            # 第一個全連接層：使用 Dropout 防止過擬合
            nn.Dropout(), # 使用Dropout防止過擬合

            # 256是特徵圖的通道數，6x6是最終特徵圖的尺寸
            # 256 * 6 * 6 = 9216 個特徵點
            # 4096是隱藏層的節點數，這個數字是經驗值，提供了足夠的模型容量
            nn.Linear(256 * 6 * 6, 4096),  # 將特徵圖展平後連接全連接層
            nn.ReLU(inplace=True),

            # 第二個全連接層
            nn.Dropout(), # 使用Dropout防止過擬合
            nn.Linear(4096, 4096), # 將4096個節點連接4096個節點
            nn.ReLU(inplace=True),

            # 輸出層：4096 -> num_classes
            nn.Linear(4096, num_classes), # 將4096個節點連接num_classes個節點
        )

    def forward(self, x):
        """
        前向傳播函數
        
        參數:
            x (torch.Tensor): 輸入張量，形狀為 (batch_size, 3, 227, 227)
            
        返回:
            torch.Tensor: 模型輸出，形狀為 (batch_size, num_classes)
        """
        # 通過特徵提取層
        # batch_size: 批量大小
        # 3: 圖片通道數
        # 227: 圖片高度
        # 227: 圖片寬度
        # 輸入圖片x的形狀為(batch_size, 3, 227, 227)
        # 通過特徵提取層後，輸出圖片的形狀為(batch_size, 256, 6, 6)
        x = self.features(x)  # 通過特徵提取層
        # 將特徵圖展平
        # 輸入圖片x的形狀為(batch_size, 256, 6, 6)
        # 通過展平後，輸出圖片的形狀為(batch_size, 256 * 6 * 6 = 9216)
        x = torch.flatten(x, 1) 
        # 通過分類器層
        # 輸入圖片x的形狀為(batch_size, 9216)
        # 通過分類器層後，輸出圖片的形狀為(batch_size, num_classes)
        x = self.classifier(x)  # 通過分類器層
        return x

