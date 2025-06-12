import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    基本的殘差塊（Residual Block）
    
    這是 ResNet 的基本構建塊，包含兩個卷積層和一個殘差連接。
    每個卷積層後面都跟著一個批標準化層。
    
    殘差連接的作用：
    1. 解決深度網絡的梯度消失問題
    2. 允許網絡直接學習殘差函數 F(x) = H(x) - x
    3. 使得網絡更容易優化
    
    參數:
        in_planes (int): 輸入通道數，決定了輸入特徵圖的通道數
        planes (int): 輸出通道數，決定了輸出特徵圖的通道數
        stride (int): 卷積步長，默認為 1
                     - 當 stride=1 時，輸出特徵圖大小與輸入相同
                     - 當 stride=2 時，輸出特徵圖的寬高減半
        downsample (nn.Module, optional): 下採樣層，用於調整殘差連接的維度
                                        - 當輸入輸出通道數不匹配時使用
                                        - 當 stride 不為 1 時使用
    """
    expansion = 1  # 擴展係數，用於調整輸出通道數
                   # 在 BasicBlock 中為 1，在 Bottleneck 中為 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        # 第一個卷積層：3x3 卷積，可選的步長
        # kernel_size=3: 使用 3x3 的卷積核
        # stride=stride: 可變步長，用於下採樣
        # padding=1: 保持特徵圖大小不變（當 stride=1 時）
        # bias=False: 不使用偏置項，因為後面有 BatchNorm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # 批標準化層，用於加速訓練和提高穩定性
        
        # 第二個卷積層：3x3 卷積，步長固定為 1
        # 注意：第二個卷積層的步長始終為 1，保持特徵圖大小
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # 批標準化層
        
        self.downsample = downsample  # 下採樣層，用於調整殘差連接的維度
                                     # 當輸入輸出通道數不匹配或需要下採樣時使用

    def forward(self, x):
        """
        前向傳播函數
        
        參數:
            x (torch.Tensor): 輸入張量，形狀為 [batch_size, in_planes, height, width]
            
        返回:
            torch.Tensor: 輸出張量，形狀為 [batch_size, planes, height, width]
                         （當 stride=1 時）或 [batch_size, planes, height/2, width/2]（當 stride=2 時）
        
        計算過程：
        1. 保存輸入用於殘差連接
        2. 通過第一個卷積塊（卷積 + 批標準化 + ReLU）
        3. 通過第二個卷積塊（卷積 + 批標準化）
        4. 如果存在下採樣層，對輸入進行下採樣
        5. 將卷積結果與處理後的輸入相加
        6. 應用 ReLU 激活函數
        """
        identity = x  # 保存輸入，用於殘差連接
        
        # 第一個卷積塊：卷積 -> 批標準化 -> ReLU
        # 當 stride=2 時，特徵圖大小減半
        out = F.relu(self.bn1(self.conv1(x)))
        
        # 第二個卷積塊：卷積 -> 批標準化
        # 特徵圖大小保持不變
        out = self.bn2(self.conv2(out))
        
        # 如果存在下採樣層，則對輸入進行下採樣
        # 下採樣層包含 1x1 卷積和批標準化
        if self.downsample:
            identity = self.downsample(x)
        
        # 殘差連接：將卷積結果與輸入相加
        # 這使得網絡可以更容易地學習恆等映射
        out += identity
        
        # 應用 ReLU 激活函數
        # 注意：ReLU 在殘差連接之後應用
        return F.relu(out)

class ResNet(nn.Module):
    """
    ResNet 模型實現
    
    這是一個經典的深度殘差網絡，通過堆疊多個殘差塊來構建。
    支持不同深度的 ResNet 變體（如 ResNet-18, ResNet-34 等）。
    
    網絡結構：
    1. 初始卷積層：7x7 卷積，步長為 2
    2. 最大池化層：3x3 池化，步長為 2
    3. 四個階段的殘差塊：
       - 第一階段：64 通道，2 個殘差塊
       - 第二階段：128 通道，2 個殘差塊
       - 第三階段：256 通道，2 個殘差塊
       - 第四階段：512 通道，2 個殘差塊
    4. 全局平均池化
    5. 全連接分類層
    
    參數:
        block (nn.Module): 殘差塊類型（BasicBlock 或 Bottleneck）
        layers (list): 每個階段包含的殘差塊數量
        num_classes (int): 分類的類別數量
    """
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_planes = 64  # 初始通道數
        
        # 第一個卷積層：7x7 卷積，步長為 2
        # 輸入：3 通道（RGB 圖像）
        # 輸出：64 通道
        # 特徵圖大小：輸入大小的一半
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)  # 批標準化層, 用於加速訓練和提高穩定性
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化層, 用於減少特徵圖大小
        
        # 構建四個階段的殘差塊
        # 每個階段的通道數翻倍，特徵圖大小減半
        self.layer1 = self._make_layer(block, 64,  layers[0])  # 第一階段：64 通道
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 第二階段：128 通道
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 第三階段：256 通道
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 第四階段：512 通道
        
        # 全局平均池化和全連接層
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自適應平均池化，輸出固定大小, 輸出形狀為 (1, 1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全連接層，用於分類

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        構建一個階段的殘差塊
        
        參數:
            block (nn.Module): 殘差塊類型（BasicBlock 或 Bottleneck）
            planes (int): 輸出通道數
            blocks (int): 殘差塊數量
            stride (int): 第一個殘差塊的步長
            
        返回:
            nn.Sequential: 包含多個殘差塊的序列
        
        注意：
        1. 只有第一個殘差塊可能改變特徵圖大小
        2. 後續殘差塊保持特徵圖大小不變
        3. 當需要改變通道數或特徵圖大小時，使用下採樣層
        """
        if not hasattr(self, 'in_planes'):
            raise RuntimeError("self.in_planes 未初始化，請確保在調用 _make_layer 之前初始化 ResNet")
            
        downsample = None
        # 如果步長不為 1 或通道數不匹配，則需要下採樣
        # block.expansion 是擴展係數，用於調整輸出通道數
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                # 1x1 卷積用於調整通道數和空間尺寸
                # kernel_size=1: 1x1 卷積，只改變通道數
                # stride=stride: 可變步長，用於下採樣
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), # 1x1 卷積，只改變通道數
                nn.BatchNorm2d(planes * block.expansion) # 批標準化層, 用於加速訓練和提高穩定性
            )
        
        # 創建第一個殘差塊
        # 這個塊可能改變特徵圖大小
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        
        # 創建剩餘的殘差塊
        # 這些塊保持特徵圖大小不變
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向傳播函數
        
        參數:
            x (torch.Tensor): 輸入張量，形狀為 [batch_size, 3, height, width]
            
        返回:
            torch.Tensor: 模型輸出，形狀為 [batch_size, num_classes]
        
        數據流動過程：
        1. 初始卷積和池化：特徵圖大小減半兩次
        2. 通過四個階段的殘差塊：每個階段可能減半特徵圖大小
        3. 全局平均池化：將特徵圖轉換為固定大小
        4. 全連接層：進行最終分類
        """
        # 初始卷積和池化
        # 輸入：224x224 -> 112x112 -> 56x56
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 通過四個階段的殘差塊
        # 特徵圖大小變化：56x56 -> 28x28 -> 14x14 -> 7x7
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化和分類
        x = self.avgpool(x)  # 7x7 -> 1x1
        x = torch.flatten(x, 1)  # 展平特徵圖
        return self.fc(x)  # 全連接層分類

def resnet18(num_classes=1000):
    """
    創建 ResNet-18 模型
    
    參數:
        num_classes (int): 分類的類別數量
        
    返回:
        ResNet: ResNet-18 模型實例
        
    網絡結構詳解：
    1. 初始卷積層：7x7 卷積，步長為 2
       - 輸入：224x224x3
       - 輸出：112x112x64
    
    2. 最大池化層：3x3 池化，步長為 2
       - 輸入：112x112x64
       - 輸出：56x56x64
    
    3. 四個階段的殘差塊：
       - 第一階段（2個殘差塊）：
         * 輸入：56x56x64
         * 輸出：56x56x64
       - 第二階段（2個殘差塊）：
         * 輸入：56x56x64
         * 輸出：28x28x128
       - 第三階段（2個殘差塊）：
         * 輸入：28x28x128
         * 輸出：14x14x256
       - 第四階段（2個殘差塊）：
         * 輸入：14x14x256
         * 輸出：7x7x512
    
    4. 全局平均池化：
       - 輸入：7x7x512
       - 輸出：1x1x512
    
    5. 全連接層：
       - 輸入：512
       - 輸出：num_classes
    
    總層數計算：
    - 1 個初始卷積層
    - 1 個最大池化層
    - 4 個階段，每個階段 2 個殘差塊，每個殘差塊 2 個卷積層
    - 1 個全連接層
    - 總計：1 + 1 + 4*2*2 + 1 = 19 層
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
