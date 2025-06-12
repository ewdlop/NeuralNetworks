import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    U-Net 的基本構建塊：(卷積 => 批標準化 => ReLU) * 2
    
    這個模塊包含兩個連續的 3x3 卷積層。
    每個卷積層後面跟著一個批標準化（BatchNorm）層和一個 ReLU 激活函數。
    使用 padding=1 來確保在 3x3 卷積後特徵圖的尺寸保持不變。
    
    參數:
        in_channels (int): 輸入特徵圖的通道數
        out_channels (int): 輸出特徵圖的通道數
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # inplace=True 可以節省內存

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    下採樣模塊（編碼器部分）
    
    這個模塊首先使用 2x2 的最大池化（Max Pooling）將特徵圖的寬高減半，
    然後再通過一個 DoubleConv 模塊進行特徵提取。
    
    參數:
        in_channels (int): 輸入通道數
        out_channels (int): 輸出通道數
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    """
    上採樣模塊（解碼器部分）
    
    這個模塊首先對特徵圖進行上採樣，將其寬高加倍。
    然後，將上採樣的結果與來自編碼器對應層級的特徵圖（通過跳躍連接傳遞）進行拼接。
    最後，通過一個 DoubleConv 模塊進行特徵融合和提取。
    
    參數:
        in_channels (int): 輸入通道數（來自上一層解碼器和跳躍連接的通道數之和）
        out_channels (int): 輸出通道數
        bilinear (bool): 是否使用雙線性插值進行上採樣。
                         若為 True，使用 nn.Upsample。
                         若為 False，使用 nn.ConvTranspose2d（可學習的反卷積）。
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 選擇上採樣方法
        if bilinear:
            # 雙線性插值，無可學習參數，計算速度快
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 轉置卷積，有可學習參數，理論上更強大但可能引入棋盤效應
            # 棋盤效應：在反卷積中，如果步長為 2，則會在特徵圖中引入棋盤效應，即在特徵圖中出現棋盤狀的紋理。
            # in_channels // 2 是因為輸入 x1 的通道數是 in_channels 的一半
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        前向傳播
        
        參數:
            x1 (torch.Tensor): 來自上一層解碼器的特徵圖，需要被上採樣
            x2 (torch.Tensor): 來自編碼器對應層級的特徵圖（跳躍連接）
        """
        # 1. 上採樣 x1
        x1 = self.up(x1)

        # 2. 處理尺寸不匹配問題
        # 如果輸入圖像的寬或高不是 16 的倍數，下採樣和上採樣後尺寸可能會有 1 個像素的差異。
        # 這裡使用 padding 來補齊 x1，使其與 x2 的尺寸一致。
        diffY = x2.size()[2] - x1.size()[2] # 計算高度差
        diffX = x2.size()[3] - x1.size()[3] # 計算寬度差

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,   # 左右填充 diffX // 2 是左填充，diffX - diffX // 2 是右填充
                        diffY // 2, diffY - diffY // 2])  # 上下填充 diffY // 2 是上填充，diffY - diffY // 2 是下填充
        
        # 3. 拼接特徵圖
        # 將來自編碼器的特徵圖 x2 和上採樣後的特徵圖 x1 沿通道維度拼接。
        # 這是 U-Net 的核心思想：結合高層語義特徵和低層細節特徵。
        x = torch.cat([x2, x1], dim=1)
        
        # 4. 進行雙卷積
        return self.conv(x)

class OutConv(nn.Module):
    """
    輸出卷積層
    
    使用一個 1x1 的卷積層將多通道的特徵圖映射到指定數量的輸出通道。
    例如，在二元分割中，out_channels 通常為 1。
    
    參數:
        in_channels (int): 輸入通道數
        out_channels (int): 輸出通道數（通常等於分割的類別數）
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.out_conv(x)

class UNet(nn.Module):
    """
    完整的 U-Net 模型
    
    網絡結構：
    1.  一個對稱的編碼器-解碼器結構。
    2.  編碼器（收縮路徑）包含 4 個下採樣模塊，逐步提取特徵並減小特徵圖尺寸。
    3.  解碼器（擴張路徑）包含 4 個上採樣模塊，逐步恢復細節並增大特徵圖尺寸。
    4.  編碼器和解碼器之間通過跳躍連接傳遞特徵圖，保留空間信息。
    
    參數:
        in_channels (int): 輸入圖像的通道數（例如，RGB 圖像為 3）。
        out_channels (int): 輸出的類別數（例如，二元分割為 1）。
        bilinear (bool): 上採樣時是否使用雙線性插值。
    """
    def __init__(self, in_channels=3, out_channels=1, bilinear=True):
        super().__init__()
        # 編碼器（收縮路徑）
        self.in_conv = DoubleConv(in_channels, 64)  # 初始卷積塊
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # 在最底層，如果使用反卷積，通道數會減半再加倍，所以這裡需要調整
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # 解碼器（擴張路徑）
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out_conv = OutConv(64, out_channels) # 最終輸出層

    def forward(self, x):
        # 編碼器路徑
        x1 = self.in_conv(x)    # -> 64  x C x H x W
        x2 = self.down1(x1)   # -> 128 x C/2 x H/2
        x3 = self.down2(x2)   # -> 256 x C/4 x H/4
        x4 = self.down3(x3)   # -> 512 x C/8 x H/8
        x5 = self.down4(x4)   # -> 1024/512 x C/16 x H/16

        # 解碼器路徑 + 跳躍連接
        x = self.up1(x5, x4) # x4 是來自 down3 的輸出
        x = self.up2(x, x3)  # x3 是來自 down2 的輸出
        x = self.up3(x, x2)  # x2 是來自 down1 的輸出
        x = self.up4(x, x1)  # x1 是來自 in_conv 的輸出
        
        # 輸出
        logits = self.out_conv(x)
        return logits
