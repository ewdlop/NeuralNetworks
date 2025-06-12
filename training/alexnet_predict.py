import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from alexnet import AlexNet

def download_image(url):
    """
    從 URL 下載圖像
    
    參數:
        url (str): 圖像的 URL
        
    返回:
        PIL.Image: 下載的圖像
        
    功能說明:
    1. 使用 requests 庫發送 HTTP GET 請求獲取圖像
    2. 檢查請求是否成功
    3. 將圖像數據轉換為 PIL Image 對象
    4. 如果下載失敗，返回 None
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # 檢查請求是否成功
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print(f"下載圖像時出錯: {e}")
        return None

def predict_image(model, image, class_names):
    """
    使用模型預測圖像的類別
    
    參數:
        model (nn.Module): 訓練好的模型
        image (PIL.Image): 輸入圖像
        class_names (list): 類別名稱列表
        
    返回:
        tuple: (預測的類別名稱, 預測的機率)
        
    功能說明:
    1. 圖像預處理：
       - 調整大小為 227x227（AlexNet 的輸入要求）
       - 轉換為張量
       - 使用 ImageNet 數據集的均值和標準差進行標準化
    2. 模型預測：
       - 將圖像轉換為模型可接受的格式
       - 使用 softmax 函數計算每個類別的概率
       - 選擇概率最高的類別作為預測結果
    """
    # 圖像預處理
    transform = transforms.Compose([
        transforms.Resize((227, 227)),  # 調整圖像大小為 227x227
        transforms.ToTensor(),  # 將圖像轉換為張量，並將像素值歸一化到 [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 數據集的均值
                           std=[0.229, 0.224, 0.225])  # ImageNet 數據集的標準差
    ])
    
    # 轉換圖像
    image_tensor = transform(image).unsqueeze(0)  # 添加批次維度，形狀從 [C,H,W] 變為 [1,C,H,W]
    
    # 設置模型為評估模式
    model.eval()  # 關閉 dropout 等訓練時的特殊層
    
    # 進行預測
    with torch.no_grad():  # 不計算梯度，節省內存
        outputs = model(image_tensor)  # 前向傳播
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # 計算每個類別的概率
        predicted_class = torch.argmax(probabilities).item()  # 獲取概率最高的類別索引
        confidence = probabilities[predicted_class].item()  # 獲取預測的置信度
    
    return class_names[predicted_class], confidence

def main():
    """
    主函數
    
    功能說明:
    1. 設置設備（GPU/CPU）
    2. 定義示例圖像 URL
    3. 定義類別名稱
    4. 加載模型和權重
    5. 對每個圖像進行預測並顯示結果
    """
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    # 示例圖像 URL
    image_urls = [
        # 可以添加更多圖像 URL
    ]
    
    # 假設的類別名稱（應該與訓練時使用的類別相同）
    class_names = ["貓", "狗", "鳥", "魚"]  # 根據您的數據集修改
    
    # 加載模型
    model = AlexNet(num_classes=len(class_names))  # 創建模型實例
    model = model.to(device)  # 將模型移動到指定設備（GPU/CPU）
    
    # 加載預訓練權重（如果有的話）
    try:
        model.load_state_dict(torch.load('alexnet_model.pth'))  # 加載模型權重
        print("成功加載模型權重")
    except:
        print("未找到預訓練權重，使用隨機初始化的模型")
    
    # 處理每個圖像
    for url in image_urls:
        print(f"\n處理圖像: {url}")
        
        # 下載圖像
        image = download_image(url)
        if image is None:
            continue
        
        # 顯示圖像信息
        print(f"圖像大小: {image.size}")  # 顯示圖像的寬度和高度
        print(f"圖像模式: {image.mode}")  # 顯示圖像的顏色模式（RGB/灰度等）
        
        # 進行預測
        predicted_class, confidence = predict_image(model, image, class_names)
        
        # 顯示結果
        print(f"預測類別: {predicted_class}")  # 顯示預測的類別
        print(f"預測機率: {confidence:.2%}")  # 顯示預測的置信度（百分比格式）

if __name__ == '__main__':
    main() 