# 操作系統模組，用於與操作系統交互
import os

# 用於生成隨機數的模組
import random

# 用於數值運算的模組
import numpy as np

# OpenCV 庫，用於圖像處理
import cv2

# Python 圖像處理庫
from PIL import Image, ImageDraw, ImageFont

# PyTorch 深度學習庫
import torch

# PyTorch 中的數據集類，用於創建自定義數據集
from torch.utils.data import Dataset

# 圖像轉換模組
import torchvision.transforms as transforms

# PyTorch 中的神經網絡模組
import torch.nn as nn

# PyTorch 中的優化算法
import torch.optim as optim

# PyTorch 中用於填充序列的函數
from torch.nn.utils.rnn import pad_sequence

# PyTorch 中用於保存圖像的函數
from torchvision.utils import save_image

# 用於繪製圖表和圖像的模組
import matplotlib.pyplot as plt

# 用於在 IPython 環境中顯示豐富內容的模組
from IPython.display import clear_output, display, HTML

# 用於編碼和解碼二進制數據到文本的模組
import base64

# 創建訓練數據集的目錄
os.makedirs('training_dataset', exist_ok=True)

# 定義要為數據集生成的視頻數量
num_videos = 30000

# 定義每個視頻的幀數（1秒視頻）
frames_per_video = 10

# 定義數據集中每個圖像的大小
img_size = (64, 64)

# 定義形狀（圓形）的大小
shape_size = 10

# 定義文本提示和相應的圓形運動
prompts_and_movements = [
    ("circle moving down", "circle", "down"),  # 圓形向下移動
    ("circle moving left", "circle", "left"),  # 圓形向左移動
    ("circle moving right", "circle", "right"),  # 圓形向右移動
    ("circle moving diagonally up-right", "circle", "diagonal_up_right"),  # 圓形對角線向上右移動
    ("circle moving diagonally down-left", "circle", "diagonal_down_left"),  # 圓形對角線向下左移動
    ("circle moving diagonally up-left", "circle", "diagonal_up_left"),  # 圓形對角線向上左移動
    ("circle moving diagonally down-right", "circle", "diagonal_down_right"),  # 圓形對角線向下右移動
    ("circle rotating clockwise", "circle", "rotate_clockwise"),  # 圓形順時針旋轉
    ("circle rotating counter-clockwise", "circle", "rotate_counter_clockwise"),  # 圓形逆時針旋轉
    ("circle bouncing vertically", "circle", "bounce_vertical"),  # 圓形垂直彈跳
    ("circle bouncing horizontally", "circle", "bounce_horizontal"),  # 圓形水平彈跳
    ("circle zigzagging vertically", "circle", "zigzag_vertical"),  # 圓形垂直之字形移動
    ("circle zigzagging horizontally", "circle", "zigzag_horizontal"),  # 圓形水平之字形移動
    ("circle moving up-left", "circle", "up_left"),  # 圓形向上左移動
    ("circle moving down-right", "circle", "down_right"),  # 圓形向下右移動
    ("circle moving down-left", "circle", "down_left")  # 圓形向下左移動
]

# 添加更多形狀和運動
shapes = ["circle", "square", "triangle", "star", "heart"]
complex_movements = [
    "spiral_clockwise",
    "wave_motion", 
    "accelerating_right",
    "bouncing_with_rotation",
    "figure_eight"
]

# 定義一個函數來創建帶有移動形狀的圖像
def create_image_with_moving_shape(size, frame_num, shape, direction):
    # 創建一個新的 RGB 圖像，指定大小和白色背景
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 計算形狀的初始位置（圖像中心）
    center_x, center_y = size[0] // 2, size[1] // 2

    # 根據移動方向確定形狀位置
    if direction == "down":
        position = (center_x, (center_y + frame_num * 5) % size[1])
    elif direction == "left":
        position = ((center_x - frame_num * 5) % size[0], center_y)
    elif direction == "right":
        position = ((center_x + frame_num * 5) % size[0], center_y)
    elif direction == "diagonal_up_right":
        position = ((center_x + frame_num * 5) % size[0], (center_y - frame_num * 5) % size[1])
    elif direction == "diagonal_down_left":
        position = ((center_x - frame_num * 5) % size[0], (center_y + frame_num * 5) % size[1])
    elif direction == "diagonal_up_left":
        position = ((center_x - frame_num * 5) % size[0], (center_y - frame_num * 5) % size[1])
    elif direction == "diagonal_down_right":
        position = ((center_x + frame_num * 5) % size[0], (center_y + frame_num * 5) % size[1])
    elif direction == "rotate_clockwise":
        img = img.rotate(frame_num * 10, center=(center_x, center_y), fillcolor=(255, 255, 255))
        position = (center_x, center_y)
    elif direction == "rotate_counter_clockwise":
        img = img.rotate(-frame_num * 10, center=(center_x, center_y), fillcolor=(255, 255, 255))
        position = (center_x, center_y)
    elif direction == "bounce_vertical":
        position = (center_x, center_y - abs(frame_num * 5 % size[1] - center_y))
    elif direction == "bounce_horizontal":
        position = (center_x - abs(frame_num * 5 % size[0] - center_x), center_y)
    elif direction == "zigzag_vertical":
        position = (center_x, center_y - frame_num * 5 % size[1] if frame_num % 2 == 0 else center_y + frame_num * 5 % size[1])
    elif direction == "zigzag_horizontal":
        position = (center_x - frame_num * 5 % size[0] if frame_num % 2 == 0 else center_x + frame_num * 5 % size[0], center_y)
    elif direction == "up_left":
        position = ((center_x - frame_num * 5) % size[0], (center_y - frame_num * 5) % size[1])
    elif direction == "down_right":
        position = ((center_x + frame_num * 5) % size[0], (center_y + frame_num * 5) % size[1])
    elif direction == "down_left":
        position = ((center_x - frame_num * 5) % size[0], (center_y + frame_num * 5) % size[1])
    else:
        position = (center_x, center_y)

    # 在計算出的位置繪製形狀（圓形）
    if shape == "circle":
        draw.ellipse([position[0] - shape_size // 2, position[1] - shape_size // 2, position[0] + shape_size // 2, position[1] + shape_size // 2], fill=(0, 0, 255))

    # 將圖像作為 numpy 數組返回
    return np.array(img)

# 定義一個繼承自 torch.utils.data.Dataset 的數據集類
class TextToVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # 使用根目錄和可選的轉換初始化數據集
        self.root_dir = root_dir
        self.transform = transform
        # 列出根目錄中的所有子目錄
        self.video_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        # 初始化列表以存儲幀路徑和相應的提示
        self.frame_paths = []
        self.prompts = []

        # 遍歷每個視頻目錄
        for video_dir in self.video_dirs:
            # 列出視頻目錄中的所有 PNG 文件並存儲它們的路徑
            frames = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.png')]
            self.frame_paths.extend(frames)
            # 讀取視頻目錄中的提示文本文件並存儲其內容
            with open(os.path.join(video_dir, 'prompt.txt'), 'r') as f:
                prompt = f.read().strip()
            # 為視頻中的每一幀重複提示並存儲在提示列表中
            self.prompts.extend([prompt] * len(frames))

    # 返回數據集中的樣本總數
    def __len__(self):
        return len(self.frame_paths)

    # 根據索引從數據集中檢索樣本
    def __getitem__(self, idx):
        # 獲取對應於給定索引的幀的路徑
        frame_path = self.frame_paths[idx]
        # 使用 PIL（Python 圖像庫）打開圖像
        image = Image.open(frame_path)
        # 獲取對應於給定索引的提示
        prompt = self.prompts[idx]

        # 如果指定了轉換，則應用轉換
        if self.transform:
            image = self.transform(image)

        # 返回轉換後的圖像和提示
        return image, prompt

# 使用預訓練模型替代簡單嵌入
from transformers import BertModel, BertTokenizer

class AdvancedTextEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def forward(self, text):
        tokens = self.tokenizer(text, return_tensors='pt', padding=True)
        outputs = self.bert(**tokens)
        return outputs.last_hidden_state.mean(dim=1)  # 平均池化

# 生成器類
class Generator(nn.Module):
    def __init__(self, text_embed_size):
        super(Generator, self).__init__()

        # 全連接層，接收噪聲和文本嵌入作為輸入
        self.fc1 = nn.Linear(100 + text_embed_size, 256 * 8 * 8)

        # 轉置卷積層，用於上採樣輸入
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, 4, 2, 1)  # 輸出有 3 個通道用於 RGB 圖像

        # 激活函數
        self.relu = nn.ReLU(True)  # ReLU 激活函數
        self.tanh = nn.Tanh()       # Tanh 激活函數用於最終輸出

    def forward(self, noise, text_embed):
        # 沿通道維度連接噪聲和文本嵌入
        x = torch.cat((noise, text_embed), dim=1)

        # 全連接層後重塑為 4D 張量
        x = self.fc1(x).view(-1, 256, 8, 8)

        # 通過轉置卷積層進行上採樣，使用 ReLU 激活
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))

        # 最終層使用 Tanh 激活，確保輸出值在 -1 和 1 之間（用於圖像）
        x = self.tanh(self.deconv3(x))

        return x

# 判別器類
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 卷積層，用於處理輸入圖像
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)   # 3 個輸入通道（RGB），64 個輸出通道，4x4 核大小，步長 2，填充 1
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1) # 64 個輸入通道，128 個輸出通道，4x4 核大小，步長 2，填充 1
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1) # 128 個輸入通道，256 個輸出通道，4x4 核大小，步長 2，填充 1

        # 用於分類的全連接層
        self.fc1 = nn.Linear(256 * 8 * 8, 1)  # 輸入大小 256x8x8（最後一個卷積的輸出大小），輸出大小 1（二分類）

        # 激活函數
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)  # Leaky ReLU 激活，負斜率 0.2
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活用於最終輸出（概率）

    def forward(self, input):
        # 通過卷積層傳遞輸入，使用 LeakyReLU 激活
        x = self.leaky_relu(self.conv1(input))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))

        # 展平卷積層的輸出
        x = x.view(-1, 256 * 8 * 8)

        # 通過全連接層，使用 Sigmoid 激活進行二分類
        x = self.sigmoid(self.fc1(x))

        return x

# 檢查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 為文本提示創建簡單的詞彙表
all_prompts = [prompt for prompt, _, _ in prompts_and_movements]  # 從 prompts_and_movements 列表中提取所有提示
vocab = {word: idx for idx, word in enumerate(set(" ".join(all_prompts).split()))}  # 創建詞彙表字典，其中每個唯一單詞都被分配一個索引
vocab_size = len(vocab)  # 詞彙表的大小
embed_size = 10  # 文本嵌入向量的大小

def encode_text(prompt):
    # 使用詞彙表將給定的提示編碼為索引張量
    return torch.tensor([vocab[word] for word in prompt.split()])

# 初始化模型、損失函數和優化器
text_embedding = AdvancedTextEmbedding().to(device)  # 使用 AdvancedTextEmbedding 模型
netG = Generator(embed_size).to(device)  # 使用 embed_size 初始化生成器模型
netD = Discriminator().to(device)  # 初始化判別器模型
criterion = nn.BCELoss().to(device)  # 二元交叉熵損失函數
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))  # 判別器的 Adam 優化器
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))  # 生成器的 Adam 優化器

# 訓練循環
num_epochs = 13

# 遍歷每個 epoch
for epoch in range(num_epochs):
    # 遍歷每個數據批次
    for i, (data, prompts) in enumerate(dataloader):
        # 將真實數據移到設備上
        real_data = data.to(device)

        # 將提示轉換為列表
        prompts = [prompt for prompt in prompts]

        # 更新判別器
        netD.zero_grad()  # 將判別器的梯度清零
        batch_size = real_data.size(0)  # 獲取批次大小
        labels = torch.ones(batch_size, 1).to(device)  # 為真實數據創建標籤（1）
        output = netD(real_data)  # 將真實數據通過判別器進行前向傳播
        lossD_real = criterion(output, labels)  # 計算真實數據的損失
        lossD_real.backward()  # 反向傳播以計算梯度

        # 生成假數據
        noise = torch.randn(batch_size, 100).to(device)  # 生成隨機噪聲
        text_embeds = torch.stack([text_embedding(prompt).to(device) for prompt in prompts])  # 將提示編碼為文本嵌入
        fake_data = netG(noise, text_embeds)  # 從噪聲和文本嵌入生成假數據
        labels = torch.zeros(batch_size, 1).to(device)  # 為假數據創建標籤（0）
        output = netD(fake_data.detach())  # 將假數據通過判別器進行前向傳播（detach 以避免梯度流回生成器）
        lossD_fake = criterion(output, labels)  # 計算假數據的損失
        lossD_fake.backward()  # 反向傳播以計算梯度
        optimizerD.step()  # 更新判別器參數

        # 更新生成器
        netG.zero_grad()  # 將生成器的梯度清零
        labels = torch.ones(batch_size, 1).to(device)  # 為假數據創建標籤（1）以欺騙判別器
        output = netD(fake_data)  # 將假數據（現在已更新）通過判別器進行前向傳播
        lossG = criterion(output, labels)  # 根據判別器的響應計算生成器的損失
        lossG.backward()  # 反向傳播以計算梯度
        optimizerG.step()  # 更新生成器參數

    # 打印 epoch 信息
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss D: {lossD_real + lossD_fake}, Loss G: {lossG}")

# 保存生成器模型的狀態字典到名為 'generator.pth' 的文件
torch.save(netG.state_dict(), 'generator.pth')

# 保存判別器模型的狀態字典到名為 'discriminator.pth' 的文件
torch.save(netD.state_dict(), 'discriminator.pth')

# 用於根據給定的文本提示生成視頻的推理函數
def generate_video(text_prompt, num_frames=10):
    # 根據文本提示為生成的視頻幀創建目錄
    os.makedirs(f'generated_video_{text_prompt.replace(" ", "_")}', exist_ok=True)

    # 將文本提示編碼為文本嵌入張量
    text_embed = text_embedding(prompt).to(device).unsqueeze(0)

    # 為視頻生成幀
    for frame_num in range(num_frames):
        # 生成隨機噪聲
        noise = torch.randn(1, 100).to(device)

        # 使用生成器網絡生成假幀
        with torch.no_grad():
            fake_frame = netG(noise, text_embed)

        # 將生成的假幀保存為圖像文件
        save_image(fake_frame, f'generated_video_{text_prompt.replace(" ", "_")}/frame_{frame_num}.png')

# 使用特定的文本提示使用 generate_video 函數
generate_video('circle moving up-right')

# 定義包含 PNG 幀的文件夾路徑
folder_path = 'generated_video_circle_moving_up-right'

# 獲取文件夾中所有 PNG 文件的列表
image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# 按名稱排序圖像（假設它們是按順序編號的）
image_files.sort()

# 創建一個列表來存儲幀
frames = []

# 讀取每個圖像並將其附加到幀列表中
for image_file in image_files:
  image_path = os.path.join(folder_path, image_file)
  frame = cv2.imread(image_path)
  frames.append(frame)

# 將幀列表轉換為 numpy 數組以便於處理
frames = np.array(frames)

# 定義幀率（每秒幀數）
fps = 10

# 創建視頻寫入器對象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('generated_video.avi', fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))

# 將每個幀寫入視頻
for frame in frames:
  out.write(frame)

# 釋放視頻寫入器
out.release() 