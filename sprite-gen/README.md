# 🎮 Sprite Sheet 生成器

一個強大的 AI 驅動的 Sprite Sheet 生成工具，結合 ControlNet + Stable Diffusion + Pose 模型，將單張角色圖片轉換為完整的動畫 Sprite Sheet。

## ✨ 功能特點

- **🤖 AI 驅動**: 使用 Stable Diffusion 和 ControlNet 技術
- **🎯 姿勢控制**: 精確的 OpenPose 關鍵點操作
- **🎬 動畫生成**: 自動生成行走、跳躍、攻擊等動畫序列
- **📐 靈活配置**: 可自定義精靈圖尺寸、間距、動畫類型
- **🎨 風格保持**: 保持原始角色的視覺風格和特徵
- **📊 元數據輸出**: 包含動畫信息的詳細元數據文件

## 🛠️ 系統要求

### 硬體要求
- **GPU**: NVIDIA GPU (6GB+ VRAM 推薦)
- **記憶體**: 12GB+ RAM
- **存儲**: 10GB+ 可用空間

### 軟體要求
- **Python**: 3.8+
- **CUDA**: 11.8+ (如果使用 GPU)
- **操作系統**: Windows 10+, Linux, macOS

## 📦 安裝指南

### 1. 克隆專案
```bash
git clone <repository-url>
cd sprite-gen
```

### 2. 創建虛擬環境
```bash
# 使用 conda
conda create -n sprite-gen python=3.9
conda activate sprite-gen

# 或使用 venv
python -m venv sprite-gen
# Windows
sprite-gen\Scripts\activate
# Linux/macOS
source sprite-gen/bin/activate
```

### 3. 安裝依賴項
```bash
pip install -r requirements.txt
```

### 4. 驗證安裝
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"
```

## 🚀 快速開始

### 基本使用

```bash
python sprite-gen.py --input_image character.png
```

這會：
1. 載入角色圖像
2. 提取姿勢信息
3. 生成標準動畫序列 (idle, walk, jump, attack)
4. 輸出 Sprite Sheet 和元數據

### 自定義配置

```bash
python sprite-gen.py \
    --input_image character.png \
    --output_dir ./my_sprites \
    --character_prompt "pixel art warrior, blue armor, fantasy character" \
    --sprite_size 128 128 \
    --animations idle walk jump attack magic \
    --padding 4
```

## 📖 詳細使用說明

### 命令行參數

#### 基本參數
- `--input_image`: 輸入角色圖像路徑 (必需)
- `--output_dir`: 輸出目錄 (預設: `./sprite_output`)
- `--character_prompt`: 角色描述 (留空自動生成)

#### Sprite Sheet 設置
- `--sprite_size WIDTH HEIGHT`: 單個精靈圖尺寸 (預設: 64x64)
- `--animations`: 動畫類型列表 (預設: idle walk jump attack)
- `--padding`: 精靈圖間距 (預設: 2)

#### 生成品質
- `--num_inference_steps`: 推理步數 (預設: 20)
- `--guidance_scale`: 引導比例 (預設: 7.5)
- `--controlnet_scale`: ControlNet 影響強度 (預設: 1.0)

#### 模型設置
- `--model_id`: Stable Diffusion 模型 (預設: runwayml/stable-diffusion-v1-5)
- `--controlnet_id`: ControlNet 模型 (預設: lllyasviel/sd-controlnet-openpose)
- `--device`: 計算設備 (預設: cuda)

### 支持的動畫類型

- **idle**: 靜態站立姿勢
- **walk**: 8幀行走循環
- **jump**: 6幀跳躍序列 (準備→空中→落地)
- **attack**: 4幀攻擊動作
- **run**: 快速跑步動畫
- **crouch**: 蹲下姿勢
- **death**: 倒下動畫

### 輸出文件結構

```
sprite_output/
├── sprite_sheet.png          # 主要的 Sprite Sheet
├── sprite_sheet_metadata.json # 動畫元數據
└── individual_frames/         # 單獨的幀文件 (可選)
    ├── idle_000.png
    ├── walk_000.png
    └── ...
```

### 元數據格式

```json
{
  "sprite_size": [64, 64],
  "animations": {
    "idle": {
      "start_index": 0,
      "frame_count": 1,
      "fps": 4
    },
    "walk": {
      "start_index": 1,
      "frame_count": 8,
      "fps": 8
    }
  },
  "total_frames": 19,
  "sheet_size": [512, 192]
}
```

## 🎯 使用技巧

### 1. 輸入圖像準備
- **分辨率**: 512x512 或更高
- **格式**: PNG, JPG, JPEG
- **背景**: 單色背景效果最佳
- **姿勢**: 清晰的正面或側面姿勢

### 2. 角色描述優化
```bash
# 良好的描述例子
--character_prompt "pixel art knight, silver armor, medieval fantasy, 2D game character, detailed sprite"

# 避免的描述
--character_prompt "person"  # 太籠統
```

### 3. 品質設置
```bash
# 快速測試
--num_inference_steps 15 --guidance_scale 5.0

# 平衡品質
--num_inference_steps 20 --guidance_scale 7.5

# 高品質
--num_inference_steps 50 --guidance_scale 10.0
```

### 4. 記憶體優化
```bash
# 如果遇到記憶體不足
--sprite_size 32 32  # 減小精靈圖尺寸
--device cpu        # 使用 CPU (較慢)
```

## 🔧 進階功能

### 自定義姿勢序列

```python
# 使用 pose_manipulation.py
from pose_manipulation import PoseManipulator

manipulator = PoseManipulator((512, 512))
base_pose = manipulator.create_default_pose()

# 創建自定義動畫
custom_poses = manipulator.create_walking_animation(base_pose, steps=12)
manipulator.save_animation_sequence(custom_poses, "./custom_output", "custom_walk")
```

### 批量處理

```bash
# 批量處理多個角色
for file in characters/*.png; do
    python sprite-gen.py --input_image "$file" --output_dir "./batch_output/$(basename "$file" .png)"
done
```

### 遊戲引擎整合

生成的 Sprite Sheet 可以直接用於：
- **Unity**: 使用 Sprite Editor 切割
- **Godot**: 導入為 SpriteFrames 資源
- **Game Maker**: 使用 Sprite 編輯器
- **Phaser**: 配合 JSON 元數據載入

## 🔍 故障排除

### 常見問題

#### 1. 記憶體不足錯誤
```
RuntimeError: CUDA out of memory
```
**解決方案**:
```bash
# 減少精靈圖尺寸
--sprite_size 32 32

# 使用 CPU
--device cpu

# 減少推理步數
--num_inference_steps 15
```

#### 2. 姿勢檢測失敗
**症狀**: 生成的精靈圖姿勢不正確
**解決方案**:
- 確保輸入圖像有清晰的人物輪廓
- 嘗試調整 `--controlnet_scale` 參數
- 使用簡單背景的圖像

#### 3. 風格不一致
**症狀**: 生成的精靈圖風格差異很大
**解決方案**:
- 提供更詳細的 `--character_prompt`
- 降低 `--guidance_scale` 值
- 使用風格一致的輸入圖像

#### 4. 動畫不流暢
**症狀**: 動畫幀之間跳躍明顯
**解決方案**:
- 增加動畫幀數 (修改代碼中的 steps 參數)
- 調整姿勢插值算法
- 使用更高的 ControlNet 影響強度

### 除錯工具

```bash
# 檢查姿勢檢測
python pose_manipulation.py  # 生成測試姿勢

# 驗證模型載入
python -c "from sprite_gen import SpriteSheetGenerator; print('Models loaded successfully')"
```

## 🎨 範例畫廊

### 輸入 vs 輸出

| 輸入圖像 | 生成的 Sprite Sheet | 動畫類型 |
|---------|-------------------|----------|
| 騎士角色 | 8x4 網格 | 行走、攻擊、防禦 |
| 法師角色 | 6x3 網格 | 施法、移動、待機 |
| 忍者角色 | 10x2 網格 | 跳躍、攻擊、隱身 |

### 支持的風格

- 🎮 像素藝術風格
- 🎨 手繪卡通風格  
- ⚔️ 奇幻 RPG 風格
- 🚀 科幻風格
- 🏰 中世紀風格

## 📚 API 參考

### SpriteSheetGenerator 類

```python
from sprite_gen import SpriteSheetGenerator

generator = SpriteSheetGenerator(
    model_id="runwayml/stable-diffusion-v1-5",
    controlnet_id="lllyasviel/sd-controlnet-openpose",
    device="cuda"
)

sprite_sheet = generator.create_sprite_sheet(
    base_image=base_image,
    character_prompt="your character description",
    animations=["idle", "walk", "jump"],
    sprite_size=(64, 64),
    padding=2
)
```

### PoseManipulator 類

```python
from pose_manipulation import PoseManipulator

manipulator = PoseManipulator((512, 512))
base_pose = manipulator.create_default_pose()
walk_poses = manipulator.create_walking_animation(base_pose, 8)
```

## 🤝 貢獻指南

我們歡迎社群貢獻！您可以：

1. 🐛 報告錯誤
2. 💡 提出新功能建議
3. 📝 改進文檔
4. 🎨 分享範例作品
5. 🔧 提交代碼改進

## 📄 授權

本專案僅供學習和研究使用。生成的內容請遵守相關版權法律。

## 🙏 致謝

- Stable Diffusion 團隊
- ControlNet 開發者
- OpenPose 專案
- Hugging Face 社群

---

**開始創建您的專屬角色動畫吧！** 🎮✨

## 📞 聯繫支援

如有問題或建議，請：
- 創建 GitHub Issue
- 查看 FAQ 部分
- 參考故障排除指南

祝您使用愉快！ 