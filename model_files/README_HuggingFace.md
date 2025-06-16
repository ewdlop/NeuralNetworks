# 🤗 Hugging Face Hub 推送指南

這個指南將幫助您將訓練好的 Transformer 模型推送到 Hugging Face Hub，讓全世界都能使用您的模型！

## 🚀 快速開始

### 1. 準備工作

首先安裝所需依賴：
```bash
pip install -r requirements_transformer.txt
```

### 2. 獲取 Hugging Face Token

1. 訪問 [Hugging Face](https://huggingface.co/) 並註冊/登錄
2. 前往 [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. 創建一個新的 token（選擇 "Write" 權限）
4. 複製您的 token

### 3. 訓練並推送模型

#### 方法 1: 訓練時自動推送
```bash
python train_transformer.py \
    --mode train \
    --push_to_hf \
    --hf_repo_name "my-shakespeare-transformer" \
    --hf_username "your_username" \
    --hf_token "your_token_here"
```

#### 方法 2: 訓練後手動推送
```bash
# 先訓練模型
python train_transformer.py --mode train

# 然後推送到 Hugging Face
python push_to_huggingface.py \
    --repo_name "my-shakespeare-transformer" \
    --username "your_username" \
    --token "your_token_here"
```

## 📋 命令行參數

### 訓練腳本參數
- `--push_to_hf`: 訓練完成後自動推送
- `--hf_repo_name`: 倉庫名稱（默認: shakespeare-transformer）
- `--hf_username`: 您的 Hugging Face 用戶名
- `--hf_token`: 您的 Hugging Face token
- `--hf_private`: 創建私有倉庫

### 推送腳本參數
- `--model_path`: 模型文件路徑（默認: transformer_model.pth）
- `--repo_name`: 倉庫名稱
- `--username`: Hugging Face 用戶名
- `--token`: Hugging Face token
- `--private`: 創建私有倉庫
- `--commit_message`: 提交信息

## 📦 推送的文件內容

推送到 Hugging Face 後，您的倉庫將包含：

```
your-repo/
├── README.md                    # 模型卡片（自動生成）
├── config.json                  # 模型配置
├── pytorch_model.bin            # PyTorch 模型權重
├── vocab.json                   # 詞彙表
├── idx_to_vocab.json           # 索引到詞彙映射
├── training_args.json          # 訓練參數
├── requirements.txt            # 依賴列表
├── usage_example.py            # 使用示例
└── encoder_decoder_transformer.py  # 模型定義
```

## 🎯 模型使用示例

推送成功後，其他人可以這樣使用您的模型：

### 下載模型
```python
from huggingface_hub import hf_hub_download

# 下載模型文件
model_path = hf_hub_download("username/repo-name", "pytorch_model.bin")
config_path = hf_hub_download("username/repo-name", "config.json")
vocab_path = hf_hub_download("username/repo-name", "vocab.json")
```

### 載入和使用
```python
import torch
import json
from encoder_decoder_transformer import Transformer

# 載入配置和詞彙表
with open(config_path) as f:
    config = json.load(f)
    
with open(vocab_path) as f:
    vocab = json.load(f)

# 重建模型
model = Transformer(**config)

# 載入權重
checkpoint = torch.load(model_path, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

# 使用模型進行推理
model.eval()
# ... 推理代碼 ...
```

## 🔒 安全提示

1. **永遠不要在代碼中硬編碼您的 token**
2. **使用環境變量存儲敏感信息**：
   ```bash
   export HF_TOKEN="your_token_here"
   export HF_USERNAME="your_username"
   ```
3. **如果不想公開模型，使用 `--private` 參數**

## 🛠️ 故障排除

### 常見錯誤及解決方案

#### 1. Token 驗證失敗
```
❌ 推送失敗: HTTP 401 Unauthorized
```
**解決方案**: 檢查您的 token 是否正確，並確保具有 "Write" 權限

#### 2. 倉庫名稱衝突
```
❌ Repository already exists
```
**解決方案**: 更改倉庫名稱或添加版本號

#### 3. 網絡連接問題
```
❌ 推送失敗: Connection timeout
```
**解決方案**: 檢查網絡連接，或嘗試使用代理

#### 4. 文件大小限制
```
❌ File too large
```
**解決方案**: 考慮使用 Git LFS 或減小模型大小

## 🌟 最佳實踐

1. **使用描述性的倉庫名稱**：`shakespeare-transformer-v2` 而不是 `model1`
2. **填寫完整的模型卡片**：自動生成的 README.md 可以進一步編輯
3. **添加使用示例**：幫助其他人更好地使用您的模型
4. **定期更新**：改進模型後推送新版本
5. **添加適當的標籤**：便於其他人發現您的模型

## 📞 獲取幫助

如果遇到問題：
1. 檢查 [Hugging Face 文檔](https://huggingface.co/docs)
2. 查看 [社區論壇](https://discuss.huggingface.co/)
3. 檢查您的網絡連接和 token 權限

## 🎉 恭喜！

成功推送模型後，您的模型將在以下地址可用：
`https://huggingface.co/your_username/your_repo_name`

現在全世界的人都可以使用您訓練的莎士比亞風格文本生成模型了！🎭📚 