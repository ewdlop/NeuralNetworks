import torch
import json
import os
import getpass  # 添加安全輸入模組
from huggingface_hub import HfApi, Repository, create_repo
from encoder_decoder_transformer import Transformer
import shutil

def create_model_card(model_name, vocab_size, d_model, n_heads, n_layers, dataset_info):
    """創建模型卡片"""
    model_card = f"""---
language: en
license: mit
tags:
- transformer
- text-generation
- shakespeare
- pytorch
datasets:
- custom
widget:
- text: "To be or not to be"
  example_title: "Shakespeare Style"
---

# {model_name}

## 模型描述

這是一個基於 Transformer 架構的文本生成模型，使用莎士比亞文本進行訓練。

## 模型架構

- **模型類型**: Encoder-Decoder Transformer
- **詞彙表大小**: {vocab_size:,}
- **模型維度**: {d_model}
- **注意力頭數**: {n_heads}
- **編碼器層數**: {n_layers}
- **解碼器層數**: {n_layers}

## 訓練數據

{dataset_info}

## 使用方法

```python
import torch
from transformers import AutoModel

# 載入模型
model = AutoModel.from_pretrained("{model_name}")

# 進行推理（需要自定義推理代碼）
# 詳見倉庫中的使用示例
```

## 訓練細節

- **優化器**: Adam
- **學習率調度**: StepLR
- **訓練框架**: PyTorch
- **任務類型**: 文本生成（復述任務）

## 限制和偏見

- 模型在莎士比亞文本上訓練，可能會反映該時代的語言特點
- 適用於古典英文文本生成，現代文本效果可能有限

```
"""
    return model_card

def create_config_json(config):
    """創建配置文件"""
    hf_config = {
        "architectures": ["Transformer"],
        "model_type": "transformer",
        "vocab_size": config["src_vocab_size"],
        "d_model": config["d_model"],
        "n_heads": config["n_heads"],
        "n_encoder_layers": config["n_encoder_layers"],
        "n_decoder_layers": config["n_decoder_layers"],
        "d_ff": config["d_ff"],
        "dropout": config["dropout"],
        "pad_token_id": config["pad_idx"],
        "bos_token_id": 2,  # 假設 BOS token ID
        "eos_token_id": 3,  # 假設 EOS token ID
        "max_position_embeddings": 512,
        "torch_dtype": "float32",
        "transformers_version": "4.0.0"
    }
    return hf_config

def create_training_args_json():
    """創建訓練參數文件"""
    training_args = {
        "output_dir": "./results",
        "num_train_epochs": 10,
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "logging_steps": 100,
        "save_steps": 1000,
        "eval_steps": 500,
        "warmup_steps": 500,
        "max_grad_norm": 1.0,
        "fp16": False,
        "dataloader_num_workers": 4,
        "load_best_model_at_end": True,
        "metric_for_best_model": "loss",
        "greater_is_better": False,
        "report_to": [],
        "seed": 42
    }
    return training_args

def create_usage_example():
    """創建使用示例文件"""
    example_code = '''
import torch
from encoder_decoder_transformer import Transformer
import json

# 載入模型和配置
def load_model(model_path):
    checkpoint = torch.load(f"{model_path}/pytorch_model.bin", map_location="cpu")
    
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    
    # 重建模型
    model = Transformer(
        src_vocab_size=config["vocab_size"],
        tgt_vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_encoder_layers=config["n_encoder_layers"],
        n_decoder_layers=config["n_decoder_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        pad_idx=config["pad_token_id"]
    )
    
    # 載入權重
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, checkpoint["vocab"], checkpoint["idx_to_vocab"]

# 使用示例
def generate_text(model, vocab, idx_to_vocab, input_text, max_length=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 簡單的tokenization
    words = input_text.lower().split()
    tokens = [vocab.get("<BOS>", 2)] + [vocab.get(word, vocab.get("<UNK>", 1)) for word in words] + [vocab.get("<EOS>", 3)]
    
    # 填充
    max_len = 64
    if len(tokens) < max_len:
        tokens += [vocab.get("<PAD>", 0)] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    
    src = torch.tensor(tokens).unsqueeze(0).to(device)
    
    # 生成
    with torch.no_grad():
        generated = model.generate(src, max_len=max_length, start_token=2, end_token=3)
    
    # 轉換為文本
    words = []
    for token in generated[0]:
        word = idx_to_vocab.get(token.item(), "<UNK>")
        if word in ["<PAD>", "<BOS>", "<EOS>"]:
            if word == "<EOS>":
                break
            continue
        words.append(word)
    
    return " ".join(words)

# 示例使用
if __name__ == "__main__":
    model, vocab, idx_to_vocab = load_model("./")
    
    input_texts = [
        "To be or not to be",
        "What is your name",
        "The king is dead"
    ]
    
    for text in input_texts:
        generated = generate_text(model, vocab, idx_to_vocab, text)
        print(f"輸入: {text}")
        print(f"生成: {generated}")
        print("-" * 50)
'''
    return example_code

def push_to_huggingface(
    model_path="transformer_model.pth",
    repo_name="shakespeare-transformer",
    username=None,
    token=None,
    private=False,
    commit_message="Add Shakespeare Transformer model"
):
    """
    將模型推送到 Hugging Face Hub
    
    參數:
    - model_path: 本地模型文件路徑
    - repo_name: 倉庫名稱
    - username: Hugging Face 用戶名
    - token: Hugging Face token
    - private: 是否為私有倉庫
    - commit_message: 提交信息
    """
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 安全地獲取用戶名和token
    if username is None:
        username = input("請輸入您的 Hugging Face 用戶名: ")
    
    if token is None:
        print("請輸入您的 Hugging Face token（輸入時不會顯示）:")
        print("💡 提示: 您可以在 https://huggingface.co/settings/tokens 獲取 token")
        token = getpass.getpass("Token: ")
        
        # 驗證 token 不為空
        while not token.strip():
            print("❌ Token 不能為空，請重新輸入:")
            token = getpass.getpass("Token: ")
    
    # 完整的倉庫名稱
    full_repo_name = f"{username}/{repo_name}"
    
    try:
        # 初始化 API
        api = HfApi()
        
        # 載入模型檢查點
        print("載入模型...")
        checkpoint = torch.load(model_path, map_location="cpu")
        config = checkpoint["config"]
        vocab = checkpoint["vocab"]
        idx_to_vocab = checkpoint["idx_to_vocab"]
        
        # 創建倉庫
        print(f"創建倉庫: {full_repo_name}")
        try:
            create_repo(
                repo_id=full_repo_name,
                token=token,
                private=private,
                exist_ok=True
            )
        except Exception as e:
            print(f"倉庫創建警告: {e}")
        
        # 創建臨時目錄
        temp_dir = f"./temp_{repo_name}"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        try:
            # 克隆倉庫
            print("克隆倉庫...")
            repo = Repository(
                local_dir=temp_dir,
                clone_from=full_repo_name,
                use_auth_token=token
            )
            
            # 保存 PyTorch 模型權重
            print("保存模型文件...")
            torch.save({
                "model_state_dict": checkpoint["model_state_dict"],
                "vocab": vocab,
                "idx_to_vocab": idx_to_vocab
            }, os.path.join(temp_dir, "pytorch_model.bin"))
            
            # 創建配置文件
            print("創建配置文件...")
            hf_config = create_config_json(config)
            with open(os.path.join(temp_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(hf_config, f, indent=2, ensure_ascii=False)
            
            # 保存詞彙表
            print("保存詞彙表...")
            with open(os.path.join(temp_dir, "vocab.json"), "w", encoding="utf-8") as f:
                json.dump(vocab, f, indent=2, ensure_ascii=False)
            
            with open(os.path.join(temp_dir, "idx_to_vocab.json"), "w", encoding="utf-8") as f:
                json.dump(idx_to_vocab, f, indent=2, ensure_ascii=False)
            
            # 創建訓練參數文件
            training_args = create_training_args_json()
            with open(os.path.join(temp_dir, "training_args.json"), "w", encoding="utf-8") as f:
                json.dump(training_args, f, indent=2)
            
            # 創建模型卡片
            print("創建模型卡片...")
            dataset_info = f"使用莎士比亞文本進行訓練，包含 {len(checkpoint.get('dataset_info', {}).get('sentences', []))} 個句子"
            model_card = create_model_card(
                full_repo_name, 
                config["src_vocab_size"], 
                config["d_model"], 
                config["n_heads"], 
                config["n_encoder_layers"],
                dataset_info
            )
            with open(os.path.join(temp_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write(model_card)
            
            # 創建使用示例文件
            print("創建使用示例...")
            example_code = create_usage_example()
            with open(os.path.join(temp_dir, "usage_example.py"), "w", encoding="utf-8") as f:
                f.write(example_code)
            
            # 複製模型定義文件
            if os.path.exists("encoder_decoder_transformer.py"):
                shutil.copy("encoder_decoder_transformer.py", temp_dir)
                print("複製模型定義文件...")
            
            # 創建需求文件
            requirements = """torch>=2.0.0
transformers>=4.21.0
numpy>=1.21.0
huggingface_hub>=0.16.0
"""
            with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
                f.write(requirements)
            
            # 提交和推送
            print("提交更改...")
            repo.git_add()
            repo.git_commit(commit_message)
            repo.git_push()
            
            print(f"✅ 模型成功推送到 Hugging Face Hub!")
            print(f"🔗 倉庫連結: https://huggingface.co/{full_repo_name}")
            print(f"📦 模型可以通過以下方式載入:")
            print(f"   from huggingface_hub import hf_hub_download")
            print(f"   hf_hub_download('{full_repo_name}', 'pytorch_model.bin')")
            
        finally:
            # 清理臨時目錄
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print("清理臨時文件...")
                
    except Exception as e:
        print(f"❌ 推送失敗: {str(e)}")
        print("請檢查:")
        print("1. Hugging Face token 是否正確")
        print("2. 用戶名是否正確")
        print("3. 網絡連接是否正常")
        print("4. 是否已安裝 huggingface_hub: pip install huggingface_hub")
        raise

def get_user_credentials():
    """安全地獲取用戶憑證"""
    print("🤗 Hugging Face 登錄")
    print("-" * 30)
    
    username = input("用戶名 (ewdlop): ").strip()
    if not username:
        username = "ewdlop"  # 默認用戶名
    
    print("\n💡 獲取 Token 的步驟:")
    print("1. 訪問 https://huggingface.co/settings/tokens")
    print("2. 點擊 'New token'")
    print("3. 選擇 'Write' 權限")
    print("4. 複製生成的 token")
    print()
    
    token = getpass.getpass("請輸入您的 Hugging Face token (輸入時不會顯示): ")
    
    while not token.strip():
        print("❌ Token 不能為空，請重新輸入:")
        token = getpass.getpass("Token: ")
    
    # 確認信息
    print(f"\n✅ 用戶名: {username}")
    print("✅ Token: [已設置]")
    
    confirm = input("\n確認信息正確嗎? (Y/n): ").lower().strip()
    if confirm in ['n', 'no']:
        print("重新輸入...")
        return get_user_credentials()
    
    return username, token

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="推送 Transformer 模型到 Hugging Face Hub")
    parser.add_argument("--model_path", default="transformer_model.pth", 
                       help="模型文件路徑")
    parser.add_argument("--repo_name", default="shakespeare-transformer", 
                       help="倉庫名稱")
    parser.add_argument("--username", default="ewdlop", help="Hugging Face 用戶名")
    parser.add_argument("--token", help="Hugging Face token (不建議在命令行中使用)")
    parser.add_argument("--private", action="store_true", 
                       help="創建私有倉庫")
    parser.add_argument("--commit_message", 
                       default="Add Shakespeare Transformer model",
                       help="提交信息")
    parser.add_argument("--interactive", action="store_true",
                       help="交互式輸入憑證")
    
    args = parser.parse_args()
    
    print("🤗 準備推送模型到 Hugging Face Hub...")
    print(f"模型文件: {args.model_path}")
    print(f"倉庫名稱: {args.repo_name}")
    print(f"私有倉庫: {args.private}")
    print()
    
    # 獲取憑證
    username = args.username
    token = args.token
    
    # 如果使用交互模式或沒有提供 token
    if args.interactive or not token:
        username, token = get_user_credentials()
    
    push_to_huggingface(
        model_path=args.model_path,
        repo_name=args.repo_name,
        username=username,
        token=token,
        private=args.private,
        commit_message=args.commit_message
    )

if __name__ == "__main__":
    main() 