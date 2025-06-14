import torch
import os
import json
import getpass
from huggingface_hub import HfApi, create_repo

def create_config_file(model_path, save_dir):
    """
    創建模型配置文件
    
    參數：
        model_path: 模型文件路徑
        save_dir: 保存目錄
    返回：
        配置字典
    """
    # 從模型文件名獲取基本信息
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # 創建基本配置
    config = {
        "model_name": model_name,
        "input_size": 2312,  # 2 * 34 * 34 = 2312 (NMNIST 數據集)
        "hidden_size": 100,
        "output_size": 10,
        "time_window": 20,
        "description": "脈衝神經網路模型，用於 NMNIST 數據集分類"
    }
    
    # 保存配置文件
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print("✓ 成功創建配置文件")
    return config

def publish_snn_to_hub(model_path, model_name, username, token):
    """
    將脈衝神經網路模型發布到 Hugging Face Hub
    
    參數：
        model_path: 模型文件路徑（.pth 文件）
        model_name: 模型名稱
        username: Hugging Face 用戶名
        token: Hugging Face 訪問令牌
    """
    print("開始發布模型到 Hugging Face Hub...")
    
    # 創建倉庫
    repo_name = f"{username}/{model_name}"
    try:
        create_repo(repo_name, private=False)
        print(f"✓ 成功創建倉庫：{repo_name}")
    except Exception as e:
        print(f"! 倉庫可能已存在: {e}")

    # 準備上傳文件
    save_dir = "model_save"
    os.makedirs(save_dir, exist_ok=True)
    print(f"✓ 創建臨時保存目錄：{save_dir}")
    
    # 複製模型文件
    model_file = os.path.join(save_dir, "model.pt")
    config_file = os.path.join(save_dir, "config.json")
    
    # 加載或創建模型配置
    try:
        config_path = os.path.join(os.path.dirname(model_path), "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            print("✓ 成功加載模型配置")
        else:
            print("! 未找到配置文件，將創建新的配置文件")
            config = create_config_file(model_path, save_dir)
    except Exception as e:
        print(f"! 無法處理配置文件: {e}")
        return
    
    # 複製模型文件
    try:
        torch.save(torch.load(model_path), model_file)
        print("✓ 成功複製模型文件")
    except Exception as e:
        print(f"! 無法複製模型文件: {e}")
        return
    
    # 創建 README.md
    readme_content = f"""# {model_name}

這是一個使用 PyTorch 實現的脈衝神經網路模型，用於處理 NMNIST 數據集。

## 模型架構

- 輸入層：{config["input_size"]} 個神經元
- 隱藏層：{config["hidden_size"]} 個神經元
- 輸出層：{config["output_size"]} 個神經元
- 時間窗口：{config["time_window"]} 個時間步

## 模型特點

- 使用積分發放神經元（Integrate-and-Fire Neurons）
- 實現了生物神經元的基本特性：膜電位積分、閾值觸發和脈衝發放
- 使用代理梯度方法進行訓練
- 能夠處理時序數據
- 具有生物學合理性
- 能量效率高（只在必要時發放脈衝）

## 使用方法

```python
import torch
from spiking_neural_net import SimpleSpikingNeuralNetwork

# 加載模型
model = SimpleSpikingNeuralNetwork(
    input_size={config["input_size"]},
    hidden_size={config["hidden_size"]},
    output_size={config["output_size"]},
    time_window={config["time_window"]}
)
model.load_state_dict(torch.load("model.pt"))

# 使用模型進行預測
model.eval()
with torch.no_grad():
    predictions = model(input_data)
```

## 數據集

本模型使用 NMNIST 數據集進行訓練，這是一個基於事件的神經形態數據集，由 MNIST 數字通過事件相機記錄而成。

## 訓練參數

- 優化器：Adam
- 學習率：1e-3
- 批次大小：64
- 訓練輪數：5
"""
    
    try:
        with open(os.path.join(save_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)
        print("✓ 成功創建 README.md")
    except Exception as e:
        print(f"! 無法創建 README.md: {e}")
        return

    # 上傳到 Hugging Face
    try:
        api = HfApi()
        api.upload_folder(
            folder_path=save_dir,
            repo_id=repo_name,
            repo_type="model",
            token=token
        )
        print(f"✓ 模型已成功上傳到 https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"! 上傳過程中發生錯誤: {e}")
        return

def main():
    print("=== 脈衝神經網路模型發布工具 ===")
    print("此工具將幫助您將訓練好的脈衝神經網路模型發布到 Hugging Face Hub")
    print("請準備以下信息：")
    print("1. 模型文件路徑（.pth 文件）")
    print("2. 模型名稱")
    print("3. Hugging Face 用戶名")
    print("4. Hugging Face 訪問令牌")
    print("=" * 40)
    
    # 獲取用戶輸入
    model_path = input("\n請輸入模型文件路徑（例如：spiking_neural_network.pth）：").strip()
    model_name = input("請輸入模型名稱（例如：spiking-neural-network-nmnist）：").strip()
    username = input("請輸入您的 Hugging Face 用戶名：").strip()
    token = getpass.getpass("請輸入您的 Hugging Face 訪問令牌：").strip()

    # 驗證輸入
    if not all([model_path, model_name, username, token]):
        print("錯誤：所有字段都必須填寫")
        return

    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型文件：{model_path}")
        return

    # 發布模型
    publish_snn_to_hub(model_path, model_name, username, token)

if __name__ == "__main__":
    main() 