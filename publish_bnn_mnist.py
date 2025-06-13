import os
import torch
from huggingface_hub import HfApi, create_repo
from bayesian_neural_net import BayesianMLP
import json
import getpass

def save_model_to_hub(model_path, repo_name, model_name, description, token):
    """
    將貝葉斯神經網絡模型保存到Hugging Face Hub
    
    參數:
        model_path: 模型文件路徑
        repo_name: Hugging Face倉庫名稱
        model_name: 模型名稱
        description: 模型描述
        token: Hugging Face訪問令牌
    """
    # 初始化Hugging Face API
    api = HfApi(token=token)
    
    try:
        # 創建倉庫
        create_repo(repo_name, repo_type="model", exist_ok=True, token=token)
        print(f"已創建/確認倉庫: {repo_name}")
        
        # 載入模型
        print("正在加載模型...")
        model = BayesianMLP()
        
        # 嘗試加載模型權重
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
            print("模型加載成功")
        except Exception as e:
            print(f"加載模型時發生錯誤: {str(e)}")
            raise
        
        # 保存模型配置
        config = {
            "model_name": model_name,
            "description": description,
            "architecture": "BayesianMLP",
            "input_size": 784,  # MNIST 圖像大小
            "hidden_sizes": [400, 400],
            "output_size": 10,  # MNIST 類別數
            "prior": {
                "type": "Gaussian",
                "mu": 0.0,
                "sigma": 1.0
            },
            "variational": {
                "type": "Gaussian",
                "parametrization": "mean-log_std"
            }
        }
        
        # 保存模型和配置
        model_save_path = os.path.join("model_files", model_name)
        os.makedirs(model_save_path, exist_ok=True)
        
        # 保存模型權重
        torch.save(model.state_dict(), os.path.join(model_save_path, "pytorch_model.bin"))
        
        # 保存配置
        with open(os.path.join(model_save_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 創建 README
        readme_content = f"""# {model_name}

{description}

## 模型架構

- 輸入層：784 神經元（28x28 MNIST 圖像）
- 隱藏層1：400 神經元
- 隱藏層2：400 神經元
- 輸出層：10 神經元（數字 0-9）

## 貝葉斯特性

- 使用變分推斷進行參數估計
- 權重和偏置都服從高斯分佈
- 支持不確定性估計
- 使用蒙特卡羅採樣進行預測

## 使用方法

```python
from bayesian_neural_net import BayesianMLP
import torch

# 加載模型
model = BayesianMLP()
model.load_state_dict(torch.load("pytorch_model.bin"))

# 進行預測
x = torch.randn(1, 784)  # 示例輸入
probs, uncertainty = mc_predict(model, x, mc_runs=50)
```

## 訓練細節

- 優化器：Adam
- 學習率：1e-3
- 批次大小：128
- 訓練輪次：10
- 損失函數：證據下界（ELBO）
"""
        
        # 保存 README
        with open(os.path.join(model_save_path, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        # 上傳到Hub
        print("正在上傳文件到 Hugging Face Hub...")
        api.upload_folder(
            folder_path=model_save_path,
            repo_id=repo_name,
            repo_type="model",
            token=token
        )
        
        print(f"模型已成功上傳到 {repo_name}")
        print(f"模型配置：")
        print(json.dumps(config, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"上傳過程中發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    # 設置模型信息
    MODEL_PATH = "models/bnn_mnist.pt"
    REPO_NAME = "ewdlop/bnn-mnist"
    MODEL_NAME = "bnn-mnist-classifier"
    DESCRIPTION = """
    貝葉斯神經網絡模型用於MNIST手寫數字分類。
    該模型使用變分推斷來學習參數的不確定性，並能夠提供預測的不確定性估計。
    
    特點：
    - 使用貝葉斯神經網絡架構
    - 輸入：784維特徵（28x28 MNIST圖像）
    - 輸出：10個數字類別（0-9）
    - 支持不確定性估計
    """
    
    # 獲取用戶的Hugging Face訪問令牌
    print("請輸入您的Hugging Face訪問令牌（輸入時不會顯示）：")
    token = getpass.getpass()
    
    if not token:
        print("錯誤：未提供訪問令牌")
        exit(1)
    
    # 發布模型
    save_model_to_hub(MODEL_PATH, REPO_NAME, MODEL_NAME, DESCRIPTION, token) 