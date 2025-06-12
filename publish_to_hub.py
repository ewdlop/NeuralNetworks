import os
import torch
from huggingface_hub import HfApi, create_repo
from liquid_net import LiquidNet
import json
import getpass

def save_model_to_hub(model_path, repo_name, model_name, description, token):
    """
    將模型保存到Hugging Face Hub
    
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
        checkpoint = torch.load(model_path, map_location='cpu')
        model = LiquidNet(
            input_size=9,
            hidden_size=128,
            output_size=6,
            steps=128
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 保存模型配置
        config = {
            "model_name": model_name,
            "description": description,
            "architecture": "LiquidNet",
            "input_size": 9,
            "hidden_size": 128,
            "output_size": 6,
            "steps": 128,
            "training_metrics": {
                "best_val_acc": checkpoint['val_acc'],
                "epoch": checkpoint['epoch']
            }
        }
        
        # 保存模型和配置
        model_save_path = os.path.join("model_files", model_name)
        os.makedirs(model_save_path, exist_ok=True)
        
        # 保存模型權重
        torch.save(model.state_dict(), os.path.join(model_save_path, "pytorch_model.bin"))
        
        # 保存配置
        with open(os.path.join(model_save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # 上傳到Hub
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

if __name__ == "__main__":
    # 設置模型信息
    MODEL_PATH = "models/best_liquid_har_model.pth"
    REPO_NAME = "ewdlop/liquidnet-har"
    MODEL_NAME = "liquidnet-har-classifier"
    DESCRIPTION = """
    LiquidNet模型用於UCI HAR數據集的人體活動識別。
    該模型使用9個原始感測器信號（3軸加速度計和3軸陀螺儀）進行訓練，
    可以識別6種不同的人體活動。
    
    特點：
    - 使用LiquidNet架構處理時間序列數據
    - 輸入：9個特徵（3軸加速度計和3軸陀螺儀）
    - 輸出：6種活動類別
    - 時間步長：128
    """
    
    # 獲取用戶的Hugging Face訪問令牌
    print("請輸入您的Hugging Face訪問令牌（輸入時不會顯示）：")
    token = getpass.getpass()
    
    if not token:
        print("錯誤：未提供訪問令牌")
        exit(1)
    
    # 發布模型
    save_model_to_hub(MODEL_PATH, REPO_NAME, MODEL_NAME, DESCRIPTION, token) 