import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import os
from huggingface_hub import HfApi, Repository
from transformers import AutoModel, AutoTokenizer
import json

class GCN(torch.nn.Module):
    """
    圖卷積神經網絡 (Graph Convolutional Network)
    
    參數:
        input_dim (int): 輸入特徵維度
        hidden_dim (int): 隱藏層維度
        output_dim (int): 輸出類別數
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim) # 第一層圖卷積層
        self.conv2 = GCNConv(hidden_dim, output_dim) # 第二層圖卷積層

    def forward(self, data):
        """
        前向傳播
        
        參數:
            data: 包含節點特徵 x 和邊索引 edge_index 的數據對象
        返回:
            log_softmax 後的預測結果
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index)) # 第一層圖卷積層
        x = F.dropout(x, training=self.training) # 丟棄層
        x = self.conv2(x, edge_index) # 第二層圖卷積層
        return F.log_softmax(x, dim=1) # 輸出層 ，dim=1 是將輸出結果的維度設置為 1 ，log_softmax 是將輸出結果轉換為 log 軟最大值

def train_model(model, data, optimizer, epochs=200):
    """
    訓練模型
    
    參數:
        model: GCN 模型
        data: 圖數據
        optimizer: 優化器
        epochs: 訓練輪數
    """
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        # 訓練
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # 評估
        model.eval()
        logits = model(data)
        pred = logits.argmax(dim=1)
        
        # 計算準確率
        train_acc = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
        val_acc = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    # 載入最佳模型
    model.load_state_dict(best_model_state)
    return model

def save_to_huggingface(model, dataset, repo_name, token):
    """
    將模型保存到 Hugging Face Hub
    
    參數:
        model: 訓練好的模型
        dataset: 數據集信息
        repo_name: Hugging Face 倉庫名稱
        token: Hugging Face API token
    """
    # 創建倉庫
    api = HfApi()
    api.create_repo(repo_name, token=token, private=False)
    
    # 保存模型
    save_dir = f"models/{repo_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型狀態
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': dataset.num_node_features,
        'hidden_dim': 16,
        'output_dim': dataset.num_classes
    }, f"{save_dir}/model.pt")
    
    # 保存模型配置
    config = {
        'input_dim': dataset.num_node_features,
        'hidden_dim': 16,
        'output_dim': dataset.num_classes,
        'dataset': 'Cora'
    }
    
    with open(f"{save_dir}/config.json", 'w') as f:
        json.dump(config, f)
    
    # 上傳到 Hugging Face
    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_name,
        token=token
    )

def main():
    # 設置隨機種子
    torch.manual_seed(42)
    
    # 載入數據集
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    # 建立模型
    model = GCN(
        input_dim=dataset.num_node_features,
        hidden_dim=16,
        output_dim=dataset.num_classes
    )
    
    # 設置優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 訓練模型
    model = train_model(model, data, optimizer)
    
    # 獲取 Hugging Face token
    token = input("請輸入你的 Hugging Face token: ")
    repo_name = input("請輸入要創建的倉庫名稱 (格式: username/repo-name): ")
    
    # 保存到 Hugging Face
    save_to_huggingface(model, dataset, repo_name, token)
    print(f"模型已成功上傳到 https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    main() 