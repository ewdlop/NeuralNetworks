"""
模型训练器
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
import time
from tqdm import tqdm
import wandb


class Trainer:
    """
    PyTorch 模型训练器
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        config: Optional[Dict] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config or {}
        
        # 默认损失函数
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        self.criterion = criterion.to(device)
        
        # 默认优化器
        if optimizer is None:
            lr = self.config.get('learning_rate', 1e-3)
            optimizer = optim.Adam(model.parameters(), lr=lr)
        self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # 早停参数
        self.best_val_loss = float('inf')
        self.patience = self.config.get('patience', 10)
        self.patience_counter = 0
        
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> tuple[float, float]:
        """验证模型"""
        if self.val_loader is None:
            return 0.0, 0.0
            
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        num_epochs: int,
        save_path: Optional[str] = None,
        use_wandb: bool = False
    ) -> Dict[str, List]:
        """完整训练流程"""
        
        if use_wandb:
            wandb.init(project="pytorch-training", config=self.config)
            wandb.watch(self.model)
        
        print(f"开始训练，设备: {self.device}")
        print(f"模型参数数量: {self.model.get_num_parameters():,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            # 打印结果
            print(f"训练损失: {train_loss:.4f}")
            if self.val_loader:
                print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            
            # WandB 日志
            if use_wandb:
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                if self.val_loader:
                    log_dict.update({
                        'val_loss': val_loss,
                        'val_accuracy': val_acc
                    })
                wandb.log(log_dict)
            
            # 早停检查
            if self.val_loader and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # 保存最佳模型
                if save_path:
                    self.model.save_model(f"{save_path}_best.pth")
                    print(f"保存最佳模型到 {save_path}_best.pth")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                print(f"早停：验证损失在 {self.patience} 个epoch内没有改善")
                break
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总用时: {total_time:.2f} 秒")
        
        # 保存最终模型
        if save_path:
            self.model.save_model(f"{save_path}_final.pth")
            print(f"保存最终模型到 {save_path}_final.pth")
        
        if use_wandb:
            wandb.finish()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
    
    def predict(self, data_loader: DataLoader) -> List:
        """模型预测"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for data, _ in tqdm(data_loader, desc='Predicting'):
                data = data.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return predictions 
