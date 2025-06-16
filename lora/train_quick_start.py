#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mega Man LoRA 快速開始訓練腳本
使用預設的最佳配置進行訓練
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse

def check_requirements():
    """檢查必要的依賴項"""
    required_packages = [
        'torch', 'torchvision', 'diffusers', 'transformers', 
        'accelerate', 'peft', 'PIL', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package == 'PIL':
                missing_packages.append('Pillow')
            else:
                missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少必要的依賴項: {', '.join(missing_packages)}")
        print("請運行: pip install -r requirements_megaman_lora.txt")
        return False
    
    print("✅ 所有依賴項已安裝")
    return True

def setup_data_directory(data_dir):
    """設置數據目錄"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"📁 創建數據目錄: {data_dir}")
        data_path.mkdir(parents=True)
        
        # 創建說明文件
        readme_content = """
# Mega Man 訓練圖像目錄

請將您的 Mega Man 圖像放在這個目錄中：

## 支持的格式：
- .jpg, .jpeg, .png, .bmp, .webp

## 建議：
- 圖像分辨率：512x512 或更高
- 圖像數量：20+ 張 (越多越好)
- 每張圖像可以有對應的 .txt 描述文件

## 示例描述：
- "mega man, blue robot, helmet, game character"
- "mega man X, futuristic armor, action pose"
- "robot master, colorful design, game boss"

準備好圖像後，重新運行此腳本開始訓練。
"""
        with open(data_path / "README.txt", "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        print(f"📝 已創建說明文件: {data_path / 'README.txt'}")
        return False
    
    # 檢查是否有圖像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(data_path.rglob(f'*{ext}')))
    
    if len(image_files) == 0:
        print(f"❌ 在 {data_dir} 中沒有找到圖像文件！")
        print("請添加一些 Mega Man 圖像到數據目錄中")
        return False
    
    print(f"✅ 找到 {len(image_files)} 張圖像，可以開始訓練")
    return True

def run_training(args):
    """運行訓練"""
    print("🚀 開始 Mega Man LoRA 訓練...")
    
    # 構建訓練命令
    cmd = [
        sys.executable, "megaman_lora.py",
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--resolution", str(args.resolution),
        "--train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.grad_accum),
        "--learning_rate", str(args.learning_rate),
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--max_train_steps", str(args.max_steps),
        "--save_steps", str(args.save_steps),
        "--validation_steps", str(args.validation_steps),
        "--mixed_precision", args.mixed_precision,
    ]
    
    if args.seed:
        cmd.extend(["--seed", str(args.seed)])
    
    print(f"📝 訓練命令: {' '.join(cmd)}")
    print("=" * 60)
    
    # 運行訓練
    try:
        subprocess.run(cmd, check=True)
        print("=" * 60)
        print("✅ 訓練完成！")
        print(f"📁 模型已保存到: {args.output_dir}")
        
        # 提示下一步
        print("\n🎨 現在您可以生成圖像:")
        print(f"python generate_megaman.py --lora_path {args.output_dir}/lora_weights")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 訓練失敗: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ 訓練被用戶中斷")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Mega Man LoRA 快速訓練")
    
    # 基本設置
    parser.add_argument("--data_dir", type=str, default="./megaman_images",
                       help="訓練數據目錄 (預設: ./megaman_images)")
    parser.add_argument("--output_dir", type=str, default="./megaman_lora_output",
                       help="輸出目錄 (預設: ./megaman_lora_output)")
    
    # 訓練配置預設 (初學者友好)
    parser.add_argument("--config", type=str, default="balanced",
                       choices=["fast", "balanced", "quality"],
                       help="訓練配置預設")
    
    # 可選的自定義參數
    parser.add_argument("--resolution", type=int, default=None,
                       help="圖像分辨率 (預設會根據配置選擇)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="批次大小")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="最大訓練步數")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="學習率")
    parser.add_argument("--seed", type=int, default=42,
                       help="隨機種子 (預設: 42)")
    
    args = parser.parse_args()
    
    # 根據配置設置預設值
    configs = {
        "fast": {
            "resolution": 512,
            "batch_size": 1,
            "grad_accum": 2,
            "learning_rate": 1e-4,
            "lora_rank": 4,
            "lora_alpha": 16,
            "max_steps": 800,
            "save_steps": 200,
            "validation_steps": 100,
            "mixed_precision": "fp16",
        },
        "balanced": {
            "resolution": 512,
            "batch_size": 1,
            "grad_accum": 4,
            "learning_rate": 8e-5,
            "lora_rank": 8,
            "lora_alpha": 32,
            "max_steps": 1500,
            "save_steps": 250,
            "validation_steps": 100,
            "mixed_precision": "fp16",
        },
        "quality": {
            "resolution": 768,
            "batch_size": 1,
            "grad_accum": 8,
            "learning_rate": 5e-5,
            "lora_rank": 16,
            "lora_alpha": 64,
            "max_steps": 2500,
            "save_steps": 300,
            "validation_steps": 150,
            "mixed_precision": "fp16",
        }
    }
    
    config = configs[args.config]
    
    # 應用預設值 (如果沒有手動指定)
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    
    print("🤖 Mega Man LoRA 快速訓練助手 🎮")
    print("=" * 50)
    print(f"📊 配置: {args.config}")
    print(f"📁 數據目錄: {args.data_dir}")
    print(f"📁 輸出目錄: {args.output_dir}")
    print(f"🖼️  分辨率: {args.resolution}")
    print(f"📦 批次大小: {args.batch_size}")
    print(f"🔄 梯度累積: {args.grad_accum}")
    print(f"📈 學習率: {args.learning_rate}")
    print(f"🔢 LoRA Rank: {args.lora_rank}")
    print(f"🔢 LoRA Alpha: {args.lora_alpha}")
    print(f"🎯 最大步數: {args.max_steps}")
    print("=" * 50)
    
    # 檢查依賴項
    if not check_requirements():
        return
    
    # 設置數據目錄
    if not setup_data_directory(args.data_dir):
        return
    
    # 開始訓練
    run_training(args)

if __name__ == "__main__":
    main()

"""
使用示例:

1. 快速訓練 (適合測試):
python train_quick_start.py --config fast

2. 平衡訓練 (推薦):
python train_quick_start.py --config balanced

3. 高質量訓練 (需要更多時間和資源):
python train_quick_start.py --config quality

4. 自定義配置:
python train_quick_start.py --config balanced --max_steps 2000 --resolution 768

5. 指定數據目錄:
python train_quick_start.py --data_dir ./my_megaman_images --output_dir ./my_lora_output
""" 