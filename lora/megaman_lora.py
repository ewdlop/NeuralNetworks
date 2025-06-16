#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mega Man LoRA 訓練腳本
使用 Stable Diffusion 和 LoRA 技術生成 Mega Man 風格的圖像

需要的依賴:
pip install diffusers transformers accelerate datasets Pillow torch torchvision
pip install peft xformers bitsandbytes
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import random
import numpy as np
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import math
from datetime import datetime

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

class MegaManDataset(Dataset):
    """
    Mega Man 數據集類
    """
    def __init__(self, data_dir, tokenizer, size=512, center_crop=True):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        
        # 查找所有圖像文件
        self.image_paths = []
        self.captions = []
        
        # 支持的圖像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        for img_path in self.data_dir.rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                self.image_paths.append(img_path)
                
                # 查找對應的描述文件
                caption_path = img_path.with_suffix('.txt')
                if caption_path.exists():
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                else:
                    # 如果沒有描述文件，使用默認描述
                    caption = self._generate_default_caption(img_path.name)
                
                self.captions.append(caption)
        
        logger.info(f"找到 {len(self.image_paths)} 張 Mega Man 圖像")
        
        # 圖像預處理
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def _generate_default_caption(self, filename):
        """生成默認的圖像描述"""
        default_captions = [
            "mega man, blue robot, helmet, megaman X, capcom game character",
            "megaman, robot master, blue armor, sci-fi character",
            "mega man X, futuristic robot, blue and white armor",
            "classic megaman, 8-bit style, robot hero",
            "megaman boss, robot design, game art style",
            "mega man zero, red and white robot, sword wielding",
            "megaman battle network, digital world, cyber style"
        ]
        return random.choice(default_captions)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        # 載入圖像
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # 處理文本描述
        caption = self.captions[index]
        
        # Tokenize 文本
        text_inputs = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids.squeeze(0),
            "attention_mask": text_inputs.attention_mask.squeeze(0),
            "caption": caption
        }

def collate_fn(examples):
    """數據整理函數"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    input_ids = torch.stack([example["input_ids"] for example in examples])
    attention_mask = torch.stack([example["attention_mask"] for example in examples])
    
    batch = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    
    return batch

def setup_lora_model(unet, rank=4, alpha=32):
    """設置 LoRA 模型"""
    
    # 定義要應用 LoRA 的模組
    target_modules = [
        "to_k", "to_q", "to_v", "to_out.0",
        "ff.net.0.proj", "ff.net.2"
    ]
    
    # LoRA 配置
    lora_config = LoraConfig(
        r=rank,  # rank
        lora_alpha=alpha,  # alpha
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.DIFFUSION,
    )
    
    # 應用 LoRA
    unet = get_peft_model(unet, lora_config)
    
    logger.info(f"LoRA 設置完成 - Rank: {rank}, Alpha: {alpha}")
    logger.info(f"可訓練參數數量: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")
    
    return unet

def create_sample_images_dir():
    """創建示例圖像目錄並提供說明"""
    sample_dir = Path("megaman_images")
    sample_dir.mkdir(exist_ok=True)
    
    # 創建說明文件
    readme_content = """
# Mega Man LoRA 訓練數據目錄

請將您的 Mega Man 圖像放在此目錄中。

## 文件格式要求：
- 圖像格式：.jpg, .jpeg, .png, .bmp, .webp
- 建議分辨率：512x512 或更高
- 每張圖像可以有對應的 .txt 描述文件

## 示例文件結構：
```
megaman_images/
├── megaman_01.png
├── megaman_01.txt  (內容: "mega man, blue robot, helmet, game character")
├── megaman_x_02.jpg
├── megaman_x_02.txt  (內容: "mega man X, futuristic armor, blue and white")
└── ...
```

## 描述文件說明：
- 每個 .txt 文件應包含對應圖像的描述
- 使用英文，用逗號分隔關鍵詞
- 包含 "mega man", "megaman", "robot" 等關鍵詞
- 描述顏色、風格、動作等特徵

## 建議的描述詞：
- mega man, megaman, mega man X, mega man zero
- blue robot, red robot, green robot
- helmet, armor, cannon, sword
- sci-fi, futuristic, 8-bit style
- game character, robot master, maverick
- capcom, classic gaming
"""
    
    with open(sample_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    return sample_dir

def train_lora(args):
    """主要的 LoRA 訓練函數"""
    
    # 初始化 accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard" if args.logging_dir else None,
        project_dir=args.logging_dir,
    )
    
    # 設置隨機種子
    if args.seed is not None:
        set_seed(args.seed)
    
    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入預訓練模型
    logger.info("載入 Stable Diffusion 模型...")
    
    # Tokenizer 和 Text Encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name, subfolder="text_encoder"
    )
    
    # VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name, subfolder="vae"
    )
    
    # UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name, subfolder="unet"
    )
    
    # 設置 LoRA
    unet = setup_lora_model(unet, rank=args.lora_rank, alpha=args.lora_alpha)
    
    # 凍結不需要訓練的模型
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # 只訓練 LoRA 參數
    for name, param in unet.named_parameters():
        if "lora" not in name:
            param.requires_grad_(False)
    
    # 噪聲調度器
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name, subfolder="scheduler"
    )
    
    # 創建數據集
    logger.info("創建數據集...")
    dataset = MegaManDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop
    )
    
    # 數據加載器
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    
    # 優化器
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # 學習率調度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    # 使用 accelerator 準備模型、優化器等
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    
    # 移動 vae 和 text_encoder 到設備
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=torch.float32)
    
    # 計算訓練步驟
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    logger.info("***** 開始訓練 *****")
    logger.info(f"  數據集大小 = {len(dataset)}")
    logger.info(f"  訓練 epochs = {args.num_train_epochs}")
    logger.info(f"  批次大小 = {args.train_batch_size}")
    logger.info(f"  梯度累積步數 = {args.gradient_accumulation_steps}")
    logger.info(f"  總訓練步數 = {args.max_train_steps}")
    
    # 訓練循環
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("訓練")
    
    for epoch in range(args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                # 編碼圖像到潛在空間
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # 添加噪聲
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 編碼文本
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # 預測噪聲
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # 計算損失
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"未知的預測類型 {noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # 反向傳播
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 記錄和保存
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                train_loss += loss.detach().item()
                
                if global_step % args.save_steps == 0:
                    save_path = output_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(save_path)
                    logger.info(f"保存檢查點到 {save_path}")
                
                if global_step % args.validation_steps == 0:
                    logger.info(f"Step {global_step}: 平均損失 = {train_loss / args.validation_steps:.4f}")
                    
                    # 生成示例圖像
                    if args.validation_prompt:
                        generate_sample_images(
                            unet, vae, text_encoder, tokenizer, noise_scheduler,
                            args.validation_prompt, accelerator.device, 
                            output_dir / f"samples-{global_step}"
                        )
                    
                    train_loss = 0.0
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= args.max_train_steps:
                break
    
    # 保存最終模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_final_model(unet, output_dir, args)
    
    accelerator.end_training()

def generate_sample_images(unet, vae, text_encoder, tokenizer, scheduler, prompt, device, save_dir):
    """生成示例圖像"""
    logger.info(f"生成示例圖像: {prompt}")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 創建管道
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    # 生成圖像
    with torch.no_grad():
        images = pipeline(
            prompt,
            num_images_per_prompt=4,
            num_inference_steps=25,
            guidance_scale=7.5,
            height=512,
            width=512,
        ).images
    
    # 保存圖像
    for i, image in enumerate(images):
        image.save(save_dir / f"sample_{i}.png")
    
    del pipeline
    torch.cuda.empty_cache()

def save_final_model(unet, output_dir, args):
    """保存最終的 LoRA 模型"""
    logger.info("保存最終 LoRA 模型...")
    
    # 保存 LoRA 權重
    unet.save_pretrained(output_dir / "lora_weights")
    
    # 保存訓練配置
    config = {
        "base_model": args.pretrained_model_name,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "resolution": args.resolution,
        "learning_rate": args.learning_rate,
        "max_train_steps": args.max_train_steps,
        "train_batch_size": args.train_batch_size,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"模型已保存到 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="訓練 Mega Man LoRA")
    
    # 基本參數
    parser.add_argument("--pretrained_model_name", type=str, default="runwayml/stable-diffusion-v1-5",
                       help="預訓練模型名稱")
    parser.add_argument("--data_dir", type=str, default="./megaman_images",
                       help="訓練數據目錄")
    parser.add_argument("--output_dir", type=str, default="./megaman_lora_output",
                       help="輸出目錄")
    parser.add_argument("--logging_dir", type=str, default="./logs",
                       help="日誌目錄")
    
    # 訓練參數
    parser.add_argument("--resolution", type=int, default=512,
                       help="圖像分辨率")
    parser.add_argument("--center_crop", action="store_true",
                       help="是否中心裁剪")
    parser.add_argument("--train_batch_size", type=int, default=1,
                       help="訓練批次大小")
    parser.add_argument("--num_train_epochs", type=int, default=100,
                       help="訓練輪數")
    parser.add_argument("--max_train_steps", type=int, default=None,
                       help="最大訓練步數")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="梯度累積步數")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="學習率")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                       help="學習率調度器")
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                       help="學習率預熱步數")
    
    # LoRA 參數
    parser.add_argument("--lora_rank", type=int, default=4,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    
    # 優化器參數
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                       help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                       help="Adam beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2,
                       help="Adam weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                       help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="最大梯度範數")
    
    # 其他參數
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                       help="數據加載器線程數")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="保存檢查點的步數間隔")
    parser.add_argument("--validation_steps", type=int, default=100,
                       help="驗證的步數間隔")
    parser.add_argument("--validation_prompt", type=str, 
                       default="mega man, blue robot, helmet, game character, high quality",
                       help="驗證用的提示詞")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                       choices=["no", "fp16", "bf16"], help="混合精度")
    parser.add_argument("--seed", type=int, default=None,
                       help="隨機種子")
    
    args = parser.parse_args()
    
    # 檢查數據目錄
    if not Path(args.data_dir).exists():
        logger.info(f"數據目錄 {args.data_dir} 不存在，正在創建示例目錄...")
        sample_dir = create_sample_images_dir()
        logger.info(f"請將 Mega Man 圖像放入 {sample_dir} 目錄中，然後重新運行訓練")
        return
    
    # 檢查是否有圖像文件
    data_path = Path(args.data_dir)
    image_files = list(data_path.rglob('*.png')) + list(data_path.rglob('*.jpg')) + list(data_path.rglob('*.jpeg'))
    
    if len(image_files) == 0:
        logger.error(f"在 {args.data_dir} 中沒有找到圖像文件！")
        logger.info("請確保數據目錄中包含 .png, .jpg 或 .jpeg 格式的圖像")
        return
    
    logger.info(f"找到 {len(image_files)} 個圖像文件，開始訓練...")
    
    # 開始訓練
    train_lora(args)

if __name__ == "__main__":
    main()

"""
使用示例:

1. 基本訓練:
python megaman_lora.py --data_dir ./megaman_images --output_dir ./megaman_lora_output

2. 自定義參數訓練:
python megaman_lora.py \
    --data_dir ./megaman_images \
    --output_dir ./megaman_lora_output \
    --resolution 512 \
    --train_batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --max_train_steps 1000

3. 高質量訓練 (需要更多 GPU 記憶體):
python megaman_lora.py \
    --data_dir ./megaman_images \
    --output_dir ./megaman_lora_output \
    --resolution 768 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --lora_rank 16 \
    --lora_alpha 64 \
    --max_train_steps 2000 \
    --mixed_precision fp16

訓練完成後，您可以使用生成的 LoRA 模型來生成 Mega Man 風格的圖像!
"""
