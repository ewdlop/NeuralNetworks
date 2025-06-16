#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mega Man LoRA 圖像生成腳本
使用訓練好的 LoRA 模型生成 Mega Man 風格的圖像
"""

import torch
import argparse
from pathlib import Path
from PIL import Image
import json
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_lora_model(base_model_path, lora_path, device="cuda"):
    """載入 LoRA 模型"""
    logger.info(f"載入基礎模型: {base_model_path}")
    
    # 載入基礎 Stable Diffusion 管道
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    
    # 使用更快的調度器
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # 載入 LoRA 權重
    if Path(lora_path).exists():
        logger.info(f"載入 LoRA 權重: {lora_path}")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        
        # 載入訓練配置
        config_path = Path(lora_path).parent / "training_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"LoRA 配置: Rank={config.get('lora_rank', 'Unknown')}, "
                           f"Alpha={config.get('lora_alpha', 'Unknown')}")
    else:
        logger.warning(f"LoRA 路徑不存在: {lora_path}")
    
    # 移動到設備
    pipe = pipe.to(device)
    
    # 啟用記憶體效率優化
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_vae_slicing()
    
    return pipe

def generate_images(pipe, prompts, output_dir, args):
    """生成圖像"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"開始生成 {len(prompts)} 個提示詞的圖像...")
    
    all_images = []
    
    for i, prompt in enumerate(prompts):
        logger.info(f"生成提示詞 {i+1}/{len(prompts)}: {prompt}")
        
        # 生成圖像
        with torch.no_grad():
            images = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                num_images_per_prompt=args.num_images_per_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=torch.Generator(device=pipe.device).manual_seed(args.seed + i) if args.seed else None,
            ).images
        
        # 保存圖像
        for j, image in enumerate(images):
            filename = f"megaman_{i:03d}_{j:02d}.png"
            image_path = output_path / filename
            image.save(image_path)
            all_images.append((prompt, image_path))
            logger.info(f"保存圖像: {image_path}")
    
    # 創建圖像索引
    create_image_index(all_images, output_path)
    
    logger.info(f"所有圖像已保存到: {output_path}")

def create_image_index(images, output_dir):
    """創建圖像索引 HTML 文件"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Mega Man LoRA 生成圖像</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .image-item { border: 1px solid #ddd; padding: 10px; border-radius: 8px; }
        .image-item img { max-width: 100%; height: auto; border-radius: 4px; }
        .prompt { margin-top: 10px; font-size: 14px; color: #666; }
        h1 { color: #333; text-align: center; }
    </style>
</head>
<body>
    <h1>🤖 Mega Man LoRA 生成圖像 🎮</h1>
    <div class="gallery">
"""
    
    for prompt, image_path in images:
        filename = image_path.name
        html_content += f"""
        <div class="image-item">
            <img src="{filename}" alt="Generated Mega Man">
            <div class="prompt"><strong>提示詞:</strong> {prompt}</div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def get_default_prompts():
    """獲取默認的 Mega Man 提示詞"""
    return [
        "mega man, blue robot, helmet, game character, high quality, detailed",
        "mega man X, futuristic armor, blue and white, sci-fi style, 4k",
        "mega man zero, red and white robot, sword, action pose, dynamic",
        "classic mega man, 8-bit style, retro gaming, pixelated",
        "mega man battle network, digital world, cyber style, neon lights",
        "mega man boss robot, unique design, colorful armor, threatening pose",
        "mega man charging mega buster, energy beam, blue glow, powerful",
        "mega man running, side view, classic platformer style, detailed",
        "mega man helmet close-up, detailed face, blue armor, high resolution",
        "mega man vs robot master, epic battle, explosions, dynamic scene"
    ]

def main():
    parser = argparse.ArgumentParser(description="使用 Mega Man LoRA 生成圖像")
    
    # 模型參數
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5",
                       help="基礎 Stable Diffusion 模型")
    parser.add_argument("--lora_path", type=str, default="./megaman_lora_output/lora_weights",
                       help="LoRA 權重路徑")
    parser.add_argument("--output_dir", type=str, default="./generated_megaman",
                       help="生成圖像輸出目錄")
    
    # 生成參數
    parser.add_argument("--prompts", type=str, nargs="*", default=None,
                       help="自定義提示詞列表")
    parser.add_argument("--prompts_file", type=str, default=None,
                       help="包含提示詞的文本文件 (每行一個)")
    parser.add_argument("--negative_prompt", type=str, 
                       default="blurry, low quality, distorted, deformed, ugly, bad anatomy",
                       help="負面提示詞")
    parser.add_argument("--num_images_per_prompt", type=int, default=1,
                       help="每個提示詞生成的圖像數量")
    parser.add_argument("--num_inference_steps", type=int, default=25,
                       help="推理步數")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="引導比例")
    parser.add_argument("--height", type=int, default=512,
                       help="圖像高度")
    parser.add_argument("--width", type=int, default=512,
                       help="圖像寬度")
    parser.add_argument("--seed", type=int, default=42,
                       help="隨機種子")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"], help="計算設備")
    
    args = parser.parse_args()
    
    # 檢查設備
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，使用 CPU")
        args.device = "cpu"
    
    # 準備提示詞
    if args.prompts_file:
        logger.info(f"從文件載入提示詞: {args.prompts_file}")
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.prompts:
        prompts = args.prompts
    else:
        logger.info("使用默認提示詞")
        prompts = get_default_prompts()
    
    logger.info(f"將生成 {len(prompts)} 個提示詞的圖像")
    
    # 載入模型
    pipe = load_lora_model(args.base_model, args.lora_path, args.device)
    
    # 生成圖像
    generate_images(pipe, prompts, args.output_dir, args)
    
    logger.info("圖像生成完成！")
    logger.info(f"查看結果: 打開 {args.output_dir}/index.html")

if __name__ == "__main__":
    main()

"""
使用示例:

1. 使用默認提示詞生成:
python generate_megaman.py --lora_path ./megaman_lora_output/lora_weights

2. 使用自定義提示詞:
python generate_megaman.py \
    --lora_path ./megaman_lora_output/lora_weights \
    --prompts "mega man X, detailed armor, action pose" "classic mega man, 8-bit style"

3. 從文件載入提示詞:
python generate_megaman.py \
    --lora_path ./megaman_lora_output/lora_weights \
    --prompts_file prompts.txt

4. 高質量生成:
python generate_megaman.py \
    --lora_path ./megaman_lora_output/lora_weights \
    --height 768 \
    --width 768 \
    --num_inference_steps 50 \
    --guidance_scale 10.0 \
    --num_images_per_prompt 4
""" 