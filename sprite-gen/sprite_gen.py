#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sprite Sheet 生成器
將單張角色圖片轉換為動畫用的 Sprite Sheet
結合 ControlNet + Stable Diffusion + Pose 模型

功能:
- 從單張角色圖片生成多個動作姿勢
- 自動生成標準動畫序列 (行走、跳躍、攻擊等)
- 支持自定義姿勢和動作
- 輸出標準格式的 Sprite Sheet
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import argparse
from pathlib import Path
import json
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math

# Diffusion 相關
from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    UniPCMultistepScheduler,
    StableDiffusionPipeline
)
from diffusers.utils import load_image
from transformers import pipeline

# ControlNet 預處理器
from controlnet_aux import OpenposeDetector

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PoseConfig:
    """姿勢配置"""
    name: str
    description: str
    keypoints: List[List[float]]  # OpenPose 關鍵點
    angle: float = 0.0  # 旋轉角度
    scale: float = 1.0  # 縮放比例

@dataclass
class AnimationSequence:
    """動畫序列配置"""
    name: str
    poses: List[PoseConfig]
    fps: int = 8
    loop: bool = True

class PoseGenerator:
    """姿勢生成器"""
    
    def __init__(self):
        self.openpose_detector = OpenposeDetector.from_pretrained('lllyasviel/Annotators')
    
    def extract_pose_from_image(self, image: Image.Image) -> np.ndarray:
        """從圖像中提取姿勢"""
        pose_image = self.openpose_detector(image)
        return np.array(pose_image)
    
    def create_walking_poses(self, base_pose: np.ndarray, steps: int = 8) -> List[np.ndarray]:
        """創建行走動畫姿勢"""
        poses = []
        
        # 基於基礎姿勢生成行走序列
        for i in range(steps):
            # 計算行走週期
            cycle = (i / steps) * 2 * math.pi
            
            # 複製基礎姿勢
            pose = base_pose.copy()
            
            # 修改腿部關鍵點來模擬行走
            # 這裡需要根據 OpenPose 的關鍵點索引進行調整
            if len(pose.shape) == 3:  # RGB 圖像
                # 簡單的變換 - 實際應用中需要更複雜的姿勢插值
                pose = self._modify_pose_for_walking(pose, cycle)
            
            poses.append(pose)
        
        return poses
    
    def create_jumping_poses(self, base_pose: np.ndarray, steps: int = 6) -> List[np.ndarray]:
        """創建跳躍動畫姿勢"""
        poses = []
        
        # 跳躍序列: 準備 -> 起跳 -> 空中 -> 落地
        for i in range(steps):
            pose = base_pose.copy()
            
            # 根據跳躍階段修改姿勢
            if i < 2:  # 準備階段
                pose = self._modify_pose_for_crouch(pose, i / 2)
            elif i < 4:  # 空中階段
                pose = self._modify_pose_for_jump(pose, (i - 2) / 2)
            else:  # 落地階段
                pose = self._modify_pose_for_landing(pose, (i - 4) / 2)
            
            poses.append(pose)
        
        return poses
    
    def create_attack_poses(self, base_pose: np.ndarray, steps: int = 4) -> List[np.ndarray]:
        """創建攻擊動畫姿勢"""
        poses = []
        
        for i in range(steps):
            pose = base_pose.copy()
            
            # 攻擊動作: 準備 -> 出擊 -> 收回
            if i < 2:  # 準備和出擊
                pose = self._modify_pose_for_attack(pose, i / 2)
            else:  # 收回
                pose = self._modify_pose_for_return(pose, (i - 2) / 2)
            
            poses.append(pose)
        
        return poses
    
    def _modify_pose_for_walking(self, pose: np.ndarray, cycle: float) -> np.ndarray:
        """修改姿勢用於行走動畫"""
        # 這裡應該實現具體的姿勢修改邏輯
        # 簡化版本 - 實際需要根據 OpenPose 關鍵點進行精確修改
        return pose
    
    def _modify_pose_for_crouch(self, pose: np.ndarray, progress: float) -> np.ndarray:
        """修改姿勢用於蹲下"""
        return pose
    
    def _modify_pose_for_jump(self, pose: np.ndarray, progress: float) -> np.ndarray:
        """修改姿勢用於跳躍"""
        return pose
    
    def _modify_pose_for_landing(self, pose: np.ndarray, progress: float) -> np.ndarray:
        """修改姿勢用於落地"""
        return pose
    
    def _modify_pose_for_attack(self, pose: np.ndarray, progress: float) -> np.ndarray:
        """修改姿勢用於攻擊"""
        return pose
    
    def _modify_pose_for_return(self, pose: np.ndarray, progress: float) -> np.ndarray:
        """修改姿勢用於收回"""
        return pose

class SpriteSheetGenerator:
    """Sprite Sheet 生成器"""
    
    def __init__(self, 
                 model_id: str = "runwayml/stable-diffusion-v1-5",
                 controlnet_id: str = "lllyasviel/sd-controlnet-openpose",
                 device: str = "cuda"):
        
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"使用設備: {self.device}")
        
        # 載入 ControlNet
        logger.info("載入 ControlNet 模型...")
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # 載入 Stable Diffusion + ControlNet 管道
        logger.info("載入 Stable Diffusion 管道...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # 優化設置
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)
        
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_xformers_memory_efficient_attention()
        
        # 姿勢生成器
        self.pose_generator = PoseGenerator()
        
        logger.info("Sprite Sheet 生成器初始化完成")
    
    def extract_character_info(self, image: Image.Image) -> Dict:
        """從輸入圖像中提取角色信息"""
        # 使用 CLIP 或其他模型分析圖像內容
        # 簡化版本 - 返回基本信息
        
        width, height = image.size
        
        return {
            "width": width,
            "height": height,
            "aspect_ratio": width / height,
            "dominant_colors": self._extract_dominant_colors(image),
            "estimated_style": self._estimate_art_style(image)
        }
    
    def _extract_dominant_colors(self, image: Image.Image) -> List[Tuple[int, int, int]]:
        """提取主要顏色"""
        # 簡化版本 - 實際可以使用更複雜的顏色分析
        image_small = image.resize((100, 100))
        colors = image_small.getcolors(maxcolors=256*256*256)
        if colors:
            # 按出現頻率排序
            colors.sort(key=lambda x: x[0], reverse=True)
            return [color[1] for color in colors[:5]]
        return [(128, 128, 128)]  # 預設灰色
    
    def _estimate_art_style(self, image: Image.Image) -> str:
        """估計藝術風格"""
        # 這裡可以使用預訓練的分類模型
        # 簡化版本
        return "pixel art style, game character, 2D sprite"
    
    def generate_sprite_from_pose(self, 
                                  pose_image: Image.Image,
                                  prompt: str,
                                  negative_prompt: str = None,
                                  num_inference_steps: int = 20,
                                  guidance_scale: float = 7.5,
                                  controlnet_conditioning_scale: float = 1.0) -> Image.Image:
        """根據姿勢圖像生成精靈圖"""
        
        if negative_prompt is None:
            negative_prompt = "blurry, low quality, distorted, deformed, extra limbs, missing limbs, bad anatomy"
        
        # 生成圖像
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                image=pose_image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=torch.Generator(device=self.device).manual_seed(42)
            )
        
        return result.images[0]
    
    def create_animation_sequence(self, 
                                  base_image: Image.Image,
                                  animation_type: str,
                                  character_prompt: str,
                                  steps: int = 8) -> List[Image.Image]:
        """創建動畫序列"""
        
        logger.info(f"創建 {animation_type} 動畫序列...")
        
        # 提取基礎姿勢
        base_pose = self.pose_generator.extract_pose_from_image(base_image)
        
        # 根據動畫類型生成姿勢序列
        if animation_type == "walk":
            poses = self.pose_generator.create_walking_poses(base_pose, steps)
        elif animation_type == "jump":
            poses = self.pose_generator.create_jumping_poses(base_pose, steps)
        elif animation_type == "attack":
            poses = self.pose_generator.create_attack_poses(base_pose, steps)
        else:
            # 預設：靜態姿勢變化
            poses = [base_pose] * steps
        
        # 為每個姿勢生成圖像
        generated_images = []
        for i, pose in enumerate(poses):
            logger.info(f"生成第 {i+1}/{len(poses)} 幀...")
            
            # 將姿勢轉換為 PIL 圖像
            pose_pil = Image.fromarray(pose.astype(np.uint8))
            
            # 生成對應的角色圖像
            generated_image = self.generate_sprite_from_pose(
                pose_image=pose_pil,
                prompt=character_prompt,
                negative_prompt="blurry, low quality, distorted, bad anatomy, extra limbs"
            )
            
            generated_images.append(generated_image)
        
        return generated_images
    
    def create_sprite_sheet(self, 
                           base_image: Image.Image,
                           character_prompt: str,
                           animations: List[str] = None,
                           sprite_size: Tuple[int, int] = (64, 64),
                           padding: int = 2,
                           background_color: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> Image.Image:
        """創建完整的 Sprite Sheet"""
        
        if animations is None:
            animations = ["idle", "walk", "jump", "attack"]
        
        logger.info("開始創建 Sprite Sheet...")
        
        # 提取角色信息
        char_info = self.extract_character_info(base_image)
        
        # 構建角色描述
        full_prompt = f"{character_prompt}, {char_info['estimated_style']}, consistent character design"
        
        all_sprites = []
        animation_metadata = {}
        
        # 為每個動畫生成精靈圖
        for animation in animations:
            logger.info(f"生成 {animation} 動畫...")
            
            if animation == "idle":
                # 靜態姿勢
                base_pose = self.pose_generator.extract_pose_from_image(base_image)
                pose_pil = Image.fromarray(base_pose.astype(np.uint8))
                sprite = self.generate_sprite_from_pose(pose_pil, full_prompt)
                sprites = [sprite]
            else:
                # 動態動畫
                steps = 8 if animation == "walk" else 6 if animation == "jump" else 4
                sprites = self.create_animation_sequence(
                    base_image, animation, full_prompt, steps
                )
            
            # 調整精靈圖大小
            resized_sprites = []
            for sprite in sprites:
                # 移除背景並調整大小
                sprite_processed = self._process_sprite(sprite, sprite_size, background_color)
                resized_sprites.append(sprite_processed)
            
            all_sprites.extend(resized_sprites)
            animation_metadata[animation] = {
                "start_index": len(all_sprites) - len(resized_sprites),
                "frame_count": len(resized_sprites),
                "fps": 8 if animation == "walk" else 6 if animation == "jump" else 4
            }
        
        # 計算 Sprite Sheet 尺寸
        total_sprites = len(all_sprites)
        cols = min(8, total_sprites)  # 最多8列
        rows = math.ceil(total_sprites / cols)
        
        sheet_width = cols * (sprite_size[0] + padding) - padding
        sheet_height = rows * (sprite_size[1] + padding) - padding
        
        # 創建 Sprite Sheet
        sprite_sheet = Image.new('RGBA', (sheet_width, sheet_height), background_color)
        
        # 放置精靈圖
        for i, sprite in enumerate(all_sprites):
            col = i % cols
            row = i // cols
            x = col * (sprite_size[0] + padding)
            y = row * (sprite_size[1] + padding)
            sprite_sheet.paste(sprite, (x, y))
        
        # 添加網格線（可選）
        sprite_sheet = self._add_grid_lines(sprite_sheet, sprite_size, padding, cols, rows)
        
        # 保存元數據
        self.animation_metadata = animation_metadata
        
        logger.info("Sprite Sheet 創建完成")
        return sprite_sheet
    
    def _process_sprite(self, 
                       sprite: Image.Image, 
                       target_size: Tuple[int, int],
                       background_color: Tuple[int, int, int, int]) -> Image.Image:
        """處理單個精靈圖"""
        
        # 轉換為RGBA
        if sprite.mode != 'RGBA':
            sprite = sprite.convert('RGBA')
        
        # 移除背景（簡單版本 - 基於邊緣顏色）
        sprite = self._remove_background(sprite)
        
        # 調整大小保持比例
        sprite.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # 創建目標大小的畫布
        processed = Image.new('RGBA', target_size, background_color)
        
        # 居中放置
        x = (target_size[0] - sprite.width) // 2
        y = (target_size[1] - sprite.height) // 2
        processed.paste(sprite, (x, y), sprite)
        
        return processed
    
    def _remove_background(self, image: Image.Image) -> Image.Image:
        """簡單的背景移除"""
        data = np.array(image)
        
        # 假設背景是邊緣的主要顏色
        bg_color = data[0, 0]  # 左上角作為背景色
        
        # 創建遮罩
        mask = np.all(data[:, :, :3] != bg_color[:3], axis=2)
        
        # 應用遮罩
        data[:, :, 3] = mask * 255
        
        return Image.fromarray(data)
    
    def _add_grid_lines(self, 
                       sprite_sheet: Image.Image, 
                       sprite_size: Tuple[int, int], 
                       padding: int, 
                       cols: int, 
                       rows: int) -> Image.Image:
        """添加網格線"""
        
        draw = ImageDraw.Draw(sprite_sheet)
        
        # 垂直線
        for i in range(cols + 1):
            x = i * (sprite_size[0] + padding) - padding // 2
            if x >= 0 and x < sprite_sheet.width:
                draw.line([(x, 0), (x, sprite_sheet.height)], fill=(255, 255, 255, 128), width=1)
        
        # 水平線
        for i in range(rows + 1):
            y = i * (sprite_size[1] + padding) - padding // 2
            if y >= 0 and y < sprite_sheet.height:
                draw.line([(0, y), (sprite_sheet.width, y)], fill=(255, 255, 255, 128), width=1)
        
        return sprite_sheet
    
    def save_sprite_sheet_with_metadata(self, 
                                       sprite_sheet: Image.Image, 
                                       output_path: str,
                                       sprite_size: Tuple[int, int],
                                       animations: List[str]):
        """保存 Sprite Sheet 和元數據"""
        
        # 保存圖像
        sprite_sheet.save(output_path)
        logger.info(f"Sprite Sheet 已保存: {output_path}")
        
        # 保存元數據
        metadata = {
            "sprite_size": sprite_size,
            "animations": self.animation_metadata,
            "total_frames": sum(anim["frame_count"] for anim in self.animation_metadata.values()),
            "sheet_size": [sprite_sheet.width, sprite_sheet.height]
        }
        
        metadata_path = output_path.replace('.png', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"元數據已保存: {metadata_path}")

def create_character_prompt(base_image: Image.Image, custom_description: str = None) -> str:
    """根據輸入圖像創建角色描述"""
    
    if custom_description:
        return custom_description
    
    # 這裡可以使用 CLIP 或其他視覺語言模型來自動描述
    # 簡化版本
    return "game character, 2D sprite, pixel art style, consistent design, clean background"

def main():
    parser = argparse.ArgumentParser(description="Sprite Sheet 生成器")
    
    # 基本參數
    parser.add_argument("--input_image", type=str, required=True,
                       help="輸入角色圖像路徑")
    parser.add_argument("--output_dir", type=str, default="./sprite_output",
                       help="輸出目錄")
    parser.add_argument("--character_prompt", type=str, default=None,
                       help="角色描述 (留空自動生成)")
    
    # Sprite Sheet 設置
    parser.add_argument("--sprite_size", type=int, nargs=2, default=[64, 64],
                       metavar=('WIDTH', 'HEIGHT'), help="單個精靈圖尺寸")
    parser.add_argument("--animations", type=str, nargs="+", 
                       default=["idle", "walk", "jump", "attack"],
                       help="要生成的動畫類型")
    parser.add_argument("--padding", type=int, default=2,
                       help="精靈圖之間的間距")
    
    # 生成參數
    parser.add_argument("--num_inference_steps", type=int, default=20,
                       help="推理步數")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="引導比例")
    parser.add_argument("--controlnet_scale", type=float, default=1.0,
                       help="ControlNet 影響強度")
    
    # 模型設置
    parser.add_argument("--model_id", type=str, 
                       default="runwayml/stable-diffusion-v1-5",
                       help="Stable Diffusion 模型")
    parser.add_argument("--controlnet_id", type=str,
                       default="lllyasviel/sd-controlnet-openpose", 
                       help="ControlNet 模型")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"], help="計算設備")
    
    args = parser.parse_args()
    
    # 檢查輸入文件
    if not os.path.exists(args.input_image):
        logger.error(f"輸入圖像不存在: {args.input_image}")
        return
    
    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入輸入圖像
    logger.info(f"載入輸入圖像: {args.input_image}")
    base_image = Image.open(args.input_image).convert('RGB')
    
    # 創建角色描述
    character_prompt = create_character_prompt(base_image, args.character_prompt)
    logger.info(f"角色描述: {character_prompt}")
    
    # 初始化生成器
    generator = SpriteSheetGenerator(
        model_id=args.model_id,
        controlnet_id=args.controlnet_id,
        device=args.device
    )
    
    # 生成 Sprite Sheet
    sprite_sheet = generator.create_sprite_sheet(
        base_image=base_image,
        character_prompt=character_prompt,
        animations=args.animations,
        sprite_size=tuple(args.sprite_size),
        padding=args.padding
    )
    
    # 保存結果
    output_path = output_dir / "sprite_sheet.png"
    generator.save_sprite_sheet_with_metadata(
        sprite_sheet=sprite_sheet,
        output_path=str(output_path),
        sprite_size=tuple(args.sprite_size),
        animations=args.animations
    )
    
    logger.info("Sprite Sheet 生成完成！")
    logger.info(f"輸出文件: {output_path}")
    logger.info(f"元數據: {output_path.with_suffix('')}_metadata.json")

if __name__ == "__main__":
    main()

"""
使用示例:

1. 基本使用:
python sprite-gen.py --input_image character.png

2. 自定義設置:
python sprite-gen.py \
    --input_image character.png \
    --output_dir ./my_sprites \
    --character_prompt "pixel art warrior, blue armor, game character" \
    --sprite_size 128 128 \
    --animations idle walk jump attack

3. 高質量生成:
python sprite-gen.py \
    --input_image character.png \
    --sprite_size 256 256 \
    --num_inference_steps 50 \
    --guidance_scale 10.0

4. 簡單動畫:
python sprite-gen.py \
    --input_image character.png \
    --animations walk jump \
    --sprite_size 64 64
"""
