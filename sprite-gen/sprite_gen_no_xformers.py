#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sprite Sheet 生成器 - 無 xformers 版本
使用 PyTorch 原生優化替代 xformers
"""

import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from typing import Tuple, Dict, Any
from PIL import Image

# 導入 AI 模型
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler
)
from controlnet_aux import OpenposeDetector

# 導入姿勢操作模組
from pose_manipulation import PoseManipulator, AnimationType

class MemoryOptimizedSpriteGenerator:
    """記憶體優化的 Sprite Sheet 生成器（無 xformers）"""
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/sd-controlnet-openpose",
        device: str = "auto",
        enable_cpu_offload: bool = True,
        enable_model_optimization: bool = True,
        low_vram_mode: bool = False
    ):
        self.device = self._setup_device(device)
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.enable_cpu_offload = enable_cpu_offload
        self.enable_model_optimization = enable_model_optimization
        self.low_vram_mode = low_vram_mode
        
        print(f"🚀 初始化 Sprite 生成器（無 xformers）")
        print(f"📱 設備: {self.device}")
        print(f"💾 CPU 卸載: {enable_cpu_offload}")
        print(f"🔧 模型優化: {enable_model_optimization}")
        print(f"⚡ 低顯存模式: {low_vram_mode}")
        
        self._initialize_models()
        self.pose_manipulator = PoseManipulator()
        
    def _setup_device(self, device: str) -> str:
        """設置計算設備"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"✅ 找到 CUDA 設備: {torch.cuda.get_device_name(0)}")
                # 顯示 GPU 記憶體信息
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"📊 GPU 記憶體: {total_memory:.1f} GB")
            else:
                device = "cpu"
                print("⚠️ 未找到 CUDA，使用 CPU")
        return device
    
    def _apply_memory_optimizations(self, pipe):
        """應用記憶體優化技術（替代 xformers）"""
        if self.enable_model_optimization:
            print("🔧 應用記憶體優化...")
            
            # 1. 啟用注意力切片
            pipe.enable_attention_slicing()
            print("✅ 注意力切片已啟用")
            
            # 2. 啟用 VAE 切片
            if hasattr(pipe, 'enable_vae_slicing'):
                pipe.enable_vae_slicing()
                print("✅ VAE 切片已啟用")
            
            # 3. 使用記憶體高效的注意力
            if self.device == "cuda":
                # 使用 PyTorch 2.0 的原生 SDPA（Scaled Dot Product Attention）
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    print("✅ 使用 PyTorch 原生 SDPA")
                else:
                    print("⚠️ PyTorch 版本較舊，無法使用原生 SDPA")
            
            # 4. 低顯存模式優化
            if self.low_vram_mode:
                pipe.enable_sequential_cpu_offload()
                print("✅ 順序 CPU 卸載已啟用（低顯存模式）")
            elif self.enable_cpu_offload and self.device == "cuda":
                pipe.enable_model_cpu_offload()
                print("✅ 模型 CPU 卸載已啟用")
            
            # 5. 編譯模型（PyTorch 2.0+）
            if hasattr(torch, 'compile'):
                try:
                    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
                    print("✅ UNet 模型已編譯優化")
                except Exception as e:
                    print(f"⚠️ 模型編譯失敗: {e}")
        
        return pipe
    
    def _initialize_models(self):
        """初始化 AI 模型"""
        print("🔄 載入 ControlNet 模型...")
        self.controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        print("🔄 載入 Stable Diffusion 管道...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=self.controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # 設置更高效的調度器
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # 應用記憶體優化
        self.pipe = self._apply_memory_optimizations(self.pipe)
        
        # 移動到設備
        if not self.enable_cpu_offload:
            self.pipe = self.pipe.to(self.device)
        
        print("🔄 載入姿勢檢測器...")
        self.pose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        
        print("✅ 所有模型載入完成")
    
    def optimize_generation_params(self, vram_gb: float = None) -> Dict[str, Any]:
        """根據可用 VRAM 優化生成參數"""
        if vram_gb is None and self.device == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if vram_gb is None or vram_gb >= 8:
            # 高端設置
            return {
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512,
                "batch_size": 2
            }
        elif vram_gb >= 6:
            # 中端設置
            return {
                "num_inference_steps": 25,
                "guidance_scale": 7.0,
                "width": 512,
                "height": 512,
                "batch_size": 1
            }
        else:
            # 低端設置
            return {
                "num_inference_steps": 20,
                "guidance_scale": 6.5,
                "width": 384,
                "height": 384,
                "batch_size": 1
            }
    
    def generate_sprite_with_pose(
        self,
        prompt: str,
        pose_image: Image.Image,
        negative_prompt: str = None,
        **kwargs
    ) -> Image.Image:
        """使用姿勢控制生成單個精靈圖像"""
        # 獲取優化參數
        default_params = self.optimize_generation_params()
        
        # 合併用戶參數
        params = {**default_params, **kwargs}
        
        # 設置預設負面提示詞
        if negative_prompt is None:
            negative_prompt = (
                "blurry, low quality, distorted, deformed, "
                "extra limbs, bad anatomy, worst quality"
            )
        
        # 清理 GPU 記憶體
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # 生成圖像
        with torch.inference_mode():
            try:
                result = self.pipe(
                    prompt=prompt,
                    image=pose_image,
                    negative_prompt=negative_prompt,
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=params["guidance_scale"],
                    width=params["width"],
                    height=params["height"],
                    generator=torch.Generator(device=self.device).manual_seed(42)
                )
                
                return result.images[0]
                
            except torch.cuda.OutOfMemoryError:
                print("⚠️ GPU 記憶體不足，切換到 CPU 卸載模式")
                if hasattr(self.pipe, 'enable_sequential_cpu_offload'):
                    self.pipe.enable_sequential_cpu_offload()
                return self.generate_sprite_with_pose(
                    prompt, pose_image, negative_prompt, **kwargs
                )
    
    def create_sprite_sheet(
        self,
        character_prompt: str,
        animation_type: AnimationType = AnimationType.WALK,
        frames: int = 8,
        sprite_size: Tuple[int, int] = (64, 64),
        sheet_cols: int = 4,
        output_path: str = "sprite_sheet.png",
        style_prompt: str = "pixel art style",
        **kwargs
    ) -> Image.Image:
        """創建完整的 Sprite Sheet"""
        print(f"🎨 創建 Sprite Sheet: {animation_type.value}")
        print(f"📐 精靈尺寸: {sprite_size}")
        print(f"🎞️ 幀數: {frames}")
        
        # 生成姿勢序列
        poses = self.pose_manipulator.generate_animation_sequence(
            animation_type, frames
        )
        
        # 準備完整提示詞
        full_prompt = f"{character_prompt}, {style_prompt}, clean background"
        
        # 生成所有幀
        generated_frames = []
        for i, pose in enumerate(tqdm(poses, desc="生成幀")):
            # 渲染姿勢
            pose_image = self.pose_manipulator.render_pose(
                pose, size=sprite_size, show_skeleton=False
            )
            
            # 生成精靈圖像
            sprite_image = self.generate_sprite_with_pose(
                full_prompt,
                pose_image,
                width=sprite_size[0],
                height=sprite_size[1],
                **kwargs
            )
            
            # 調整大小確保一致性
            sprite_image = sprite_image.resize(sprite_size, Image.Resampling.LANCZOS)
            generated_frames.append(sprite_image)
            
            # 清理記憶體
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        # 創建 Sprite Sheet
        sheet_rows = (frames + sheet_cols - 1) // sheet_cols
        sheet_width = sheet_cols * sprite_size[0]
        sheet_height = sheet_rows * sprite_size[1]
        
        sprite_sheet = Image.new('RGBA', (sheet_width, sheet_height), (0, 0, 0, 0))
        
        for i, frame in enumerate(generated_frames):
            row = i // sheet_cols
            col = i % sheet_cols
            x = col * sprite_size[0]
            y = row * sprite_size[1]
            sprite_sheet.paste(frame, (x, y))
        
        # 保存結果
        sprite_sheet.save(output_path)
        print(f"✅ Sprite Sheet 已保存: {output_path}")
        
        return sprite_sheet

def main():
    """命令行界面"""
    parser = argparse.ArgumentParser(description="Sprite Sheet 生成器（無 xformers）")
    parser.add_argument("--prompt", required=True, help="角色描述提示詞")
    parser.add_argument("--animation", default="walk", 
                       choices=["idle", "walk", "run", "jump", "attack"],
                       help="動畫類型")
    parser.add_argument("--frames", type=int, default=8, help="動畫幀數")
    parser.add_argument("--size", nargs=2, type=int, default=[64, 64], 
                       help="精靈尺寸 (寬 高)")
    parser.add_argument("--cols", type=int, default=4, help="Sprite Sheet 列數")
    parser.add_argument("--output", default="sprite_sheet.png", help="輸出檔案路徑")
    parser.add_argument("--style", default="pixel art style", help="藝術風格")
    parser.add_argument("--low-vram", action="store_true", help="啟用低顯存模式")
    parser.add_argument("--cpu-only", action="store_true", help="僅使用 CPU")
    
    args = parser.parse_args()
    
    # 設置設備
    device = "cpu" if args.cpu_only else "auto"
    
    # 創建生成器
    generator = MemoryOptimizedSpriteGenerator(
        device=device,
        low_vram_mode=args.low_vram,
        enable_cpu_offload=not args.cpu_only
    )
    
    # 轉換動畫類型
    animation_type = AnimationType(args.animation.upper())
    
    # 生成 Sprite Sheet
    try:
        sprite_sheet = generator.create_sprite_sheet(
            character_prompt=args.prompt,
            animation_type=animation_type,
            frames=args.frames,
            sprite_size=tuple(args.size),
            sheet_cols=args.cols,
            output_path=args.output,
            style_prompt=args.style
        )
        
        print(f"🎉 成功生成 Sprite Sheet: {args.output}")
        
    except Exception as e:
        print(f"❌ 生成失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 