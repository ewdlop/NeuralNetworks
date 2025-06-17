#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sprite Sheet 生成器 - 快速開始腳本
簡化的界面，適合初學者使用
"""

import os
import sys
from pathlib import Path
import argparse
from PIL import Image

def check_dependencies():
    """檢查必要的依賴項"""
    required_packages = [
        'torch', 'diffusers', 'transformers', 'controlnet_aux', 
        'PIL', 'numpy', 'cv2'
    ]
    
    missing = []
    for pkg in required_packages:
        try:
            if pkg == 'PIL':
                import PIL
            elif pkg == 'cv2':
                import cv2
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"❌ 缺少必要依賴項: {', '.join(missing)}")
        print("請運行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依賴項已安裝")
    return True

def get_user_preferences():
    """獲取用戶偏好設置"""
    print("\n🎮 Sprite Sheet 生成器配置")
    print("=" * 40)
    
    # 獲取輸入圖像
    while True:
        image_path = input("📁 請輸入角色圖像路徑: ").strip()
        if os.path.exists(image_path):
            break
        print(f"❌ 文件不存在: {image_path}")
    
    # 角色描述
    character_desc = input("📝 角色描述 (留空自動生成): ").strip()
    if not character_desc:
        character_desc = "game character, 2D sprite, detailed, high quality"
    
    # 精靈圖尺寸
    print("\n📐 選擇精靈圖尺寸:")
    print("1. 32x32 (小型，快速)")
    print("2. 64x64 (標準，推薦)")
    print("3. 128x128 (大型，高品質)")
    print("4. 自定義")
    
    size_choice = input("選擇 (1-4): ").strip()
    if size_choice == "1":
        sprite_size = (32, 32)
    elif size_choice == "3":
        sprite_size = (128, 128)
    elif size_choice == "4":
        try:
            w = int(input("寬度: "))
            h = int(input("高度: "))
            sprite_size = (w, h)
        except ValueError:
            sprite_size = (64, 64)
    else:
        sprite_size = (64, 64)
    
    # 動畫類型
    print("\n🎬 選擇動畫類型:")
    print("1. 基本動畫 (idle, walk)")
    print("2. 標準動畫 (idle, walk, jump, attack)")
    print("3. 完整動畫 (idle, walk, jump, attack, run)")
    print("4. 自定義")
    
    anim_choice = input("選擇 (1-4): ").strip()
    if anim_choice == "1":
        animations = ["idle", "walk"]
    elif anim_choice == "3":
        animations = ["idle", "walk", "jump", "attack", "run"]
    elif anim_choice == "4":
        custom_anims = input("輸入動畫類型 (用空格分隔): ").strip().split()
        animations = custom_anims if custom_anims else ["idle", "walk", "jump", "attack"]
    else:
        animations = ["idle", "walk", "jump", "attack"]
    
    # 品質設置
    print("\n⚙️ 選擇生成品質:")
    print("1. 快速 (適合測試)")
    print("2. 平衡 (推薦)")
    print("3. 高品質 (耗時較長)")
    
    quality_choice = input("選擇 (1-3): ").strip()
    if quality_choice == "1":
        inference_steps = 15
        guidance_scale = 5.0
    elif quality_choice == "3":
        inference_steps = 50
        guidance_scale = 10.0
    else:
        inference_steps = 20
        guidance_scale = 7.5
    
    # 輸出目錄
    output_dir = input("\n📁 輸出目錄 (留空使用預設): ").strip()
    if not output_dir:
        # 基於輸入文件名創建輸出目錄
        base_name = Path(image_path).stem
        output_dir = f"./sprite_output_{base_name}"
    
    return {
        'image_path': image_path,
        'character_desc': character_desc,
        'sprite_size': sprite_size,
        'animations': animations,
        'inference_steps': inference_steps,
        'guidance_scale': guidance_scale,
        'output_dir': output_dir
    }

def display_preview(preferences):
    """顯示配置預覽"""
    print("\n" + "="*50)
    print("🔍 配置預覽")
    print("="*50)
    print(f"📁 輸入圖像: {preferences['image_path']}")
    print(f"📝 角色描述: {preferences['character_desc']}")
    print(f"📐 精靈圖尺寸: {preferences['sprite_size'][0]}x{preferences['sprite_size'][1]}")
    print(f"🎬 動畫類型: {', '.join(preferences['animations'])}")
    print(f"⚙️ 推理步數: {preferences['inference_steps']}")
    print(f"🎯 引導比例: {preferences['guidance_scale']}")
    print(f"📁 輸出目錄: {preferences['output_dir']}")
    print("="*50)
    
    confirm = input("確認開始生成？(y/n): ").strip().lower()
    return confirm in ['y', 'yes', '是', '確認']

def run_generation(preferences):
    """運行 Sprite Sheet 生成"""
    print("\n🚀 開始生成 Sprite Sheet...")
    
    try:
        # 導入主要模組
        from sprite_gen import SpriteSheetGenerator, create_character_prompt
        
        # 載入輸入圖像
        print("📥 載入輸入圖像...")
        base_image = Image.open(preferences['image_path']).convert('RGB')
        
        # 創建角色描述
        character_prompt = create_character_prompt(base_image, preferences['character_desc'])
        print(f"🎨 角色描述: {character_prompt}")
        
        # 初始化生成器
        print("🤖 初始化 AI 模型 (首次運行需要下載模型)...")
        generator = SpriteSheetGenerator(device="cuda" if preferences.get('use_cuda', True) else "cpu")
        
        # 生成 Sprite Sheet
        print("🎨 生成 Sprite Sheet...")
        sprite_sheet = generator.create_sprite_sheet(
            base_image=base_image,
            character_prompt=character_prompt,
            animations=preferences['animations'],
            sprite_size=preferences['sprite_size'],
            padding=2
        )
        
        # 保存結果
        print("💾 保存結果...")
        output_path = Path(preferences['output_dir']) / "sprite_sheet.png"
        generator.save_sprite_sheet_with_metadata(
            sprite_sheet=sprite_sheet,
            output_path=str(output_path),
            sprite_size=preferences['sprite_size'],
            animations=preferences['animations']
        )
        
        print("\n" + "="*50)
        print("✅ Sprite Sheet 生成完成！")
        print("="*50)
        print(f"📁 輸出文件: {output_path}")
        print(f"📊 元數據: {output_path.with_suffix('_metadata.json')}")
        print(f"🖼️ 精靈圖數量: {len(preferences['animations'])} 個動畫")
        
        # 顯示如何使用
        print("\n🎮 如何使用生成的 Sprite Sheet:")
        print("1. 在遊戲引擎中導入 sprite_sheet.png")
        print("2. 使用 metadata.json 獲取動畫信息")
        print("3. 根據元數據設置動畫幀")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模組導入錯誤: {e}")
        print("請確保已安裝所有依賴項: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ 生成過程中發生錯誤: {e}")
        return False

def main():
    """主函數"""
    print("🎮 歡迎使用 Sprite Sheet 生成器！")
    print("這個工具可以將您的角色圖像轉換為動畫用的 Sprite Sheet")
    print()
    
    # 檢查依賴項
    if not check_dependencies():
        return
    
    # 獲取用戶偏好
    try:
        preferences = get_user_preferences()
    except KeyboardInterrupt:
        print("\n👋 已取消操作")
        return
    
    # 顯示預覽並確認
    if not display_preview(preferences):
        print("👋 已取消操作")
        return
    
    # 運行生成
    success = run_generation(preferences)
    
    if success:
        print("\n🎉 感謝使用 Sprite Sheet 生成器！")
        print("如有問題，請查看 README.md 或提交 Issue")
    else:
        print("\n😞 生成失敗，請檢查錯誤信息並重試")

if __name__ == "__main__":
    main()

"""
快速開始指南:

1. 確保已安裝依賴項:
   pip install -r requirements.txt

2. 運行快速開始腳本:
   python quick_start.py

3. 按照提示操作:
   - 選擇輸入圖像
   - 設置角色描述
   - 選擇尺寸和動畫類型
   - 確認並開始生成

4. 等待生成完成，檢查輸出文件

注意: 首次運行會下載 AI 模型，需要穩定的網絡連接
""" 