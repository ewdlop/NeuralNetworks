#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡單安裝腳本 - 無 xformers 版本
快速安裝所需依賴，避免 Windows 路徑問題
"""

import subprocess
import sys
import os

def run_pip_install(packages, description=""):
    """安裝 pip 包"""
    try:
        print(f"🔄 {description}...")
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失敗")
        print(f"錯誤: {e.stderr}")
        return False

def main():
    print("🚀 Sprite 生成系統安裝器（無 xformers）")
    print("=" * 50)
    
    # 升級 pip
    print("🔄 升級 pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=False)
    
    # 核心包列表（僅必需包）
    core_packages = [
        ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu118"],
        ["diffusers"],
        ["transformers"],
        ["accelerate"],
        ["controlnet-aux"],
        ["Pillow"],
        ["tqdm"],
        ["safetensors"],
        ["streamlit"]
    ]
    
    # 移除的無用包（已在註釋中說明原因）
    # - torchaudio: 僅音頻項目需要
    # - opencv-python: controlnet-aux 已包含
    # - numpy: torch 依賴會自動安裝
    # - matplotlib: 僅可視化需要
    # - requests: 基礎庫通常已安裝
    # - rich, psutil, bitsandbytes: 可選優化包
    
    print("\n📦 安裝核心依賴...")
    for packages in core_packages:
        package_name = packages[0] if len(packages) == 1 else packages[0]
        if not run_pip_install(packages, f"安裝 {package_name}"):
            print(f"⚠️ {package_name} 安裝失敗，但繼續安裝其他包")
    
    print("\n🎉 安裝完成！")
    print("\n📋 使用說明:")
    print("1. 基本使用: python sprite_gen_no_xformers.py --prompt '你的角色描述'")
    print("2. 低顯存模式: python sprite_gen_no_xformers.py --prompt '角色' --low-vram")
    print("3. 僅 CPU 模式: python sprite_gen_no_xformers.py --prompt '角色' --cpu-only")
    
    print("\n💡 提示:")
    print("- 此版本不使用 xformers，避免 Windows 路徑問題")
    print("- 使用 PyTorch 原生優化，性能接近 xformers")
    print("- 支持自動記憶體管理和設備優化")

if __name__ == "__main__":
    main() 