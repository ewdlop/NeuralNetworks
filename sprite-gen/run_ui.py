#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI 啟動腳本
"""

import subprocess
import sys
import os
from pathlib import Path

def check_streamlit():
    """檢查 Streamlit 是否已安裝"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """安裝 Streamlit"""
    print("🔄 正在安裝 Streamlit...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit>=1.25.0"], check=True)
        print("✅ Streamlit 安裝成功")
        return True
    except subprocess.CalledProcessError:
        print("❌ Streamlit 安裝失敗")
        return False

def main():
    print("🚀 Sprite Sheet 生成器 - Web UI 啟動器")
    print("=" * 50)
    
    # 檢查 Streamlit
    if not check_streamlit():
        print("⚠️ 未找到 Streamlit，正在安裝...")
        if not install_streamlit():
            print("❌ 無法安裝 Streamlit，請手動安裝：pip install streamlit")
            return
    
    # 獲取腳本路徑
    script_dir = Path(__file__).parent
    streamlit_app = script_dir / "streamlit_app.py"
    
    if not streamlit_app.exists():
        print(f"❌ 找不到 Streamlit 應用文件: {streamlit_app}")
        return
    
    print("✅ 啟動 Streamlit Web UI...")
    print("🌐 瀏覽器將自動打開 http://localhost:8501")
    print("💡 按 Ctrl+C 停止服務器")
    print("-" * 50)
    
    # 啟動 Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_app),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit 服務器已停止")
    except Exception as e:
        print(f"❌ 啟動失敗: {e}")

if __name__ == "__main__":
    main() 