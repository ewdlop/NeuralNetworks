#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sprite Sheet 生成器 - Streamlit UI
提供用戶友好的圖形界面來生成 Sprite Sheet
"""

import streamlit as st
import torch
from PIL import Image
import io
import os
import sys
from typing import Dict, Any
import traceback
import time

# 添加當前目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 導入本地模組
try:
    from sprite_gen_no_xformers import MemoryOptimizedSpriteGenerator
    from pose_manipulation import AnimationType
except ImportError as e:
    st.error(f"導入錯誤: {e}")
    st.error("請確保所有必需的模組都已安裝")
    st.stop()

# 頁面配置
st.set_page_config(
    page_title="🎨 Sprite Sheet 生成器",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ecdc4;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 主標題
st.markdown('<h1 class="main-header">🎨 Sprite Sheet 生成器</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">使用 AI 技術生成遊戲角色動畫</p>', unsafe_allow_html=True)

# 初始化 session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'generation_in_progress' not in st.session_state:
    st.session_state.generation_in_progress = False

# 側邊欄 - 系統設置
st.sidebar.header("🔧 系統設置")

# 設備選擇
device_option = st.sidebar.selectbox(
    "計算設備",
    ["auto", "cuda", "cpu"],
    help="選擇用於 AI 計算的設備"
)

# 記憶體設置
memory_settings = st.sidebar.expander("💾 記憶體設置")
with memory_settings:
    enable_cpu_offload = st.checkbox("啟用 CPU 卸載", value=True, help="減少 GPU 記憶體使用")
    low_vram_mode = st.checkbox("低顯存模式", value=False, help="適用於 4GB 以下顯卡")
    enable_optimization = st.checkbox("啟用模型優化", value=True, help="使用 PyTorch 原生優化")

# 初始化模型按鈕
if st.sidebar.button("🚀 初始化/重新載入模型"):
    with st.spinner("正在載入 AI 模型..."):
        try:
            st.session_state.generator = MemoryOptimizedSpriteGenerator(
                device=device_option,
                enable_cpu_offload=enable_cpu_offload,
                low_vram_mode=low_vram_mode,
                enable_model_optimization=enable_optimization
            )
            st.sidebar.success("✅ 模型載入成功！")
        except Exception as e:
            st.sidebar.error(f"❌ 模型載入失敗: {str(e)}")
            st.session_state.generator = None

# 顯示系統狀態
st.sidebar.subheader("📊 系統狀態")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    st.sidebar.success(f"🔥 GPU: {gpu_name}")
    st.sidebar.info(f"💾 GPU 記憶體: {gpu_memory:.1f} GB")
else:
    st.sidebar.warning("⚠️ 未檢測到 CUDA GPU")

model_status = "✅ 已載入" if st.session_state.generator else "❌ 未載入"
st.sidebar.write(f"🤖 AI 模型: {model_status}")

# 主要內容區域
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎭 角色設置")
    
    # 角色描述
    character_prompt = st.text_area(
        "角色描述",
        value="pixel art knight character with sword and shield",
        height=100,
        help="詳細描述您想要生成的角色外觀和特徵"
    )
    
    # 藝術風格
    style_presets = {
        "像素藝術": "pixel art style, 8-bit, retro gaming",
        "卡通風格": "cartoon style, cute, colorful",
        "奇幻風格": "fantasy style, detailed, magical",
        "科幻風格": "sci-fi style, futuristic, technological",
        "中世紀風格": "medieval style, historical, authentic",
        "自訂": ""
    }
    
    style_choice = st.selectbox("藝術風格", list(style_presets.keys()))
    
    if style_choice == "自訂":
        style_prompt = st.text_input("自訂風格描述", value="")
    else:
        style_prompt = style_presets[style_choice]
        st.info(f"風格提示詞: {style_prompt}")
    
    # 動畫設置
    st.header("🎬 動畫設置")
    
    col_anim1, col_anim2 = st.columns(2)
    
    with col_anim1:
        animation_options = {
            "待機": "IDLE",
            "行走": "WALK", 
            "跑步": "RUN",
            "跳躍": "JUMP",
            "攻擊": "ATTACK"
        }
        
        animation_choice = st.selectbox("動畫類型", list(animation_options.keys()))
        animation_type = AnimationType(animation_options[animation_choice])
        
        frames = st.slider("動畫幀數", min_value=4, max_value=16, value=8)
    
    with col_anim2:
        sprite_width = st.number_input("精靈寬度", min_value=32, max_value=512, value=64, step=32)
        sprite_height = st.number_input("精靈高度", min_value=32, max_value=512, value=64, step=32)
        
        sheet_cols = st.slider("Sprite Sheet 列數", min_value=2, max_value=8, value=4)
    
    # 高級設置
    advanced_settings = st.expander("⚙️ 高級設置")
    with advanced_settings:
        negative_prompt = st.text_area(
            "負面提示詞",
            value="blurry, low quality, distorted, deformed, extra limbs, bad anatomy, worst quality",
            help="描述您不希望在圖像中出現的內容"
        )
        
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            num_inference_steps = st.slider("推理步數", min_value=10, max_value=50, value=25)
            guidance_scale = st.slider("引導強度", min_value=1.0, max_value=20.0, value=7.5, step=0.5)
        
        with col_adv2:
            seed = st.number_input("隨機種子", min_value=0, max_value=1000000, value=42)
            custom_size = st.checkbox("自訂生成尺寸")
            
            if custom_size:
                gen_width = st.number_input("生成寬度", min_value=256, max_value=1024, value=512, step=64)
                gen_height = st.number_input("生成高度", min_value=256, max_value=1024, value=512, step=64)
            else:
                gen_width = gen_height = 512

with col2:
    st.header("🎮 生成控制")
    
    # 生成按鈕
    generate_disabled = (st.session_state.generator is None or 
                        st.session_state.generation_in_progress or 
                        not character_prompt.strip())
    
    if st.button("🎨 生成 Sprite Sheet", disabled=generate_disabled, use_container_width=True):
        if not st.session_state.generator:
            st.error("請先初始化 AI 模型！")
        else:
            st.session_state.generation_in_progress = True
            
            # 生成參數
            generation_params = {
                "character_prompt": character_prompt,
                "animation_type": animation_type,
                "frames": frames,
                "sprite_size": (sprite_width, sprite_height),
                "sheet_cols": sheet_cols,
                "style_prompt": style_prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": gen_width,
                "height": gen_height
            }
            
            # 顯示生成參數
            st.subheader("📋 生成參數")
            with st.expander("查看詳細參數"):
                st.json({
                    "角色描述": character_prompt,
                    "動畫類型": animation_choice,
                    "幀數": frames,
                    "精靈尺寸": f"{sprite_width}x{sprite_height}",
                    "Sheet 列數": sheet_cols,
                    "藝術風格": style_prompt,
                    "推理步數": num_inference_steps,
                    "引導強度": guidance_scale,
                    "生成尺寸": f"{gen_width}x{gen_height}"
                })
            
            # 執行生成
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔄 正在生成 Sprite Sheet...")
                
                # 創建輸出路徑
                output_path = f"generated_sprite_{int(time.time())}.png"
                
                # 生成 Sprite Sheet
                sprite_sheet = st.session_state.generator.create_sprite_sheet(
                    output_path=output_path,
                    **generation_params
                )
                
                progress_bar.progress(100)
                status_text.text("✅ 生成完成！")
                
                # 顯示結果
                st.subheader("🎉 生成結果")
                st.image(sprite_sheet, caption=f"生成的 Sprite Sheet - {animation_choice}", use_column_width=True)
                
                # 提供下載鏈接
                img_buffer = io.BytesIO()
                sprite_sheet.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label="📥 下載 Sprite Sheet",
                    data=img_buffer.getvalue(),
                    file_name=f"sprite_sheet_{animation_choice.lower()}_{int(time.time())}.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                # 顯示生成統計
                st.success(f"成功生成 {frames} 幀 {animation_choice} 動畫")
                
            except Exception as e:
                st.error(f"❌ 生成失敗: {str(e)}")
                with st.expander("查看錯誤詳情"):
                    st.code(traceback.format_exc())
            
            finally:
                st.session_state.generation_in_progress = False
                progress_bar.empty()
                status_text.empty()
    
    # 生成狀態提示
    if st.session_state.generation_in_progress:
        st.warning("⏳ 正在生成中，請稍候...")
    elif not st.session_state.generator:
        st.info("💡 請先在側邊欄初始化 AI 模型")
    elif not character_prompt.strip():
        st.warning("⚠️ 請輸入角色描述")
    
    # 預設範例
    st.subheader("💡 範例提示詞")
    examples = {
        "🗡️ 騎士": "pixel art medieval knight with silver armor and blue cape",
        "🧙‍♂️ 法師": "pixel art wizard with purple robes and magical staff",
        "🏹 弓箭手": "pixel art elf archer with green cloak and bow",
        "⚔️ 戰士": "pixel art barbarian warrior with axe and fur armor",
        "🛡️ 守護者": "pixel art paladin with golden armor and holy symbol"
    }
    
    for name, prompt in examples.items():
        if st.button(name, key=f"example_{name}", use_container_width=True):
            # 直接更新文本區域的值（需要重新渲染）
            st.info(f"已選擇範例: {name}")
            st.info(f"提示詞: {prompt}")

# 頁腳
st.markdown("---")
st.markdown("### 📚 使用說明")

with st.expander("如何使用此應用程式"):
    st.markdown("""
    1. **初始化模型**: 在側邊欄點擊「初始化/重新載入模型」
    2. **設置角色**: 輸入詳細的角色描述
    3. **選擇風格**: 選擇預設風格或自訂風格
    4. **配置動畫**: 選擇動畫類型和幀數
    5. **調整參數**: 根據需要調整精靈尺寸和其他設置
    6. **生成**: 點擊「生成 Sprite Sheet」開始生成
    7. **下載**: 生成完成後可下載結果
    
    **提示**: 
    - 初次載入模型需要下載大約 4-6GB 的文件
    - 生成時間取決於您的硬體配置
    - 使用 GPU 可以大幅提升生成速度
    """)

with st.expander("系統要求"):
    st.markdown("""
    **最低要求**:
    - Python 3.8+
    - 8GB 系統記憶體
    - 20GB 可用硬碟空間
    
    **推薦配置**:
    - NVIDIA GPU (6GB+ 顯存)
    - 16GB+ 系統記憶體
    - SSD 硬碟
    
    **支持的動畫類型**:
    - 待機 (Idle): 靜態呼吸動畫
    - 行走 (Walk): 標準行走循環
    - 跑步 (Run): 快速移動動畫
    - 跳躍 (Jump): 跳躍動作序列
    - 攻擊 (Attack): 攻擊動作動畫
    """)

# 資源監控（僅在有 GPU 時顯示）
if torch.cuda.is_available():
    with st.expander("📊 GPU 資源監控"):
        if st.button("刷新 GPU 狀態"):
            try:
                allocated = torch.cuda.memory_allocated(0) / 1e9
                cached = torch.cuda.memory_reserved(0) / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                col_gpu1, col_gpu2, col_gpu3 = st.columns(3)
                col_gpu1.metric("已分配記憶體", f"{allocated:.2f} GB")
                col_gpu2.metric("已緩存記憶體", f"{cached:.2f} GB") 
                col_gpu3.metric("總記憶體", f"{total:.2f} GB")
                
                # 記憶體使用率條
                usage_percent = (allocated / total) * 100
                st.progress(usage_percent / 100)
                st.caption(f"GPU 記憶體使用率: {usage_percent:.1f}%")
                
            except Exception as e:
                st.error(f"無法獲取 GPU 狀態: {e}")

# JavaScript 用於清理
st.markdown("""
<script>
// 頁面卸載時清理資源
window.addEventListener('beforeunload', function (e) {
    // 清理 GPU 記憶體
    fetch('/clear_gpu_cache', {method: 'POST'});
});
</script>
""", unsafe_allow_html=True) 