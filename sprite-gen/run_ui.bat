@echo off
echo 🚀 啟動 Sprite Sheet 生成器 Web UI
echo =====================================
echo.

REM 檢查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到 Python，請先安裝 Python 3.8+
    pause
    exit /b 1
)

REM 切換到腳本目錄
cd /d "%~dp0"

REM 啟動 UI
echo ✅ 正在啟動 Web UI...
echo 🌐 瀏覽器將自動打開 http://localhost:8501
echo 💡 按 Ctrl+C 停止服務器
echo.

python run_ui.py

echo.
echo �� Web UI 已停止
pause 