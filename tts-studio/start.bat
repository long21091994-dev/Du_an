@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ===================================================
echo   YouTube Local TTS Studio (Kokoro + Chatterbox)
echo ===================================================
if not exist ".venv\Scripts\python.exe" (
    echo [LOI] Chua tim thay moi truong .venv!
    pause
    exit /b 1
)
echo Dang khoi dong giao dien web tai http://127.0.0.1:7860 ...
start "" http://127.0.0.1:7860
".venv\Scripts\python.exe" app.py
pause
