@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo =================================================================
echo   Cai dat YouTube Local TTS Studio (Kokoro + Chatterbox CUDA)
echo   Toi uu cho NVIDIA RTX 2060 6GB
echo =================================================================

where python >nul 2>&1
if errorlevel 1 (
    echo [LOI] May chua cai Python! Hay cai Python 3.10+ tai https://www.python.org/downloads/
    pause
    exit /b 1
)

python -m uv --version >nul 2>&1
if errorlevel 1 (
    echo [0/3] Dang cai trinh quan ly goi uv...
    python -m pip install --user uv
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Dang tao moi truong Python 3.11 doc lap trong .venv...
    python -m uv venv --python 3.11 .venv
)

echo [2/3] Dang cai Kokoro-82M + Gradio WebUI...
python -m uv pip install --python .venv\Scripts\python.exe kokoro-onnx gradio soundfile numpy

echo [3/3] Dang cai PyTorch CUDA 12.4 + Chatterbox Multilingual (~2.5GB)...
python -m uv pip install --python .venv\Scripts\python.exe --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match torch==2.6.0 torchaudio==2.6.0 chatterbox-tts

echo.
echo [HOAN TAT] Nhan dup start.bat de mo giao dien web tai http://127.0.0.1:7860
pause
