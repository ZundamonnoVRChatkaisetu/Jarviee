@echo off
REM Jarviee GPU診断スクリプト
REM GPUのサポート状況を詳細にチェックします

echo ===== Jarviee GPU診断スクリプト =====

REM Pythonのバージョン確認
echo Pythonバージョン:
python --version

REM 環境変数を設定
set USE_GPU=true
set GPU_LAYERS=-1
set JARVIEE_DEBUG=1

REM NVIDIA-SMIの実行
echo.
echo NVIDIA-SMI情報:
nvidia-smi

REM PyTorchのCUDAサポート確認
echo.
echo PyTorchのCUDAサポート:
python -c "import torch; print(f'CUDA利用可能: {torch.cuda.is_available()}'); print(f'GPU数: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('GPUが見つかりません');"

REM llama-cpp-pythonのGPUサポート確認
echo.
echo llama-cpp-pythonのGPUサポート:
python -c "try: from llama_cpp import Llama; import inspect; gpuSupport = 'n_gpu_layers' in inspect.signature(Llama.__init__).parameters; print(f'GPU対応のllama-cpp-python: {'有効' if gpuSupport else '無効'}'); except ImportError: print('llama-cpp-pythonがインストールされていません');"

REM 診断ツールの実行
echo.
echo GPUインストーラー診断ツールを実行しますか？(y/n)
set /p choice=

if /i "%choice%"=="y" (
    python scripts\install_gpu_support.py --force
)

echo.
echo GPUサポートフルチェックを実行しますか？(y/n)
set /p choice=

if /i "%choice%"=="y" (
    python scripts\diagnose_jarviee.py
)

pause
