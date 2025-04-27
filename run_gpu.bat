@echo off
REM Jarviee GPU起動スクリプト
REM GPUを使用してJarvieeを起動するためのバッチファイル

echo ===== Jarviee GPU起動スクリプト =====
echo GPUサポートを有効にしてJarvieeを起動します

REM 環境変数を設定
set USE_GPU=true
set GPU_LAYERS=-1
set JARVIEE_DEBUG=1

REM llama-cpp-pythonがGPU対応かチェック
python -c "from llama_cpp import Llama; import inspect; gpuSupport = 'n_gpu_layers' in inspect.signature(Llama.__init__).parameters; print('GPUサポート: ' + ('有効' if gpuSupport else '無効'))"

echo.
echo Jarvieeを起動中...
python jarviee.py --gpu --debug %*

pause
