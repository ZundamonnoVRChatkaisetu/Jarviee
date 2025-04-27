@echo off
REM Jarviee GPU診断・修正済みテストバッチ
REM GPU動作確認の強化版：エンコーディング問題を回避し、明示的なGPU初期化を行います

REM 現在日時を取得してログファイル名に利用
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set LOGFILE=logs\gpu_test_fixed_%datetime:~0,8%_%datetime:~8,6%.log

REM ログディレクトリがなければ作成
if not exist logs mkdir logs

REM ログファイルのヘッダー出力
echo ===== Jarviee GPU診断（修正版） ===== > %LOGFILE%
echo 実行日時: %date% %time% >> %LOGFILE%
echo. >> %LOGFILE%

echo ===== Jarviee GPU診断（修正版） =====
echo 結果は %LOGFILE% に記録されます

REM 必須環境変数の設定
set USE_GPU=true
set GPU_LAYERS=-1
set JARVIEE_DEBUG=1
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

REM CUDA環境変数の明示的設定
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_MODULE_LOADING=LAZY

echo 環境変数設定: >> %LOGFILE%
echo USE_GPU=%USE_GPU% >> %LOGFILE%
echo GPU_LAYERS=%GPU_LAYERS% >> %LOGFILE%
echo JARVIEE_DEBUG=%JARVIEE_DEBUG% >> %LOGFILE%
echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES% >> %LOGFILE%
echo PYTORCH_CUDA_ALLOC_CONF=%PYTORCH_CUDA_ALLOC_CONF% >> %LOGFILE%
echo CUDA_PATH=%CUDA_PATH% >> %LOGFILE%
echo CUDA_MODULE_LOADING=%CUDA_MODULE_LOADING% >> %LOGFILE%
echo. >> %LOGFILE%

REM CUDA基本情報
echo NVIDIA GPU情報: >> %LOGFILE%
nvidia-smi >> %LOGFILE% 2>&1
echo. >> %LOGFILE%

REM GPUメモリ強制割り当て
echo GPU強制初期化を実行しています... >> %LOGFILE%
python scripts\force_gpu_alloc.py >> %LOGFILE% 2>&1
echo. >> %LOGFILE%

REM 簡易PyTorchテスト
echo PyTorch GPU動作テスト: >> %LOGFILE%
python -c "import torch; print(f'CUDA利用可能: {torch.cuda.is_available()}'); print(f'GPU数: {torch.cuda.device_count()}'); print(f'GPUデバイス名: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"なし\"}'); x=torch.ones(10, device='cuda' if torch.cuda.is_available() else 'cpu'); print(f'テストテンソル: {x.device}')" >> %LOGFILE% 2>&1
echo. >> %LOGFILE%

REM llama-cpp-pythonのテスト
echo llama-cpp-python GPU対応確認: >> %LOGFILE%
python -c "import llama_cpp; from inspect import signature; params = signature(llama_cpp.Llama.__init__).parameters; print(f'GPU対応: {\"あり\" if \"n_gpu_layers\" in params else \"なし\"}'); print(f'llama-cpp-python バージョン: {llama_cpp.__version__}')" >> %LOGFILE% 2>&1
echo. >> %LOGFILE%

REM ユーザーに確認
echo モデルロードテストを実行しますか？(処理に時間がかかります) (y/n)
set /p choice=
echo 選択: %choice% >> %LOGFILE%

if /i "%choice%"=="y" (
    echo モデルロードテストを実行中... >> %LOGFILE%
    echo. >> %LOGFILE%
    
    REM モデルロードテスト用Pythonスクリプト
    echo import os, sys, time >> test_gpu_load.py
    echo from pathlib import Path >> test_gpu_load.py
    echo sys.path.append(".") >> test_gpu_load.py
    echo print("モデルテスト開始...") >> test_gpu_load.py
    echo print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'なし')}") >> test_gpu_load.py
    echo try: >> test_gpu_load.py
    echo     from llama_cpp import Llama >> test_gpu_load.py
    echo     models_dir = Path("models") >> test_gpu_load.py
    echo     model_files = list(models_dir.glob("*.gguf")) >> test_gpu_load.py
    echo     if model_files: >> test_gpu_load.py
    echo         model_path = str(model_files[0]) >> test_gpu_load.py
    echo         print(f"テスト用モデル: {model_path}") >> test_gpu_load.py
    echo         print("GPU版モデルロード開始...") >> test_gpu_load.py
    echo         start_time = time.time() >> test_gpu_load.py
    echo         llm = Llama(model_path=model_path, n_ctx=512, n_gpu_layers=-1) >> test_gpu_load.py
    echo         load_time = time.time() - start_time >> test_gpu_load.py
    echo         print(f"ロード完了: {load_time:.2f}秒") >> test_gpu_load.py
    echo         print("短いテキスト生成を実行...") >> test_gpu_load.py
    echo         output = llm.create_completion("こんにちは、元気ですか？", max_tokens=20) >> test_gpu_load.py
    echo         print(f"生成結果: {output}") >> test_gpu_load.py
    echo         print("テスト完了!") >> test_gpu_load.py
    echo     else: >> test_gpu_load.py
    echo         print("モデルファイル(.gguf)が見つかりません。") >> test_gpu_load.py
    echo except Exception as e: >> test_gpu_load.py
    echo     print(f"エラー: {e}") >> test_gpu_load.py
    
    python test_gpu_load.py >> %LOGFILE% 2>&1
    del test_gpu_load.py
)

echo. >> %LOGFILE%
echo テスト完了: %date% %time% >> %LOGFILE%
echo ========================================== >> %LOGFILE%

echo テスト結果は %LOGFILE% に記録されています。続行するには何かキーを押してください...
pause
