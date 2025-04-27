@echo off
REM Jarviee GPU修復スクリプト
REM GPUを確実に使用するための修復を実行します

REM 現在の日時を環境変数として取得
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set LOGFILE=logs\gpu_fix_%datetime:~0,8%_%datetime:~8,6%.log

REM ログディレクトリが存在しない場合は作成
if not exist logs mkdir logs

REM ログファイルのヘッダー
echo ===== Jarviee GPU修復ツール ===== > %LOGFILE%
echo 実行日時: %date% %time% >> %LOGFILE%
echo. >> %LOGFILE%

echo ===== Jarviee GPU修復ツール =====
echo 出力は %LOGFILE% に記録されています

REM 環境確認
echo CUDA環境を確認しています...
echo ===== CUDA環境確認 ===== >> %LOGFILE%
nvcc --version >> %LOGFILE% 2>&1
echo. >> %LOGFILE%
nvidia-smi >> %LOGFILE% 2>&1
echo. >> %LOGFILE%

REM 依存ライブラリのインストール
echo GPU依存ライブラリをインストールしています...
echo ===== GPU依存ライブラリのインストール ===== >> %LOGFILE%
python scripts\install_gpu_deps.py >> %LOGFILE% 2>&1
echo. >> %LOGFILE%

REM llama-cpp-pythonの再インストール
echo ===== CUDA対応のllama-cpp-pythonを再インストールしています =====
echo ===== llama-cpp-python再インストール ===== >> %LOGFILE%
python scripts\reinstall_llama_cpp.py >> %LOGFILE% 2>&1
echo. >> %LOGFILE%

REM 設定ファイルの確認
echo CUDA環境変数を確認・設定しています...
echo ===== CUDA環境変数設定 ===== >> %LOGFILE%
set CUDA_VISIBLE_DEVICES=0
set USE_GPU=true
set GPU_LAYERS=-1
echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES% >> %LOGFILE%
echo USE_GPU=%USE_GPU% >> %LOGFILE%
echo GPU_LAYERS=%GPU_LAYERS% >> %LOGFILE%
echo. >> %LOGFILE%

REM GPUサポート診断の実行
echo GPUサポート診断を実行します...
echo ===== GPUサポート診断 ===== >> %LOGFILE%
python scripts\diagnose_jarviee.py >> %LOGFILE% 2>&1
echo. >> %LOGFILE%

echo ===== 修復プロセスが完了しました =====
echo 修復プロセスが完了しました >> %LOGFILE%
echo テスト完了: %date% %time% >> %LOGFILE%
echo ========================================== >> %LOGFILE%

echo 問題が解決しない場合は以下のコマンドを使用して直接GPUテストを実行してください:
echo python -c "import torch; print(f'CUDA利用可能: {torch.cuda.is_available()}'); print(f'デバイス数: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
echo このコマンドも %LOGFILE% に記録されています >> %LOGFILE%
echo python -c "import torch; print(f'CUDA利用可能: {torch.cuda.is_available()}'); print(f'デバイス数: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" >> %LOGFILE%

echo 修復が完了しました。結果は %LOGFILE% に保存されています。
pause
