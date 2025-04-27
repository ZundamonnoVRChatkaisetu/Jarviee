@echo off
REM Jarviee GPUテストバッチ
REM GPUの動作確認用バッチ。GPUの動作状況をログに記録します。

REM 日時情報を取得してログファイル名に利用
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set LOGFILE=logs\gpu_test_%datetime:~0,8%_%datetime:~8,6%.log

REM ログディレクトリがなければ作成
if not exist logs mkdir logs

REM ログファイルのヘッダー出力
echo ===== Jarviee GPUテスト開始 ===== > %LOGFILE%
echo 実行日時: %date% %time% >> %LOGFILE%
echo. >> %LOGFILE%

echo ===== Jarviee GPUテスト開始 =====
echo 結果は %LOGFILE% に記録されます

REM 環境変数の設定
set USE_GPU=true
set GPU_LAYERS=-1
set JARVIEE_DEBUG=1
echo 環境変数設定: >> %LOGFILE%
echo USE_GPU=%USE_GPU% >> %LOGFILE%
echo GPU_LAYERS=%GPU_LAYERS% >> %LOGFILE%
echo JARVIEE_DEBUG=%JARVIEE_DEBUG% >> %LOGFILE%
echo. >> %LOGFILE%

REM CUDA情報の取得
echo CUDA情報: >> %LOGFILE%
nvidia-smi >> %LOGFILE% 2>&1
echo. >> %LOGFILE%

REM GPU依存のPythonパッケージのインストール
echo GPU依存のPythonパッケージのインストールを開始します...
echo ===== GPU依存のPythonパッケージのインストール ===== >> %LOGFILE%
python scripts\install_gpu_deps.py >> %LOGFILE% 2>&1
echo. >> %LOGFILE%

REM GPUの動作確認
echo GPUの動作確認を行います...
echo ===== GPUの動作確認 ===== >> %LOGFILE%
python scripts\diagnose_jarviee.py >> %LOGFILE% 2>&1
echo. >> %LOGFILE%

REM 続行するかユーザーに確認
echo 続行しますか？(y/n)
set /p choice=
echo 選択肢: %choice% >> %LOGFILE%

if /i "%choice%"=="y" (
    echo 続行します...
    echo ===== 続行します ===== >> %LOGFILE%
    set CUDA_VISIBLE_DEVICES=0
    set USE_GPU=true
    set GPU_LAYERS=-1
    echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES% >> %LOGFILE%
    
    echo 処理開始: %time% >> %LOGFILE%
    python -c "import sys; sys.path.append('.'); from src.core.llm.providers.gemma_provider import GemmaProvider; provider = GemmaProvider('./models/gemma-3-12B-it-QAT-Q4_0.gguf', {'use_gpu': True, 'n_gpu_layers': -1, 'verbose': True}); result = provider.generate('縺薙?ｮ譁?遶?繧定恭隱槭↓鄙ｻ險ｳ縺励※縺上□縺輔＞: 縺薙ｓ縺ｫ縺｡縺ｯ縲∽ｸ也阜?ｼ∝??豌励〒縺吶°?ｼ?'); print(f'逕滓?千ｵ先棡: {result}')" >> %LOGFILE% 2>&1
    echo 処理終了: %time% >> %LOGFILE%
)

echo. >> %LOGFILE%
echo 処理終了: %date% %time% >> %LOGFILE%
echo ========================================== >> %LOGFILE%

echo 処理結果は %LOGFILE% に記録されています。続行するには何かキーを押してください...
pause
