@echo off
REM Jarviee 最適化版起動スクリプト
REM CPU最適化設定を使用してJarvieeを起動するためのバッチファイル

echo ===== Jarviee 最適化版起動スクリプト =====
echo CPU性能を最大化して実行します。GPU未使用環境向け。

REM 設定ファイルをバックアップ
if exist config\config.json (
    copy config\config.json config\config.json.bak > nul
    echo 設定ファイルをバックアップしました: config\config.json.bak
)

REM 環境変数を設定
set USE_GPU=true
set GPU_LAYERS=-1
set FORCE_GPU=true
set CPU_THREADS=%NUMBER_OF_PROCESSORS%
set BATCH_SIZE=1024
set JARVIEE_DEBUG=1

echo CPU最適化モードで実行します。スレッド数: %CPU_THREADS%

REM CPUコア数と使用可能メモリを表示
echo 使用可能なCPUコア数: %NUMBER_OF_PROCESSORS%

REM HyperThreadingが有効か確認する簡易テスト
set /a logical_cores=%NUMBER_OF_PROCESSORS%
set /a physical_cores=%NUMBER_OF_PROCESSORS% / 2
echo 論理コア数: %logical_cores%（推定物理コア数: %physical_cores%）

REM llama-cpp-pythonのバージョン確認
python -c "try: from llama_cpp import __version__ as v; print(f'llama-cpp-python バージョン: {v}'); except: print('llama-cpp-python がインストールされていません')"

REM Jarvieeを起動
echo.
echo Jarvieeを起動中...
python jarviee.py -d %*

pause