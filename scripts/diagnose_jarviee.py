#!/usr/bin/env python
"""
Jarviee診断ツール

システムの状態、とくにGPU利用状況を診断します。
"""
import os
import sys
import logging
import importlib.util
import time
from pathlib import Path

# パスを追加して親ディレクトリのモジュールをインポートできるようにする
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

def check_system_path():
    """システムパスに必要なディレクトリが含まれているか確認"""
    print("\n[1] システムパスの確認")
    print(f"現在の作業ディレクトリ: {os.getcwd()}")
    
    sys_paths = sys.path
    project_dir = parent_dir
    
    print(f"プロジェクトディレクトリ: {project_dir}")
    if project_dir in sys_paths:
        print(f"[OK] プロジェクトディレクトリはシステムパスに含まれています。")
    else:
        print(f"[ERROR] プロジェクトディレクトリがシステムパスに含まれていません。")
        sys.path.insert(0, project_dir)
        print(f"  => パスを追加しました。")

def check_modules():
    """必要なモジュールが利用可能か確認"""
    print("\n[2] 必要なモジュールの確認")
    
    modules = [
        ("rich", "リッチな出力表示"),
        ("dotenv", "環境変数の読み込み"),
        ("llama_cpp", "LLaMAモデルのロード"),
        ("torch", "PyTorch (GPU対応用)"),
        ("numpy", "数値演算ライブラリ"),
        ("pynvml", "NVIDIA管理ライブラリ (GPU診断用)"),
        ("ctypes", "Cライブラリ呼び出し (GPU連携用)")
    ]
    
    for module_name, description in modules:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            print(f"[OK] {module_name}: 利用可能 ({description})")
            
            # PyTorchの場合、CUDA利用可能性をチェック
            if module_name == "torch":
                try:
                    import torch
                    if torch.cuda.is_available():
                        device_count = torch.cuda.device_count()
                        devices = ", ".join([torch.cuda.get_device_name(i) for i in range(device_count)])
                        print(f"  => CUDA利用可能 (GPUデバイス: {devices})")
                    else:
                        print(f"  => CUDA利用不可")
                except Exception as e:
                    print(f"  => CUDA確認エラー: {e}")
            
            # llama-cppの場合、GPUサポートをチェック
            if module_name == "llama_cpp":
                try:
                    from llama_cpp import Llama
                    import inspect
                    params = inspect.signature(Llama.__init__).parameters
                    if "n_gpu_layers" in params:
                        print(f"  => GPU対応ビルド (n_gpu_layers パラメータあり)")
                    else:
                        print(f"  => CPU専用ビルド (n_gpu_layers パラメータなし)")
                except Exception as e:
                    print(f"  => GPUサポート確認エラー: {e}")
        else:
            print(f"[ERROR] {module_name}: 見つかりません ({description})")

def check_config():
    """設定ファイルを確認"""
    print("\n[3] 設定ファイルの確認")
    
    config_paths = [
        os.path.join(parent_dir, "config", "config.json"),
        os.path.join(parent_dir, ".env")
    ]
    
    for path in config_paths:
        if os.path.exists(path):
            print(f"[OK] {os.path.basename(path)}: 存在します")
            
            # config.jsonの場合、GPUの設定を確認
            if path.endswith("config.json"):
                try:
                    import json
                    with open(path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    if "llm" in config and "providers" in config["llm"] and "gemma" in config["llm"]["providers"]:
                        gemma_config = config["llm"]["providers"]["gemma"]
                        use_gpu = gemma_config.get("use_gpu", False)
                        n_gpu_layers = gemma_config.get("n_gpu_layers", 0)
                        model_path = gemma_config.get("path", "")
                        
                        print(f"  => Gemma設定: use_gpu={use_gpu}, n_gpu_layers={n_gpu_layers}")
                        print(f"  => モデルパス: {model_path}")
                        
                        if use_gpu:
                            print(f"  => GPU設定: 有効")
                        else:
                            print(f"  => GPU設定: 無効")
                            
                        if not os.path.exists(model_path) and model_path:
                            print(f"  [ERROR] モデルファイルが見つかりません: {model_path}")
                    else:
                        print("  => Gemma設定が見つかりません")
                except Exception as e:
                    print(f"  => 設定ファイル解析エラー: {e}")
            
            # .envファイルの場合、環境変数を表示（セキュアな情報は隠す）
            if path.endswith(".env"):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        env_content = f.readlines()
                    
                    gpu_vars = []
                    for line in env_content:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        if '=' in line:
                            key, value = line.split('=', 1)
                            if key in ['USE_GPU', 'GPU_LAYERS']:
                                gpu_vars.append(f"{key}={value}")
                    
                    if gpu_vars:
                        print(f"  => GPU関連環境変数: {', '.join(gpu_vars)}")
                    else:
                        print(f"  => GPU関連環境変数は設定されていません")
                except Exception as e:
                    print(f"  => .env解析エラー: {e}")
        else:
            print(f"[ERROR] {os.path.basename(path)}: 見つかりません")

def check_models():
    """モデルファイルを確認"""
    print("\n[4] モデルファイルの確認")
    
    models_dir = os.path.join(parent_dir, "models")
    if not os.path.exists(models_dir):
        print(f"[ERROR] models ディレクトリが見つかりません: {models_dir}")
        return
    
    print(f"[OK] models ディレクトリ: {models_dir}")
    
    gguf_files = []
    for file in os.listdir(models_dir):
        if file.lower().endswith('.gguf'):
            file_path = os.path.join(models_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            gguf_files.append((file, size_mb))
    
    if gguf_files:
        print(f"モデルファイル一覧:")
        for file, size_mb in gguf_files:
            print(f"  • {file} ({size_mb:.1f} MB)")
    else:
        print("[ERROR] .gguf モデルファイルが見つかりません")

def test_load_model():
    """モデルのロードテスト"""
    print("\n[5] モデルロードテスト")
    
    # GPUサポートのチェック
    gpu_supported = False
    try:
        from llama_cpp import Llama
        import inspect
        params = inspect.signature(Llama.__init__).parameters
        gpu_supported = "n_gpu_layers" in params
        
        if gpu_supported:
            print("[OK] llama-cpp-python はGPUをサポートしています")
            # GPUビルドのバージョン確認
            try:
                import llama_cpp
                print(f"  => llama-cpp-python バージョン: {llama_cpp.__version__}")
                
                # CUDAバージョン確認
                try:
                    import torch
                    if torch.cuda.is_available():
                        cuda_version = torch.version.cuda
                        print(f"  => CUDA バージョン: {cuda_version}")
                        
                        # CUDAバージョンの互換性チェック
                        if cuda_version != '12.8':
                            print(f"  [WARNING] CUDAバージョン不一致の可能性: llama-cpp-pythonはcu128用にビルドされていますが、検出されたCUDAは{cuda_version}です")
                            print(f"  => 解決策: pip install llama-cpp-python --force-reinstall --no-cache-dir --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu{cuda_version.replace('.', '')}")
                except Exception as e:
                    print(f"  => CUDA確認エラー: {e}")
            except Exception as e:
                print(f"  => バージョン確認エラー: {e}")
        else:
            print("[ERROR] llama-cpp-python はGPU非対応ビルドです")
            print("  => GPU対応版をインストールするには:")
            print("  pip install llama-cpp-python --force-reinstall --no-cache-dir --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu128")
            return
            
    except ImportError:
        print("[ERROR] llama-cpp-python がインストールされていません")
        return
    
    # 設定ファイルからモデルパスを取得
    model_path = None
    try:
        import json
        config_path = os.path.join(parent_dir, "config", "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if "llm" in config and "providers" in config["llm"] and "gemma" in config["llm"]["providers"]:
            model_path = config["llm"]["providers"]["gemma"].get("path", "")
    except Exception as e:
        print(f"[ERROR] 設定ファイルからモデルパスを取得できません: {e}")
        return
    
    if not model_path:
        print("[ERROR] モデルパスが設定されていません")
        return
    
    # モデルファイルの存在確認
    if not os.path.exists(model_path):
        # models ディレクトリからの相対パスの可能性を確認
        alt_path = os.path.join(parent_dir, model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"[ERROR] モデルファイルが見つかりません: {model_path}")
            return
    
    print(f"[OK] モデルパス: {model_path}")
    
    # モデルロードのテスト
    print("モデルをロード中...")
    start_time = time.time()
    
    try:
        # CPU版のみでロード
        llm_cpu = Llama(
            model_path=model_path,
            n_ctx=512,  # 小さいコンテキスト窓で高速化
            n_threads=2,  # 少ないスレッド数で高速化
            n_gpu_layers=0  # CPU専用モード
        )
        cpu_time = time.time() - start_time
        print(f"[OK] CPU版ロード完了: {cpu_time:.2f}秒")
        
        # GPU版でロード（CPUとの比較用）
        if gpu_supported:
            print("GPU版をロード中...")
            gpu_start = time.time()
            try:
                llm_gpu = Llama(
                    model_path=model_path,
                    n_ctx=512,
                    n_threads=2,
                    n_gpu_layers=-1  # 全レイヤーGPU使用
                )
                gpu_time = time.time() - gpu_start
                print(f"[OK] GPU版ロード完了: {gpu_time:.2f}秒 (CPU比: {cpu_time/gpu_time:.1f}倍)")
                
                # 簡単な生成テスト
                print("\n生成テスト実行中...")
                prompt = "こんにちは、元気ですか？"
                
                # CPU生成
                cpu_gen_start = time.time()
                cpu_result = llm_cpu.create_completion(prompt=prompt, max_tokens=20)
                cpu_gen_time = time.time() - cpu_gen_start
                
                # GPU生成
                gpu_gen_start = time.time()
                gpu_result = llm_gpu.create_completion(prompt=prompt, max_tokens=20)
                gpu_gen_time = time.time() - gpu_gen_start
                
                # 結果表示
                print(f"CPU生成時間: {cpu_gen_time:.2f}秒")
                print(f"GPU生成時間: {gpu_gen_time:.2f}秒")
                print(f"速度比: {cpu_gen_time/gpu_gen_time:.1f}倍")
                
                # GPUメモリ使用状況確認（Pythonから確認できる範囲で）
                try:
                    import torch
                    if torch.cuda.is_available():
                        # 現在のGPUメモリ確保状況
                        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                        print(f"GPU メモリ確保状況: 割当済 {allocated:.2f} GB / 予約済 {reserved:.2f} GB")
                        
                        # メモリが確保されているかチェック
                        if allocated > 0.1:  # 0.1GB以上を使っているか
                            print("[OK] GPUメモリが確保されています - GPUが使用されている可能性が高い")
                        else:
                            print("[WARNING] GPUメモリ割り当てが少ないか確認できません")
                            
                except Exception as e:
                    print(f"GPU メモリ確認エラー: {e}")
                
                # NVML経由でGPU使用状況を確認（可能な場合）
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        print(f"GPU使用率: {util.gpu}%")
                        print(f"GPUメモリ使用率: {util.memory}%")
                        
                        if util.gpu > 5:  # 5%以上のGPU使用率
                            print("[OK] GPUが実際に計算に使用されています！")
                        else:
                            print("[WARNING] GPU使用率が低いです - 計算がGPUで実行されていない可能性があります")
                    pynvml.nvmlShutdown()
                except Exception as e:
                    print(f"NVML確認エラー: {e}")
                
                if cpu_gen_time > gpu_gen_time:
                    print("[OK] GPUモードは CPU より高速です！")
                else:
                    print("[WARNING] GPUモードが速度向上してません。原因を調査します...")
                    print("  ・CUDAバージョンが不適切かもしれません")
                    print("  ・モデルがGPU処理に最適化されていない可能性があります")
                    print("  ・llama-cpp-pythonがGPUを正しく認識していない可能性があります")
                
            except Exception as e:
                print(f"[ERROR] GPU版ロードエラー: {e}")
        
    except Exception as e:
        print(f"[ERROR] モデルロードエラー: {e}")

def check_jarviee():
    """Jarvieeのコアコンポーネントをチェック"""
    print("\n[6] Jarvieeコンポーネントの確認")
    
    core_modules = [
        "src.core.llm.engine",
        "src.core.knowledge.query_engine",
        "src.core.integration.framework",
        "src.interfaces.cli.jarviee_cli"
    ]
    
    for module_name in core_modules:
        try:
            module = importlib.import_module(module_name)
            print(f"[OK] {module_name}: インポート成功")
        except ImportError as e:
            print(f"[ERROR] {module_name}: インポートエラー - {e}")
        except Exception as e:
            print(f"[ERROR] {module_name}: その他のエラー - {e}")

def main():
    """メイン実行関数"""
    print("===== Jarviee システム診断ツール =====")
    
    # 各種診断を実行
    check_system_path()
    check_modules()
    check_config()
    check_models()
    check_jarviee()
    
    # GPU診断は選択可能に
    print("\nGPUテストを行いますか？モデルのロードに時間がかかる場合があります。(y/n): ", end="")
    choice = input().strip().lower()
    if choice == 'y':
        test_load_model()
    
    print("\n診断完了しました。")
    
if __name__ == "__main__":
    main()
