#!/usr/bin/env python
"""
Jarviee GPU診断スクリプト

GPUの認識とllama-cpp-pythonのGPU利用状況を詳細に診断します。
"""

import sys
import os
import time
import traceback
import subprocess
import platform
import logging
from datetime import datetime

# ログディレクトリの作成
if not os.path.exists('logs'):
    os.makedirs('logs')

# ログファイル名の設定
log_filename = f"logs/gpu_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_gpu_torch():
    """PyTorchを使用してGPUをチェック"""
    logger.info("=== PyTorch GPU診断 ===")
    
    try:
        import torch
        logger.info(f"PyTorchバージョン: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA利用可能: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            logger.info(f"GPUデバイス数: {device_count}")
            
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                capability = torch.cuda.get_device_capability(i)
                memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                
                logger.info(f"GPU {i}: {name}")
                logger.info(f"  - CUDA Capability: {capability[0]}.{capability[1]}")
                logger.info(f"  - 総メモリ: {memory:.2f} GB")
            
            # メモリ使用量テスト
            current_device = torch.cuda.current_device()
            logger.info(f"現在のデバイス: {current_device}")
            
            # メモリ使用量（テスト前）
            allocated_before = torch.cuda.memory_allocated(current_device) / (1024**3)
            reserved_before = torch.cuda.memory_reserved(current_device) / (1024**3)
            logger.info(f"テスト前メモリ使用量:")
            logger.info(f"  - 割当済み: {allocated_before:.4f} GB")
            logger.info(f"  - 予約済み: {reserved_before:.4f} GB")
            
            # メモリ使用量テスト
            logger.info("GPUメモリ使用テスト実行中...")
            
            # テストテンソルを作成（1GB程度）
            test_size = 1_000_000_000 // 4  # float32で約1GB
            try:
                test_tensor = torch.ones(test_size, dtype=torch.float32, device=f"cuda:{current_device}")
                torch.cuda.synchronize()
                
                # メモリ使用量（テスト後）
                allocated_after = torch.cuda.memory_allocated(current_device) / (1024**3)
                reserved_after = torch.cuda.memory_reserved(current_device) / (1024**3)
                
                logger.info(f"テスト後メモリ使用量:")
                logger.info(f"  - 割当済み: {allocated_after:.4f} GB")
                logger.info(f"  - 予約済み: {reserved_after:.4f} GB")
                logger.info(f"  - 差分: {allocated_after-allocated_before:.4f} GB")
                
                # テストテンソルを解放
                del test_tensor
                torch.cuda.empty_cache()
                logger.info("テストテンソルを解放しました")
                
                # 最終メモリ状態
                allocated_final = torch.cuda.memory_allocated(current_device) / (1024**3)
                reserved_final = torch.cuda.memory_reserved(current_device) / (1024**3)
                logger.info(f"解放後メモリ使用量:")
                logger.info(f"  - 割当済み: {allocated_final:.4f} GB")
                logger.info(f"  - 予約済み: {reserved_final:.4f} GB")
                
                if abs(allocated_after - allocated_before) > 0.5:  # 500MB以上の差
                    logger.info("✅ GPUメモリテスト成功: GPUメモリを使用できました")
                else:
                    logger.warning("⚠️ GPUメモリテスト警告: メモリ割り当てが予想より少ないです")
                
            except Exception as e:
                logger.error(f"GPUメモリテストエラー: {e}")
                logger.error(traceback.format_exc())
        
        return cuda_available
    
    except ImportError:
        logger.error("PyTorchがインストールされていません")
        return False
    
    except Exception as e:
        logger.error(f"PyTorch GPU診断エラー: {e}")
        logger.error(traceback.format_exc())
        return False

def check_gpu_nvml():
    """NVMLを使用してGPUをチェック"""
    logger.info("\n=== NVML GPU診断 ===")
    
    try:
        import pynvml
        pynvml.nvmlInit()
        
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        logger.info(f"NVIDIAドライバーバージョン: {driver_version}")
        
        device_count = pynvml.nvmlDeviceGetCount()
        logger.info(f"GPUデバイス数: {device_count}")
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # デバイス名
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            # メモリ情報
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = memory_info.total / (1024**3)
            used_memory = memory_info.used / (1024**3)
            free_memory = memory_info.free / (1024**3)
            
            # 使用率
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util_rates.gpu
            memory_util = util_rates.memory
            
            # 情報表示
            logger.info(f"GPU {i}: {name}")
            logger.info(f"  - メモリ合計: {total_memory:.2f} GB")
            logger.info(f"  - メモリ使用: {used_memory:.2f} GB ({free_memory:.2f} GB空き)")
            logger.info(f"  - GPU使用率: {gpu_util}%")
            logger.info(f"  - メモリ使用率: {memory_util}%")
            
            # プロセス情報
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                if processes:
                    logger.info(f"  - 実行中GPUプロセス: {len(processes)}")
                    for p in processes:
                        try:
                            process_name = pynvml.nvmlSystemGetProcessName(p.pid)
                            if isinstance(process_name, bytes):
                                process_name = process_name.decode('utf-8')
                            memory_used = p.usedGpuMemory / (1024**3) if p.usedGpuMemory else 0
                            logger.info(f"    * PID {p.pid}: {process_name} ({memory_used:.2f} GB)")
                        except Exception:
                            pass
                else:
                    logger.info("  - 実行中GPUプロセスなし")
            except Exception as e:
                logger.warning(f"  - プロセス情報取得エラー: {e}")
        
        pynvml.nvmlShutdown()
        return True
    
    except ImportError:
        logger.error("pynvmlがインストールされていません")
        return False
    
    except Exception as e:
        logger.error(f"NVML GPU診断エラー: {e}")
        logger.error(traceback.format_exc())
        return False

def check_llama_cpp_gpu():
    """llama-cpp-pythonのGPUサポートをチェック"""
    logger.info("\n=== llama-cpp-python GPU診断 ===")
    
    try:
        import llama_cpp
        logger.info(f"llama-cpp-pythonバージョン: {getattr(llama_cpp, '__version__', '不明')}")
        
        # GPU対応チェック
        import inspect
        params = inspect.signature(llama_cpp.Llama.__init__).parameters
        has_gpu_param = 'n_gpu_layers' in params
        
        if has_gpu_param:
            logger.info("✅ llama-cpp-pythonはGPU対応のビルドです")
            
            # パラメータ一覧表示
            gpu_params = [p for p in params.keys() if 'gpu' in p.lower()]
            logger.info(f"GPU関連パラメータ: {', '.join(gpu_params)}")
            
            # モデルテスト前の環境変数設定
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            logger.info("CUDA_VISIBLE_DEVICES=0 に設定しました")
            
            # テスト用の小さなモデルパスの探索
            test_model_path = None
            model_dirs = [
                os.path.join(os.getcwd(), "models"),
                "models",
                "../models",
                "../../models"
            ]
            
            for model_dir in model_dirs:
                if os.path.exists(model_dir):
                    for filename in os.listdir(model_dir):
                        if filename.endswith('.gguf'):
                            test_model_path = os.path.join(model_dir, filename)
                            logger.info(f"テスト用モデル発見: {test_model_path}")
                            break
                    if test_model_path:
                        break
            
            if test_model_path:
                logger.info("小規模なモデルロードテストを試みます...")
                
                try:
                    # モデルロード（GPUあり）
                    start_time = time.time()
                    
                    model_gpu = llama_cpp.Llama(
                        model_path=test_model_path,
                        n_ctx=512,  # 小さなコンテキストサイズ
                        n_threads=4,
                        n_gpu_layers=-1,  # すべてのレイヤーをGPUにロード
                        n_batch=1,  # 小さなバッチサイズ
                        f16_kv=True,  # KVキャッシュにFP16を使用
                        verbose=True  # 詳細出力を有効化
                    )
                    
                    gpu_load_time = time.time() - start_time
                    logger.info(f"✅ GPUモードでモデルを正常にロードしました: {gpu_load_time:.2f}秒")
                    
                    # 簡単な推論テスト
                    logger.info("GPUモードで推論テストを実行します...")
                    start_time = time.time()
                    
                    if hasattr(model_gpu, "create_completion"):
                        result = model_gpu.create_completion(
                            prompt="Hello, world!",
                            max_tokens=1
                        )
                        gpu_infer_time = time.time() - start_time
                        logger.info(f"GPU推論時間: {gpu_infer_time:.2f}秒")
                    
                    # GPU使用状況確認
                    try:
                        import torch
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated(0) / (1024**3)
                            reserved = torch.cuda.memory_reserved(0) / (1024**3)
                            logger.info(f"推論後のGPUメモリ: 割当済 {allocated:.4f} GB / 予約済 {reserved:.4f} GB")
                            
                            if allocated > 0.01:  # 10MB以上使用
                                logger.info("✅ GPUメモリが使用されています - GPUが正しく利用されています")
                            else:
                                logger.warning("⚠️ GPUメモリ使用が検出されません - GPUが正しく使用されていない可能性")
                    except Exception as e:
                        logger.error(f"GPUメモリ確認エラー: {e}")
                    
                    # モデル解放
                    del model_gpu
                    
                    # CPU比較（オプション）
                    logger.info("\nCPUモードでの比較をしますか？ (y/n)")
                    response = input().strip().lower()
                    
                    if response == 'y':
                        logger.info("CPUモードで同じテストを実行します...")
                        
                        # モデルロード（CPUのみ）
                        start_time = time.time()
                        
                        model_cpu = llama_cpp.Llama(
                            model_path=test_model_path,
                            n_ctx=512,
                            n_threads=4,
                            n_gpu_layers=0,  # GPUレイヤーなし
                            verbose=True
                        )
                        
                        cpu_load_time = time.time() - start_time
                        logger.info(f"CPUモードでモデルをロードしました: {cpu_load_time:.2f}秒")
                        
                        # 比較
                        if gpu_load_time < cpu_load_time:
                            speedup = cpu_load_time / gpu_load_time
                            logger.info(f"✅ GPUモードは{speedup:.1f}倍高速です")
                        else:
                            logger.warning(f"⚠️ GPUモードの方が遅いです - GPUが正しく使用されていない可能性")
                
                except Exception as e:
                    logger.error(f"モデルテストエラー: {e}")
                    logger.error(traceback.format_exc())
            else:
                logger.warning("テスト用モデルが見つかりませんでした")
            
            return True
        else:
            logger.error("❌ llama-cpp-pythonはCPU専用ビルドです")
            logger.info("GPU対応版をインストールするには次のコマンドを実行してください:")
            logger.info("python scripts/reinstall_llama_cpp.py")
            return False
    
    except ImportError:
        logger.error("llama-cpp-pythonがインストールされていません")
        return False
    
    except Exception as e:
        logger.error(f"llama-cpp-python診断エラー: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """メイン実行関数"""
    logger.info("===== Jarviee GPU詳細診断ツール =====")
    
    # システム情報
    system_info = platform.uname()
    logger.info(f"OS: {system_info.system} {system_info.release}")
    logger.info(f"Python: {platform.python_version()}")
    
    # 環境変数確認
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "設定なし")
    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    # 各種診断の実行
    torch_result = check_gpu_torch()
    nvml_result = check_gpu_nvml()
    llama_cpp_result = check_llama_cpp_gpu()
    
    # 総合診断
    logger.info("\n===== 総合診断結果 =====")
    
    if torch_result:
        logger.info("✅ PyTorch GPU診断: 正常")
    else:
        logger.error("❌ PyTorch GPU診断: 異常")
    
    if nvml_result:
        logger.info("✅ NVML GPU診断: 正常")
    else:
        logger.error("❌ NVML GPU診断: 異常")
    
    if llama_cpp_result:
        logger.info("✅ llama-cpp-python GPU診断: 正常")
    else:
        logger.error("❌ llama-cpp-python GPU診断: 異常")
    
    # 問題がある場合の解決策
    if not (torch_result and nvml_result and llama_cpp_result):
        logger.info("\n推奨される対応策:")
        
        if not torch_result:
            logger.info("1. CUDAとPyTorchの再インストール:")
            logger.info("   pip uninstall torch")
            logger.info("   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128")
        
        if not llama_cpp_result:
            logger.info("2. llama-cpp-pythonの再インストール:")
            logger.info("   python scripts/reinstall_llama_cpp.py")
        
        logger.info("3. GPUドライバーの更新を検討してください")
    else:
        logger.info("\n✅ すべての診断が正常に完了しました！")
        logger.info("GPUが正しく認識され、使用可能な状態です。")
        logger.info("Jarvieeのパフォーマンスが改善するはずです。")
    
    # ログファイル情報
    logger.info(f"\n診断ログは次のファイルに保存されました: {log_filename}")

if __name__ == "__main__":
    main()
