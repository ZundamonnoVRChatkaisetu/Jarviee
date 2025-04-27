#!/usr/bin/env python
"""
GPU対応llama-cpp-pythonインストーラー

このスクリプトは、GPUに対応したllama-cpp-pythonパッケージを正しくインストールします。
"""
import os
import sys
import platform
import subprocess
import argparse
import logging

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def get_cuda_version():
    """
    システムのCUDAバージョンを取得
    
    Returns:
        CUDAバージョン (例: "11.7")、見つからない場合は None
    """
    try:
        # nvccコマンドを試す
        result = subprocess.run(
            ["nvcc", "--version"],
            check=True,
            capture_output=True,
            text=True
        )
        
        # バージョン情報を抽出
        for line in result.stdout.split('\n'):
            if "release" in line and "V" in line:
                # 例: "Cuda compilation tools, release 11.7, V11.7.99"
                parts = line.split("release")
                if len(parts) >= 2:
                    version_parts = parts[1].strip().split(",")[0].strip()
                    return version_parts
        
        logger.warning("nvccからCUDAバージョンを取得できませんでした")
        return None
    except Exception:
        # nvidia-smiを試す
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                check=True,
                capture_output=True,
                text=True
            )
            
            # 例: "| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |"
            for line in result.stdout.split('\n'):
                if "CUDA Version:" in line:
                    cuda_version = line.split("CUDA Version:")[1].strip().split()[0]
                    return cuda_version
            
            logger.warning("nvidia-smiからCUDAバージョンを取得できませんでした")
            return None
        except Exception as e:
            logger.error(f"CUDAバージョンの検出中にエラーが発生しました: {e}")
            return None

def check_torch_cuda():
    """
    PyTorchがCUDAと互換性があるかチェック
    
    Returns:
        (bool, str): (CUDAが利用可能か, GPUデバイス情報)
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            devices = []
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                devices.append(f"{name} ({mem:.1f} GB)")
            
            return cuda_available, ", ".join(devices)
        else:
            return False, "利用可能なGPUが見つかりません"
    except ImportError:
        logger.warning("PyTorchがインストールされていません")
        return False, "PyTorchがインストールされていません"
    except Exception as e:
        logger.error(f"PyTorch CUDA検出エラー: {e}")
        return False, f"エラー: {str(e)}"

def install_llama_cpp(cuda_version=None, force=False):
    """
    llama-cpp-pythonパッケージをインストール
    
    Args:
        cuda_version: CUDAバージョン、指定がない場合は自動検出
        force: 既存のインストールに関わらず強制的に再インストールするかどうか
    
    Returns:
        bool: インストールが成功したかどうか
    """
    try:
        # 現在のインストール状態を確認
        try:
            import llama_cpp
            if not force:
                logger.info("llama-cpp-pythonは既にインストールされています")
                
                # GPU対応を確認
                import inspect
                if 'n_gpu_layers' in inspect.signature(llama_cpp.Llama.__init__).parameters:
                    logger.info("インストール済みのllama-cpp-pythonはGPU対応です")
                    return True
                else:
                    logger.warning("インストール済みのllama-cpp-pythonはGPU非対応です。再インストールを実行します")
        except ImportError:
            logger.info("llama-cpp-pythonはインストールされていません。新規インストールを実行します")
        
        # CUDAバージョンの取得・検証
        if cuda_version is None:
            cuda_version = get_cuda_version()
        
        if cuda_version:
            # バージョン番号を正規化（例: "11.7.0" → "117"）
            normalized_version = cuda_version.split(".")[0] + cuda_version.split(".")[1]
            normalized_version = normalized_version.replace(".", "")
            
            # サポートされているバージョンの確認
            supported_versions = ["110", "112", "113", "116", "117", "118", "120", "121", "122", "123", "128"]
            closest_version = None
            
            # 完全一致を確認
            if normalized_version in supported_versions:
                closest_version = normalized_version
            else:
                # メジャーバージョンで部分一致
                major_version = normalized_version[:3]  # 例: "117" → "11"+"7"="117"
                matching = [v for v in supported_versions if v.startswith(major_version[0:2])]
                if matching:
                    # 最も近いバージョンを選択
                    closest_version = matching[-1]  # 通常、最新のマイナーバージョン
            
            if closest_version:
                logger.info(f"検出されたCUDAバージョン: {cuda_version}")
                logger.info(f"使用する正規化バージョン: cu{closest_version}")
                
                # インストールコマンドの構築
                cuda_tag = f"cu{closest_version}"
                install_cmd = [
                    sys.executable, "-m", "pip", "install", 
                    "llama-cpp-python", "--force-reinstall", "--no-cache-dir",
                    f"--extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/{cuda_tag}"
                ]
                
                # インストールの実行
                logger.info(f"コマンド実行: {' '.join(install_cmd)}")
                subprocess.run(install_cmd, check=True)
                
                # インストールの確認
                try:
                    import llama_cpp
                    import importlib
                    importlib.reload(llama_cpp)
                    
                    import inspect
                    if 'n_gpu_layers' in inspect.signature(llama_cpp.Llama.__init__).parameters:
                        logger.info("GPU対応のllama-cpp-pythonのインストールに成功しました")
                        return True
                    else:
                        logger.error("インストールされたパッケージがGPUをサポートしていません")
                        return False
                except Exception as e:
                    logger.error(f"インストール後の確認に失敗しました: {e}")
                    return False
            else:
                logger.warning(f"サポートされていないCUDAバージョン: {cuda_version}")
                logger.info("安全なデフォルトとしてCUDA 11.7用のパッケージを使用します")
                
                # CUDA 11.7用にインストール
                install_cmd = [
                    sys.executable, "-m", "pip", "install", 
                    "llama-cpp-python", "--force-reinstall", "--no-cache-dir",
                    "--extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117"
                ]
                
                logger.info(f"コマンド実行: {' '.join(install_cmd)}")
                subprocess.run(install_cmd, check=True)
                
                # インストールの確認
                try:
                    import llama_cpp
                    import importlib
                    importlib.reload(llama_cpp)
                    
                    import inspect
                    if 'n_gpu_layers' in inspect.signature(llama_cpp.Llama.__init__).parameters:
                        logger.info("GPU対応のllama-cpp-pythonのインストールに成功しました")
                        return True
                    else:
                        logger.error("インストールされたパッケージがGPUをサポートしていません")
                        return False
                except Exception as e:
                    logger.error(f"インストール後の確認に失敗しました: {e}")
                    return False
        else:
            logger.error("CUDAバージョンを検出できません")
            return False
        
    except Exception as e:
        logger.error(f"インストール中にエラーが発生しました: {e}")
        return False

def test_gpu_support():
    """
    GPUサポートをテスト
    
    Returns:
        bool: GPUサポートが利用可能かどうか
    """
    try:
        import llama_cpp
        
        # GPU対応チェック
        import inspect
        if 'n_gpu_layers' not in inspect.signature(llama_cpp.Llama.__init__).parameters:
            logger.error("llama-cpp-pythonはGPU対応していません")
            return False
        
        # PyTorchでCUDA利用可能性をチェック
        cuda_available, device_info = check_torch_cuda()
        if not cuda_available:
            logger.warning("PyTorchがCUDAを検出できません")
            logger.warning("llama-cpp-pythonはGPU対応していますが、GPUが利用できない可能性があります")
            return False
        
        logger.info(f"GPU情報: {device_info}")
        return True
    except ImportError:
        logger.error("llama-cpp-pythonがインストールされていません")
        return False
    except Exception as e:
        logger.error(f"テスト中にエラーが発生しました: {e}")
        return False

def setup_environment():
    """
    Jarviee環境をGPU対応に設定
    
    Returns:
        bool: 設定が成功したかどうか
    """
    try:
        # プロジェクトのルートディレクトリを特定
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # .envファイルへGPU設定を追加
        env_path = os.path.join(project_root, ".env")
        env_lines = []
        
        # 既存の設定を読み込み
        gpu_settings_exist = False
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("USE_GPU=") or line.strip().startswith("GPU_LAYERS="):
                        gpu_settings_exist = True
                        # 既存の設定を更新
                        if line.strip().startswith("USE_GPU="):
                            env_lines.append("USE_GPU=true\n")
                        elif line.strip().startswith("GPU_LAYERS="):
                            env_lines.append("GPU_LAYERS=-1\n")
                    else:
                        env_lines.append(line)
        
        # GPU設定がなければ追加
        if not gpu_settings_exist:
            env_lines.append("\n# GPU設定\n")
            env_lines.append("USE_GPU=true\n")
            env_lines.append("GPU_LAYERS=-1\n")
        
        # 設定を書き込み
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(env_lines)
        
        logger.info(f"環境設定を更新しました: {env_path}")
        return True
    except Exception as e:
        logger.error(f"環境設定の更新中にエラーが発生しました: {e}")
        return False

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="GPU対応llama-cpp-pythonインストーラー")
    parser.add_argument("--cuda-version", help="CUDAバージョンを指定（例: 11.7）")
    parser.add_argument("--force", action="store_true", help="既存のインストールに関わらず強制的に再インストール")
    parser.add_argument("--no-env-setup", action="store_true", help="環境設定を更新しない")
    args = parser.parse_args()
    
    logger.info("=== GPU対応llama-cpp-pythonインストーラー ===")
    
    # システム情報
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    
    # CUDA/GPU情報
    cuda_version = args.cuda_version or get_cuda_version()
    if cuda_version:
        logger.info(f"検出されたCUDAバージョン: {cuda_version}")
    else:
        logger.warning("CUDAバージョンを検出できませんでした")
    
    cuda_available, device_info = check_torch_cuda()
    if cuda_available:
        logger.info(f"利用可能なGPU: {device_info}")
    else:
        logger.warning(f"GPU利用不可: {device_info}")
    
    # llama-cpp-pythonをインストール
    success = install_llama_cpp(cuda_version, args.force)
    
    if success:
        # GPUサポートをテスト
        if test_gpu_support():
            logger.info("GPUサポートテスト成功")
        else:
            logger.warning("GPUサポートテスト失敗")
        
        # 環境設定
        if not args.no_env_setup:
            if setup_environment():
                logger.info("環境設定を更新しました")
            else:
                logger.warning("環境設定の更新に失敗しました")
        
        logger.info("インストールが完了しました")
    else:
        logger.error("インストールに失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    main()
