#!/usr/bin/env python
"""
llama-cpp-python再インストールスクリプト

CUDAバージョンに完全に対応したllama-cpp-pythonをインストールします。
"""

import os
import sys
import subprocess
import logging
import platform
import importlib.util

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def get_cuda_version():
    """
    システムのCUDAバージョンを正確に取得
    
    Returns:
        str: CUDAバージョン（正規化済み）
    """
    cuda_version = None
    
    # 方法0: 環境変数から直接取得（最も信頼性が高い）
    cuda_path = os.environ.get('CUDA_PATH', '')
    if cuda_path and os.path.exists(cuda_path):
        cuda_dir = os.path.basename(cuda_path)
        if cuda_dir.startswith('v'):
            version = cuda_dir[1:]  # 'v12.8' -> '12.8'
            logger.info(f"環境変数から検出されたCUDAバージョン: {version}")
            if '.' in version:
                major, minor = version.split('.')[:2]
                return f"{major}{minor}"
    
    # 方法1: nvcc
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            check=True,
            capture_output=True,
            text=True
        )
        
        for line in result.stdout.split('\n'):
            if "release" in line:
                parts = line.split("release")
                if len(parts) >= 2:
                    version_parts = parts[1].strip().split(",")[0].strip()
                    logger.info(f"nvccから検出されたCUDAバージョン: {version_parts}")
                    cuda_version = version_parts
                    break
    except Exception as e:
        logger.warning(f"nvccからのCUDAバージョン取得に失敗: {e}")
    
    # 方法2: nvidia-smi
    if not cuda_version:
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                check=True,
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.split('\n'):
                if "CUDA Version:" in line:
                    cuda_version = line.split("CUDA Version:")[1].strip().split()[0]
                    logger.info(f"nvidia-smiから検出されたCUDAバージョン: {cuda_version}")
                    break
        except Exception as e:
            logger.warning(f"nvidia-smiからのCUDAバージョン取得に失敗: {e}")
    
    # 方法3: PyTorch
    if not cuda_version:
        try:
            if importlib.util.find_spec("torch") is not None:
                import torch
                if torch.cuda.is_available():
                    cuda_version = torch.version.cuda
                    logger.info(f"PyTorchから検出されたCUDAバージョン: {cuda_version}")
        except Exception as e:
            logger.warning(f"PyTorchからのCUDAバージョン取得に失敗: {e}")
    
    if cuda_version:
        # バージョン形式の正規化
        if '.' in cuda_version:
            major, minor = cuda_version.split('.')[:2]
            normalized = f"{major}{minor}"
            return normalized
        return cuda_version
    
    # 方法4: デフォルト値の設定 (通常は推奨されないが、他の方法がすべて失敗した場合)
    logger.warning("[WARNING] CUDAバージョンを検出できませんでしたが、CUDA 12.8が存在すると仮定します")
    return "128"  # CUDA 12.8をデフォルト値として設定

def uninstall_llama_cpp():
    """
    既存のllama-cpp-pythonをアンインストール
    """
    logger.info("既存のllama-cpp-pythonをアンインストールしています...")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"],
            check=True
        )
        logger.info("[OK] llama-cpp-pythonのアンインストールに成功しました")
        return True
    except Exception as e:
        logger.error(f"[ERROR] llama-cpp-pythonのアンインストールに失敗しました: {e}")
        return False

def install_llama_cpp(cuda_version=None):
    """
    CUDA対応のllama-cpp-pythonをインストール
    
    Args:
        cuda_version: CUDAバージョン (正規化済み)
    """
    if not cuda_version:
        cuda_version = get_cuda_version()
    
    # サポートされているCUDAバージョンチェック
    supported_versions = ["110", "111", "112", "113", "116", "117", "118", "120", "121", "122", "123", "128"]
    
    # 完全一致または最も近いバージョンを選択
    if cuda_version not in supported_versions:
        closest_versions = [v for v in supported_versions if v.startswith(cuda_version[0])]
        if closest_versions:
            closest_version = closest_versions[-1]  # 最新のマイナーバージョン
            logger.warning(f"[WARNING] 完全一致するCUDAバージョンがないため、最も近い {closest_version} を使用します")
            cuda_version = closest_version
        else:
            logger.error(f"[ERROR] 互換性のあるCUDAバージョンが見つかりません: {cuda_version}")
            # CUDA 12.8をデフォルトとして使用
            cuda_version = "128"
            logger.warning(f"[WARNING] デフォルトとしてCUDA 12.8を使用します")
    
    # インストールURLの設定
    logger.info(f"CUDA {cuda_version} 用のllama-cpp-pythonをインストールします")
    
    # AVX2とAVX512をサポート
    install_url = f"https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu{cuda_version}"
    
    # インストールコマンド
    install_cmd = [
        sys.executable, "-m", "pip", "install", 
        "llama-cpp-python", "--no-cache-dir", "--force-reinstall",
        f"--extra-index-url={install_url}"
    ]
    
    # 追加のpipオプション
    install_cmd.extend(["--verbose", "--upgrade"])
    
    # インストール実行
    logger.info(f"コマンド実行: {' '.join(install_cmd)}")
    
    try:
        result = subprocess.run(
            install_cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # 詳細なログ出力
        for line in result.stdout.split('\n'):
            if "Successfully" in line or "Error" in line:
                logger.info(line)
        
        # バージョン確認
        try:
            # llama_cppモジュールをリロード（既にロードされている場合）
            if "llama_cpp" in sys.modules:
                import llama_cpp
                importlib.reload(llama_cpp)
            else:
                import llama_cpp
            
            # バージョン情報表示
            version = getattr(llama_cpp, "__version__", "不明")
            logger.info(f"インストールされたllama-cpp-pythonのバージョン: {version}")
            
            # GPU対応確認
            import inspect
            has_gpu_support = 'n_gpu_layers' in inspect.signature(llama_cpp.Llama.__init__).parameters
            
            if has_gpu_support:
                logger.info("[SUCCESS] GPU対応のllama-cpp-pythonのインストールに成功しました")
                return True
            else:
                logger.error("[ERROR] インストールされたllama-cpp-pythonはGPU非対応です")
                return False
        except ImportError:
            logger.error("[ERROR] llama-cpp-pythonのインポートに失敗しました - インストールに問題がある可能性があります")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] インストール中にエラーが発生しました: {e}")
        logger.error(f"エラー出力: {e.stderr}")
        return False

def main():
    """メイン実行関数"""
    logger.info("=== llama-cpp-python再インストールツール ===")
    
    # システム情報
    system_info = platform.uname()
    logger.info(f"OS: {system_info.system} {system_info.release} ({system_info.version})")
    logger.info(f"Python: {platform.python_version()}")
    
    # CUDA環境変数の確認
    cuda_vars = {k: v for k, v in os.environ.items() if 'CUDA' in k}
    if cuda_vars:
        logger.info("CUDA環境変数:")
        for k, v in cuda_vars.items():
            logger.info(f"  {k}: {v}")
    else:
        logger.warning("[WARNING] CUDA環境変数が設定されていません")
    
    # 現在のCUDAバージョンを取得
    cuda_version = get_cuda_version()
    if cuda_version:
        logger.info(f"検出されたCUDAバージョン (正規化済み): {cuda_version}")
    
    # 既存のllama-cpp-pythonをアンインストール
    if not uninstall_llama_cpp():
        logger.warning("[WARNING] アンインストールに問題がありましたが、続行します")
    
    # llama-cpp-pythonをインストール
    if install_llama_cpp(cuda_version):
        logger.info("[SUCCESS] llama-cpp-pythonの再インストールが完了しました")
    else:
        logger.error("[ERROR] llama-cpp-pythonの再インストールに失敗しました")
        sys.exit(1)
    
    logger.info("GPUサポートテストを行ってください: python scripts/diagnose_jarviee.py")

if __name__ == "__main__":
    main()
