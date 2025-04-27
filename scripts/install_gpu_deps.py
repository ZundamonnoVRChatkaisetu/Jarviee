#!/usr/bin/env python
"""
Jarviee GPU依存ライブラリインストーラー

GPUをより効果的に利用するための依存パッケージをインストールします。
"""
import sys
import os
import subprocess
import logging
import platform

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def check_and_install(package_name, install_name=None):
    """
    パッケージのインストール状態を確認し、必要に応じてインストール
    
    Args:
        package_name: インポート時のパッケージ名
        install_name: pipでのインストール名（指定がなければpackage_nameを使用）
    
    Returns:
        bool: インストールが成功したかどうか
    """
    install_name = install_name or package_name
    
    # パッケージのインストール状態を確認
    try:
        __import__(package_name)
        logger.info(f"[OK] {package_name} は既にインストールされています")
        return True
    except ImportError:
        logger.info(f"[WARNING] {package_name} はインストールされていません。インストールを試みます...")
        
    # パッケージのインストール
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", install_name],
            check=True,
            capture_output=True,
            text=True
        )
        
        # インストールの確認
        try:
            __import__(package_name)
            logger.info(f"[OK] {package_name} のインストールに成功しました")
            return True
        except ImportError:
            logger.error(f"[ERROR] {package_name} のインストールに失敗しました")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] {package_name} のインストール中にエラーが発生しました: {e}")
        logger.error(f"エラー出力: {e.stderr}")
        return False

def main():
    """メイン関数"""
    logger.info("=== Jarviee GPU依存ライブラリインストーラー ===")
    
    # システム情報の表示
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
    
    # インストールするパッケージリスト
    packages = [
        ("pynvml", "pynvml"),               # NVIDIA GPUモニタリング
        ("psutil", "psutil"),               # システムリソースモニタリング
        ("torch", "torch"),                 # PyTorch（GPU検出用）
        # nvidia-ml-pyはpynvmlで代替するため削除
    ]
    
    # 各パッケージのインストール
    success_count = 0
    for package_name, install_name in packages:
        if check_and_install(package_name, install_name):
            success_count += 1
    
    # 結果の表示
    logger.info(f"{success_count}/{len(packages)} パッケージのインストールに成功しました")
    
    # 追加のチェック - GPU状態の確認
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"[SUCCESS] PyTorchがGPUを検出しました！ 検出されたGPU数: {device_count}")
            
            # GPUデバイス情報
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                device_cap = torch.cuda.get_device_capability(i)
                logger.info(f"GPU {i}: {device_name} (CUDA Capability: {device_cap[0]}.{device_cap[1]})")
        else:
            logger.warning("[WARNING] PyTorchがGPUを検出できませんでした")
    except Exception as e:
        logger.error(f"[ERROR] GPU検出中にエラーが発生しました: {e}")
    
    # NVMLの動作確認
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        logger.info(f"[SUCCESS] NVMLが正常に動作しています！ 検出されたGPU数: {device_count}")
        
        # GPUデバイス情報
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mem = memory.total / (1024**3)  # GB単位
            logger.info(f"GPU {i}: {name} (メモリ: {total_mem:.2f} GB)")
        
        pynvml.nvmlShutdown()
    except Exception as e:
        logger.error(f"[ERROR] NVML検出中にエラーが発生しました: {e}")
    
    logger.info("インストールプロセスが完了しました")

if __name__ == "__main__":
    main()
