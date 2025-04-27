#!/usr/bin/env python
"""
GPU強制メモリ割り当てモジュール

GPUメモリを明示的に確保し、LLMモデルがGPUを確実に使用できるようにします。
"""
import os
import sys
import logging

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def force_gpu_memory_allocation():
    """GPUメモリを明示的に確保"""
    try:
        import torch
        
        # CUDAが利用可能か確認
        if not torch.cuda.is_available():
            logger.warning("[WARNING] CUDA is not available.")
            return False
            
        # 現在のGPUメモリ状態を確認
        logger.info(f"GPU初期状態: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB / {torch.cuda.memory_reserved(0)/(1024**2):.2f} MB")
        
        # 小さなテンソルの確保でGPUをアクティブ化
        dummy = torch.ones(1, device="cuda")
        logger.info(f"ダミーテンソル作成後: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB")
        
        # テンソル演算でキャッシュをウォームアップ
        for _ in range(10):
            dummy = dummy + dummy
            
        # GPUキャッシュを明示的にクリア
        torch.cuda.empty_cache()
        
        # テンソル割り当てを維持
        _ = torch.ones(100, 100, device="cuda")
        logger.info(f"GPU割り当て後: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB / {torch.cuda.memory_reserved(0)/(1024**2):.2f} MB")
        
        return True
    except Exception as e:
        logger.error(f"[ERROR] GPU memory allocation failed: {e}")
        return False

def check_cuda_setup():
    """CUDA設定の検証"""
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            
            # CUDA環境変数の表示
            cuda_vars = {k: v for k, v in os.environ.items() if 'CUDA' in k}
            if cuda_vars:
                logger.info("CUDA environment variables:")
                for k, v in cuda_vars.items():
                    logger.info(f"  {k}: {v}")
            
            return True
        else:
            logger.warning("[WARNING] CUDA is not available in PyTorch.")
            return False
    except Exception as e:
        logger.error(f"[ERROR] CUDA setup check failed: {e}")
        return False

def main():
    """メイン関数"""
    logger.info("=== GPU強制メモリ割り当てツール ===")
    
    # CUDA設定を確認
    check_cuda_setup()
    
    # GPUメモリを割り当て
    if force_gpu_memory_allocation():
        logger.info("[SUCCESS] GPU memory allocation successful.")
    else:
        logger.error("[ERROR] GPU memory allocation failed.")

if __name__ == "__main__":
    main()
