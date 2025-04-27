"""
Gemmaモデルプロバイダー

ローカルのGemmaモデルを使用したLLMプロバイダー実装
"""

import os
import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Union
import subprocess

# ベースプロバイダーのインポート
from .base import LLMProvider

# llama-cpp-pythonの依存関係
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logging.warning("llama-cpp-python is not installed. GemmaProvider will not be fully functional.")


class GemmaProvider(LLMProvider):
    """Gemmaモデルを利用したLLMプロバイダー"""
    
    def __init__(self, model_path: str, config: Dict[str, Any] = None):
        """
        Gemmaプロバイダーを初期化

        Args:
            model_path: モデルファイルのパス
            config: 設定パラメータ
        """
        self.model_path = model_path
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # デフォルト設定
        self.n_ctx = self.config.get("context_window", 8192)  # コンテキストウィンドウサイズ
        self.n_threads = self.config.get("threads", os.cpu_count() or 4)  # スレッド数
        self.temp = self.config.get("temperature", 0.7)  # 温度
        self.max_tokens = self.config.get("max_tokens", 2048)  # 最大トークン数
        
        # 環境変数から設定を優先的に読み込む
        self.use_gpu = os.environ.get("USE_GPU", "").lower() == "true" if "USE_GPU" in os.environ else self.config.get("use_gpu", True)
        self.n_gpu_layers = int(os.environ.get("GPU_LAYERS", "-1")) if "GPU_LAYERS" in os.environ else self.config.get("n_gpu_layers", -1)
        
        # CPU最適化設定
        self.n_batch = self.config.get("n_batch", 512)
        if "gpu_settings" in self.config and "n_batch" in self.config["gpu_settings"]:
            self.n_batch = self.config["gpu_settings"]["n_batch"]
        
        # GPU使用状態のフラグ
        self.is_using_gpu = False
        self.gpu_info = "Unknown"
        
        # 設定をコンソールにロギング
        self.logger.info(f"GPU設定: use_gpu={self.use_gpu}, n_gpu_layers={self.n_gpu_layers}")
        print(f"[DEBUG] GPU設定: use_gpu={self.use_gpu}, n_gpu_layers={self.n_gpu_layers}")
        
        # GPUサポートを確認
        self._check_gpu_support()
        
        # モデルのロード
        self._model = None
        self._load_model()
        
    def _check_gpu_support(self) -> bool:
        """
        GPU対応版llama-cpp-pythonがインストールされているか確認
        
        Returns:
            GPU対応かどうか
        """
        try:
            # GPU関連の引数があるかチェック
            import inspect
            llama_init_args = inspect.signature(Llama.__init__).parameters
            has_gpu_support = 'n_gpu_layers' in llama_init_args
            
            if has_gpu_support:
                print("[DEBUG] llama-cpp-pythonはGPU対応版です")
                
                # CUDAが利用可能かも確認
                try:
                    import torch
                    has_cuda = torch.cuda.is_available()
                    if has_cuda:
                        print(f"[DEBUG] PyTorch CUDA利用可能: {torch.cuda.get_device_name(0)}")
                    else:
                        print("[DEBUG] PyTorch CUDAは使用できませんが、llama-cpp-pythonのGPU対応は確認できています")
                        print("[DEBUG] llama-cpp-pythonはCUDAなしでもGPUパラメータを使用可能なため、GPU設定を有効化します")
                except Exception as e:
                    print(f"[DEBUG] PyTorch CUDA確認エラー: {e}")
            else:
                print("[DEBUG] llama-cpp-pythonはCPU版のみです。GPU機能は使用できません")
                
            return has_gpu_support
        except Exception as e:
            print(f"[DEBUG] GPUサポート確認エラー: {e}")
            return False
    
    def _load_model(self) -> None:
        """モデルをロード"""
        if not LLAMA_CPP_AVAILABLE:
            self.logger.error("llama-cpp-python is not installed. Cannot load Gemma model.")
            print("[ERROR] llama-cpp-pythonがインストールされていません。モデルをロードできません。")
            return
            
        if not os.path.exists(self.model_path):
            # 相対パスの場合に対応
            alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), self.model_path)
            if os.path.exists(alt_path):
                self.model_path = alt_path
                print(f"[DEBUG] 相対パスからモデルファイルを見つけました: {self.model_path}")
            else:
                self.logger.error(f"Model file not found: {self.model_path}")
                print(f"[ERROR] モデルファイルが見つかりません: {self.model_path}")
                print(f"[DEBUG] 試行パス: {self.model_path}, {alt_path}")
                return
            
        try:
            self.logger.info(f"Loading Gemma model from {self.model_path}...")
            print(f"[INFO] Gemmaモデルをロード中: {self.model_path}")
            start_time = time.time()
            
            # GPU対応チェック
            gpu_supported = self._check_gpu_support()
            
            # モデル設定パラメータ
            model_params = {
                "model_path": self.model_path,
                "n_ctx": self.n_ctx,
                "n_threads": self.n_threads,
                "verbose": self.config.get("verbose", True),  # デバッグのためにverboseを有効化
                "n_batch": self.n_batch,  # バッチサイズ設定
            }
            
            # GPUサポートが確認できた場合のみパラメータを追加
            if gpu_supported and self.use_gpu:
                print(f"[DEBUG] GPU対応確認済み - n_gpu_layers={self.n_gpu_layers}に設定します")
                model_params["n_gpu_layers"] = self.n_gpu_layers
                
                # GPU最適化パラメータを設定
                if "gpu_settings" in self.config:
                    for key, value in self.config["gpu_settings"].items():
                        if key not in model_params and key != "n_gpu_layers":  # 既に設定済みのものを除く
                            model_params[key] = value
                            print(f"[DEBUG] GPU最適化パラメータ設定: {key}={value}")
                    
                # 追加のGPU最適化設定 - より明示的に設定
                model_params["n_gpu_layers"] = -1  # すべてのレイヤーをGPUにロード
                model_params["f16_kv"] = True  # KVキャッシュにFP16を使用
                model_params["use_mmap"] = False  # GPUモードではmmapをオフにする方が安定
                model_params["use_mlock"] = True  # メモリをロック（スワップ防止）
                
                # 明示的なCUDA初期化
                try:
                    import torch
                    if not torch.cuda.is_initialized():
                        torch.cuda.init()
                    print(f"[DEBUG] CUDA初期化状態: {torch.cuda.is_initialized()}")
                    
                    # CUDAデバイスの明示的選択
                    current_device = torch.cuda.current_device()
                    print(f"[DEBUG] 現在のCUDAデバイス: {current_device}")
                    
                    # GPUをウォームアップ
                    dummy_tensor = torch.zeros(1, device=f'cuda:{current_device}')
                    torch.cuda.synchronize()
                    print(f"[DEBUG] GPUウォームアップ完了")
                except Exception as e:
                    print(f"[DEBUG] CUDA初期化エラー: {e}")
                
                # GPUが高速に動作するようにローカルGPUメモリを多く使用
                model_params["main_gpu"] = 0  # プライマリGPU
                model_params["tensor_split"] = [1.0]  # すべてのテンソルをGPU 0に配置
                
                # CUDA環境設定を強制的に有効化
                if "CUDA_VISIBLE_DEVICES" not in os.environ:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0を明示的に使用
                
                # PyTorchからGPU情報を取得（可能な場合）
                try:
                    import torch
                    if torch.cuda.is_available():
                        device_name = torch.cuda.get_device_name(0)
                        device_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        self.gpu_info = f"{device_name} ({device_mem:.1f} GB)"
                        print(f"[DEBUG] GPU検出: {self.gpu_info}")
                except Exception as e:
                    print(f"[DEBUG] PyTorch GPU情報取得エラー: {e}")
                
                self.is_using_gpu = True
                
                # CUDA環境変数をデバッグ出力
                cuda_env_vars = {k: v for k, v in os.environ.items() if "CUDA" in k}
                print(f"[DEBUG] CUDA環境変数: {cuda_env_vars}")
            else:
                # CPU最適化
                print("[DEBUG] CPUモードで最適化設定を使用")
                model_params["n_batch"] = 512  # CPU用バッチサイズ
                
                # スレッド設定
                model_params["n_threads"] = os.cpu_count() or 4  # 全CPUコアを使用
                model_params["n_threads_batch"] = os.cpu_count() or 4  # バッチ処理用スレッド
                
                self.is_using_gpu = False
            
            # モデルパラメータ表示
            print(f"[DEBUG] モデルパラメータ: {model_params}")
            
            # モデルロード
            try:
                self._model = Llama(**model_params)
                
                # GPU使用確認
                if self.is_using_gpu:
                    print(f"[INFO] GPUモードでGemmaモデルをロードしました")
                    
                    # レイヤー数を確認
                    if hasattr(self._model, "n_gpu_layers"):
                        print(f"[DEBUG] GPUレイヤー数: {self._model.n_gpu_layers}")
                    
                else:
                    print("[INFO] CPUモードでGemmaモデルをロードしました")
                
                # モデルのロード時間記録
                load_time = time.time() - start_time
                self.logger.info(f"Gemma model loaded in {load_time:.2f} seconds")
                print(f"[INFO] Gemmaモデルを {load_time:.2f} 秒でロードしました")
                
                # GPU使用状況を確認
                if self.is_using_gpu:
                    try:
                        # NVML経由でGPU情報を取得
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            mem_used = mem_info.used / (1024**3)
                            mem_total = mem_info.total / (1024**3)
                            print(f"[DEBUG] NVML GPU状態: {gpu_name}, メモリ使用: {mem_used:.2f}/{mem_total:.2f} GB")
                            pynvml.nvmlShutdown()
                        except Exception as e:
                            print(f"[DEBUG] NVML GPU情報取得エラー: {e}")
                            
                        # PyTorchでもGPU状態を確認
                        try:
                            import torch
                            if torch.cuda.is_available():
                                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                                print(f"[DEBUG] PyTorch GPU状態: 割当済 {allocated:.2f} GB / 予約済 {reserved:.2f} GB")
                        except Exception as e:
                            print(f"[DEBUG] PyTorch GPU情報取得エラー: {e}")
                    except Exception as e:
                        print(f"[DEBUG] GPU情報取得エラー: {e}")
                
                # 簡単なテストプロンプトでパフォーマンスを確認
                try:
                    print("[DEBUG] 簡単なプロンプトで動作テスト中...")
                    test_start = time.time()
                    _ = self._model.create_completion(prompt="hello", max_tokens=1)
                    test_time = time.time() - test_start
                    print(f"[DEBUG] テスト完了: {test_time:.2f} 秒")
                    
                    # GPUメモリ使用の変化を再確認
                    if self.is_using_gpu:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                                print(f"[DEBUG] テスト後のGPUメモリ: 割当済 {allocated:.2f} GB / 予約済 {reserved:.2f} GB")
                                
                                # メモリ使用があればGPUが使われている可能性が高い
                                if allocated > 0.01 or reserved > 0.01:  # 10MB以上使用
                                    print("[DEBUG] ✅ GPUメモリが確保されています - GPUが使用されている可能性が高い")
                                else:
                                    print("[DEBUG] ⚠️ GPUメモリ使用がほとんど見られません - GPUが正しく使用されていない可能性")
                        except Exception as e:
                            print(f"[DEBUG] PyTorch GPU情報取得エラー: {e}")
                except Exception as e:
                    print(f"[DEBUG] テストプロンプトエラー: {e}")
                
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                print(f"[ERROR] モデルロードエラー: {e}")
                print(f"[DEBUG] 詳細エラー情報:\n{traceback.format_exc()}")
                
                # CPU専用モードで再試行
                print("[DEBUG] CPUモードで再試行します...")
                
                # GPUパラメータを削除
                if "n_gpu_layers" in model_params:
                    del model_params["n_gpu_layers"]
                if "main_gpu" in model_params:
                    del model_params["main_gpu"]
                if "tensor_split" in model_params:
                    del model_params["tensor_split"]
                
                # その他のGPU関連パラメータを削除
                gpu_keys = [k for k in model_params.keys() if "gpu" in k.lower()]
                for key in gpu_keys:
                    del model_params[key]
                
                print(f"[DEBUG] CPU専用パラメータ: {model_params}")
                
                try:
                    self._model = Llama(**model_params)
                    self.is_using_gpu = False
                    print("[INFO] CPUモードでGemmaモデルをロードしました（フォールバック）")
                    
                    # ロード時間記録
                    load_time = time.time() - start_time
                    self.logger.info(f"Gemma model (CPU fallback) loaded in {load_time:.2f} seconds")
                    print(f"[INFO] Gemmaモデル(CPU) {load_time:.2f} 秒でロードしました")
                except Exception as e2:
                    self.logger.error(f"Failed to load model in CPU mode: {e2}")
                    print(f"[ERROR] CPU専用モードでもロード失敗: {e2}")
                    self._model = None
                
        except Exception as e:
            self.logger.error(f"Failed to load Gemma model: {e}")
            print(f"[ERROR] Gemmaモデルのロードに失敗しました: {e}")
            print(f"[DEBUG] 詳細エラー情報:\n{traceback.format_exc()}")
            self._model = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        テキスト生成を実行

        Args:
            prompt: 入力プロンプト
            **kwargs: 生成パラメータ
            
        Returns:
            生成されたテキスト
        """
        if self._model is None:
            return "モデルが正しく読み込まれていません。モデルファイルのパスを確認してください。"
            
        # パラメータ設定
        temperature = kwargs.get("temperature", self.temp)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        try:
            # 現在のGPU状態をログ
            if self.is_using_gpu:
                print(f"[DEBUG] 生成: GPU使用中")
                
                # 生成前のGPUメモリ状態確認
                try:
                    import torch
                    if torch.cuda.is_available():
                        allocated_before = torch.cuda.memory_allocated(0) / (1024**3)
                        reserved_before = torch.cuda.memory_reserved(0) / (1024**3)
                        print(f"[DEBUG] 生成前GPUメモリ: 割当済 {allocated_before:.2f} GB / 予約済 {reserved_before:.2f} GB")
                except Exception:
                    pass
            else:
                print("[DEBUG] 生成: CPU使用中")
                
            # 生成実行
            start_time = time.time()
            response = self._model.create_completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=kwargs.get("top_p", 0.9),
                repeat_penalty=kwargs.get("repeat_penalty", 1.1),
                stop=kwargs.get("stop", None)
            )
            gen_time = time.time() - start_time
            
            # パフォーマンスログ
            token_count = len(response.get("tokens", []))
            if token_count > 0:
                tokens_per_sec = token_count / gen_time
                print(f"[DEBUG] 生成パフォーマンス: {gen_time:.2f}秒, {token_count}トークン, {tokens_per_sec:.2f}トークン/秒")
                
            # 生成後のGPUメモリ状態確認
            if self.is_using_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        allocated_after = torch.cuda.memory_allocated(0) / (1024**3)
                        reserved_after = torch.cuda.memory_reserved(0) / (1024**3)
                        print(f"[DEBUG] 生成後GPUメモリ: 割当済 {allocated_after:.2f} GB / 予約済 {reserved_after:.2f} GB")
                        
                        # メモリ使用の変化を確認
                        try:
                            memory_diff = allocated_after - allocated_before
                            if memory_diff > 0.001:  # 1MB以上の変化
                                print(f"[DEBUG] ✅ GPUメモリ増加: {memory_diff:.3f} GB - GPUが使用されています")
                            else:
                                print("[DEBUG] ⚠️ GPUメモリ変化なし - GPUが正しく使用されていない可能性")
                        except:
                            pass
                        
                        # GPU使用率も確認（可能であれば）
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            print(f"[DEBUG] 現在のGPU使用率: {util.gpu}%, メモリ使用率: {util.memory}%")
                            pynvml.nvmlShutdown()
                        except:
                            pass
                except Exception:
                    pass
            
            # テキスト抽出
            if isinstance(response, dict) and "choices" in response:
                text = response["choices"][0]["text"]
                return text
            else:
                return str(response)
                
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            print(f"[ERROR] テキスト生成エラー: {e}")
            return f"生成エラー: {str(e)}"
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        チャット形式での生成を実行

        Args:
            messages: メッセージの履歴
            **kwargs: 生成パラメータ
            
        Returns:
            生成結果を含む辞書
        """
        if self._model is None:
            return {"content": "モデルが正しく読み込まれていません。モデルファイルのパスを確認してください。"}
            
        # チャット形式を通常のプロンプトに変換
        prompt = self._format_chat_prompt(messages)
        
        # 生成実行
        response_text = self.generate(prompt, **kwargs)
        
        return {"content": response_text}
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        チャットメッセージをGemmaフォーマットのプロンプトに変換

        Args:
            messages: メッセージの履歴
            
        Returns:
            フォーマットされたプロンプト
        """
        formatted_prompt = ""
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted_prompt += f"<s>\n{content}\n</s>\n\n"
            elif role == "user":
                formatted_prompt += f"<user>\n{content}\n</user>\n\n"
            elif role == "assistant":
                formatted_prompt += f"<assistant>\n{content}\n</assistant>\n\n"
                
        # 最後にアシスタントの応答を求めるプロンプトを追加
        if messages and messages[-1]["role"] != "assistant":
            formatted_prompt += "<assistant>\n"
            
        return formatted_prompt
        
    def get_capabilities(self) -> Dict[str, Any]:
        """
        プロバイダーの機能情報を返す

        Returns:
            機能情報を含む辞書
        """
        return {
            "supports_embedding": False,
            "supports_streaming": False,
            "supports_vision": False,
            "max_tokens": self.max_tokens,
            "context_window": self.n_ctx,
            "gpu_enabled": self.is_using_gpu,
            "gpu_info": self.gpu_info
        }
