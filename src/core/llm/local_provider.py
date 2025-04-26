"""
ローカルLLMプロバイダー実装

このモジュールでは、ローカルにインストールされた言語モデルを使用するためのプロバイダーを実装します。
GGUFフォーマットのモデルをサポートし、GPUを活用して推論を高速化します。
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import asyncio
from pathlib import Path

from .engine import LLMProvider

class LocalModelProvider(LLMProvider):
    """ローカルモデルを利用したLLMプロバイダー（GPU対応）"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        ローカルモデルプロバイダーを初期化
        
        Args:
            model_path: モデルファイルのパス（デフォルトでは特定のモデルを使用）
            config: 設定ディクショナリ
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # モデルパスの解決（環境変数や相対パスの処理）
        if model_path.startswith("${") and model_path.endswith("}"):
            env_var = model_path[2:-1]
            model_path = os.environ.get(env_var, "")
            
        if not model_path or not os.path.exists(model_path):
            # デフォルトのモデルディレクトリから利用可能なモデルを検索
            models_dir = Path("./models")
            if not models_dir.exists():
                models_dir = Path("../models")
            if not models_dir.exists():
                models_dir = Path("../../models")
                
            if models_dir.exists():
                # 一般的なモデル拡張子を持つファイルを検索
                model_files = list(models_dir.glob("*.gguf"))
                if model_files:
                    model_path = str(model_files[0])
                    self.logger.info(f"自動的にモデルを選択しました: {model_path}")
                else:
                    raise ValueError("利用可能なモデルが見つかりません")
            else:
                raise ValueError("モデルディレクトリが見つかりません")
        
        self.model_path = model_path
        self.model_instance = None
        self.model_loaded = False
        
        # GPU設定
        self.use_gpu = config.get("use_gpu", True)
        self.gpu_layers = config.get("gpu_layers", -1)  # -1 = 全レイヤー
        
        # モデルパラメータ
        self.temperature = config.get("parameters", {}).get("temperature", 0.7)
        self.max_tokens = config.get("parameters", {}).get("max_tokens", 2048)
        self.top_p = config.get("parameters", {}).get("top_p", 0.9)
        self.repeat_penalty = config.get("parameters", {}).get("repeat_penalty", 1.1)
        
        self.logger.info(f"ローカルモデルプロバイダーを初期化: {model_path}")
        self.logger.info(f"GPU使用: {self.use_gpu}, GPUレイヤー: {self.gpu_layers}")
    
    def _load_model(self):
        """モデルを必要に応じてロード"""
        if self.model_loaded:
            return
            
        try:
            # llama-cpp-pythonのインポート
            from llama_cpp import Llama
            
            # n_gpu_layersが-1の場合、モデルのすべてのレイヤーをGPUでロード
            n_gpu_layers = self.gpu_layers if self.gpu_layers > 0 else None
            
            # モデルロード（GPU使用）
            self.logger.info(f"モデルをロード中: {self.model_path}")
            self.logger.info(f"GPU設定: use_gpu={self.use_gpu}, n_gpu_layers={n_gpu_layers}")
            
            self.model_instance = Llama(
                model_path=self.model_path,
                n_ctx=self.max_tokens,
                n_gpu_layers=n_gpu_layers if self.use_gpu else 0,
                seed=42
            )
            
            self.model_loaded = True
            self.logger.info("モデルのロードが完了しました")
            
        except ImportError:
            self.logger.error("llama-cpp-pythonがインストールされていません")
            raise ImportError("llama-cpp-pythonをインストールしてください: pip install llama-cpp-python")
        except Exception as e:
            self.logger.error(f"モデルのロード中にエラーが発生しました: {str(e)}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        テキスト生成を実行
        
        Args:
            prompt: 入力プロンプト
            **kwargs: 生成パラメータ（temperature、max_tokensなど）
            
        Returns:
            生成されたテキスト
        """
        # モデルが未ロードの場合はロード
        if not self.model_loaded:
            self._load_model()
        
        # パラメータの設定
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        repeat_penalty = kwargs.get("repeat_penalty", self.repeat_penalty)
        
        # 非同期で計算負荷の高い処理を実行
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.model_instance.create_completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop=["</s>", "\n\n"]
            )
        )
        
        # 結果の取得
        generated_text = result["choices"][0]["text"]
        return generated_text
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        チャット形式での生成を実行
        
        Args:
            messages: 会話履歴
            **kwargs: 生成パラメータ
            
        Returns:
            生成結果を含む辞書
        """
        # モデルが未ロードの場合はロード
        if not self.model_loaded:
            self._load_model()
        
        # メッセージをプロンプト形式に変換
        prompt = self._format_chat_messages(messages)
        
        # パラメータの設定
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        repeat_penalty = kwargs.get("repeat_penalty", self.repeat_penalty)
        
        # 非同期で計算負荷の高い処理を実行
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.model_instance.create_completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop=["</s>", "\n\nユーザー:"]
            )
        )
        
        # 結果の取得
        generated_text = result["choices"][0]["text"]
        
        return {
            "role": "assistant",
            "content": generated_text
        }
    
    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        チャットメッセージをプロンプト形式に変換
        
        Args:
            messages: 会話履歴のリスト
            
        Returns:
            フォーマットされたプロンプト文字列
        """
        formatted_messages = []
        
        for message in messages:
            role = message["role"].lower()
            content = message["content"]
            
            if role == "system":
                formatted_messages.append(f"システム: {content}")
            elif role == "user":
                formatted_messages.append(f"ユーザー: {content}")
            elif role == "assistant":
                formatted_messages.append(f"アシスタント: {content}")
            else:
                # その他の役割はそのまま追加
                formatted_messages.append(f"{role}: {content}")
        
        # 最後のメッセージがユーザーでない場合、生成すべき内容がアシスタントの応答
        if messages and messages[-1]["role"].lower() != "user":
            formatted_messages.append("アシスタント:")
        else:
            formatted_messages.append("アシスタント:")
        
        return "\n\n".join(formatted_messages)
    
    async def embed(self, text: str) -> List[float]:
        """
        テキストの埋め込みベクトルを生成
        
        Args:
            text: 入力テキスト
            
        Returns:
            埋め込みベクトル
        """
        # 埋め込みはまだサポートされていない
        raise NotImplementedError("ローカルモデルでの埋め込みはまだサポートされていません")
