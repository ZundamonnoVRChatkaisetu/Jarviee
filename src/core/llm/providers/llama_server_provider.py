"""
Llama Server Provider

ローカルで実行されているllama-serverに接続するプロバイダー実装
"""

import os
import json
import logging
import time
import requests
from typing import Dict, List, Any, Optional, Union

# ベースプロバイダーのインポート
from .base import LLMProvider

class LlamaServerProvider(LLMProvider):
    """ローカルllama-serverと連携するLLMプロバイダー"""
    
    def __init__(self, server_url: str = None, config: Dict[str, Any] = None):
        """
        LlamaServerプロバイダーを初期化

        Args:
            server_url: llama-serverのURL
            config: 設定パラメータ
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # サーバーURL
        self.server_url = server_url or os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
        
        # デフォルト設定
        self.temp = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 2048)
        self.context_window = self.config.get("context_window", 8192)
        self.top_p = self.config.get("top_p", 0.9)
        self.top_k = self.config.get("top_k", 40)
        self.repeat_penalty = self.config.get("repeat_penalty", 1.1)
        
        # ログ
        self.logger.info(f"LlamaServerProvider initialized with server URL: {self.server_url}")
        
        # 接続確認
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """サーバー接続を確認"""
        try:
            response = requests.get(f"{self.server_url}/health")
            if response.status_code == 200:
                self.logger.info("Successfully connected to Llama Server")
                return True
            else:
                self.logger.warning(f"Failed to connect to Llama Server: {response.status_code}")
                return False
        except Exception as e:
            self.logger.warning(f"Failed to connect to Llama Server: {e}")
            return False
        
    def _get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        try:
            response = requests.get(f"{self.server_url}/model")
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.warning(f"Failed to get model info: {response.status_code}")
                return {}
        except Exception as e:
            self.logger.warning(f"Failed to get model info: {e}")
            return {}
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        テキスト生成を実行

        Args:
            prompt: 入力プロンプト
            **kwargs: 生成パラメータ
            
        Returns:
            生成されたテキスト
        """
        # パラメータ設定
        temperature = kwargs.get("temperature", self.temp)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", self.top_k)
        repeat_penalty = kwargs.get("repeat_penalty", self.repeat_penalty)
        stop = kwargs.get("stop", None)
        
        # リクエストデータ
        data = {
            "prompt": prompt,
            "temperature": temperature,
            "n_predict": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stop": stop if isinstance(stop, list) else [stop] if stop else []
        }
        
        try:
            # 生成リクエスト送信
            response = requests.post(f"{self.server_url}/completion", json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("content", "")
            else:
                error_msg = f"Error generating text: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return f"生成エラー: {error_msg}"
        except Exception as e:
            self.logger.error(f"Error calling Llama Server: {e}")
            return f"サーバー接続エラー: {str(e)}"
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        チャット形式での生成を実行

        Args:
            messages: メッセージの履歴
            **kwargs: 生成パラメータ
            
        Returns:
            生成結果を含む辞書
        """
        # パラメータ設定
        temperature = kwargs.get("temperature", self.temp)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", self.top_k)
        repeat_penalty = kwargs.get("repeat_penalty", self.repeat_penalty)
        stop = kwargs.get("stop", None)
        
        # リクエストデータ
        data = {
            "messages": messages,
            "temperature": temperature,
            "n_predict": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stop": stop if isinstance(stop, list) else [stop] if stop else []
        }
        
        try:
            # チャットリクエスト送信
            response = requests.post(f"{self.server_url}/chat", json=data)
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "")
                return {"content": content}
            else:
                error_msg = f"Error in chat generation: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {"content": f"生成エラー: {error_msg}"}
        except Exception as e:
            self.logger.error(f"Error calling Llama Server chat: {e}")
            return {"content": f"サーバー接続エラー: {str(e)}"}
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        プロバイダーの機能情報を返す

        Returns:
            機能情報を含む辞書
        """
        # モデル情報取得
        model_info = self._get_model_info()
        
        return {
            "supports_embedding": False,
            "supports_streaming": True,
            "supports_vision": False,
            "max_tokens": model_info.get("n_ctx", self.max_tokens),
            "context_window": model_info.get("n_ctx", self.context_window),
            "model_name": model_info.get("model_name", "unknown"),
            "using_gpu": model_info.get("gpu_layers", 0) > 0
        }
