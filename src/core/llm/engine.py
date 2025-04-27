"""
LLMエンジン - 言語モデル処理の中核コンポーネント

このモジュールは、異なるLLMプロバイダー（OpenAI、Anthropic、ローカルモデルなど）との
統一的なインターフェースを提供し、高レベルの言語処理機能を実装します。
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from pydantic import BaseModel, Field

# 共通インターフェースをインポート
# プロバイダーモジュールからインポート
from .providers import LLMProvider
try:
    from .providers.gemma_provider import GemmaProvider
except ImportError:
    GemmaProvider = None


class LLMEngine:
    """
    複数のLLMプロバイダーを統合し、高レベルの言語処理機能を提供するエンジン
    """
    
    def __init__(self, config_path: str = None):
        """
        LLMエンジンを初期化
        
        Args:
            config_path: 設定ファイルパス。省略時は環境変数から設定を読み込む
        """
        # ロガーを最初に初期化
        self.logger = logging.getLogger(__name__)
        
        self.config = self._load_config(config_path)
        self.providers = {}
        self._init_providers()
        
        # デフォルトプロバイダーの設定
        default_provider = self.config["llm"]["default_provider"]
        if default_provider in self.providers:
            self.default_provider = default_provider
        else:
            # 利用可能なプロバイダーから選択
            self.default_provider = next(iter(self.providers)) if self.providers else None
            
        self.logger.info(f"LLMEngine initialized with {len(self.providers)} providers")
        if self.default_provider:
            self.logger.info(f"Default provider set to: {self.default_provider}")
        else:
            self.logger.warning("No LLM providers available!")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """設定を読み込む"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # デフォルト設定パス
        default_paths = [
            './config/config.json',
            '../config/config.json',
            '../../config/config.json',
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        # 設定が見つからない場合は最小限の設定を返す
        return {
            "llm": {
                "default_provider": "openai",
                "providers": {
                    "openai": {
                        "api_key": os.environ.get("OPENAI_API_KEY", ""),
                        "models": {
                            "default": "gpt-3.5-turbo"
                        }
                    }
                }
            }
        }
    
    def _init_providers(self):
        """利用可能なLLMプロバイダーを初期化"""
        providers_config = self.config["llm"]["providers"]
        
        # Llama Server
        if "llama_server" in providers_config and "url" in providers_config["llama_server"]:
            try:
                # 遅延インポート
                from .providers.llama_server_provider import LlamaServerProvider
                server_url = providers_config["llama_server"]["url"]
                self.logger.info(f"Initializing Llama Server provider with URL: {server_url}")
                self.providers["llama_server"] = LlamaServerProvider(server_url, providers_config["llama_server"])
            except ImportError:
                self.logger.warning("LlamaServerProvider could not be imported")
                
        # Gemmaモデル
        if "gemma" in providers_config and "path" in providers_config["gemma"]:
            if GemmaProvider is not None:
                model_path = providers_config["gemma"]["path"]
                # 環境変数で設定されていれば優先
                model_path = os.environ.get("GEMMA_MODEL_PATH", model_path)
                
                if os.path.exists(model_path):
                    self.logger.info(f"Initializing Gemma provider with model: {model_path}")
                    self.providers["gemma"] = GemmaProvider(model_path, providers_config["gemma"])
                else:
                    self.logger.warning(f"Gemma model not found at path: {model_path}")
            else:
                self.logger.warning("GemmaProvider could not be imported. Make sure llama-cpp-python is installed.")
        
        # OpenAI
        if "openai" in providers_config and os.environ.get("OPENAI_API_KEY"):
            try:
                # 遅延インポート
                from .providers.openai_provider import OpenAIProvider
                api_key = os.environ.get("OPENAI_API_KEY")
                self.providers["openai"] = OpenAIProvider(api_key, providers_config["openai"])
            except ImportError:
                self.logger.warning("OpenAIProvider could not be imported")
        
        # Anthropic
        if "anthropic" in providers_config and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                # 遅延インポート
                from .providers.anthropic_provider import AnthropicProvider
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                self.providers["anthropic"] = AnthropicProvider(api_key, providers_config["anthropic"])
            except ImportError:
                self.logger.warning("AnthropicProvider could not be imported")
        
        # ローカルモデル
        if "local" in providers_config and "path" in providers_config["local"]:
            try:
                # 遅延インポート
                from .providers.local_provider import LocalProvider
                model_path = providers_config["local"]["path"]
                if os.path.exists(model_path):
                    self.providers["local"] = LocalProvider(model_path, providers_config["local"])
            except ImportError:
                self.logger.warning("LocalProvider could not be imported")
    
    async def generate(self, prompt: str, provider: str = None, **kwargs) -> str:
        """
        テキスト生成を実行（非同期版）
        
        Args:
            prompt: 入力プロンプト
            provider: 使用するプロバイダー（省略時はデフォルト）
            **kwargs: プロバイダー固有のパラメータ
            
        Returns:
            生成されたテキスト
        """
        provider_name = provider or self.default_provider
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")
            
        return await self.providers[provider_name].generate(prompt, **kwargs)
    
    async def chat(self, messages: List[Dict[str, str]], provider: str = None, **kwargs) -> Dict[str, Any]:
        """
        チャット形式での生成を実行（非同期版）
        
        Args:
            messages: 会話履歴
            provider: 使用するプロバイダー（省略時はデフォルト）
            **kwargs: プロバイダー固有のパラメータ
            
        Returns:
            生成結果を含む辞書
        """
        provider_name = provider or self.default_provider
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")
            
        return await self.providers[provider_name].chat(messages, **kwargs)
    
    async def embed(self, text: str, provider: str = None) -> List[float]:
        """
        テキストの埋め込みベクトルを生成（非同期版）
        
        Args:
            text: 入力テキスト
            provider: 使用するプロバイダー（省略時はデフォルト）
            
        Returns:
            埋め込みベクトル
        """
        provider_name = provider or self.default_provider
        if provider_name not in self.providers:
            # 埋め込みをサポートするプロバイダーを探す
            for p_name, provider in self.providers.items():
                try:
                    return await provider.embed(text)
                except NotImplementedError:
                    continue
            raise ValueError("No provider available for embeddings")
            
        return await self.providers[provider_name].embed(text)

    def generate_sync(self, prompt: str, provider: str = None, **kwargs) -> str:
        """
        テキスト生成を実行（同期版）
        
        Args:
            prompt: 入力プロンプト
            provider: 使用するプロバイダー（省略時はデフォルト）
            **kwargs: プロバイダー固有のパラメータ
            
        Returns:
            生成されたテキスト
        """
        provider_name = provider or self.default_provider
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")
        
        # プロバイダーが同期メソッドを持っているか確認
        if hasattr(self.providers[provider_name], "generate") and not self.providers[provider_name].generate.__name__.startswith("async"):
            # 同期メソッドを直接呼び出し
            return self.providers[provider_name].generate(prompt, **kwargs)
        else:
            # 非同期メソッドを同期的に呼び出す（簡易版）
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # イベントループがない場合は新規作成
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(self.generate(prompt, provider, **kwargs))
    
    def chat_sync(self, messages: List[Dict[str, str]], provider: str = None, **kwargs) -> Dict[str, Any]:
        """
        チャット形式での生成を実行（同期版）
        
        Args:
            messages: 会話履歴
            provider: 使用するプロバイダー（省略時はデフォルト）
            **kwargs: プロバイダー固有のパラメータ
            
        Returns:
            生成結果を含む辞書
        """
        provider_name = provider or self.default_provider
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")
        
        # プロバイダーが同期メソッドを持っているか確認
        if hasattr(self.providers[provider_name], "chat") and not self.providers[provider_name].chat.__name__.startswith("async"):
            # 同期メソッドを直接呼び出し
            return self.providers[provider_name].chat(messages, **kwargs)
        else:
            # 非同期メソッドを同期的に呼び出す（簡易版）
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # イベントループがない場合は新規作成
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(self.chat(messages, provider, **kwargs))
