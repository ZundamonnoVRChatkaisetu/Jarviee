"""
LLMエンジン - 言語モデル処理の中核コンポーネント

このモジュールは、異なるLLMプロバイダー（OpenAI、Anthropic、ローカルモデルなど）との
統一的なインターフェースを提供し、高レベルの言語処理機能を実装します。
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field

# プロバイダー実装は将来的に別モジュールに移動
class LLMProvider(ABC):
    """LLMプロバイダーの抽象基底クラス"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """テキスト生成を実行"""
        pass
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """チャット形式での生成を実行"""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """テキストの埋め込みベクトルを生成"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI APIを利用したLLMプロバイダー"""
    
    def __init__(self, api_key: str, config: Dict[str, Any]):
        self.api_key = api_key
        self.config = config
        # 実際のAPIクライアント初期化は遅延して行う
        self._client = None
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """テキスト生成を実行（未実装）"""
        # TODO: OpenAI APIを用いた実装
        return f"OpenAI generated response for: {prompt[:30]}..."
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """チャット形式での生成を実行（未実装）"""
        # TODO: OpenAI APIを用いた実装
        return {"content": f"OpenAI chat response for {len(messages)} messages"}
    
    async def embed(self, text: str) -> List[float]:
        """テキストの埋め込みベクトルを生成（未実装）"""
        # TODO: OpenAI APIを用いた実装
        return [0.1, 0.2, 0.3]  # ダミーデータ


class AnthropicProvider(LLMProvider):
    """Anthropic APIを利用したLLMプロバイダー"""
    
    def __init__(self, api_key: str, config: Dict[str, Any]):
        self.api_key = api_key
        self.config = config
        # 実際のAPIクライアント初期化は遅延して行う
        self._client = None
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """テキスト生成を実行（未実装）"""
        # TODO: Anthropic APIを用いた実装
        return f"Anthropic generated response for: {prompt[:30]}..."
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """チャット形式での生成を実行（未実装）"""
        # TODO: Anthropic APIを用いた実装
        return {"content": f"Anthropic chat response for {len(messages)} messages"}
    
    async def embed(self, text: str) -> List[float]:
        """テキストの埋め込みベクトルを生成（未実装）"""
        # Anthropicは現在埋め込みをサポートしていないため、別のプロバイダーを使用
        raise NotImplementedError("Anthropic does not support embeddings yet")


class LocalProvider(LLMProvider):
    """ローカルモデルを利用したLLMプロバイダー"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        # 実際のモデルロードは遅延して行う
        self._model = None
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """テキスト生成を実行（未実装）"""
        # TODO: ローカルモデルを用いた実装
        return f"Local model generated response for: {prompt[:30]}..."
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """チャット形式での生成を実行（未実装）"""
        # TODO: ローカルモデルを用いた実装
        return {"content": f"Local model chat response for {len(messages)} messages"}
    
    async def embed(self, text: str) -> List[float]:
        """テキストの埋め込みベクトルを生成（未実装）"""
        # TODO: ローカルモデルを用いた実装
        return [0.1, 0.2, 0.3]  # ダミーデータ


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
            
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"LLMEngine initialized with {len(self.providers)} providers")
        if self.default_provider:
            self.logger.info(f"Default provider set to: {self.default_provider}")
        else:
            self.logger.warning("No LLM providers available!")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """設定を読み込む"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # デフォルト設定パス
        default_paths = [
            './config/config.json',
            '../config/config.json',
            '../../config/config.json',
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
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
        
        # OpenAI
        if "openai" in providers_config and os.environ.get("OPENAI_API_KEY"):
            api_key = os.environ.get("OPENAI_API_KEY")
            self.providers["openai"] = OpenAIProvider(api_key, providers_config["openai"])
        
        # Anthropic
        if "anthropic" in providers_config and os.environ.get("ANTHROPIC_API_KEY"):
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            self.providers["anthropic"] = AnthropicProvider(api_key, providers_config["anthropic"])
        
        # ローカルモデル
        if "local" in providers_config and "path" in providers_config["local"]:
            model_path = providers_config["local"]["path"]
            if os.path.exists(model_path):
                self.providers["local"] = LocalProvider(model_path, providers_config["local"])
    
    async def generate(self, prompt: str, provider: str = None, **kwargs) -> str:
        """
        テキスト生成を実行
        
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
        チャット形式での生成を実行
        
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
        テキストの埋め込みベクトルを生成
        
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
