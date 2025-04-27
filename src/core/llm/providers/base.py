"""
LLMプロバイダーの抽象基底クラス

異なるLLMプロバイダー（OpenAI、Anthropic、ローカルモデルなど）の
共通インターフェースを定義します。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class LLMProvider(ABC):
    """LLMプロバイダーの抽象基底クラス"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        テキスト生成を実行

        Args:
            prompt: 入力プロンプト
            **kwargs: プロバイダー固有のパラメータ
            
        Returns:
            生成されたテキスト
        """
        pass
        
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        チャット形式での生成を実行

        Args:
            messages: メッセージの履歴
                     各メッセージは {"role": "user"|"assistant"|"system", "content": str} の形式
            **kwargs: プロバイダー固有のパラメータ
            
        Returns:
            生成結果を含む辞書
        """
        pass
        
    def embed(self, text: str) -> List[float]:
        """
        テキストの埋め込みベクトルを生成

        Args:
            text: 入力テキスト
            
        Returns:
            埋め込みベクトル
        """
        # デフォルトでは未実装
        raise NotImplementedError("This provider does not support embeddings")
        
    def get_capabilities(self) -> Dict[str, Any]:
        """
        プロバイダーの機能情報を返す

        Returns:
            機能情報を含む辞書
        """
        return {
            "supports_embedding": hasattr(self, "embed") and callable(self.embed),
            "supports_streaming": False,  # サブクラスでオーバーライド可能
            "supports_vision": False,     # サブクラスでオーバーライド可能
            "max_tokens": 2048,           # サブクラスでオーバーライド可能
            "context_window": 4096        # サブクラスでオーバーライド可能
        }
