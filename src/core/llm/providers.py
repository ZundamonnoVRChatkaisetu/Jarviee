"""
LLMプロバイダーインターフェース定義

このモジュールでは、すべてのLLMプロバイダーが実装すべき共通インターフェースを定義します。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any

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
