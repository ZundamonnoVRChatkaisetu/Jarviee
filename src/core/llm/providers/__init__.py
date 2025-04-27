"""
LLMプロバイダーモジュール

各種LLMプロバイダーの実装を提供します。
"""

from .base import LLMProvider

# 可能であれば各プロバイダーをインポート
try:
    from .gemma_provider import GemmaProvider
except ImportError:
    pass

# プロバイダー名からクラスへのマッピング
PROVIDER_MAP = {
    "gemma": "GemmaProvider",
    "openai": "OpenAIProvider",
    "anthropic": "AnthropicProvider",
    "local": "LocalProvider"
}

__all__ = [
    "LLMProvider",
    "PROVIDER_MAP"
]
