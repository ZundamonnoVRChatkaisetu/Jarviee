"""
知識獲得モジュール

自律的に新しい知識を獲得するための機能を提供するモジュール群。
- 興味生成エンジン: 何を学ぶべきかを決定
- 情報収集エージェント: 多様な情報源からデータを収集
- 知識検証フレームワーク: 収集した情報の信頼性を検証
- 知識構造化: 情報を構造化された知識に変換
"""

from .interest_engine import InterestEngine
from .collector import InformationCollector
from .validator import KnowledgeValidator
