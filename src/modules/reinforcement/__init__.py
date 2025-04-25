"""
強化学習モジュール

このパッケージは、LLMコアと強化学習（RL）を連携させるコンポーネントを提供します。
LLMの言語理解能力と、強化学習の環境適応・最適化能力を組み合わせることで、
より自律的で適応的なAI能力を実現します。
"""

from src.modules.reinforcement.rl_adapter import RLAdapter
from src.modules.reinforcement.reward_generator import RewardGenerator
from src.modules.reinforcement.environment_manager import EnvironmentManager
from src.modules.reinforcement.action_optimizer import ActionOptimizer
from src.modules.reinforcement.llm_rl_interface import LLMRLInterface

__all__ = [
    'RLAdapter',
    'RewardGenerator',
    'EnvironmentManager',
    'ActionOptimizer',
    'LLMRLInterface'
]
