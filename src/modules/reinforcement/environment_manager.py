"""
環境状態管理モジュール

このモジュールは強化学習における環境の抽象化、状態管理、
状態空間と行動空間の定義などを担当します。
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple, Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.core.utils.logger import Logger


class Environment:
    """強化学習環境の基底クラス"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        環境の初期化
        
        Args:
            config (Dict[str, Any], optional): 環境設定
        """
        self.logger = Logger(__name__)
        self.config = config or {}
        self.reset()
        
        self.logger.info("Environment initialized")
    
    def reset(self) -> Dict[str, Any]:
        """
        環境を初期状態にリセット
        
        Returns:
            Dict[str, Any]: 初期状態
        """
        self.state = {}
        self.done = False
        self.info = {}
        self.steps = 0
        self.max_steps = self.config.get('max_steps', 100)
        
        self.logger.debug("Environment reset")
        return self.get_state()
    
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        アクションを実行して環境を更新
        
        Args:
            action (Any): 実行するアクション
            
        Returns:
            Tuple[Dict[str, Any], float, bool, Dict[str, Any]]: 
                - 新しい状態
                - 報酬
                - 終了フラグ
                - 追加情報
        """
        # このメソッドはサブクラスでオーバーライドする必要がある
        raise NotImplementedError("step method must be implemented by subclass")
    
    def get_state(self) -> Dict[str, Any]:
        """
        現在の状態を取得
        
        Returns:
            Dict[str, Any]: 現在の状態
        """
        return self.state.copy()
    
    def get_observation(self) -> Dict[str, Any]:
        """
        エージェントが観測できる状態を取得
        (部分観測環境の場合、状態の一部のみ返す)
        
        Returns:
            Dict[str, Any]: 観測可能な状態
        """
        # デフォルトでは完全観測とし、すべての状態を返す
        return self.get_state()
    
    def render(self, mode: str = 'text') -> Optional[str]:
        """
        環境の現在状態を視覚化
        
        Args:
            mode (str): 描画モード (text/html/rgb_array)
            
        Returns:
            Optional[str]: テキスト表現またはNone
        """
        if mode == 'text':
            # シンプルなテキスト表現
            return f"State: {self.state}, Steps: {self.steps}/{self.max_steps}, Done: {self.done}"
        else:
            return None
    
    def close(self) -> None:
        """環境のクリーンアップ"""
        self.logger.debug("Environment closed")


class TextEnvironment(Environment):
    """テキストベース環境の実装"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        テキスト環境の初期化
        
        Args:
            config (Dict[str, Any], optional): 環境設定
        """
        super().__init__(config)
        self.context = self.config.get('initial_context', "")
        self.goal = self.config.get('goal', "")
        self.history = []
        
        self.logger.info("TextEnvironment initialized")
    
    def reset(self) -> Dict[str, Any]:
        """
        テキスト環境のリセット
        
        Returns:
            Dict[str, Any]: 初期状態
        """
        super().reset()
        self.context = self.config.get('initial_context', "")
        self.goal = self.config.get('goal', "")
        self.history = []
        
        self.state = {
            'context': self.context,
            'goal': self.goal,
            'history': self.history
        }
        
        self.logger.debug("TextEnvironment reset")
        return self.get_state()
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        テキストアクションの実行
        
        Args:
            action (str): 実行するテキストアクション
            
        Returns:
            Tuple[Dict[str, Any], float, bool, Dict[str, Any]]: 
                - 新しい状態
                - 報酬
                - 終了フラグ
                - 追加情報
        """
        if self.done:
            return self.get_state(), 0.0, True, {'warning': 'Environment already done'}
        
        self.steps += 1
        
        # アクションを履歴に追加
        self.history.append(action)
        
        # コンテキストの更新 (例: シンプルな連結)
        self.context += f"\n{action}"
        
        # 状態の更新
        self.state = {
            'context': self.context,
            'goal': self.goal,
            'history': self.history,
            'steps': self.steps
        }
        
        # 終了条件のチェック
        self.done = self._check_done()
        
        # 報酬計算 (このシンプルな実装では常に0)
        # 実際のアプリケーションでは、報酬関数を使用
        reward = 0.0
        
        # 情報の更新
        self.info = {
            'steps': self.steps,
            'max_steps': self.max_steps
        }
        
        return self.get_state(), reward, self.done, self.info
    
    def _check_done(self) -> bool:
        """
        環境が終了状態かどうかを確認
        
        Returns:
            bool: 終了しているかどうか
        """
        # 最大ステップ数に達した場合
        if self.steps >= self.max_steps:
            return True
        
        # 目標達成条件の確認 (シンプルな例)
        if self.goal in self.context:
            return True
        
        return False
    
    def render(self, mode: str = 'text') -> Optional[str]:
        """
        テキスト環境の視覚化
        
        Args:
            mode (str): 描画モード
            
        Returns:
            Optional[str]: テキスト表現またはNone
        """
        if mode == 'text':
            output = [
                f"Steps: {self.steps}/{self.max_steps}",
                f"Goal: {self.goal}",
                f"Context: {self.context[:100]}..." if len(self.context) > 100 else f"Context: {self.context}",
                f"History: {len(self.history)} actions"
            ]
            return "\n".join(output)
        
        return None


class EnvironmentManager:
    """環境の管理とラッピングを行うクラス"""
    
    def __init__(self):
        """環境マネージャーの初期化"""
        self.logger = Logger(__name__)
        self.environments = {}
        self.active_env = None
        
        # 環境タイプの登録
        self.env_types = {
            'text': TextEnvironment,
            # 他の環境タイプをここに追加
        }
        
        self.logger.info("EnvironmentManager initialized")
    
    def create_environment(self, env_type: str, env_id: str, 
                         config: Optional[Dict[str, Any]] = None) -> Optional[Environment]:
        """
        新しい環境の作成
        
        Args:
            env_type (str): 環境タイプ
            env_id (str): 環境ID
            config (Dict[str, Any], optional): 環境設定
            
        Returns:
            Optional[Environment]: 作成された環境またはNone
        """
        if env_type not in self.env_types:
            self.logger.error(f"Unknown environment type: {env_type}")
            return None
            
        try:
            # 環境のインスタンス化
            env_class = self.env_types[env_type]
            env = env_class(config)
            
            # 環境の登録
            self.environments[env_id] = env
            
            # アクティブ環境の設定
            if self.active_env is None:
                self.active_env = env_id
                
            self.logger.info(f"Created environment: {env_id} of type {env_type}")
            return env
            
        except Exception as e:
            self.logger.error(f"Error creating environment: {str(e)}")
            return None
    
    def get_environment(self, env_id: Optional[str] = None) -> Optional[Environment]:
        """
        環境の取得
        
        Args:
            env_id (str, optional): 環境ID (未指定の場合はアクティブ環境)
            
        Returns:
            Optional[Environment]: 環境インスタンスまたはNone
        """
        # 環境IDの決定
        if env_id is None:
            env_id = self.active_env
            
        if env_id is None:
            self.logger.warning("No active environment")
            return None
            
        if env_id not in self.environments:
            self.logger.error(f"Environment not found: {env_id}")
            return None
            
        return self.environments[env_id]
    
    def set_active_environment(self, env_id: str) -> bool:
        """
        アクティブ環境の設定
        
        Args:
            env_id (str): 環境ID
            
        Returns:
            bool: 成功したかどうか
        """
        if env_id not in self.environments:
            self.logger.error(f"Environment not found: {env_id}")
            return False
            
        self.active_env = env_id
        self.logger.info(f"Active environment set to {env_id}")
        return True
    
    def delete_environment(self, env_id: str) -> bool:
        """
        環境の削除
        
        Args:
            env_id (str): 環境ID
            
        Returns:
            bool: 成功したかどうか
        """
        if env_id not in self.environments:
            self.logger.error(f"Environment not found: {env_id}")
            return False
            
        # 環境のクリーンアップ
        try:
            self.environments[env_id].close()
        except Exception as e:
            self.logger.warning(f"Error closing environment: {str(e)}")
            
        # 環境の削除
        del self.environments[env_id]
        
        # アクティブ環境の更新
        if self.active_env == env_id:
            self.active_env = next(iter(self.environments.keys())) if self.environments else None
            
        self.logger.info(f"Deleted environment: {env_id}")
        return True
    
    def register_environment_type(self, env_type: str, env_class) -> bool:
        """
        新しい環境タイプの登録
        
        Args:
            env_type (str): 環境タイプ名
            env_class: 環境クラス
            
        Returns:
            bool: 成功したかどうか
        """
        if env_type in self.env_types:
            self.logger.warning(f"Environment type already exists: {env_type}")
            return False
            
        self.env_types[env_type] = env_class
        self.logger.info(f"Registered environment type: {env_type}")
        return True
    
    def get_environment_info(self, env_id: Optional[str] = None) -> Dict[str, Any]:
        """
        環境の情報を取得
        
        Args:
            env_id (str, optional): 環境ID (未指定の場合はアクティブ環境)
            
        Returns:
            Dict[str, Any]: 環境情報
        """
        env = self.get_environment(env_id)
        
        if env is None:
            return {
                'error': 'Environment not found',
                'available_environments': list(self.environments.keys())
            }
            
        # 環境情報の収集
        info = {
            'env_id': env_id or self.active_env,
            'env_type': type(env).__name__,
            'state': env.get_state(),
            'steps': getattr(env, 'steps', 0),
            'max_steps': getattr(env, 'max_steps', 'unknown'),
            'done': getattr(env, 'done', False)
        }
        
        return info
    
    def convert_state_to_vector(self, state: Dict[str, Any], 
                              feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        状態を特徴ベクトルに変換
        
        Args:
            state (Dict[str, Any]): 状態辞書
            feature_names (List[str], optional): 使用する特徴の名前
            
        Returns:
            np.ndarray: 特徴ベクトル
        """
        # 特徴名が指定されていない場合は、状態のすべてのキーを使用
        if feature_names is None:
            feature_names = list(state.keys())
            
        # 各特徴の値を抽出・変換
        feature_vector = []
        
        for feature in feature_names:
            if feature in state:
                value = state[feature]
                
                # 型に応じた処理
                if isinstance(value, (int, float)):
                    feature_vector.append(float(value))
                elif isinstance(value, bool):
                    feature_vector.append(1.0 if value else 0.0)
                elif isinstance(value, str):
                    # 文字列の簡易ベクトル化 (長さなど)
                    feature_vector.append(float(len(value)))
                elif isinstance(value, (list, tuple)):
                    # リストの簡易ベクトル化 (長さなど)
                    feature_vector.append(float(len(value)))
                else:
                    # その他の型は0としておく
                    feature_vector.append(0.0)
            else:
                # 存在しない特徴は0で埋める
                feature_vector.append(0.0)
                
        return np.array(feature_vector, dtype=np.float32)
