"""
強化学習モジュール統合テスト

このモジュールはLLMコアと強化学習モジュール間の連携をテストします。
主要コンポーネント間の通信、データ変換、および機能統合を検証します。
"""

import os
import sys
import json
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.core.utils.logger import Logger
from src.core.utils.config import ConfigManager
from src.core.llm.engine import LLMEngine
from src.core.autonomy.rl.adapter import RLAdapter
from src.core.autonomy.rl.reward_generator import RewardFunctionGenerator
from src.core.autonomy.rl.state_manager import EnvironmentStateManager
from src.core.autonomy.rl.action_optimizer import ActionOptimizer
from src.tests.integration.rl_module.test_environment import RLTestEnvironment

class RLIntegrationTest(unittest.TestCase):
    """強化学習モジュール統合テストクラス"""
    
    @classmethod
    def setUpClass(cls):
        """テストケース実行前の一度だけの準備"""
        cls.logger = Logger(__name__)
        cls.logger.info("Setting up RL integration test suite")
        
        # 設定の読み込み
        config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '../../../config/test_config.json'
        ))
        
        if os.path.exists(config_path):
            cls.config = ConfigManager(config_path).get_config()
        else:
            # デフォルト設定
            cls.config = {
                'llm': {
                    'model': 'gpt-3.5-turbo',
                    'temperature': 0.2,
                    'max_tokens': 500,
                    'use_mock': True  # テスト用モック
                },
                'rl': {
                    'adapter': {
                        'reward_scaling': 1.0,
                        'state_normalization': True,
                        'action_smoothing': 0.1
                    },
                    'environment': {
                        'dimensions': [10, 10],
                        'action_space': {
                            'type': 'discrete',
                            'n': 4
                        },
                        'max_steps': 100
                    },
                    'algorithm': {
                        'name': 'dqn',
                        'learning_rate': 0.001,
                        'gamma': 0.99,
                        'epsilon': 0.1
                    }
                },
                'test': {
                    'episodes': 10,
                    'render': False
                }
            }
        
        # LLMモックの設定
        if cls.config['llm'].get('use_mock', True):
            cls.llm_engine = MagicMock(spec=LLMEngine)
            cls.llm_engine.generate.return_value = "This is a mock LLM response"
            
            # 報酬関数生成モック
            cls.llm_engine.generate_for_reward_function = MagicMock(return_value="""
            def reward_function(state, action, next_state, done, info):
                reward = 0.0
                
                # ゴール到達報酬
                if done and info.get('success', False):
                    reward += 10.0
                
                # 時間ペナルティ
                reward -= 0.01
                
                # ゴールへの接近報酬
                if 'agent_pos' in state and 'goal_pos' in state:
                    prev_dist = ((state['agent_pos'][0] - state['goal_pos'][0]) ** 2 + 
                                 (state['agent_pos'][1] - state['goal_pos'][1]) ** 2) ** 0.5
                    curr_dist = ((next_state['agent_pos'][0] - next_state['goal_pos'][0]) ** 2 + 
                                 (next_state['agent_pos'][1] - next_state['goal_pos'][1]) ** 2) ** 0.5
                    
                    if curr_dist < prev_dist:
                        reward += 0.1
                
                return reward
            """)
        else:
            # 実際のLLMエンジンをロード
            cls.llm_engine = LLMEngine(cls.config['llm'])
            
        # RLコンポーネントの初期化
        cls.reward_generator = RewardFunctionGenerator(cls.llm_engine)
        cls.state_manager = EnvironmentStateManager()
        cls.action_optimizer = ActionOptimizer(cls.config['rl']['algorithm'])
        cls.rl_adapter = RLAdapter(
            cls.reward_generator,
            cls.state_manager,
            cls.action_optimizer
        )
        
        # テスト環境の初期化
        cls.test_env = RLTestEnvironment(cls.config['rl']['environment'])
        
        cls.logger.info("RL integration test suite setup completed")
    
    def setUp(self):
        """各テストケース前の準備"""
        self.logger.info("Setting up individual test case")
    
    def test_llm_rl_communication(self):
        """LLMとRL間の通信テスト"""
        self.logger.info("Testing LLM-RL communication")
        
        # 自然言語指示
        instruction = "Minimize energy consumption while maintaining optimal performance"
        
        # LLMから報酬関数への変換
        reward_function = self.reward_generator.generate_from_instruction(instruction)
        
        # 検証
        self.assertIsNotNone(reward_function, "Reward function should not be None")
        self.assertTrue(callable(reward_function), "Reward function should be callable")
        
        # 報酬関数の動作検証
        mock_state = {'agent_pos': [5, 5], 'goal_pos': [9, 9]}
        mock_next_state = {'agent_pos': [6, 6], 'goal_pos': [9, 9]}
        mock_action = 1
        mock_done = False
        mock_info = {}
        
        reward = reward_function(mock_state, mock_action, mock_next_state, mock_done, mock_info)
        self.assertIsInstance(reward, (int, float), "Reward should be a numeric value")
        
        self.logger.info(f"Generated reward function returned {reward} for test input")
    
    def test_state_conversion(self):
        """環境状態とRL状態表現の変換テスト"""
        self.logger.info("Testing environment state conversion")
        
        # 環境からの観測
        test_observation = np.zeros((10, 10))
        test_observation[5, 5] = 0.5  # エージェント位置
        test_observation[9, 9] = 1.0  # ゴール位置
        
        # 状態マネージャを使用して変換
        rl_state = self.state_manager.preprocess_observation(test_observation)
        
        # 検証
        self.assertIsNotNone(rl_state, "Processed state should not be None")
        
        # 変換された状態が正しい形式か確認
        if isinstance(rl_state, dict):
            self.assertIn('agent_pos', rl_state, "State should contain agent position")
            self.assertIn('goal_pos', rl_state, "State should contain goal position")
        elif isinstance(rl_state, np.ndarray):
            self.assertEqual(rl_state.shape, test_observation.shape, 
                            "State shape should be preserved")
        
        self.logger.info("State conversion test completed")
    
    def test_action_selection(self):
        """行動選択と最適化のテスト"""
        self.logger.info("Testing action selection and optimization")
        
        # 環境観測のセットアップ
        test_observation = np.zeros((10, 10))
        test_observation[5, 5] = 0.5  # エージェント位置
        test_observation[9, 9] = 1.0  # ゴール位置
        
        # RLアダプタを通じた行動選択
        action = self.rl_adapter.select_action(test_observation)
        
        # 検証
        self.assertIsNotNone(action, "Selected action should not be None")
        self.assertIsInstance(action, (int, np.integer, np.ndarray), 
                            "Action should be a valid type (int or array)")
        
        if isinstance(action, (int, np.integer)):
            self.assertGreaterEqual(action, 0, "Action index should be non-negative")
            self.assertLess(action, self.config['rl']['environment']['action_space']['n'], 
                           "Action index should be within action space")
        
        self.logger.info(f"Action selection test completed, selected action: {action}")
    
    def test_adaptation_mechanism(self):
        """RLの適応メカニズムテスト"""
        self.logger.info("Testing RL adaptation mechanism")
        
        # RLアダプタのリセット
        self.rl_adapter.reset()
        
        # 複数ステップでの適応テスト
        obs = self.test_env.simulator.reset()
        done = False
        episode_rewards = []
        
        while not done:
            # 行動選択
            action = self.rl_adapter.select_action(obs)
            
            # 環境ステップ
            next_obs, reward, done, info = self.test_env.simulator.step(action)
            episode_rewards.append(reward)
            
            # 更新
            self.rl_adapter.update(obs, action, reward, next_obs, done)
            
            obs = next_obs
        
        # 検証
        total_reward = sum(episode_rewards)
        self.logger.info(f"Adaptation test completed with total reward: {total_reward}")
        
        # モデルの更新確認（実際のパフォーマンス向上はより長期的なテストが必要）
        updated_params = self.rl_adapter.get_model_params() if hasattr(self.rl_adapter, 'get_model_params') else None
        self.logger.info(f"Model parameters after update: {updated_params}")
    
    def test_reward_shaping(self):
        """報酬形状調整のテスト"""
        self.logger.info("Testing reward shaping mechanism")
        
        # 自然言語指示
        instruction = "Prioritize efficiency over speed, and ensure safe operation"
        
        # 報酬関数の生成
        reward_function = self.reward_generator.generate_from_instruction(instruction)
        
        # RewardShaper（ある場合）を経由した報酬形状調整
        if hasattr(self.rl_adapter, 'reward_shaper'):
            original_reward = 1.0
            shaped_reward = self.rl_adapter.reward_shaper.shape_reward(
                original_reward, 
                {'instruction': instruction, 'efficiency': 0.8, 'safety': 0.9}
            )
            self.assertIsInstance(shaped_reward, (int, float), "Shaped reward should be numeric")
            self.logger.info(f"Original reward: {original_reward}, Shaped reward: {shaped_reward}")
        else:
            self.logger.info("RewardShaper not implemented, skipping detailed test")
            
            # 代わりに報酬関数の動作を検証
            mock_state = {'agent_pos': [5, 5], 'goal_pos': [9, 9], 'safety': 1.0}
            mock_next_state = {'agent_pos': [6, 6], 'goal_pos': [9, 9], 'safety': 0.8}
            mock_action = 1
            mock_done = False
            mock_info = {'efficiency': 0.9}
            
            reward = reward_function(mock_state, mock_action, mock_next_state, mock_done, mock_info)
            self.assertIsInstance(reward, (int, float), "Reward should be a numeric value")
            
            # 安全性低下の場合のペナルティ
            mock_next_state_unsafe = {'agent_pos': [6, 6], 'goal_pos': [9, 9], 'safety': 0.2}
            reward_unsafe = reward_function(mock_state, mock_action, mock_next_state_unsafe, mock_done, mock_info)
            
            # 安全性が低い場合は報酬が下がることを期待
            self.assertLess(reward_unsafe, reward, 
                          "Lower safety should result in lower reward")
            
            self.logger.info(f"Safe operation reward: {reward}, Unsafe operation reward: {reward_unsafe}")
    
    def test_end_to_end_simple_task(self):
        """シンプルなタスクのエンドツーエンドテスト"""
        self.logger.info("Testing end-to-end on a simple navigation task")
        
        # タスク指示
        instruction = "Navigate to the goal position efficiently"
        
        # 指示からの報酬関数生成
        self.rl_adapter.set_instruction(instruction)
        
        # テスト環境でのエピソード実行
        results = []
        n_episodes = min(5, self.config['test'].get('episodes', 5))
        
        for i in range(n_episodes):
            self.logger.info(f"Starting test episode {i+1}/{n_episodes}")
            render = self.config['test'].get('render', False)
            result = self.test_env.run_test_episode(self.rl_adapter, render)
            results.append(result)
            
            self.logger.info(f"Episode {i+1} result: {result}")
        
        # 成功率と平均報酬の計算
        success_rate = sum(1 for r in results if r['success']) / len(results)
        avg_reward = sum(r['total_reward'] for r in results) / len(results)
        
        self.logger.info(f"End-to-end test completed with success rate: {success_rate:.2f}, "
                      f"average reward: {avg_reward:.2f}")
        
        # エンドツーエンドテストでは具体的なアサーションは限定的
        # 実行が完了したことを確認
        self.assertEqual(len(results), n_episodes, 
                       f"All {n_episodes} episodes should complete")
    
    def tearDown(self):
        """各テストケース後の処理"""
        pass
    
    @classmethod
    def tearDownClass(cls):
        """全テストケース終了後の処理"""
        cls.logger.info("Cleaning up RL integration test suite")
        # 必要に応じてリソースの解放等を行う

if __name__ == "__main__":
    unittest.main()
