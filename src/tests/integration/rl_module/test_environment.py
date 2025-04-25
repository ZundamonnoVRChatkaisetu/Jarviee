"""
強化学習モジュールのテスト環境

このモジュールは強化学習（RL）コンポーネントのテスト環境を定義します。
シミュレーション環境、メトリクス収集、ベースライン測定機能を提供します。
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.core.utils.logger import Logger
from src.core.autonomy.rl.environment import RLEnvironmentInterface
from src.core.autonomy.rl.metrics import MetricsCollector

class SimulationEnvironment:
    """シミュレーション環境クラス"""
    
    def __init__(self, config):
        """
        シミュレーション環境の初期化
        
        Args:
            config (dict): 環境設定パラメータ
                - dimensions (list): 状態空間の次元情報
                - action_space (dict): 行動空間の定義
                - max_steps (int): エピソードの最大ステップ数
                - reward_scale (float): 報酬スケーリング係数
                - noise_level (float): 環境ノイズレベル
        """
        self.logger = Logger(__name__)
        self.config = config
        self.state_dim = config.get('dimensions', [10, 10])
        self.action_space = config.get('action_space', {
            'type': 'discrete',
            'n': 4  # デフォルト: 上下左右
        })
        self.max_steps = config.get('max_steps', 100)
        self.reward_scale = config.get('reward_scale', 1.0)
        self.noise_level = config.get('noise_level', 0.05)
        
        # 現在の状態と環境変数の初期化
        self.reset()
        
        self.logger.info(f"SimulationEnvironment initialized with dimensions: {self.state_dim}")
        
    def reset(self):
        """環境を初期状態にリセット"""
        # 状態空間の初期化
        if isinstance(self.state_dim, list):
            # 多次元状態空間
            self.state = np.zeros(self.state_dim)
            self.agent_pos = [0, 0]  # エージェントの初期位置
            self.goal_pos = [self.state_dim[0]-1, self.state_dim[1]-1]  # ゴールの位置
            
            # ゴール位置の設定
            self._set_goal()
        else:
            # 単一次元状態空間
            self.state = 0
            
        self.steps = 0
        self.done = False
        self.info = {}
        
        return self._get_observation()
    
    def _set_goal(self):
        """環境内にゴールを設定"""
        if isinstance(self.state_dim, list) and len(self.state_dim) >= 2:
            self.state[self.goal_pos[0], self.goal_pos[1]] = 1.0
    
    def step(self, action):
        """
        環境で一歩進める
        
        Args:
            action: エージェントの行動
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        self.steps += 1
        
        # 行動の実行
        if self.action_space['type'] == 'discrete':
            self._discrete_step(action)
        else:
            self._continuous_step(action)
            
        # 終了条件の確認
        if self.steps >= self.max_steps:
            self.done = True
            self.info['timeout'] = True
            
        # ゴール到達チェック
        if isinstance(self.state_dim, list) and len(self.state_dim) >= 2:
            if self.agent_pos == self.goal_pos:
                self.done = True
                self.info['success'] = True
                
        # 報酬計算
        reward = self._calculate_reward()
        
        # ノイズの追加
        self._add_noise()
                
        return self._get_observation(), reward, self.done, self.info
        
    def _discrete_step(self, action):
        """離散行動空間での行動実行"""
        if isinstance(self.state_dim, list) and len(self.state_dim) >= 2:
            # 2D環境での移動
            if action == 0:  # 上
                self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
            elif action == 1:  # 右
                self.agent_pos[1] = min(self.state_dim[1] - 1, self.agent_pos[1] + 1)
            elif action == 2:  # 下
                self.agent_pos[0] = min(self.state_dim[0] - 1, self.agent_pos[0] + 1)
            elif action == 3:  # 左
                self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        else:
            # 1D環境での移動
            if action == 0:  # 左
                self.state = max(0, self.state - 1)
            elif action == 1:  # 右
                self.state = min(self.state_dim - 1, self.state + 1)
    
    def _continuous_step(self, action):
        """連続行動空間での行動実行"""
        # 連続空間での行動の実装（具体的な実装はタスクによる）
        if isinstance(self.state_dim, list):
            # 多次元状態の更新
            self.state += action * 0.1  # 簡易的な実装
            self.state = np.clip(self.state, -1.0, 1.0)
        else:
            # 1次元状態の更新
            self.state += action[0] * 0.1
            self.state = max(0, min(self.state_dim - 1, self.state))
    
    def _calculate_reward(self):
        """報酬の計算"""
        reward = 0.0
        
        # タスク完了報酬
        if self.done and self.info.get('success', False):
            reward += 10.0
            
        # タスク失敗ペナルティ
        if self.done and not self.info.get('success', False):
            reward -= 1.0
            
        # ステップごとの小さなペナルティ（より早く解くよう促進）
        reward -= 0.01
        
        # 目標への接近報酬（2D環境の場合）
        if isinstance(self.state_dim, list) and len(self.state_dim) >= 2:
            prev_dist = np.linalg.norm(np.array([self.agent_pos[0], self.agent_pos[1]]) - 
                                       np.array(self.goal_pos))
            
            if prev_dist < self.prev_distance:
                reward += 0.1
            
            self.prev_distance = prev_dist
            
        return reward * self.reward_scale
    
    def _add_noise(self):
        """環境ノイズの追加"""
        if self.noise_level > 0:
            if isinstance(self.state_dim, list):
                noise = np.random.normal(0, self.noise_level, self.state.shape)
                self.state += noise
            else:
                self.state += np.random.normal(0, self.noise_level)
    
    def _get_observation(self):
        """現在の観測の取得"""
        if isinstance(self.state_dim, list) and len(self.state_dim) >= 2:
            obs = np.zeros(self.state_dim)
            # エージェント位置
            obs[self.agent_pos[0], self.agent_pos[1]] = 0.5
            # ゴール位置
            obs[self.goal_pos[0], self.goal_pos[1]] = 1.0
            return obs
        else:
            return np.array([self.state])
            
    def render(self, mode='human'):
        """環境の視覚化（デバッグ用）"""
        if mode == 'human':
            if isinstance(self.state_dim, list) and len(self.state_dim) >= 2:
                grid = np.zeros(self.state_dim)
                grid[self.agent_pos[0], self.agent_pos[1]] = 2  # エージェント
                grid[self.goal_pos[0], self.goal_pos[1]] = 1    # ゴール
                
                print("\n" + "-" * (self.state_dim[1] + 2))
                for i in range(self.state_dim[0]):
                    line = "|"
                    for j in range(self.state_dim[1]):
                        if grid[i, j] == 0:
                            line += " "
                        elif grid[i, j] == 1:
                            line += "G"
                        elif grid[i, j] == 2:
                            line += "A"
                    line += "|"
                    print(line)
                print("-" * (self.state_dim[1] + 2))
                print(f"Steps: {self.steps}/{self.max_steps}")
            else:
                # 1次元環境の視覚化
                line = "-" * self.state_dim
                pos = int(self.state)
                line = line[:pos] + "A" + line[pos+1:]
                print(line)
                print(f"Position: {pos}/{self.state_dim-1}, Steps: {self.steps}/{self.max_steps}")


class RLTestEnvironment:
    """強化学習テスト環境クラス"""
    
    def __init__(self, config=None):
        """
        テスト環境の初期化
        
        Args:
            config (dict, optional): 環境設定
                デフォルトはシンプルなグリッドワールド設定
        """
        if config is None:
            config = {
                'dimensions': [10, 10],
                'action_space': {
                    'type': 'discrete',
                    'n': 4
                },
                'max_steps': 100,
                'reward_scale': 1.0,
                'noise_level': 0.05
            }
            
        self.logger = Logger(__name__)
        self.simulator = SimulationEnvironment(config)
        self.metrics = MetricsCollector(['reward', 'completion_rate', 'efficiency', 'stability'])
        self.results_dir = os.path.join(os.path.dirname(__file__), 'results')
        
        # 結果保存ディレクトリの作成
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        # ベースラインの確立
        self.baseline = self._establish_baseline()
        
        self.logger.info("RLTestEnvironment initialized successfully")
        self.logger.info(f"Baseline performance: {self.baseline}")
        
    def _establish_baseline(self):
        """ベースラインパフォーマンスの測定"""
        self.logger.info("Establishing baseline performance...")
        
        # ランダム行動によるベースライン
        n_episodes = 100
        rewards = []
        completion = []
        steps = []
        
        for _ in range(n_episodes):
            obs = self.simulator.reset()
            done = False
            total_reward = 0
            episode_steps = 0
            
            while not done:
                # ランダム行動
                action = np.random.randint(0, self.simulator.action_space['n'])
                obs, reward, done, info = self.simulator.step(action)
                total_reward += reward
                episode_steps += 1
            
            rewards.append(total_reward)
            completion.append(1 if info.get('success', False) else 0)
            steps.append(episode_steps)
        
        # ベースラインメトリクスの計算
        baseline = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'completion_rate': np.mean(completion),
            'mean_steps': np.mean(steps),
            'timestamp': datetime.now().isoformat()
        }
        
        # ベースラインの保存
        with open(os.path.join(self.results_dir, 'baseline.json'), 'w') as f:
            json.dump(baseline, f, indent=2)
            
        return baseline
    
    def run_test_episode(self, agent, render=False):
        """
        テストエピソードの実行
        
        Args:
            agent: テスト対象のRLエージェント
            render (bool): 環境の視覚化を行うか
            
        Returns:
            dict: エピソードの結果メトリクス
        """
        obs = self.simulator.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # エージェントの行動選択
            action = agent.select_action(obs)
            next_obs, reward, done, info = self.simulator.step(action)
            
            # 行動後の更新
            agent.update(obs, action, reward, next_obs, done)
            
            if render:
                self.simulator.render()
                
            obs = next_obs
            total_reward += reward
            steps += 1
        
        # エピソード結果の収集
        success = info.get('success', False)
        timeout = info.get('timeout', False)
        
        result = {
            'total_reward': total_reward,
            'steps': steps,
            'success': success,
            'timeout': timeout,
            'efficiency': 0 if not success else 1.0 / steps
        }
        
        # メトリクス更新
        self.metrics.update(
            reward=total_reward,
            completion_rate=1 if success else 0,
            efficiency=result['efficiency'],
            stability=1 if not timeout else 0
        )
        
        return result
    
    def evaluate_agent(self, agent, n_episodes=50, render_interval=0):
        """
        エージェントの総合評価
        
        Args:
            agent: 評価対象のRLエージェント
            n_episodes (int): 評価エピソード数
            render_interval (int): 視覚化の間隔（0なら視覚化なし）
            
        Returns:
            dict: 評価結果
        """
        self.logger.info(f"Starting evaluation of agent over {n_episodes} episodes")
        
        all_results = []
        for i in range(n_episodes):
            render = render_interval > 0 and i % render_interval == 0
            result = self.run_test_episode(agent, render)
            all_results.append(result)
            
            if i % 10 == 0:
                self.logger.info(f"Completed {i}/{n_episodes} evaluation episodes")
        
        # 集計結果の計算
        rewards = [r['total_reward'] for r in all_results]
        success_rate = np.mean([1 if r['success'] else 0 for r in all_results])
        avg_steps = np.mean([r['steps'] for r in all_results])
        
        evaluation = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'success_rate': success_rate,
            'average_steps': avg_steps,
            'baseline_comparison': {
                'reward_improvement': np.mean(rewards) - self.baseline['mean_reward'],
                'success_improvement': success_rate - self.baseline['completion_rate'],
                'steps_improvement': self.baseline['mean_steps'] - avg_steps
            },
            'timestamp': datetime.now().isoformat(),
            'episodes': n_episodes
        }
        
        # 結果の保存
        results_file = os.path.join(
            self.results_dir, 
            f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, 'w') as f:
            json.dump(evaluation, f, indent=2)
        
        self.logger.info(f"Evaluation completed, results saved to {results_file}")
        self.logger.info(f"Success rate: {success_rate:.2f}, Mean reward: {np.mean(rewards):.2f}")
        
        return evaluation
    
    def get_metrics(self):
        """現在のメトリクスを取得"""
        return self.metrics.get_metrics()
    
    def reset_metrics(self):
        """メトリクスをリセット"""
        self.metrics.reset()
