"""
強化学習モジュールのエンドツーエンドシナリオテスト

このモジュールは強化学習モジュールの実際のユースケースシナリオに対する
エンドツーエンドテストを実装します。複数の現実的なタスクに対して
LLM-RL連携パイプラインの全体動作を検証します。
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.core.utils.logger import Logger
from src.core.utils.config import ConfigManager
from src.core.llm.engine import LLMEngine
from src.core.autonomy.rl.adapter import RLAdapter
from src.core.autonomy.rl.reward_generator import RewardFunctionGenerator
from src.core.autonomy.rl.state_manager import EnvironmentStateManager
from src.core.autonomy.rl.action_optimizer import ActionOptimizer
from src.core.autonomy.pipeline import AutonomyPipeline

class RLScenarioTest:
    """強化学習シナリオテストクラス"""
    
    def __init__(self, config_path=None):
        """
        テスト環境の初期化
        
        Args:
            config_path (str, optional): 設定ファイルのパス
        """
        self.logger = Logger(__name__)
        self.logger.info("Initializing RL scenario test")
        
        # 設定のロード
        if config_path and os.path.exists(config_path):
            self.config = ConfigManager(config_path).get_config()
        else:
            # デフォルト設定
            self.config = {
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
                    'algorithm': {
                        'name': 'dqn',
                        'learning_rate': 0.001,
                        'gamma': 0.99,
                        'epsilon': 0.1
                    }
                },
                'scenarios': {
                    'resource_optimization': {
                        'max_steps': 100,
                        'threshold': 0.7,
                        'resource_types': ['cpu', 'memory', 'network'],
                        'initial_state': {'cpu': 80, 'memory': 70, 'network': 50}
                    },
                    'dynamic_scheduling': {
                        'max_steps': 200,
                        'n_tasks': 10,
                        'n_resources': 3,
                        'deadline_factor': 1.5
                    },
                    'adaptive_security': {
                        'max_steps': 150,
                        'threat_levels': ['low', 'medium', 'high'],
                        'resources': ['cpu', 'memory', 'network', 'user_experience'],
                        'initial_security': 0.5
                    }
                },
                'test': {
                    'n_episodes': 5,
                    'verbose': True
                }
            }
            
        # コンポーネントの初期化
        self._initialize_components()
        
        # シナリオ環境の初期化
        self.environments = {
            'resource_optimization': ResourceOptimizationEnvironment(self.config['scenarios']['resource_optimization']),
            'dynamic_scheduling': DynamicSchedulingEnvironment(self.config['scenarios']['dynamic_scheduling']),
            'adaptive_security': AdaptiveSecurityEnvironment(self.config['scenarios']['adaptive_security'])
        }
        
        # 結果保存ディレクトリの設定
        self.results_dir = os.path.join(os.path.dirname(__file__), 'results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        self.logger.info("RL scenario test initialized")
        
    def _initialize_components(self):
        """RLコンポーネントの初期化"""
        # LLMエンジンの設定
        if self.config['llm'].get('use_mock', True):
            from unittest.mock import MagicMock
            self.llm_engine = MagicMock(spec=LLMEngine)
            
            # リソース最適化のモック応答
            self.llm_engine.generate_for_resource_optimization = MagicMock(return_value="""
            def optimize_resources(state, goal):
                # 現在のリソース使用量を分析
                high_usage = [resource for resource, value in state.items() 
                            if resource in ['cpu', 'memory', 'network'] and value > 70]
                
                # 最適化アクション（リソースタイプごとに値を減少させる量）
                actions = {}
                for resource in ['cpu', 'memory', 'network']:
                    if resource in high_usage:
                        actions[resource] = -5  # 高使用リソースは大きく削減
                    else:
                        actions[resource] = -1  # その他は小さく削減
                
                return actions
            """)
            
            # スケジューリングのモック応答
            self.llm_engine.generate_for_scheduling = MagicMock(return_value="""
            def schedule_tasks(tasks, resources, current_time):
                # 優先度順にタスクをソート (締切が近いものが優先)
                sorted_tasks = sorted(tasks, key=lambda x: x['deadline'] - current_time)
                
                # リソース割り当て
                allocation = {}
                for resource_id in range(len(resources)):
                    allocation[resource_id] = []
                
                # 最も負荷の少ないリソースにタスクを割り当て
                for task in sorted_tasks:
                    if task['status'] == 'pending':
                        resource_loads = [sum(r['load'] for r in allocation[rid]) for rid in range(len(resources))]
                        min_load_resource = resource_loads.index(min(resource_loads))
                        allocation[min_load_resource].append(task)
                
                return allocation
            """)
            
            # セキュリティ最適化のモック応答
            self.llm_engine.generate_for_security = MagicMock(return_value="""
            def optimize_security(state, threat_level):
                # 脅威レベルに応じたセキュリティレベルの調整
                if threat_level == 'high':
                    security_level = 0.9  # 高いセキュリティ要求
                    user_exp_factor = 0.7  # ユーザー体験の制約
                elif threat_level == 'medium':
                    security_level = 0.7
                    user_exp_factor = 0.8
                else:  # low
                    security_level = 0.5
                    user_exp_factor = 0.9
                
                # リソース配分の計算
                resource_allocation = {
                    'cpu': min(state['cpu'] * 1.2, 100) if security_level > 0.7 else state['cpu'],
                    'memory': min(state['memory'] * 1.1, 100) if security_level > 0.6 else state['memory'],
                    'network': max(state['network'] * 0.9, 20) if security_level > 0.8 else state['network'],
                    'user_experience': max(state['user_experience'] * user_exp_factor, 60)
                }
                
                return resource_allocation
            """)
            
            # 一般的なLLM応答のモック
            self.llm_engine.generate.return_value = "This is a mock LLM response for testing"
        else:
            # 実際のLLMエンジンをロード
            self.llm_engine = LLMEngine(self.config['llm'])
        
        # RL関連コンポーネント
        self.reward_generator = RewardFunctionGenerator(self.llm_engine)
        self.state_manager = EnvironmentStateManager()
        self.action_optimizer = ActionOptimizer(self.config['rl']['algorithm'])
        
        # RLアダプタ
        self.rl_adapter = RLAdapter(
            self.reward_generator,
            self.state_manager,
            self.action_optimizer
        )
        
        # 自律性パイプライン
        self.rl_pipeline = AutonomyPipeline(
            llm_engine=self.llm_engine,
            rl_adapter=self.rl_adapter
        )
    
    def test_resource_optimization_scenario(self):
        """リソース最適化シナリオのテスト"""
        self.logger.info("Starting resource optimization scenario test")
        
        scenario_config = self.config['scenarios']['resource_optimization']
        env = self.environments['resource_optimization']
        
        # パイプラインの初期化
        goal = "Optimize resource usage without affecting user experience"
        initial_state = scenario_config['initial_state'].copy()
        
        # テスト実行
        results = []
        n_episodes = self.config['test'].get('n_episodes', 3)
        threshold = scenario_config.get('threshold', 0.7)
        
        for i in range(n_episodes):
            self.logger.info(f"Running resource optimization episode {i+1}/{n_episodes}")
            
            # 環境のリセット
            state = env.reset(initial_state.copy())
            
            # エピソード実行
            done = False
            steps = 0
            total_reward = 0
            states_history = [state.copy()]
            
            while not done and steps < scenario_config.get('max_steps', 100):
                # RLパイプラインを通じたアクション生成
                action = self.rl_pipeline.decide_action(state, goal)
                
                # 環境でアクションを実行
                next_state, reward, done, info = env.step(action)
                
                # 履歴の記録
                states_history.append(next_state.copy())
                total_reward += reward
                steps += 1
                
                # 状態の更新
                state = next_state.copy()
                
                if self.config['test'].get('verbose', False) and steps % 10 == 0:
                    self.logger.info(f"Step {steps}, State: {state}, Reward: {reward:.2f}")
            
            # エピソード結果
            success = all(v < 70 for k, v in state.items() 
                         if k in scenario_config.get('resource_types', []))
            user_exp_ok = state.get('user_experience', 100) >= threshold * 100
            
            episode_result = {
                'success': success and user_exp_ok,
                'steps': steps,
                'final_state': state,
                'total_reward': total_reward,
                'state_trajectory': states_history
            }
            
            results.append(episode_result)
            self.logger.info(f"Episode {i+1} completed: Success={episode_result['success']}, "
                          f"Reward={total_reward:.2f}, Steps={steps}")
        
        # 結果の集計
        success_rate = sum(1 for r in results if r['success']) / len(results)
        avg_reward = sum(r['total_reward'] for r in results) / len(results)
        avg_steps = sum(r['steps'] for r in results) / len(results)
        
        summary = {
            'scenario': 'resource_optimization',
            'success_rate': success_rate,
            'average_reward': avg_reward,
            'average_steps': avg_steps,
            'n_episodes': n_episodes,
            'timestamp': datetime.now().isoformat()
        }
        
        # 結果の保存
        result_file = os.path.join(
            self.results_dir, 
            f"resource_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Resource optimization test completed: Success rate={success_rate:.2f}, "
                      f"Avg reward={avg_reward:.2f}, Avg steps={avg_steps:.1f}")
        
        return summary
    
    def test_dynamic_scheduling_scenario(self):
        """動的タスクスケジューリングシナリオのテスト"""
        self.logger.info("Starting dynamic scheduling scenario test")
        
        scenario_config = self.config['scenarios']['dynamic_scheduling']
        env = self.environments['dynamic_scheduling']
        
        # パイプラインの初期化
        goal = "Schedule tasks efficiently to meet deadlines with balanced resource utilization"
        
        # テスト実行
        results = []
        n_episodes = self.config['test'].get('n_episodes', 3)
        
        for i in range(n_episodes):
            self.logger.info(f"Running dynamic scheduling episode {i+1}/{n_episodes}")
            
            # 環境のリセット
            state = env.reset(scenario_config.get('n_tasks', 10), scenario_config.get('n_resources', 3))
            
            # エピソード実行
            done = False
            steps = 0
            total_reward = 0
            deadline_met = 0
            total_tasks = len(state['tasks'])
            
            while not done and steps < scenario_config.get('max_steps', 200):
                # スケジューリングアクションの生成
                action = self.rl_pipeline.schedule_tasks(
                    state['tasks'], 
                    state['resources'], 
                    state['current_time']
                )
                
                # 環境でアクションを実行
                next_state, reward, done, info = env.step(action)
                
                # 報酬と状態の記録
                total_reward += reward
                deadline_met += info.get('completed_on_time', 0)
                steps += 1
                
                # 状態の更新
                state = next_state.copy()
                
                if self.config['test'].get('verbose', False) and steps % 20 == 0:
                    self.logger.info(f"Step {steps}, Tasks remaining: {len([t for t in state['tasks'] if t['status'] == 'pending'])}, "
                                  f"Completed: {len([t for t in state['tasks'] if t['status'] == 'completed'])}")
            
            # エピソード結果
            completion_rate = deadline_met / total_tasks if total_tasks > 0 else 0
            
            episode_result = {
                'completion_rate': completion_rate,
                'steps': steps,
                'total_reward': total_reward,
                'deadline_met': deadline_met,
                'total_tasks': total_tasks
            }
            
            results.append(episode_result)
            self.logger.info(f"Episode {i+1} completed: Completion={completion_rate:.2f}, "
                          f"Reward={total_reward:.2f}, Steps={steps}")
        
        # 結果の集計
        avg_completion = sum(r['completion_rate'] for r in results) / len(results)
        avg_reward = sum(r['total_reward'] for r in results) / len(results)
        avg_steps = sum(r['steps'] for r in results) / len(results)
        
        summary = {
            'scenario': 'dynamic_scheduling',
            'average_completion_rate': avg_completion,
            'average_reward': avg_reward,
            'average_steps': avg_steps,
            'n_episodes': n_episodes,
            'timestamp': datetime.now().isoformat()
        }
        
        # 結果の保存
        result_file = os.path.join(
            self.results_dir, 
            f"dynamic_scheduling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Dynamic scheduling test completed: Completion rate={avg_completion:.2f}, "
                      f"Avg reward={avg_reward:.2f}, Avg steps={avg_steps:.1f}")
        
        return summary
    
    def test_adaptive_security_scenario(self):
        """適応的セキュリティシナリオのテスト"""
        self.logger.info("Starting adaptive security scenario test")
        
        scenario_config = self.config['scenarios']['adaptive_security']
        env = self.environments['adaptive_security']
        
        # パイプラインの初期化
        goal = "Maintain optimal security level while adapting to changing threat conditions"
        
        # テスト実行
        results = []
        n_episodes = self.config['test'].get('n_episodes', 3)
        
        for i in range(n_episodes):
            self.logger.info(f"Running adaptive security episode {i+1}/{n_episodes}")
            
            # 環境のリセット
            state = env.reset(security_level=scenario_config.get('initial_security', 0.5))
            
            # エピソード実行
            done = False
            steps = 0
            total_reward = 0
            threat_changes = 0
            security_breaches = 0
            
            while not done and steps < scenario_config.get('max_steps', 150):
                # 現在の脅威レベル
                current_threat = state['threat_level']
                
                # セキュリティアクションの生成
                action = self.rl_pipeline.optimize_security(state, current_threat)
                
                # 環境でアクションを実行
                next_state, reward, done, info = env.step(action)
                
                # 脅威レベル変更の検出
                if next_state['threat_level'] != current_threat:
                    threat_changes += 1
                
                # セキュリティ侵害の記録
                if info.get('security_breach', False):
                    security_breaches += 1
                
                # 報酬と状態の記録
                total_reward += reward
                steps += 1
                
                # 状態の更新
                state = next_state.copy()
                
                if self.config['test'].get('verbose', False) and steps % 15 == 0:
                    self.logger.info(f"Step {steps}, Threat level: {state['threat_level']}, "
                                  f"Security score: {state.get('security_score', 0):.2f}, "
                                  f"User exp: {state.get('user_experience', 0):.2f}")
            
            # エピソード結果
            security_score = state.get('security_score', 0)
            user_exp = state.get('user_experience', 0)
            balance = min(security_score, user_exp) / max(security_score, user_exp) if max(security_score, user_exp) > 0 else 0
            
            episode_result = {
                'security_score': security_score,
                'user_experience': user_exp,
                'balance': balance,
                'threat_changes': threat_changes,
                'security_breaches': security_breaches,
                'steps': steps,
                'total_reward': total_reward
            }
            
            results.append(episode_result)
            self.logger.info(f"Episode {i+1} completed: Security={security_score:.2f}, "
                          f"User exp={user_exp:.2f}, Balance={balance:.2f}, "
                          f"Breaches={security_breaches}")
        
        # 結果の集計
        avg_security = sum(r['security_score'] for r in results) / len(results)
        avg_user_exp = sum(r['user_experience'] for r in results) / len(results)
        avg_balance = sum(r['balance'] for r in results) / len(results)
        avg_breaches = sum(r['security_breaches'] for r in results) / len(results)
        
        summary = {
            'scenario': 'adaptive_security',
            'average_security_score': avg_security,
            'average_user_experience': avg_user_exp,
            'average_balance': avg_balance,
            'average_security_breaches': avg_breaches,
            'n_episodes': n_episodes,
            'timestamp': datetime.now().isoformat()
        }
        
        # 結果の保存
        result_file = os.path.join(
            self.results_dir, 
            f"adaptive_security_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Adaptive security test completed: Security={avg_security:.2f}, "
                      f"User exp={avg_user_exp:.2f}, Balance={avg_balance:.2f}")
        
        return summary
    
    def run_all_tests(self):
        """すべてのシナリオテストを実行"""
        self.logger.info("Running all scenario tests")
        
        results = {
            'resource_optimization': self.test_resource_optimization_scenario(),
            'dynamic_scheduling': self.test_dynamic_scheduling_scenario(),
            'adaptive_security': self.test_adaptive_security_scenario()
        }
        
        # 総合結果の保存
        summary_file = os.path.join(
            self.results_dir, 
            f"all_scenarios_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"All scenario tests completed, summary saved to {summary_file}")
        return results


class ResourceOptimizationEnvironment:
    """リソース最適化シナリオのシミュレーション環境"""
    
    def __init__(self, config):
        """
        環境の初期化
        
        Args:
            config (dict): 環境設定
        """
        self.logger = Logger(__name__)
        self.config = config
        self.resource_types = config.get('resource_types', ['cpu', 'memory', 'network'])
        self.max_steps = config.get('max_steps', 100)
        self.threshold = config.get('threshold', 0.7)  # ユーザー体験の最低しきい値
        
        # 状態の初期化
        self.state = None
        self.steps = 0
        
    def reset(self, initial_state=None):
        """環境の初期化"""
        if initial_state is None:
            # デフォルト初期状態
            self.state = {resource: np.random.randint(50, 90) for resource in self.resource_types}
            self.state['user_experience'] = 100.0  # 最大値からスタート
        else:
            self.state = initial_state.copy()
            if 'user_experience' not in self.state:
                self.state['user_experience'] = 100.0
        
        self.steps = 0
        return self.state.copy()
    
    def step(self, action):
        """
        環境を一歩進める
        
        Args:
            action (dict): 各リソースに対する調整値
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        self.steps += 1
        
        # アクションの適用
        next_state = self.state.copy()
        for resource, adjustment in action.items():
            if resource in self.state:
                next_state[resource] = max(0, min(100, self.state[resource] + adjustment))
        
        # ユーザー体験への影響
        # リソース削減によるユーザー体験の低下
        user_exp_impact = 0
        for resource in self.resource_types:
            if resource in action and action[resource] < 0:
                # 高いリソース使用率からの削減は体験へのネガティブ影響が少ない
                if self.state[resource] > 70:
                    impact_factor = 0.1
                else:
                    impact_factor = 0.3
                user_exp_impact -= abs(action[resource]) * impact_factor
                
        next_state['user_experience'] = max(0, min(100, next_state['user_experience'] + user_exp_impact))
        
        # 報酬の計算
        reward = 0
        
        # リソース使用率の改善に対する報酬
        for resource in self.resource_types:
            if self.state[resource] > 70 and next_state[resource] < self.state[resource]:
                # 高使用率リソースの削減に報酬
                reward += (self.state[resource] - next_state[resource]) * 0.1
            elif self.state[resource] < 30 and next_state[resource] > self.state[resource]:
                # 低使用率リソースの増加に小報酬（無駄なリソースの有効活用）
                reward += (next_state[resource] - self.state[resource]) * 0.05
        
        # ユーザー体験維持の報酬
        if next_state['user_experience'] >= self.threshold * 100:
            reward += 0.5
        else:
            # しきい値を下回るとペナルティ
            reward -= (self.threshold * 100 - next_state['user_experience']) * 0.1
        
        # 最適化目標達成の報酬
        high_usage_resources = [r for r in self.resource_types if next_state[r] > 70]
        if len(high_usage_resources) == 0 and next_state['user_experience'] >= self.threshold * 100:
            reward += 10.0  # 大きな報酬
            done = True
        else:
            done = False
        
        # ステップ数制限
        if self.steps >= self.max_steps:
            done = True
        
        # 状態の更新
        self.state = next_state.copy()
        
        # 追加情報
        info = {
            'high_usage_resources': high_usage_resources,
            'user_experience_ok': next_state['user_experience'] >= self.threshold * 100
        }
        
        return next_state.copy(), reward, done, info


class DynamicSchedulingEnvironment:
    """動的タスクスケジューリングのシミュレーション環境"""
    
    def __init__(self, config):
        """
        環境の初期化
        
        Args:
            config (dict): 環境設定
        """
        self.logger = Logger(__name__)
        self.config = config
        self.max_steps = config.get('max_steps', 200)
        self.n_tasks = config.get('n_tasks', 10)
        self.n_resources = config.get('n_resources', 3)
        self.deadline_factor = config.get('deadline_factor', 1.5)
        
        # 状態の初期化
        self.state = None
        self.steps = 0
        
    def reset(self, n_tasks=None, n_resources=None):
        """環境の初期化"""
        self.steps = 0
        
        if n_tasks is not None:
            self.n_tasks = n_tasks
        if n_resources is not None:
            self.n_resources = n_resources
        
        # タスク生成
        tasks = []
        for i in range(self.n_tasks):
            # タスク特性をランダム生成
            duration = np.random.randint(5, 20)
            complexity = np.random.uniform(0.5, 2.0)
            priority = np.random.randint(1, 5)
            
            # 締め切りの設定（現在時刻からduration * deadline_factorの範囲で）
            deadline = int(duration * self.deadline_factor * (0.8 + 0.4 * np.random.random()))
            
            task = {
                'id': i,
                'duration': duration,
                'complexity': complexity,
                'priority': priority,
                'deadline': deadline,
                'status': 'pending',
                'progress': 0,
                'resource_assigned': None,
                'start_time': None
            }
            tasks.append(task)
        
        # リソース生成
        resources = []
        for i in range(self.n_resources):
            # リソース特性をランダム生成
            capacity = np.random.uniform(0.8, 1.2)
            efficiency = np.random.uniform(0.7, 1.3)
            
            resource = {
                'id': i,
                'capacity': capacity,
                'efficiency': efficiency,
                'load': [],  # 現在のタスク負荷
                'history': []  # 処理履歴
            }
            resources.append(resource)
        
        # 初期状態の設定
        self.state = {
            'tasks': tasks,
            'resources': resources,
            'current_time': 0,
            'completed_tasks': []
        }
        
        return self.state.copy()
    
    def step(self, action):
        """
        環境を一歩進める
        
        Args:
            action (dict): リソースID -> タスクリストのマッピング
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        self.steps += 1
        next_state = self.state.copy()
        next_state['tasks'] = [task.copy() for task in self.state['tasks']]
        next_state['resources'] = [resource.copy() for resource in self.state['resources']]
        next_state['current_time'] = self.state['current_time'] + 1
        
        # アクションの適用（タスクのスケジュール）
        for resource_id, task_list in action.items():
            if resource_id < len(next_state['resources']):
                resource = next_state['resources'][resource_id]
                
                # リソースのロード更新
                resource['load'] = []
                
                # タスクの割り当てと進捗更新
                for task_id in task_list:
                    # タスクの取得
                    task_idx = next(i for i, t in enumerate(next_state['tasks']) 
                                  if t['id'] == task_id and t['status'] == 'pending')
                    task = next_state['tasks'][task_idx]
                    
                    # タスクの開始時刻設定
                    if task['start_time'] is None:
                        task['start_time'] = next_state['current_time']
                    
                    # タスクの進捗更新
                    progress_rate = resource['efficiency'] / task['complexity']
                    task['progress'] += progress_rate
                    task['resource_assigned'] = resource_id
                    
                    # リソース負荷に追加
                    resource['load'].append({
                        'task_id': task['id'],
                        'load': 1.0  # 単純化のため一律負荷
                    })
                    
                    # タスク完了判定
                    if task['progress'] >= task['duration']:
                        task['status'] = 'completed'
                        task['completion_time'] = next_state['current_time']
                        next_state['completed_tasks'].append(task.copy())
                        
                        # 処理履歴に追加
                        resource['history'].append({
                            'task_id': task['id'],
                            'start_time': task['start_time'],
                            'completion_time': task['completion_time']
                        })
        
        # 報酬の計算
        reward = 0
        completed_on_time = 0
        completed_late = 0
        
        # 今回完了したタスクのチェック
        newly_completed = [t for t in next_state['tasks'] 
                         if t['status'] == 'completed' and 
                         t['completion_time'] == next_state['current_time']]
        
        for task in newly_completed:
            # 締め切り前に完了で大報酬
            if task['completion_time'] <= task['deadline']:
                reward += 5.0 * task['priority']
                completed_on_time += 1
            else:
                # 締め切り超過でペナルティ（超過時間に比例）
                lateness = task['completion_time'] - task['deadline']
                reward -= 2.0 * lateness * task['priority']
                completed_late += 1
        
        # リソース使用効率の報酬
        total_capacity = sum(r['capacity'] for r in next_state['resources'])
        used_capacity = sum(len(r['load']) for r in next_state['resources'])
        if total_capacity > 0:
            efficiency = used_capacity / total_capacity
            # 高効率（0.7-0.9）が最適、過負荷や低負荷は非効率
            if 0.7 <= efficiency <= 0.9:
                reward += 1.0
            elif efficiency > 0.9:
                reward -= (efficiency - 0.9) * 5.0  # 過負荷ペナルティ
            else:
                reward -= (0.7 - efficiency) * 2.0  # 低負荷ペナルティ
        
        # 終了条件：すべてのタスクが完了または最大ステップ数に到達
        pending_tasks = [t for t in next_state['tasks'] if t['status'] == 'pending']
        if len(pending_tasks) == 0 or self.steps >= self.max_steps:
            done = True
        else:
            done = False
        
        # 情報の設定
        info = {
            'completed_on_time': completed_on_time,
            'completed_late': completed_late,
            'remaining_tasks': len(pending_tasks),
            'resource_efficiency': efficiency if 'efficiency' in locals() else 0
        }
        
        # 状態の更新
        self.state = next_state
        
        return next_state, reward, done, info


class AdaptiveSecurityEnvironment:
    """適応的セキュリティのシミュレーション環境"""
    
    def __init__(self, config):
        """
        環境の初期化
        
        Args:
            config (dict): 環境設定
        """
        self.logger = Logger(__name__)
        self.config = config
        self.max_steps = config.get('max_steps', 150)
        self.threat_levels = config.get('threat_levels', ['low', 'medium', 'high'])
        self.resources = config.get('resources', ['cpu', 'memory', 'network', 'user_experience'])
        
        # 状態の初期化
        self.state = None
        self.steps = 0
        
        # 脅威シナリオの定義
        self.threat_scenarios = {
            'low': {
                'breach_prob': 0.05,
                'impact': 0.2,
                'resource_demands': {
                    'cpu': 0.3,
                    'memory': 0.3,
                    'network': 0.2
                }
            },
            'medium': {
                'breach_prob': 0.15,
                'impact': 0.5,
                'resource_demands': {
                    'cpu': 0.5,
                    'memory': 0.5,
                    'network': 0.4
                }
            },
            'high': {
                'breach_prob': 0.3,
                'impact': 0.8,
                'resource_demands': {
                    'cpu': 0.8,
                    'memory': 0.7,
                    'network': 0.6
                }
            }
        }
        
    def reset(self, security_level=0.5):
        """環境の初期化"""
        self.steps = 0
        
        # 初期状態の設定
        self.state = {
            # リソース初期状態
            'cpu': 50.0,
            'memory': 50.0,
            'network': 50.0,
            'user_experience': 100.0,
            
            # セキュリティパラメータ
            'security_level': security_level,
            'threat_level': 'low',  # 初期脅威レベル
            'security_score': self._calculate_security_score(security_level, 'low'),
            'breach_history': []
        }
        
        return self.state.copy()
    
    def step(self, action):
        """
        環境を一歩進める
        
        Args:
            action (dict): リソース割り当てと設定
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        self.steps += 1
        next_state = self.state.copy()
        
        # アクションの適用
        for resource, value in action.items():
            if resource in self.resources:
                next_state[resource] = max(0, min(100, value))
        
        # 脅威レベルの更新（一定確率で変化）
        if np.random.random() < 0.1:  # 10%の確率で変化
            next_threat = np.random.choice(self.threat_levels)
            next_state['threat_level'] = next_threat
            self.logger.info(f"Threat level changed to {next_threat}")
        
        # 現在の脅威レベルとセキュリティ設定に基づくセキュリティスコアの計算
        threat_level = next_state['threat_level']
        security_score = self._calculate_security_score(
            next_state['security_level'], 
            threat_level,
            next_state
        )
        next_state['security_score'] = security_score
        
        # セキュリティ侵害シミュレーション
        threat_info = self.threat_scenarios[threat_level]
        breach_prob = max(0, threat_info['breach_prob'] - security_score/100.0)
        
        security_breach = False
        if np.random.random() < breach_prob:
            security_breach = True
            impact = threat_info['impact']
            
            # セキュリティ侵害の影響
            next_state['user_experience'] *= (1 - impact * 0.5)
            for resource in ['cpu', 'memory', 'network']:
                next_state[resource] *= (1 + impact * 0.3)  # リソース使用量増加
            
            # 侵害履歴の記録
            next_state['breach_history'].append({
                'time': self.steps,
                'threat_level': threat_level,
                'security_score': security_score,
                'impact': impact
            })
            
            self.logger.warning(f"Security breach occurred at step {self.steps} with impact {impact:.2f}")
        
        # 報酬の計算
        reward = 0
        
        # セキュリティスコアに基づく報酬
        reward += security_score * 0.05
        
        # ユーザー体験の維持報酬
        user_exp = next_state['user_experience']
        reward += user_exp * 0.01
        
        # セキュリティと体験のバランス報酬
        balance = min(security_score, user_exp) / max(security_score, user_exp) if max(security_score, user_exp) > 0 else 0
        reward += balance * 3.0
        
        # セキュリティ侵害のペナルティ
        if security_breach:
            reward -= 20.0 * threat_info['impact']
        
        # リソース効率の報酬/ペナルティ
        threat_demands = threat_info['resource_demands']
        for resource, demand in threat_demands.items():
            required = demand * 100
            actual = next_state[resource]
            
            # 必要量に近いリソース配分に報酬
            if abs(actual - required) < 10:
                reward += 1.0
            elif actual < required * 0.8:
                # 著しく不足している場合のペナルティ
                reward -= (required * 0.8 - actual) * 0.1
            elif actual > required * 1.2:
                # 著しく過剰な場合のペナルティ（無駄）
                reward -= (actual - required * 1.2) * 0.05
        
        # 終了条件：最大ステップ数に到達または重大なセキュリティ侵害
        critical_breach = (security_breach and threat_level == 'high' and security_score < 40)
        if self.steps >= self.max_steps or critical_breach:
            done = True
        else:
            done = False
        
        # 情報の設定
        info = {
            'security_breach': security_breach,
            'security_score': security_score,
            'user_experience': user_exp,
            'balance_score': balance,
            'critical_breach': critical_breach
        }
        
        # 状態の更新
        self.state = next_state
        
        return next_state, reward, done, info
    
    def _calculate_security_score(self, security_level, threat_level, state=None):
        """
        セキュリティスコアの計算
        
        Args:
            security_level (float): 基本セキュリティレベル設定（0-1）
            threat_level (str): 現在の脅威レベル
            state (dict, optional): 現在の状態
            
        Returns:
            float: セキュリティスコア（0-100）
        """
        # 基本スコア
        base_score = security_level * 100
        
        # 脅威レベルに基づく要求スコア
        if threat_level == 'high':
            required_score = 80
        elif threat_level == 'medium':
            required_score = 60
        else:  # low
            required_score = 40
        
        # リソース配分に基づくスコア調整
        resource_score = 0
        if state is not None:
            threat_demands = self.threat_scenarios[threat_level]['resource_demands']
            for resource, demand in threat_demands.items():
                required = demand * 100
                actual = state.get(resource, 0)
                
                # リソース充足率の計算
                fulfillment = min(1.0, actual / required if required > 0 else 1.0)
                resource_score += fulfillment * 100 / len(threat_demands)
        
        # 最終スコアの計算（基本スコアとリソーススコアの加重平均）
        if state is not None:
            final_score = 0.3 * base_score + 0.7 * resource_score
        else:
            # リソース情報がない場合は基本スコアのみ
            final_score = base_score
            
        # 必要スコアに対する達成度によるボーナス/ペナルティ
        achievement = final_score / required_score if required_score > 0 else 1.0
        if achievement >= 1.0:
            # 要求を満たしている場合はボーナス
            final_score *= min(1.2, 1.0 + (achievement - 1.0) * 0.5)
        else:
            # 要求を満たしていない場合はペナルティ
            final_score *= max(0.5, achievement)
        
        return min(100, max(0, final_score))


if __name__ == "__main__":
    # 設定ファイルのパス
    config_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../../../config/test_config.json'
    ))
    
    # テストの実行
    test = RLScenarioTest(config_path)
    results = test.run_all_tests()
    
    print("\n=== Scenario Test Results ===")
    for scenario, result in results.items():
        print(f"\n{scenario.replace('_', ' ').title()}:")
        for key, value in result.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            elif key != 'timestamp':
                print(f"  {key}: {value}")
