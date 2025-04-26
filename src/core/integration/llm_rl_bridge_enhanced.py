"""
LLMと強化学習（RL）の拡張連携ブリッジ

このモジュールは、LLM（大規模言語モデル）と強化学習（RL）モジュール間の
高度な連携を実現するための拡張ブリッジ機能を提供します。
基本的なllm_rl_bridgeを拡張し、より柔軟で効率的な連携を可能にします。

主な拡張機能:
1. 高度なコンテキスト管理
2. 動的報酬関数生成
3. 状態表現の自動最適化
4. マルチモーダル入力の統合
5. 説明可能な行動選択
"""

import json
import logging
import numpy as np
from typing import Dict, Any, List, Callable, Optional, Tuple, Union

from src.core.llm.engine import LLMEngine
from src.core.integration.llm_rl_bridge import LLMRLBridge
from src.core.integration.adapters.reinforcement_learning.adapter import RLAdapter
from src.core.integration.adapters.reinforcement_learning.environment import Environment
from src.core.integration.adapters.reinforcement_learning.action import Action
from src.core.integration.adapters.reinforcement_learning.reward import RewardFunction
from src.core.utils.logger import get_logger
from src.core.utils.event_bus import EventBus
from src.core.llm.context import ContextManager

logger = get_logger(__name__)

class EnhancedLLMRLBridge(LLMRLBridge):
    """
    LLMと強化学習を連携させるための拡張ブリッジクラス
    
    基本的なLLMRLBridgeを拡張し、より高度な連携機能を提供します。
    コンテキスト認識、説明可能なポリシー、マルチモーダル入力などの
    機能を強化しています。
    """
    
    def __init__(
        self, 
        llm_engine: LLMEngine, 
        rl_adapter: RLAdapter,
        context_manager: ContextManager,
        event_bus: Optional[EventBus] = None
    ):
        """
        拡張LLM-RL連携ブリッジの初期化
        
        Args:
            llm_engine: LLMエンジンインスタンス
            rl_adapter: 強化学習アダプタインスタンス
            context_manager: コンテキスト管理インスタンス
            event_bus: イベントバスインスタンス（オプション）
        """
        super().__init__(llm_engine, rl_adapter, event_bus)
        self.context_manager = context_manager
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.explanation_cache = {}
        
    def generate_reward_function(
        self, 
        goal_description: str, 
        constraints: List[str] = None,
        context_id: Optional[str] = None
    ) -> RewardFunction:
        """
        自然言語の目標記述から強化学習用の報酬関数を生成
        
        基本実装を拡張し、制約条件とコンテキスト情報を考慮します。
        
        Args:
            goal_description: 自然言語での目標記述
            constraints: 制約条件のリスト（オプション）
            context_id: コンテキストID（オプション）
            
        Returns:
            生成された報酬関数
        """
        # コンテキスト情報の取得
        context = {}
        if context_id and self.context_manager:
            context = self.context_manager.get_context(context_id)
        
        # 制約条件の処理
        constraints_text = ""
        if constraints and len(constraints) > 0:
            constraints_text = "制約条件:\n" + "\n".join([f"- {c}" for c in constraints])
        
        # 報酬関数生成のためのプロンプト
        prompt = f"""
あなたの役割は、強化学習エージェントのための報酬関数を設計することです。
以下の目標に基づいて、状態と行動から報酬値を計算する関数を生成してください。

# 目標
{goal_description}

{constraints_text}

# 入力情報
- state: エージェントの現在の状態を表す辞書
- action: エージェントが選択した行動を表すオブジェクト
- done: エピソードが終了したかを示すブール値

# 報酬関数の要件
- 目標達成を促進する行動に正の報酬を与えること
- 制約に違反する行動にはペナルティを与えること
- 短期的な利益より長期的な目標達成を優先すること
- 報酬値の範囲は通常 -10 から 10 の間に収めること

# 出力形式
pythonの関数として報酬計算ロジックを記述してください。関数定義は除き、
関数本体のみを記述してください。

例：
```
reward = 0
if state['progress'] > 0.5:
    reward += 2
if action.type == 'risky' and state['stability'] < 0.3:
    reward -= 5
if done and state['goal_achieved']:
    reward += 10
return reward
```
        """
        
        # LLMを使用して報酬関数のロジックを生成
        response = self.llm_engine.generate(prompt, temperature=0.2)
        
        # コード部分を抽出
        import re
        code_match = re.search(r'```(?:python)?(.*?)```', response, re.DOTALL)
        if code_match:
            reward_logic = code_match.group(1).strip()
        else:
            # コードブロックがない場合は全体を使用
            reward_logic = response.strip()
        
        # 報酬関数の組み立てと安全性チェック
        reward_function = self._build_safe_reward_function(reward_logic)
        
        # イベント通知
        if self.event_bus:
            self.event_bus.publish(
                "llm_rl_bridge.reward_function_generated",
                {
                    "goal": goal_description,
                    "constraints": constraints,
                    "reward_logic": reward_logic
                }
            )
        
        return reward_function
    
    def _build_safe_reward_function(self, reward_logic: str) -> RewardFunction:
        """
        安全な報酬関数を構築する
        
        Args:
            reward_logic: 報酬計算ロジック（文字列のPythonコード）
            
        Returns:
            実行可能な報酬関数
        """
        # 安全な名前空間
        safe_globals = {
            'np': np,
            'min': min,
            'max': max,
            'abs': abs,
            'sum': sum,
            'len': len
        }
        
        # 報酬関数の構築
        reward_code = f"""
def reward_function(state, action, done=False):
    try:
        {reward_logic}
    except Exception as e:
        logger.error(f"Error in reward function: {{e}}")
        return 0.0
"""
        
        try:
            # 実行環境の準備
            local_vars = {}
            exec(reward_code, safe_globals, local_vars)
            reward_function = RewardFunction(local_vars['reward_function'])
            
            # 簡単なテスト
            test_state = {'test': True}
            test_action = Action('test', {})
            test_reward = reward_function(test_state, test_action)
            logger.debug(f"Test reward calculation: {test_reward}")
            
            return reward_function
            
        except Exception as e:
            logger.error(f"Failed to build reward function: {e}")
            # フォールバック報酬関数
            def fallback_reward(state, action, done=False):
                return 0.0
            
            return RewardFunction(fallback_reward)
    
    def interpret_state(
        self,
        environment: Environment,
        state: Dict[str, Any],
        context_id: Optional[str] = None
    ) -> str:
        """
        環境状態の自然言語解釈を生成
        
        Args:
            environment: 環境インスタンス
            state: 現在の状態
            context_id: コンテキストID（オプション）
            
        Returns:
            状態の自然言語による解釈
        """
        # 環境メタデータの取得
        env_meta = environment.get_metadata()
        
        # コンテキスト情報の取得
        context = {}
        if context_id and self.context_manager:
            context = self.context_manager.get_context(context_id)
        
        # 状態解釈のためのプロンプト
        prompt = f"""
現在の環境状態を自然言語で説明してください。

# 環境情報
環境名: {env_meta.get('name', 'Unknown')}
環境説明: {env_meta.get('description', 'No description')}

# 現在の状態
```
{json.dumps(state, indent=2)}
```

{f"# コンテキスト情報\n{json.dumps(context, indent=2)}" if context else ""}

# 出力要件
- 状態の主要な特徴を簡潔に説明すること
- 重要なメトリクスや状態変数の値を含めること
- 環境内での進捗や位置づけを明確にすること
- 300文字程度で簡潔にまとめること
        """
        
        # LLMを使用して状態解釈を生成
        response = self.llm_engine.generate(prompt, temperature=0.3, max_tokens=512)
        
        # 状態履歴の更新
        self.state_history.append({
            'state': state,
            'interpretation': response
        })
        
        return response
    
    def explain_action(
        self,
        environment: Environment,
        state: Dict[str, Any],
        action: Action,
        next_state: Dict[str, Any],
        reward: float,
        context_id: Optional[str] = None
    ) -> str:
        """
        強化学習エージェントの行動選択を説明
        
        Args:
            environment: 環境インスタンス
            state: 元の状態
            action: 実行された行動
            next_state: 行動後の新しい状態
            reward: 受け取った報酬
            context_id: コンテキストID（オプション）
            
        Returns:
            行動選択の説明
        """
        # キャッシュキーの作成
        cache_key = f"{hash(str(state))}-{hash(str(action))}"
        if cache_key in self.explanation_cache:
            return self.explanation_cache[cache_key]
        
        # 環境メタデータの取得
        env_meta = environment.get_metadata()
        
        # コンテキスト情報の取得
        context = {}
        if context_id and self.context_manager:
            context = self.context_manager.get_context(context_id)
        
        # 行動説明のためのプロンプト
        prompt = f"""
強化学習エージェントが選択した行動の理由と結果を説明してください。

# 環境情報
環境名: {env_meta.get('name', 'Unknown')}
環境説明: {env_meta.get('description', 'No description')}

# 行動の詳細
- 元の状態: {json.dumps(state, indent=2)}
- 選択された行動: {action.type} {json.dumps(action.parameters, indent=2)}
- 行動後の状態: {json.dumps(next_state, indent=2)}
- 受け取った報酬: {reward}

{f"# コンテキスト情報\n{json.dumps(context, indent=2)}" if context else ""}

# 出力要件
- なぜこの行動が選択されたと考えられるかを説明すること
- 行動がどのような変化をもたらしたかを説明すること
- 報酬の意味を解釈すること
- 人間にとって理解しやすい自然な説明であること
- 200文字程度で簡潔にまとめること
        """
        
        # LLMを使用して行動説明を生成
        response = self.llm_engine.generate(prompt, temperature=0.3, max_tokens=512)
        
        # 行動履歴の更新
        self.action_history.append({
            'state': state,
            'action': action.to_dict(),
            'next_state': next_state,
            'reward': reward,
            'explanation': response
        })
        
        # キャッシュに保存
        self.explanation_cache[cache_key] = response
        
        return response
    
    def optimize_state_representation(
        self,
        raw_state: Dict[str, Any],
        goal_description: str,
        history_length: int = 5
    ) -> Dict[str, Any]:
        """
        強化学習に最適化された状態表現を生成
        
        Args:
            raw_state: 生の状態データ
            goal_description: 目標の説明
            history_length: 考慮する履歴の長さ
            
        Returns:
            最適化された状態表現
        """
        # 状態履歴の取得（最新のN個）
        recent_history = self.state_history[-history_length:] if len(self.state_history) > 0 else []
        
        # 状態最適化のためのプロンプト
        prompt = f"""
強化学習エージェントのために生の状態データから最適化された状態表現を生成してください。

# 目標
{goal_description}

# 現在の生の状態
```
{json.dumps(raw_state, indent=2)}
```

{f"# 最近の状態履歴\n```\n{json.dumps([h['state'] for h in recent_history], indent=2)}\n```" if recent_history else ""}

# 要件
- 目標達成に関連する重要な特徴を特定すること
- 不要な情報を削除し、状態空間を単純化すること
- 連続値は適切な範囲に正規化すること
- カテゴリ変数はone-hotエンコーディングすること
- 時間的な傾向を捉える特徴を追加すること

# 出力形式
JSON形式で最適化された状態表現を出力してください。例：
```json
{
  "position_normalized": [0.45, 0.67],
  "velocity_normalized": [0.2, -0.1],
  "distance_to_goal": 0.34,
  "obstacle_present": 1,
  "resource_level": 0.78
}
```
        """
        
        # LLMを使用して最適化された状態表現を生成
        response = self.llm_engine.generate(prompt, temperature=0.2, max_tokens=1024)
        
        # JSONを抽出
        import re
        json_match = re.search(r'```(?:json)?(.*?)```', response, re.DOTALL)
        
        try:
            if json_match:
                optimized_state = json.loads(json_match.group(1).strip())
            else:
                # JSON形式でない場合は解析を試みる
                optimized_state = json.loads(response.strip())
                
            # 最適化された状態の検証
            if not isinstance(optimized_state, dict):
                logger.warning("Optimized state is not a dictionary, falling back to raw state")
                return raw_state
                
            return optimized_state
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse optimized state: {e}")
            return raw_state
    
    def suggest_action_improvements(
        self, 
        environment: Environment,
        episodes_data: List[Dict[str, Any]],
        goal_description: str
    ) -> Dict[str, Any]:
        """
        強化学習エージェントの行動改善を提案
        
        Args:
            environment: 環境インスタンス
            episodes_data: エピソードデータのリスト
            goal_description: 目標の説明
            
        Returns:
            行動改善提案
        """
        # 環境メタデータの取得
        env_meta = environment.get_metadata()
        
        # エピソードサマリーの作成
        episodes_summary = []
        for i, episode in enumerate(episodes_data):
            # 各エピソードの要約情報を抽出
            total_reward = sum(step['reward'] for step in episode['steps'])
            steps_count = len(episode['steps'])
            final_state = episode['steps'][-1]['next_state'] if steps_count > 0 else {}
            success = episode.get('success', False)
            
            episodes_summary.append({
                'episode_id': i,
                'total_reward': total_reward,
                'steps_count': steps_count,
                'final_state': final_state,
                'success': success
            })
        
        # 詳細分析用のサンプルステップ
        sample_steps = []
        if episodes_data and 'steps' in episodes_data[-1]:
            # 最新エピソードから重要なステップをサンプリング
            steps = episodes_data[-1]['steps']
            
            # 重要なステップの選択（最初、最後、高い/低い報酬のステップ）
            if len(steps) > 0:
                sample_steps.append(steps[0])  # 最初のステップ
            
            if len(steps) > 1:
                sample_steps.append(steps[-1])  # 最後のステップ
            
            # 報酬が最も高いステップ
            if len(steps) > 2:
                max_reward_step = max(steps[1:-1], key=lambda x: x['reward'], default=None)
                if max_reward_step:
                    sample_steps.append(max_reward_step)
            
            # 報酬が最も低いステップ
            if len(steps) > 3:
                min_reward_step = min(steps[1:-1], key=lambda x: x['reward'], default=None)
                if min_reward_step:
                    sample_steps.append(min_reward_step)
        
        # 行動改善提案のためのプロンプト
        prompt = f"""
強化学習エージェントの行動パターンを分析し、改善提案を行ってください。

# 目標
{goal_description}

# 環境情報
環境名: {env_meta.get('name', 'Unknown')}
環境説明: {env_meta.get('description', 'No description')}
利用可能な行動: {env_meta.get('available_actions', 'Unknown')}

# エピソードサマリー
```
{json.dumps(episodes_summary, indent=2)}
```

# サンプルステップ詳細
```
{json.dumps(sample_steps, indent=2)}
```

# 分析要件
- エージェントの現在の行動パターンの強みと弱みを特定する
- 目標達成のために改善すべき具体的な行動戦略を提案する
- 報酬が低いケースの原因を分析する
- 探索と活用のバランスに関する提案を行う
- 環境の特性を活かした効率的な戦略を提案する

# 出力形式
以下の項目を含むJSON形式で出力してください：
1. "current_pattern": 現在の行動パターンの分析
2. "strengths": 現在の戦略の強み（リスト）
3. "weaknesses": 現在の戦略の弱み（リスト）
4. "improvement_suggestions": 改善提案（リスト）
5. "priority_actions": 優先的に改善すべき行動（リスト）
        """
        
        # LLMを使用して行動改善提案を生成
        response = self.llm_engine.generate(prompt, temperature=0.3, max_tokens=1536)
        
        # JSONを抽出
        import re
        json_match = re.search(r'```(?:json)?(.*?)```', response, re.DOTALL)
        
        try:
            if json_match:
                improvement_suggestions = json.loads(json_match.group(1).strip())
            else:
                # JSON形式でない場合は解析を試みる
                improvement_suggestions = json.loads(response.strip())
            
            return improvement_suggestions
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse improvement suggestions: {e}")
            # 解析に失敗した場合はテキスト形式で返す
            return {
                "error": "Failed to parse JSON",
                "text_response": response
            }
    
    def generate_curriculum(
        self,
        environment: Environment,
        goal_description: str,
        current_skill_level: float = 0.0,
        target_skill_level: float = 1.0,
        stages_count: int = 5
    ) -> List[Dict[str, Any]]:
        """
        段階的なカリキュラム学習のための目標シーケンスを生成
        
        Args:
            environment: 環境インスタンス
            goal_description: 最終目標の説明
            current_skill_level: 現在のスキルレベル（0.0～1.0）
            target_skill_level: 目標スキルレベル（0.0～1.0）
            stages_count: カリキュラムのステージ数
            
        Returns:
            カリキュラムステージのリスト
        """
        # 環境メタデータの取得
        env_meta = environment.get_metadata()
        
        # カリキュラム生成のためのプロンプト
        prompt = f"""
強化学習エージェントのためのカリキュラム学習シーケンスを設計してください。

# 最終目標
{goal_description}

# 環境情報
環境名: {env_meta.get('name', 'Unknown')}
環境説明: {env_meta.get('description', 'No description')}
利用可能な行動: {env_meta.get('available_actions', 'Unknown')}

# 学習パラメータ
- 現在のスキルレベル: {current_skill_level} (0.0～1.0)
- 目標スキルレベル: {target_skill_level} (0.0～1.0)
- 要求ステージ数: {stages_count}

# 要件
- 簡単なタスクから始めて徐々に難しくしていくこと
- 各ステージは前のステージで学んだスキルを基にすること
- 各ステージには明確な成功基準を設定すること
- 環境パラメータの段階的な調整を含めること
- 報酬関数の調整提案を含めること

# 出力形式
各ステージを含むJSON配列で出力してください。各ステージには以下の情報を含めてください：
1. "stage_id": ステージID（整数）
2. "stage_name": ステージ名（文字列）
3. "description": 簡潔な説明（文字列）
4. "goal": このステージでの具体的な目標（文字列）
5. "success_criteria": 成功基準（文字列）
6. "environment_params": 環境パラメータの調整（オブジェクト）
7. "reward_function_description": 報酬関数の説明（文字列）
8. "estimated_episodes": 学習に必要なエピソード数の推定（整数）
        """
        
        # LLMを使用してカリキュラムを生成
        response = self.llm_engine.generate(prompt, temperature=0.4, max_tokens=2048)
        
        # JSONを抽出
        import re
        json_match = re.search(r'```(?:json)?(.*?)```', response, re.DOTALL)
        
        try:
            if json_match:
                curriculum = json.loads(json_match.group(1).strip())
            else:
                # JSON形式でない場合は解析を試みる
                curriculum = json.loads(response.strip())
            
            # カリキュラムの検証
            if not isinstance(curriculum, list):
                logger.warning("Generated curriculum is not a list, creating fallback curriculum")
                # フォールバックカリキュラム
                curriculum = self._generate_fallback_curriculum(
                    goal_description, 
                    stages_count, 
                    current_skill_level, 
                    target_skill_level
                )
            
            return curriculum
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse curriculum: {e}")
            # 解析に失敗した場合はフォールバックカリキュラムを生成
            return self._generate_fallback_curriculum(
                goal_description, 
                stages_count, 
                current_skill_level, 
                target_skill_level
            )
    
    def _generate_fallback_curriculum(
        self, 
        goal_description: str, 
        stages_count: int, 
        current_skill_level: float, 
        target_skill_level: float
    ) -> List[Dict[str, Any]]:
        """
        フォールバックカリキュラムを生成
        
        Args:
            goal_description: 最終目標の説明
            stages_count: カリキュラムのステージ数
            current_skill_level: 現在のスキルレベル
            target_skill_level: 目標スキルレベル
            
        Returns:
            基本的なカリキュラムステージのリスト
        """
        curriculum = []
        
        # スキルレベルの段階的増加
        skill_step = (target_skill_level - current_skill_level) / stages_count
        
        for i in range(stages_count):
            stage_skill_level = current_skill_level + skill_step * (i + 1)
            stage = {
                "stage_id": i + 1,
                "stage_name": f"Stage {i + 1}",
                "description": f"Learning stage {i + 1} of {stages_count}",
                "goal": f"Progress towards {goal_description} with {int(stage_skill_level * 100)}% proficiency",
                "success_criteria": f"Achieve average reward above {0.5 + i * 0.1} over 10 episodes",
                "environment_params": {
                    "difficulty": (i + 1) / stages_count,
                    "complexity": min(1.0, 0.3 + i * 0.15)
                },
                "reward_function_description": "Basic reward function with increasing emphasis on efficiency",
                "estimated_episodes": 100 + i * 50
            }
            curriculum.append(stage)
        
        return curriculum
    
    def generate_reflection(
        self,
        learning_history: Dict[str, Any],
        goal_description: str
    ) -> Dict[str, Any]:
        """
        学習プロセスの振り返りと洞察を生成
        
        Args:
            learning_history: 学習履歴データ
            goal_description: 目標の説明
            
        Returns:
            振り返りと洞察
        """
        # 学習履歴のサマリー作成
        episodes_count = learning_history.get('episodes_count', 0)
        total_steps = learning_history.get('total_steps', 0)
        success_rate = learning_history.get('success_rate', 0.0)
        reward_trend = learning_history.get('reward_trend', [])
        
        # 重要な学習イベント
        key_events = learning_history.get('key_events', [])
        
        # 振り返り生成のためのプロンプト
        prompt = f"""
強化学習プロセスの振り返りと洞察を生成してください。

# 学習目標
{goal_description}

# 学習履歴サマリー
- エピソード数: {episodes_count}
- 総ステップ数: {total_steps}
- 成功率: {success_rate}
- 報酬トレンド: {reward_trend}

# 重要な学習イベント
```
{json.dumps(key_events, indent=2)}
```

# 要件
- 学習プロセスの成功要因と障壁を分析すること
- 学習効率に関する洞察を提供すること
- エージェントの能力の成長パターンを特定すること
- 将来の改善のための具体的な提案を行うこと
- 学習から得られた一般的な教訓を抽出すること

# 出力形式
以下の項目を含むJSON形式で出力してください：
1. "summary": 全体の振り返り（文字列）
2. "key_insights": 主要な洞察（リスト）
3. "learning_patterns": 学習パターンの分析（オブジェクト）
4. "challenges": 直面した課題（リスト）
5. "improvement_suggestions": 改善提案（リスト）
6. "generalizable_lessons": 一般化可能な教訓（リスト）
        """
        
        # LLMを使用して振り返りを生成
        response = self.llm_engine.generate(prompt, temperature=0.4, max_tokens=1536)
        
        # JSONを抽出
        import re
        json_match = re.search(r'```(?:json)?(.*?)```', response, re.DOTALL)
        
        try:
            if json_match:
                reflection = json.loads(json_match.group(1).strip())
            else:
                # JSON形式でない場合は解析を試みる
                reflection = json.loads(response.strip())
            
            return reflection
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse reflection: {e}")
            # 解析に失敗した場合はテキスト形式で返す
            return {
                "summary": response,
                "error": "Failed to parse JSON",
                "text_response": response
            }
