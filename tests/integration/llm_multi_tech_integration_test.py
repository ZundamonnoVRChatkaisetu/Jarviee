"""
複合AI技術連携テスト

このモジュールは、LLMと複数のAI技術（強化学習、シンボリックAI、マルチモーダルAI）を
同時に連携させるテストケースを実装しています。これにより、異なるAI技術が
協調して動作する場合の相互作用や性能を検証します。
"""

import os
import sys
import json
import unittest
from unittest.mock import MagicMock, patch
import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple

# テスト用にパスを設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 統合ハブと各種AI技術アダプターをインポート
from src.core.integration.ai_integration.hub import AIIntegrationHub
from src.core.integration.coordinator.coordinator import IntegrationCoordinator
from src.core.integration.coordinator.context_manager import ContextManager
from src.core.integration.adapters.reinforcement_learning.adapter import RLAdapter
from src.core.integration.adapters.symbolic_ai.adapter import SymbolicAIAdapter
from src.core.integration.adapters.multimodal.adapter import MultimodalAdapter
from src.core.integration.adapters.agent.adapter import AgentAdapter
from src.core.llm.engine import LLMEngine
from src.core.utils.event_bus import EventBus, Event
from src.core.integration.base import IntegrationMessage, ComponentType


class TestLLMMultiTechIntegration(unittest.TestCase):
    """LLMと複数AI技術の連携テストケース"""
    
    def setUp(self):
        """テスト環境のセットアップ"""
        # イベントバスの初期化
        self.event_bus = EventBus()
        
        # モックコンポーネントの作成
        self.mock_llm = MagicMock(spec=LLMEngine)
        self.mock_rl_adapter = MagicMock(spec=RLAdapter)
        self.mock_symbolic_adapter = MagicMock(spec=SymbolicAIAdapter)
        self.mock_multimodal_adapter = MagicMock(spec=MultimodalAdapter)
        self.mock_agent_adapter = MagicMock(spec=AgentAdapter)
        
        # コンポーネントIDの設定
        self.mock_llm.component_id = "llm_engine"
        self.mock_rl_adapter.component_id = "rl_adapter"
        self.mock_symbolic_adapter.component_id = "symbolic_adapter"
        self.mock_multimodal_adapter.component_id = "multimodal_adapter"
        self.mock_agent_adapter.component_id = "agent_adapter"
        
        # タイプの設定
        self.mock_llm.component_type = ComponentType.LLM
        self.mock_rl_adapter.component_type = ComponentType.REINFORCEMENT_LEARNING
        self.mock_symbolic_adapter.component_type = ComponentType.SYMBOLIC_AI
        self.mock_multimodal_adapter.component_type = ComponentType.MULTIMODAL_AI
        self.mock_agent_adapter.component_type = ComponentType.AGENT_AI
        
        # コンテキストマネージャーの初期化
        self.context_manager = ContextManager()
        
        # 統合ハブの初期化
        self.integration_hub = AIIntegrationHub(
            hub_id="test_hub",
            event_bus=self.event_bus,
            components={
                ComponentType.LLM: [self.mock_llm],
                ComponentType.REINFORCEMENT_LEARNING: [self.mock_rl_adapter],
                ComponentType.SYMBOLIC_AI: [self.mock_symbolic_adapter],
                ComponentType.MULTIMODAL_AI: [self.mock_multimodal_adapter],
                ComponentType.AGENT_AI: [self.mock_agent_adapter]
            },
            context_manager=self.context_manager,
            config={
                "default_timeout": 30,
                "load_balancing": "round_robin",
                "enable_monitoring": True,
                "log_level": "debug"
            }
        )
        
        # 統合コーディネーターの初期化
        self.coordinator = IntegrationCoordinator(
            coordinator_id="test_coordinator",
            hub=self.integration_hub,
            event_bus=self.event_bus,
            context_manager=self.context_manager
        )
        
        # モックの戻り値を設定
        self._configure_mock_responses()
        
        # 送信されたメッセージの追跡
        self.sent_messages = []
        
        # send_message メソッドを置き換えてメッセージを追跡
        self.original_send = self.integration_hub.send_message
        self.integration_hub.send_message = self._mock_send_message
    
    def _configure_mock_responses(self):
        """モックコンポーネントの応答を設定"""
        # LLMのモック応答
        self.mock_llm.process_message.side_effect = self._mock_llm_process
        
        # RLアダプターのモック応答
        self.mock_rl_adapter.process_message.side_effect = self._mock_rl_process
        
        # シンボリックAIアダプターのモック応答
        self.mock_symbolic_adapter.process_message.side_effect = self._mock_symbolic_process
        
        # マルチモーダルアダプターのモック応答
        self.mock_multimodal_adapter.process_message.side_effect = self._mock_multimodal_process
        
        # エージェントアダプターのモック応答
        self.mock_agent_adapter.process_message.side_effect = self._mock_agent_process
    
    def _mock_send_message(self, message):
        """メッセージ送信をモックしてトラッキング"""
        self.sent_messages.append(message)
        
        # メッセージのタイプに基づいて適切なモックに転送
        target_component = message.target_component
        
        # コンポーネントの検索
        if target_component == self.mock_llm.component_id:
            response = self.mock_llm.process_message(message)
        elif target_component == self.mock_rl_adapter.component_id:
            response = self.mock_rl_adapter.process_message(message)
        elif target_component == self.mock_symbolic_adapter.component_id:
            response = self.mock_symbolic_adapter.process_message(message)
        elif target_component == self.mock_multimodal_adapter.component_id:
            response = self.mock_multimodal_adapter.process_message(message)
        elif target_component == self.mock_agent_adapter.component_id:
            response = self.mock_agent_adapter.process_message(message)
        else:
            # 対象コンポーネントが見つからない場合
            return None
        
        # レスポンスがあれば、イベントとして発行
        if response:
            event = Event(
                event_type=f"integration.message.{response.message_type}",
                source=response.source_component,
                data={"message": response}
            )
            self.event_bus.publish(event)
        
        return message.message_id
    
    def _mock_llm_process(self, message):
        """LLMのメッセージ処理をモック"""
        # メッセージタイプに基づいて応答を生成
        if message.message_type == "query.text_analysis":
            # テキスト分析クエリへの応答
            return IntegrationMessage(
                source_component=self.mock_llm.component_id,
                target_component=message.source_component,
                message_type="response.text_analysis",
                content={
                    "analysis": {
                        "intent": "navigation_request",
                        "entities": [
                            {"type": "location", "value": "target_position", "confidence": 0.95},
                            {"type": "constraint", "value": "avoid_obstacles", "confidence": 0.98}
                        ],
                        "sentiment": "neutral",
                        "complexity": "medium"
                    },
                    "response": "テキスト分析が完了しました。ナビゲーションリクエストが検出されました。"
                },
                correlation_id=message.correlation_id
            )
        elif message.message_type == "command.generate_response":
            # レスポンス生成コマンドへの応答
            return IntegrationMessage(
                source_component=self.mock_llm.component_id,
                target_component=message.source_component,
                message_type="result.generated_response",
                content={
                    "text": "ナビゲーションプランを生成しました。最短経路で目標に到達し、すべての障害物を回避します。最初に北に3ユニット、次に東に2ユニット移動してください。",
                    "reasoning": [
                        "現在の位置と目標位置の分析",
                        "障害物の位置の考慮",
                        "最短経路の計算",
                        "安全な通過点の決定"
                    ]
                },
                correlation_id=message.correlation_id
            )
        elif message.message_type == "query.reward_formulation":
            # 報酬関数の定式化クエリへの応答
            return IntegrationMessage(
                source_component=self.mock_llm.component_id,
                target_component=message.source_component,
                message_type="response.reward_function",
                content={
                    "reward_function": {
                        "components": {
                            "goal_reached": 1.0,
                            "distance_reduction": 0.3,
                            "energy_efficiency": 0.4,
                            "obstacle_avoidance": 0.8,
                            "time_efficiency": 0.5
                        },
                        "computation": {
                            "type": "weighted_sum",
                            "normalization": True
                        }
                    }
                },
                correlation_id=message.correlation_id
            )
        
        # デフォルトの応答
        return IntegrationMessage(
            source_component=self.mock_llm.component_id,
            target_component=message.source_component,
            message_type="response.generic",
            content={"status": "processed"},
            correlation_id=message.correlation_id
        )
    
    def _mock_rl_process(self, message):
        """強化学習アダプターのメッセージ処理をモック"""
        # メッセージタイプに基づいて応答を生成
        if message.message_type == "command.optimize_action":
            # 行動最適化コマンドへの応答
            return IntegrationMessage(
                source_component=self.mock_rl_adapter.component_id,
                target_component=message.source_component,
                message_type="result.optimized_action",
                content={
                    "action": {
                        "type": "move",
                        "direction": [1, 0],  # 東に移動
                        "speed": 0.8,
                        "expected_reward": 0.75
                    },
                    "alternatives": [
                        {
                            "type": "move",
                            "direction": [0, 1],  # 北に移動
                            "speed": 0.7,
                            "expected_reward": 0.65
                        }
                    ]
                },
                correlation_id=message.correlation_id
            )
        elif message.message_type == "command.create_policy":
            # ポリシー作成コマンドへの応答
            return IntegrationMessage(
                source_component=self.mock_rl_adapter.component_id,
                target_component=message.source_component,
                message_type="result.policy_created",
                content={
                    "policy_id": str(uuid.uuid4()),
                    "policy_type": "dqn",
                    "state_space": message.content.get("state_space"),
                    "action_space": message.content.get("action_space"),
                    "status": "training",
                    "estimated_completion": time.time() + 10  # 10秒後に完了予定
                },
                correlation_id=message.correlation_id
            )
        
        # デフォルトの応答
        return IntegrationMessage(
            source_component=self.mock_rl_adapter.component_id,
            target_component=message.source_component,
            message_type="response.generic",
            content={"status": "processed"},
            correlation_id=message.correlation_id
        )
    
    def _mock_symbolic_process(self, message):
        """シンボリックAIアダプターのメッセージ処理をモック"""
        # メッセージタイプに基づいて応答を生成
        if message.message_type == "query.logical_inference":
            # 論理推論クエリへの応答
            return IntegrationMessage(
                source_component=self.mock_symbolic_adapter.component_id,
                target_component=message.source_component,
                message_type="response.logical_inference",
                content={
                    "conclusions": [
                        {"statement": "path_is_safe", "confidence": 0.92},
                        {"statement": "goal_is_reachable", "confidence": 0.85}
                    ],
                    "reasoning_steps": [
                        "障害物の位置を分析",
                        "エージェントの現在位置を確認",
                        "空間内の安全な経路の存在を検証",
                        "目標到達可能性を計算"
                    ]
                },
                correlation_id=message.correlation_id
            )
        elif message.message_type == "command.verify_plan":
            # 計画検証コマンドへの応答
            return IntegrationMessage(
                source_component=self.mock_symbolic_adapter.component_id,
                target_component=message.source_component,
                message_type="result.plan_verification",
                content={
                    "is_valid": True,
                    "safety_score": 0.95,
                    "efficiency_score": 0.88,
                    "constraint_violations": [],
                    "suggestions": [
                        {
                            "step": 2,
                            "suggestion": "エネルギー効率を向上させるために速度を若干下げる",
                            "impact": "minor"
                        }
                    ]
                },
                correlation_id=message.correlation_id
            )
        
        # デフォルトの応答
        return IntegrationMessage(
            source_component=self.mock_symbolic_adapter.component_id,
            target_component=message.source_component,
            message_type="response.generic",
            content={"status": "processed"},
            correlation_id=message.correlation_id
        )
    
    def _mock_multimodal_process(self, message):
        """マルチモーダルAIアダプターのメッセージ処理をモック"""
        # メッセージタイプに基づいて応答を生成
        if message.message_type == "query.environment_analysis":
            # 環境分析クエリへの応答
            return IntegrationMessage(
                source_component=self.mock_multimodal_adapter.component_id,
                target_component=message.source_component,
                message_type="response.environment_analysis",
                content={
                    "detected_objects": [
                        {"type": "obstacle", "position": [3, 4], "size": [1, 1], "confidence": 0.97},
                        {"type": "obstacle", "position": [7, 2], "size": [2, 1], "confidence": 0.94},
                        {"type": "goal", "position": [9, 9], "confidence": 0.99}
                    ],
                    "spatial_map": {
                        "dimensions": [10, 10],
                        "resolution": 1.0,
                        "occupied_cells": [[3, 4], [7, 2], [8, 2]]
                    },
                    "agent_position": [1, 1],
                    "visual_features": {
                        "lighting": "good",
                        "visibility": "clear",
                        "terrain": "flat"
                    }
                },
                correlation_id=message.correlation_id
            )
        elif message.message_type == "command.generate_visualization":
            # 可視化生成コマンドへの応答
            return IntegrationMessage(
                source_component=self.mock_multimodal_adapter.component_id,
                target_component=message.source_component,
                message_type="result.visualization",
                content={
                    "visualization_type": "path_map",
                    "data": {
                        "map_dimensions": [10, 10],
                        "agent_position": [1, 1],
                        "goal_position": [9, 9],
                        "obstacles": [[3, 4], [7, 2], [8, 2]],
                        "planned_path": [[1, 1], [2, 1], [2, 2], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [6, 4], [7, 4], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [9, 8], [9, 9]]
                    },
                    "format": "json"  # 実際には画像データや3Dモデルなどになる
                },
                correlation_id=message.correlation_id
            )
        
        # デフォルトの応答
        return IntegrationMessage(
            source_component=self.mock_multimodal_adapter.component_id,
            target_component=message.source_component,
            message_type="response.generic",
            content={"status": "processed"},
            correlation_id=message.correlation_id
        )
    
    def _mock_agent_process(self, message):
        """エージェントAIアダプターのメッセージ処理をモック"""
        # メッセージタイプに基づいて応答を生成
        if message.message_type == "command.execute_task":
            # タスク実行コマンドへの応答
            return IntegrationMessage(
                source_component=self.mock_agent_adapter.component_id,
                target_component=message.source_component,
                message_type="result.task_execution",
                content={
                    "task_id": message.content.get("task_id", str(uuid.uuid4())),
                    "status": "in_progress",
                    "progress": 0.25,
                    "current_step": {
                        "name": "path_planning",
                        "status": "completed",
                        "output": {
                            "planned_path": [[1, 1], [2, 1], [2, 2], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [6, 4], [7, 4], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [9, 8], [9, 9]]
                        }
                    },
                    "next_step": {
                        "name": "execute_movement",
                        "status": "pending"
                    },
                    "estimated_completion": time.time() + 20  # 20秒後に完了予定
                },
                correlation_id=message.correlation_id
            )
        elif message.message_type == "query.task_status":
            # タスク状態クエリへの応答
            return IntegrationMessage(
                source_component=self.mock_agent_adapter.component_id,
                target_component=message.source_component,
                message_type="response.task_status",
                content={
                    "task_id": message.content.get("task_id"),
                    "status": "in_progress",
                    "progress": 0.65,
                    "current_step": {
                        "name": "execute_movement",
                        "status": "in_progress",
                        "progress": 0.7
                    }
                },
                correlation_id=message.correlation_id
            )
        
        # デフォルトの応答
        return IntegrationMessage(
            source_component=self.mock_agent_adapter.component_id,
            target_component=message.source_component,
            message_type="response.generic",
            content={"status": "processed"},
            correlation_id=message.correlation_id
        )
        
    def test_navigation_task_all_tech(self):
        """ナビゲーションタスクでの全AI技術連携テスト (TC-MT-01)"""
        # テスト用のタスクコンテキスト
        task_context = {
            "task_id": "nav_task_1",
            "task_type": "navigation",
            "description": "障害物を回避しながら目標位置に到達する",
            "priority": "high",
            "start_time": time.time()
        }
        
        # コンテキストマネージャーにタスクを登録
        context_id = self.context_manager.create_context(task_context)
        
        # 初期メッセージの送信（ユーザーからの指示を想定）
        initial_message = IntegrationMessage(
            source_component="user_interface",
            target_component=self.integration_hub.hub_id,
            message_type="command.execute_complex_task",
            content={
                "task_description": "障害物を回避しながら位置[9,9]に効率的に移動してください。",
                "context_id": context_id,
                "requirements": {
                    "safety": "必須",
                    "efficiency": "高",
                    "energy_conservation": "中"
                }
            },
            correlation_id=str(uuid.uuid4())
        )
        
        # メッセージを処理
        # 実際のシステムではイベントバス経由で送信されるが、テストでは直接コーディネーターに転送
        self.coordinator.process_task_request(initial_message)
        
        # 送信されたメッセージを検証
        # 各AI技術に対して、適切なタイプのメッセージが送信されていることを確認
        
        # LLMへのメッセージ（テキスト分析）
        llm_messages = [msg for msg in self.sent_messages 
                       if msg.target_component == self.mock_llm.component_id]
        self.assertGreaterEqual(len(llm_messages), 1)
        
        # テキスト分析メッセージを検証
        text_analysis_msg = next((msg for msg in llm_messages 
                                if msg.message_type == "query.text_analysis"), None)
        self.assertIsNotNone(text_analysis_msg)
        self.assertEqual(text_analysis_msg.content["text"], 
                        "障害物を回避しながら位置[9,9]に効率的に移動してください。")
        
        # マルチモーダルAIへのメッセージ（環境分析）
        multimodal_messages = [msg for msg in self.sent_messages 
                              if msg.target_component == self.mock_multimodal_adapter.component_id]
        self.assertGreaterEqual(len(multimodal_messages), 1)
        
        # 環境分析メッセージを検証
        env_analysis_msg = next((msg for msg in multimodal_messages 
                               if msg.message_type == "query.environment_analysis"), None)
        self.assertIsNotNone(env_analysis_msg)
        
        # 強化学習アダプターへのメッセージ（ポリシー作成または行動最適化）
        rl_messages = [msg for msg in self.sent_messages 
                      if msg.target_component == self.mock_rl_adapter.component_id]
        self.assertGreaterEqual(len(rl_messages), 1)
        
        # エージェントへのタスク実行メッセージ
        agent_messages = [msg for msg in self.sent_messages 
                         if msg.target_component == self.mock_agent_adapter.component_id]
        self.assertGreaterEqual(len(agent_messages), 1)
        execute_task_msg = next((msg for msg in agent_messages 
                               if msg.message_type == "command.execute_task"), None)
        self.assertIsNotNone(execute_task_msg)
        
        # コンテキストの更新を確認
        updated_context = self.context_manager.get_context(context_id)
        self.assertEqual(updated_context["task_id"], "nav_task_1")
        # テスト終了時にはタスクが進行中または完了のいずれかになっている
        self.assertIn("status", updated_context)
        self.assertIn(updated_context["status"], ["in_progress", "completed"])
    
    def test_symbolic_reinforcement_integration(self):
        """シンボリックAIと強化学習の連携テスト (TC-MT-02)"""
        # テスト用のタスクコンテキスト
        task_context = {
            "task_id": "verify_plan_task",
            "task_type": "plan_verification",
            "description": "最適化されたナビゲーション計画の安全性と有効性を検証する",
            "priority": "medium",
            "start_time": time.time()
        }
        
        # コンテキストマネージャーにタスクを登録
        context_id = self.context_manager.create_context(task_context)
        
        # ナビゲーション計画（強化学習で生成されたと想定）
        navigation_plan = {
            "planned_path": [[1, 1], [2, 1], [2, 2], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [6, 4], [7, 4], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [9, 8], [9, 9]],
            "estimated_time": 42,
            "energy_usage": 0.65,
            "safety_margin": 1.0  # 障害物からの最小距離
        }
        
        # 環境情報（マルチモーダルAIからの情報と想定）
        environment_info = {
            "dimensions": [10, 10],
            "obstacles": [
                {"position": [3, 4], "size": [1, 1]},
                {"position": [7, 2], "size": [2, 1]}
            ],
            "goal": {"position": [9, 9]},
            "agent": {"position": [1, 1], "capabilities": {"max_speed": 1.0, "sensor_range": 3.0}}
        }
        
        # コンテキストを更新
        self.context_manager.update_context(context_id, {
            "navigation_plan": navigation_plan,
            "environment_info": environment_info
        })
        
        # 初期メッセージの送信（LLMからの指示を想定）
        initial_message = IntegrationMessage(
            source_component=self.mock_llm.component_id,
            target_component=self.integration_hub.hub_id,
            message_type="command.verify_and_optimize_plan",
            content={
                "context_id": context_id,
                "verification_criteria": {
                    "safety": "高優先度",
                    "efficiency": "中優先度",
                    "energy_usage": "中優先度"
                },
                "optimization_goals": {
                    "minimize_time": 0.6,
                    "minimize_energy": 0.4
                }
            },
            correlation_id=str(uuid.uuid4())
        )
        
        # メッセージを処理
        # 統合ハブを通してコーディネーターに転送
        event = Event(
            event_type="integration.message.command.verify_and_optimize_plan",
            source=self.mock_llm.component_id,
            data={"message": initial_message}
        )
        self.coordinator.handle_verification_optimization_request(event)
        
        # 送信されたメッセージを検証
        
        # シンボリックAIへの検証リクエスト
        symbolic_messages = [msg for msg in self.sent_messages 
                           if msg.target_component == self.mock_symbolic_adapter.component_id]
        self.assertGreaterEqual(len(symbolic_messages), 1)
        
        # 計画検証メッセージを検証
        verify_plan_msg = next((msg for msg in symbolic_messages 
                              if msg.message_type == "command.verify_plan"), None)
        self.assertIsNotNone(verify_plan_msg)
        self.assertIn("plan", verify_plan_msg.content)
        self.assertIn("criteria", verify_plan_msg.content)
        
        # 強化学習アダプターへの最適化リクエスト
        rl_messages = [msg for msg in self.sent_messages 
                      if msg.target_component == self.mock_rl_adapter.component_id]
        
        # 検証結果に基づいて強化学習が呼び出されたかを確認
        # （計画が有効な場合のみ最適化が呼び出される）
        if verify_plan_msg is not None:
            optimization_msg = next((msg for msg in rl_messages 
                                   if msg.message_type == "command.optimize_policy" or
                                   msg.message_type == "command.optimize_action"), None)
            # 注: 検証結果が無効な場合は最適化は呼び出されない
            
            # コンテキストの更新を確認
            updated_context = self.context_manager.get_context(context_id)
            self.assertIn("verification_result", updated_context)
            
            # 検証と最適化の結果がコンテキストに記録されていることを確認
            if optimization_msg is not None:
                self.assertIn("optimization_result", updated_context)
    
    def test_llm_multimodal_integration(self):
        """LLMとマルチモーダルAIの連携テスト (TC-MT-03)"""
        # テスト用のタスクコンテキスト
        task_context = {
            "task_id": "environment_description_task",
            "task_type": "environment_description",
            "description": "環境のセンサーデータを分析し、自然言語で説明を生成する",
            "priority": "medium",
            "start_time": time.time()
        }
        
        # コンテキストマネージャーにタスクを登録
        context_id = self.context_manager.create_context(task_context)
        
        # 環境センサーデータ（マルチモーダルAIの入力と想定）
        sensor_data = {
            "visual": {
                "raw_images": "バイナリデータ（テスト用に省略）",
                "resolution": [1024, 768],
                "format": "RGB"
            },
            "lidar": {
                "point_cloud": "バイナリデータ（テスト用に省略）",
                "range": 10,
                "density": "high"
            },
            "position": [45.3, 22.7],
            "orientation": [0, 1, 0]  # 北向き
        }
        
        # コンテキストを更新
        self.context_manager.update_context(context_id, {
            "sensor_data": sensor_data
        })
        
        # 初期メッセージの送信（ユーザーインターフェースからの要求を想定）
        initial_message = IntegrationMessage(
            source_component="user_interface",
            target_component=self.integration_hub.hub_id,
            message_type="command.analyze_and_describe_environment",
            content={
                "context_id": context_id,
                "format_preferences": {
                    "detail_level": "high",
                    "include_visual_elements": True,
                    "language_style": "descriptive"
                }
            },
            correlation_id=str(uuid.uuid4())
        )
        
        # メッセージを処理
        event = Event(
            event_type="integration.message.command.analyze_and_describe_environment",
            source="user_interface",
            data={"message": initial_message}
        )
        self.coordinator.handle_environment_description_request(event)
        
        # 送信されたメッセージを検証
        
        # マルチモーダルAIへの分析リクエスト
        multimodal_messages = [msg for msg in self.sent_messages 
                              if msg.target_component == self.mock_multimodal_adapter.component_id]
        self.assertGreaterEqual(len(multimodal_messages), 1)
        
        # 環境分析メッセージを検証
        env_analysis_msg = next((msg for msg in multimodal_messages 
                               if msg.message_type == "query.environment_analysis"), None)
        self.assertIsNotNone(env_analysis_msg)
        
        # LLMへの説明生成リクエスト
        llm_messages = [msg for msg in self.sent_messages 
                       if msg.target_component == self.mock_llm.component_id]
        self.assertGreaterEqual(len(llm_messages), 1)
        
        # 説明生成メッセージを検証
        description_msg = next((msg for msg in llm_messages 
                              if msg.message_type == "command.generate_description"), None)
        if description_msg is None:
            # 代替として、より一般的なテキスト生成コマンドを検索
            description_msg = next((msg for msg in llm_messages 
                                  if msg.message_type == "command.generate_response" or 
                                  msg.message_type == "command.generate_text"), None)
            
        self.assertIsNotNone(description_msg)
        
        # 可視化生成リクエスト（オプション）
        visualization_msg = next((msg for msg in multimodal_messages 
                                if msg.message_type == "command.generate_visualization"), None)
        
        # コンテキストの更新を確認
        updated_context = self.context_manager.get_context(context_id)
        self.assertIn("analysis_result", updated_context)
        
        # 説明が生成された場合はコンテキストに記録されていることを確認
        if description_msg is not None:
            # 応答またはテキスト生成の結果がどこかに保存されているはず
            found_result = False
            for key in updated_context:
                if "description" in key or "text" in key or "response" in key:
                    found_result = True
                    break
            self.assertTrue(found_result)
    
    def test_agent_rl_symbolic_integration(self):
        """エージェント、強化学習、シンボリックAIの連携テスト (TC-MT-04)"""
        # テスト用のタスクコンテキスト
        task_context = {
            "task_id": "complex_navigation_task",
            "task_type": "complex_navigation",
            "description": "複数のサブタスクを含む複雑なナビゲーションミッションを実行する",
            "priority": "high",
            "start_time": time.time()
        }
        
        # コンテキストマネージャーにタスクを登録
        context_id = self.context_manager.create_context(task_context)
        
        # ミッション情報
        mission_info = {
            "objective": "3つのチェックポイントを訪問し、最終目的地に到達する",
            "checkpoints": [
                {"id": "CP1", "position": [3, 3], "task": "データ収集"},
                {"id": "CP2", "position": [7, 2], "task": "機器の点検"},
                {"id": "CP3", "position": [5, 8], "task": "サンプル採取"}
            ],
            "final_destination": {"position": [9, 9], "task": "ミッション完了報告"},
            "constraints": [
                "エネルギー消費を最小限に抑える",
                "経路上の障害物をすべて回避する",
                "指定されたチェックポイントをすべて訪問する"
            ]
        }
        
        # 環境情報
        environment_info = {
            "dimensions": [10, 10],
            "obstacles": [
                {"position": [2, 4], "size": [1, 1]},
                {"position": [6, 3], "size": [1, 3]},
                {"position": [8, 7], "size": [2, 1]}
            ],
            "start_position": [1, 1],
            "terrain": "mixed",
            "dynamic_elements": True
        }
        
        # コンテキストを更新
        self.context_manager.update_context(context_id, {
            "mission_info": mission_info,
            "environment_info": environment_info
        })
        
        # 初期メッセージの送信（LLMからのタスク分解要求を想定）
        initial_message = IntegrationMessage(
            source_component=self.mock_llm.component_id,
            target_component=self.integration_hub.hub_id,
            message_type="command.execute_complex_mission",
            content={
                "context_id": context_id,
                "execution_strategy": "sequential",  # チェックポイントを順番に訪問
                "adaptivity_level": "high",  # 環境の変化に積極的に適応
                "reporting_frequency": "medium"  # 進捗の報告頻度
            },
            correlation_id=str(uuid.uuid4())
        )
        
        # メッセージを処理
        event = Event(
            event_type="integration.message.command.execute_complex_mission",
            source=self.mock_llm.component_id,
            data={"message": initial_message}
        )
        self.coordinator.handle_complex_mission_request(event)
        
        # 送信されたメッセージを検証
        
        # エージェントへのタスク実行リクエスト
        agent_messages = [msg for msg in self.sent_messages 
                         if msg.target_component == self.mock_agent_adapter.component_id]
        self.assertGreaterEqual(len(agent_messages), 1)
        
        # タスク実行メッセージを検証
        execute_task_msg = next((msg for msg in agent_messages 
                               if msg.message_type == "command.execute_task"), None)
        self.assertIsNotNone(execute_task_msg)
        
        # 計画（タスク分解）が含まれていることを確認
        self.assertIn("task_plan", execute_task_msg.content)
        self.assertIn("checkpoints", execute_task_msg.content["task_plan"])
        
        # シンボリックAIへの検証リクエスト
        symbolic_messages = [msg for msg in self.sent_messages 
                           if msg.target_component == self.mock_symbolic_adapter.component_id]
        self.assertGreaterEqual(len(symbolic_messages), 1)
        
        # 検証メッセージを確認
        verify_msg = next((msg for msg in symbolic_messages 
                         if "verify" in msg.message_type.lower()), None)
        self.assertIsNotNone(verify_msg)
        
        # 強化学習への最適化リクエスト
        rl_messages = [msg for msg in self.sent_messages 
                      if msg.target_component == self.mock_rl_adapter.component_id]
        self.assertGreaterEqual(len(rl_messages), 1)
        
        # ポリシーまたは行動最適化メッセージを確認
        optimize_msg = next((msg for msg in rl_messages 
                           if "optimize" in msg.message_type.lower() or 
                           "policy" in msg.message_type.lower() or
                           "action" in msg.message_type.lower()), None)
        self.assertIsNotNone(optimize_msg)
        
        # コンテキストの更新を確認
        updated_context = self.context_manager.get_context(context_id)
        self.assertIn("task_plan", updated_context)
        self.assertIn("status", updated_context)
        
        # 各モジュールの結果がコンテキストに反映されていることを確認
        for component_type in ["agent", "symbolic", "rl"]:
            found_result = False
            for key in updated_context:
                if component_type in key.lower() and "result" in key.lower():
                    found_result = True
                    break
            self.assertTrue(found_result, f"{component_type}の結果がコンテキストに反映されていません")
    
    def test_all_technology_performance(self):
        """全AI技術を使用した場合のパフォーマンステスト (TC-MT-05)"""
        # テスト用のタスクコンテキスト
        task_context = {
            "task_id": "all_tech_performance_task",
            "task_type": "comprehensive_analysis",
            "description": "全てのAI技術を活用した総合分析と行動計画の作成",
            "priority": "high",
            "start_time": time.time()
        }
        
        # コンテキストマネージャーにタスクを登録
        context_id = self.context_manager.create_context(task_context)
        
        # 入力データ
        input_data = {
            "text_query": "複雑な地形と動的障害物を含む環境で、最小のエネルギー消費で目標地点[9,9]に到達するための最適な計画を立ててください。",
            "environment": {
                "dimensions": [10, 10],
                "obstacles": [
                    {"position": [2, 4], "size": [1, 1], "type": "static"},
                    {"position": [6, 3], "size": [1, 3], "type": "static"},
                    {"position": [4, 6], "size": [1, 1], "type": "dynamic", "velocity": [0.1, 0.2]}
                ],
                "start_position": [1, 1],
                "goal_position": [9, 9],
                "terrain": [
                    {"position": [3, 3], "type": "rough", "energy_factor": 1.5},
                    {"position": [7, 7], "type": "slippery", "stability_factor": 0.7}
                ]
            },
            "constraints": [
                "全ての障害物を回避すること",
                "エネルギー消費を最小化すること",
                "できるだけ短時間で目標に到達すること"
            ]
        }
        
        # コンテキストを更新
        self.context_manager.update_context(context_id, {
            "input_data": input_data
        })
        
        # パフォーマンス計測開始
        start_time = time.time()
        
        # 初期メッセージの送信
        initial_message = IntegrationMessage(
            source_component="user_interface",
            target_component=self.integration_hub.hub_id,
            message_type="command.comprehensive_task",
            content={
                "context_id": context_id,
                "required_technologies": ["llm", "rl", "symbolic", "multimodal", "agent"],
                "output_format": {
                    "include_visualization": True,
                    "include_reasoning": True,
                    "include_alternatives": True
                }
            },
            correlation_id=str(uuid.uuid4())
        )
        
        # 複数回実行してパフォーマンスを計測
        iterations = 3
        execution_times = []
        
        for i in range(iterations):
            # 初期化
            self.sent_messages = []
            
            # メッセージを処理
            iteration_start = time.time()
            
            event = Event(
                event_type="integration.message.command.comprehensive_task",
                source="user_interface",
                data={"message": initial_message}
            )
            
            # 統合コーディネーターでの処理
            self.coordinator.handle_comprehensive_task(event)
            
            # 実行時間を記録
            execution_time = time.time() - iteration_start
            execution_times.append(execution_time)
            
            # メッセージの数を確認
            message_counts = {
                "llm": len([msg for msg in self.sent_messages if msg.target_component == self.mock_llm.component_id]),
                "rl": len([msg for msg in self.sent_messages if msg.target_component == self.mock_rl_adapter.component_id]),
                "symbolic": len([msg for msg in self.sent_messages if msg.target_component == self.mock_symbolic_adapter.component_id]),
                "multimodal": len([msg for msg in self.sent_messages if msg.target_component == self.mock_multimodal_adapter.component_id]),
                "agent": len([msg for msg in self.sent_messages if msg.target_component == self.mock_agent_adapter.component_id])
            }
            
            print(f"イテレーション {i+1}: 実行時間 = {execution_time:.3f}秒, メッセージ数 = {sum(message_counts.values())}")
            print(f"  技術別メッセージ数: {message_counts}")
        
        # 平均実行時間を計算
        avg_execution_time = sum(execution_times) / len(execution_times)
        
        # 総実行時間
        total_time = time.time() - start_time
        
        # 結果の表示
        print(f"\n全技術連携パフォーマンステスト結果:")
        print(f"平均実行時間: {avg_execution_time:.3f}秒")
        print(f"標準偏差: {(sum((t - avg_execution_time) ** 2 for t in execution_times) / len(execution_times)) ** 0.5:.3f}秒")
        print(f"メッセージ総数: {len(self.sent_messages)}")
        print(f"イテレーション数: {iterations}")
        print(f"総実行時間: {total_time:.3f}秒")
        
        # 各技術への分布を確認
        message_distribution = {
            "llm": len([msg for msg in self.sent_messages if msg.target_component == self.mock_llm.component_id]),
            "rl": len([msg for msg in self.sent_messages if msg.target_component == self.mock_rl_adapter.component_id]),
            "symbolic": len([msg for msg in self.sent_messages if msg.target_component == self.mock_symbolic_adapter.component_id]),
            "multimodal": len([msg for msg in self.sent_messages if msg.target_component == self.mock_multimodal_adapter.component_id]),
            "agent": len([msg for msg in self.sent_messages if msg.target_component == self.mock_agent_adapter.component_id])
        }
        
        print(f"メッセージ分布: {message_distribution}")
        
        # すべての技術が使用されていることを確認
        for tech, count in message_distribution.items():
            self.assertGreater(count, 0, f"{tech}への呼び出しがありません")
        
        # パフォーマンス基準を満たしていることを確認（モックなので高速）
        # 実際のシステムでは適切な閾値に調整
        self.assertLess(avg_execution_time, 1.0, "パフォーマンスが基準を満たしていません")


if __name__ == "__main__":
    # ロギングを設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # テストを実行
    unittest.main()
