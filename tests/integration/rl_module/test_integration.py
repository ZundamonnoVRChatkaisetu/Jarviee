"""
Integration Tests for Reinforcement Learning Module.

This file contains tests for verifying the integration between the reinforcement
learning adapter and other components of the Jarviee system, with a focus on
LLM-to-RL communication and functionality.
"""

import os
import sys
import json
import unittest
from unittest.mock import MagicMock, patch
import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Union

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.core.utils.logger import Logger
from src.core.utils.event_bus import EventBus
from src.core.integration.base import IntegrationMessage, ComponentType
from src.core.llm.engine import LLMEngine
from src.core.integration.adapters.reinforcement_learning.adapter import RLAdapter
from src.core.integration.adapters.reinforcement_learning.action import ActionOptimizer
from src.core.integration.adapters.reinforcement_learning.environment import EnvironmentStateManager
from src.core.integration.adapters.reinforcement_learning.reward import RewardFunctionGenerator

from tests.integration.rl_module.test_environment import SimulationEnvironment, RLTestEnvironment


class RLModuleIntegrationTest(unittest.TestCase):
    """Integration tests for the Reinforcement Learning module."""
    
    def setUp(self):
        """Set up the test environment."""
        # Initialize shared event bus and components
        self.event_bus = EventBus()
        
        # Mock LLMEngine for testing
        self.llm_engine = MagicMock(spec=LLMEngine)
        self.llm_engine.component_id = "test_llm"
        
        # Configure RL adapter for testing
        self.rl_adapter = RLAdapter(
            adapter_id="test_rl_adapter",
            llm_component_id="test_llm",
            config={
                "max_optimization_steps": 100,
                "optimization_timeout": 5
            }
        )
        
        # Connect event bus
        self.event_bus.subscribe("integration.*", self.rl_adapter.process_message)
        
        # Set up message tracking for tests
        self.sent_messages = []
        self.rl_adapter.send_message = self._mock_send_message
        
        # Mock reward function generator to produce deterministic results
        self.rl_adapter.reward_generator.generate_from_text = MagicMock(
            return_value=lambda state: 1.0 if state.get("goal_achieved", False) else 0.0
        )
        
        # Prepare test environment
        self.setup_test_environment()
        
        # Common test data
        self.test_task_id = "integration_test_task"
        self.test_goal = "Maximize efficiency while maintaining performance"
        
        # Set up a standard task for tests
        self.setup_standard_task()
    
    def _mock_send_message(self, target, message_type, content, correlation_id=None):
        """Mock send_message to track messages."""
        message = IntegrationMessage(
            source_component=self.rl_adapter.component_id,
            target_component=target,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id
        )
        self.sent_messages.append(message)
        return message.message_id
    
    def setup_test_environment(self):
        """Set up the RL test environment for testing."""
        test_config = {
            "simulation": {
                "type": "resource_management",
                "max_steps": 50,
                "initial_resources": {"cpu": 0.5, "memory": 0.5, "network": 0.5},
                "initial_demand": {"cpu": 0.3, "memory": 0.4, "network": 0.2}
            },
            "metrics": ["reward", "completion_rate", "efficiency", "performance"],
            "establish_baseline": True,
            "baseline_episodes": 3
        }
        self.test_env = RLTestEnvironment(test_config)
    
    def setup_standard_task(self):
        """Set up a standard task for tests."""
        self.standard_task = {
            "task_id": self.test_task_id,
            "goal_description": self.test_goal,
            "environment_state": {
                "resources": {"cpu": 0.5, "memory": 0.5, "network": 0.5},
                "demand": {"cpu": 0.3, "memory": 0.4, "network": 0.2},
                "performance": 1.0,
                "efficiency": 1.0
            }
        }
        
        # Create the task in the RL adapter
        self.rl_adapter.active_tasks[self.test_task_id] = {
            "created_at": time.time(),
            "goal_description": self.test_goal,
            "environment_state": self.standard_task["environment_state"].copy(),
            "reward_function": lambda state: state.get("performance", 0.0) * 0.6 + state.get("efficiency", 0.0) * 0.4
        }
    
    def tearDown(self):
        """Clean up after the test."""
        self.rl_adapter.active_tasks.clear()
        self.sent_messages.clear()
    
    def test_rl_adapter_initialization(self):
        """Test that the RL adapter initializes correctly."""
        # Create a fresh adapter
        adapter = RLAdapter(
            adapter_id="init_test_adapter",
            llm_component_id="test_llm"
        )
        
        # Check that components are initialized
        self.assertIsNotNone(adapter.reward_generator)
        self.assertIsNotNone(adapter.environment_manager)
        self.assertIsNotNone(adapter.action_optimizer)
        
        # Check that capabilities are defined
        self.assertIn("language_to_reward_conversion", adapter.capabilities)
        self.assertIn("action_optimization", adapter.capabilities)
        
        # Check that initial state is correct
        self.assertEqual({}, adapter.active_tasks)
        self.assertFalse(adapter.learning_state["training_in_progress"])
        
        # Initialize the adapter
        success = adapter._initialize_impl()
        self.assertTrue(success)
    
    def test_task_creation(self):
        """Test creation of a reinforcement learning task."""
        # Create a new task ID
        task_id = "task_creation_test"
        
        # Define task creation message
        task_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.create_task",
            content={
                "task_id": task_id,
                "goal_description": "Navigate to the target efficiently",
                "environment_state": {
                    "type": "grid_world",
                    "size": [5, 5],
                    "agent_position": [0, 0],
                    "target_position": [4, 4]
                }
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process task creation message
        self.rl_adapter.process_message(task_message)
        
        # Verify task was created
        self.assertIn(task_id, self.rl_adapter.active_tasks)
        self.assertEqual("Navigate to the target efficiently", 
                         self.rl_adapter.active_tasks[task_id]["goal_description"])
        
        # Check that response was sent
        self.assertEqual(1, len(self.sent_messages))
        response = self.sent_messages[0]
        self.assertEqual("response", response.message_type)
        self.assertEqual("test_llm", response.target_component)
        self.assertTrue(response.content.get("success", False))
    
    def test_llm_goal_interpretation(self):
        """Test interaction with LLM for goal interpretation."""
        # Mock the LLM response
        interpreted_goal = {
            "objective": "maximize_resource_efficiency",
            "constraints": ["maintain_performance_above_0.8"],
            "priorities": {"efficiency": 0.7, "performance": 0.3}
        }
        
        # Create a goal interpretation message
        interpretation_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="llm.goal_interpretation",
            content={
                "task_id": self.test_task_id,
                "goal_data": interpreted_goal
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process interpretation message
        self.rl_adapter.process_message(interpretation_message)
        
        # Verify interpretation was stored
        self.assertIn("interpreted_goal", self.rl_adapter.active_tasks[self.test_task_id])
        self.assertEqual(interpreted_goal, 
                         self.rl_adapter.active_tasks[self.test_task_id]["interpreted_goal"])
        
        # Verify reward function was generated
        self.assertIn("reward_function", self.rl_adapter.active_tasks[self.test_task_id])
        
        # Verify notification was sent
        self.assertEqual(1, len(self.sent_messages))
        notification = self.sent_messages[0]
        self.assertEqual("technology_notification", notification.message_type)
        self.assertEqual("reward_function_generated", notification.content.get("notification_type"))
    
    def test_action_optimization(self):
        """Test optimization of actions based on RL."""
        # Mock the action optimizer
        original_optimize = self.rl_adapter.action_optimizer.optimize
        self.rl_adapter.action_optimizer.optimize = MagicMock(
            return_value=("allocate_resources", 0.85, 10)
        )
        
        # Create optimization request
        optimization_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.optimize_action",
            content={
                "task_id": self.test_task_id,
                "action_space": ["allocate_resources", "reduce_resources", "balance_resources"],
                "optimization_criteria": {
                    "time_constraint": 2,
                    "max_steps": 50
                }
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process optimization request
        self.rl_adapter.process_message(optimization_message)
        
        # Verify optimize was called
        self.rl_adapter.action_optimizer.optimize.assert_called_once()
        
        # Verify response was sent
        self.assertEqual(1, len(self.sent_messages))
        response = self.sent_messages[0]
        self.assertEqual("response", response.message_type)
        self.assertEqual("test_llm", response.target_component)
        self.assertTrue(response.content.get("success", False))
        self.assertEqual("allocate_resources", response.content.get("best_action"))
        
        # Restore original method
        self.rl_adapter.action_optimizer.optimize = original_optimize
    
    def test_environment_state_update(self):
        """Test updating environment state."""
        # Initial state
        initial_resources = self.rl_adapter.active_tasks[self.test_task_id]["environment_state"]["resources"]
        
        # Create state update message
        update_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.update_environment",
            content={
                "task_id": self.test_task_id,
                "update": {
                    "resources": {
                        "cpu": 0.8,
                        "memory": 0.6,
                        "network": 0.4
                    }
                }
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process update message
        self.rl_adapter.process_message(update_message)
        
        # Verify state was updated
        updated_resources = self.rl_adapter.active_tasks[self.test_task_id]["environment_state"]["resources"]
        self.assertEqual(0.8, updated_resources["cpu"])
        self.assertEqual(0.6, updated_resources["memory"])
        self.assertEqual(0.4, updated_resources["network"])
        
        # Verify response was sent
        self.assertEqual(1, len(self.sent_messages))
        response = self.sent_messages[0]
        self.assertEqual("response", response.message_type)
        self.assertTrue(response.content.get("success", False))
    
    def test_feedback_incorporation(self):
        """Test incorporation of feedback."""
        # Mock the feedback incorporation method
        original_incorporate = self.rl_adapter.action_optimizer.incorporate_feedback
        self.rl_adapter.action_optimizer.incorporate_feedback = MagicMock(return_value=True)
        
        # Create feedback message
        feedback_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.provide_feedback",
            content={
                "task_id": self.test_task_id,
                "feedback": {
                    "action": "allocate_resources",
                    "reward": 0.75,
                    "comments": "Good allocation but could be more efficient"
                }
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process feedback message
        self.rl_adapter.process_message(feedback_message)
        
        # Verify feedback incorporation was called
        self.rl_adapter.action_optimizer.incorporate_feedback.assert_called_once()
        
        # Verify LLM was consulted for feedback interpretation (if comments provided)
        if hasattr(self.rl_adapter, "send_to_llm"):
            original_send_to_llm = self.rl_adapter.send_to_llm
            self.rl_adapter.send_to_llm = MagicMock()
            
            # Process feedback message again
            self.rl_adapter.process_message(feedback_message)
            
            # Verify LLM was consulted
            self.rl_adapter.send_to_llm.assert_called_once()
            args = self.rl_adapter.send_to_llm.call_args[0]
            self.assertEqual("feedback_interpretation_request", args[0])
            
            # Restore original method
            self.rl_adapter.send_to_llm = original_send_to_llm
        
        # Restore original method
        self.rl_adapter.action_optimizer.incorporate_feedback = original_incorporate
    
    def test_error_handling(self):
        """Test error handling in the RL adapter."""
        # Invalid task ID
        invalid_task_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.optimize_action",
            content={
                "task_id": "nonexistent_task",
                "action_space": ["action1", "action2"]
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process invalid task message
        self.rl_adapter.process_message(invalid_task_message)
        
        # Verify error response was sent
        self.assertEqual(1, len(self.sent_messages))
        response = self.sent_messages[0]
        self.assertEqual("error", response.message_type)
        self.assertEqual("test_llm", response.target_component)
        self.assertIn("error_code", response.content)
        self.assertFalse(response.content.get("success", True))
        
        # Missing required content
        missing_content_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.create_task",
            content={
                "task_id": "missing_content_task"
                # Missing goal_description
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process missing content message
        self.rl_adapter.process_message(missing_content_message)
        
        # Verify error response was sent
        self.assertEqual(1, len(self.sent_messages))
        response = self.sent_messages[0]
        self.assertEqual("error", response.message_type)
        self.assertFalse(response.content.get("success", True))
    
    def test_task_cancellation(self):
        """Test cancellation of an ongoing task."""
        # Set up a task to be "running"
        self.rl_adapter.active_tasks[self.test_task_id]["status"] = "running"
        
        # Create cancellation message
        cancel_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.cancel_optimization",
            content={
                "task_id": self.test_task_id
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process cancellation message
        self.rl_adapter.process_message(cancel_message)
        
        # Verify task was cancelled
        self.assertEqual("cancelled", self.rl_adapter.active_tasks[self.test_task_id]["status"])
        
        # Verify response was sent
        self.assertEqual(1, len(self.sent_messages))
        response = self.sent_messages[0]
        self.assertEqual("response", response.message_type)
        self.assertTrue(response.content.get("success", False))
        self.assertEqual("cancelled", response.content.get("status"))
    
    def test_learning_params_update(self):
        """Test updating of learning parameters."""
        # Initial parameters
        initial_exploration_rate = self.rl_adapter.learning_state["exploration_rate"]
        
        # Create parameter update message
        params_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.update_learning_params",
            content={
                "command_type": "update_learning_params",
                "params": {
                    "exploration_rate": 0.05,
                    "learning_rate": 0.002
                }
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process parameter update message
        self.rl_adapter.process_message(params_message)
        
        # Verify parameters were updated
        self.assertEqual(0.05, self.rl_adapter.learning_state["exploration_rate"])
        self.assertEqual(0.002, self.rl_adapter.learning_state["learning_rate"])
        
        # Verify response was sent
        self.assertEqual(1, len(self.sent_messages))
        response = self.sent_messages[0]
        self.assertEqual("response", response.message_type)
        self.assertTrue(response.content.get("success", False))
    
    def test_query_handling(self):
        """Test handling of query messages."""
        # Task status query
        status_query = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="technology_query",
            content={
                "query_type": "rl_task_status",
                "task_id": self.test_task_id
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process task status query
        self.rl_adapter.process_message(status_query)
        
        # Verify response was sent
        self.assertEqual(1, len(self.sent_messages))
        response = self.sent_messages[0]
        self.assertEqual("response", response.message_type)
        self.assertTrue(response.content.get("success", False))
        self.assertEqual(self.test_task_id, response.content.get("task_id"))
        
        # Learning state query
        learning_query = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="technology_query",
            content={
                "query_type": "rl_learning_state"
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process learning state query
        self.rl_adapter.process_message(learning_query)
        
        # Verify response was sent
        self.assertEqual(1, len(self.sent_messages))
        response = self.sent_messages[0]
        self.assertEqual("response", response.message_type)
        self.assertTrue(response.content.get("success", False))
        self.assertIn("learning_state", response.content)
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from goal to action optimization."""
        # Create a new task for this test
        task_id = "end_to_end_test"
        
        # 1. Task creation
        task_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.create_task",
            content={
                "task_id": task_id,
                "goal_description": "Optimize resource usage to maximize efficiency",
                "environment_state": {
                    "resources": {"cpu": 0.5, "memory": 0.5, "network": 0.5},
                    "demand": {"cpu": 0.3, "memory": 0.4, "network": 0.2},
                    "performance": 1.0,
                    "efficiency": 1.0
                }
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process task creation
        self.rl_adapter.process_message(task_message)
        
        # 2. Goal interpretation (normally from LLM, mocked here)
        interpretation_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="llm.goal_interpretation",
            content={
                "task_id": task_id,
                "goal_data": {
                    "objective": "maximize_efficiency",
                    "constraints": ["maintain_performance"],
                    "metrics": {"efficiency": "high", "resource_usage": "minimize"}
                }
            }
        )
        
        # Process interpretation
        self.rl_adapter.process_message(interpretation_message)
        
        # 3. Action optimization request
        # Mock optimize method for deterministic results
        original_optimize = self.rl_adapter.action_optimizer.optimize
        self.rl_adapter.action_optimizer.optimize = MagicMock(
            return_value=(
                {"cpu": 0.35, "memory": 0.45, "network": 0.25},  # Best action
                0.9,  # Best reward
                20  # Steps taken
            )
        )
        
        optimization_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.optimize_action",
            content={
                "task_id": task_id,
                "optimization_criteria": {
                    "max_steps": 100
                }
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process optimization request
        self.rl_adapter.process_message(optimization_message)
        
        # Verify response with optimized action
        self.assertEqual(1, len(self.sent_messages))
        response = self.sent_messages[0]
        self.assertEqual("response", response.message_type)
        self.assertTrue(response.content.get("success", False))
        self.assertIn("best_action", response.content)
        
        # 4. Provide feedback
        feedback_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.provide_feedback",
            content={
                "task_id": task_id,
                "feedback": {
                    "reward": 0.85,
                    "comments": "Good allocation, performance maintained"
                }
            }
        )
        
        # Mock feedback incorporation
        self.rl_adapter.action_optimizer.incorporate_feedback = MagicMock(return_value=True)
        
        # Reset tracking
        self.sent_messages = []
        
        # Process feedback
        self.rl_adapter.process_message(feedback_message)
        
        # Verify feedback was received
        self.assertEqual(1, len(self.sent_messages))
        response = self.sent_messages[0]
        self.assertEqual("response", response.message_type)
        self.assertTrue(response.content.get("success", False))
        
        # Restore original methods
        self.rl_adapter.action_optimizer.optimize = original_optimize


if __name__ == "__main__":
    unittest.main()
