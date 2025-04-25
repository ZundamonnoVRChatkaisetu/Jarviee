"""
Test Suite for Reinforcement Learning Adapter.

This module provides comprehensive tests for the Reinforcement Learning adapter,
focusing on its integration with the LLM core. These tests validate the adapter's
functionality, performance, and compatibility with the broader Jarviee system.
"""

import asyncio
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from src.core.integration.adapters.reinforcement_learning.adapter import RLAdapter
from src.core.integration.adapters.reinforcement_learning.reward import RewardFunctionGenerator
from src.core.integration.adapters.reinforcement_learning.environment import EnvironmentStateManager
from src.core.integration.adapters.reinforcement_learning.action import ActionOptimizer
from src.core.integration.base import ComponentType, IntegrationMessage
from src.core.utils.event_bus import EventBus


class MockLLMEngine:
    """Mock LLM Engine for testing."""
    
    def __init__(self):
        self.generate_calls = []
        
    def generate(self, prompt, **kwargs):
        """Mock generate method."""
        self.generate_calls.append((prompt, kwargs))
        
        if "goal" in prompt.lower() and "interpretation" in prompt.lower():
            return json.dumps({
                "interpreted_goal": {
                    "main_objective": "Navigate to target efficiently",
                    "constraints": ["Avoid obstacles", "Minimize energy usage"],
                    "priority": "High",
                    "context": {"environment": "dynamic", "time_sensitive": True}
                },
                "reward_components": {
                    "goal_reached": 1.0,
                    "distance_reduction": 0.3,
                    "energy_efficiency": 0.4,
                    "obstacle_avoidance": 0.8,
                    "time_efficiency": 0.5
                }
            })
        
        elif "enhance" in prompt.lower() and "reward function" in prompt.lower():
            return """```json
{
    "added_components": {
        "smoothness": 0.2,
        "predictability": 0.15
    },
    "adjusted_weights": {
        "energy_efficiency": 0.5,
        "obstacle_avoidance": 0.9
    },
    "refined_constraints": [
        {
            "type": "hard",
            "description": "Never collide with dynamic obstacles",
            "condition": "collision_detected == True"
        }
    ],
    "domain_specific": {
        "domain": "robotics",
        "enhancements": ["Added robot-specific motion constraints"]
    }
}
```"""
        
        elif "explain" in prompt.lower() and "action" in prompt.lower():
            return "The agent chose to move diagonally to reduce distance while avoiding the obstacle to the right. This balances the goal of reaching the target efficiently with the constraint of avoiding collisions."
        
        else:
            return "Generic LLM response for testing"


class TestRLAdapter(unittest.TestCase):
    """Test cases for the Reinforcement Learning Adapter."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.event_bus = EventBus()
        self.mock_llm = MockLLMEngine()
        
        # Initialize adapter with mocked dependencies
        self.adapter = RLAdapter(
            adapter_id="test_rl_adapter",
            llm_component_id="mock_llm"
        )
        
        # Replace components with mocks
        self.adapter.reward_generator = RewardFunctionGenerator()
        self.adapter.reward_generator.initialize(self.mock_llm)
        
        self.adapter.environment_manager = MagicMock()
        self.adapter.action_optimizer = MagicMock()
        
        # Initialize adapter
        self.adapter.initialize()
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        self.assertTrue(self.adapter.is_initialized)
        self.assertEqual(self.adapter.component_id, "test_rl_adapter")
        self.assertEqual(self.adapter.component_type, ComponentType.REINFORCEMENT_LEARNING)
    
    def test_generate_reward_function_from_description(self):
        """Test generating a reward function from a description."""
        goal_description = "Navigate to the target while avoiding obstacles and minimizing energy usage"
        
        # Generate reward function
        reward_function = self.adapter.reward_generator.generate_from_description(
            goal_description=goal_description,
            domain="navigation",
            complexity="medium"
        )
        
        # Verify structure
        self.assertIn("components", reward_function)
        self.assertIn("computation", reward_function)
        self.assertIn("normalization", reward_function)
        self.assertIn("shaping", reward_function)
        self.assertIn("metadata", reward_function)
        
        # Verify components
        components = reward_function["components"]
        self.assertIn("goal_reached", components)
        self.assertIn("obstacle_avoidance", components)
        
        # Values should be reasonable
        for component, value in components.items():
            self.assertIsInstance(value, (int, float))
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)
    
    def test_enhance_reward_function_with_llm(self):
        """Test enhancing a reward function using LLM."""
        goal_description = "Navigate to the target while avoiding obstacles and minimizing energy usage"
        
        # Generate base reward function
        base_reward_function = self.adapter.reward_generator.generate_from_description(
            goal_description=goal_description,
            domain="navigation",
            complexity="medium"
        )
        
        # Enhance with LLM
        enhanced_reward_function = self.adapter.reward_generator.enhance_with_llm(
            reward_function=base_reward_function,
            goal_description=goal_description
        )
        
        # Verify enhancements
        self.assertIn("smoothness", enhanced_reward_function["components"])
        self.assertIn("predictability", enhanced_reward_function["components"])
        
        # Verify weight adjustments
        if "energy_efficiency" in enhanced_reward_function["components"]:
            self.assertEqual(enhanced_reward_function["components"]["energy_efficiency"], 0.5)
            
        if "obstacle_avoidance" in enhanced_reward_function["components"]:
            self.assertEqual(enhanced_reward_function["components"]["obstacle_avoidance"], 0.9)
            
        # Verify metadata
        self.assertIn("llm_enhanced", enhanced_reward_function["metadata"])
        self.assertTrue(enhanced_reward_function["metadata"]["llm_enhanced"])
    
    def test_process_goal_interpretation_message(self):
        """Test processing a goal interpretation message."""
        # Create message
        message = IntegrationMessage(
            source_component="mock_llm",
            target_component="test_rl_adapter",
            message_type="llm.goal_interpretation",
            content={
                "goal_data": {
                    "main_objective": "Navigate to target efficiently",
                    "constraints": ["Avoid obstacles", "Minimize energy usage"],
                    "priority": "High",
                    "context": {"environment": "dynamic", "time_sensitive": True}
                },
                "task_id": "test_task"
            },
            message_id="test_message_id"
        )
        
        # Mock active_tasks
        self.adapter.active_tasks = {
            "test_task": {
                "status": "running",
                "goal_description": "Navigate to target efficiently",
                "environment_state": {},
                "action_type": "navigate",
                "max_steps": 100,
                "current_step": 0
            }
        }
        
        # Patch send_message to capture responses
        with patch.object(self.adapter, 'send_message') as mock_send_message:
            # Process message
            self.adapter._handle_technology_llm_message(message, "goal_interpretation")
            
            # Verify task was updated
            self.assertIn("interpreted_goal", self.adapter.active_tasks["test_task"])
            self.assertIn("reward_function", self.adapter.active_tasks["test_task"])
            
            # Verify notification was sent
            mock_send_message.assert_called()
            args, kwargs = mock_send_message.call_args
            self.assertEqual(args[1], "technology.notification")
    
    def test_optimize_action(self):
        """Test action optimization based on goal and environment."""
        # Create command message
        message = IntegrationMessage(
            source_component="test_component",
            target_component="test_rl_adapter",
            message_type="technology.command",
            content={
                "command_type": "optimize_action",
                "task_id": "test_optimize_task",
                "goal_description": "Navigate to target efficiently",
                "environment_state": {
                    "agent_position": [0, 0],
                    "target_position": [10, 10],
                    "obstacles": [[2, 3], [5, 7]]
                },
                "action_type": "navigate",
                "max_steps": 100
            },
            message_id="test_command_message_id"
        )
        
        # Mock action_optimizer.optimize to return results
        self.adapter.action_optimizer.optimize.return_value = (
            {"type": "move", "direction": [1, 1], "speed": 0.8},  # Best action
            0.75,  # Best reward
            50     # Steps taken
        )
        
        # Patch send_message to capture responses
        with patch.object(self.adapter, 'send_message') as mock_send_message:
            # Handle command
            self.adapter._handle_technology_command(message)
            
            # Verify task was created
            self.assertIn("test_optimize_task", self.adapter.active_tasks)
            
            # Verify acknowledgment was sent
            mock_send_message.assert_called()
            args, kwargs = mock_send_message.call_args_list[0]
            self.assertEqual(args[0], "test_component")
            self.assertEqual(args[1], "response")
            
            # Mock _run_optimization_task to simulate completion
            self.adapter._run_optimization_task("test_optimize_task", "test_component", "test_command_message_id")
            
            # Verify optimize was called
            self.adapter.action_optimizer.optimize.assert_called()
            
            # Verify result was sent
            self.assertEqual(self.adapter.active_tasks["test_optimize_task"]["status"], "completed")
            
            # Check final response
            result_args = mock_send_message.call_args_list[-1][0]
            self.assertEqual(result_args[0], "test_component")
            self.assertEqual(result_args[1], "response")
            self.assertEqual(result_args[2]["status"], "completed")
    
    def test_incorporate_feedback(self):
        """Test incorporating feedback to improve future optimization."""
        # Setup a completed task
        self.adapter.active_tasks = {
            "test_feedback_task": {
                "status": "completed",
                "goal_description": "Navigate to target efficiently",
                "environment_state": {
                    "agent_position": [0, 0],
                    "target_position": [10, 10],
                    "obstacles": [[2, 3], [5, 7]]
                },
                "best_action": {"type": "move", "direction": [1, 1], "speed": 0.8},
                "best_reward": 0.75,
                "reward_function": {"components": {"goal_reached": 1.0}}
            }
        }
        
        # Test feedback incorporation
        result = self.adapter._incorporate_feedback(
            task_id="test_feedback_task",
            feedback={
                "reward": 0.6,
                "comments": "Action was good but could be more energy efficient"
            }
        )
        
        # Verify result
        self.assertTrue(result)
        
        # Verify action optimizer was called
        self.adapter.action_optimizer.incorporate_feedback.assert_called_once()
        
        # Verify learning state was updated
        self.assertIn("feedback_history", self.adapter.learning_state)
        
        # Verify LLM was called for comment interpretation
        self.assertEqual(len(self.mock_llm.generate_calls), 1)
    
    def test_get_component_status(self):
        """Test getting component status."""
        # Setup some active tasks
        self.adapter.active_tasks = {
            "task1": {"status": "running"},
            "task2": {"status": "completed"},
            "task3": {"status": "error"}
        }
        
        # Get status
        status = self.adapter._get_status_impl()
        
        # Verify status content
        self.assertIn("active_tasks", status)
        self.assertEqual(status["active_tasks"], 3)
        self.assertEqual(status["running_tasks"], 1)
        self.assertEqual(status["completed_tasks"], 1)
        self.assertIn("learning_state", status)


class TestLLMRLIntegration(unittest.TestCase):
    """Test cases for the LLM-RL integration."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.event_bus = EventBus()
        self.mock_llm = MockLLMEngine()
        
        # Initialize adapter
        self.rl_adapter = RLAdapter(
            adapter_id="test_rl_adapter",
            llm_component_id="mock_llm"
        )
        
        # Replace components with mocks
        self.rl_adapter.reward_generator = RewardFunctionGenerator()
        self.rl_adapter.reward_generator.initialize(self.mock_llm)
        
        self.rl_adapter.environment_manager = MagicMock()
        self.rl_adapter.action_optimizer = MagicMock()
        
        # Initialize adapter
        self.rl_adapter.initialize()
        
        # Create mock LLM component
        self.mock_llm_component = MagicMock()
        self.mock_llm_component.component_id = "mock_llm"
        self.mock_llm_component.component_type = ComponentType.LLM
        
        # Setup bridge
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
        from src.core.integration.llm_rl_bridge import LLMtoRLBridge
        
        self.bridge = LLMtoRLBridge(
            bridge_id="test_llm_rl_bridge",
            llm_component_id="mock_llm",
            rl_component_id="test_rl_adapter",
            event_bus=self.event_bus
        )
    
    def test_end_to_end_goal_to_action(self):
        """Test end-to-end flow from goal definition to action execution."""
        # Mock components
        self.rl_adapter.action_optimizer.optimize.return_value = (
            {"type": "move", "direction": [1, 1], "speed": 0.8},  # Best action
            0.75,  # Best reward
            50     # Steps taken
        )
        
        # Create goal definition message
        goal_message = IntegrationMessage(
            source_component="mock_llm",
            target_component="test_llm_rl_bridge",
            message_type="llm.goal_definition",
            content={
                "goal_id": "test_e2e_goal",
                "goal_description": "Navigate to the target while avoiding obstacles and minimizing energy usage",
                "priority": 1,
                "constraints": ["Avoid all obstacles", "Keep energy usage below 80%"],
                "deadline": None
            },
            message_id="test_goal_message_id"
        )
        
        # Patch event_bus.publish to capture events
        with patch.object(self.event_bus, 'publish') as mock_publish:
            # Setup mock_publish to simulate event flow
            def simulate_event_flow(event):
                # Extract message if present
                message = None
                if "message" in event.data:
                    message = event.data["message"]
                elif isinstance(event.data, dict) and "content" in event.data:
                    # Create message from event data
                    message = IntegrationMessage(
                        source_component=event.source,
                        target_component=event.data.get("target"),
                        message_type=event.event_type,
                        content=event.data.get("content", {}),
                        message_id=event.event_id
                    )
                
                if not message:
                    return
                    
                # Simulate RL adapter receiving task message
                if (message.target_component == "test_rl_adapter" and 
                        "task_id" in message.content):
                    # Create optimization complete message
                    complete_message = IntegrationMessage(
                        source_component="test_rl_adapter",
                        target_component="test_llm_rl_bridge",
                        message_type="reinforcement_learning.task_completed",
                        content={
                            "task_id": message.content["task_id"],
                            "results": {
                                "best_action": {"type": "move", "direction": [1, 1], "speed": 0.8},
                                "reward": 0.75,
                                "steps_taken": 50,
                                "execution_time": 0.5
                            }
                        },
                        message_id="test_complete_message_id"
                    )
                    
                    # Simulate the bridge receiving and processing the message
                    # This would normally happen through the event bus
                    if hasattr(self.bridge, '_handle_task_completed'):
                        self.bridge._handle_task_completed(Event(
                            event_type="integration.reinforcement_learning.task_completed",
                            source="test_rl_adapter",
                            data={"message": complete_message}
                        ))
            
            mock_publish.side_effect = simulate_event_flow
            
            # Process goal message
            # Normally this would be through the event bus
            self.bridge._handle_goal_definition(Event(
                event_type="integration.llm.goal_definition",
                source="mock_llm",
                data={"message": goal_message}
            ))
            
            # Verify goal was created
            self.assertIn("test_e2e_goal", self.bridge.active_goals)
            
            # Verify task was created
            self.assertEqual(len(self.bridge.active_tasks), 1)
            
            # Get the task ID
            task_id = next(iter(self.bridge.active_tasks))
            
            # Verify task maps to goal
            self.assertEqual(self.bridge.task_to_goal[task_id], "test_e2e_goal")
            
            # Verify bridge sent task to RL component
            mock_publish.assert_called()
            
            # Verify goal is now completed
            task = self.bridge.active_tasks[task_id]
            self.assertEqual(task["status"], "completed")
            self.assertIn("best_action", task)
    
    def test_goal_to_reward_function_conversion(self):
        """Test conversion from goal description to reward function."""
        # Create goal with complex description
        goal_description = """
        Navigate the robot to the charging station when battery is below 20%.
        Prioritize safety by avoiding all obstacles with at least 1 meter margin.
        Use energy efficiently by taking the shortest path when possible.
        If humans are present, reduce speed by 50% and maintain safe distance.
        """
        
        # Generate reward function directly
        reward_function = self.rl_adapter.reward_generator.generate_from_description(
            goal_description=goal_description,
            domain="robotics",
            complexity="complex"
        )
        
        # Verify complex reward function structure
        self.assertIn("components", reward_function)
        self.assertIn("computation", reward_function)
        self.assertIn("normalization", reward_function)
        self.assertIn("shaping", reward_function)
        self.assertIn("explanations", reward_function)
        
        # Verify computation method is appropriate for complex function
        self.assertEqual(reward_function["computation"]["type"], "hierarchical")
        
        # Verify shaping is appropriate for complex function
        self.assertEqual(reward_function["shaping"]["type"], "potential_based")
        
        # Verify key components are included
        components = reward_function["components"]
        self.assertIn("goal_reached", components)
        self.assertIn("obstacle_avoidance", components)
        self.assertIn("energy_efficiency", components)
        
        # Verify safety is highly weighted
        self.assertGreaterEqual(components.get("obstacle_avoidance", 0), 0.7)


if __name__ == "__main__":
    unittest.main()
