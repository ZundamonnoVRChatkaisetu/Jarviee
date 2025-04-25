"""
LLM-RL Integration Test for Jarviee System.

This script tests the integration between the LLM and Reinforcement Learning 
components of the Jarviee system. It verifies that natural language goals and 
instructions can be effectively translated into RL-optimized actions.
"""

import asyncio
import json
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.integration.base import IntegrationMessage, ComponentType
from src.core.integration.adapters.reinforcement_learning.adapter import RLAdapter
from src.core.llm.engine import LLMEngine
from src.core.utils.event_bus import EventBus


class LLMtoRLIntegrationTest(unittest.TestCase):
    """Test the integration between LLM and RL components."""
    
    def setUp(self):
        """Set up the test environment."""
        # Initialize shared event bus
        self.event_bus = EventBus()
        
        # Initialize components with mocked dependencies
        self.llm_engine = LLMEngine(
            engine_id="test_llm",
            model_name="test_model",
            event_bus=self.event_bus
        )
        
        self.rl_adapter = RLAdapter(
            adapter_id="test_rl_adapter",
            llm_component_id="test_llm",
            model_path=None  # No actual model for tests
        )
        
        # Register components with the event bus
        self.event_bus.subscribe("integration.*", self.rl_adapter.process_message)
        
        # Set up message tracking for tests
        self.sent_messages = []
        self.rl_adapter.send_message = self._mock_send_message
        
    def _mock_send_message(self, target, message_type, content, correlation_id=None):
        """Mock the send_message method to track messages."""
        message = IntegrationMessage(
            source_component=self.rl_adapter.component_id,
            target_component=target,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id
        )
        self.sent_messages.append(message)
        return message.message_id
    
    def test_goal_to_reward_conversion(self):
        """Test conversion of language goal to reward function."""
        # Mock the reward generation to avoid actual LLM calls
        self.rl_adapter.reward_generator.generate_from_text = MagicMock(
            return_value=lambda state: 1.0 if "target" in state else 0.0
        )
        
        # Create a test language goal
        goal = "Find the shortest path to the target"
        
        # Send goal message to RL adapter
        goal_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.create_reward",
            content={
                "goal_description": goal,
                "task_id": "test_task_1",
                "environment_context": {
                    "type": "grid_world",
                    "size": [10, 10],
                    "obstacles": [[2, 3], [5, 7]],
                    "target": [9, 9]
                }
            }
        )
        
        # Process the message
        self.rl_adapter.process_message(goal_message)
        
        # Check that a reward function was created
        self.assertIn("test_task_1", self.rl_adapter.active_tasks)
        self.assertIn("reward_function", self.rl_adapter.active_tasks["test_task_1"])
        
        # Check that a response was sent
        self.assertEqual(len(self.sent_messages), 1)
        response = self.sent_messages[0]
        self.assertEqual(response.message_type, "response")
        self.assertEqual(response.target_component, "test_llm")
        self.assertEqual(response.content["status"], "success")
    
    def test_environment_state_management(self):
        """Test environment state representation and updates."""
        # Create an initial environment state
        env_state = {
            "type": "grid_world",
            "size": [10, 10],
            "agent_position": [0, 0],
            "target_position": [9, 9],
            "obstacles": [[2, 3], [5, 7]]
        }
        
        # Set up the environment
        env_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.initialize_environment",
            content={
                "task_id": "test_task_2",
                "environment_state": env_state
            }
        )
        
        # Process the message
        self.rl_adapter.process_message(env_message)
        
        # Check that environment was created
        self.assertIn("test_task_2", self.rl_adapter.active_tasks)
        self.assertIn("environment_state", self.rl_adapter.active_tasks["test_task_2"])
        
        # Update environment state
        update_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.update_environment",
            content={
                "task_id": "test_task_2",
                "update": {
                    "agent_position": [1, 1]
                }
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process the update
        self.rl_adapter.process_message(update_message)
        
        # Check that environment was updated
        updated_state = self.rl_adapter.active_tasks["test_task_2"]["environment_state"]
        self.assertEqual(updated_state["agent_position"], [1, 1])
        
        # Check that a response was sent
        self.assertEqual(len(self.sent_messages), 1)
        response = self.sent_messages[0]
        self.assertEqual(response.message_type, "response")
        self.assertEqual(response.content["status"], "success")
    
    def test_action_optimization(self):
        """Test optimization of actions based on RL."""
        # Mock the action optimizer to avoid actual RL computation
        self.rl_adapter.action_optimizer.optimize = MagicMock(
            return_value={"action": "move_right", "expected_reward": 0.8, "confidence": 0.9}
        )
        
        # Set up task with environment and reward
        setup_task(self.rl_adapter, "test_task_3")
        
        # Request action optimization
        action_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.optimize_action",
            content={
                "task_id": "test_task_3",
                "action_space": ["move_up", "move_down", "move_left", "move_right"],
                "optimization_criteria": {
                    "time_constraint": 5,  # seconds
                    "max_steps": 100
                }
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process the action request
        self.rl_adapter.process_message(action_message)
        
        # Check that a response with optimized action was sent
        self.assertEqual(len(self.sent_messages), 1)
        response = self.sent_messages[0]
        self.assertEqual(response.message_type, "response")
        self.assertEqual(response.content["status"], "success")
        self.assertEqual(response.content["action"], "move_right")
        self.assertIn("expected_reward", response.content)
        self.assertIn("confidence", response.content)
    
    def test_feedback_incorporation(self):
        """Test incorporation of feedback into the learning process."""
        # Set up task with environment and reward
        setup_task(self.rl_adapter, "test_task_4")
        
        # Mock the action optimizer's feedback method
        self.rl_adapter.action_optimizer.incorporate_feedback = MagicMock(return_value=True)
        
        # Send feedback message
        feedback_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.provide_feedback",
            content={
                "task_id": "test_task_4",
                "action": "move_right",
                "feedback": {
                    "reward": 0.5,
                    "state_transition": {
                        "from": {"agent_position": [0, 0]},
                        "to": {"agent_position": [0, 1]}
                    },
                    "feedback_text": "Partial progress toward goal"
                }
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process the feedback
        self.rl_adapter.process_message(feedback_message)
        
        # Verify feedback was incorporated
        self.rl_adapter.action_optimizer.incorporate_feedback.assert_called_once()
        
        # Check that a response was sent
        self.assertEqual(len(self.sent_messages), 1)
        response = self.sent_messages[0]
        self.assertEqual(response.message_type, "response")
        self.assertEqual(response.content["status"], "success")
        self.assertIn("feedback_incorporated", response.content)
        self.assertTrue(response.content["feedback_incorporated"])
    
    def test_end_to_end_workflow(self):
        """Test the complete LLM-to-RL-to-LLM workflow."""
        # Mock all the RL components
        self.rl_adapter.reward_generator.generate_from_text = MagicMock(
            return_value=lambda state: 1.0 if state["agent_position"] == [9, 9] else 0.0
        )
        self.rl_adapter.environment_manager.update_state = MagicMock(
            return_value={"agent_position": [1, 0], "target_position": [9, 9]}
        )
        self.rl_adapter.action_optimizer.optimize = MagicMock(
            return_value={"action": "move_right", "expected_reward": 0.1, "confidence": 0.8}
        )
        
        # 1. Create task with goal
        goal_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.create_task",
            content={
                "task_id": "end_to_end_task",
                "goal_description": "Navigate to the target efficiently",
                "environment_context": {
                    "type": "grid_world",
                    "size": [10, 10],
                    "agent_position": [0, 0],
                    "target_position": [9, 9]
                }
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process goal message
        self.rl_adapter.process_message(goal_message)
        
        # Verify task created
        self.assertIn("end_to_end_task", self.rl_adapter.active_tasks)
        self.assertEqual(len(self.sent_messages), 1)
        
        # 2. Request action
        action_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.optimize_action",
            content={
                "task_id": "end_to_end_task",
                "action_space": ["move_up", "move_down", "move_left", "move_right"],
                "current_state": {"agent_position": [0, 0], "target_position": [9, 9]}
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process action request
        self.rl_adapter.process_message(action_message)
        
        # Verify action response
        self.assertEqual(len(self.sent_messages), 1)
        action_response = self.sent_messages[0]
        self.assertEqual(action_response.content["action"], "move_right")
        
        # 3. Provide feedback
        feedback_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.provide_feedback",
            content={
                "task_id": "end_to_end_task",
                "action": "move_right",
                "feedback": {
                    "reward": 0.1,
                    "state_transition": {
                        "from": {"agent_position": [0, 0]},
                        "to": {"agent_position": [1, 0]}
                    }
                }
            }
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process feedback
        self.rl_adapter.process_message(feedback_message)
        
        # Verify feedback response
        self.assertEqual(len(self.sent_messages), 1)
        feedback_response = self.sent_messages[0]
        self.assertEqual(feedback_response.content["status"], "success")
        
        # 4. Request explanation
        explanation_message = IntegrationMessage(
            source_component="test_llm",
            target_component=self.rl_adapter.component_id,
            message_type="reinforcement_learning.explain_action",
            content={
                "task_id": "end_to_end_task",
                "action": "move_right",
                "format": "natural_language"
            }
        )
        
        # Mock the explanation generator
        self.rl_adapter.action_optimizer.explain_action = MagicMock(
            return_value="Moving right is the best action because it brings the agent closer to the target."
        )
        
        # Reset tracking
        self.sent_messages = []
        
        # Process explanation request
        self.rl_adapter.process_message(explanation_message)
        
        # Verify explanation response
        self.assertEqual(len(self.sent_messages), 1)
        explanation_response = self.sent_messages[0]
        self.assertEqual(explanation_response.content["status"], "success")
        self.assertIn("explanation", explanation_response.content)


def setup_task(adapter, task_id):
    """Helper function to set up a task with environment and reward function."""
    adapter.active_tasks[task_id] = {
        "created_at": time.time(),
        "goal_description": "Test goal",
        "environment_state": {
            "agent_position": [0, 0],
            "target_position": [9, 9]
        },
        "reward_function": lambda state: 1.0 if state["agent_position"] == state["target_position"] else 0.0
    }


if __name__ == "__main__":
    unittest.main()
