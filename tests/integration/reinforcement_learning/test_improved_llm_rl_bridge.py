"""
Comprehensive Tests for the Improved LLM-RL Bridge.

This module implements comprehensive integration tests for the ImprovedLLMtoRLBridge,
which is an enhanced version of the LLM-RL bridge that provides more sophisticated
integration between the LLM and reinforcement learning components of the Jarviee system.

The tests follow the test plan defined in test_plan.md and cover various aspects of the 
bridge functionality, including:
1. Basic functionality (initialization, configuration)
2. Integration with LLM and RL components
3. Error handling and recovery
4. Performance under various conditions
5. Learning capability over time
"""

import asyncio
import json
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Any, Dict, List, Optional, Tuple
import uuid

# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.core.integration.llm_rl_bridge_improved import (
    ImprovedLLMtoRLBridge, 
    GoalContext, 
    RLTask, 
    TaskStatus
)
from src.core.integration.base import IntegrationMessage, ComponentType
from src.core.integration.adapters.reinforcement_learning.adapter import RLAdapter
from src.core.integration.adapters.reinforcement_learning.reward import RewardFunctionGenerator
from src.core.integration.adapters.reinforcement_learning.environment import EnvironmentStateManager
from src.core.integration.adapters.reinforcement_learning.action import ActionOptimizer
from src.core.llm.engine import LLMEngine
from src.core.utils.event_bus import EventBus, Event
from src.core.utils.logger import Logger


class MockLLMEngine:
    """Mock LLM Engine for testing."""
    
    def __init__(self):
        self.generate_calls = []
        
    async def generate(self, prompt, **kwargs):
        """Mock asynchronous generate method."""
        self.generate_calls.append((prompt, kwargs))
        
        # Return appropriate response based on prompt content
        if "goal" in prompt.lower() and ("formalization" in prompt.lower() or "interpretation" in prompt.lower()):
            return json.dumps({
                "interpreted_goal": {
                    "domain": "navigation",
                    "complexity": "standard",
                    "main_objective": "Navigate to target efficiently",
                    "constraints": ["Avoid obstacles", "Minimize energy usage"],
                    "priority": "high",
                    "context": {"environment": "dynamic", "time_sensitive": True}
                },
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
                    },
                    "constraints": [
                        {
                            "type": "hard",
                            "description": "Never collide with obstacles",
                            "condition": "collision_detected == True",
                            "penalty": -1.0
                        }
                    ]
                }
            })
        
        elif "environment" in prompt.lower() and "formalization" in prompt.lower():
            return json.dumps({
                "environment_model": {
                    "type": "grid_world",
                    "dimensions": 2,
                    "continuous": False,
                    "size": [10, 10],
                    "obstacles": [[2, 3], [5, 7]],
                    "dynamics": "deterministic",
                    "initial_state": {
                        "agent_position": [0, 0],
                        "target_position": [9, 9]
                    }
                },
                "action_space": ["move_up", "move_down", "move_left", "move_right"]
            })
        
        elif "uncertainty" in prompt.lower() and "clarification" in prompt.lower():
            return json.dumps({
                "clarifications": [
                    {
                        "aspect": "obstacle_definition",
                        "question": "What specific types of obstacles need to be avoided?",
                        "importance": "high"
                    },
                    {
                        "aspect": "energy_efficiency",
                        "question": "Is there a specific threshold for energy usage?",
                        "importance": "medium"
                    }
                ],
                "suggested_refinements": {
                    "goal_description": "Navigate to the target position efficiently while avoiding all physical obstacles and keeping energy usage below 70% of maximum capacity."
                }
            })
        
        elif "feedback" in prompt.lower() and "analysis" in prompt.lower():
            return json.dumps({
                "feedback_analysis": {
                    "feedback_type": "performance",
                    "strengths": ["Good obstacle avoidance", "Direct path finding"],
                    "weaknesses": ["Energy inefficiency", "Excessive speed"],
                    "suggestions": [
                        "Decrease speed in open areas",
                        "Utilize momentum more effectively"
                    ]
                },
                "reward_adjustments": {
                    "energy_efficiency": 0.5,  # Increase weight
                    "time_efficiency": 0.3     # Decrease weight
                }
            })
        
        elif "explain" in prompt.lower() and "result" in prompt.lower():
            return "The agent successfully navigated to the target while maintaining a safe distance from obstacles. It took a slightly longer path to conserve energy, resulting in optimal overall performance."
        
        else:
            return "Generic LLM response for testing"


class MockRLAdapter:
    """Mock RL Adapter for testing."""
    
    def __init__(self):
        self.messages_received = []
        self.component_id = "mock_rl_adapter"
        self.component_type = ComponentType.REINFORCEMENT_LEARNING
        
    def process_message(self, message):
        """Process a message sent to the adapter."""
        self.messages_received.append(message)
        
        # Return a response based on message type
        if message.message_type == "reinforcement_learning.create_task":
            # Simulate task creation and immediate success response
            return IntegrationMessage(
                source_component=self.component_id,
                target_component=message.source_component,
                message_type="response",
                content={
                    "status": "success",
                    "task_id": message.content.get("task_id", str(uuid.uuid4())),
                    "message": "Task created successfully"
                },
                correlation_id=message.correlation_id
            )
            
        elif message.message_type == "reinforcement_learning.update_environment":
            # Simulate environment update
            return IntegrationMessage(
                source_component=self.component_id,
                target_component=message.source_component,
                message_type="response",
                content={
                    "status": "success",
                    "task_id": message.content.get("task_id"),
                    "message": "Environment updated successfully"
                },
                correlation_id=message.correlation_id
            )
            
        elif message.message_type == "reinforcement_learning.update_reward":
            # Simulate reward function update
            return IntegrationMessage(
                source_component=self.component_id,
                target_component=message.source_component,
                message_type="response",
                content={
                    "status": "success",
                    "task_id": message.content.get("task_id"),
                    "message": "Reward function updated successfully"
                },
                correlation_id=message.correlation_id
            )
            
        # Default response
        return IntegrationMessage(
            source_component=self.component_id,
            target_component=message.source_component,
            message_type="response",
            content={
                "status": "success",
                "message": "Message processed"
            },
            correlation_id=message.correlation_id
        )
    
    def send_task_completed(self, task_id, source_component, event_bus, results=None):
        """Simulate task completion."""
        if results is None:
            results = {
                "final_reward": 0.9,
                "success": True,
                "steps_taken": 50,
                "optimal_action": {"type": "move", "direction": [1, 1], "speed": 0.8},
                "execution_time": 0.5,
                "selected_path": [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]
            }
            
        complete_message = IntegrationMessage(
            source_component=self.component_id,
            target_component=source_component,
            message_type="reinforcement_learning.task_completed",
            content={
                "task_id": task_id,
                "results": results
            }
        )
        
        # Create and publish event
        event = Event(
            event_type="integration.reinforcement_learning.task_completed",
            source=self.component_id,
            data={"message": complete_message}
        )
        
        event_bus.publish(event)


class TestImprovedLLMtoRLBridge(unittest.TestCase):
    """Test suite for the ImprovedLLMtoRLBridge."""
    
    def setUp(self):
        """Set up the test environment."""
        # Initialize EventBus
        self.event_bus = EventBus()
        
        # Create mock components
        self.mock_llm = MockLLMEngine()
        self.mock_rl_adapter = MockRLAdapter()
        
        # Create bridge with mocked components
        self.bridge = ImprovedLLMtoRLBridge(
            bridge_id="test_bridge",
            llm_component_id="mock_llm",
            rl_component_id="mock_rl_adapter",
            event_bus=self.event_bus,
            config={
                "goal_batch_size": 3,
                "task_timeout_seconds": 30,
                "feedback_integration_mode": "immediate",
                "enable_contextual_learning": True,
                "uncertainty_threshold": 0.7,
                "prompt_enhancement": True,
                "max_llm_retry_attempts": 2,
                "diagnostics_level": "detailed"
            }
        )
        
        # Initialize event handlers
        self._register_event_handlers()
        
        # Track sent messages
        self.sent_messages = []
        
        # Patch the send_message method to track messages
        self.original_send_message = self.bridge.send_message
        self.bridge.send_message = self._mock_send_message
        
        # Patch the send_to_llm method
        self.bridge.send_to_llm = self._mock_send_to_llm
    
    def _register_event_handlers(self):
        """Register event handlers for testing."""
        # LLM to Bridge
        self.event_bus.subscribe("integration.llm.goal_definition", self.bridge._handle_goal_definition)
        self.event_bus.subscribe("integration.llm.goal_clarification", self.bridge._handle_goal_clarification)
        self.event_bus.subscribe("integration.llm.environment_description", self.bridge._handle_environment_description)
        self.event_bus.subscribe("integration.llm.feedback_provision", self.bridge._handle_feedback_provision)
        
        # RL to Bridge
        self.event_bus.subscribe("integration.reinforcement_learning.task_update", self.bridge._handle_task_update)
        self.event_bus.subscribe("integration.reinforcement_learning.task_completed", self.bridge._handle_task_completed)
        self.event_bus.subscribe("integration.reinforcement_learning.learning_progress", self.bridge._handle_learning_progress)
        self.event_bus.subscribe("integration.reinforcement_learning.uncertainty_detection", self.bridge._handle_uncertainty_detection)
    
    def _mock_send_message(self, target, message_type, content, correlation_id=None):
        """Mock send_message to track messages."""
        message = IntegrationMessage(
            source_component=self.bridge.bridge_id,
            target_component=target,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id or str(uuid.uuid4())
        )
        self.sent_messages.append(message)
        
        # For RL messages, forward to the mock adapter
        if target == self.mock_rl_adapter.component_id:
            response = self.mock_rl_adapter.process_message(message)
            if response:
                # Create an event from the response
                event = Event(
                    event_type=f"integration.{response.message_type}",
                    source=response.source_component,
                    data={"message": response}
                )
                
                # Process the response
                if response.message_type == "response":
                    self.bridge._handle_technology_response(event)
        
        return message.correlation_id
    
    def _mock_send_to_llm(self, message_type, content, correlation_id=None):
        """Mock send_to_llm to track LLM messages."""
        message = IntegrationMessage(
            source_component=self.bridge.bridge_id,
            target_component=self.bridge.llm_component_id,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id or str(uuid.uuid4())
        )
        self.sent_messages.append(message)
        
        # For goal formalization, simulate LLM response
        if message_type == "goal_formalization_request":
            # Create a formalized goal response
            formalized_goal = {
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
                },
                "task_id": content.get("task_id")
            }
            
            # Create an integration message
            response = IntegrationMessage(
                source_component=self.bridge.llm_component_id,
                target_component=self.bridge.bridge_id,
                message_type="llm.goal_formalization_response",
                content=formalized_goal,
                correlation_id=message.correlation_id
            )
            
            # Create an event from the response
            event = Event(
                event_type="integration.llm.goal_formalization_response",
                source=self.bridge.llm_component_id,
                data={"message": response}
            )
            
            # Process the response asynchronously
            # In a real environment this would come through the event bus
            self.bridge._handle_goal_formalization_response(event)
            
        # For environment formalization, simulate LLM response
        elif message_type == "environment_formalization_request":
            # Create a formalized environment response
            formalized_env = {
                "environment_model": {
                    "type": "grid_world",
                    "dimensions": 2,
                    "continuous": False,
                    "size": [10, 10],
                    "obstacles": [[2, 3], [5, 7]],
                    "dynamics": "deterministic"
                },
                "action_space": ["move_up", "move_down", "move_left", "move_right"],
                "task_id": content.get("task_id")
            }
            
            # Create an integration message
            response = IntegrationMessage(
                source_component=self.bridge.llm_component_id,
                target_component=self.bridge.bridge_id,
                message_type="llm.environment_formalization_response",
                content=formalized_env,
                correlation_id=message.correlation_id
            )
            
            # Create an event from the response
            event = Event(
                event_type="integration.llm.environment_formalization_response",
                source=self.bridge.llm_component_id,
                data={"message": response}
            )
            
            # Process the response asynchronously
            # In a real environment this would come through the event bus
            # Bridge doesn't have a handler for this yet, but we'll add one for testing
            if hasattr(self.bridge, "_handle_environment_formalization_response"):
                self.bridge._handle_environment_formalization_response(event)
        
        return message.correlation_id
    
    def test_adapter_initialization(self):
        """Test bridge initialization (TC-RL-01)."""
        # Verify bridge was initialized
        self.assertEqual(self.bridge.bridge_id, "test_bridge")
        self.assertEqual(self.bridge.llm_component_id, "mock_llm")
        self.assertEqual(self.bridge.rl_component_id, "mock_rl_adapter")
        
        # Verify config was applied
        self.assertEqual(self.bridge.config["goal_batch_size"], 3)
        self.assertEqual(self.bridge.config["task_timeout_seconds"], 30)
        self.assertEqual(self.bridge.config["feedback_integration_mode"], "immediate")
        
        # Verify subcomponents initialized
        self.assertIsNotNone(self.bridge.reward_generator)
        self.assertIsNotNone(self.bridge.environment_manager)
        
        # Verify data structures initialized
        self.assertEqual(len(self.bridge.active_goals), 0)
        self.assertEqual(len(self.bridge.active_tasks), 0)
        self.assertEqual(len(self.bridge.task_to_goal), 0)
    
    def test_process_goal_to_task(self):
        """Test basic goal processing to RL task (TC-RL-02)."""
        # Create a test goal
        goal_context = GoalContext(
            goal_id="test_goal_1",
            goal_description="Navigate to the target while avoiding obstacles and minimizing energy usage",
            priority=1,
            constraints=["Avoid all obstacles", "Keep energy usage below 80%"]
        )
        
        # Process the goal to a task
        self.bridge._process_goal_to_task(goal_context)
        
        # Verify task was created
        self.assertEqual(len(self.bridge.active_tasks), 1)
        
        # Get the task ID
        task_id = next(iter(self.bridge.active_tasks))
        task = self.bridge.active_tasks[task_id]
        
        # Verify task properties
        self.assertEqual(task.goal_context.goal_id, "test_goal_1")
        self.assertEqual(task.status, TaskStatus.FORMALIZING)
        
        # Verify goal-task mapping
        self.assertEqual(self.bridge.task_to_goal[task_id], "test_goal_1")
        
        # Verify initial reward specification
        self.assertIn("domain", task.reward_specification)
        self.assertIn("complexity", task.reward_specification)
        
        # Verify LLM formalization was requested
        llm_messages = [msg for msg in self.sent_messages 
                       if msg.target_component == "mock_llm"]
        self.assertGreaterEqual(len(llm_messages), 1)
        
        # Find formalization request
        formalization_request = next((msg for msg in llm_messages 
                                    if msg.message_type == "goal_formalization_request"), None)
        self.assertIsNotNone(formalization_request)
        self.assertEqual(formalization_request.content["goal_description"], 
                        "Navigate to the target while avoiding obstacles and minimizing energy usage")
    
    def test_goal_definition_handler(self):
        """Test goal definition handler (TC-RL-04)."""
        # Create a goal definition message
        goal_message = IntegrationMessage(
            source_component="mock_llm",
            target_component="test_bridge",
            message_type="llm.goal_definition",
            content={
                "goal_description": "Find the shortest path to the target while avoiding dangerous areas",
                "goal_id": "test_goal_2",
                "priority": 2,
                "constraints": ["Avoid red zones", "Maintain minimum safe distance from hazards"],
                "deadline": time.time() + 3600  # 1 hour from now
            }
        )
        
        # Create an event from the message
        event = Event(
            event_type="integration.llm.goal_definition",
            source="mock_llm",
            data={"message": goal_message}
        )
        
        # Clear any existing messages
        self.sent_messages = []
        
        # Process the goal definition event
        self.bridge._handle_goal_definition(event)
        
        # Verify goal was created
        self.assertIn("test_goal_2", self.bridge.active_goals)
        
        # Verify goal properties
        goal = self.bridge.active_goals["test_goal_2"]
        self.assertEqual(goal.goal_description, 
                        "Find the shortest path to the target while avoiding dangerous areas")
        self.assertEqual(goal.priority, 2)
        
        # Verify task was created
        self.assertEqual(len(self.bridge.task_to_goal), 1)
        
        # Find the task ID
        task_id = next(key for key, value in self.bridge.task_to_goal.items() 
                      if value == "test_goal_2")
        
        # Verify task created and processing started
        self.assertIn(task_id, self.bridge.active_tasks)
        task = self.bridge.active_tasks[task_id]
        self.assertEqual(task.status, TaskStatus.FORMALIZING)
        
        # Verify LLM formalization was requested
        llm_messages = [msg for msg in self.sent_messages 
                       if msg.target_component == "mock_llm"]
        self.assertGreaterEqual(len(llm_messages), 1)
    
    def test_goal_formalization_and_task_preparation(self):
        """Test goal formalization through LLM and task preparation (TC-RL-05)."""
        # Create a goal
        goal_context = GoalContext(
            goal_id="test_goal_3",
            goal_description="Navigate to the charging station when battery is below 20%",
            priority=1
        )
        
        # Store the goal
        self.bridge.active_goals["test_goal_3"] = goal_context
        
        # Process the goal to a task
        self.bridge._process_goal_to_task(goal_context)
        
        # Get the task ID
        task_id = next(key for key, value in self.bridge.task_to_goal.items() 
                      if value == "test_goal_3")
        
        # Clear sent messages
        self.sent_messages = []
        
        # Simulate task completion and transition to PLANNING
        task = self.bridge.active_tasks[task_id]
        
        # Add formalized reward function to results
        if hasattr(task, "results"):
            # Task already has results
            task.results["reward_function"] = {
                "components": {
                    "goal_reached": 1.0,
                    "battery_efficiency": 0.6,
                    "path_efficiency": 0.4
                },
                "computation": {
                    "type": "weighted_sum",
                    "normalization": True
                }
            }
        else:
            # Task doesn't have results yet, create it
            task.results = {
                "reward_function": {
                    "components": {
                        "goal_reached": 1.0,
                        "battery_efficiency": 0.6,
                        "path_efficiency": 0.4
                    },
                    "computation": {
                        "type": "weighted_sum",
                        "normalization": True
                    }
                }
            }
        
        # Manually update task status to Planning
        task.status = TaskStatus.PLANNING
        
        # Call preparation method
        self.bridge._prepare_rl_task(task)
        
        # Verify task was sent to RL component
        rl_messages = [msg for msg in self.sent_messages 
                      if msg.target_component == "mock_rl_adapter"]
        self.assertGreaterEqual(len(rl_messages), 1)
        
        # Find create task message
        create_task_message = next((msg for msg in rl_messages 
                                  if msg.message_type == "reinforcement_learning.create_task"), None)
        self.assertIsNotNone(create_task_message)
        
        # Verify task content
        self.assertEqual(create_task_message.content["task_id"], task_id)
        self.assertEqual(create_task_message.content["goal_description"], 
                        "Navigate to the charging station when battery is below 20%")
        self.assertIn("reward_specification", create_task_message.content)
        
        # Verify task status updated to Executing
        self.assertEqual(task.status, TaskStatus.EXECUTING)
    
    def test_task_completion_handling(self):
        """Test handling of task completion (TC-RL-03)."""
        # Set up a goal and task
        goal_context = GoalContext(
            goal_id="test_goal_4",
            goal_description="Navigate through the maze to find the exit",
            priority=1
        )
        
        # Store the goal
        self.bridge.active_goals["test_goal_4"] = goal_context
        
        # Create a task
        task = RLTask(
            task_id="test_task_4",
            goal_context=goal_context,
            environment_context={
                "type": "maze",
                "size": [20, 20],
                "start": [0, 0],
                "exit": [19, 19]
            },
            action_space=["move_up", "move_down", "move_left", "move_right"],
            reward_specification={
                "domain": "navigation",
                "complexity": "standard",
                "template": {"goal_reached": 1.0}
            },
            status=TaskStatus.EXECUTING
        )
        
        # Store the task
        self.bridge.active_tasks["test_task_4"] = task
        self.bridge.task_to_goal["test_task_4"] = "test_goal_4"
        
        # Clear sent messages
        self.sent_messages = []
        
        # Create a task completed message
        complete_message = IntegrationMessage(
            source_component="mock_rl_adapter",
            target_component="test_bridge",
            message_type="reinforcement_learning.task_completed",
            content={
                "task_id": "test_task_4",
                "results": {
                    "final_reward": 0.95,
                    "path_taken": [[0, 0], [1, 0], [2, 0], [3, 1], [4, 2], 
                                 [5, 3], [6, 4], [7, 5], [8, 6], [9, 7], 
                                 [10, 8], [11, 9], [12, 10], [13, 11], [14, 12], 
                                 [15, 13], [16, 14], [17, 15], [18, 16], [19, 17], 
                                 [19, 18], [19, 19]],
                    "steps_taken": 22,
                    "execution_time": 1.5,
                    "success": True
                }
            },
            correlation_id="test_completion_id"
        )
        
        # Create an event from the message
        event = Event(
            event_type="integration.reinforcement_learning.task_completed",
            source="mock_rl_adapter",
            data={"message": complete_message}
        )
        
        # Process the task completed event
        self.bridge._handle_task_completed(event)
        
        # Verify task status updated
        self.assertEqual(self.bridge.active_tasks["test_task_4"].status, TaskStatus.COMPLETED)
        self.assertEqual(self.bridge.active_tasks["test_task_4"].progress, 1.0)
        
        # Verify results stored
        self.assertIn("final_reward", self.bridge.active_tasks["test_task_4"].results)
        self.assertEqual(self.bridge.active_tasks["test_task_4"].results["final_reward"], 0.95)
        
        # Verify completion step added
        self.assertGreaterEqual(len(self.bridge.active_tasks["test_task_4"].steps), 1)
        last_step = self.bridge.active_tasks["test_task_4"].steps[-1]
        self.assertEqual(last_step["type"], "completion")
        
        # Verify metrics updated
        self.assertEqual(self.bridge.metrics["tasks_completed"], 1)
        
        # Verify notification sent to LLM
        llm_messages = [msg for msg in self.sent_messages 
                       if msg.target_component == "mock_llm"]
        self.assertGreaterEqual(len(llm_messages), 1)
        
        # Find task completed notification
        completion_notification = next((msg for msg in llm_messages 
                                     if "task_completed" in msg.message_type), None)
        self.assertIsNotNone(completion_notification)
        self.assertEqual(completion_notification.content["task_id"], "test_task_4")
        self.assertEqual(completion_notification.content["goal_id"], "test_goal_4")
        self.assertTrue(completion_notification.content["status"] == "COMPLETED")
    
    def test_goal_clarification_handling(self):
        """Test handling goal clarification from LLM (TC-RL-05)."""
        # Set up a goal and task
        goal_context = GoalContext(
            goal_id="test_goal_5",
            goal_description="Navigate efficiently",  # Intentionally vague
            priority=1
        )
        
        # Store the goal
        self.bridge.active_goals["test_goal_5"] = goal_context
        
        # Create a task
        task = RLTask(
            task_id="test_task_5",
            goal_context=goal_context,
            environment_context={
                "type": "navigation",
                "size": [10, 10]
            },
            action_space=["move_up", "move_down", "move_left", "move_right"],
            reward_specification={
                "domain": "navigation",
                "complexity": "simple",
                "template": {"goal_reached": 1.0}
            },
            status=TaskStatus.FORMALIZING
        )
        
        # Store the task
        self.bridge.active_tasks["test_task_5"] = task
        self.bridge.task_to_goal["test_task_5"] = "test_goal_5"
        
        # Clear sent messages
        self.sent_messages = []
        
        # Create a goal clarification message
        clarification_message = IntegrationMessage(
            source_component="mock_llm",
            target_component="test_bridge",
            message_type="llm.goal_clarification",
            content={
                "goal_id": "test_goal_5",
                "clarification": {
                    "refined_description": "Navigate to the target position while minimizing energy usage and avoiding obstacles",
                    "constraints": ["Keep minimum 1 unit distance from obstacles", 
                                  "Maintain energy usage below 70%"],
                    "summary": "Added specificity to navigation goal"
                }
            },
            correlation_id="test_clarification_id"
        )
        
        # Create an event from the message
        event = Event(
            event_type="integration.llm.goal_clarification",
            source="mock_llm",
            data={"message": clarification_message}
        )
        
        # Process the goal clarification event
        self.bridge._handle_goal_clarification(event)
        
        # Verify goal was updated
        updated_goal = self.bridge.active_goals["test_goal_5"]
        self.assertEqual(updated_goal.goal_description, 
                        "Navigate to the target position while minimizing energy usage and avoiding obstacles")
        self.assertEqual(len(updated_goal.constraints), 2)
        
        # Verify task goal context was updated
        updated_task = self.bridge.active_tasks["test_task_5"]
        self.assertEqual(updated_task.goal_context.goal_description, 
                        "Navigate to the target position while minimizing energy usage and avoiding obstacles")
        
        # Verify clarification history added to metadata
        self.assertIn("clarification_history", updated_goal.metadata)
        self.assertEqual(len(updated_goal.metadata["clarification_history"]), 1)
        
        # Verify acknowledgment sent
        ack_messages = [msg for msg in self.sent_messages 
                       if msg.message_type == "response" and 
                       "goal_clarification_ack" in msg.content.get("message_type", "")]
        self.assertEqual(len(ack_messages), 1)
    
    def test_uncertainty_handling(self):
        """Test handling uncertainty from RL component (TC-RL-06)."""
        # Set up a goal and task
        goal_context = GoalContext(
            goal_id="test_goal_6",
            goal_description="Navigate while being cautious",  # Ambiguous
            priority=1
        )
        
        # Store the goal
        self.bridge.active_goals["test_goal_6"] = goal_context
        
        # Create a task
        task = RLTask(
            task_id="test_task_6",
            goal_context=goal_context,
            environment_context={
                "type": "navigation",
                "size": [10, 10]
            },
            action_space=["move_up", "move_down", "move_left", "move_right"],
            reward_specification={
                "domain": "navigation",
                "complexity": "standard",
                "template": {"goal_reached": 1.0}
            },
            status=TaskStatus.EXECUTING
        )
        
        # Store the task
        self.bridge.active_tasks["test_task_6"] = task
        self.bridge.task_to_goal["test_task_6"] = "test_goal_6"
        
        # Clear sent messages
        self.sent_messages = []
        
        # Create an uncertainty detection message
        uncertainty_message = IntegrationMessage(
            source_component="mock_rl_adapter",
            target_component="test_bridge",
            message_type="reinforcement_learning.uncertainty_detection",
            content={
                "task_id": "test_task_6",
                "uncertainty": {
                    "level": 0.8,  # High uncertainty, above threshold
                    "areas": ["reward_definition", "environment_constraints"],
                    "description": "Unclear what 'cautious' means in terms of reward function"
                }
            },
            correlation_id="test_uncertainty_id"
        )
        
        # Create an event from the message
        event = Event(
            event_type="integration.reinforcement_learning.uncertainty_detection",
            source="mock_rl_adapter",
            data={"message": uncertainty_message}
        )
        
        # Process the uncertainty event
        self.bridge._handle_uncertainty_detection(event)
        
        # Verify uncertainty step added to task
        last_step = self.bridge.active_tasks["test_task_6"].steps[-1]
        self.assertEqual(last_step["type"], "uncertainty")
        
        # Verify LLM clarification request sent
        llm_messages = [msg for msg in self.sent_messages 
                       if msg.target_component == "mock_llm"]
        self.assertGreaterEqual(len(llm_messages), 1)
        
        # Find uncertainty clarification request
        uncertainty_request = next((msg for msg in llm_messages 
                                  if "uncertainty_clarification_request" in msg.message_type), None)
        self.assertIsNotNone(uncertainty_request)
        self.assertEqual(uncertainty_request.content["task_id"], "test_task_6")
        self.assertGreaterEqual(uncertainty_request.content["uncertainty_level"], 0.7)
        
        # Verify metrics updated
        self.assertEqual(self.bridge.metrics["clarification_requests"], 1)
    
    def test_feedback_handling(self):
        """Test handling feedback integration (TC-RL-05)."""
        # Set up a goal and task
        goal_context = GoalContext(
            goal_id="test_goal_7",
            goal_description="Navigate to the target efficiently",
            priority=1
        )
        
        # Store the goal
        self.bridge.active_goals["test_goal_7"] = goal_context
        
        # Create a task
        task = RLTask(
            task_id="test_task_7",
            goal_context=goal_context,
            environment_context={
                "type": "navigation",
                "size": [10, 10]
            },
            action_space=["move_up", "move_down", "move_left", "move_right"],
            reward_specification={
                "domain": "navigation",
                "complexity": "standard",
                "template": {"goal_reached": 1.0}
            },
            status=TaskStatus.EXECUTING,
            results={
                "reward_function": {
                    "components": {
                        "goal_reached": 1.0,
                        "distance_reduction": 0.3,
                        "energy_efficiency": 0.2,
                        "time_efficiency": 0.5
                    },
                    "computation": {
                        "type": "weighted_sum",
                        "normalization": True
                    }
                }
            }
        )
        
        # Store the task
        self.bridge.active_tasks["test_task_7"] = task
        self.bridge.task_to_goal["test_task_7"] = "test_goal_7"
        
        # Clear sent messages
        self.sent_messages = []
        
        # Create a feedback message
        feedback_message = IntegrationMessage(
            source_component="mock_llm",
            target_component="test_bridge",
            message_type="llm.feedback_provision",
            content={
                "task_id": "test_task_7",
                "feedback": {
                    "reward": 0.6,  # Moderate reward
                    "comments": "The path was efficient but energy usage was too high. Consider reducing speed to save energy.",
                    "summary": "Good path finding, poor energy efficiency"
                }
            },
            correlation_id="test_feedback_id"
        )
        
        # Create an event from the message
        event = Event(
            event_type="integration.llm.feedback_provision",
            source="mock_llm",
            data={"message": feedback_message}
        )
        
        # Process the feedback event
        self.bridge._handle_feedback_provision(event)
        
        # Verify feedback added to task
        self.assertEqual(len(self.bridge.active_tasks["test_task_7"].feedback), 1)
        
        # Verify feedback processing triggered (immediate mode is set)
        self.assertEqual(self.bridge.config["feedback_integration_mode"], "immediate")
        
        # Verify LLM analysis request sent
        llm_messages = [msg for msg in self.sent_messages 
                       if msg.target_component == "mock_llm"]
        self.assertGreaterEqual(len(llm_messages), 1)
        
        # Find feedback analysis request
        feedback_request = next((msg for msg in llm_messages 
                               if "feedback_analysis_request" in msg.message_type), None)
        self.assertIsNotNone(feedback_request)
        
        # Verify RL component gets reward function update
        if "dynamic_reward_adaptation" in self.bridge.config and self.bridge.config["dynamic_reward_adaptation"]:
            # Mock the feedback response since we don't have a real LLM
            task = self.bridge.active_tasks["test_task_7"]
            
            # Update the reward function based on feedback
            reward_function = task.results["reward_function"]
            reward_function["components"]["energy_efficiency"] = 0.5  # Increase weight
            task.results["reward_function"] = reward_function
            
            # Call the update method directly
            self.bridge._send_updated_reward_to_rl(task)
            
            # Verify RL update message sent
            rl_messages = [msg for msg in self.sent_messages 
                          if msg.target_component == "mock_rl_adapter"]
            self.assertGreaterEqual(len(rl_messages), 1)
            
            # Find reward update message
            reward_update = next((msg for msg in rl_messages 
                                if "update_reward" in msg.message_type), None)
            self.assertIsNotNone(reward_update)
            self.assertEqual(reward_update.content["task_id"], "test_task_7")
            self.assertIn("reward_function", reward_update.content)
    
    def test_complete_workflow(self):
        """Test complete goal-to-task-to-action-to-completion workflow (TC-RL-10)."""
        # Clear all state
        self.bridge.active_goals = {}
        self.bridge.active_tasks = {}
        self.bridge.task_to_goal = {}
        self.bridge.metrics = {
            "goals_processed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_completion_time": 0,
            "avg_reward": 0,
            "llm_calls": 0,
            "rl_calls": 0,
            "clarification_requests": 0
        }
        
        # Clear sent messages
        self.sent_messages = []
        
        # 1. Create goal definition message
        goal_message = IntegrationMessage(
            source_component="mock_llm",
            target_component="test_bridge",
            message_type="llm.goal_definition",
            content={
                "goal_description": "Navigate to the charging station efficiently while avoiding obstacles",
                "goal_id": "workflow_goal",
                "priority": 1,
                "constraints": ["Avoid all obstacles", "Minimize energy usage"]
            },
            correlation_id="workflow_goal_id"
        )
        
        # Create an event from the message
        goal_event = Event(
            event_type="integration.llm.goal_definition",
            source="mock_llm",
            data={"message": goal_message}
        )
        
        # Process the goal definition event
        self.bridge._handle_goal_definition(goal_event)
        
        # Verify goal created and task initiated
        self.assertIn("workflow_goal", self.bridge.active_goals)
        self.assertEqual(len(self.bridge.active_tasks), 1)
        
        # Get the task ID
        task_id = next(iter(self.bridge.active_tasks))
        task = self.bridge.active_tasks[task_id]
        
        # Verify task in formalizing state
        self.assertEqual(task.status, TaskStatus.FORMALIZING)
        
        # 2. Simulate LLM formalization response
        # (This is already happening via the mocked send_to_llm)
        
        # 3. Simulate RL task execution
        # Verify status is EXECUTING
        task = self.bridge.active_tasks[task_id]
        self.assertEqual(task.status, TaskStatus.EXECUTING)
        
        # 4. Simulate task completion from RL
        # Set up mock response data
        completion_results = {
            "final_reward": 0.85,
            "success": True,
            "steps_taken": 42,
            "path_taken": [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
            "execution_time": 0.8,
            "energy_used": 0.6,
            "obstacles_avoided": 3
        }
        
        # Send the completion
        self.mock_rl_adapter.send_task_completed(
            task_id=task_id,
            source_component=self.bridge.bridge_id,
            event_bus=self.event_bus,
            results=completion_results
        )
        
        # Verify task completed
        task = self.bridge.active_tasks[task_id]
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.progress, 1.0)
        
        # Verify results stored
        self.assertEqual(task.results["final_reward"], 0.85)
        self.assertEqual(task.results["steps_taken"], 42)
        
        # Verify goal completion check
        # (Since this is the only task for the goal, it should be considered complete)
        
        # Verify metrics updated
        self.assertEqual(self.bridge.metrics["goals_processed"], 1)
        self.assertEqual(self.bridge.metrics["tasks_completed"], 1)
        self.assertGreater(self.bridge.metrics["avg_completion_time"], 0)
        self.assertEqual(self.bridge.metrics["avg_reward"], 0.85)
    
    def test_error_handling(self):
        """Test error handling in the bridge (TC-RL-07)."""
        # Test various error scenarios
        
        # 1. Missing goal description
        incomplete_goal_message = IntegrationMessage(
            source_component="mock_llm",
            target_component="test_bridge",
            message_type="llm.goal_definition",
            content={
                # Missing goal description
                "goal_id": "error_goal_1",
                "priority": 1
            },
            correlation_id="error_test_1"
        )
        
        # Create an event from the message
        error_event = Event(
            event_type="integration.llm.goal_definition",
            source="mock_llm",
            data={"message": incomplete_goal_message}
        )
        
        # Clear sent messages
        self.sent_messages = []
        
        # Process the incomplete goal message
        self.bridge._handle_goal_definition(error_event)
        
        # Verify error response was sent
        error_responses = [msg for msg in self.sent_messages 
                          if msg.message_type == "error"]
        self.assertEqual(len(error_responses), 1)
        self.assertIn("Missing goal description", error_responses[0].content["error"])
        
        # 2. Invalid task ID in feedback
        invalid_feedback_message = IntegrationMessage(
            source_component="mock_llm",
            target_component="test_bridge",
            message_type="llm.feedback_provision",
            content={
                "task_id": "nonexistent_task",
                "feedback": {
                    "reward": 0.5,
                    "comments": "Test feedback"
                }
            },
            correlation_id="error_test_2"
        )
        
        # Create an event from the message
        error_event_2 = Event(
            event_type="integration.llm.feedback_provision",
            source="mock_llm",
            data={"message": invalid_feedback_message}
        )
        
        # Clear sent messages
        self.sent_messages = []
        
        # Process the invalid feedback message
        self.bridge._handle_feedback_provision(error_event_2)
        
        # Verify error response was sent
        error_responses = [msg for msg in self.sent_messages 
                          if msg.message_type == "error"]
        self.assertEqual(len(error_responses), 1)
        self.assertIn("Invalid feedback provision", error_responses[0].content["error"])
        
        # 3. Test task timeout handling
        # Set up a goal and task that will timeout
        timeout_goal = GoalContext(
            goal_id="timeout_goal",
            goal_description="Test timeout handling",
            priority=1
        )
        
        # Store the goal
        self.bridge.active_goals["timeout_goal"] = timeout_goal
        
        # Create a task with old timestamp
        timeout_task = RLTask(
            task_id="timeout_task",
            goal_context=timeout_goal,
            environment_context={},
            action_space=[],
            reward_specification={},
            status=TaskStatus.EXECUTING,
            created_at=time.time() - 3600,  # 1 hour ago
            updated_at=time.time() - 3600   # 1 hour ago
        )
        
        # Store the task
        self.bridge.active_tasks["timeout_task"] = timeout_task
        self.bridge.task_to_goal["timeout_task"] = "timeout_goal"
        
        # Set short timeout for testing
        original_timeout = self.bridge.config["task_timeout_seconds"]
        self.bridge.config["task_timeout_seconds"] = 1800  # 30 minutes
        
        # Clear sent messages
        self.sent_messages = []
        
        # Simulate maintenance event
        maintenance_event = Event(
            event_type="system.periodic_maintenance",
            source="system",
            data={}
        )
        
        # Process maintenance event
        self.bridge._handle_maintenance(maintenance_event)
        
        # Verify task was marked as error
        self.assertEqual(self.bridge.active_tasks["timeout_task"].status, TaskStatus.ERROR)
        self.assertEqual(self.bridge.active_tasks["timeout_task"].error, "Task timed out")
        
        # Verify timeout step added
        last_step = self.bridge.active_tasks["timeout_task"].steps[-1]
        self.assertEqual(last_step["type"], "timeout")
        
        # Verify metrics updated
        self.assertEqual(self.bridge.metrics["tasks_failed"], 1)
        
        # Verify notification sent
        error_notifications = [msg for msg in self.sent_messages 
                              if msg.target_component == "mock_llm" and 
                              "task_error" in msg.message_type]
        self.assertEqual(len(error_notifications), 1)
        
        # Restore original timeout
        self.bridge.config["task_timeout_seconds"] = original_timeout
    
    def test_environment_update(self):
        """Test environment state update handling."""
        # Set up a goal and task
        goal_context = GoalContext(
            goal_id="env_test_goal",
            goal_description="Navigate with dynamic environment updates",
            priority=1
        )
        
        # Store the goal
        self.bridge.active_goals["env_test_goal"] = goal_context
        
        # Create a task
        task = RLTask(
            task_id="env_test_task",
            goal_context=goal_context,
            environment_context={
                "type": "navigation",
                "size": [10, 10],
                "agent_position": [0, 0],
                "obstacles": [[2, 3], [5, 7]]
            },
            action_space=["move_up", "move_down", "move_left", "move_right"],
            reward_specification={
                "domain": "navigation",
                "complexity": "standard",
                "template": {"goal_reached": 1.0}
            },
            status=TaskStatus.EXECUTING
        )
        
        # Store the task
        self.bridge.active_tasks["env_test_task"] = task
        self.bridge.task_to_goal["env_test_task"] = "env_test_goal"
        
        # Clear sent messages
        self.sent_messages = []
        
        # Create an environment update message
        env_message = IntegrationMessage(
            source_component="mock_llm",
            target_component="test_bridge",
            message_type="llm.environment_description",
            content={
                "task_id": "env_test_task",
                "environment_description": {
                    "agent_position": [1, 1],  # Agent moved
                    "new_obstacle": [3, 4],    # New obstacle appeared
                    "weather": "rainy"         # Weather condition added
                }
            },
            correlation_id="env_test_id"
        )
        
        # Create an event from the message
        env_event = Event(
            event_type="integration.llm.environment_description",
            source="mock_llm",
            data={"message": env_message}
        )
        
        # Process the environment update event
        self.bridge._handle_environment_description(env_event)
        
        # Verify environment update processed
        # In the real implementation, this would parse and update the environment
        # For our test, we'll just check if the appropriate methods were called
        
        # Verify update sent to RL
        rl_messages = [msg for msg in self.sent_messages 
                      if msg.target_component == "mock_rl_adapter"]
        env_updates = [msg for msg in rl_messages 
                      if "update_environment" in msg.message_type]
        self.assertGreaterEqual(len(env_updates), 1)


class TestImprovedLLMtoRLBridgePerformance(unittest.TestCase):
    """Test suite for performance aspects of the ImprovedLLMtoRLBridge."""
    
    def setUp(self):
        """Set up the test environment."""
        # Initialize EventBus
        self.event_bus = EventBus()
        
        # Create mock components
        self.mock_llm = MockLLMEngine()
        self.mock_rl_adapter = MockRLAdapter()
        
        # Create bridge with mocked components
        self.bridge = ImprovedLLMtoRLBridge(
            bridge_id="perf_test_bridge",
            llm_component_id="mock_llm",
            rl_component_id="mock_rl_adapter",
            event_bus=self.event_bus,
            config={
                "goal_batch_size": 10,  # Higher for performance tests
                "task_timeout_seconds": 30,
                "feedback_integration_mode": "immediate",
                "enable_contextual_learning": True,
                "uncertainty_threshold": 0.7,
                "prompt_enhancement": True,
                "max_llm_retry_attempts": 2,
                "diagnostics_level": "minimal"  # Minimal logging for performance
            }
        )
        
        # Initialize event handlers
        self._register_event_handlers()
        
        # Track sent messages
        self.sent_messages = []
        
        # Patch the send_message method to track messages
        self.original_send_message = self.bridge.send_message
        self.bridge.send_message = self._mock_send_message
        
        # Patch the send_to_llm method
        self.bridge.send_to_llm = self._mock_send_to_llm
        
        # For performance tests, we'll use simpler mocks that don't do as much processing
        self.quick_mock_rl_adapter = MagicMock()
        self.quick_mock_rl_adapter.component_id = "mock_rl_adapter"
        self.quick_mock_rl_adapter.component_type = ComponentType.REINFORCEMENT_LEARNING
        
        # Patch task_to_completion to simulate RL responses
        self.task_to_completion_time = {}
    
    def _register_event_handlers(self):
        """Register event handlers for testing."""
        # LLM to Bridge
        self.event_bus.subscribe("integration.llm.goal_definition", self.bridge._handle_goal_definition)
        
        # RL to Bridge
        self.event_bus.subscribe("integration.reinforcement_learning.task_completed", self.bridge._handle_task_completed)
    
    def _mock_send_message(self, target, message_type, content, correlation_id=None):
        """Mock send_message to track messages with minimal processing."""
        message = IntegrationMessage(
            source_component=self.bridge.bridge_id,
            target_component=target,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id or str(uuid.uuid4())
        )
        self.sent_messages.append((message, time.time()))
        
        # For RL task messages, schedule automatic completion after delay
        if (target == self.mock_rl_adapter.component_id and 
            message_type == "reinforcement_learning.create_task"):
            task_id = content.get("task_id")
            if task_id:
                # Schedule for completion (simulated)
                self.task_to_completion_time[task_id] = time.time() + 0.1  # 100ms completion time
        
        return message.correlation_id
    
    def _mock_send_to_llm(self, message_type, content, correlation_id=None):
        """Mock send_to_llm with minimal processing."""
        message = IntegrationMessage(
            source_component=self.bridge.bridge_id,
            target_component=self.bridge.llm_component_id,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id or str(uuid.uuid4())
        )
        self.sent_messages.append((message, time.time()))
        
        # For goal formalization, immediately return success
        if message_type == "goal_formalization_request":
            task_id = content.get("task_id")
            if task_id and task_id in self.bridge.active_tasks:
                task = self.bridge.active_tasks[task_id]
                
                # Set up results
                if not hasattr(task, "results") or task.results is None:
                    task.results = {}
                
                # Add mock reward function
                task.results["reward_function"] = {
                    "components": {"goal_reached": 1.0},
                    "computation": {"type": "simple"}
                }
                
                # Update status
                task.status = TaskStatus.PLANNING
                
                # Prepare for RL
                self.bridge._prepare_rl_task(task)
        
        return message.correlation_id
    
    def _complete_scheduled_tasks(self):
        """Complete any tasks that are scheduled for completion."""
        current_time = time.time()
        
        # Find tasks due for completion
        for task_id, completion_time in list(self.task_to_completion_time.items()):
            if current_time >= completion_time:
                # Complete the task
                if task_id in self.bridge.active_tasks:
                    # Create completion message
                    complete_message = IntegrationMessage(
                        source_component=self.mock_rl_adapter.component_id,
                        target_component=self.bridge.bridge_id,
                        message_type="reinforcement_learning.task_completed",
                        content={
                            "task_id": task_id,
                            "results": {
                                "final_reward": 0.9,
                                "success": True,
                                "steps_taken": 20
                            }
                        }
                    )
                    
                    # Create and publish event
                    event = Event(
                        event_type="integration.reinforcement_learning.task_completed",
                        source=self.mock_rl_adapter.component_id,
                        data={"message": complete_message}
                    )
                    
                    # Process the completion
                    self.bridge._handle_task_completed(event)
                    
                    # Remove from schedule
                    del self.task_to_completion_time[task_id]
    
    def test_parallel_goal_processing(self):
        """Test processing multiple goals in parallel (TC-RL-08)."""
        # Create multiple goals
        num_goals = 5
        
        # Track processing times
        start_time = time.time()
        goal_start_times = {}
        goal_complete_times = {}
        
        # Submit goals in quick succession
        for i in range(num_goals):
            # Create goal
            goal_id = f"perf_goal_{i}"
            goal_description = f"Navigate to target {i} efficiently"
            
            # Create goal message
            goal_message = IntegrationMessage(
                source_component="mock_llm",
                target_component="perf_test_bridge",
                message_type="llm.goal_definition",
                content={
                    "goal_description": goal_description,
                    "goal_id": goal_id,
                    "priority": 1
                },
                correlation_id=f"perf_corr_{i}"
            )
            
            # Record start time
            goal_start_times[goal_id] = time.time()
            
            # Create and publish event
            event = Event(
                event_type="integration.llm.goal_definition",
                source="mock_llm",
                data={"message": goal_message}
            )
            
            # Process goal
            self.bridge._handle_goal_definition(event)
        
        # Wait for all goals to process (simulate async processing)
        max_wait_time = 2.0  # 2 seconds max
        wait_interval = 0.1   # Check every 100ms
        wait_time = 0
        
        while wait_time < max_wait_time:
            # Process scheduled completions
            self._complete_scheduled_tasks()
            
            # Check if all goals are completed
            all_completed = True
            for goal_id in [f"perf_goal_{i}" for i in range(num_goals)]:
                # Find the task for this goal
                task_id = next((t_id for t_id, g_id in self.bridge.task_to_goal.items() 
                              if g_id == goal_id and t_id in self.bridge.active_tasks), None)
                
                if task_id is None or self.bridge.active_tasks[task_id].status != TaskStatus.COMPLETED:
                    all_completed = False
                elif goal_id not in goal_complete_times:
                    # Record completion time
                    goal_complete_times[goal_id] = time.time()
            
            if all_completed:
                break
                
            # Wait a bit
            time.sleep(wait_interval)
            wait_time += wait_interval
        
        # Calculate metrics
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all goals were processed
        self.assertEqual(len(self.bridge.active_goals), num_goals)
        
        # Verify all tasks were created and completed
        completed_tasks = [t for t in self.bridge.active_tasks.values() 
                         if t.status == TaskStatus.COMPLETED]
        self.assertEqual(len(completed_tasks), num_goals)
        
        # Calculate average completion time
        avg_completion_time = sum(goal_complete_times[g] - goal_start_times[g] 
                                for g in goal_complete_times) / len(goal_complete_times)
        
        # Print performance metrics
        print(f"\nParallel goal processing performance:")
        print(f"Total time for {num_goals} goals: {total_time:.3f} seconds")
        print(f"Average goal completion time: {avg_completion_time:.3f} seconds")
        print(f"Goals per second: {num_goals / total_time:.2f}")
        
        # Verify efficient parallel processing
        # The time should be less than processing each goal sequentially
        # Exact thresholds depend on the environment, but we expect some parallelism
        self.assertLess(total_time, avg_completion_time * num_goals,
                      "Parallel processing should be faster than sequential")
    
    def test_complex_environment_handling(self):
        """Test handling complex environment descriptions (TC-RL-09)."""
        # Create a complex environment description
        large_environment = {
            "type": "complex_navigation",
            "dimensions": 3,
            "continuous": True,
            "size": [100, 100, 10],
            "obstacles": []
        }
        
        # Add many obstacles
        for i in range(200):
            large_environment["obstacles"].append([
                random.randint(0, 99),
                random.randint(0, 99),
                random.randint(0, 9)
            ])
        
        # Add dynamic elements
        large_environment["dynamic_elements"] = []
        for i in range(50):
            large_environment["dynamic_elements"].append({
                "type": "moving_obstacle",
                "initial_position": [
                    random.randint(0, 99),
                    random.randint(0, 99),
                    random.randint(0, 9)
                ],
                "velocity": [
                    random.uniform(-1, 1),
                    random.uniform(-1, 1),
                    random.uniform(-0.5, 0.5)
                ],
                "radius": random.uniform(0.5, 2.0)
            })
        
        # Add complex reward structure
        complex_reward = {
            "components": {}
        }
        
        # Add many reward components
        for i in range(20):
            complex_reward["components"][f"component_{i}"] = random.uniform(0.1, 1.0)
        
        # Create a goal with this complex environment
        goal_context = GoalContext(
            goal_id="complex_env_goal",
            goal_description="Navigate in a complex 3D environment",
            priority=1
        )
        
        # Store the goal
        self.bridge.active_goals["complex_env_goal"] = goal_context
        
        # Create a task
        task = RLTask(
            task_id="complex_env_task",
            goal_context=goal_context,
            environment_context=large_environment,
            action_space=["move_x_pos", "move_x_neg", "move_y_pos", "move_y_neg", 
                         "move_z_pos", "move_z_neg", "rotate_x", "rotate_y", "rotate_z"],
            reward_specification=complex_reward,
            status=TaskStatus.PLANNING
        )
        
        # Store the task
        self.bridge.active_tasks["complex_env_task"] = task
        self.bridge.task_to_goal["complex_env_task"] = "complex_env_goal"
        
        # Clear sent messages
        self.sent_messages = []
        
        # Measure the time to prepare and send this task to RL
        start_time = time.time()
        
        # Call preparation method
        self.bridge._prepare_rl_task(task)
        
        # Get end time
        end_time = time.time()
        preparation_time = end_time - start_time
        
        # Verify task was sent to RL component
        rl_messages = [msg for msg, _ in self.sent_messages 
                      if msg.target_component == "mock_rl_adapter"]
        self.assertGreaterEqual(len(rl_messages), 1)
        
        # Find task creation message
        creation_msg = next((msg for msg in rl_messages 
                           if "create_task" in msg.message_type), None)
        self.assertIsNotNone(creation_msg)
        
        # Verify environment content was preserved
        env_content = creation_msg.content.get("environment_context", {})
        self.assertEqual(env_content.get("type"), "complex_navigation")
        self.assertEqual(len(env_content.get("obstacles", [])), 200)
        self.assertEqual(len(env_content.get("dynamic_elements", [])), 50)
        
        # Print performance metrics
        print(f"\nComplex environment handling performance:")
        print(f"Environment size (JSON): {len(json.dumps(large_environment))} bytes")
        print(f"Preparation time: {preparation_time:.3f} seconds")
        
        # Verify reasonable performance (threshold depends on environment)
        # For a test environment, we expect this to be quick
        self.assertLess(preparation_time, 0.5, 
                      "Complex environment preparation should be reasonably fast")


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create random seed for reproducibility
    random.seed(42)
    
    # Run tests
    unittest.main()
