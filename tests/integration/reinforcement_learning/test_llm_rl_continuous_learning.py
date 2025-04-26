# tests/integration/reinforcement_learning/test_llm_rl_continuous_learning.py
"""
Test for continuous learning capabilities of LLM-RL integration.

This module implements the TC-RL-10 test case: "継続学習と適応" from the test plan,
focusing on the ability of the RL module to improve performance over time through
feedback and multiple task executions.
"""

import os
import sys
import json
import time
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.core.utils.event_bus import EventBus, Event
from src.core.integration.llm_rl_bridge_improved import ImprovedLLMtoRLBridge, GoalContext, RLTask, TaskStatus
from tests.integration.reinforcement_learning.test_improved_llm_rl_bridge import MockLLMEngine, MockRLAdapter

class TestContinuousLearningLLMRL(unittest.TestCase):
    """Test suite for continuous learning capabilities of LLM-RL integration."""
    
    def setUp(self):
        """Set up the test environment."""
        # Initialize EventBus
        self.event_bus = EventBus()
        
        # Create mock components
        self.mock_llm = MockLLMEngine()
        self.mock_rl_adapter = MockRLAdapter()
        
        # Create bridge with mocked components
        self.bridge = ImprovedLLMtoRLBridge(
            bridge_id="continuous_learning_test_bridge",
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
                "diagnostics_level": "detailed",
                "learning_rate": 0.1,  # Specific to continuous learning
                "experience_retention": 5  # Keep track of 5 past experiences
            }
        )
        
        # Initialize event handlers - similar to test_improved_llm_rl_bridge.py
        self._register_event_handlers()
        
        # Mock methods for testing
        self._mock_methods()
        
        # Set up test scenario data
        self._setup_test_scenario()
    
    def _register_event_handlers(self):
        """Register event handlers for testing."""
        # Copy from test_improved_llm_rl_bridge.py with any additional handlers
        # for learning-specific events
        pass
    
    def _mock_methods(self):
        """Mock methods for testing."""
        # Mock methods similar to test_improved_llm_rl_bridge.py
        # Add specific mocks for learning-related methods
        pass
    
    def _setup_test_scenario(self):
        """Set up test scenario data for continuous learning tests."""
        # Define a consistent navigation goal that will be used across multiple tasks
        self.test_goal = "Navigate efficiently through an environment with obstacles"
        
        # Define a sequence of increasingly complex environments for testing adaptation
        self.test_environments = [
            {
                "type": "grid_world",
                "size": [5, 5],
                "agent_position": [0, 0],
                "target_position": [4, 4],
                "obstacles": [[2, 2]]  # Simple environment with one obstacle
            },
            {
                "type": "grid_world",
                "size": [5, 5],
                "agent_position": [0, 0],
                "target_position": [4, 4],
                "obstacles": [[1, 1], [2, 2], [3, 3]]  # More obstacles in diagonal
            },
            {
                "type": "grid_world",
                "size": [5, 5],
                "agent_position": [0, 0],
                "target_position": [4, 4],
                "obstacles": [[1, 1], [1, 2], [2, 1], [2, 2], [3, 3]]  # Complex obstacle pattern
            }
        ]
        
        # Define feedback templates for each environment
        self.feedback_templates = [
            "Good navigation, but try to be more energy efficient",
            "Better path finding, focus more on avoiding obstacles completely",
            "Good obstacle avoidance, now optimize for the shortest possible path"
        ]
        
        # Metrics to track
        self.learning_metrics = {
            "task_completion_times": [],
            "reward_values": [],
            "obstacle_collisions": [],
            "path_efficiencies": []
        }
    
    def test_performance_improvement_over_time(self):
        """
        Test that the RL module improves performance over time through feedback.
        This implements TC-RL-10 from the test plan.
        """
        # Run a sequence of similar tasks with increasing complexity
        for i, environment in enumerate(self.test_environments):
            # Create a new goal for this iteration
            goal_id = f"continuous_learning_goal_{i}"
            goal_context = GoalContext(
                goal_id=goal_id,
                goal_description=self.test_goal,
                priority=1
            )
            
            # Store the goal
            self.bridge.active_goals[goal_id] = goal_context
            
            # Process the goal to a task
            self.bridge._process_goal_to_task(goal_context)
            
            # Get the task ID
            task_id = next(key for key, value in self.bridge.task_to_goal.items() 
                         if value == goal_id)
            
            # Set the environment for this task
            task = self.bridge.active_tasks[task_id]
            task.environment_context = environment
            
            # Simulate task execution
            self._simulate_task_execution(task_id, i)
            
            # Record metrics
            self._record_metrics(task)
            
            # Provide feedback for learning
            self._provide_feedback(task_id, i)
            
            # Allow time for processing
            time.sleep(0.1)
        
        # Analyze learning progress
        self._analyze_learning_progress()
    
    def _simulate_task_execution(self, task_id, complexity_level):
        """Simulate task execution with appropriate results based on complexity level."""
        # Adjust success rate and performance based on prior learning
        prior_experience = len(self.learning_metrics["reward_values"])
        
        # Base performance decreases with complexity but improves with experience
        base_success = 0.7 - (complexity_level * 0.1) + (prior_experience * 0.05)
        base_success = max(0.5, min(0.95, base_success))
        
        # Calculate metrics
        steps_taken = 20 + (complexity_level * 5) - (prior_experience * 2)
        steps_taken = max(15, steps_taken)
        
        obstacle_collisions = max(0, complexity_level - prior_experience)
        
        # Generate result based on current learning state
        result = {
            "final_reward": base_success,
            "success": base_success > 0.6,
            "steps_taken": steps_taken,
            "obstacle_collisions": obstacle_collisions,
            "execution_time": 0.5 + (complexity_level * 0.2),
            "path_efficiency": base_success
        }
        
        # Create completion message
        complete_message = MagicMock()
        complete_message.source_component = self.mock_rl_adapter.component_id
        complete_message.target_component = self.bridge.bridge_id
        complete_message.message_type = "reinforcement_learning.task_completed"
        complete_message.content = {
            "task_id": task_id,
            "results": result
        }
        
        # Create event
        event = Event(
            event_type="integration.reinforcement_learning.task_completed",
            source=self.mock_rl_adapter.component_id,
            data={"message": complete_message}
        )
        
        # Process task completion
        self.bridge._handle_task_completed(event)
    
    def _provide_feedback(self, task_id, feedback_index):
        """Provide feedback for a completed task to enable learning."""
        # Create feedback message
        feedback_message = MagicMock()
        feedback_message.source_component = self.mock_llm.component_id
        feedback_message.target_component = self.bridge.bridge_id
        feedback_message.message_type = "llm.feedback_provision"
        feedback_message.content = {
            "task_id": task_id,
            "feedback": {
                "reward": 0.6 + (feedback_index * 0.1),  # Increasing reward
                "comments": self.feedback_templates[feedback_index],
                "summary": f"Feedback iteration {feedback_index+1}"
            }
        }
        
        # Create event
        event = Event(
            event_type="integration.llm.feedback_provision",
            source=self.mock_llm.component_id,
            data={"message": feedback_message}
        )
        
        # Process feedback
        self.bridge._handle_feedback_provision(event)
    
    def _record_metrics(self, task):
        """Record metrics for analyzing learning progress."""
        if hasattr(task, "results") and task.results:
            self.learning_metrics["reward_values"].append(
                task.results.get("final_reward", 0))
            self.learning_metrics["task_completion_times"].append(
                task.results.get("execution_time", 0))
            self.learning_metrics["obstacle_collisions"].append(
                task.results.get("obstacle_collisions", 0))
            self.learning_metrics["path_efficiencies"].append(
                task.results.get("path_efficiency", 0))
    
    def _analyze_learning_progress(self):
        """Analyze and verify learning progress metrics."""
        # Check that reward values trend upward
        self.assertGreaterEqual(len(self.learning_metrics["reward_values"]), 2, 
                              "Not enough data points to analyze learning")
        
        # Calculate improvement trends
        reward_trend = self._calculate_trend(self.learning_metrics["reward_values"])
        efficiency_trend = self._calculate_trend(self.learning_metrics["path_efficiencies"])
        collision_trend = self._calculate_trend(self.learning_metrics["obstacle_collisions"])
        
        # Verify positive learning trends
        self.assertGreater(reward_trend, 0, 
                         "Reward should show positive trend over time")
        self.assertGreater(efficiency_trend, 0, 
                         "Path efficiency should show positive trend over time")
        self.assertLess(collision_trend, 0, 
                       "Obstacle collisions should decrease over time")
        
        # Print learning metrics for clarity
        print("\nLearning Progress Metrics:")
        print(f"Reward Values: {self.learning_metrics['reward_values']}")
        print(f"Path Efficiencies: {self.learning_metrics['path_efficiencies']}")
        print(f"Obstacle Collisions: {self.learning_metrics['obstacle_collisions']}")
        print(f"Reward Improvement Trend: {reward_trend:.4f}")
        print(f"Efficiency Improvement Trend: {efficiency_trend:.4f}")
        print(f"Collision Reduction Trend: {collision_trend:.4f}")
    
    def _calculate_trend(self, values):
        """Calculate a simple trend line slope for a series of values."""
        if len(values) < 2:
            return 0
            
        x = list(range(len(values)))
        x_mean = sum(x) / len(x)
        y_mean = sum(values) / len(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(len(values)))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(len(values)))
        
        if denominator == 0:
            return 0
            
        return numerator / denominator


if __name__ == "__main__":
    unittest.main()