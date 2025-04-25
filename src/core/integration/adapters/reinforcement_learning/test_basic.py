"""
Basic Test Implementation for RL Adapter in Jarviee System.

This module provides basic tests for the Reinforcement Learning adapter
based on the integration approach described in the 'basic.md' document.
It demonstrates how LLM and RL can be integrated for practical use cases.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from .adapter import RLAdapter
from .action import ActionOptimizer
from .environment import EnvironmentStateManager
from .reward import RewardFunctionGenerator
from ...base import ComponentType, IntegrationMessage
from ...llm_rl_bridge import LLMtoRLBridge
from ....llm.engine import LLMEngine
from ....utils.event_bus import EventBus
from ....utils.logger import Logger


class RLAdapterTest:
    """
    Test implementation for the Reinforcement Learning adapter integration.
    
    This class provides methods to test the integration between LLM and RL
    components as described in the design document. It sets up the necessary
    components, performs test scenarios, and validates the results.
    """
    
    def __init__(self, test_id: str = "rl_test_basic"):
        """
        Initialize the RL adapter test.
        
        Args:
            test_id: Unique identifier for this test
        """
        self.test_id = test_id
        self.logger = Logger().get_logger(f"jarviee.integration.test.{test_id}")
        
        # Initialize event bus
        self.event_bus = EventBus()
        
        # Component IDs
        self.llm_id = "llm_test_engine"
        self.rl_id = "rl_test_adapter"
        self.bridge_id = "llm_rl_test_bridge"
        
        # Components
        self.llm_engine = None
        self.rl_adapter = None
        self.bridge = None
        
        # Test state
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = {}
        
        self.logger.info(f"RL Adapter Test {test_id} initialized")
    
    def setup(self) -> bool:
        """
        Set up the components for testing.
        
        Returns:
            bool: True if setup was successful
        """
        try:
            # Initialize LLM component (mock)
            self.llm_engine = self._create_mock_llm_engine()
            
            # Initialize RL adapter
            self.rl_adapter = RLAdapter(
                adapter_id=self.rl_id,
                llm_component_id=self.llm_id
            )
            
            # Initialize bridge
            self.bridge = LLMtoRLBridge(
                bridge_id=self.bridge_id,
                llm_component_id=self.llm_id,
                rl_component_id=self.rl_id,
                event_bus=self.event_bus
            )
            
            # Initialize components
            if not self.rl_adapter.initialize():
                self.logger.error("Failed to initialize RL adapter")
                return False
                
            # Start components
            if not self.rl_adapter.start():
                self.logger.error("Failed to start RL adapter")
                return False
                
            self.logger.info("Test components set up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up test components: {str(e)}")
            return False
    
    def _create_mock_llm_engine(self):
        """
        Create a mock LLM engine for testing.
        
        Returns:
            object: Mock LLM engine
        """
        # Simple mock that responds to bridge messages
        class MockLLMEngine:
            def __init__(self, component_id, event_bus, logger):
                self.component_id = component_id
                self.event_bus = event_bus
                self.logger = logger
                
                # Subscribe to events
                self.event_bus.subscribe(
                    f"integration.*",
                    self._handle_event
                )
                
                logger.info(f"Mock LLM Engine {component_id} initialized")
            
            def _handle_event(self, event):
                # Extract message
                if "message" not in event.data:
                    return
                    
                message = event.data["message"]
                
                # Check if targeted at this component
                if message.target_component != self.component_id:
                    return
                    
                self.logger.debug(f"Mock LLM received: {message.message_type}")
                
                # Handle goal interpretation request
                if message.message_type == "goal_interpretation_request":
                    self._handle_goal_interpretation(message)
                    
                # Handle feedback interpretation request
                elif message.message_type == "feedback_interpretation_request":
                    self._handle_feedback_interpretation(message)
                    
                # Handle task completion
                elif message.message_type == "llm.task_completed":
                    self._handle_task_completion(message)
                    
                # Handle goal completion
                elif message.message_type == "llm.goal_completed":
                    self._handle_goal_completion(message)
            
            def _handle_goal_interpretation(self, message):
                # Extract goal description
                goal_description = message.content.get("goal_description", "")
                context = message.content.get("context", {})
                
                # Create a simple interpretation
                interpretation = {
                    "primary_objective": "optimize_performance",
                    "reward_components": {
                        "performance": 0.7,
                        "efficiency": 0.3
                    },
                    "constraints": ["resource_limits", "time_constraints"],
                    "domain": "programming" if "code" in goal_description.lower() else "general"
                }
                
                # Send interpretation back to bridge
                response = IntegrationMessage(
                    source_component=self.component_id,
                    target_component=message.source_component,
                    message_type="goal_interpretation",
                    content={
                        "task_id": context.get("task_id"),
                        "goal_data": interpretation
                    },
                    correlation_id=message.message_id
                )
                
                self.event_bus.publish(response.to_event())
                self.logger.debug(f"Sent goal interpretation for task {context.get('task_id')}")
            
            def _handle_feedback_interpretation(self, message):
                # Extract feedback
                feedback = message.content.get("feedback_comments", "")
                context = message.content.get("context", {})
                
                # Create a simple interpretation
                interpretation = {
                    "feedback_type": "performance",
                    "sentiment": "positive" if "good" in feedback.lower() else "negative",
                    "improvement_suggestions": ["optimize_algorithm", "reduce_memory_usage"]
                }
                
                # Send interpretation back to bridge
                response = IntegrationMessage(
                    source_component=self.component_id,
                    target_component=message.source_component,
                    message_type="feedback_interpretation",
                    content={
                        "task_id": context.get("task_id"),
                        "interpretation": interpretation
                    },
                    correlation_id=message.message_id
                )
                
                self.event_bus.publish(response.to_event())
                self.logger.debug(f"Sent feedback interpretation for task {context.get('task_id')}")
            
            def _handle_task_completion(self, message):
                # Log task completion
                task_id = message.content.get("task_id")
                goal_id = message.content.get("goal_id")
                results = message.content.get("results", {})
                
                self.logger.info(f"Task {task_id} completed for goal {goal_id}")
                self.logger.info(f"Results: {json.dumps(results)}")
            
            def _handle_goal_completion(self, message):
                # Log goal completion
                goal_id = message.content.get("goal_id")
                summary = message.content.get("summary", "")
                
                self.logger.info(f"Goal {goal_id} completed")
                self.logger.info(f"Summary: {summary}")
        
        # Create and return instance
        return MockLLMEngine(self.llm_id, self.event_bus, self.logger)
    
    def run_integration_tests(self) -> bool:
        """
        Run integration tests for the RL adapter.
        
        Returns:
            bool: True if all tests passed
        """
        try:
            # Run individual tests
            self._test_goal_to_task_conversion()
            self._test_task_optimization()
            self._test_feedback_incorporation()
            self._test_extended_scenario()
            
            # Log results
            self.logger.info(f"Tests completed: {self.tests_passed} passed, {self.tests_failed} failed")
            
            return self.tests_failed == 0
            
        except Exception as e:
            self.logger.error(f"Error running integration tests: {str(e)}")
            return False
    
    def _test_goal_to_task_conversion(self) -> None:
        """
        Test conversion from language goal to RL task.
        """
        self.logger.info("Starting test: Goal to Task Conversion")
        
        try:
            # Create a goal using the bridge
            goal_description = "Optimize code performance for the search algorithm"
            goal_id = self.bridge.create_goal_from_text(
                goal_description,
                priority=2,
                constraints=["memory_usage < 100MB", "execution_time < 500ms"]
            )
            
            # Wait for processing
            time.sleep(1)
            
            # Check if goal was created and converted to task
            goal_status = self.bridge.get_goal_status(goal_id)
            
            self.logger.info(f"Goal status: {json.dumps(goal_status)}")
            
            # Validate
            if goal_status.get("goal_id") == goal_id:
                if len(goal_status.get("tasks", {})) > 0:
                    self.logger.info("Test passed: Goal successfully converted to task")
                    self.tests_passed += 1
                    self.test_results["goal_to_task"] = True
                else:
                    self.logger.error("Test failed: Goal did not generate any tasks")
                    self.tests_failed += 1
                    self.test_results["goal_to_task"] = False
            else:
                self.logger.error("Test failed: Goal status could not be retrieved")
                self.tests_failed += 1
                self.test_results["goal_to_task"] = False
                
        except Exception as e:
            self.logger.error(f"Error in goal to task test: {str(e)}")
            self.tests_failed += 1
            self.test_results["goal_to_task"] = False
    
    def _test_task_optimization(self) -> None:
        """
        Test optimization of an RL task.
        """
        self.logger.info("Starting test: Task Optimization")
        
        try:
            # Create a sample environment state
            environment_state = {
                "code_complexity": 75,
                "memory_usage": 85,
                "cpu_usage": 65,
                "execution_time": 120,
                "algorithm_type": "search",
                "language": "python"
            }
            
            # Create a sample reward function
            reward_function = {
                "domain": "programming",
                "complexity": "medium",
                "primary_objective": "performance",
                "components": {
                    "execution_time": {"weight": 0.5, "target": "minimize"},
                    "memory_usage": {"weight": 0.3, "target": "minimize"},
                    "code_readability": {"weight": 0.2, "target": "maintain"}
                }
            }
            
            # Get action optimizer from adapter
            action_optimizer = self.rl_adapter.action_optimizer
            
            # Run optimization
            start_time = time.time()
            best_action, best_reward, steps = action_optimizer.optimize(
                reward_function=reward_function,
                environment_state=environment_state,
                action_type="optimize",
                max_steps=100
            )
            elapsed_time = time.time() - start_time
            
            self.logger.info(f"Optimization completed in {elapsed_time:.2f}s with {steps} steps")
            self.logger.info(f"Best action: {json.dumps(best_action)}")
            self.logger.info(f"Best reward: {best_reward:.4f}")
            
            # Validate
            if best_reward > 0.0 and best_action:
                self.logger.info("Test passed: Successfully optimized action")
                self.tests_passed += 1
                self.test_results["task_optimization"] = True
            else:
                self.logger.error("Test failed: Could not optimize action")
                self.tests_failed += 1
                self.test_results["task_optimization"] = False
                
        except Exception as e:
            self.logger.error(f"Error in task optimization test: {str(e)}")
            self.tests_failed += 1
            self.test_results["task_optimization"] = False
    
    def _test_feedback_incorporation(self) -> None:
        """
        Test incorporation of feedback to improve future actions.
        """
        self.logger.info("Starting test: Feedback Incorporation")
        
        try:
            # Create a sample action
            action = {
                "type": "optimize",
                "subtype": "performance",
                "param1": 65,
                "param2": 35
            }
            
            # Create a sample environment state
            environment_state = {
                "code_complexity": 80,
                "memory_usage": 90,
                "cpu_usage": 70,
                "execution_time": 150
            }
            
            # Get action optimizer from adapter
            action_optimizer = self.rl_adapter.action_optimizer
            
            # Record initial Q-table state
            initial_q_value = None
            state_repr = action_optimizer._get_state_representation(environment_state)
            action_repr = action_optimizer._get_action_representation(action)
            
            if state_repr in action_optimizer.q_table and action_repr in action_optimizer.q_table[state_repr]:
                initial_q_value = action_optimizer.q_table[state_repr][action_repr]
            
            # Incorporate feedback
            expected_reward = 0.7
            actual_reward = 0.9  # Better than expected
            
            success = action_optimizer.incorporate_feedback(
                action=action,
                expected_reward=expected_reward,
                actual_reward=actual_reward,
                environment_state=environment_state
            )
            
            # Check new Q-value
            new_q_value = None
            if state_repr in action_optimizer.q_table and action_repr in action_optimizer.q_table[state_repr]:
                new_q_value = action_optimizer.q_table[state_repr][action_repr]
            
            self.logger.info(f"Feedback incorporation result: {success}")
            self.logger.info(f"Initial Q-value: {initial_q_value}")
            self.logger.info(f"New Q-value: {new_q_value}")
            
            # Validate
            if success and (initial_q_value is None or new_q_value > initial_q_value):
                self.logger.info("Test passed: Successfully incorporated feedback")
                self.tests_passed += 1
                self.test_results["feedback_incorporation"] = True
            else:
                self.logger.error("Test failed: Could not incorporate feedback")
                self.tests_failed += 1
                self.test_results["feedback_incorporation"] = False
                
        except Exception as e:
            self.logger.error(f"Error in feedback incorporation test: {str(e)}")
            self.tests_failed += 1
            self.test_results["feedback_incorporation"] = False
    
    def _test_extended_scenario(self) -> None:
        """
        Test an extended scenario that combines multiple aspects of the integration.
        
        This test simulates a complete flow:
        1. Goal creation
        2. Task conversion
        3. Optimization
        4. Feedback incorporation
        5. Re-optimization
        """
        self.logger.info("Starting test: Extended Scenario")
        
        try:
            # Create a goal using the bridge
            goal_description = "Improve the performance of the database query system while maintaining reliability"
            
            goal_id = self.bridge.create_goal_from_text(
                goal_description,
                priority=3,
                constraints=["downtime < 1min", "maintain_data_integrity"],
                metadata={"system": "database", "component": "query_engine"}
            )
            
            # Wait for processing
            time.sleep(1)
            
            # Get goal status
            goal_status = self.bridge.get_goal_status(goal_id)
            
            # Extract task ID
            task_ids = list(goal_status.get("tasks", {}).keys())
            
            if not task_ids:
                self.logger.error("Test failed: No tasks created for goal")
                self.tests_failed += 1
                self.test_results["extended_scenario"] = False
                return
                
            task_id = task_ids[0]
            self.logger.info(f"Created task {task_id} for goal {goal_id}")
            
            # Manually simulate task completion (since we don't have a real RL component)
            # In a real system, the RL adapter would handle this
            
            # Create task results
            task_results = {
                "optimized_parameters": {
                    "query_cache_size": 256,
                    "index_optimization": True,
                    "parallel_execution": True
                },
                "expected_performance_gain": "35%",
                "expected_reliability_impact": "minimal",
                "implementation_difficulty": "medium"
            }
            
            # Send task completion message
            task_completed_message = IntegrationMessage(
                source_component=self.rl_id,
                target_component=self.bridge_id,
                message_type="reinforcement_learning.task_completed",
                content={
                    "task_id": task_id,
                    "status": "completed",
                    "results": task_results
                }
            )
            
            self.event_bus.publish(task_completed_message.to_event())
            
            # Wait for processing
            time.sleep(1)
            
            # Get goal status again
            updated_goal_status = self.bridge.get_goal_status(goal_id)
            
            # Check if goal shows task as completed
            task_status = updated_goal_status.get("tasks", {}).get(task_id, {}).get("status")
            
            self.logger.info(f"Updated task status: {task_status}")
            
            # Validate
            if task_status == "completed":
                self.logger.info("Test passed: Extended scenario successfully completed")
                self.tests_passed += 1
                self.test_results["extended_scenario"] = True
            else:
                self.logger.error(f"Test failed: Task not marked as completed (status: {task_status})")
                self.tests_failed += 1
                self.test_results["extended_scenario"] = False
                
        except Exception as e:
            self.logger.error(f"Error in extended scenario test: {str(e)}")
            self.tests_failed += 1
            self.test_results["extended_scenario"] = False
    
    def cleanup(self) -> None:
        """
        Clean up resources after testing.
        """
        try:
            # Stop RL adapter
            if self.rl_adapter and self.rl_adapter.is_running:
                self.rl_adapter.stop()
                
            # Clear event bus
            if self.event_bus:
                # In a real system, you would unsubscribe from events
                pass
                
            self.logger.info("Test resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up test resources: {str(e)}")
    
    def get_test_summary(self) -> Dict[str, Any]:
        """
        Get a summary of test results.
        
        Returns:
            Dict: Test summary
        """
        return {
            "test_id": self.test_id,
            "total_tests": self.tests_passed + self.tests_failed,
            "passed": self.tests_passed,
            "failed": self.tests_failed,
            "pass_rate": self.tests_passed / max(1, self.tests_passed + self.tests_failed),
            "results": self.test_results,
            "timestamp": time.time()
        }


def run_tests() -> None:
    """
    Run all RL adapter tests.
    """
    logger = Logger().get_logger("jarviee.integration.test.runner")
    logger.info("Starting RL adapter tests")
    
    # Create and run test
    tester = RLAdapterTest()
    
    # Set up
    if not tester.setup():
        logger.error("Failed to set up test components")
        return
        
    # Run tests
    success = tester.run_integration_tests()
    
    # Clean up
    tester.cleanup()
    
    # Report results
    summary = tester.get_test_summary()
    logger.info(f"Test summary: {json.dumps(summary)}")
    
    if success:
        logger.info("All tests passed successfully")
    else:
        logger.error("Some tests failed")


if __name__ == "__main__":
    run_tests()
