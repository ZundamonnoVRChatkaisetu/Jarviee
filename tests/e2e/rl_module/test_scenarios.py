"""
End-to-End Test Scenarios for Reinforcement Learning Module.

This module implements end-to-end test scenarios for the Reinforcement Learning
adapter, focusing on realistic use cases and integration with the complete
Jarviee system.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from unittest.mock import MagicMock, patch

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.core.utils.logger import Logger
from src.core.utils.event_bus import EventBus
from src.core.integration.base import IntegrationMessage, ComponentType
from src.core.llm.engine import LLMEngine
from src.core.llm.mock_engine import MockLLMEngine
from src.core.integration.coordinator.coordinator import IntegrationCoordinator
from src.core.integration.adapters.reinforcement_learning.adapter import RLAdapter
from src.core.integration.adapters.reinforcement_learning.action import ActionOptimizer
from src.core.integration.adapters.reinforcement_learning.environment import EnvironmentStateManager
from src.core.integration.adapters.reinforcement_learning.reward import RewardFunctionGenerator

from tests.integration.rl_module.test_environment import SimulationEnvironment, RLTestEnvironment


class RLScenarioTester:
    """
    End-to-end test scenarios for the Reinforcement Learning module.
    
    This class provides a set of practical, real-world scenarios for testing
    the Reinforcement Learning adapter's functionality and integration with
    other system components.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the RL scenario tester.
        
        Args:
            config_path: Optional path to a config file
        """
        self.logger = Logger("RLScenarioTester")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.event_bus = EventBus()
        self.initialize_components()
        
        # Test results
        self.results = {}
        
        self.logger.info("RL Scenario Tester initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from a file or use defaults.
        
        Args:
            config_path: Path to a JSON config file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "use_real_llm": False,
            "scenarios": ["resource_optimization", "path_finding", "custom"],
            "output_dir": "test_results",
            "test_iterations": 3,
            "timeout": 30,
            "verbose": True
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    for key, value in loaded_config.items():
                        default_config[key] = value
                self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.error(f"Error loading config from {config_path}: {str(e)}")
        
        return default_config
    
    def initialize_components(self):
        """Initialize the necessary components for testing."""
        # Initialize LLM engine (real or mock)
        if self.config.get("use_real_llm", False):
            self.llm_engine = LLMEngine(
                engine_id="test_llm",
                model_name=self.config.get("llm_model", "gpt-3.5-turbo"),
                event_bus=self.event_bus
            )
        else:
            self.llm_engine = MockLLMEngine(
                engine_id="test_llm",
                response_templates={
                    "goal_interpretation_request": {
                        "type": "goal_interpretation",
                        "content": {
                            "goal_data": {
                                "objective": "{{OBJECTIVE}}",
                                "constraints": ["{{CONSTRAINT_1}}", "{{CONSTRAINT_2}}"],
                                "priorities": {"{{PRIORITY_1}}": 0.7, "{{PRIORITY_2}}": 0.3}
                            }
                        }
                    },
                    "feedback_interpretation_request": {
                        "type": "feedback_interpretation",
                        "content": {
                            "feedback_data": {
                                "analysis": "{{ANALYSIS}}",
                                "suggestions": ["{{SUGGESTION_1}}", "{{SUGGESTION_2}}"]
                            }
                        }
                    }
                },
                event_bus=self.event_bus
            )
        
        # Initialize coordinator
        self.coordinator = IntegrationCoordinator(
            component_id="test_coordinator",
            event_bus=self.event_bus
        )
        
        # Initialize RL adapter
        self.rl_adapter = RLAdapter(
            adapter_id="test_rl_adapter",
            llm_component_id="test_llm"
        )
        
        # Register components with event bus
        self.event_bus.subscribe("integration.*", self.coordinator.process_message)
        self.event_bus.subscribe("llm.*", self.llm_engine.process_message)
        self.event_bus.subscribe("integration.reinforcement_learning.*", self.rl_adapter.process_message)
        
        # Initialize test environments for different scenarios
        self._init_test_environments()
    
    def _init_test_environments(self):
        """Initialize test environments for different scenarios."""
        # Resource optimization environment
        resource_opt_config = {
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
        
        # Path finding environment
        path_finding_config = {
            "simulation": {
                "type": "grid_world",
                "size": [10, 10],
                "max_steps": 100,
                "agent_position": [0, 0],
                "target_position": [9, 9],
                "obstacles": [[2, 2], [2, 3], [3, 2], [5, 5], [5, 6], [6, 5], [6, 6], [8, 3], [3, 8]]
            },
            "metrics": ["reward", "completion_rate", "steps_to_target", "path_efficiency"],
            "establish_baseline": True,
            "baseline_episodes": 5
        }
        
        # Custom environment for extensibility demonstration
        custom_config = {
            "simulation": {
                "type": "custom",
                "max_steps": 30,
                "initial_state": {
                    "position": [0, 0],
                    "energy": 100,
                    "items_collected": 0,
                    "items_locations": [[2, 3], [5, 2], [8, 8]],
                    "hazards": [[3, 3], [6, 6]]
                },
                "transition_function": self._custom_transition_function,
                "reward_function": self._custom_reward_function,
                "done_function": self._custom_done_function,
                "valid_actions": ["move_up", "move_down", "move_left", "move_right", "collect"]
            },
            "metrics": ["reward", "items_collected", "energy_efficiency"],
            "establish_baseline": True,
            "baseline_episodes": 3
        }
        
        # Initialize environments
        self.test_environments = {
            "resource_optimization": RLTestEnvironment(resource_opt_config),
            "path_finding": RLTestEnvironment(path_finding_config),
            "custom": RLTestEnvironment(custom_config)
        }
    
    def _custom_transition_function(self, state: Dict[str, Any], action: str) -> Dict[str, Any]:
        """
        Custom transition function for the custom environment.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            New state
        """
        new_state = state.copy()
        position = list(new_state["position"])
        
        # Update position based on action
        if action == "move_up":
            position[1] = max(0, position[1] - 1)
            new_state["energy"] -= 1
        elif action == "move_down":
            position[1] = min(9, position[1] + 1)
            new_state["energy"] -= 1
        elif action == "move_left":
            position[0] = max(0, position[0] - 1)
            new_state["energy"] -= 1
        elif action == "move_right":
            position[0] = min(9, position[0] + 1)
            new_state["energy"] -= 1
        elif action == "collect":
            # Check if there's an item at the current position
            if position in new_state["items_locations"]:
                new_state["items_collected"] += 1
                new_state["items_locations"].remove(position)
                new_state["energy"] -= 2
            else:
                # Penalty for trying to collect where there's no item
                new_state["energy"] -= 5
        
        new_state["position"] = position
        
        # Check for hazards
        if position in new_state["hazards"]:
            new_state["energy"] -= 10
        
        return new_state
    
    def _custom_reward_function(self, state: Dict[str, Any], action: str) -> float:
        """
        Custom reward function for the custom environment.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Reward value
        """
        # Reward for collecting items
        reward = state["items_collected"] * 10
        
        # Penalty for energy usage
        reward -= (100 - state["energy"]) * 0.1
        
        # Penalty for being in or near hazards
        position = state["position"]
        for hazard in state["hazards"]:
            distance = abs(position[0] - hazard[0]) + abs(position[1] - hazard[1])
            if distance == 0:
                reward -= 10
            elif distance <= 2:
                reward -= 5 / distance
        
        return reward
    
    def _custom_done_function(self, state: Dict[str, Any]) -> bool:
        """
        Custom done function for the custom environment.
        
        Args:
            state: Current state
            
        Returns:
            True if the environment is done
        """
        # Done if all items collected
        if len(state["items_locations"]) == 0:
            return True
        
        # Done if out of energy
        if state["energy"] <= 0:
            return True
        
        return False
    
    def run_resource_optimization_scenario(self) -> Dict[str, Any]:
        """
        Run the resource optimization scenario.
        
        This scenario simulates a resource allocation problem, where the agent
        must optimize the allocation of resources (CPU, memory, network) based
        on changing demand.
        
        Returns:
            Scenario results
        """
        self.logger.info("Starting Resource Optimization scenario")
        
        # Use the appropriate test environment
        test_env = self.test_environments["resource_optimization"]
        
        # Define task parameters
        task_id = "resource_optimization_task"
        goal_description = "Optimize resource allocation to maximize efficiency while maintaining performance above 0.8"
        
        # Configure mock LLM responses for this scenario
        if isinstance(self.llm_engine, MockLLMEngine):
            self.llm_engine.configure_template_values({
                "OBJECTIVE": "maximize_resource_efficiency",
                "CONSTRAINT_1": "maintain_performance_above_0.8",
                "CONSTRAINT_2": "adapt_to_changing_demands",
                "PRIORITY_1": "efficiency",
                "PRIORITY_2": "performance"
            })
        
        # Create the task in the RL adapter
        initial_state = test_env.simulator.state.copy()
        
        # Set up a custom reward function for this scenario
        def resource_reward_function(state: Dict[str, Any]) -> float:
            # Balance between performance and efficiency
            performance = state.get("performance", 0.0)
            efficiency = state.get("efficiency", 0.0)
            
            # Performance must be above 0.8 (constraint)
            if performance < 0.8:
                return performance * 0.5  # Partial reward based on how close to 0.8
            
            # Once performance requirement is met, focus on efficiency
            return 0.8 + efficiency * 0.2
        
        # Create task directly in the adapter
        self.rl_adapter.active_tasks[task_id] = {
            "created_at": time.time(),
            "goal_description": goal_description,
            "environment_state": initial_state,
            "reward_function": resource_reward_function
        }
        
        # Run test using the adapter's optimization capability
        results = test_env.run_rl_adapter_test(
            self.rl_adapter, 
            task_id, 
            goal_description, 
            episodes=self.config.get("test_iterations", 3)
        )
        
        # Additional scenario-specific analysis
        results["scenario"] = "resource_optimization"
        results["efficiency_stability"] = self._calculate_stability(
            test_env.metrics.metrics.get("efficiency", [])
        )
        results["performance_constraint_violations"] = self._count_constraint_violations(
            test_env.simulator.state_history, 
            lambda s: s.get("performance", 1.0) < 0.8
        )
        
        self.logger.info(f"Resource Optimization scenario completed with mean reward: {results['mean_reward']:.2f}")
        return results
    
    def run_path_finding_scenario(self) -> Dict[str, Any]:
        """
        Run the path finding scenario.
        
        This scenario simulates a navigation problem, where the agent must find
        an efficient path from start to goal while avoiding obstacles.
        
        Returns:
            Scenario results
        """
        self.logger.info("Starting Path Finding scenario")
        
        # Use the appropriate test environment
        test_env = self.test_environments["path_finding"]
        
        # Define task parameters
        task_id = "path_finding_task"
        goal_description = "Find the shortest path to the target while avoiding obstacles"
        
        # Configure mock LLM responses for this scenario
        if isinstance(self.llm_engine, MockLLMEngine):
            self.llm_engine.configure_template_values({
                "OBJECTIVE": "find_shortest_path",
                "CONSTRAINT_1": "avoid_obstacles",
                "CONSTRAINT_2": "minimize_steps",
                "PRIORITY_1": "path_efficiency",
                "PRIORITY_2": "safety"
            })
        
        # Create the task in the RL adapter
        initial_state = test_env.simulator.state.copy()
        
        # Set up a custom reward function for this scenario
        def path_reward_function(state: Dict[str, Any]) -> float:
            agent_pos = state.get("agent_position", [0, 0])
            target_pos = state.get("target_position", [9, 9])
            obstacles = state.get("obstacles", [])
            
            # Calculate Manhattan distance to target
            distance = abs(agent_pos[0] - target_pos[0]) + abs(agent_pos[1] - target_pos[1])
            
            # Check if reached target
            if distance == 0:
                return 100.0
            
            # Check if hit obstacle
            if agent_pos in obstacles:
                return -50.0
            
            # Check if near obstacle (penalty for being close)
            for obstacle in obstacles:
                obs_distance = abs(agent_pos[0] - obstacle[0]) + abs(agent_pos[1] - obstacle[1])
                if obs_distance == 1:
                    return -5.0
            
            # Return inverse distance as reward (closer is better)
            return 10.0 / (distance + 1)
        
        # Create task directly in the adapter
        self.rl_adapter.active_tasks[task_id] = {
            "created_at": time.time(),
            "goal_description": goal_description,
            "environment_state": initial_state,
            "reward_function": path_reward_function
        }
        
        # Run test using the adapter's optimization capability
        results = test_env.run_rl_adapter_test(
            self.rl_adapter, 
            task_id, 
            goal_description, 
            episodes=self.config.get("test_iterations", 3)
        )
        
        # Additional scenario-specific analysis
        results["scenario"] = "path_finding"
        results["obstacle_collisions"] = self._count_constraint_violations(
            test_env.simulator.state_history, 
            lambda s: s.get("agent_position") in s.get("obstacles", [])
        )
        
        if "mean_steps_to_complete" in results:
            # Calculate path efficiency (ratio of manhattan distance to steps taken)
            start_pos = test_env.simulator.state["agent_position"]
            target_pos = test_env.simulator.state["target_position"]
            manhattan_distance = abs(start_pos[0] - target_pos[0]) + abs(start_pos[1] - target_pos[1])
            results["path_efficiency"] = manhattan_distance / max(1, results["mean_steps_to_complete"])
        
        self.logger.info(f"Path Finding scenario completed with mean reward: {results['mean_reward']:.2f}")
        return results
    
    def run_custom_scenario(self) -> Dict[str, Any]:
        """
        Run the custom scenario.
        
        This scenario demonstrates how to create a custom environment and test
        the RL adapter with it.
        
        Returns:
            Scenario results
        """
        self.logger.info("Starting Custom scenario")
        
        # Use the appropriate test environment
        test_env = self.test_environments["custom"]
        
        # Define task parameters
        task_id = "custom_scenario_task"
        goal_description = "Collect all items while minimizing energy usage and avoiding hazards"
        
        # Configure mock LLM responses for this scenario
        if isinstance(self.llm_engine, MockLLMEngine):
            self.llm_engine.configure_template_values({
                "OBJECTIVE": "collect_all_items",
                "CONSTRAINT_1": "minimize_energy_usage",
                "CONSTRAINT_2": "avoid_hazards",
                "PRIORITY_1": "collection_complete",
                "PRIORITY_2": "energy_efficiency"
            })
        
        # Create the task in the RL adapter
        initial_state = test_env.simulator.state.copy()
        
        # Use the environment's reward function directly
        reward_function = test_env.simulator.reward_function
        
        # Create task directly in the adapter
        self.rl_adapter.active_tasks[task_id] = {
            "created_at": time.time(),
            "goal_description": goal_description,
            "environment_state": initial_state,
            "reward_function": reward_function
        }
        
        # Run test using the adapter's optimization capability
        results = test_env.run_rl_adapter_test(
            self.rl_adapter, 
            task_id, 
            goal_description, 
            episodes=self.config.get("test_iterations", 3)
        )
        
        # Additional scenario-specific analysis
        results["scenario"] = "custom"
        results["hazard_encounters"] = self._count_constraint_violations(
            test_env.simulator.state_history, 
            lambda s: s.get("position") in s.get("hazards", [])
        )
        results["items_collected"] = np.mean([
            s.get("items_collected", 0) for s in test_env.simulator.state_history
            if "items_collected" in s
        ])
        results["energy_efficiency"] = np.mean([
            s.get("energy", 0) / 100.0 for s in test_env.simulator.state_history
            if "energy" in s
        ])
        
        self.logger.info(f"Custom scenario completed with mean reward: {results['mean_reward']:.2f}")
        return results
    
    def _calculate_stability(self, metric_values: List[float]) -> float:
        """
        Calculate stability of a metric over time.
        
        Args:
            metric_values: List of metric values
            
        Returns:
            Stability score (0-1, higher is more stable)
        """
        if not metric_values or len(metric_values) < 2:
            return 1.0
        
        # Calculate changes between consecutive values
        changes = [abs(metric_values[i] - metric_values[i-1]) for i in range(1, len(metric_values))]
        
        # Average change relative to the mean
        mean_value = np.mean(metric_values)
        mean_change = np.mean(changes)
        
        # Stability score (1 - normalized average change)
        stability = 1.0 - min(1.0, mean_change / max(0.0001, mean_value))
        
        return stability
    
    def _count_constraint_violations(self, state_history: List[Dict[str, Any]], 
                                    violation_check: Callable[[Dict[str, Any]], bool]) -> int:
        """
        Count constraint violations in state history.
        
        Args:
            state_history: List of states
            violation_check: Function that checks if a state violates a constraint
            
        Returns:
            Number of violations
        """
        violations = 0
        for state in state_history:
            if violation_check(state):
                violations += 1
        
        return violations
    
    def run_scenarios(self, selected_scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run selected or all scenarios.
        
        Args:
            selected_scenarios: List of scenario names to run, or None for all
            
        Returns:
            Dictionary of scenario results
        """
        scenarios = selected_scenarios or self.config.get("scenarios", [])
        results = {}
        
        for scenario in scenarios:
            if scenario == "resource_optimization":
                results[scenario] = self.run_resource_optimization_scenario()
            elif scenario == "path_finding":
                results[scenario] = self.run_path_finding_scenario()
            elif scenario == "custom":
                results[scenario] = self.run_custom_scenario()
            else:
                self.logger.warning(f"Unknown scenario: {scenario}")
        
        return results
    
    def export_results(self, output_dir: Optional[str] = None) -> None:
        """
        Export results to files.
        
        Args:
            output_dir: Directory to save results, or None for default
        """
        output_dir = output_dir or self.config.get("output_dir", "test_results")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save overall results
        with open(os.path.join(output_dir, "scenario_results.json"), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save individual scenario results
        for scenario, result in self.results.items():
            with open(os.path.join(output_dir, f"{scenario}_results.json"), 'w') as f:
                json.dump(result, f, indent=2)
        
        self.logger.info(f"Results exported to {output_dir}")
    
    def print_summary(self) -> None:
        """Print a summary of scenario results."""
        print("\n" + "="*80)
        print("REINFORCEMENT LEARNING SCENARIO TEST SUMMARY")
        print("="*80)
        
        for scenario, result in self.results.items():
            print(f"\nScenario: {scenario}")
            print("-" * 40)
            print(f"Mean Reward: {result.get('mean_reward', 0.0):.2f}")
            print(f"Completion Rate: {result.get('completion_rate', 0.0):.2f}")
            
            # Print scenario-specific metrics
            if scenario == "resource_optimization":
                print(f"Efficiency Stability: {result.get('efficiency_stability', 0.0):.2f}")
                print(f"Performance Constraint Violations: {result.get('performance_constraint_violations', 0)}")
                
            elif scenario == "path_finding":
                print(f"Path Efficiency: {result.get('path_efficiency', 0.0):.2f}")
                print(f"Obstacle Collisions: {result.get('obstacle_collisions', 0)}")
                if "mean_steps_to_complete" in result:
                    print(f"Average Steps to Target: {result.get('mean_steps_to_complete', 0.0):.1f}")
                
            elif scenario == "custom":
                print(f"Items Collected: {result.get('items_collected', 0.0):.1f}")
                print(f"Energy Efficiency: {result.get('energy_efficiency', 0.0):.2f}")
                print(f"Hazard Encounters: {result.get('hazard_encounters', 0)}")
            
            # Print baseline comparison if available
            if "baseline_comparison" in result:
                baseline = result["baseline_comparison"]
                print("\nComparison to Random Baseline:")
                print(f"Reward Improvement: {baseline.get('reward_improvement', 0.0):.2f} " +
                      f"({baseline.get('reward_improvement_percent', 0.0):.1f}%)")
                print(f"Completion Rate Improvement: {baseline.get('completion_rate_improvement', 0.0):.2f}")
        
        print("\n" + "="*80)


def main():
    """Main function to run scenarios from command line."""
    parser = argparse.ArgumentParser(description="Run RL module end-to-end test scenarios")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--scenarios", nargs="+", help="Scenarios to run")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--iterations", type=int, help="Number of test iterations per scenario")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    # Create tester
    tester = RLScenarioTester(args.config)
    
    # Override config with command line arguments
    if args.scenarios:
        tester.config["scenarios"] = args.scenarios
    if args.output:
        tester.config["output_dir"] = args.output
    if args.iterations:
        tester.config["test_iterations"] = args.iterations
    if args.quiet:
        tester.config["verbose"] = False
    
    # Run scenarios
    tester.results = tester.run_scenarios()
    
    # Export results
    tester.export_results()
    
    # Print summary
    if tester.config.get("verbose", True):
        tester.print_summary()


if __name__ == "__main__":
    main()
