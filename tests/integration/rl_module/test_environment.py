"""
Test Environment for Reinforcement Learning Module Integration Tests.

This module provides the test environment and simulation infrastructure for 
testing the Reinforcement Learning adapter and its integration with other
components of the Jarviee system.
"""

import os
import sys
import json
import numpy as np
import time
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.core.utils.logger import Logger
from src.core.integration.base import IntegrationMessage, ComponentType
from src.core.integration.adapters.reinforcement_learning.adapter import RLAdapter
from src.core.integration.adapters.reinforcement_learning.action import ActionOptimizer
from src.core.integration.adapters.reinforcement_learning.environment import EnvironmentStateManager
from src.core.integration.adapters.reinforcement_learning.reward import RewardFunctionGenerator


class MetricsCollector:
    """
    Collects and analyzes metrics from the RL test environment.
    
    This class is responsible for collecting various performance metrics
    during test execution, calculating statistics, and providing evaluation
    results.
    """
    
    def __init__(self, metric_names: List[str]):
        """
        Initialize the metrics collector.
        
        Args:
            metric_names: List of metric names to track
        """
        self.metrics = {name: [] for name in metric_names}
        self.logger = Logger("RLMetricsCollector")
    
    def record(self, metric_name: str, value: float) -> None:
        """
        Record a metric value.
        
        Args:
            metric_name: Name of the metric to record
            value: Value to record
        """
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
        else:
            self.logger.warning(f"Unknown metric: {metric_name}")
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """
        Get statistics for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Dict containing statistics (mean, median, min, max, std)
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
                "count": 0
            }
        
        values = np.array(self.metrics[metric_name])
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
            "count": len(values)
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all metrics.
        
        Returns:
            Dict mapping metric names to their statistics
        """
        return {name: self.get_statistics(name) for name in self.metrics}
    
    def clear(self) -> None:
        """Clear all recorded metrics."""
        for name in self.metrics:
            self.metrics[name] = []
    
    def export_to_json(self, file_path: str) -> None:
        """
        Export metrics to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        stats = self.get_all_statistics()
        with open(file_path, 'w') as f:
            json.dump(stats, f, indent=2)


class SimulationEnvironment:
    """
    Simulation environment for RL module testing.
    
    This class provides a configurable environment for testing RL components
    in various scenarios. It supports different environment types and can
    simulate state transitions, rewards, and observations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the simulation environment.
        
        Args:
            config: Environment configuration
        """
        self.logger = Logger("RLSimulationEnvironment")
        self.config = config
        self.env_type = config.get("type", "grid_world")
        self.state = self._initialize_state()
        self.step_count = 0
        self.max_steps = config.get("max_steps", 1000)
        self.done = False
        self.reward_function = self._create_reward_function()
        
        # Optional noise to add to state transitions
        self.noise_level = config.get("noise_level", 0.0)
        
        # Track history for analysis
        self.state_history = [self.state.copy()]
        self.action_history = []
        self.reward_history = []
        
        self.logger.info(f"Initialized {self.env_type} environment with config: {json.dumps(config)}")
    
    def _initialize_state(self) -> Dict[str, Any]:
        """
        Initialize the environment state based on the environment type.
        
        Returns:
            The initial state
        """
        if self.env_type == "grid_world":
            size = self.config.get("size", [10, 10])
            return {
                "agent_position": self.config.get("agent_position", [0, 0]),
                "target_position": self.config.get("target_position", [size[0]-1, size[1]-1]),
                "obstacles": self.config.get("obstacles", []),
                "size": size
            }
        elif self.env_type == "resource_management":
            return {
                "resources": self.config.get("initial_resources", {"cpu": 0.5, "memory": 0.5, "network": 0.5}),
                "demand": self.config.get("initial_demand", {"cpu": 0.3, "memory": 0.3, "network": 0.3}),
                "performance": 1.0,
                "efficiency": 1.0
            }
        elif self.env_type == "custom":
            return self.config.get("initial_state", {})
        else:
            self.logger.warning(f"Unknown environment type: {self.env_type}, using empty state")
            return {}
    
    def _create_reward_function(self) -> Callable[[Dict[str, Any], Any], float]:
        """
        Create a reward function based on the environment type.
        
        Returns:
            A function that takes state and action and returns a reward
        """
        if self.env_type == "grid_world":
            def grid_world_reward(state, action):
                # Calculate Manhattan distance to target
                agent_pos = state["agent_position"]
                target_pos = state["target_position"]
                distance = abs(agent_pos[0] - target_pos[0]) + abs(agent_pos[1] - target_pos[1])
                
                # Check if reached target
                if distance == 0:
                    return 10.0
                
                # Check if hit obstacle
                if agent_pos in state["obstacles"]:
                    return -5.0
                
                # Return inverse distance as reward (closer is better)
                return 1.0 / (distance + 1)
            
            return grid_world_reward
            
        elif self.env_type == "resource_management":
            def resource_management_reward(state, action):
                # Calculate reward based on performance and efficiency
                performance = state["performance"]
                efficiency = state["efficiency"]
                
                # Balance between performance and efficiency
                return performance * 0.7 + efficiency * 0.3
            
            return resource_management_reward
            
        elif self.env_type == "custom" and "reward_function" in self.config:
            # Use provided reward function if available
            return self.config["reward_function"]
            
        else:
            # Default reward function returns 0 always
            self.logger.warning(f"No specific reward function for {self.env_type}, using zero reward")
            return lambda state, action: 0.0
    
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment using the given action.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (new_state, reward, done, info)
        """
        if self.done:
            return self.state.copy(), 0.0, True, {"message": "Environment is already done"}
        
        # Increment step counter
        self.step_count += 1
        
        # Update state based on action and environment type
        new_state = self._update_state(action)
        
        # Add noise if configured
        if self.noise_level > 0:
            new_state = self._add_noise(new_state)
        
        # Calculate reward
        reward = self.reward_function(new_state, action)
        
        # Check if done
        done = self._check_done(new_state)
        self.done = done
        
        # Store history
        self.state = new_state
        self.state_history.append(new_state.copy())
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        # Return step results
        return new_state, reward, done, {"step": self.step_count}
    
    def _update_state(self, action: Any) -> Dict[str, Any]:
        """
        Update state based on the action and environment type.
        
        Args:
            action: The action to take
            
        Returns:
            The new state
        """
        new_state = self.state.copy()
        
        if self.env_type == "grid_world":
            agent_pos = list(new_state["agent_position"])
            
            # Update position based on action
            if action == "move_up":
                agent_pos[1] = max(0, agent_pos[1] - 1)
            elif action == "move_down":
                agent_pos[1] = min(new_state["size"][1] - 1, agent_pos[1] + 1)
            elif action == "move_left":
                agent_pos[0] = max(0, agent_pos[0] - 1)
            elif action == "move_right":
                agent_pos[0] = min(new_state["size"][0] - 1, agent_pos[0] + 1)
            
            new_state["agent_position"] = agent_pos
        
        elif self.env_type == "resource_management":
            resources = new_state["resources"]
            demand = new_state["demand"]
            
            # Update resources based on action
            for resource, allocation in action.items():
                if resource in resources:
                    resources[resource] = max(0.0, min(1.0, allocation))
            
            # Calculate performance based on how well resources match demand
            performance = 1.0
            efficiency = 1.0
            
            for resource in resources:
                if resource in demand:
                    # Performance drops if resources are below demand
                    if resources[resource] < demand[resource]:
                        performance *= resources[resource] / demand[resource]
                    
                    # Efficiency drops if resources exceed demand significantly
                    if resources[resource] > demand[resource] * 1.2:
                        efficiency *= demand[resource] * 1.2 / resources[resource]
            
            # Random fluctuation in demand
            for resource in demand:
                # Change demand by up to ±10%
                change = (np.random.random() - 0.5) * 0.2
                demand[resource] = max(0.1, min(1.0, demand[resource] + change))
            
            new_state["resources"] = resources
            new_state["demand"] = demand
            new_state["performance"] = performance
            new_state["efficiency"] = efficiency
        
        elif self.env_type == "custom" and "transition_function" in self.config:
            # Use provided transition function if available
            transition_func = self.config["transition_function"]
            new_state = transition_func(self.state, action)
        
        return new_state
    
    def _add_noise(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add noise to the state to simulate imperfect observations.
        
        Args:
            state: The state to add noise to
            
        Returns:
            The noisy state
        """
        noisy_state = state.copy()
        
        if self.env_type == "grid_world":
            # Add small position noise (but ensure it stays in bounds)
            if np.random.random() < self.noise_level:
                pos = noisy_state["agent_position"]
                noise = [int(np.random.choice([-1, 0, 1])), int(np.random.choice([-1, 0, 1]))]
                noisy_state["agent_position"] = [
                    max(0, min(noisy_state["size"][0] - 1, pos[0] + noise[0])),
                    max(0, min(noisy_state["size"][1] - 1, pos[1] + noise[1]))
                ]
        
        elif self.env_type == "resource_management":
            # Add noise to resource measurements
            for resource in noisy_state["resources"]:
                noise = (np.random.random() - 0.5) * self.noise_level
                noisy_state["resources"][resource] = max(0.0, min(1.0, noisy_state["resources"][resource] + noise))
        
        return noisy_state
    
    def _check_done(self, state: Dict[str, Any]) -> bool:
        """
        Check if the environment is done.
        
        Args:
            state: The current state
            
        Returns:
            True if the environment is done
        """
        # Check if maximum steps reached
        if self.step_count >= self.max_steps:
            return True
        
        if self.env_type == "grid_world":
            # Check if agent reached target
            agent_pos = state["agent_position"]
            target_pos = state["target_position"]
            if agent_pos[0] == target_pos[0] and agent_pos[1] == target_pos[1]:
                return True
        
        elif self.env_type == "custom" and "done_function" in self.config:
            # Use provided done function if available
            done_func = self.config["done_function"]
            return done_func(state)
        
        return False
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to its initial state.
        
        Returns:
            The initial state
        """
        self.state = self._initialize_state()
        self.step_count = 0
        self.done = False
        
        # Clear history
        self.state_history = [self.state.copy()]
        self.action_history = []
        self.reward_history = []
        
        return self.state.copy()
    
    def render(self) -> str:
        """
        Render the environment as a string for visualization.
        
        Returns:
            String representation of the environment
        """
        if self.env_type == "grid_world":
            # Create a grid representation
            size = self.state["size"]
            grid = []
            
            for y in range(size[1]):
                row = []
                for x in range(size[0]):
                    pos = [x, y]
                    
                    if pos == self.state["agent_position"]:
                        row.append("A")
                    elif pos == self.state["target_position"]:
                        row.append("T")
                    elif pos in self.state["obstacles"]:
                        row.append("O")
                    else:
                        row.append(".")
                
                grid.append(" ".join(row))
            
            return "\n".join(grid)
        
        elif self.env_type == "resource_management":
            # Create a text representation of resources and demand
            lines = ["Resource Management:"]
            for resource in self.state["resources"]:
                res_val = self.state["resources"][resource]
                dem_val = self.state["demand"].get(resource, 0.0)
                res_bar = "=" * int(res_val * 10)
                dem_bar = "*" * int(dem_val * 10)
                lines.append(f"{resource.ljust(8)}: {res_bar.ljust(10)} ({res_val:.2f})")
                lines.append(f"{'Demand'.ljust(8)}: {dem_bar.ljust(10)} ({dem_val:.2f})")
            
            lines.append(f"Performance: {self.state['performance']:.2f}")
            lines.append(f"Efficiency : {self.state['efficiency']:.2f}")
            
            return "\n".join(lines)
        
        else:
            return f"State: {json.dumps(self.state, indent=2)}"
    
    def get_valid_actions(self) -> List[Any]:
        """
        Get the list of valid actions for the current state.
        
        Returns:
            List of valid actions
        """
        if self.env_type == "grid_world":
            return ["move_up", "move_down", "move_left", "move_right"]
        
        elif self.env_type == "resource_management":
            # In resource management, actions are allocations so there are infinite possibilities
            # Here we return a predefined set of allocation strategies
            resources = list(self.state["resources"].keys())
            actions = []
            
            # Add some common allocation strategies
            actions.append({res: 0.3 for res in resources})  # Low allocation
            actions.append({res: 0.5 for res in resources})  # Medium allocation
            actions.append({res: 0.7 for res in resources})  # High allocation
            
            # Add specific allocations matching current demand
            actions.append(self.state["demand"].copy())
            
            # Add slightly higher than demand
            actions.append({res: min(1.0, self.state["demand"].get(res, 0.5) * 1.2) for res in resources})
            
            return actions
        
        elif self.env_type == "custom" and "valid_actions" in self.config:
            # Use provided valid actions if available
            if callable(self.config["valid_actions"]):
                return self.config["valid_actions"](self.state)
            else:
                return self.config["valid_actions"]
        
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the environment execution.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "steps": self.step_count,
            "done": self.done,
            "final_state": self.state if self.done else None
        }
        
        if len(self.reward_history) > 0:
            stats["total_reward"] = sum(self.reward_history)
            stats["mean_reward"] = np.mean(self.reward_history)
            stats["min_reward"] = min(self.reward_history)
            stats["max_reward"] = max(self.reward_history)
        
        return stats


class RLTestEnvironment:
    """
    Testing environment for reinforcement learning components.
    
    This class provides infrastructure for testing RL components in 
    various scenarios. It can be configured with different simulation
    environments and evaluation metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RL test environment.
        
        Args:
            config: Test environment configuration
        """
        self.logger = Logger("RLTestEnvironment")
        self.config = config
        
        # Initialize components
        self.simulator = SimulationEnvironment(config.get("simulation", {}))
        self.metrics = MetricsCollector(config.get("metrics", ["reward", "completion_rate", "efficiency"]))
        
        # Track test results
        self.results = {}
        self.baseline = self._establish_baseline()
        
        self.logger.info("RL test environment initialized")
    
    def _establish_baseline(self) -> Dict[str, Any]:
        """
        Establish baseline performance for comparison.
        
        Runs simulations with random actions to establish a baseline.
        
        Returns:
            Baseline performance metrics
        """
        if not self.config.get("establish_baseline", True):
            return {}
        
        self.logger.info("Establishing baseline performance with random actions...")
        
        # Number of baseline episodes
        episodes = self.config.get("baseline_episodes", 10)
        total_rewards = []
        completion_rates = []
        
        for episode in range(episodes):
            # Reset environment
            state = self.simulator.reset()
            done = False
            rewards = []
            
            while not done:
                # Choose random action
                valid_actions = self.simulator.get_valid_actions()
                if not valid_actions:
                    break
                    
                action = np.random.choice(valid_actions)
                
                # Take step
                state, reward, done, _ = self.simulator.step(action)
                rewards.append(reward)
            
            # Record metrics
            total_reward = sum(rewards)
            total_rewards.append(total_reward)
            completion_rates.append(1.0 if self.simulator.done else 0.0)
        
        # Calculate baseline metrics
        baseline = {
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "min_reward": float(np.min(total_rewards)),
            "max_reward": float(np.max(total_rewards)),
            "completion_rate": float(np.mean(completion_rates)),
            "episodes": episodes
        }
        
        self.logger.info(f"Baseline established: {json.dumps(baseline)}")
        return baseline
    
    def run_test(self, test_name: str, action_selector: Callable[[Dict[str, Any], List[Any]], Any], 
                episodes: int = 10) -> Dict[str, Any]:
        """
        Run a test with the provided action selector.
        
        Args:
            test_name: Name of the test
            action_selector: Function that selects actions based on state and valid actions
            episodes: Number of episodes to run
            
        Returns:
            Test results
        """
        self.logger.info(f"Running test: {test_name} for {episodes} episodes")
        
        # Reset metrics
        self.metrics.clear()
        
        total_rewards = []
        completion_rates = []
        steps_to_complete = []
        
        for episode in range(episodes):
            # Reset environment
            state = self.simulator.reset()
            done = False
            rewards = []
            
            while not done:
                # Get valid actions
                valid_actions = self.simulator.get_valid_actions()
                if not valid_actions:
                    break
                
                # Select action using provided selector
                action = action_selector(state, valid_actions)
                
                # Take step
                state, reward, done, _ = self.simulator.step(action)
                rewards.append(reward)
            
            # Record metrics
            total_reward = sum(rewards)
            total_rewards.append(total_reward)
            completion_rates.append(1.0 if self.simulator.done else 0.0)
            
            if self.simulator.done and self.simulator.step_count < self.simulator.max_steps:
                steps_to_complete.append(self.simulator.step_count)
            
            # Record episode metrics
            self.metrics.record("reward", total_reward)
            self.metrics.record("completion_rate", 1.0 if self.simulator.done else 0.0)
            
            # Efficiency is measured as completion in minimum steps
            if self.simulator.done:
                # Normalize by maximum possible steps
                efficiency = 1.0 - ((self.simulator.step_count - 1) / self.simulator.max_steps)
                self.metrics.record("efficiency", efficiency)
        
        # Calculate test results
        results = {
            "test_name": test_name,
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "min_reward": float(np.min(total_rewards)),
            "max_reward": float(np.max(total_rewards)),
            "completion_rate": float(np.mean(completion_rates)),
            "episodes": episodes
        }
        
        # Add mean steps to complete if available
        if steps_to_complete:
            results["mean_steps_to_complete"] = float(np.mean(steps_to_complete))
            results["min_steps_to_complete"] = int(np.min(steps_to_complete))
            results["max_steps_to_complete"] = int(np.max(steps_to_complete))
        
        # Add baseline comparison if available
        if self.baseline:
            results["baseline_comparison"] = {
                "reward_improvement": results["mean_reward"] - self.baseline["mean_reward"],
                "completion_rate_improvement": results["completion_rate"] - self.baseline["completion_rate"],
                "reward_improvement_percent": 
                    (results["mean_reward"] / max(0.0001, self.baseline["mean_reward"]) - 1.0) * 100
                    if self.baseline["mean_reward"] > 0 else 0.0
            }
        
        # Store results
        self.results[test_name] = results
        
        self.logger.info(f"Test completed: {test_name} with mean reward {results['mean_reward']:.2f}")
        return results
    
    def compare_models(self, models: Dict[str, Callable[[Dict[str, Any], List[Any]], Any]], 
                      episodes: int = 10) -> Dict[str, Any]:
        """
        Compare multiple models against each other.
        
        Args:
            models: Dictionary mapping model names to action selectors
            episodes: Number of episodes to run per model
            
        Returns:
            Comparison results
        """
        self.logger.info(f"Comparing {len(models)} models: {', '.join(models.keys())}")
        
        comparison = {}
        
        # Run tests for each model
        for name, selector in models.items():
            results = self.run_test(name, selector, episodes)
            comparison[name] = results
        
        # Find best model
        best_model = max(comparison.keys(), key=lambda k: comparison[k]["mean_reward"])
        best_reward = comparison[best_model]["mean_reward"]
        
        # Calculate relative performance
        for name in comparison:
            comparison[name]["relative_performance"] = comparison[name]["mean_reward"] / best_reward
        
        self.logger.info(f"Model comparison completed. Best model: {best_model}")
        return {
            "models": comparison,
            "best_model": best_model,
            "best_reward": best_reward
        }
    
    def run_rl_adapter_test(self, adapter: RLAdapter, task_id: str, 
                          goal_description: str, episodes: int = 5) -> Dict[str, Any]:
        """
        Test a complete RL adapter implementation.
        
        Args:
            adapter: The RL adapter to test
            task_id: Task ID for the test
            goal_description: Goal description
            episodes: Number of episodes
            
        Returns:
            Test results
        """
        self.logger.info(f"Testing RL adapter with goal: {goal_description}")
        
        # Define action selector that uses the RL adapter
        def adapter_action_selector(state, valid_actions):
            # Update task environment state
            if task_id in adapter.active_tasks:
                adapter.active_tasks[task_id]["environment_state"] = state
            else:
                # Initialize task
                adapter.active_tasks[task_id] = {
                    "created_at": time.time(),
                    "goal_description": goal_description,
                    "environment_state": state,
                    "reward_function": self.simulator.reward_function
                }
            
            # Request action optimization
            best_action, best_reward, _ = adapter.action_optimizer.optimize(
                reward_function=adapter.active_tasks[task_id]["reward_function"],
                environment_state=state,
                action_type="optimize",
                max_steps=100,
                exploration_rate=adapter.learning_state["exploration_rate"],
                available_actions=valid_actions
            )
            
            return best_action
        
        # Run the test
        results = self.run_test(f"RLAdapter_{task_id}", adapter_action_selector, episodes)
        
        # Add adapter-specific metrics
        if hasattr(adapter.action_optimizer, "get_performance_metrics"):
            results["adapter_metrics"] = adapter.action_optimizer.get_performance_metrics()
        
        return results
    
    def export_results(self, file_path: str) -> None:
        """
        Export test results to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        with open(file_path, 'w') as f:
            json.dump({
                "results": self.results,
                "baseline": self.baseline,
                "metrics": self.metrics.get_all_statistics()
            }, f, indent=2)
        
        self.logger.info(f"Results exported to {file_path}")
