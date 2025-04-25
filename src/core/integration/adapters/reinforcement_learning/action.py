"""
Action Optimizer Module for Reinforcement Learning Adapter.

This module provides functionality for optimizing actions using reinforcement
learning algorithms based on reward functions and environment states. It handles
action selection, exploration vs. exploitation, and learning from feedback.
"""

import json
import random
import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ....utils.logger import Logger


class ActionOptimizationMethod(Enum):
    """Methods for action optimization in reinforcement learning."""
    Q_LEARNING = "q_learning"
    SARSA = "sarsa"
    DQN = "dqn"
    PPO = "ppo"
    A2C = "a2c"


class ActionOptimizer:
    """
    Optimizer for selecting actions using reinforcement learning algorithms.
    
    This class provides functionality for selecting optimal actions based on
    reward functions and environment states using various reinforcement learning
    algorithms. It handles the exploration-exploitation tradeoff and learns from
    feedback on past actions.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the action optimizer.
        
        Args:
            model_path: Optional path to pre-trained RL model
        """
        self.logger = Logger().get_logger("jarviee.integration.rl.action_optimizer")
        self.initialized = False
        self.running = False
        self.optimization_lock = threading.RLock()
        
        # RL parameters
        self.exploration_rate = 0.1
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        
        # Model state
        self.model_path = model_path
        self.q_table = {}  # Simple Q-table for basic RL
        self.models = {}  # More sophisticated models
        
        # Learning history
        self.action_history = []
        self.max_history_size = 1000
        
        # Supported optimization methods
        self.optimization_methods = {
            ActionOptimizationMethod.Q_LEARNING: self._optimize_q_learning,
            ActionOptimizationMethod.SARSA: self._optimize_sarsa,
            ActionOptimizationMethod.DQN: self._optimize_dqn,
            ActionOptimizationMethod.PPO: self._optimize_ppo,
            ActionOptimizationMethod.A2C: self._optimize_a2c
        }
        
        # Default method
        self.default_method = ActionOptimizationMethod.Q_LEARNING
    
    def initialize(self, exploration_rate: float = 0.1, 
                 learning_rate: float = 0.001, 
                 discount_factor: float = 0.99) -> bool:
        """
        Initialize the action optimizer.
        
        Args:
            exploration_rate: Epsilon value for exploration (0-1)
            learning_rate: Alpha value for learning rate (0-1)
            discount_factor: Gamma value for future reward discounting (0-1)
            
        Returns:
            bool: True if initialization was successful
        """
        if self.initialized:
            return True
            
        try:
            # Set parameters
            self.exploration_rate = exploration_rate
            self.learning_rate = learning_rate
            self.discount_factor = discount_factor
            
            # Load model if path provided
            if self.model_path:
                self._load_model(self.model_path)
                
            self.initialized = True
            self.logger.info("Action optimizer initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing action optimizer: {str(e)}")
            return False
    
    def _load_model(self, model_path: str) -> bool:
        """
        Load a pre-trained RL model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            bool: True if loading was successful
        """
        try:
            # In a real implementation, this would load model weights
            # Here, we simulate loading with a simple mock model
            
            self.logger.info(f"Simulating model load from {model_path}")
            
            # For demonstration, just initialize a basic Q-table
            self.q_table = {}
            
            # Initialize more sophisticated models as needed
            self.models = {
                ActionOptimizationMethod.DQN: {"initialized": True},
                ActionOptimizationMethod.PPO: {"initialized": True},
                ActionOptimizationMethod.A2C: {"initialized": True}
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def start(self) -> bool:
        """
        Start the action optimizer.
        
        Returns:
            bool: True if start was successful
        """
        if not self.initialized:
            if not self.initialize():
                return False
                
        self.running = True
        self.logger.info("Action optimizer started")
        return True
    
    def stop(self) -> bool:
        """
        Stop the action optimizer.
        
        Returns:
            bool: True if stop was successful
        """
        self.running = False
        self.logger.info("Action optimizer stopped")
        return True
    
    def shutdown(self) -> bool:
        """
        Shut down the action optimizer.
        
        Returns:
            bool: True if shutdown was successful
        """
        try:
            # Stop if running
            if self.running:
                self.stop()
                
            # Clear resources
            self.q_table = {}
            self.models = {}
            self.action_history = []
            
            self.initialized = False
            self.logger.info("Action optimizer shut down")
            return True
            
        except Exception as e:
            self.logger.error(f"Error shutting down action optimizer: {str(e)}")
            return False
    
    def optimize(self, reward_function: Dict[str, Any], 
                environment_state: Dict[str, Any],
                action_type: str, max_steps: int = 1000,
                exploration_rate: Optional[float] = None) -> Tuple[Dict[str, Any], float, int]:
        """
        Optimize an action based on a reward function and environment state.
        
        Args:
            reward_function: Reward function definition
            environment_state: Current environment state
            action_type: Type of action to optimize
            max_steps: Maximum optimization steps
            exploration_rate: Optional override for exploration rate
            
        Returns:
            Tuple[Dict, float, int]: Best action, best reward, steps taken
        """
        if not self.initialized or not self.running:
            self.logger.warning("Action optimizer not initialized or not running")
            return {}, 0.0, 0
            
        with self.optimization_lock:
            try:
                # Determine optimization method
                method_name = action_type.upper() if action_type.upper() in [m.name for m in ActionOptimizationMethod] else self.default_method.name
                method = ActionOptimizationMethod[method_name]
                
                # Set exploration rate
                current_exploration = exploration_rate if exploration_rate is not None else self.exploration_rate
                
                # Get optimizer function
                optimizer_func = self.optimization_methods.get(method, self._optimize_q_learning)
                
                # Run optimization
                best_action, best_reward, steps_taken = optimizer_func(
                    reward_function, environment_state, current_exploration, max_steps
                )
                
                # Record in history
                self._record_optimization(
                    reward_function, environment_state, action_type,
                    best_action, best_reward, steps_taken
                )
                
                return best_action, best_reward, steps_taken
                
            except Exception as e:
                self.logger.error(f"Error in action optimization: {str(e)}")
                return {}, 0.0, 0
    
    def incorporate_feedback(self, action: Dict[str, Any], 
                           expected_reward: float, actual_reward: float,
                           environment_state: Dict[str, Any],
                           reward_function: Optional[Dict[str, Any]] = None) -> bool:
        """
        Incorporate feedback on a previous action to improve future optimization.
        
        Args:
            action: The action that was taken
            expected_reward: The reward that was expected
            actual_reward: The actual reward received
            environment_state: The environment state when the action was taken
            reward_function: Optional reward function definition
            
        Returns:
            bool: True if feedback was successfully incorporated
        """
        if not self.initialized:
            return False
            
        try:
            # Create state representation
            state_repr = self._get_state_representation(environment_state)
            action_repr = self._get_action_representation(action)
            
            # Update Q-table
            if state_repr not in self.q_table:
                self.q_table[state_repr] = {}
                
            if action_repr not in self.q_table[state_repr]:
                self.q_table[state_repr][action_repr] = expected_reward
                
            # Apply temporal difference update
            current_q = self.q_table[state_repr][action_repr]
            error = actual_reward - current_q
            self.q_table[state_repr][action_repr] += self.learning_rate * error
            
            # Log the update
            self.logger.info(f"Updated Q-value for state-action: expected={expected_reward:.2f}, actual={actual_reward:.2f}, new_q={self.q_table[state_repr][action_repr]:.2f}")
            
            # Record in history
            self._record_feedback(action, expected_reward, actual_reward, environment_state)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error incorporating feedback: {str(e)}")
            return False
    
    def update_model(self, model_info: Dict[str, Any]) -> bool:
        """
        Update the underlying RL model with new information.
        
        Args:
            model_info: Information for model update
            
        Returns:
            bool: True if update was successful
        """
        if not self.initialized:
            return False
            
        try:
            # Update parameters if provided
            if "exploration_rate" in model_info:
                self.exploration_rate = model_info["exploration_rate"]
                
            if "learning_rate" in model_info:
                self.learning_rate = model_info["learning_rate"]
                
            if "discount_factor" in model_info:
                self.discount_factor = model_info["discount_factor"]
                
            # Update specific model if provided
            if "model_updates" in model_info:
                model_updates = model_info["model_updates"]
                model_type = model_updates.get("type")
                
                if model_type == "q_table":
                    # Update Q-table entries
                    updates = model_updates.get("updates", {})
                    for state_repr, actions in updates.items():
                        if state_repr not in self.q_table:
                            self.q_table[state_repr] = {}
                            
                        for action_repr, value in actions.items():
                            self.q_table[state_repr][action_repr] = value
                
                elif model_type in ["dqn", "ppo", "a2c"]:
                    # Update neural network based models
                    # In a real implementation, this would update model weights
                    model_method = ActionOptimizationMethod[model_type.upper()]
                    if model_method in self.models:
                        self.models[model_method]["last_update"] = time.time()
                        
            self.logger.info("Model updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating model: {str(e)}")
            return False
    
    def get_exploration_rate(self) -> float:
        """
        Get the current exploration rate.
        
        Returns:
            float: Current exploration rate
        """
        return self.exploration_rate
    
    def set_exploration_rate(self, rate: float) -> None:
        """
        Set the exploration rate.
        
        Args:
            rate: New exploration rate (0-1)
        """
        self.exploration_rate = max(0.0, min(1.0, rate))
        self.logger.info(f"Exploration rate set to {self.exploration_rate}")
    
    def _optimize_q_learning(self, reward_function: Dict[str, Any],
                           environment_state: Dict[str, Any],
                           exploration_rate: float, max_steps: int) -> Tuple[Dict[str, Any], float, int]:
        """
        Optimize using Q-learning algorithm.
        
        Args:
            reward_function: Reward function definition
            environment_state: Current environment state
            exploration_rate: Exploration rate for this optimization
            max_steps: Maximum optimization steps
            
        Returns:
            Tuple[Dict, float, int]: Best action, best reward, steps taken
        """
        # Create state representation
        state_repr = self._get_state_representation(environment_state)
        
        # Initialize tracking variables
        best_action = {}
        best_reward = float("-inf")
        steps_taken = 0
        
        # Get action space
        action_space = self._get_action_space(environment_state, reward_function)
        
        # Q-learning optimization loop
        current_state = state_repr
        
        for step in range(max_steps):
            # Increment step counter
            steps_taken += 1
            
            # Exploration vs. exploitation
            if random.random() < exploration_rate:
                # Exploration: random action
                action = random.choice(action_space)
            else:
                # Exploitation: best known action
                action = self._get_best_action(current_state, action_space)
                
            # Convert action to proper format
            action_dict = self._action_to_dict(action)
            
            # Simulate next state and reward
            next_state, reward = self._simulate_action(current_state, action, environment_state, reward_function)
            
            # Update best action if better reward found
            if reward > best_reward:
                best_action = action_dict
                best_reward = reward
                
            # Update Q-table
            self._update_q_value(current_state, action, next_state, reward)
            
            # Move to next state
            current_state = next_state
            
            # Early stopping if reward is good enough
            if reward > 0.95:  # Assuming reward is normalized to [0, 1]
                break
                
        return best_action, best_reward, steps_taken
    
    def _optimize_sarsa(self, reward_function: Dict[str, Any],
                       environment_state: Dict[str, Any],
                       exploration_rate: float, max_steps: int) -> Tuple[Dict[str, Any], float, int]:
        """
        Optimize using SARSA algorithm.
        
        Args:
            reward_function: Reward function definition
            environment_state: Current environment state
            exploration_rate: Exploration rate for this optimization
            max_steps: Maximum optimization steps
            
        Returns:
            Tuple[Dict, float, int]: Best action, best reward, steps taken
        """
        # For this demo, we'll just use a simplified version similar to Q-learning
        # In a real implementation, SARSA would be properly implemented
        
        self.logger.info("Using SARSA optimization (simplified implementation)")
        
        # Call Q-learning for now
        return self._optimize_q_learning(reward_function, environment_state, exploration_rate, max_steps)
    
    def _optimize_dqn(self, reward_function: Dict[str, Any],
                     environment_state: Dict[str, Any],
                     exploration_rate: float, max_steps: int) -> Tuple[Dict[str, Any], float, int]:
        """
        Optimize using Deep Q-Network algorithm.
        
        Args:
            reward_function: Reward function definition
            environment_state: Current environment state
            exploration_rate: Exploration rate for this optimization
            max_steps: Maximum optimization steps
            
        Returns:
            Tuple[Dict, float, int]: Best action, best reward, steps taken
        """
        # For this demo, we'll implement a simplified version that doesn't require TensorFlow/PyTorch
        # In a real implementation, DQN would use a neural network
        
        self.logger.info("Using DQN optimization (simplified implementation)")
        
        # Initialize tracking variables
        best_action = {}
        best_reward = float("-inf")
        steps_taken = 0
        
        # Get action space
        action_space = self._get_action_space(environment_state, reward_function)
        
        # Simulate deep exploration with more random sampling
        for step in range(max_steps):
            # Increment step counter
            steps_taken += 1
            
            # Generate a candidate action
            if random.random() < exploration_rate:
                # Pure exploration
                action = random.choice(action_space)
            else:
                # Guided exploration - slightly perturb previous best actions
                if best_action and random.random() < 0.5:
                    action = self._perturb_action(best_action, action_space)
                else:
                    action = random.choice(action_space)
                    
            # Convert action to proper format
            action_dict = self._action_to_dict(action)
            
            # Evaluate action using reward function
            reward = self._evaluate_action(action_dict, environment_state, reward_function)
            
            # Update best action if better reward found
            if reward > best_reward:
                best_action = action_dict
                best_reward = reward
                
                # Early stopping if reward is good enough
                if reward > 0.95:
                    break
                    
        return best_action, best_reward, steps_taken
    
    def _optimize_ppo(self, reward_function: Dict[str, Any],
                     environment_state: Dict[str, Any],
                     exploration_rate: float, max_steps: int) -> Tuple[Dict[str, Any], float, int]:
        """
        Optimize using Proximal Policy Optimization algorithm.
        
        Args:
            reward_function: Reward function definition
            environment_state: Current environment state
            exploration_rate: Exploration rate for this optimization
            max_steps: Maximum optimization steps
            
        Returns:
            Tuple[Dict, float, int]: Best action, best reward, steps taken
        """
        # For this demo, we'll call a simpler algorithm
        # In a real implementation, PPO would be properly implemented
        
        self.logger.info("Using PPO optimization (placeholder implementation)")
        
        # Call DQN for now
        return self._optimize_dqn(reward_function, environment_state, exploration_rate, max_steps)
    
    def _optimize_a2c(self, reward_function: Dict[str, Any],
                     environment_state: Dict[str, Any],
                     exploration_rate: float, max_steps: int) -> Tuple[Dict[str, Any], float, int]:
        """
        Optimize using Advantage Actor-Critic algorithm.
        
        Args:
            reward_function: Reward function definition
            environment_state: Current environment state
            exploration_rate: Exploration rate for this optimization
            max_steps: Maximum optimization steps
            
        Returns:
            Tuple[Dict, float, int]: Best action, best reward, steps taken
        """
        # For this demo, we'll call a simpler algorithm
        # In a real implementation, A2C would be properly implemented
        
        self.logger.info("Using A2C optimization (placeholder implementation)")
        
        # Call DQN for now
        return self._optimize_dqn(reward_function, environment_state, exploration_rate, max_steps)
    
    def _get_state_representation(self, state: Dict[str, Any]) -> str:
        """
        Convert a state dictionary to a string representation for the Q-table.
        
        Args:
            state: State dictionary
            
        Returns:
            str: String representation of the state
        """
        # Simple implementation - in a real system, this would be more sophisticated
        # Sort keys for consistency
        sorted_items = sorted(
            [(k, v) for k, v in state.items() if not isinstance(v, (dict, list, tuple))],
            key=lambda x: x[0]
        )
        
        # Create string representation
        return "|".join(f"{k}:{v}" for k, v in sorted_items)
    
    def _get_action_representation(self, action: Dict[str, Any]) -> str:
        """
        Convert an action dictionary to a string representation for the Q-table.
        
        Args:
            action: Action dictionary
            
        Returns:
            str: String representation of the action
        """
        # Simple implementation - in a real system, this would be more sophisticated
        # Sort keys for consistency
        sorted_items = sorted(action.items(), key=lambda x: x[0])
        
        # Create string representation
        return "|".join(f"{k}:{v}" for k, v in sorted_items)
    
    def _get_action_space(self, state: Dict[str, Any], 
                         reward_function: Dict[str, Any]) -> List[str]:
        """
        Generate a list of possible actions for the current state.
        
        Args:
            state: Current environment state
            reward_function: Reward function definition
            
        Returns:
            List[str]: List of action representations
        """
        # In a real implementation, this would generate meaningful actions
        # Here, we'll generate some placeholder actions
        
        # Generate a small set of random action representations
        action_space = []
        
        # Extract action domain from reward function if available
        domain = reward_function.get("domain", "general")
        
        # Generate different actions based on domain
        if domain == "programming":
            actions = [
                "refactor|optimize_performance",
                "refactor|improve_readability",
                "add|feature",
                "fix|bug",
                "optimize|memory",
                "optimize|cpu",
                "test|unit",
                "test|integration"
            ]
            action_space.extend(actions)
            
        elif domain == "dialog":
            actions = [
                "respond|brief",
                "respond|detailed",
                "ask|clarification",
                "suggest|alternative",
                "provide|example",
                "summarize|conversation",
                "acknowledge|sentiment",
                "change|topic"
            ]
            action_space.extend(actions)
            
        else:  # general
            actions = [
                "search|information",
                "analyze|data",
                "generate|content",
                "organize|resources",
                "validate|input",
                "optimize|workflow",
                "interact|user",
                "modify|system"
            ]
            action_space.extend(actions)
            
        # Add some random variations
        for _ in range(5):
            param1 = random.randint(0, 100)
            param2 = random.randint(0, 100)
            action_space.append(f"param1:{param1}|param2:{param2}")
            
        return action_space
    
    def _get_best_action(self, state: str, action_space: List[str]) -> str:
        """
        Get the best known action for a state from the Q-table.
        
        Args:
            state: State representation
            action_space: List of possible actions
            
        Returns:
            str: Best action representation
        """
        # If state not in Q-table, return random action
        if state not in self.q_table:
            return random.choice(action_space)
            
        # Filter to actions that are in both Q-table and action space
        valid_actions = [a for a in action_space if a in self.q_table[state]]
        
        # If no valid actions, return random action
        if not valid_actions:
            return random.choice(action_space)
            
        # Return action with highest Q-value
        return max(valid_actions, key=lambda a: self.q_table[state].get(a, 0))
    
    def _action_to_dict(self, action: str) -> Dict[str, Any]:
        """
        Convert an action representation to a dictionary.
        
        Args:
            action: Action representation string
            
        Returns:
            Dict: Action dictionary
        """
        # Parse the action string into a dictionary
        action_dict = {}
        
        # Split on pipe character
        parts = action.split("|")
        
        if len(parts) == 2:
            # Simple action with type and subtype
            action_dict["type"] = parts[0]
            action_dict["subtype"] = parts[1]
        else:
            # Parameter-based action
            for part in action.split("|"):
                if ":" in part:
                    key, value = part.split(":", 1)
                    
                    # Try to convert numeric values
                    try:
                        if "." in value:
                            action_dict[key] = float(value)
                        else:
                            action_dict[key] = int(value)
                    except ValueError:
                        action_dict[key] = value
                else:
                    # No value specified
                    action_dict[part] = True
                    
        return action_dict
    
    def _simulate_action(self, state: str, action: str, 
                        environment_state: Dict[str, Any],
                        reward_function: Dict[str, Any]) -> Tuple[str, float]:
        """
        Simulate taking an action in the environment.
        
        Args:
            state: Current state representation
            action: Action representation
            environment_state: Full environment state
            reward_function: Reward function definition
            
        Returns:
            Tuple[str, float]: Next state representation and reward
        """
        # Convert action to dictionary
        action_dict = self._action_to_dict(action)
        
        # In a real implementation, this would simulate the environment
        # Here, we'll use a simple model
        
        # Create a slightly modified next state
        next_state_dict = environment_state.copy()
        
        # Make some basic changes based on action
        if "type" in action_dict:
            action_type = action_dict["type"]
            
            if action_type == "refactor":
                next_state_dict["code_quality"] = next_state_dict.get("code_quality", 0) + 1
                
            elif action_type == "optimize":
                next_state_dict["performance"] = next_state_dict.get("performance", 0) + 1
                
            elif action_type == "respond":
                next_state_dict["user_satisfaction"] = next_state_dict.get("user_satisfaction", 0) + 1
                
        # Add some randomness
        next_state_dict["random_factor"] = random.random()
        
        # Convert to state representation
        next_state = self._get_state_representation(next_state_dict)
        
        # Calculate reward
        reward = self._evaluate_action(action_dict, environment_state, reward_function)
        
        return next_state, reward
    
    def _evaluate_action(self, action: Dict[str, Any], 
                        environment_state: Dict[str, Any],
                        reward_function: Dict[str, Any]) -> float:
        """
        Evaluate an action using the reward function.
        
        Args:
            action: Action dictionary
            environment_state: Environment state
            reward_function: Reward function definition
            
        Returns:
            float: Reward value
        """
        # In a real implementation, this would use the actual reward function
        # Here, we'll use a simplified model
        
        # Extract relevant information
        domain = reward_function.get("domain", "general")
        complexity = reward_function.get("complexity", "medium")
        
        # Base reward
        reward = 0.0
        
        # Domain-specific evaluation
        if domain == "programming":
            if "type" in action and action["type"] == "refactor":
                reward += 0.7
            elif "type" in action and action["type"] == "optimize":
                reward += 0.8
            elif "type" in action and action["type"] == "test":
                reward += 0.6
                
        elif domain == "dialog":
            if "type" in action and action["type"] == "respond":
                reward += 0.7
            elif "type" in action and action["type"] == "ask":
                reward += 0.5
            elif "type" in action and action["type"] == "acknowledge":
                reward += 0.6
                
        else:  # general
            if "type" in action and action["type"] == "search":
                reward += 0.5
            elif "type" in action and action["type"] == "analyze":
                reward += 0.7
            elif "type" in action and action["type"] == "generate":
                reward += 0.6
                
        # Adjust based on complexity
        if complexity == "simple":
            # Simpler reward function
            reward = min(1.0, reward)
        elif complexity == "medium":
            # Add some noise
            reward = min(1.0, reward + random.uniform(-0.1, 0.1))
        else:  # complex
            # More factors and noise
            extra_factor = 0.0
            if "param1" in action and "param2" in action:
                # Optimize for param1 around 60 and param2 around 40
                p1_score = 1.0 - abs(action["param1"] - 60) / 60
                p2_score = 1.0 - abs(action["param2"] - 40) / 40
                extra_factor = (p1_score + p2_score) / 2
                
            reward = min(1.0, reward * 0.7 + extra_factor * 0.3 + random.uniform(-0.15, 0.15))
            
        return max(0.0, reward)  # Ensure non-negative reward
    
    def _update_q_value(self, state: str, action: str, next_state: str, reward: float) -> None:
        """
        Update Q-value for a state-action pair.
        
        Args:
            state: Current state representation
            action: Action representation
            next_state: Next state representation
            reward: Reward received
        """
        # Ensure state exists in Q-table
        if state not in self.q_table:
            self.q_table[state] = {}
            
        # Ensure action exists in state's Q-values
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
            
        # Get current Q-value
        current_q = self.q_table[state][action]
        
        # Get max Q-value for next state
        next_max_q = 0.0
        if next_state in self.q_table:
            next_actions = self.q_table[next_state]
            if next_actions:
                next_max_q = max(next_actions.values())
                
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        
        # Update Q-value
        self.q_table[state][action] = new_q
    
    def _perturb_action(self, action: Dict[str, Any], action_space: List[str]) -> str:
        """
        Slightly modify an action to explore similar alternatives.
        
        Args:
            action: Current best action
            action_space: List of possible actions
            
        Returns:
            str: Modified action representation
        """
        # Simple implementation - randomly pick an action or modify a parameter
        if random.random() < 0.5:
            # Pick a random action
            return random.choice(action_space)
        else:
            # Modify a parameter if present
            if "param1" in action and "param2" in action:
                # Perturb parameters slightly
                param1 = max(0, min(100, action["param1"] + random.randint(-10, 10)))
                param2 = max(0, min(100, action["param2"] + random.randint(-10, 10)))
                return f"param1:{param1}|param2:{param2}"
            else:
                # No parameters to modify, pick random action
                return random.choice(action_space)
    
    def _record_optimization(self, reward_function: Dict[str, Any],
                           environment_state: Dict[str, Any],
                           action_type: str, best_action: Dict[str, Any],
                           best_reward: float, steps_taken: int) -> None:
        """
        Record optimization results in history.
        
        Args:
            reward_function: Reward function used
            environment_state: Environment state
            action_type: Type of action optimized
            best_action: Best action found
            best_reward: Best reward achieved
            steps_taken: Number of steps taken
        """
        # Create history entry
        entry = {
            "timestamp": time.time(),
            "action_type": action_type,
            "reward_function": {
                "domain": reward_function.get("domain", "general"),
                "complexity": reward_function.get("complexity", "medium")
            },
            "environment_state_summary": {
                k: v for k, v in environment_state.items() 
                if not isinstance(v, (dict, list, tuple))
            },
            "best_action": best_action,
            "best_reward": best_reward,
            "steps_taken": steps_taken
        }
        
        # Add to history
        self.action_history.append(entry)
        
        # Trim history if needed
        if len(self.action_history) > self.max_history_size:
            self.action_history.pop(0)
    
    def _record_feedback(self, action: Dict[str, Any],
                        expected_reward: float, actual_reward: float,
                        environment_state: Dict[str, Any]) -> None:
        """
        Record feedback results in history.
        
        Args:
            action: Action that was taken
            expected_reward: Expected reward
            actual_reward: Actual reward received
            environment_state: Environment state when action was taken
        """
        # Create history entry
        entry = {
            "timestamp": time.time(),
            "action": action,
            "expected_reward": expected_reward,
            "actual_reward": actual_reward,
            "environment_state_summary": {
                k: v for k, v in environment_state.items() 
                if not isinstance(v, (dict, list, tuple))
            },
            "reward_delta": actual_reward - expected_reward
        }
        
        # Add to history
        self.action_history.append(entry)
        
        # Trim history if needed
        if len(self.action_history) > self.max_history_size:
            self.action_history.pop(0)
