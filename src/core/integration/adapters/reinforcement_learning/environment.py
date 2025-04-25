"""
Environment State Manager Module for Reinforcement Learning Adapter.

This module provides functionality for managing and representing the state of
the environment for reinforcement learning tasks. It enables the translation
of real-world states into representations that RL algorithms can process.
"""

import json
import threading
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ....utils.logger import Logger


class StateDetailLevel(Enum):
    """Detail levels for environment state representation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EnvironmentStateManager:
    """
    Manager for environment state representations in reinforcement learning.
    
    This class provides functionality for creating, updating, and managing
    representations of the environment state that are suitable for use in
    reinforcement learning algorithms.
    """
    
    def __init__(self):
        """Initialize the environment state manager."""
        self.logger = Logger().get_logger("jarviee.integration.rl.environment_manager")
        self.initialized = False
        self.state_lock = threading.RLock()
        
        # Current environment state
        self.current_state = {}
        
        # State history for tracking changes
        self.state_history = []
        self.max_history_size = 100
        
        # Feature extractors for different domains
        self.feature_extractors = {}
    
    def initialize(self) -> bool:
        """
        Initialize the environment state manager.
        
        Returns:
            bool: True if initialization was successful
        """
        if self.initialized:
            return True
            
        try:
            # Register default feature extractors
            self._register_default_extractors()
            
            self.initialized = True
            self.logger.info("Environment state manager initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing environment state manager: {str(e)}")
            return False
    
    def _register_default_extractors(self) -> None:
        """Register default feature extractors for different domains."""
        # General purpose feature extractor
        self.feature_extractors["general"] = self._extract_general_features
        
        # Programming domain feature extractor
        self.feature_extractors["programming"] = self._extract_programming_features
        
        # Dialog domain feature extractor
        self.feature_extractors["dialog"] = self._extract_dialog_features
    
    def update_state(self, state_data: Dict[str, Any]) -> bool:
        """
        Update the current environment state.
        
        Args:
            state_data: New state data to incorporate
            
        Returns:
            bool: True if update was successful
        """
        try:
            with self.state_lock:
                # Save current state to history
                if self.current_state:
                    self.state_history.append(self.current_state.copy())
                    
                    # Trim history if needed
                    if len(self.state_history) > self.max_history_size:
                        self.state_history.pop(0)
                
                # Update current state
                self.current_state.update(state_data)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating environment state: {str(e)}")
            return False
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current environment state.
        
        Returns:
            Dict: Current environment state
        """
        with self.state_lock:
            return self.current_state.copy()
    
    def get_state_history(self, limit: int = -1) -> List[Dict[str, Any]]:
        """
        Get the history of environment states.
        
        Args:
            limit: Maximum number of states to return (-1 for all)
            
        Returns:
            List[Dict]: History of environment states
        """
        with self.state_lock:
            if limit < 0:
                return self.state_history.copy()
            else:
                return self.state_history[-limit:].copy()
    
    def extract_features(self, state: Optional[Dict[str, Any]] = None, 
                        domain: str = "general", 
                        detail_level: Union[str, StateDetailLevel] = StateDetailLevel.MEDIUM) -> Dict[str, Any]:
        """
        Extract features from an environment state for RL algorithms.
        
        Args:
            state: State to extract features from (default: current state)
            domain: Domain for feature extraction
            detail_level: Level of detail for extracted features
            
        Returns:
            Dict: Extracted features suitable for RL algorithms
        """
        # Use current state if none provided
        if state is None:
            state = self.get_current_state()
            
        # Convert string detail level to enum if needed
        if isinstance(detail_level, str):
            try:
                detail_level = StateDetailLevel(detail_level.lower())
            except ValueError:
                detail_level = StateDetailLevel.MEDIUM
                
        # Get appropriate feature extractor
        if domain in self.feature_extractors:
            extractor = self.feature_extractors[domain]
        else:
            # Fall back to general extractor
            extractor = self.feature_extractors["general"]
            
        # Extract features
        try:
            features = extractor(state, detail_level)
            return features
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            # Return empty features on error
            return {}
    
    def register_feature_extractor(self, domain: str, extractor_func: callable) -> None:
        """
        Register a custom feature extractor for a domain.
        
        Args:
            domain: Domain for the feature extractor
            extractor_func: Function that extracts features from state
        """
        self.feature_extractors[domain] = extractor_func
        self.logger.info(f"Registered feature extractor for domain: {domain}")
    
    def get_state_diff(self, previous_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get the difference between current state and a previous state.
        
        Args:
            previous_state: Previous state to compare against (default: previous in history)
            
        Returns:
            Dict: Differences between states
        """
        with self.state_lock:
            # Get current state
            current = self.current_state
            
            # Get previous state (from argument or history)
            if previous_state is None:
                if not self.state_history:
                    return current.copy()  # No history, return full current state
                previous_state = self.state_history[-1]
                
            # Calculate differences
            diff = {}
            
            # Find added or modified keys
            for key, value in current.items():
                if key not in previous_state or previous_state[key] != value:
                    diff[key] = value
                    
            # Mark removed keys
            for key in previous_state:
                if key not in current:
                    diff[key] = None
                    
            return diff
    
    def reset_state(self) -> None:
        """Reset the current environment state."""
        with self.state_lock:
            # Save current state to history if it exists
            if self.current_state:
                self.state_history.append(self.current_state.copy())
                
                # Trim history if needed
                if len(self.state_history) > self.max_history_size:
                    self.state_history.pop(0)
            
            # Reset current state
            self.current_state = {}
            
            self.logger.info("Environment state reset")
    
    def _extract_general_features(self, state: Dict[str, Any], 
                                detail_level: StateDetailLevel) -> Dict[str, Any]:
        """
        Extract general features from an environment state.
        
        Args:
            state: State to extract features from
            detail_level: Level of detail for extracted features
            
        Returns:
            Dict: Extracted features
        """
        features = {}
        
        # Basic flattening of state dictionary
        features = self._flatten_state(state, detail_level)
        
        return features
    
    def _extract_programming_features(self, state: Dict[str, Any], 
                                     detail_level: StateDetailLevel) -> Dict[str, Any]:
        """
        Extract programming-specific features from an environment state.
        
        Args:
            state: State to extract features from
            detail_level: Level of detail for extracted features
            
        Returns:
            Dict: Extracted features
        """
        features = {}
        
        # Basic features
        code_state = state.get("code", {})
        features["code_length"] = len(str(code_state.get("content", "")))
        features["language"] = code_state.get("language", "unknown")
        
        # Extract metrics if available
        metrics = code_state.get("metrics", {})
        if metrics:
            features["complexity"] = metrics.get("complexity", 0)
            features["bugs"] = metrics.get("bugs", 0)
            features["test_coverage"] = metrics.get("test_coverage", 0)
            
        # Detail level specific features
        if detail_level == StateDetailLevel.HIGH:
            # Include more detailed metrics
            if "detailed_metrics" in code_state:
                detailed = code_state["detailed_metrics"]
                features.update(detailed)
                
            # Include AST features if available
            if "ast" in code_state:
                ast_features = self._extract_ast_features(code_state["ast"])
                features.update(ast_features)
                
        return features
    
    def _extract_dialog_features(self, state: Dict[str, Any], 
                               detail_level: StateDetailLevel) -> Dict[str, Any]:
        """
        Extract dialog-specific features from an environment state.
        
        Args:
            state: State to extract features from
            detail_level: Level of detail for extracted features
            
        Returns:
            Dict: Extracted features
        """
        features = {}
        
        # Basic features
        dialog_state = state.get("dialog", {})
        features["turn_count"] = dialog_state.get("turn_count", 0)
        features["user_sentiment"] = dialog_state.get("user_sentiment", 0)
        
        # Extract message features
        messages = dialog_state.get("messages", [])
        if messages:
            features["last_message_length"] = len(str(messages[-1].get("content", "")))
            features["average_message_length"] = sum(len(str(m.get("content", ""))) for m in messages) / len(messages)
            
        # Detail level specific features
        if detail_level == StateDetailLevel.HIGH:
            # Include more detailed features
            features["user_messages"] = sum(1 for m in messages if m.get("role") == "user")
            features["system_messages"] = sum(1 for m in messages if m.get("role") == "system")
            
            # Include topic features if available
            if "topics" in dialog_state:
                topics = dialog_state["topics"]
                for topic, score in topics.items():
                    features[f"topic_{topic}"] = score
                    
        return features
    
    def _flatten_state(self, state: Dict[str, Any], detail_level: StateDetailLevel,
                     prefix: str = "", max_depth: int = 3) -> Dict[str, Any]:
        """
        Flatten a nested state dictionary for feature extraction.
        
        Args:
            state: State dictionary to flatten
            detail_level: Level of detail to include
            prefix: Prefix for flattened keys
            max_depth: Maximum depth to flatten
            
        Returns:
            Dict: Flattened state dictionary
        """
        features = {}
        
        # Set depth limit based on detail level
        if detail_level == StateDetailLevel.LOW:
            depth_limit = 1
        elif detail_level == StateDetailLevel.MEDIUM:
            depth_limit = 2
        else:  # HIGH
            depth_limit = max_depth
            
        # Recursive helper function
        def _flatten_recursive(d, current_prefix, current_depth):
            if current_depth > depth_limit:
                return
                
            for key, value in d.items():
                new_key = f"{current_prefix}{key}" if current_prefix else key
                
                if isinstance(value, dict) and current_depth < depth_limit:
                    # Recurse into nested dictionary
                    _flatten_recursive(value, f"{new_key}.", current_depth + 1)
                elif isinstance(value, (list, tuple)):
                    # Handle lists/tuples based on detail level
                    if detail_level == StateDetailLevel.LOW:
                        # Just include length for low detail
                        features[f"{new_key}_length"] = len(value)
                    else:
                        # Include summary statistics for medium/high detail
                        if value and all(isinstance(x, (int, float)) for x in value):
                            # Numeric list - include statistics
                            features[f"{new_key}_length"] = len(value)
                            features[f"{new_key}_sum"] = sum(value)
                            features[f"{new_key}_avg"] = sum(value) / len(value)
                            features[f"{new_key}_max"] = max(value)
                            features[f"{new_key}_min"] = min(value)
                        else:
                            # Mixed or non-numeric list - just include length
                            features[f"{new_key}_length"] = len(value)
                            
                            # Include first few items for high detail
                            if detail_level == StateDetailLevel.HIGH:
                                for i, item in enumerate(value[:3]):
                                    if isinstance(item, (int, float, str, bool)):
                                        features[f"{new_key}_{i}"] = item
                else:
                    # Include the value directly if it's a basic type
                    if isinstance(value, (int, float, str, bool)):
                        features[new_key] = value
                        
        # Start recursive flattening
        _flatten_recursive(state, prefix, 1)
        
        return features
    
    def _extract_ast_features(self, ast_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from an Abstract Syntax Tree representation.
        
        Args:
            ast_data: AST data to extract features from
            
        Returns:
            Dict: Extracted AST features
        """
        features = {}
        
        # Basic AST metrics
        features["ast_node_count"] = ast_data.get("node_count", 0)
        features["ast_depth"] = ast_data.get("max_depth", 0)
        
        # Node type counts
        node_types = ast_data.get("node_types", {})
        for node_type, count in node_types.items():
            features[f"ast_nodes_{node_type}"] = count
            
        return features
