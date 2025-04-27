"""
Configuration Management Module for Jarviee System.

This module provides a centralized configuration management system that handles
loading, validating, and accessing configuration settings across the entire
Jarviee application.
"""

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml


class Config:
    """
    Configuration manager for Jarviee system.
    
    This class handles loading configuration from files and environment variables,
    providing a unified interface for accessing and modifying configuration
    settings with change notifications and validation.
    """
    
    _instance = None  # Singleton instance
    _lock = threading.RLock()  # Thread-safe lock
    
    def __new__(cls, *args, **kwargs):
        """Ensure Config is a singleton."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: str = None):
        """Initialize the config manager if not already initialized."""
        if self._initialized:
            return
        self._config_data = {}
        self._observers = {}
        self._loaded_files = set()
        self._env_prefix = "JARVIEE_"
        self._config_dir = os.environ.get("JARVIEE_CONFIG_DIR", "config")
        self._default_configs = [
            Path(self._config_dir) / "default.yaml",
            Path(self._config_dir) / "default.json",
            Path(self._config_dir) / "local.yaml",
            Path(self._config_dir) / "local.json",
        ]
        self._initialized = True
        self.load_defaults()
        if config_path:
            self.load_file(config_path)
    
    def load_defaults(self) -> bool:
        """
        Load default configuration files.
        
        Returns:
            bool: True if at least one file was loaded successfully.
        """
        success = False
        for config_file in self._default_configs:
            if config_file.exists():
                success = self.load_file(str(config_file)) or success
        return success
    
    def load_file(self, file_path: str) -> bool:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file (YAML or JSON)
            
        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        path = Path(file_path)
        if not path.exists():
            return False
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    return False
                    
            # Merge with existing config
            with self._lock:
                self._deep_update(self._config_data, config)
                self._loaded_files.add(str(path.resolve()))
                
            return True
        except Exception as e:
            # Log error
            print(f"Error loading config file {file_path}: {str(e)}")
            return False
    
    def _deep_update(self, target: Dict, source: Dict) -> None:
        """
        Recursively update a dictionary without overwriting entire sub-dictionaries.
        
        Args:
            target: The dictionary to update
            source: The dictionary with updates
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
                # Notify observers for this key
                if key in self._observers:
                    for callback in self._observers[key]:
                        try:
                            callback(key, value)
                        except Exception:
                            # Just continue if callback fails
                            pass
    
    def load_env_vars(self) -> int:
        """
        Load configuration from environment variables.
        
        Environment variables starting with JARVIEE_ will be added to the config.
        Nested keys can be specified using double underscores, e.g.,
        JARVIEE_DATABASE__HOST will set config['database']['host'].
        
        Returns:
            int: Number of environment variables loaded.
        """
        count = 0
        with self._lock:
            for key, value in os.environ.items():
                if key.startswith(self._env_prefix):
                    # Remove prefix and convert to lowercase
                    config_key = key[len(self._env_prefix):].lower()
                    
                    # Handle nested keys (using double underscore as separator)
                    if '__' in config_key:
                        parts = config_key.split('__')
                        # Navigate to the right level in the config
                        curr = self._config_data
                        for part in parts[:-1]:
                            if part not in curr:
                                curr[part] = {}
                            elif not isinstance(curr[part], dict):
                                # Can't go deeper if it's not a dict
                                break
                            curr = curr[part]
                        else:  # This else belongs to the for loop (no break)
                            curr[parts[-1]] = self._parse_env_value(value)
                            count += 1
                    else:
                        # Top-level key
                        self._config_data[config_key] = self._parse_env_value(value)
                        count += 1
        return count
    
    def _parse_env_value(self, value: str) -> Any:
        """
        Parse environment variable value into appropriate Python type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Parsed value (bool, int, float, or string)
        """
        # Handle boolean values
        if value.lower() in ['true', 'yes', '1']:
            return True
        if value.lower() in ['false', 'no', '0']:
            return False
            
        # Try to convert to numeric
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            # Leave as string
            return value
    
    def get(self, key: str, default: Any = None, as_type: Optional[type] = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key, using dot notation for nested keys
            default: Default value if key not found
            as_type: Type to convert the value to
            
        Returns:
            The configuration value or default
        """
        with self._lock:
            # Handle dot notation for nested keys
            parts = key.split('.')
            curr = self._config_data
            
            # Navigate the nested structure
            for part in parts:
                if part not in curr:
                    return default
                curr = curr[part]
                
            # Apply type conversion if requested
            if as_type and curr is not None:
                try:
                    return as_type(curr)
                except (ValueError, TypeError):
                    return default
                    
            return curr
    
    def set(self, key: str, value: Any, persist: bool = False) -> None:
        """
        Set a configuration value.
        
        Args:
            key: The configuration key, using dot notation for nested keys
            value: The value to set
            persist: If True, save to local config file
        """
        with self._lock:
            # Handle dot notation for nested keys
            parts = key.split('.')
            curr = self._config_data
            
            # Navigate and create the nested structure as needed
            for part in parts[:-1]:
                if part not in curr or not isinstance(curr[part], dict):
                    curr[part] = {}
                curr = curr[part]
                
            # Set the value
            curr[parts[-1]] = value
            
            # Notify observers
            if key in self._observers:
                for callback in self._observers[key]:
                    try:
                        callback(key, value)
                    except Exception:
                        # Just continue if callback fails
                        pass
                        
        # Persist if requested
        if persist:
            self.save_local_config()
    
    def observe(self, key: str, callback: callable) -> None:
        """
        Register an observer for a configuration key.
        
        Args:
            key: The configuration key to observe
            callback: Function to call when the key changes
        """
        with self._lock:
            if key not in self._observers:
                self._observers[key] = []
            if callback not in self._observers[key]:
                self._observers[key].append(callback)
    
    def unobserve(self, key: str, callback: callable) -> bool:
        """
        Remove an observer for a configuration key.
        
        Args:
            key: The configuration key
            callback: The callback function to remove
            
        Returns:
            bool: True if successfully removed, False otherwise
        """
        with self._lock:
            if key in self._observers and callback in self._observers[key]:
                self._observers[key].remove(callback)
                return True
        return False
    
    def save_local_config(self, file_path: Optional[str] = None) -> bool:
        """
        Save current configuration to a local file.
        
        Args:
            file_path: Path to save to, defaults to config/local.yaml
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if file_path is None:
            file_path = os.path.join(self._config_dir, "local.yaml")
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write config to file
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config_data, f, default_flow_style=False)
                
            self._loaded_files.add(str(Path(file_path).resolve()))
            return True
        except Exception as e:
            # Log error
            print(f"Error saving config to {file_path}: {str(e)}")
            return False
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get a copy of the entire configuration.
        
        Returns:
            Dict: Copy of the configuration data
        """
        with self._lock:
            # Return a deep copy to prevent modification
            return json.loads(json.dumps(self._config_data))
    
    def reset(self) -> None:
        """Reset configuration to defaults."""
        with self._lock:
            self._config_data = {}
            self._loaded_files = set()
            
        # Reload defaults
        self.load_defaults()
        
    def get_loaded_files(self) -> List[str]:
        """
        Get list of loaded configuration files.
        
        Returns:
            List of file paths that have been loaded.
        """
        with self._lock:
            return list(self._loaded_files)
