"""
Logging Module for Jarviee System.

This module implements a centralized logging system for the Jarviee application,
providing standardized logging across all components with configurable output
formats, levels, and destinations.
"""

import inspect
import logging
import os
import sys
import time
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Union

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"

# Log levels
class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class Logger:
    """
    Centralized logging manager for Jarviee system.
    
    This class provides a simplified interface to Python's logging module with
    additional features such as automatic context detection, structured logging,
    and multi-destination output.
    """
    
    _instance = None  # Singleton instance
    
    def __new__(cls):
        """Ensure Logger is a singleton."""
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the logger if not already initialized."""
        if self._initialized:
            return
            
        # Create root logger
        self.root_logger = logging.getLogger("jarviee")
        self.root_logger.setLevel(logging.DEBUG)  # Capture all, filter at handlers
        
        # Default paths
        self.logs_dir = os.environ.get("JARVIEE_LOGS_DIR", "logs")
        Path(self.logs_dir).mkdir(exist_ok=True)
        
        # Setup default handlers
        self.console_handler = None
        self.file_handler = None
        self.setup_default_handlers()
        
        # Track module loggers
        self.module_loggers = {}
        
        self._initialized = True
    
    def setup_default_handlers(self):
        """Setup default console and file handlers."""
        # Console handler
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        self.console_handler.setFormatter(console_formatter)
        self.root_logger.addHandler(self.console_handler)
        
        # File handler (daily rotation)
        log_file = Path(self.logs_dir) / "jarviee.log"
        self.file_handler = TimedRotatingFileHandler(
            log_file, when='midnight', backupCount=7
        )
        self.file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        self.file_handler.setFormatter(file_formatter)
        self.root_logger.addHandler(self.file_handler)
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Get a logger for a specific module.
        
        Args:
            name: Module name. If None, will attempt to detect the caller module.
            
        Returns:
            A configured logger instance.
        """
        if name is None:
            # Attempt to determine caller module
            frame = inspect.currentframe()
            try:
                if frame is not None and frame.f_back is not None:
                    module = inspect.getmodule(frame.f_back)
                    if module is not None:
                        name = module.__name__
            finally:
                del frame  # Avoid reference cycles
        
        if name is None:
            name = "jarviee.unknown"
        elif not name.startswith("jarviee."):
            name = f"jarviee.{name}"
            
        # Check if we already have this logger
        if name in self.module_loggers:
            return self.module_loggers[name]
            
        # Create new logger
        logger = logging.getLogger(name)
        self.module_loggers[name] = logger
        return logger
    
    def set_level(self, level: Union[LogLevel, int, str], handler_type: Optional[str] = None):
        """
        Set logging level for handlers.
        
        Args:
            level: Log level (can be LogLevel enum, int, or string name)
            handler_type: 'console', 'file', or None for both
        """
        # Convert level to int if needed
        if isinstance(level, LogLevel):
            level_value = level.value
        elif isinstance(level, str):
            level_value = getattr(logging, level.upper(), logging.INFO)
        else:
            level_value = level
            
        # Apply to specified handlers
        if handler_type is None or handler_type.lower() == 'console':
            self.console_handler.setLevel(level_value)
            
        if handler_type is None or handler_type.lower() == 'file':
            self.file_handler.setLevel(level_value)
    
    def add_file_handler(self, filename: str, level: Union[LogLevel, int, str] = LogLevel.DEBUG, 
                         max_size_mb: Optional[int] = None, backup_count: int = 5):
        """
        Add an additional file handler.
        
        Args:
            filename: Name of the log file
            level: Log level for this handler
            max_size_mb: If set, use size-based rotation with this max size (in MB)
            backup_count: Number of backup files to keep
        """
        # Convert level to int if needed
        if isinstance(level, LogLevel):
            level_value = level.value
        elif isinstance(level, str):
            level_value = getattr(logging, level.upper(), logging.INFO)
        else:
            level_value = level
            
        # Create path if needed
        log_path = Path(self.logs_dir) / filename
        log_path.parent.mkdir(exist_ok=True)
        
        # Create appropriate handler
        if max_size_mb:
            handler = RotatingFileHandler(
                log_path, maxBytes=max_size_mb * 1024 * 1024, 
                backupCount=backup_count
            )
        else:
            handler = TimedRotatingFileHandler(
                log_path, when='midnight', backupCount=backup_count
            )
            
        handler.setLevel(level_value)
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        handler.setFormatter(formatter)
        self.root_logger.addHandler(handler)
        return handler
    
    def set_format(self, format_str: str, handler_type: Optional[str] = None):
        """
        Set log format for handlers.
        
        Args:
            format_str: Format string for log messages
            handler_type: 'console', 'file', or None for both
        """
        formatter = logging.Formatter(format_str)
        
        if handler_type is None or handler_type.lower() == 'console':
            self.console_handler.setFormatter(formatter)
            
        if handler_type is None or handler_type.lower() == 'file':
            self.file_handler.setFormatter(formatter)
