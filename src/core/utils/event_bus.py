"""
Event Bus Module for Jarviee System.

This module implements a centralized event bus system that allows different
components of the Jarviee system to communicate through an asynchronous,
event-driven architecture. The EventBus facilitates loose coupling between
components and enables efficient integration of multiple AI technologies.
"""

import asyncio
import inspect
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

# Setup module logger
logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Priority levels for events in the event bus."""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class Event:
    """Base event class for all events in the system."""
    event_type: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: EventPriority = EventPriority.NORMAL
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate event data after initialization."""
        if not self.event_type:
            raise ValueError("Event type cannot be empty")
        if not self.source:
            raise ValueError("Event source cannot be empty")


class EventBus:
    """
    Central event bus for the Jarviee system.
    
    The EventBus implements a publish-subscribe pattern that allows components
    to communicate without direct dependencies. It supports synchronous and 
    asynchronous event handling, event filtering, and priority-based processing.
    """
    
    _instance = None  # Singleton instance
    
    def __new__(cls):
        """Ensure EventBus is a singleton."""
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the event bus if not already initialized."""
        if self._initialized:
            return
            
        self._subscribers: Dict[str, List[Callable]] = {}
        self._wildcard_subscribers: List[Callable] = []
        self._event_history: List[Event] = []
        self._max_history_size = 1000  # Default history size
        self._event_filters: Dict[str, List[Callable]] = {}
        self._loop = asyncio.get_event_loop() if asyncio.get_event_loop_policy().get_event_loop().is_running() else None
        self._initialized = True
        
        logger.info("EventBus initialized")
    
    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """
        Subscribe to a specific event type.
        
        Args:
            event_type: The type of event to subscribe to. Use "*" for all events.
            callback: Function to call when an event of this type is published.
                      The callback will receive the Event object as its argument.
        """
        if event_type == "*":
            if callback not in self._wildcard_subscribers:
                self._wildcard_subscribers.append(callback)
                logger.debug(f"Added wildcard subscription for {callback.__name__}")
        else:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)
                logger.debug(f"Added subscription for {event_type} to {callback.__name__}")
    
    def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        """
        Unsubscribe from a specific event type.
        
        Args:
            event_type: The event type to unsubscribe from. Use "*" for wildcard subscriptions.
            callback: The callback function to remove.
            
        Returns:
            bool: True if successfully unsubscribed, False otherwise.
        """
        if event_type == "*":
            if callback in self._wildcard_subscribers:
                self._wildcard_subscribers.remove(callback)
                logger.debug(f"Removed wildcard subscription for {callback.__name__}")
                return True
        elif event_type in self._subscribers:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Removed subscription for {event_type} from {callback.__name__}")
                return True
        return False
    
    def add_filter(self, event_type: str, filter_func: Callable[[Event], bool]) -> None:
        """
        Add a filter for a specific event type.
        
        Args:
            event_type: The event type to filter.
            filter_func: A function that returns True if the event should be processed.
        """
        if event_type not in self._event_filters:
            self._event_filters[event_type] = []
        self._event_filters[event_type].append(filter_func)
        logger.debug(f"Added filter for {event_type}")
    
    def remove_filter(self, event_type: str, filter_func: Callable[[Event], bool]) -> bool:
        """
        Remove a filter for a specific event type.
        
        Args:
            event_type: The event type to remove the filter from.
            filter_func: The filter function to remove.
            
        Returns:
            bool: True if successfully removed, False otherwise.
        """
        if event_type in self._event_filters and filter_func in self._event_filters[event_type]:
            self._event_filters[event_type].remove(filter_func)
            logger.debug(f"Removed filter for {event_type}")
            return True
        return False
    
    def _should_process_event(self, event: Event) -> bool:
        """
        Check if an event should be processed based on filters.
        
        Args:
            event: The event to check.
            
        Returns:
            bool: True if the event should be processed, False otherwise.
        """
        if event.event_type in self._event_filters:
            for filter_func in self._event_filters[event.event_type]:
                if not filter_func(event):
                    return False
        return True
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to the bus.
        
        Args:
            event: The event to publish.
        """
        if not self._should_process_event(event):
            logger.debug(f"Event {event.event_id} filtered out: {event.event_type}")
            return
            
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)
            
        logger.debug(f"Publishing event: {event.event_type} (ID: {event.event_id})")
        
        # Notify specific subscribers
        if event.event_type in self._subscribers:
            for callback in self._subscribers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback) and self._loop:
                        asyncio.run_coroutine_threadsafe(callback(event), self._loop)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in subscriber callback {callback.__name__}: {str(e)}")
        
        # Notify wildcard subscribers
        for callback in self._wildcard_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback) and self._loop:
                    asyncio.run_coroutine_threadsafe(callback(event), self._loop)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in wildcard subscriber callback {callback.__name__}: {str(e)}")
    
    async def publish_async(self, event: Event) -> None:
        """
        Publish an event asynchronously.
        
        Args:
            event: The event to publish.
        """
        if not self._should_process_event(event):
            logger.debug(f"Event {event.event_id} filtered out: {event.event_type}")
            return
            
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)
            
        logger.debug(f"Async publishing event: {event.event_type} (ID: {event.event_id})")
        
        # Notify specific subscribers
        tasks = []
        
        if event.event_type in self._subscribers:
            for callback in self._subscribers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        tasks.append(asyncio.create_task(callback(event)))
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in subscriber callback {callback.__name__}: {str(e)}")
        
        # Notify wildcard subscribers
        for callback in self._wildcard_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(asyncio.create_task(callback(event)))
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in wildcard subscriber callback {callback.__name__}: {str(e)}")
                
        # Wait for all async tasks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """
        Get event history, optionally filtered by type.
        
        Args:
            event_type: If provided, only return events of this type.
            limit: Maximum number of events to return.
            
        Returns:
            List of events, most recent first.
        """
        if event_type:
            filtered = [e for e in self._event_history if e.event_type == event_type]
            return filtered[-limit:] if limit > 0 else filtered
        else:
            return self._event_history[-limit:] if limit > 0 else self._event_history.copy()
    
    def clear_history(self) -> None:
        """Clear the event history."""
        self._event_history.clear()
        logger.debug("Event history cleared")
    
    def set_max_history_size(self, size: int) -> None:
        """
        Set the maximum number of events to keep in history.
        
        Args:
            size: Maximum number of events to store.
        """
        if size < 0:
            raise ValueError("History size cannot be negative")
            
        self._max_history_size = size
        # Trim if needed
        if len(self._event_history) > size:
            self._event_history = self._event_history[-size:]
        logger.debug(f"Max history size set to {size}")

    def on(self, event_name, callback):
        # ダミー実装: イベント購読用
        pass
