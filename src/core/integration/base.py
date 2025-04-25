"""
Base Integration Components for Jarviee AI Technology Integration.

This module defines the foundational classes and interfaces for integrating
different AI technologies with the Jarviee system. It provides abstract base
classes that standardize communication between components and establish
common patterns for extensible AI integration.
"""

import abc
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Type, Union

from ..utils.event_bus import Event, EventBus


class ComponentType(Enum):
    """Types of AI components that can be integrated with the system."""
    LLM = auto()
    REINFORCEMENT_LEARNING = auto()
    SYMBOLIC_AI = auto()
    MULTIMODAL = auto()
    AGENT = auto()
    NEUROMORPHIC = auto()
    KNOWLEDGE_BASE = auto()
    USER_INTERFACE = auto()
    SYSTEM = auto()


@dataclass
class IntegrationMessage:
    """
    Standard message format for communication between AI components.
    
    This class defines a common message structure that can be passed between
    different AI technologies, allowing them to exchange information in a
    standardized way regardless of their internal representations.
    """
    
    source_component: str
    target_component: Optional[str]
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_event(self) -> Event:
        """Convert the integration message to an event bus event."""
        return Event(
            event_type=f"integration.{self.message_type}",
            source=self.source_component,
            data={
                "message": self,
                "target": self.target_component,
                "content": self.content,
                "correlation_id": self.correlation_id,
            }
        )
    
    @classmethod
    def from_event(cls, event: Event) -> 'IntegrationMessage':
        """Create an integration message from an event bus event."""
        if "message" in event.data and isinstance(event.data["message"], IntegrationMessage):
            # The message is already in the event
            return event.data["message"]
        
        # Create a new message from the event data
        return cls(
            source_component=event.source,
            target_component=event.data.get("target"),
            message_type=event.event_type.replace("integration.", "", 1),
            content=event.data.get("content", {}),
            timestamp=event.timestamp,
            message_id=event.event_id,
            correlation_id=event.data.get("correlation_id"),
            metadata=event.data.get("metadata", {})
        )


class AIComponent(abc.ABC):
    """
    Abstract base class for all AI components in the Jarviee system.
    
    This class defines the common interface that all AI technology integrations
    must implement to interact with the rest of the system. It handles
    subscription to relevant events, message processing, and provides a
    standardized way to send messages to other components.
    """
    
    def __init__(self, component_id: str, component_type: ComponentType):
        """
        Initialize the AI component.
        
        Args:
            component_id: Unique identifier for this component
            component_type: Type of AI technology this component represents
        """
        self.component_id = component_id
        self.component_type = component_type
        self.event_bus = EventBus()
        self.is_initialized = False
        self.is_running = False
        
        # Subscribe to events targeted at this component
        self.event_bus.subscribe(
            f"integration.*", 
            self._handle_integration_event
        )
    
    def _handle_integration_event(self, event: Event) -> None:
        """
        Handle events from the event bus that might be relevant to this component.
        
        Args:
            event: The event to process
        """
        # Check if this message is targeted at this component
        if "target" in event.data and event.data["target"] != self.component_id:
            return
            
        # Convert to an integration message
        message = IntegrationMessage.from_event(event)
        
        # Process the message
        if asyncio.iscoroutinefunction(self.process_message):
            # Schedule async processing
            asyncio.create_task(self.process_message(message))
        else:
            # Direct synchronous processing
            self.process_message(message)
    
    @abc.abstractmethod
    def process_message(self, message: IntegrationMessage) -> None:
        """
        Process an incoming integration message.
        
        Args:
            message: The message to process
        """
        pass
    
    def send_message(self, target_component: Optional[str], 
                     message_type: str, content: Dict[str, Any],
                     correlation_id: Optional[str] = None,
                     priority: int = 0) -> str:
        """
        Send a message to another component or broadcast to all.
        
        Args:
            target_component: Target component ID, or None for broadcast
            message_type: Type of message being sent
            content: Message payload
            correlation_id: Optional ID to correlate related messages
            priority: Message priority (higher numbers = higher priority)
            
        Returns:
            str: The message ID
        """
        message = IntegrationMessage(
            source_component=self.component_id,
            target_component=target_component,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id,
            priority=priority
        )
        
        # Convert to event and publish
        self.event_bus.publish(message.to_event())
        
        return message.message_id
    
    async def send_message_async(self, target_component: Optional[str],
                                message_type: str, content: Dict[str, Any],
                                correlation_id: Optional[str] = None,
                                priority: int = 0) -> str:
        """
        Send a message asynchronously to another component or broadcast to all.
        
        Args:
            target_component: Target component ID, or None for broadcast
            message_type: Type of message being sent
            content: Message payload
            correlation_id: Optional ID to correlate related messages
            priority: Message priority (higher numbers = higher priority)
            
        Returns:
            str: The message ID
        """
        message = IntegrationMessage(
            source_component=self.component_id,
            target_component=target_component,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id,
            priority=priority
        )
        
        # Convert to event and publish asynchronously
        await self.event_bus.publish_async(message.to_event())
        
        return message.message_id
    
    def initialize(self) -> bool:
        """
        Initialize the component.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self.is_initialized:
            return True
            
        success = self._initialize_impl()
        if success:
            self.is_initialized = True
            # Announce component initialization
            self.send_message(
                None,  # Broadcast
                "component.initialized",
                {
                    "component_id": self.component_id,
                    "component_type": self.component_type.name
                }
            )
        
        return success
    
    def _initialize_impl(self) -> bool:
        """
        Implementation of component initialization logic.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        # Default implementation
        return True
    
    def start(self) -> bool:
        """
        Start the component.
        
        Returns:
            bool: True if start was successful, False otherwise
        """
        if not self.is_initialized:
            if not self.initialize():
                return False
                
        if self.is_running:
            return True
            
        success = self._start_impl()
        if success:
            self.is_running = True
            # Announce component start
            self.send_message(
                None,  # Broadcast
                "component.started",
                {
                    "component_id": self.component_id,
                    "component_type": self.component_type.name
                }
            )
        
        return success
    
    def _start_impl(self) -> bool:
        """
        Implementation of component start logic.
        
        Returns:
            bool: True if start was successful, False otherwise
        """
        # Default implementation
        return True
    
    def stop(self) -> bool:
        """
        Stop the component.
        
        Returns:
            bool: True if stop was successful, False otherwise
        """
        if not self.is_running:
            return True
            
        success = self._stop_impl()
        if success:
            self.is_running = False
            # Announce component stop
            self.send_message(
                None,  # Broadcast
                "component.stopped",
                {
                    "component_id": self.component_id,
                    "component_type": self.component_type.name
                }
            )
        
        return success
    
    def _stop_impl(self) -> bool:
        """
        Implementation of component stop logic.
        
        Returns:
            bool: True if stop was successful, False otherwise
        """
        # Default implementation
        return True
    
    def shutdown(self) -> bool:
        """
        Shut down the component.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        if self.is_running:
            if not self.stop():
                return False
                
        success = self._shutdown_impl()
        if success:
            self.is_initialized = False
            # Announce component shutdown
            self.send_message(
                None,  # Broadcast
                "component.shutdown",
                {
                    "component_id": self.component_id,
                    "component_type": self.component_type.name
                }
            )
            
            # Unsubscribe from events
            self.event_bus.unsubscribe(
                f"integration.*", 
                self._handle_integration_event
            )
        
        return success
    
    def _shutdown_impl(self) -> bool:
        """
        Implementation of component shutdown logic.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        # Default implementation
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the component.
        
        Returns:
            Dict: Component status information
        """
        status = {
            "component_id": self.component_id,
            "component_type": self.component_type.name,
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
        }
        
        # Add implementation-specific status
        status.update(self._get_status_impl())
        
        return status
    
    def _get_status_impl(self) -> Dict[str, Any]:
        """
        Implementation of component-specific status information.
        
        Returns:
            Dict: Component-specific status
        """
        # Default implementation
        return {}
