"""
Base Adapter for AI Technology Integration.

This module defines the base adapter class that serves as a foundation for
all AI technology adapters in the Jarviee system. It provides common functionality
for data transformation, communication standardization, and lifecycle management.
"""

import abc
from typing import Any, Dict, List, Optional, Type, Union

from ...utils.logger import Logger
from ..base import AIComponent, ComponentType, IntegrationMessage


class TechnologyAdapter(AIComponent):
    """
    Base class for all AI technology adapters.
    
    This class extends the AIComponent base class to provide specific functionality
    for adapting different AI technologies to work with the Jarviee system. It
    handles message transformation, protocol adaptation, and provides a standardized
    interface for technology-specific implementations.
    """
    
    def __init__(self, adapter_id: str, technology_type: ComponentType, 
                 llm_component_id: str = "llm_core"):
        """
        Initialize the technology adapter.
        
        Args:
            adapter_id: Unique identifier for this adapter
            technology_type: Type of AI technology this adapter represents
            llm_component_id: ID of the LLM core component to connect with
        """
        super().__init__(adapter_id, technology_type)
        self.llm_component_id = llm_component_id
        self.logger = Logger().get_logger(f"jarviee.integration.adapter.{adapter_id}")
        
        # Additional adapter-specific state
        self.capabilities = []
        self.config = {}
        self.status_info = {}
        
        self.logger.info(f"Adapter {adapter_id} for {technology_type.name} initialized")
    
    def process_message(self, message: IntegrationMessage) -> None:
        """
        Process an incoming integration message.
        
        This method identifies the type of message and routes it to the
        appropriate handler method based on the message type.
        
        Args:
            message: The message to process
        """
        self.logger.debug(f"Processing message: {message.message_type} from {message.source_component}")
        
        # Extract message type for routing
        message_type = message.message_type
        
        # Route to appropriate handler
        if message_type == "query":
            self._handle_query(message)
        elif message_type == "command":
            self._handle_command(message)
        elif message_type == "notification":
            self._handle_notification(message)
        elif message_type == "response":
            self._handle_response(message)
        elif message_type == "error":
            self._handle_error(message)
        elif message_type.startswith("llm."):
            self._handle_llm_message(message)
        elif message_type.startswith(f"{self.component_type.name.lower()}."):
            self._handle_technology_message(message)
        elif message_type.startswith("config."):
            self._handle_config_message(message)
        else:
            # Unknown message type, pass to generic handler
            self._handle_unknown_message(message)
    
    def _handle_query(self, message: IntegrationMessage) -> None:
        """
        Handle a query message that requests information.
        
        Args:
            message: The query message
        """
        # Get query type from content
        query_type = message.content.get("query_type", "unknown")
        
        # Handle based on query type
        if query_type == "capability":
            # Return adapter capabilities
            self.send_message(
                message.source_component,
                "response",
                {
                    "query_type": "capability",
                    "capabilities": self.capabilities,
                    "success": True
                },
                correlation_id=message.message_id
            )
        elif query_type == "status":
            # Return adapter status
            self.send_message(
                message.source_component,
                "response",
                {
                    "query_type": "status",
                    "status": self.get_status(),
                    "success": True
                },
                correlation_id=message.message_id
            )
        else:
            # Unknown query type, delegate to implementation
            self._handle_technology_query(message)
    
    @abc.abstractmethod
    def _handle_technology_query(self, message: IntegrationMessage) -> None:
        """
        Handle a technology-specific query.
        
        Args:
            message: The query message
        """
        pass
    
    def _handle_command(self, message: IntegrationMessage) -> None:
        """
        Handle a command message that requests an action.
        
        Args:
            message: The command message
        """
        # Get command type from content
        command_type = message.content.get("command_type", "unknown")
        
        # Handle based on command type
        if command_type == "configure":
            # Update adapter configuration
            config_updates = message.content.get("config", {})
            self.config.update(config_updates)
            
            # Notify configuration applied
            self.send_message(
                message.source_component,
                "response",
                {
                    "command_type": "configure",
                    "success": True
                },
                correlation_id=message.message_id
            )
        else:
            # Unknown command type, delegate to implementation
            self._handle_technology_command(message)
    
    @abc.abstractmethod
    def _handle_technology_command(self, message: IntegrationMessage) -> None:
        """
        Handle a technology-specific command.
        
        Args:
            message: The command message
        """
        pass
    
    def _handle_notification(self, message: IntegrationMessage) -> None:
        """
        Handle a notification message that provides information.
        
        Args:
            message: The notification message
        """
        # Get notification type from content
        notification_type = message.content.get("notification_type", "unknown")
        
        # Common notification types can be handled here
        
        # Delegate to implementation for technology-specific notifications
        self._handle_technology_notification(message)
    
    @abc.abstractmethod
    def _handle_technology_notification(self, message: IntegrationMessage) -> None:
        """
        Handle a technology-specific notification.
        
        Args:
            message: The notification message
        """
        pass
    
    def _handle_response(self, message: IntegrationMessage) -> None:
        """
        Handle a response message that replies to a previous message.
        
        Args:
            message: The response message
        """
        # Responses typically need specific handling
        self._handle_technology_response(message)
    
    @abc.abstractmethod
    def _handle_technology_response(self, message: IntegrationMessage) -> None:
        """
        Handle a technology-specific response.
        
        Args:
            message: The response message
        """
        pass
    
    def _handle_error(self, message: IntegrationMessage) -> None:
        """
        Handle an error message.
        
        Args:
            message: The error message
        """
        # Log the error
        error_code = message.content.get("error_code", "unknown")
        error_message = message.content.get("error_message", "Unknown error")
        self.logger.error(f"Received error {error_code}: {error_message}")
        
        # Delegate to implementation for technology-specific error handling
        self._handle_technology_error(message)
    
    @abc.abstractmethod
    def _handle_technology_error(self, message: IntegrationMessage) -> None:
        """
        Handle a technology-specific error.
        
        Args:
            message: The error message
        """
        pass
    
    def _handle_llm_message(self, message: IntegrationMessage) -> None:
        """
        Handle a message from the LLM core.
        
        Args:
            message: The LLM message
        """
        # Extract the specific LLM message type
        llm_message_type = message.message_type.replace("llm.", "", 1)
        
        # Delegate to implementation for technology-specific LLM message handling
        self._handle_technology_llm_message(message, llm_message_type)
    
    @abc.abstractmethod
    def _handle_technology_llm_message(self, message: IntegrationMessage, 
                                       llm_message_type: str) -> None:
        """
        Handle a technology-specific message from the LLM core.
        
        Args:
            message: The LLM message
            llm_message_type: The specific type of LLM message
        """
        pass
    
    def _handle_technology_message(self, message: IntegrationMessage) -> None:
        """
        Handle a message specific to this technology.
        
        Args:
            message: The technology message
        """
        # Extract the specific technology message type
        tech_prefix = f"{self.component_type.name.lower()}."
        tech_message_type = message.message_type.replace(tech_prefix, "", 1)
        
        # Delegate to implementation for specific technology message handling
        self._handle_specific_technology_message(message, tech_message_type)
    
    @abc.abstractmethod
    def _handle_specific_technology_message(self, message: IntegrationMessage,
                                           tech_message_type: str) -> None:
        """
        Handle a specific technology message type.
        
        Args:
            message: The technology message
            tech_message_type: The specific type of technology message
        """
        pass
    
    def _handle_config_message(self, message: IntegrationMessage) -> None:
        """
        Handle a configuration message.
        
        Args:
            message: The configuration message
        """
        # Extract the specific config message type
        config_message_type = message.message_type.replace("config.", "", 1)
        
        if config_message_type == "update":
            # Update configuration
            config_updates = message.content.get("config", {})
            self.config.update(config_updates)
            self.logger.info(f"Configuration updated with {len(config_updates)} settings")
            
            # Apply configuration changes
            self._apply_config_updates(config_updates)
            
            # Send acknowledgment
            self.send_message(
                message.source_component,
                "response",
                {
                    "config_action": "update",
                    "success": True
                },
                correlation_id=message.message_id
            )
        elif config_message_type == "reset":
            # Reset configuration to defaults
            self.config = self._get_default_config()
            self.logger.info("Configuration reset to defaults")
            
            # Apply default configuration
            self._apply_config_updates(self.config)
            
            # Send acknowledgment
            self.send_message(
                message.source_component,
                "response",
                {
                    "config_action": "reset",
                    "success": True
                },
                correlation_id=message.message_id
            )
        elif config_message_type == "get":
            # Return current configuration
            self.send_message(
                message.source_component,
                "response",
                {
                    "config_action": "get",
                    "config": self.config,
                    "success": True
                },
                correlation_id=message.message_id
            )
    
    def _apply_config_updates(self, config_updates: Dict[str, Any]) -> None:
        """
        Apply configuration updates to the adapter.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        # Default implementation - can be overridden by subclasses
        pass
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration for this adapter.
        
        Returns:
            Dict: Default configuration settings
        """
        # Default implementation - should be overridden by subclasses
        return {}
    
    def _handle_unknown_message(self, message: IntegrationMessage) -> None:
        """
        Handle an unknown message type.
        
        Args:
            message: The unknown message
        """
        self.logger.warning(f"Received unknown message type: {message.message_type}")
        
        # Send error response
        self.send_message(
            message.source_component,
            "error",
            {
                "error_code": "unknown_message_type",
                "error_message": f"Unknown message type: {message.message_type}",
                "original_message_type": message.message_type
            },
            correlation_id=message.message_id
        )
    
    def send_to_llm(self, message_type: str, content: Dict[str, Any],
                   correlation_id: Optional[str] = None) -> str:
        """
        Send a message to the LLM core.
        
        Args:
            message_type: Type of message to send
            content: Message payload
            correlation_id: Optional ID to correlate related messages
            
        Returns:
            str: The message ID
        """
        return self.send_message(
            self.llm_component_id,
            message_type,
            content,
            correlation_id=correlation_id
        )
    
    def send_technology_notification(self, notification_type: str, 
                                    content: Dict[str, Any]) -> str:
        """
        Send a technology-specific notification to interested components.
        
        Args:
            notification_type: Type of notification
            content: Notification payload
            
        Returns:
            str: The message ID
        """
        tech_name = self.component_type.name.lower()
        message_type = f"{tech_name}.{notification_type}"
        
        # Add standard fields to content
        notification_content = {
            "notification_type": notification_type,
            "technology": tech_name,
            "adapter_id": self.component_id,
            **content
        }
        
        return self.send_message(
            None,  # Broadcast
            message_type,
            notification_content
        )
    
    def _get_status_impl(self) -> Dict[str, Any]:
        """
        Get adapter-specific status information.
        
        Returns:
            Dict: Adapter-specific status
        """
        return {
            "capabilities": self.capabilities,
            "config": self.config,
            **self.status_info
        }
