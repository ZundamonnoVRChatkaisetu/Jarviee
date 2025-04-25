"""
Technology Dispatcher for AI Technology Orchestration.

This module implements the dispatcher component of the integration coordinator,
which is responsible for routing messages to the appropriate technology adapters
and managing the dispatch strategies for different integration patterns.
"""

import threading
from typing import Any, Dict, List, Optional, Set, Type, Union

from ...utils.event_bus import Event, EventBus
from ...utils.logger import Logger
from ..adapters.registry import AdapterRegistry
from ..base import AIComponent, ComponentType, IntegrationMessage
from ..registry import ComponentRegistry


class TechnologyDispatcher(AIComponent):
    """
    Dispatcher for routing tasks to appropriate AI technologies.
    
    This component manages the routing of messages between the integration
    coordinator and various technology adapters, handling dispatch strategies
    and load balancing across available technology instances.
    """
    
    def __init__(self, component_id: str, coordinator):
        """
        Initialize the technology dispatcher.
        
        Args:
            component_id: Unique identifier for this component
            coordinator: The integration coordinator component
        """
        super().__init__(component_id, ComponentType.SYSTEM)
        
        # Store reference to coordinator
        self.coordinator = coordinator
        
        # Initialize logger
        self.logger = Logger().get_logger("jarviee.integration.dispatcher")
        
        # Initialize dependencies
        self.component_registry = ComponentRegistry()
        self.adapter_registry = AdapterRegistry()
        
        # Dispatch state
        self.dispatch_count: Dict[ComponentType, int] = {
            ctype: 0 for ctype in ComponentType
        }
        self.pending_dispatches: Set[str] = set()
        
        # Configuration (will be updated from coordinator)
        self.config = {
            "load_balancing": "round_robin",  # round_robin, least_loaded, capability_based
            "retry_attempts": 3,
            "retry_delay": 1.0,  # seconds
            "fallback_enabled": True
        }
        
        self.logger.info("Technology Dispatcher initialized")
    
    def process_message(self, message: IntegrationMessage) -> None:
        """
        Process an incoming integration message.
        
        Args:
            message: The message to process
        """
        message_type = message.message_type
        
        if message_type == "dispatch":
            # Handle explicit dispatch request
            self._handle_dispatch_request(message)
        elif message_type == "component.initialized" or message_type == "component.started":
            # Technology component has become available
            self._handle_component_availability(message, True)
        elif message_type == "component.stopped" or message_type == "component.shutdown":
            # Technology component has become unavailable
            self._handle_component_availability(message, False)
        else:
            # For other messages, just dispatch to the appropriate technology
            self.dispatch_message(message)
    
    def _handle_dispatch_request(self, message: IntegrationMessage) -> None:
        """
        Handle an explicit request to dispatch a message to technologies.
        
        Args:
            message: The dispatch request message
        """
        target_type = message.content.get("target_type")
        target_message = message.content.get("message", {})
        message_type = message.content.get("message_type", "execute_task")
        dispatch_id = message.content.get("dispatch_id", message.message_id)
        
        if not target_type:
            self.send_message(
                message.source_component,
                "error",
                {
                    "error_code": "missing_target_type",
                    "error_message": "Target technology type is required for dispatch",
                    "dispatch_id": dispatch_id,
                    "success": False
                },
                correlation_id=message.message_id
            )
            return
        
        # Convert string type to enum if needed
        if isinstance(target_type, str):
            try:
                target_type = ComponentType[target_type.upper()]
            except (KeyError, ValueError):
                self.send_message(
                    message.source_component,
                    "error",
                    {
                        "error_code": "invalid_target_type",
                        "error_message": f"Invalid technology type: {target_type}",
                        "dispatch_id": dispatch_id,
                        "success": False
                    },
                    correlation_id=message.message_id
                )
                return
        
        # Dispatch to the target technology
        success = self.dispatch_to_technology(
            target_type,
            message_type,
            target_message,
            dispatch_id,
            message.correlation_id
        )
        
        if success:
            self.send_message(
                message.source_component,
                "response",
                {
                    "message_type": "dispatch",
                    "dispatch_id": dispatch_id,
                    "target_type": target_type.name,
                    "success": True
                },
                correlation_id=message.message_id
            )
        else:
            self.send_message(
                message.source_component,
                "error",
                {
                    "error_code": "dispatch_failed",
                    "error_message": f"Failed to dispatch to {target_type.name}",
                    "dispatch_id": dispatch_id,
                    "target_type": target_type.name,
                    "success": False
                },
                correlation_id=message.message_id
            )
    
    def _handle_component_availability(self, message: IntegrationMessage, 
                                      available: bool) -> None:
        """
        Handle a component becoming available or unavailable.
        
        Args:
            message: The component lifecycle message
            available: Whether the component is becoming available
        """
        component_id = message.content.get("component_id")
        component_type_str = message.content.get("component_type")
        
        if not component_id or not component_type_str:
            return
        
        try:
            component_type = ComponentType[component_type_str]
            
            if available:
                self.logger.info(f"Technology component {component_id} of type {component_type.name} is now available")
            else:
                self.logger.info(f"Technology component {component_id} of type {component_type.name} is now unavailable")
                
        except (KeyError, ValueError):
            # Not a valid component type, ignore
            pass
    
    def dispatch_message(self, message: IntegrationMessage) -> bool:
        """
        Dispatch a message to the appropriate technology based on message type.
        
        Args:
            message: The message to dispatch
            
        Returns:
            bool: True if dispatch was successful
        """
        # First, try to get explicit target from the message
        target_component = message.target_component
        
        if target_component:
            # Explicit target, send directly to that component
            self.event_bus.publish(message.to_event())
            return True
        
        # No explicit target, try to infer from message type
        message_type = message.message_type
        target_type = None
        
        # Infer technology type from message type prefix (e.g., "reinforcement_learning.execute")
        for ctype in ComponentType:
            type_prefix = ctype.name.lower() + "."
            if message_type.startswith(type_prefix):
                target_type = ctype
                break
        
        if not target_type:
            # No target type inferred, check for generic keywords
            keywords = {
                ComponentType.REINFORCEMENT_LEARNING: ["reward", "action", "environment", "agent"],
                ComponentType.SYMBOLIC_AI: ["logic", "inference", "reasoning", "knowledge"],
                ComponentType.MULTIMODAL: ["image", "audio", "vision", "speech"],
                ComponentType.AGENT: ["goal", "task", "plan", "autonomous"],
                ComponentType.NEUROMORPHIC: ["neural", "spike", "neuron", "brain"]
            }
            
            # Check message content for keywords
            content_str = str(message.content).lower()
            for ctype, kwords in keywords.items():
                if any(kw in content_str for kw in kwords):
                    target_type = ctype
                    break
        
        if target_type:
            # Send to the best available component of this type
            return self._dispatch_to_best_component(target_type, message)
        else:
            # No target determined, log and return false
            self.logger.warning(f"Unable to determine target for message type: {message_type}")
            return False
    
    def dispatch_to_technology(self, technology_type: ComponentType, message_type: str,
                              content: Dict[str, Any], dispatch_id: str,
                              correlation_id: Optional[str] = None) -> bool:
        """
        Dispatch a message to a specific technology type.
        
        Args:
            technology_type: Type of technology to dispatch to
            message_type: Type of message to send
            content: Message payload
            dispatch_id: ID to track this dispatch
            correlation_id: Optional correlation ID for responses
            
        Returns:
            bool: True if dispatch was successful
        """
        # Create a message for this dispatch
        message = IntegrationMessage(
            source_component=self.component_id,
            target_component=None,  # Will be determined by _dispatch_to_best_component
            message_type=message_type,
            content=content,
            correlation_id=correlation_id,
            message_id=dispatch_id
        )
        
        # Track this dispatch
        self.pending_dispatches.add(dispatch_id)
        
        # Send to the best available component of this type
        return self._dispatch_to_best_component(technology_type, message)
    
    def _dispatch_to_best_component(self, technology_type: ComponentType, 
                                   message: IntegrationMessage) -> bool:
        """
        Dispatch a message to the best available component of a specific type.
        
        Args:
            technology_type: Type of technology to dispatch to
            message: The message to dispatch
            
        Returns:
            bool: True if dispatch was successful
        """
        # Get all components of this type
        components = self.component_registry.get_components_by_type(technology_type)
        
        if not components:
            self.logger.warning(f"No components available for type {technology_type.name}")
            return False
        
        # Determine the best component based on load balancing strategy
        best_component = None
        
        if self.config["load_balancing"] == "round_robin":
            # Round-robin selection
            count = self.dispatch_count[technology_type]
            index = count % len(components)
            best_component = components[index]
            self.dispatch_count[technology_type] = count + 1
            
        elif self.config["load_balancing"] == "least_loaded":
            # Select the least loaded component
            best_load = float('inf')
            
            for component in components:
                status = self.component_registry.get_component_status(component.component_id)
                if status:
                    load = status.get("current_load", 0.0)
                    if load < best_load:
                        best_load = load
                        best_component = component
            
            # If no load information, fall back to first component
            if not best_component:
                best_component = components[0]
                
        elif self.config["load_balancing"] == "capability_based":
            # Select based on capabilities (for adapters)
            required_capabilities = self._extract_required_capabilities(message)
            
            if required_capabilities:
                best_match = 0
                
                for component in components:
                    # Get component capabilities
                    status = self.component_registry.get_component_status(component.component_id)
                    if status and "capabilities" in status:
                        capabilities = status["capabilities"]
                        match_count = sum(1 for cap in required_capabilities if cap in capabilities)
                        
                        if match_count > best_match:
                            best_match = match_count
                            best_component = component
            
            # If no capability match, fall back to first component
            if not best_component:
                best_component = components[0]
                
        else:
            # Default to first component
            best_component = components[0]
        
        # Set the target component in the message
        message.target_component = best_component.component_id
        
        # Publish the message
        self.event_bus.publish(message.to_event())
        self.logger.info(f"Dispatched {message.message_type} to {best_component.component_id}")
        
        return True
    
    def _extract_required_capabilities(self, message: IntegrationMessage) -> List[str]:
        """
        Extract required capabilities from a message.
        
        Args:
            message: The message to analyze
            
        Returns:
            List[str]: Required capabilities
        """
        required_capabilities = []
        
        # Check if message content explicitly lists required capabilities
        if "required_capabilities" in message.content:
            caps = message.content["required_capabilities"]
            if isinstance(caps, list):
                required_capabilities.extend(caps)
            elif isinstance(caps, str):
                required_capabilities.append(caps)
        
        # Infer from message type and content
        message_type = message.message_type
        
        # Add capabilities based on message type
        if "optimize" in message_type:
            required_capabilities.append("optimization")
        if "learning" in message_type:
            required_capabilities.append("learning")
        if "inference" in message_type:
            required_capabilities.append("inference")
        if "reasoning" in message_type:
            required_capabilities.append("reasoning")
        
        return required_capabilities
    
    def handle_dispatch_response(self, dispatch_id: str, technology_type: ComponentType,
                               result: Dict[str, Any], success: bool) -> None:
        """
        Handle a response from a technology for a previous dispatch.
        
        Args:
            dispatch_id: ID of the dispatch
            technology_type: Type of technology that responded
            result: The result from the technology
            success: Whether the technology operation was successful
        """
        # Remove from pending dispatches
        if dispatch_id in self.pending_dispatches:
            self.pending_dispatches.remove(dispatch_id)
        
        # Forward to the coordinator
        self.coordinator.handle_technology_response(
            technology_type,
            dispatch_id,
            result,
            success
        )
    
    def apply_config(self, config: Dict[str, Any]) -> None:
        """
        Apply configuration from the coordinator.
        
        Args:
            config: Configuration dictionary
        """
        # Update dispatcher-specific config
        self.config["load_balancing"] = config.get("load_balancing", self.config["load_balancing"])
        self.config["retry_attempts"] = config.get("retry_attempts", self.config["retry_attempts"])
        self.config["retry_delay"] = config.get("retry_delay", self.config["retry_delay"])
        self.config["fallback_enabled"] = config.get("auto_fallback", self.config["fallback_enabled"])
    
    def _initialize_impl(self) -> bool:
        """
        Implementation of component initialization.
        
        Returns:
            bool: True if initialization was successful
        """
        # Nothing specific to initialize
        return True
    
    def _start_impl(self) -> bool:
        """
        Implementation of component start.
        
        Returns:
            bool: True if start was successful
        """
        # Reset dispatch counters
        self.dispatch_count = {ctype: 0 for ctype in ComponentType}
        self.pending_dispatches.clear()
        
        return True
    
    def _stop_impl(self) -> bool:
        """
        Implementation of component stop.
        
        Returns:
            bool: True if stop was successful
        """
        # Nothing specific to stop
        return True
    
    def _shutdown_impl(self) -> bool:
        """
        Implementation of component shutdown.
        
        Returns:
            bool: True if shutdown was successful
        """
        # Clear state
        self.pending_dispatches.clear()
        
        return True
    
    def _get_status_impl(self) -> Dict[str, Any]:
        """
        Get component-specific status information.
        
        Returns:
            Dict: Component-specific status
        """
        return {
            "pending_dispatches": len(self.pending_dispatches),
            "dispatch_counts": {t.name: c for t, c in self.dispatch_count.items()},
            "load_balancing_strategy": self.config["load_balancing"],
            "fallback_enabled": self.config["fallback_enabled"]
        }
