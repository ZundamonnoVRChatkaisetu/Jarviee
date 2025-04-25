"""
Component Registry for Jarviee AI Technology Integration.

This module implements a registry system for managing AI components in the
Jarviee system. It provides a centralized way to register, discover, and
manage the lifecycle of AI components, enabling dynamic integration of
different AI technologies.
"""

import threading
from typing import Dict, List, Optional, Type, Union

from ..utils.event_bus import Event, EventBus
from ..utils.logger import Logger
from .base import AIComponent, ComponentType


class ComponentRegistry:
    """
    Registry for managing AI components in the Jarviee system.
    
    This class provides a centralized registry for all AI components, allowing
    components to be registered, discovered, and managed throughout their
    lifecycle. It plays a key role in enabling the dynamic integration of
    different AI technologies.
    """
    
    _instance = None  # Singleton instance
    _lock = threading.RLock()  # Thread-safe lock
    
    def __new__(cls):
        """Ensure ComponentRegistry is a singleton."""
        if cls._instance is None:
            cls._instance = super(ComponentRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the component registry if not already initialized."""
        if self._initialized:
            return
            
        # Initialize registry data structures
        self._components: Dict[str, AIComponent] = {}
        self._component_types: Dict[ComponentType, List[str]] = {
            ctype: [] for ctype in ComponentType
        }
        
        # Initialize dependencies
        self.event_bus = EventBus()
        self.logger = Logger().get_logger("jarviee.core.integration.registry")
        
        # Subscribe to component lifecycle events
        self.event_bus.subscribe("component.initialized", self._handle_component_event)
        self.event_bus.subscribe("component.started", self._handle_component_event)
        self.event_bus.subscribe("component.stopped", self._handle_component_event)
        self.event_bus.subscribe("component.shutdown", self._handle_component_event)
        
        self._initialized = True
        self.logger.info("Component Registry initialized")
    
    def _handle_component_event(self, event: Event) -> None:
        """
        Handle component lifecycle events from the event bus.
        
        Args:
            event: The event to process
        """
        if "component_id" not in event.data:
            return
            
        component_id = event.data["component_id"]
        event_type = event.event_type
        
        if component_id in self._components:
            self.logger.debug(f"Component {component_id} sent {event_type}")
    
    def register_component(self, component: AIComponent) -> bool:
        """
        Register an AI component with the registry.
        
        Args:
            component: The component to register
            
        Returns:
            bool: True if registration was successful, False otherwise
        """
        with self._lock:
            component_id = component.component_id
            
            # Check if already registered
            if component_id in self._components:
                self.logger.warning(f"Component {component_id} is already registered")
                return False
                
            # Add to registry
            self._components[component_id] = component
            self._component_types[component.component_type].append(component_id)
            
            self.logger.info(f"Registered component {component_id} of type {component.component_type.name}")
            
            # Publish registration event
            self.event_bus.publish(Event(
                event_type="registry.component_registered",
                source="component_registry",
                data={
                    "component_id": component_id,
                    "component_type": component.component_type.name
                }
            ))
            
            return True
    
    def unregister_component(self, component_id: str) -> bool:
        """
        Unregister an AI component from the registry.
        
        Args:
            component_id: ID of the component to unregister
            
        Returns:
            bool: True if unregistration was successful, False otherwise
        """
        with self._lock:
            # Check if registered
            if component_id not in self._components:
                self.logger.warning(f"Component {component_id} is not registered")
                return False
                
            # Get component and remove from registry
            component = self._components[component_id]
            del self._components[component_id]
            
            # Remove from type list
            if component_id in self._component_types[component.component_type]:
                self._component_types[component.component_type].remove(component_id)
                
            self.logger.info(f"Unregistered component {component_id}")
            
            # Publish unregistration event
            self.event_bus.publish(Event(
                event_type="registry.component_unregistered",
                source="component_registry",
                data={
                    "component_id": component_id,
                    "component_type": component.component_type.name
                }
            ))
            
            return True
    
    def get_component(self, component_id: str) -> Optional[AIComponent]:
        """
        Get a registered component by ID.
        
        Args:
            component_id: ID of the component to retrieve
            
        Returns:
            AIComponent: The component, or None if not found
        """
        with self._lock:
            return self._components.get(component_id)
    
    def get_components_by_type(self, component_type: ComponentType) -> List[AIComponent]:
        """
        Get all registered components of a specific type.
        
        Args:
            component_type: Type of components to retrieve
            
        Returns:
            List[AIComponent]: List of components of the specified type
        """
        with self._lock:
            component_ids = self._component_types.get(component_type, [])
            return [self._components[cid] for cid in component_ids if cid in self._components]
    
    def get_all_components(self) -> Dict[str, AIComponent]:
        """
        Get all registered components.
        
        Returns:
            Dict[str, AIComponent]: Dictionary of component ID to component
        """
        with self._lock:
            # Return a copy to prevent modification
            return dict(self._components)
    
    def initialize_component(self, component_id: str) -> bool:
        """
        Initialize a specific component.
        
        Args:
            component_id: ID of the component to initialize
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        with self._lock:
            component = self.get_component(component_id)
            if not component:
                self.logger.warning(f"Cannot initialize unknown component {component_id}")
                return False
                
            try:
                result = component.initialize()
                if result:
                    self.logger.info(f"Component {component_id} initialized successfully")
                else:
                    self.logger.error(f"Component {component_id} initialization failed")
                return result
            except Exception as e:
                self.logger.error(f"Error initializing component {component_id}: {str(e)}")
                return False
    
    def start_component(self, component_id: str) -> bool:
        """
        Start a specific component.
        
        Args:
            component_id: ID of the component to start
            
        Returns:
            bool: True if start was successful, False otherwise
        """
        with self._lock:
            component = self.get_component(component_id)
            if not component:
                self.logger.warning(f"Cannot start unknown component {component_id}")
                return False
                
            try:
                result = component.start()
                if result:
                    self.logger.info(f"Component {component_id} started successfully")
                else:
                    self.logger.error(f"Component {component_id} start failed")
                return result
            except Exception as e:
                self.logger.error(f"Error starting component {component_id}: {str(e)}")
                return False
    
    def stop_component(self, component_id: str) -> bool:
        """
        Stop a specific component.
        
        Args:
            component_id: ID of the component to stop
            
        Returns:
            bool: True if stop was successful, False otherwise
        """
        with self._lock:
            component = self.get_component(component_id)
            if not component:
                self.logger.warning(f"Cannot stop unknown component {component_id}")
                return False
                
            try:
                result = component.stop()
                if result:
                    self.logger.info(f"Component {component_id} stopped successfully")
                else:
                    self.logger.error(f"Component {component_id} stop failed")
                return result
            except Exception as e:
                self.logger.error(f"Error stopping component {component_id}: {str(e)}")
                return False
    
    def shutdown_component(self, component_id: str) -> bool:
        """
        Shut down a specific component.
        
        Args:
            component_id: ID of the component to shut down
            
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        with self._lock:
            component = self.get_component(component_id)
            if not component:
                self.logger.warning(f"Cannot shut down unknown component {component_id}")
                return False
                
            try:
                result = component.shutdown()
                if result:
                    self.logger.info(f"Component {component_id} shut down successfully")
                else:
                    self.logger.error(f"Component {component_id} shutdown failed")
                return result
            except Exception as e:
                self.logger.error(f"Error shutting down component {component_id}: {str(e)}")
                return False
    
    def initialize_all_components(self) -> Dict[str, bool]:
        """
        Initialize all registered components.
        
        Returns:
            Dict[str, bool]: Dictionary of component ID to initialization result
        """
        with self._lock:
            results = {}
            for component_id in self._components:
                results[component_id] = self.initialize_component(component_id)
            return results
    
    def start_all_components(self) -> Dict[str, bool]:
        """
        Start all registered components.
        
        Returns:
            Dict[str, bool]: Dictionary of component ID to start result
        """
        with self._lock:
            results = {}
            for component_id in self._components:
                results[component_id] = self.start_component(component_id)
            return results
    
    def stop_all_components(self) -> Dict[str, bool]:
        """
        Stop all registered components.
        
        Returns:
            Dict[str, bool]: Dictionary of component ID to stop result
        """
        with self._lock:
            results = {}
            for component_id in self._components:
                results[component_id] = self.stop_component(component_id)
            return results
    
    def shutdown_all_components(self) -> Dict[str, bool]:
        """
        Shut down all registered components.
        
        Returns:
            Dict[str, bool]: Dictionary of component ID to shutdown result
        """
        with self._lock:
            results = {}
            for component_id in self._components:
                results[component_id] = self.shutdown_component(component_id)
            return results
    
    def get_component_status(self, component_id: str) -> Optional[Dict]:
        """
        Get the status of a specific component.
        
        Args:
            component_id: ID of the component to query
            
        Returns:
            Dict: Component status, or None if component not found
        """
        with self._lock:
            component = self.get_component(component_id)
            if not component:
                return None
                
            try:
                return component.get_status()
            except Exception as e:
                self.logger.error(f"Error getting status for component {component_id}: {str(e)}")
                return {
                    "component_id": component_id,
                    "error": str(e),
                    "is_initialized": component.is_initialized,
                    "is_running": component.is_running
                }
    
    def get_all_component_statuses(self) -> Dict[str, Dict]:
        """
        Get the status of all registered components.
        
        Returns:
            Dict[str, Dict]: Dictionary of component ID to status
        """
        with self._lock:
            statuses = {}
            for component_id in self._components:
                statuses[component_id] = self.get_component_status(component_id)
            return statuses
