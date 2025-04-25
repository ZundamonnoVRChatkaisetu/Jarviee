"""
Adapter Registry for AI Technology Integration.

This module implements a specialized registry for managing technology adapters
in the Jarviee system. It extends the functionality of the ComponentRegistry
to provide adapter-specific operations and management capabilities.
"""

import threading
from typing import Dict, List, Optional, Type

from ...utils.logger import Logger
from ..base import ComponentType
from ..registry import ComponentRegistry
from .base import TechnologyAdapter


class AdapterRegistry:
    """
    Registry for managing AI technology adapters.
    
    This class provides a specialized registry for AI technology adapters,
    offering adapter-specific operations and management capabilities beyond
    the general component registry.
    """
    
    _instance = None  # Singleton instance
    _lock = threading.RLock()  # Thread-safe lock
    
    def __new__(cls):
        """Ensure AdapterRegistry is a singleton."""
        if cls._instance is None:
            cls._instance = super(AdapterRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the adapter registry if not already initialized."""
        if self._initialized:
            return
            
        # Initialize registry data structures
        self._adapters_by_type: Dict[ComponentType, List[str]] = {
            ctype: [] for ctype in ComponentType
        }
        self._adapter_factories: Dict[str, Type[TechnologyAdapter]] = {}
        
        # Initialize dependencies
        self.component_registry = ComponentRegistry()
        self.logger = Logger().get_logger("jarviee.core.integration.adapters.registry")
        
        self._initialized = True
        self.logger.info("Adapter Registry initialized")
    
    def register_adapter(self, adapter: TechnologyAdapter) -> bool:
        """
        Register an AI technology adapter.
        
        Args:
            adapter: The adapter to register
            
        Returns:
            bool: True if registration was successful, False otherwise
        """
        with self._lock:
            # Register with component registry
            if not self.component_registry.register_component(adapter):
                return False
                
            # Add to adapter-specific tracking
            adapter_id = adapter.component_id
            adapter_type = adapter.component_type
            
            if adapter_id not in self._adapters_by_type[adapter_type]:
                self._adapters_by_type[adapter_type].append(adapter_id)
                
            self.logger.info(f"Registered adapter {adapter_id} for {adapter_type.name}")
            return True
    
    def unregister_adapter(self, adapter_id: str) -> bool:
        """
        Unregister an AI technology adapter.
        
        Args:
            adapter_id: ID of the adapter to unregister
            
        Returns:
            bool: True if unregistration was successful, False otherwise
        """
        with self._lock:
            # Get adapter from component registry
            adapter = self.component_registry.get_component(adapter_id)
            if not adapter or not isinstance(adapter, TechnologyAdapter):
                self.logger.warning(f"Cannot unregister unknown adapter {adapter_id}")
                return False
                
            # Remove from adapter-specific tracking
            adapter_type = adapter.component_type
            if adapter_id in self._adapters_by_type[adapter_type]:
                self._adapters_by_type[adapter_type].remove(adapter_id)
                
            # Unregister from component registry
            return self.component_registry.unregister_component(adapter_id)
    
    def get_adapter(self, adapter_id: str) -> Optional[TechnologyAdapter]:
        """
        Get a registered adapter by ID.
        
        Args:
            adapter_id: ID of the adapter to retrieve
            
        Returns:
            TechnologyAdapter: The adapter, or None if not found
        """
        component = self.component_registry.get_component(adapter_id)
        if component and isinstance(component, TechnologyAdapter):
            return component
        return None
    
    def get_adapters_by_type(self, adapter_type: ComponentType) -> List[TechnologyAdapter]:
        """
        Get all registered adapters of a specific type.
        
        Args:
            adapter_type: Type of adapters to retrieve
            
        Returns:
            List[TechnologyAdapter]: List of adapters of the specified type
        """
        with self._lock:
            adapter_ids = self._adapters_by_type.get(adapter_type, [])
            return [self.get_adapter(aid) for aid in adapter_ids if self.get_adapter(aid)]
    
    def register_adapter_factory(self, adapter_type: str,
                               factory_class: Type[TechnologyAdapter]) -> None:
        """
        Register a factory class for creating adapters of a specific type.
        
        Args:
            adapter_type: Type identifier for the adapter
            factory_class: Adapter class to use as factory
        """
        with self._lock:
            self._adapter_factories[adapter_type] = factory_class
            self.logger.info(f"Registered adapter factory for {adapter_type}")
    
    def create_adapter(self, adapter_type: str, adapter_id: str,
                     llm_component_id: str = "llm_core", **kwargs) -> Optional[TechnologyAdapter]:
        """
        Create and register an adapter using a registered factory.
        
        Args:
            adapter_type: Type identifier for the adapter
            adapter_id: ID to assign to the new adapter
            llm_component_id: ID of the LLM core to connect with
            **kwargs: Additional arguments to pass to the adapter constructor
            
        Returns:
            TechnologyAdapter: The created adapter, or None if factory not found
        """
        with self._lock:
            if adapter_type not in self._adapter_factories:
                self.logger.warning(f"No adapter factory registered for {adapter_type}")
                return None
                
            factory_class = self._adapter_factories[adapter_type]
            
            try:
                # Create adapter instance
                adapter = factory_class(adapter_id, llm_component_id=llm_component_id, **kwargs)
                
                # Register the adapter
                if self.register_adapter(adapter):
                    return adapter
                else:
                    return None
            except Exception as e:
                self.logger.error(f"Error creating adapter {adapter_type}/{adapter_id}: {str(e)}")
                return None
    
    def initialize_adapter(self, adapter_id: str) -> bool:
        """
        Initialize a specific adapter.
        
        Args:
            adapter_id: ID of the adapter to initialize
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        return self.component_registry.initialize_component(adapter_id)
    
    def start_adapter(self, adapter_id: str) -> bool:
        """
        Start a specific adapter.
        
        Args:
            adapter_id: ID of the adapter to start
            
        Returns:
            bool: True if start was successful, False otherwise
        """
        return self.component_registry.start_component(adapter_id)
    
    def stop_adapter(self, adapter_id: str) -> bool:
        """
        Stop a specific adapter.
        
        Args:
            adapter_id: ID of the adapter to stop
            
        Returns:
            bool: True if stop was successful, False otherwise
        """
        return self.component_registry.stop_component(adapter_id)
    
    def shutdown_adapter(self, adapter_id: str) -> bool:
        """
        Shut down a specific adapter.
        
        Args:
            adapter_id: ID of the adapter to shut down
            
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        return self.component_registry.shutdown_component(adapter_id)
    
    def get_available_adapter_types(self) -> List[str]:
        """
        Get a list of available adapter types that can be created.
        
        Returns:
            List[str]: List of adapter type identifiers
        """
        with self._lock:
            return list(self._adapter_factories.keys())
    
    def get_adapter_status(self, adapter_id: str) -> Optional[Dict]:
        """
        Get the status of a specific adapter.
        
        Args:
            adapter_id: ID of the adapter to query
            
        Returns:
            Dict: Adapter status, or None if adapter not found
        """
        return self.component_registry.get_component_status(adapter_id)
