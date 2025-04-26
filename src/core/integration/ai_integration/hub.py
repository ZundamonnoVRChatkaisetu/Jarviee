"""
AI Technology Integration Hub for Jarviee System.

This module implements the comprehensive AI Technology Integration Hub that
facilitates seamless cooperation between LLM and other AI technologies (Reinforcement
Learning, Symbolic AI, Multimodal AI, Agent-based AI, and Neuromorphic AI).

It serves as a unified integration layer that enables these technologies to work
together effectively, enhancing the system's capabilities beyond what any single
technology can achieve.
"""

import asyncio
import json
import logging
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..adapters.agent.adapter import AgentAdapter
from ..adapters.multimodal.adapter import MultimodalAdapter
from ..adapters.neuromorphic.adapter import NeuromorphicAdapter
from ..adapters.reinforcement_learning.adapter import RLAdapter
from ..adapters.symbolic_ai.adapter import SymbolicAIAdapter
from ..base import AIComponent, ComponentType, IntegrationMessage
from ..coordinator.integration_hub import IntegrationHub
from ..coordinator.resource_manager import ResourceManager
from ..framework import (AITechnologyIntegration, IntegrationCapabilityTag,
                      IntegrationFramework, IntegrationMethod, IntegrationPipeline,
                      IntegrationPriority, TechnologyIntegrationType)
from ..registry import ComponentRegistry
from ...llm.engine import LLMEngine
from ...utils.event_bus import Event, EventBus
from ...utils.logger import Logger


class AITechnologyIntegrationHub:
    """
    Central hub for integrating multiple AI technologies with the LLM core.
    
    This class coordinates the interactions between different AI technologies,
    manages the creation and execution of integration pipelines, and provides
    a unified interface for using these integrated capabilities.
    """
    
    def __init__(
        self, 
        hub_id: str,
        event_bus: EventBus,
        llm_component_id: str = "llm_core",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the AI Technology Integration Hub.
        
        Args:
            hub_id: Unique identifier for this hub
            event_bus: Event bus for communication
            llm_component_id: ID of the LLM component
            config: Optional configuration settings
        """
        self.hub_id = hub_id
        self.event_bus = event_bus
        self.llm_component_id = llm_component_id
        
        self.logger = Logger().get_logger(f"jarviee.integration.hub.{hub_id}")
        
        # Initialize component registry and resource manager
        self.registry = ComponentRegistry()
        self.resource_manager = ResourceManager()
        
        # Initialize the integration framework
        self.framework = IntegrationFramework()
        
        # Technology adapters
        self.adapters: Dict[ComponentType, AIComponent] = {}
        
        # Integration instances (technology-specific)
        self.integrations: Dict[str, AITechnologyIntegration] = {}
        
        # Integration pipelines
        self.pipelines: Dict[str, IntegrationPipeline] = {}
        
        # Task contexts
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Capability registry - maps capabilities to integration IDs
        self.capability_registry: Dict[IntegrationCapabilityTag, List[str]] = {
            tag: [] for tag in IntegrationCapabilityTag
        }
        
        # Default configuration
        self.config = {
            "default_pipeline_method": "adaptive",
            "auto_create_pipelines": True,
            "integration_timeout_seconds": 30,
            "max_active_tasks": 100,
            "enable_advanced_capabilities": True,
            "resource_optimization": True,
            "capability_based_routing": True,
            "persist_integration_state": True,
            "log_integration_details": True,
            "error_recovery_attempts": 3
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Register for events
        self._register_event_handlers()
        
        self.logger.info(f"AI Technology Integration Hub {hub_id} initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the hub and all required components.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Initialize resource manager
            self.resource_manager.initialize()
            
            # Discover and register available technology adapters
            self._discover_technology_adapters()
            
            # Initialize integration framework
            # (Already done in constructor)
            
            # Set up default integration pipelines
            if self.config["auto_create_pipelines"]:
                self._setup_default_pipelines()
            
            self.logger.info(f"AI Technology Integration Hub {self.hub_id} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing AI Technology Integration Hub: {str(e)}")
            return False
    
    def shutdown(self) -> bool:
        """
        Shutdown the hub and all components.
        
        Returns:
            bool: True if shutdown was successful
        """
        try:
            # Shutdown all integrations
            for integration_id in list(self.integrations.keys()):
                self.deactivate_integration(integration_id)
            
            # Shutdown framework
            self.framework.shutdown()
            
            # Shutdown resource manager
            self.resource_manager.shutdown()
            
            self.logger.info(f"AI Technology Integration Hub {self.hub_id} shut down successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error shutting down AI Technology Integration Hub: {str(e)}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the hub.
        
        Returns:
            Dictionary containing status information
        """
        # Get framework status
        framework_status = self.framework.get_framework_status()
        
        # Get adapter statuses
        adapter_statuses = {
            component_type.name: adapter.get_status()
            for component_type, adapter in self.adapters.items()
        }
        
        # Capability statistics
        capability_stats = {
            tag.name: len(ids) for tag, ids in self.capability_registry.items()
        }
        
        # Task statistics
        task_stats = {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": sum(1 for t in self.active_tasks.values() if t.get("status") == "completed"),
            "failed_tasks": sum(1 for t in self.active_tasks.values() if t.get("status") == "failed"),
            "in_progress_tasks": sum(1 for t in self.active_tasks.values() if t.get("status") == "in_progress")
        }
        
        # Hub status
        hub_status = {
            "hub_id": self.hub_id,
            "llm_component_id": self.llm_component_id,
            "adapters": adapter_statuses,
            "framework": framework_status,
            "capability_stats": capability_stats,
            "task_stats": task_stats,
            "config": self.config,
            "timestamp": time.time()
        }
        
        return hub_status
    
    def register_adapter(self, adapter: AIComponent) -> bool:
        """
        Register a technology adapter with the hub.
        
        Args:
            adapter: The adapter to register
            
        Returns:
            bool: True if registration was successful
        """
        component_type = adapter.component_type
        
        if component_type in self.adapters:
            self.logger.warning(f"Adapter for {component_type.name} already registered")
            return False
        
        self.adapters[component_type] = adapter
        self.logger.info(f"Registered {component_type.name} adapter: {adapter.component_id}")
        
        # Create integrations for this adapter
        self._create_integrations_for_adapter(adapter)
        
        return True
    
    def unregister_adapter(self, component_type: ComponentType) -> bool:
        """
        Unregister a technology adapter.
        
        Args:
            component_type: The type of adapter to unregister
            
        Returns:
            bool: True if unregistration was successful
        """
        if component_type not in self.adapters:
            self.logger.warning(f"No adapter registered for {component_type.name}")
            return False
        
        # Remove any integrations using this adapter
        adapter = self.adapters[component_type]
        integration_ids_to_remove = []
        
        for integration_id, integration in self.integrations.items():
            if integration.technology_component_id == adapter.component_id:
                integration_ids_to_remove.append(integration_id)
        
        for integration_id in integration_ids_to_remove:
            self.deactivate_integration(integration_id)
            self.unregister_integration(integration_id)
        
        # Remove adapter
        del self.adapters[component_type]
        self.logger.info(f"Unregistered {component_type.name} adapter")
        
        return True
    
    def register_integration(self, integration: AITechnologyIntegration) -> bool:
        """
        Register an integration with the hub.
        
        Args:
            integration: The integration to register
            
        Returns:
            bool: True if registration was successful
        """
        integration_id = integration.integration_id
        
        if integration_id in self.integrations:
            self.logger.warning(f"Integration {integration_id} already registered")
            return False
        
        # Register with framework
        self.framework.register_integration(integration)
        
        # Store locally
        self.integrations[integration_id] = integration
        
        # Update capability registry
        for capability in integration.capabilities:
            self.capability_registry[capability].append(integration_id)
        
        self.logger.info(f"Registered integration: {integration_id}")
        return True
    
    def unregister_integration(self, integration_id: str) -> bool:
        """
        Unregister an integration from the hub.
        
        Args:
            integration_id: ID of the integration to unregister
            
        Returns:
            bool: True if unregistration was successful
        """
        if integration_id not in self.integrations:
            self.logger.warning(f"No integration registered with ID {integration_id}")
            return False
        
        # Get the integration
        integration = self.integrations[integration_id]
        
        # Remove from capability registry
        for capability in integration.capabilities:
            if integration_id in self.capability_registry[capability]:
                self.capability_registry[capability].remove(integration_id)
        
        # Remove from framework
        self.framework.unregister_integration(integration_id)
        
        # Remove locally
        del self.integrations[integration_id]
        
        self.logger.info(f"Unregistered integration: {integration_id}")
        return True
    
    def activate_integration(self, integration_id: str) -> bool:
        """
        Activate an integration.
        
        Args:
            integration_id: ID of the integration to activate
            
        Returns:
            bool: True if activation was successful
        """
        if integration_id not in self.integrations:
            self.logger.warning(f"No integration registered with ID {integration_id}")
            return False
        
        # Activate via framework
        return self.framework.activate_integration(integration_id)
    
    def deactivate_integration(self, integration_id: str) -> bool:
        """
        Deactivate an integration.
        
        Args:
            integration_id: ID of the integration to deactivate
            
        Returns:
            bool: True if deactivation was successful
        """
        if integration_id not in self.integrations:
            self.logger.warning(f"No integration registered with ID {integration_id}")
            return False
        
        # Deactivate via framework
        return self.framework.deactivate_integration(integration_id)
    
    def create_pipeline(
        self, 
        pipeline_id: str,
        integration_ids: List[str],
        method: Union[str, IntegrationMethod] = None
    ) -> Optional[str]:
        """
        Create a new integration pipeline.
        
        Args:
            pipeline_id: ID for the new pipeline
            integration_ids: IDs of integrations to include
            method: Processing method (sequential, parallel, hybrid, adaptive)
            
        Returns:
            ID of the created pipeline, or None if creation failed
        """
        if not integration_ids:
            self.logger.error("Cannot create pipeline with no integrations")
            return None
        
        # Check if integrations exist
        for integration_id in integration_ids:
            if integration_id not in self.integrations:
                self.logger.error(f"Integration {integration_id} not found")
                return None
        
        # Convert method string to enum if needed
        if isinstance(method, str):
            try:
                method = IntegrationMethod[method.upper()]
            except KeyError:
                self.logger.warning(f"Invalid method: {method}, using default")
                method = None
        
        # Use default method if none specified
        if method is None:
            default_method = self.config["default_pipeline_method"]
            if isinstance(default_method, str):
                try:
                    method = IntegrationMethod[default_method.upper()]
                except KeyError:
                    method = IntegrationMethod.ADAPTIVE
            else:
                method = IntegrationMethod.ADAPTIVE
        
        try:
            # Create pipeline via framework
            pipeline = self.framework.create_pipeline(
                pipeline_id, integration_ids, method)
            
            # Store locally
            self.pipelines[pipeline_id] = pipeline
            
            self.logger.info(f"Created pipeline: {pipeline_id} with {len(integration_ids)} integrations")
            return pipeline_id
            
        except Exception as e:
            self.logger.error(f"Error creating pipeline: {str(e)}")
            return None
    
    def delete_pipeline(self, pipeline_id: str) -> bool:
        """
        Delete an integration pipeline.
        
        Args:
            pipeline_id: ID of the pipeline to delete
            
        Returns:
            bool: True if deletion was successful
        """
        if pipeline_id not in self.pipelines:
            self.logger.warning(f"No pipeline registered with ID {pipeline_id}")
            return False
        
        # Remove from framework
        success = self.framework.unregister_pipeline(pipeline_id)
        
        if success:
            # Remove locally
            del self.pipelines[pipeline_id]
            self.logger.info(f"Deleted pipeline: {pipeline_id}")
        
        return success
    
    def find_integrations_with_capability(self, capability: Union[str, IntegrationCapabilityTag]) -> List[str]:
        """
        Find integrations that have a specific capability.
        
        Args:
            capability: The capability to look for
            
        Returns:
            List of integration IDs with the specified capability
        """
        # Convert string to enum if needed
        if isinstance(capability, str):
            try:
                capability = IntegrationCapabilityTag[capability.upper()]
            except KeyError:
                self.logger.error(f"Invalid capability: {capability}")
                return []
        
        # Get integrations with this capability
        return self.capability_registry.get(capability, [])
    
    def find_best_integration_for_task(
        self, 
        capabilities: List[Union[str, IntegrationCapabilityTag]],
        task_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Find the best integration for a task based on required capabilities.
        
        Args:
            capabilities: List of capabilities required for the task
            task_context: Optional context about the task
            
        Returns:
            ID of the best integration, or None if no suitable integration is found
        """
        if not capabilities:
            self.logger.error("Cannot find integration without specified capabilities")
            return None
        
        # Convert string capabilities to enums
        capability_enums = []
        for cap in capabilities:
            if isinstance(cap, str):
                try:
                    cap_enum = IntegrationCapabilityTag[cap.upper()]
                    capability_enums.append(cap_enum)
                except KeyError:
                    self.logger.warning(f"Invalid capability: {cap}, ignoring")
            else:
                capability_enums.append(cap)
        
        if not capability_enums:
            return None
        
        # Find integrations with all required capabilities
        candidate_ids = set(self.integrations.keys())
        for capability in capability_enums:
            integration_ids = set(self.capability_registry.get(capability, []))
            candidate_ids = candidate_ids.intersection(integration_ids)
        
        if not candidate_ids:
            self.logger.warning(f"No integration found with all required capabilities: {capabilities}")
            return None
        
        # Filter for active integrations
        active_candidates = [
            integration_id for integration_id in candidate_ids
            if self.integrations[integration_id].active
        ]
        
        if not active_candidates:
            self.logger.warning("No active integration with required capabilities")
            return None
        
        # Select best integration based on priority and other factors
        # For now, just use the highest priority one
        selected_id = max(
            active_candidates,
            key=lambda i: self.integrations[i].priority.value
        )
        
        return selected_id
    
    def create_task_pipeline(
        self, 
        capabilities: List[Union[str, IntegrationCapabilityTag]],
        task_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a pipeline specifically for a task based on required capabilities.
        
        Args:
            capabilities: List of capabilities required for the task
            task_context: Optional context about the task
            
        Returns:
            ID of the created pipeline, or None if creation failed
        """
        if not capabilities:
            self.logger.error("Cannot create task pipeline without specified capabilities")
            return None
        
        # Use framework to create a task-specific pipeline
        task_type = task_context.get("task_type", "unknown") if task_context else "unknown"
        task_content = task_context.get("content", {}) if task_context else {}
        
        # Convert capabilities to the expected format
        capability_enums = []
        for cap in capabilities:
            if isinstance(cap, str):
                try:
                    cap_enum = IntegrationCapabilityTag[cap.upper()]
                    capability_enums.append(cap_enum)
                except KeyError:
                    self.logger.warning(f"Invalid capability: {cap}, ignoring")
            else:
                capability_enums.append(cap)
        
        pipeline_id = self.framework.create_task_pipeline(
            task_type, task_content, task_context, capability_enums)
        
        if pipeline_id:
            # Store the pipeline reference
            pipeline = self.framework.get_pipeline(pipeline_id)
            if pipeline:
                self.pipelines[pipeline_id] = pipeline
                
                self.logger.info(f"Created task pipeline: {pipeline_id} for task type: {task_type}")
                return pipeline_id
        
        self.logger.error(f"Failed to create task pipeline for capabilities: {capabilities}")
        return None
    
    def execute_task(
        self, 
        task_type: str,
        task_content: Dict[str, Any],
        capabilities: Optional[List[Union[str, IntegrationCapabilityTag]]] = None,
        integration_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a task using appropriate integration(s).
        
        This method selects the best way to execute the task based on the provided
        parameters:
        - If integration_id is provided, it uses that specific integration
        - If pipeline_id is provided, it uses that specific pipeline
        - If capabilities are provided, it finds or creates a suitable pipeline
        - Otherwise, it tries to determine the appropriate integration based on the task
        
        Args:
            task_type: Type of task to execute
            task_content: Task data
            capabilities: Optional list of required capabilities
            integration_id: Optional specific integration to use
            pipeline_id: Optional specific pipeline to use
            context: Optional execution context
            
        Returns:
            Dictionary containing the task result
        """
        # Create task ID and context
        task_id = str(uuid.uuid4())
        task_context = context or {}
        task_context.update({
            "task_id": task_id,
            "task_type": task_type,
            "start_time": time.time()
        })
        
        # Log task start
        self.logger.debug(f"Executing task {task_id} of type {task_type}")
        
        # Store task in active tasks
        self.active_tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "content": task_content,
            "status": "created",
            "timestamp": time.time(),
            "context": task_context
        }
        
        try:
            # Determine execution method
            result = None
            
            if integration_id:
                # Use specific integration
                if integration_id not in self.integrations:
                    raise ValueError(f"Integration {integration_id} not found")
                    
                self.active_tasks[task_id]["status"] = "in_progress"
                self.active_tasks[task_id]["integration_id"] = integration_id
                
                result = self.framework.process_task(
                    integration_id, task_type, task_content, task_context)
                
            elif pipeline_id:
                # Use specific pipeline
                if pipeline_id not in self.pipelines:
                    raise ValueError(f"Pipeline {pipeline_id} not found")
                    
                self.active_tasks[task_id]["status"] = "in_progress"
                self.active_tasks[task_id]["pipeline_id"] = pipeline_id
                
                result = self.framework.process_task_with_pipeline(
                    pipeline_id, task_type, task_content, task_context)
                
            elif capabilities:
                # Try to find a suitable integration or pipeline
                if self.config["capability_based_routing"]:
                    # Create a task-specific pipeline
                    created_pipeline_id = self.create_task_pipeline(
                        capabilities, 
                        {"task_type": task_type, "content": task_content, **task_context}
                    )
                    
                    if created_pipeline_id:
                        self.active_tasks[task_id]["status"] = "in_progress"
                        self.active_tasks[task_id]["pipeline_id"] = created_pipeline_id
                        
                        result = self.framework.process_task_with_pipeline(
                            created_pipeline_id, task_type, task_content, task_context)
                    else:
                        # Try to find a single integration with the required capabilities
                        found_integration_id = self.find_best_integration_for_task(
                            capabilities, task_context)
                            
                        if found_integration_id:
                            self.active_tasks[task_id]["status"] = "in_progress"
                            self.active_tasks[task_id]["integration_id"] = found_integration_id
                            
                            result = self.framework.process_task(
                                found_integration_id, task_type, task_content, task_context)
                        else:
                            raise ValueError(f"No suitable integration found for capabilities: {capabilities}")
            else:
                # Let the framework decide
                selected_integration_id = self.framework.select_integration_for_task(
                    task_type, task_content, task_context)
                    
                if selected_integration_id:
                    self.active_tasks[task_id]["status"] = "in_progress"
                    self.active_tasks[task_id]["integration_id"] = selected_integration_id
                    
                    result = self.framework.process_task(
                        selected_integration_id, task_type, task_content, task_context)
                else:
                    raise ValueError(f"No suitable integration found for task type: {task_type}")
            
            # Update task status
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["completed_at"] = time.time()
            self.active_tasks[task_id]["result"] = result
            
            # Add task ID to result
            if result is None:
                result = {}
            result["task_id"] = task_id
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing task {task_id}: {str(e)}")
            
            # Update task status
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["error"] = str(e)
            
            # Return error result
            return {
                "task_id": task_id,
                "error": str(e),
                "status": "failed"
            }
    
    async def execute_task_async(
        self, 
        task_type: str,
        task_content: Dict[str, Any],
        capabilities: Optional[List[Union[str, IntegrationCapabilityTag]]] = None,
        integration_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a task asynchronously using appropriate integration(s).
        
        Args:
            task_type: Type of task to execute
            task_content: Task data
            capabilities: Optional list of required capabilities
            integration_id: Optional specific integration to use
            pipeline_id: Optional specific pipeline to use
            context: Optional execution context
            
        Returns:
            Dictionary containing the task result
        """
        # Create task ID and context
        task_id = str(uuid.uuid4())
        task_context = context or {}
        task_context.update({
            "task_id": task_id,
            "task_type": task_type,
            "start_time": time.time()
        })
        
        # Log task start
        self.logger.debug(f"Executing task {task_id} of type {task_type} asynchronously")
        
        # Store task in active tasks
        self.active_tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "content": task_content,
            "status": "created",
            "timestamp": time.time(),
            "context": task_context,
            "async": True
        }
        
        try:
            # Determine execution method
            result = None
            
            if integration_id:
                # Use specific integration
                if integration_id not in self.integrations:
                    raise ValueError(f"Integration {integration_id} not found")
                    
                self.active_tasks[task_id]["status"] = "in_progress"
                self.active_tasks[task_id]["integration_id"] = integration_id
                
                result = await self.framework.process_task_async(
                    integration_id, task_type, task_content, task_context)
                
            elif pipeline_id:
                # Use specific pipeline
                if pipeline_id not in self.pipelines:
                    raise ValueError(f"Pipeline {pipeline_id} not found")
                    
                self.active_tasks[task_id]["status"] = "in_progress"
                self.active_tasks[task_id]["pipeline_id"] = pipeline_id
                
                result = await self.framework.process_task_with_pipeline_async(
                    pipeline_id, task_type, task_content, task_context)
                
            elif capabilities:
                # Try to find a suitable integration or pipeline
                if self.config["capability_based_routing"]:
                    # Create a task-specific pipeline
                    created_pipeline_id = self.create_task_pipeline(
                        capabilities, 
                        {"task_type": task_type, "content": task_content, **task_context}
                    )
                    
                    if created_pipeline_id:
                        self.active_tasks[task_id]["status"] = "in_progress"
                        self.active_tasks[task_id]["pipeline_id"] = created_pipeline_id
                        
                        result = await self.framework.process_task_with_pipeline_async(
                            created_pipeline_id, task_type, task_content, task_context)
                    else:
                        # Try to find a single integration with the required capabilities
                        found_integration_id = self.find_best_integration_for_task(
                            capabilities, task_context)
                            
                        if found_integration_id:
                            self.active_tasks[task_id]["status"] = "in_progress"
                            self.active_tasks[task_id]["integration_id"] = found_integration_id
                            
                            result = await self.framework.process_task_async(
                                found_integration_id, task_type, task_content, task_context)
                        else:
                            raise ValueError(f"No suitable integration found for capabilities: {capabilities}")
            else:
                # Let the framework decide
                selected_integration_id = self.framework.select_integration_for_task(
                    task_type, task_content, task_context)
                    
                if selected_integration_id:
                    self.active_tasks[task_id]["status"] = "in_progress"
                    self.active_tasks[task_id]["integration_id"] = selected_integration_id
                    
                    result = await self.framework.process_task_async(
                        selected_integration_id, task_type, task_content, task_context)
                else:
                    raise ValueError(f"No suitable integration found for task type: {task_type}")
            
            # Update task status
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["completed_at"] = time.time()
            self.active_tasks[task_id]["result"] = result
            
            # Add task ID to result
            if result is None:
                result = {}
            result["task_id"] = task_id
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing async task {task_id}: {str(e)}")
            
            # Update task status
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["error"] = str(e)
            
            # Return error result
            return {
                "task_id": task_id,
                "error": str(e),
                "status": "failed"
            }
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dictionary containing task status information
        """
        if task_id not in self.active_tasks:
            return {"error": "Task not found", "task_id": task_id}
        
        task = self.active_tasks[task_id]
        
        # Create a copy to avoid modifying the original
        status = task.copy()
        
        # Remove potentially large content from the response
        if "content" in status and len(json.dumps(status["content"])) > 1000:
            status["content"] = {"truncated": True, "size": len(json.dumps(task["content"]))}
        
        if "result" in status and len(json.dumps(status["result"])) > 1000:
            status["result"] = {"truncated": True, "size": len(json.dumps(task["result"]))}
        
        return status
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            bool: True if the task was cancelled successfully
        """
        if task_id not in self.active_tasks:
            self.logger.warning(f"Task {task_id} not found for cancellation")
            return False
        
        task = self.active_tasks[task_id]
        
        # Check if task can be cancelled
        if task["status"] in ["completed", "failed"]:
            self.logger.warning(f"Task {task_id} already {task['status']}, cannot cancel")
            return False
        
        # Update status
        task["status"] = "cancelled"
        task["cancelled_at"] = time.time()
        
        self.logger.info(f"Task {task_id} cancelled")
        return True
    
    # Event handlers
    
    def _register_event_handlers(self):
        """Register event handlers for integration events."""
        # Component events
        self.event_bus.subscribe("integration.component.registered", self._handle_component_registered)
        self.event_bus.subscribe("integration.component.unregistered", self._handle_component_unregistered)
        
        # Integration events
        self.event_bus.subscribe("integration.status_update", self._handle_integration_status)
        self.event_bus.subscribe("integration.capability_update", self._handle_capability_update)
        
        # Task events
        self.event_bus.subscribe("integration.task.created", self._handle_task_created)
        self.event_bus.subscribe("integration.task.completed", self._handle_task_completed)
        self.event_bus.subscribe("integration.task.failed", self._handle_task_failed)
        
        # Error events
        self.event_bus.subscribe("integration.error", self._handle_integration_error)
    
    def _handle_component_registered(self, event: Event):
        """
        Handle component registration event.
        
        Args:
            event: The event data
        """
        if "component" not in event.data:
            return
            
        component = event.data["component"]
        
        if not isinstance(component, AIComponent):
            return
            
        # Check if this is a technology adapter
        if component.component_type in [
            ComponentType.REINFORCEMENT_LEARNING,
            ComponentType.SYMBOLIC_AI,
            ComponentType.MULTIMODAL,
            ComponentType.AGENT,
            ComponentType.NEUROMORPHIC
        ]:
            # Register adapter if not already registered
            if component.component_type not in self.adapters:
                self.register_adapter(component)
    
    def _handle_component_unregistered(self, event: Event):
        """
        Handle component unregistration event.
        
        Args:
            event: The event data
        """
        if "component_id" not in event.data or "component_type" not in event.data:
            return
            
        component_id = event.data["component_id"]
        component_type = event.data["component_type"]
        
        # Check if this is a registered adapter
        if component_type in self.adapters and self.adapters[component_type].component_id == component_id:
            self.unregister_adapter(component_type)
    
    def _handle_integration_status(self, event: Event):
        """
        Handle integration status update event.
        
        Args:
            event: The event data
        """
        if "integration_id" not in event.data or "status" not in event.data:
            return
            
        integration_id = event.data["integration_id"]
        status = event.data["status"]
        
        if integration_id in self.integrations:
            # Update local status tracking if needed
            self.logger.debug(f"Integration {integration_id} status updated: {status}")
            
            # Framework will handle the actual status update
    
    def _handle_capability_update(self, event: Event):
        """
        Handle capability update event.
        
        Args:
            event: The event data
        """
        if "integration_id" not in event.data or "capabilities" not in event.data:
            return
            
        integration_id = event.data["integration_id"]
        capabilities = event.data["capabilities"]
        
        if integration_id not in self.integrations:
            return
            
        integration = self.integrations[integration_id]
        
        # Remove from old capability registrations
        for capability, registered_ids in self.capability_registry.items():
            if integration_id in registered_ids:
                registered_ids.remove(integration_id)
        
        # Add to new capability registrations
        for capability in capabilities:
            if isinstance(capability, str):
                try:
                    capability = IntegrationCapabilityTag[capability.upper()]
                except KeyError:
                    continue
                    
            if capability in self.capability_registry:
                self.capability_registry[capability].append(integration_id)
        
        self.logger.debug(f"Updated capabilities for integration {integration_id}")
    
    def _handle_task_created(self, event: Event):
        """
        Handle task creation event.
        
        Args:
            event: The event data
        """
        # This is mostly handled at task execution time, but we might
        # want to track externally created tasks here
        if "task_id" not in event.data:
            return
            
        task_id = event.data["task_id"]
        
        if task_id not in self.active_tasks:
            self.active_tasks[task_id] = {
                "id": task_id,
                "status": "created",
                "timestamp": time.time(),
                "external": True
            }
            
            # Add other fields if available
            for field in ["type", "content", "context"]:
                if field in event.data:
                    self.active_tasks[task_id][field] = event.data[field]
    
    def _handle_task_completed(self, event: Event):
        """
        Handle task completion event.
        
        Args:
            event: The event data
        """
        if "task_id" not in event.data:
            return
            
        task_id = event.data["task_id"]
        
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["completed_at"] = time.time()
            
            # Add result if available
            if "result" in event.data:
                self.active_tasks[task_id]["result"] = event.data["result"]
    
    def _handle_task_failed(self, event: Event):
        """
        Handle task failure event.
        
        Args:
            event: The event data
        """
        if "task_id" not in event.data:
            return
            
        task_id = event.data["task_id"]
        
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["failed_at"] = time.time()
            
            # Add error information if available
            if "error" in event.data:
                self.active_tasks[task_id]["error"] = event.data["error"]
    
    def _handle_integration_error(self, event: Event):
        """
        Handle integration error event.
        
        Args:
            event: The event data
        """
        if "integration_id" not in event.data or "error" not in event.data:
            return
            
        integration_id = event.data["integration_id"]
        error = event.data["error"]
        
        self.logger.error(f"Integration {integration_id} reported error: {error}")
        
        # Check if we need to deactivate the integration
        if event.data.get("critical", False):
            self.logger.warning(f"Critical error in integration {integration_id}, deactivating")
            self.deactivate_integration(integration_id)
    
    # Internal helper methods
    
    def _discover_technology_adapters(self) -> None:
        """Discover and register available technology adapters."""
        # Query registry for available adapters
        components = self.registry.list_components()
        
        for component in components:
            if component.component_type in [
                ComponentType.REINFORCEMENT_LEARNING,
                ComponentType.SYMBOLIC_AI,
                ComponentType.MULTIMODAL,
                ComponentType.AGENT,
                ComponentType.NEUROMORPHIC
            ]:
                self.register_adapter(component)
    
    def _create_integrations_for_adapter(self, adapter: AIComponent) -> None:
        """
        Create integration instances for a technology adapter.
        
        Args:
            adapter: The technology adapter
        """
        # Get LLM component
        llm_component = self.registry.get_component(self.llm_component_id)
        if not llm_component:
            self.logger.error(f"LLM component {self.llm_component_id} not found")
            return
        
        # Create appropriate integration based on adapter type
        if adapter.component_type == ComponentType.REINFORCEMENT_LEARNING:
            self._create_rl_integration(adapter, llm_component)
            
        elif adapter.component_type == ComponentType.SYMBOLIC_AI:
            self._create_symbolic_integration(adapter, llm_component)
            
        elif adapter.component_type == ComponentType.MULTIMODAL:
            self._create_multimodal_integration(adapter, llm_component)
            
        elif adapter.component_type == ComponentType.AGENT:
            self._create_agent_integration(adapter, llm_component)
            
        elif adapter.component_type == ComponentType.NEUROMORPHIC:
            self._create_neuromorphic_integration(adapter, llm_component)
    
    def _create_rl_integration(self, adapter: AIComponent, llm_component: AIComponent) -> None:
        """
        Create Reinforcement Learning integration.
        
        Args:
            adapter: The RL adapter
            llm_component: The LLM component
        """
        # Create integration ID
        integration_id = f"llm_rl_{adapter.component_id}"
        
        try:
            # Import the LLM-RL bridge class
            from ..llm_rl_bridge_improved import ImprovedLLMtoRLBridge
            
            # Create a custom integration wrapper
            class LLMRLIntegration(AITechnologyIntegration):
                """Integration between LLM and Reinforcement Learning."""
                
                def __init__(self, integration_id: str, bridge: ImprovedLLMtoRLBridge):
                    super().__init__(
                        integration_id=integration_id,
                        integration_type=TechnologyIntegrationType.LLM_RL,
                        llm_component_id=llm_component.component_id,
                        technology_component_id=adapter.component_id,
                        priority=IntegrationPriority.HIGH,
                        method=IntegrationMethod.SEQUENTIAL
                    )
                    self.bridge = bridge
                    
                    # Add capabilities
                    self.add_capability(IntegrationCapabilityTag.AUTONOMOUS_ACTION)
                    self.add_capability(IntegrationCapabilityTag.LEARNING_FROM_FEEDBACK)
                    self.add_capability(IntegrationCapabilityTag.GOAL_ORIENTED_PLANNING)
                
                def _activate_impl(self) -> bool:
                    # No special activation needed, bridge is already active
                    return True
                
                def _deactivate_impl(self) -> bool:
                    # No special deactivation needed
                    return True
                
                def _process_task_impl(
                    self, 
                    task_type: str, 
                    task_content: Dict[str, Any],
                    context: Dict[str, Any]
                ) -> Dict[str, Any]:
                    # Process through the bridge
                    if task_type == "goal_execution":
                        # Create goal and wait for initial processing
                        goal_id = self.bridge.create_goal_from_text(
                            text=task_content.get("goal_description", ""),
                            priority=task_content.get("priority", 0),
                            constraints=task_content.get("constraints", []),
                            deadline=task_content.get("deadline"),
                            metadata=task_content.get("metadata", {})
                        )
                        
                        # Wait briefly for initial processing
                        time.sleep(0.5)
                        
                        # Return goal information
                        return {
                            "content": {
                                "goal_id": goal_id,
                                "status": "processing",
                                "check_status": True
                            }
                        }
                        
                    elif task_type == "goal_status":
                        # Get goal status
                        goal_id = task_content.get("goal_id")
                        if not goal_id:
                            return {"content": {"error": "Missing goal_id"}}
                            
                        status = self.bridge.get_goal_status(goal_id)
                        return {"content": status}
                        
                    elif task_type == "cancel_goal":
                        # Cancel a goal
                        goal_id = task_content.get("goal_id")
                        if not goal_id:
                            return {"content": {"error": "Missing goal_id"}}
                            
                        success = self.bridge.cancel_goal(goal_id)
                        return {"content": {"success": success}}
                        
                    else:
                        # Unknown task type
                        return {"content": {"error": f"Unknown task type: {task_type}"}}
            
            # Create event bus
            event_bus = EventBus()
            
            # Create the bridge
            bridge = ImprovedLLMtoRLBridge(
                bridge_id=f"bridge_{integration_id}",
                llm_component_id=llm_component.component_id,
                rl_component_id=adapter.component_id,
                event_bus=event_bus
            )
            
            # Create the integration
            integration = LLMRLIntegration(integration_id, bridge)
            
            # Register with framework
            self.register_integration(integration)
            
            self.logger.info(f"Created LLM-RL integration: {integration_id}")
            
        except Exception as e:
            self.logger.error(f"Error creating LLM-RL integration: {str(e)}")
    
    def _create_symbolic_integration(self, adapter: AIComponent, llm_component: AIComponent) -> None:
        """
        Create Symbolic AI integration.
        
        Args:
            adapter: The Symbolic AI adapter
            llm_component: The LLM component
        """
        # Create integration ID
        integration_id = f"llm_symbolic_{adapter.component_id}"
        
        try:
            # Create a custom integration wrapper
            class LLMSymbolicIntegration(AITechnologyIntegration):
                """Integration between LLM and Symbolic AI."""
                
                def __init__(self):
                    super().__init__(
                        integration_id=integration_id,
                        integration_type=TechnologyIntegrationType.LLM_SYMBOLIC,
                        llm_component_id=llm_component.component_id,
                        technology_component_id=adapter.component_id,
                        priority=IntegrationPriority.MEDIUM,
                        method=IntegrationMethod.SEQUENTIAL
                    )
                    
                    # Add capabilities
                    self.add_capability(IntegrationCapabilityTag.LOGICAL_REASONING)
                    self.add_capability(IntegrationCapabilityTag.CAUSAL_REASONING)
                
                def _activate_impl(self) -> bool:
                    # Simple activation (would be more complex in real system)
                    return True
                
                def _deactivate_impl(self) -> bool:
                    # Simple deactivation
                    return True
                
                def _process_task_impl(
                    self, 
                    task_type: str, 
                    task_content: Dict[str, Any],
                    context: Dict[str, Any]
                ) -> Dict[str, Any]:
                    # Simple processing (would be more complex in real system)
                    if task_type == "logical_reasoning":
                        # Process logical reasoning task
                        # This would interact with the symbolic AI adapter
                        return {
                            "content": {
                                "status": "completed",
                                "result": "Simulated logical reasoning result"
                            }
                        }
                        
                    elif task_type == "causal_analysis":
                        # Process causal analysis task
                        return {
                            "content": {
                                "status": "completed",
                                "result": "Simulated causal analysis result"
                            }
                        }
                        
                    else:
                        # Unknown task type
                        return {"content": {"error": f"Unknown task type: {task_type}"}}
            
            # Create the integration
            integration = LLMSymbolicIntegration()
            
            # Register with framework
            self.register_integration(integration)
            
            self.logger.info(f"Created LLM-Symbolic integration: {integration_id}")
            
        except Exception as e:
            self.logger.error(f"Error creating LLM-Symbolic integration: {str(e)}")
    
    def _create_multimodal_integration(self, adapter: AIComponent, llm_component: AIComponent) -> None:
        """
        Create Multimodal AI integration.
        
        Args:
            adapter: The Multimodal AI adapter
            llm_component: The LLM component
        """
        # Create integration ID
        integration_id = f"llm_multimodal_{adapter.component_id}"
        
        try:
            # Create a custom integration wrapper
            class LLMMultimodalIntegration(AITechnologyIntegration):
                """Integration between LLM and Multimodal AI."""
                
                def __init__(self):
                    super().__init__(
                        integration_id=integration_id,
                        integration_type=TechnologyIntegrationType.LLM_MULTIMODAL,
                        llm_component_id=llm_component.component_id,
                        technology_component_id=adapter.component_id,
                        priority=IntegrationPriority.MEDIUM,
                        method=IntegrationMethod.PARALLEL
                    )
                    
                    # Add capabilities
                    self.add_capability(IntegrationCapabilityTag.MULTIMODAL_PERCEPTION)
                    self.add_capability(IntegrationCapabilityTag.PATTERN_RECOGNITION)
                
                def _activate_impl(self) -> bool:
                    # Simple activation
                    return True
                
                def _deactivate_impl(self) -> bool:
                    # Simple deactivation
                    return True
                
                def _process_task_impl(
                    self, 
                    task_type: str, 
                    task_content: Dict[str, Any],
                    context: Dict[str, Any]
                ) -> Dict[str, Any]:
                    # Simple processing
                    if task_type == "image_analysis":
                        # Process image analysis task
                        return {
                            "content": {
                                "status": "completed",
                                "result": "Simulated image analysis result"
                            }
                        }
                        
                    elif task_type == "multimodal_fusion":
                        # Process multimodal fusion task
                        return {
                            "content": {
                                "status": "completed",
                                "result": "Simulated multimodal fusion result"
                            }
                        }
                        
                    else:
                        # Unknown task type
                        return {"content": {"error": f"Unknown task type: {task_type}"}}
            
            # Create the integration
            integration = LLMMultimodalIntegration()
            
            # Register with framework
            self.register_integration(integration)
            
            self.logger.info(f"Created LLM-Multimodal integration: {integration_id}")
            
        except Exception as e:
            self.logger.error(f"Error creating LLM-Multimodal integration: {str(e)}")
    
    def _create_agent_integration(self, adapter: AIComponent, llm_component: AIComponent) -> None:
        """
        Create Agent-based AI integration.
        
        Args:
            adapter: The Agent-based AI adapter
            llm_component: The LLM component
        """
        # Create integration ID
        integration_id = f"llm_agent_{adapter.component_id}"
        
        try:
            # Create a custom integration wrapper
            class LLMAgentIntegration(AITechnologyIntegration):
                """Integration between LLM and Agent-based AI."""
                
                def __init__(self):
                    super().__init__(
                        integration_id=integration_id,
                        integration_type=TechnologyIntegrationType.LLM_AGENT,
                        llm_component_id=llm_component.component_id,
                        technology_component_id=adapter.component_id,
                        priority=IntegrationPriority.HIGH,
                        method=IntegrationMethod.HYBRID
                    )
                    
                    # Add capabilities
                    self.add_capability(IntegrationCapabilityTag.AUTONOMOUS_ACTION)
                    self.add_capability(IntegrationCapabilityTag.GOAL_ORIENTED_PLANNING)
                    self.add_capability(IntegrationCapabilityTag.CREATIVE_THINKING)
                
                def _activate_impl(self) -> bool:
                    # Simple activation
                    return True
                
                def _deactivate_impl(self) -> bool:
                    # Simple deactivation
                    return True
                
                def _process_task_impl(
                    self, 
                    task_type: str, 
                    task_content: Dict[str, Any],
                    context: Dict[str, Any]
                ) -> Dict[str, Any]:
                    # Simple processing
                    if task_type == "autonomous_task":
                        # Process autonomous task
                        return {
                            "content": {
                                "status": "completed",
                                "result": "Simulated autonomous task result"
                            }
                        }
                        
                    elif task_type == "multi_agent_collaboration":
                        # Process multi-agent collaboration task
                        return {
                            "content": {
                                "status": "completed",
                                "result": "Simulated multi-agent collaboration result"
                            }
                        }
                        
                    else:
                        # Unknown task type
                        return {"content": {"error": f"Unknown task type: {task_type}"}}
            
            # Create the integration
            integration = LLMAgentIntegration()
            
            # Register with framework
            self.register_integration(integration)
            
            self.logger.info(f"Created LLM-Agent integration: {integration_id}")
            
        except Exception as e:
            self.logger.error(f"Error creating LLM-Agent integration: {str(e)}")
    
    def _create_neuromorphic_integration(self, adapter: AIComponent, llm_component: AIComponent) -> None:
        """
        Create Neuromorphic AI integration.
        
        Args:
            adapter: The Neuromorphic AI adapter
            llm_component: The LLM component
        """
        # Create integration ID
        integration_id = f"llm_neuromorphic_{adapter.component_id}"
        
        try:
            # Create a custom integration wrapper
            class LLMNeuromorphicIntegration(AITechnologyIntegration):
                """Integration between LLM and Neuromorphic AI."""
                
                def __init__(self):
                    super().__init__(
                        integration_id=integration_id,
                        integration_type=TechnologyIntegrationType.LLM_NEUROMORPHIC,
                        llm_component_id=llm_component.component_id,
                        technology_component_id=adapter.component_id,
                        priority=IntegrationPriority.MEDIUM,
                        method=IntegrationMethod.SEQUENTIAL
                    )
                    
                    # Add capabilities
                    self.add_capability(IntegrationCapabilityTag.PATTERN_RECOGNITION)
                    self.add_capability(IntegrationCapabilityTag.RESOURCE_OPTIMIZATION)
                    self.add_capability(IntegrationCapabilityTag.INTUITIVE_DECISION)
                
                def _activate_impl(self) -> bool:
                    # Simple activation
                    return True
                
                def _deactivate_impl(self) -> bool:
                    # Simple deactivation
                    return True
                
                def _process_task_impl(
                    self, 
                    task_type: str, 
                    task_content: Dict[str, Any],
                    context: Dict[str, Any]
                ) -> Dict[str, Any]:
                    # Simple processing
                    if task_type == "pattern_recognition":
                        # Process pattern recognition task
                        return {
                            "content": {
                                "status": "completed",
                                "result": "Simulated pattern recognition result"
                            }
                        }
                        
                    elif task_type == "energy_optimization":
                        # Process energy optimization task
                        return {
                            "content": {
                                "status": "completed",
                                "result": "Simulated energy optimization result"
                            }
                        }
                        
                    else:
                        # Unknown task type
                        return {"content": {"error": f"Unknown task type: {task_type}"}}
            
            # Create the integration
            integration = LLMNeuromorphicIntegration()
            
            # Register with framework
            self.register_integration(integration)
            
            self.logger.info(f"Created LLM-Neuromorphic integration: {integration_id}")
            
        except Exception as e:
            self.logger.error(f"Error creating LLM-Neuromorphic integration: {str(e)}")
    
    def _setup_default_pipelines(self) -> None:
        """Set up default integration pipelines."""
        # Wait until integrations are available
        if not self.integrations:
            self.logger.warning("No integrations available, cannot create default pipelines")
            return
        
        try:
            # Create a multi-technology pipeline for comprehensive reasoning
            self._create_comprehensive_reasoning_pipeline()
            
            # Create a decision-making pipeline
            self._create_decision_making_pipeline()
            
            # Create a creative problem-solving pipeline
            self._create_creative_problem_solving_pipeline()
            
            # Create an adaptive learning pipeline
            self._create_adaptive_learning_pipeline()
            
        except Exception as e:
            self.logger.error(f"Error setting up default pipelines: {str(e)}")
    
    def _create_comprehensive_reasoning_pipeline(self) -> None:
        """Create a pipeline for comprehensive reasoning tasks."""
        # Find integrations with relevant capabilities
        logical_reasoning_integrations = self.find_integrations_with_capability(
            IntegrationCapabilityTag.LOGICAL_REASONING)
        causal_reasoning_integrations = self.find_integrations_with_capability(
            IntegrationCapabilityTag.CAUSAL_REASONING)
        pattern_recognition_integrations = self.find_integrations_with_capability(
            IntegrationCapabilityTag.PATTERN_RECOGNITION)
        
        # Combine and deduplicate
        integration_ids = list(set(
            logical_reasoning_integrations + 
            causal_reasoning_integrations + 
            pattern_recognition_integrations
        ))
        
        if not integration_ids:
            self.logger.warning("No suitable integrations for comprehensive reasoning pipeline")
            return
        
        # Create pipeline
        pipeline_id = "comprehensive_reasoning_pipeline"
        self.create_pipeline(
            pipeline_id,
            integration_ids,
            IntegrationMethod.HYBRID
        )
        
        self.logger.info(f"Created comprehensive reasoning pipeline with {len(integration_ids)} integrations")
    
    def _create_decision_making_pipeline(self) -> None:
        """Create a pipeline for decision-making tasks."""
        # Find integrations with relevant capabilities
        logical_reasoning_integrations = self.find_integrations_with_capability(
            IntegrationCapabilityTag.LOGICAL_REASONING)
        autonomous_action_integrations = self.find_integrations_with_capability(
            IntegrationCapabilityTag.AUTONOMOUS_ACTION)
        intuitive_decision_integrations = self.find_integrations_with_capability(
            IntegrationCapabilityTag.INTUITIVE_DECISION)
        
        # Combine and deduplicate
        integration_ids = list(set(
            logical_reasoning_integrations + 
            autonomous_action_integrations + 
            intuitive_decision_integrations
        ))
        
        if not integration_ids:
            self.logger.warning("No suitable integrations for decision-making pipeline")
            return
        
        # Create pipeline
        pipeline_id = "decision_making_pipeline"
        self.create_pipeline(
            pipeline_id,
            integration_ids,
            IntegrationMethod.SEQUENTIAL
        )
        
        self.logger.info(f"Created decision-making pipeline with {len(integration_ids)} integrations")
    
    def _create_creative_problem_solving_pipeline(self) -> None:
        """Create a pipeline for creative problem-solving tasks."""
        # Find integrations with relevant capabilities
        creative_thinking_integrations = self.find_integrations_with_capability(
            IntegrationCapabilityTag.CREATIVE_THINKING)
        pattern_recognition_integrations = self.find_integrations_with_capability(
            IntegrationCapabilityTag.PATTERN_RECOGNITION)
        goal_oriented_integrations = self.find_integrations_with_capability(
            IntegrationCapabilityTag.GOAL_ORIENTED_PLANNING)
        
        # Combine and deduplicate
        integration_ids = list(set(
            creative_thinking_integrations + 
            pattern_recognition_integrations + 
            goal_oriented_integrations
        ))
        
        if not integration_ids:
            self.logger.warning("No suitable integrations for creative problem-solving pipeline")
            return
        
        # Create pipeline
        pipeline_id = "creative_problem_solving_pipeline"
        self.create_pipeline(
            pipeline_id,
            integration_ids,
            IntegrationMethod.HYBRID
        )
        
        self.logger.info(f"Created creative problem-solving pipeline with {len(integration_ids)} integrations")
    
    def _create_adaptive_learning_pipeline(self) -> None:
        """Create a pipeline for adaptive learning tasks."""
        # Find integrations with relevant capabilities
        learning_feedback_integrations = self.find_integrations_with_capability(
            IntegrationCapabilityTag.LEARNING_FROM_FEEDBACK)
        pattern_recognition_integrations = self.find_integrations_with_capability(
            IntegrationCapabilityTag.PATTERN_RECOGNITION)
        
        # Combine and deduplicate
        integration_ids = list(set(
            learning_feedback_integrations + 
            pattern_recognition_integrations
        ))
        
        if not integration_ids:
            self.logger.warning("No suitable integrations for adaptive learning pipeline")
            return
        
        # Create pipeline
        pipeline_id = "adaptive_learning_pipeline"
        self.create_pipeline(
            pipeline_id,
            integration_ids,
            IntegrationMethod.SEQUENTIAL
        )
        
        self.logger.info(f"Created adaptive learning pipeline with {len(integration_ids)} integrations")
