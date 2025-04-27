"""
Enhanced Framework Integrator for Jarviee AI Technology Integration.

This module integrates all the enhanced components of the AI technology integration
framework into a unified system, providing a cohesive interface for leveraging
multiple AI technologies together effectively.
"""

import asyncio
import json
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from .base import AIComponent, ComponentType, IntegrationMessage
from .context_extension import EnhancedContextManager, ContextScope, ContextLifespan
from .data_bridge import DataBridge, DataFormat
from .framework_enhanced import EnhancedIntegrationFramework
from .resource_optimizer import (ResourceOptimizer, ComputeLocation, 
                              DeviceProfile, TaskResourceProfile)
from .xai_module import XAIModule, ExplanationComponent, ExplanationLevel, ExplanationFormat
from ..utils.event_bus import Event, EventBus
from ..utils.logger import Logger


class IntegrationStatus(Enum):
    """Status of AI technology integration."""
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class EnhancedFrameworkIntegrator:
    """
    Enhanced integration framework that combines all specialized modules.
    
    This class serves as the central hub for AI technology integration,
    coordinating the enhanced framework, data bridge, resource optimizer,
    context manager, and XAI module.
    """
    
    def __init__(self, integrator_id: str, event_bus: Optional[EventBus] = None):
        """
        Initialize the enhanced framework integrator.
        
        Args:
            integrator_id: Unique identifier for this integrator
            event_bus: Optional event bus for communication
        """
        self.integrator_id = integrator_id
        self.event_bus = event_bus or EventBus()
        
        self.logger = Logger().get_logger(f"jarviee.integration.integrator.{integrator_id}")
        
        # Initialize status
        self.status = IntegrationStatus.INITIALIZING
        
        # Initialize components
        self.framework = EnhancedIntegrationFramework()
        self.data_bridge = DataBridge(self.event_bus)
        self.resource_optimizer = ResourceOptimizer(self.event_bus)
        self.context_manager = EnhancedContextManager(self.event_bus)
        self.xai_module = XAIModule(self.event_bus)
        
        # Task tracking
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Component capabilities
        self.platform_capabilities = {
            "data_formats": [format.value for format in DataFormat],
            "compute_locations": [location.value for location in ComputeLocation],
            "explanation_levels": [level.name for level in ExplanationLevel],
            "explanation_formats": [format.value for format in ExplanationFormat],
            "explanation_components": [component.value for component in ExplanationComponent]
        }
        
        # Configuration
        self.config = {
            "auto_resource_optimization": True,
            "enable_edge_computing": True,
            "enable_explanation": True,
            "context_preservation_level": "medium",  # minimal, medium, full
            "data_conversion_strict": True,  # Whether data conversion errors should be fatal
            "resource_allocation_timeout": 30,  # seconds
            "concurrent_task_limit": 50,
            "debug_mode": False
        }
        
        # Register event handlers
        self._register_event_handlers()
        
        self.logger.info(f"Enhanced Framework Integrator {integrator_id} initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the integrator and all components.
        
        Returns:
            True if initialization was successful
        """
        try:
            # Initialize framework
            if not self.framework.initialize():
                self.logger.error("Failed to initialize integration framework")
                self.status = IntegrationStatus.ERROR
                return False
            
            # Start resource monitoring
            if self.config["auto_resource_optimization"]:
                self.resource_optimizer.start_monitoring()
            
            # Configure context manager
            self._configure_context_manager()
            
            # Update status
            self.status = IntegrationStatus.READY
            
            # Emit initialization event
            self.event_bus.publish(Event(
                "integrator.initialized",
                {"integrator_id": self.integrator_id}
            ))
            
            self.logger.info(f"Enhanced Framework Integrator {self.integrator_id} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing integrator: {str(e)}")
            self.status = IntegrationStatus.ERROR
            return False
    
    def shutdown(self) -> bool:
        """
        Shut down the integrator and all components.
        
        Returns:
            True if shutdown was successful
        """
        try:
            # Update status
            self.status = IntegrationStatus.SHUTDOWN
            
            # Stop resource monitoring
            self.resource_optimizer.stop_monitoring()
            
            # Shut down framework
            if not self.framework.shutdown():
                self.logger.error("Error shutting down integration framework")
                return False
            
            # Clean up active tasks
            for task_id in list(self.active_tasks.keys()):
                self.cancel_task(task_id)
            
            # Emit shutdown event
            self.event_bus.publish(Event(
                "integrator.shutdown",
                {"integrator_id": self.integrator_id}
            ))
            
            self.logger.info(f"Enhanced Framework Integrator {self.integrator_id} shut down successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error shutting down integrator: {str(e)}")
            return False
    
    def register_edge_device(
        self, 
        name: str,
        resources: Dict[str, float],
        capabilities: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Register an edge computing device.
        
        Args:
            name: Device name
            resources: Available resources
            capabilities: Device capabilities
            metadata: Additional metadata
            
        Returns:
            Device ID if registration was successful, or None otherwise
        """
        if not self.config["enable_edge_computing"]:
            self.logger.warning("Edge computing is disabled, cannot register device")
            return None
        
        # Create device profile
        device_id = str(uuid.uuid4())
        
        # Convert resource types
        resources_dict = {}
        for res_key, amount in resources.items():
            try:
                res_type = next(t for t in self.resource_optimizer.ResourceType 
                               if t.value == res_key)
                resources_dict[res_type] = amount
            except StopIteration:
                self.logger.warning(f"Unknown resource type: {res_key}")
        
        device = DeviceProfile(
            device_id=device_id,
            name=name,
            location=ComputeLocation.EDGE,
            resources=resources_dict,
            cost_per_hour=0.0,  # Edge devices are "free" (already owned)
            capabilities=capabilities,
            metadata=metadata
        )
        
        # Register with resource optimizer
        if not self.resource_optimizer.register_device(device):
            return None
        
        self.logger.info(f"Registered edge device {device_id} ({name})")
        return device_id
    
    def process_task(
        self, 
        integration_id: str,
        task_type: str, 
        task_content: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        input_data_format: Optional[DataFormat] = None,
        output_data_format: Optional[DataFormat] = None,
        preferred_location: Optional[ComputeLocation] = None,
        explanation_level: Optional[ExplanationLevel] = None
    ) -> Dict[str, Any]:
        """
        Process a task using an integration with enhanced features.
        
        Args:
            integration_id: ID of the integration to use
            task_type: Type of task to process
            task_content: Content of the task
            context: Optional context information
            input_data_format: Format of input data
            output_data_format: Desired format for output data
            preferred_location: Preferred compute location
            explanation_level: Desired explanation level
            
        Returns:
            Dictionary containing the result
        """
        # Create task ID and context
        task_id = str(uuid.uuid4())
        task_context = context or {}
        task_context["task_id"] = task_id
        task_context["start_time"] = time.time()
        task_context["integrator_id"] = self.integrator_id
        
        # Record the task
        self.active_tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "content": task_content,
            "integration_id": integration_id,
            "status": "created",
            "timestamp": time.time(),
            "context": task_context
        }
        
        # Check if we need to convert input data format
        if input_data_format and "content" in task_content:
            original_content = task_content["content"]
            framework_format = DataFormat.LLM_JSON  # Framework expects this format
            
            if input_data_format != framework_format:
                # Convert input data
                converted_content, success = self.data_bridge.convert(
                    original_content, input_data_format, framework_format)
                
                if success:
                    task_content["content"] = converted_content
                    task_content["original_content"] = original_content
                    task_content["input_data_format"] = input_data_format.value
                elif self.config["data_conversion_strict"]:
                    # Strict mode, data conversion failure is fatal
                    result = {
                        "status": "error",
                        "task_id": task_id,
                        "error": f"Failed to convert input data from {input_data_format.value} to {framework_format.value}"
                    }
                    
                    self.active_tasks[task_id]["status"] = "failed"
                    self.active_tasks[task_id]["result"] = result
                    
                    return result
        
        # Resource allocation for computation
        if self.config["auto_resource_optimization"] and preferred_location:
            # Register task with resource optimizer
            resource_requirements = self._estimate_resource_requirements(
                integration_id, task_type, task_content)
            
            task_profile = TaskResourceProfile(
                task_id=task_id,
                task_type=task_type,
                resource_requirements=resource_requirements,
                required_capabilities=[],  # We'd need a mapping here
                estimated_duration=60,  # Default 60 seconds
                priority=2,  # Default medium priority
                metadata={"integration_id": integration_id}
            )
            
            self.resource_optimizer.register_task(task_profile)
            
            # Allocate resources
            allocation_id = self.resource_optimizer.allocate_resources(
                task_id=task_id,
                preferred_location=preferred_location
            )
            
            if allocation_id:
                task_context["resource_allocation_id"] = allocation_id
                self.active_tasks[task_id]["allocation_id"] = allocation_id
            else:
                self.logger.warning(f"Failed to allocate resources for task {task_id}")
        
        # Create task context in context manager
        self.context_manager.create_task_context(
            task_id=task_id,
            context_data={
                "task_type": task_type,
                "integration_id": integration_id,
                "timestamp": time.time(),
                **task_context
            }
        )
        
        try:
            # Update task status
            self.active_tasks[task_id]["status"] = "processing"
            
            # Process the task using the framework
            result = self.framework.process_task(
                integration_id, task_type, task_content, task_context)
            
            # Update task status
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["completed_at"] = time.time()
            self.active_tasks[task_id]["result"] = result
            
            # Check if we need to convert output data format
            if output_data_format and "content" in result:
                framework_format = DataFormat.LLM_JSON  # Framework returns this format
                
                if output_data_format != framework_format:
                    # Convert output data
                    converted_result, success = self.data_bridge.convert(
                        result["content"], framework_format, output_data_format)
                    
                    if success:
                        # Store original content
                        result["original_content"] = result["content"]
                        result["content"] = converted_result
                        result["output_data_format"] = output_data_format.value
                    elif self.config["data_conversion_strict"]:
                        # Strict mode would handle failure differently
                        self.logger.warning(
                            f"Failed to convert output data from {framework_format.value} "
                            f"to {output_data_format.value}")
            
            # Generate explanation if requested
            if self.config["enable_explanation"] and explanation_level:
                explanation = self.xai_module.explain(
                    component=ExplanationComponent.INTEGRATION,
                    data={
                        "integration_id": integration_id,
                        "task_type": task_type,
                        "input": task_content,
                        "output": result
                    },
                    context=task_context,
                    level=explanation_level
                )
                
                # Add explanation to result
                result["explanation"] = {
                    "explanation_id": explanation.explanation_id,
                    "content": explanation.content,
                    "level": explanation.level.name
                }
            
            # Release resources if allocated
            if "allocation_id" in self.active_tasks[task_id]:
                allocation_id = self.active_tasks[task_id]["allocation_id"]
                self.resource_optimizer.release_resources(allocation_id, "completed")
            
            return result
            
        except Exception as e:
            # Handle exception
            self.logger.error(f"Error processing task {task_id}: {str(e)}")
            
            # Update task status
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["error"] = str(e)
            
            # Release resources if allocated
            if "allocation_id" in self.active_tasks[task_id]:
                allocation_id = self.active_tasks[task_id]["allocation_id"]
                self.resource_optimizer.release_resources(allocation_id, "failed")
            
            # Return error result
            return {
                "status": "error",
                "task_id": task_id,
                "error": str(e)
            }
    
    async def process_task_async(
        self, 
        integration_id: str,
        task_type: str, 
        task_content: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        input_data_format: Optional[DataFormat] = None,
        output_data_format: Optional[DataFormat] = None,
        preferred_location: Optional[ComputeLocation] = None,
        explanation_level: Optional[ExplanationLevel] = None
    ) -> Dict[str, Any]:
        """
        Process a task asynchronously using an integration with enhanced features.
        
        Args:
            integration_id: ID of the integration to use
            task_type: Type of task to process
            task_content: Content of the task
            context: Optional context information
            input_data_format: Format of input data
            output_data_format: Desired format for output data
            preferred_location: Preferred compute location
            explanation_level: Desired explanation level
            
        Returns:
            Dictionary containing the result
        """
        # This implementation follows a similar pattern to process_task
        # but uses the asynchronous version of framework processing
        
        # Create task ID and context
        task_id = str(uuid.uuid4())
        task_context = context or {}
        task_context["task_id"] = task_id
        task_context["start_time"] = time.time()
        task_context["integrator_id"] = self.integrator_id
        
        # Record the task
        self.active_tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "content": task_content,
            "integration_id": integration_id,
            "status": "created",
            "timestamp": time.time(),
            "context": task_context,
            "async": True
        }
        
        # Check if we need to convert input data format
        if input_data_format and "content" in task_content:
            original_content = task_content["content"]
            framework_format = DataFormat.LLM_JSON
            
            if input_data_format != framework_format:
                # Convert input data
                converted_content, success = self.data_bridge.convert(
                    original_content, input_data_format, framework_format)
                
                if success:
                    task_content["content"] = converted_content
                    task_content["original_content"] = original_content
                    task_content["input_data_format"] = input_data_format.value
                elif self.config["data_conversion_strict"]:
                    # Strict mode, data conversion failure is fatal
                    result = {
                        "status": "error",
                        "task_id": task_id,
                        "error": f"Failed to convert input data from {input_data_format.value} to {framework_format.value}"
                    }
                    
                    self.active_tasks[task_id]["status"] = "failed"
                    self.active_tasks[task_id]["result"] = result
                    
                    return result
        
        # Resource allocation for computation
        if self.config["auto_resource_optimization"] and preferred_location:
            # Register task with resource optimizer
            resource_requirements = self._estimate_resource_requirements(
                integration_id, task_type, task_content)
            
            task_profile = TaskResourceProfile(
                task_id=task_id,
                task_type=task_type,
                resource_requirements=resource_requirements,
                required_capabilities=[],
                estimated_duration=60,
                priority=2,
                metadata={"integration_id": integration_id}
            )
            
            self.resource_optimizer.register_task(task_profile)
            
            # Allocate resources
            allocation_id = self.resource_optimizer.allocate_resources(
                task_id=task_id,
                preferred_location=preferred_location
            )
            
            if allocation_id:
                task_context["resource_allocation_id"] = allocation_id
                self.active_tasks[task_id]["allocation_id"] = allocation_id
            else:
                self.logger.warning(f"Failed to allocate resources for task {task_id}")
        
        # Create task context in context manager
        self.context_manager.create_task_context(
            task_id=task_id,
            context_data={
                "task_type": task_type,
                "integration_id": integration_id,
                "timestamp": time.time(),
                **task_context
            }
        )
        
        try:
            # Update task status
            self.active_tasks[task_id]["status"] = "processing"
            
            # Process the task using the framework
            result = await self.framework.process_task_async(
                integration_id, task_type, task_content, task_context)
            
            # Update task status
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["completed_at"] = time.time()
            self.active_tasks[task_id]["result"] = result
            
            # Check if we need to convert output data format
            if output_data_format and "content" in result:
                framework_format = DataFormat.LLM_JSON
                
                if output_data_format != framework_format:
                    # Convert output data
                    converted_result, success = self.data_bridge.convert(
                        result["content"], framework_format, output_data_format)
                    
                    if success:
                        # Store original content
                        result["original_content"] = result["content"]
                        result["content"] = converted_result
                        result["output_data_format"] = output_data_format.value
                    elif self.config["data_conversion_strict"]:
                        self.logger.warning(
                            f"Failed to convert output data from {framework_format.value} "
                            f"to {output_data_format.value}")
            
            # Generate explanation if requested
            if self.config["enable_explanation"] and explanation_level:
                explanation = self.xai_module.explain(
                    component=ExplanationComponent.INTEGRATION,
                    data={
                        "integration_id": integration_id,
                        "task_type": task_type,
                        "input": task_content,
                        "output": result
                    },
                    context=task_context,
                    level=explanation_level
                )
                
                # Add explanation to result
                result["explanation"] = {
                    "explanation_id": explanation.explanation_id,
                    "content": explanation.content,
                    "level": explanation.level.name
                }
            
            # Release resources if allocated
            if "allocation_id" in self.active_tasks[task_id]:
                allocation_id = self.active_tasks[task_id]["allocation_id"]
                self.resource_optimizer.release_resources(allocation_id, "completed")
            
            return result
            
        except Exception as e:
            # Handle exception
            self.logger.error(f"Error processing async task {task_id}: {str(e)}")
            
            # Update task status
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["error"] = str(e)
            
            # Release resources if allocated
            if "allocation_id" in self.active_tasks[task_id]:
                allocation_id = self.active_tasks[task_id]["allocation_id"]
                self.resource_optimizer.release_resources(allocation_id, "failed")
            
            # Return error result
            return {
                "status": "error",
                "task_id": task_id,
                "error": str(e)
            }
    
    def process_task_with_pipeline(
        self, 
        pipeline_id: str,
        task_type: str, 
        task_content: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        input_data_format: Optional[DataFormat] = None,
        output_data_format: Optional[DataFormat] = None,
        preferred_location: Optional[ComputeLocation] = None,
        explanation_level: Optional[ExplanationLevel] = None
    ) -> Dict[str, Any]:
        """
        Process a task using a pipeline with enhanced features.
        
        Args:
            pipeline_id: ID of the pipeline to use
            task_type: Type of task to process
            task_content: Content of the task
            context: Optional context information
            input_data_format: Format of input data
            output_data_format: Desired format for output data
            preferred_location: Preferred compute location
            explanation_level: Desired explanation level
            
        Returns:
            Dictionary containing the result
        """
        # This implementation is similar to process_task but uses a pipeline
        
        # Create task ID and context
        task_id = str(uuid.uuid4())
        task_context = context or {}
        task_context["task_id"] = task_id
        task_context["start_time"] = time.time()
        task_context["integrator_id"] = self.integrator_id
        
        # Record the task
        self.active_tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "content": task_content,
            "pipeline_id": pipeline_id,
            "status": "created",
            "timestamp": time.time(),
            "context": task_context
        }
        
        # Check if we need to convert input data format
        if input_data_format and "content" in task_content:
            original_content = task_content["content"]
            framework_format = DataFormat.LLM_JSON
            
            if input_data_format != framework_format:
                # Convert input data
                converted_content, success = self.data_bridge.convert(
                    original_content, input_data_format, framework_format)
                
                if success:
                    task_content["content"] = converted_content
                    task_content["original_content"] = original_content
                    task_content["input_data_format"] = input_data_format.value
                elif self.config["data_conversion_strict"]:
                    # Strict mode, data conversion failure is fatal
                    result = {
                        "status": "error",
                        "task_id": task_id,
                        "error": f"Failed to convert input data from {input_data_format.value} to {framework_format.value}"
                    }
                    
                    self.active_tasks[task_id]["status"] = "failed"
                    self.active_tasks[task_id]["result"] = result
                    
                    return result
        
        # Resource allocation for computation
        if self.config["auto_resource_optimization"] and preferred_location:
            # For pipelines, we'd need to estimate combined resource requirements
            # This is a simple approach - in reality, it would be more sophisticated
            pipeline = self.framework.get_pipeline(pipeline_id)
            
            if pipeline:
                # Estimate resources based on pipeline
                resource_requirements = self._estimate_pipeline_resource_requirements(
                    pipeline, task_type, task_content)
                
                task_profile = TaskResourceProfile(
                    task_id=task_id,
                    task_type=task_type,
                    resource_requirements=resource_requirements,
                    required_capabilities=[],
                    estimated_duration=120,  # Pipelines might take longer
                    priority=2,
                    metadata={"pipeline_id": pipeline_id}
                )
                
                self.resource_optimizer.register_task(task_profile)
                
                # Allocate resources
                allocation_id = self.resource_optimizer.allocate_resources(
                    task_id=task_id,
                    preferred_location=preferred_location
                )
                
                if allocation_id:
                    task_context["resource_allocation_id"] = allocation_id
                    self.active_tasks[task_id]["allocation_id"] = allocation_id
                else:
                    self.logger.warning(f"Failed to allocate resources for pipeline task {task_id}")
        
        # Create task context in context manager
        self.context_manager.create_task_context(
            task_id=task_id,
            context_data={
                "task_type": task_type,
                "pipeline_id": pipeline_id,
                "timestamp": time.time(),
                **task_context
            }
        )
        
        try:
            # Update task status
            self.active_tasks[task_id]["status"] = "processing"
            
            # Process the task using the framework
            result = self.framework.process_task_with_pipeline(
                pipeline_id, task_type, task_content, task_context)
            
            # Update task status
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["completed_at"] = time.time()
            self.active_tasks[task_id]["result"] = result
            
            # Check if we need to convert output data format
            if output_data_format and "content" in result:
                framework_format = DataFormat.LLM_JSON
                
                if output_data_format != framework_format:
                    # Convert output data
                    converted_result, success = self.data_bridge.convert(
                        result["content"], framework_format, output_data_format)
                    
                    if success:
                        # Store original content
                        result["original_content"] = result["content"]
                        result["content"] = converted_result
                        result["output_data_format"] = output_data_format.value
                    elif self.config["data_conversion_strict"]:
                        self.logger.warning(
                            f"Failed to convert output data from {framework_format.value} "
                            f"to {output_data_format.value}")
            
            # Generate explanation if requested
            if self.config["enable_explanation"] and explanation_level:
                explanation = self.xai_module.explain(
                    component=ExplanationComponent.PIPELINE,
                    data={
                        "pipeline_id": pipeline_id,
                        "task_type": task_type,
                        "input": task_content,
                        "output": result
                    },
                    context=task_context,
                    level=explanation_level
                )
                
                # Add explanation to result
                result["explanation"] = {
                    "explanation_id": explanation.explanation_id,
                    "content": explanation.content,
                    "level": explanation.level.name
                }
            
            # Release resources if allocated
            if "allocation_id" in self.active_tasks[task_id]:
                allocation_id = self.active_tasks[task_id]["allocation_id"]
                self.resource_optimizer.release_resources(allocation_id, "completed")
            
            return result
            
        except Exception as e:
            # Handle exception
            self.logger.error(f"Error processing pipeline task {task_id}: {str(e)}")
            
            # Update task status
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["error"] = str(e)
            
            # Release resources if allocated
            if "allocation_id" in self.active_tasks[task_id]:
                allocation_id = self.active_tasks[task_id]["allocation_id"]
                self.resource_optimizer.release_resources(allocation_id, "failed")
            
            # Return error result
            return {
                "status": "error",
                "task_id": task_id,
                "error": str(e)
            }
    
    async def process_task_with_pipeline_async(
        self, 
        pipeline_id: str,
        task_type: str, 
        task_content: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        input_data_format: Optional[DataFormat] = None,
        output_data_format: Optional[DataFormat] = None,
        preferred_location: Optional[ComputeLocation] = None,
        explanation_level: Optional[ExplanationLevel] = None
    ) -> Dict[str, Any]:
        """
        Process a task with a pipeline asynchronously with enhanced features.
        
        Args:
            pipeline_id: ID of the pipeline to use
            task_type: Type of task to process
            task_content: Content of the task
            context: Optional context information
            input_data_format: Format of input data
            output_data_format: Desired format for output data
            preferred_location: Preferred compute location
            explanation_level: Desired explanation level
            
        Returns:
            Dictionary containing the result
        """
        # This implementation is similar to process_task_with_pipeline but async
        
        # Create task ID and context
        task_id = str(uuid.uuid4())
        task_context = context or {}
        task_context["task_id"] = task_id
        task_context["start_time"] = time.time()
        task_context["integrator_id"] = self.integrator_id
        
        # Record the task
        self.active_tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "content": task_content,
            "pipeline_id": pipeline_id,
            "status": "created",
            "timestamp": time.time(),
            "context": task_context,
            "async": True
        }
        
        # Check if we need to convert input data format
        if input_data_format and "content" in task_content:
            original_content = task_content["content"]
            framework_format = DataFormat.LLM_JSON
            
            if input_data_format != framework_format:
                # Convert input data
                converted_content, success = self.data_bridge.convert(
                    original_content, input_data_format, framework_format)
                
                if success:
                    task_content["content"] = converted_content
                    task_content["original_content"] = original_content
                    task_content["input_data_format"] = input_data_format.value
                elif self.config["data_conversion_strict"]:
                    # Strict mode, data conversion failure is fatal
                    result = {
                        "status": "error",
                        "task_id": task_id,
                        "error": f"Failed to convert input data from {input_data_format.value} to {framework_format.value}"
                    }
                    
                    self.active_tasks[task_id]["status"] = "failed"
                    self.active_tasks[task_id]["result"] = result
                    
                    return result
        
        # Resource allocation for computation
        if self.config["auto_resource_optimization"] and preferred_location:
            pipeline = self.framework.get_pipeline(pipeline_id)
            
            if pipeline:
                # Estimate resources based on pipeline
                resource_requirements = self._estimate_pipeline_resource_requirements(
                    pipeline, task_type, task_content)
                
                task_profile = TaskResourceProfile(
                    task_id=task_id,
                    task_type=task_type,
                    resource_requirements=resource_requirements,
                    required_capabilities=[],
                    estimated_duration=120,
                    priority=2,
                    metadata={"pipeline_id": pipeline_id}
                )
                
                self.resource_optimizer.register_task(task_profile)
                
                # Allocate resources
                allocation_id = self.resource_optimizer.allocate_resources(
                    task_id=task_id,
                    preferred_location=preferred_location
                )
                
                if allocation_id:
                    task_context["resource_allocation_id"] = allocation_id
                    self.active_tasks[task_id]["allocation_id"] = allocation_id
                else:
                    self.logger.warning(f"Failed to allocate resources for async pipeline task {task_id}")
        
        # Create task context in context manager
        self.context_manager.create_task_context(
            task_id=task_id,
            context_data={
                "task_type": task_type,
                "pipeline_id": pipeline_id,
                "timestamp": time.time(),
                **task_context
            }
        )
        
        try:
            # Update task status
            self.active_tasks[task_id]["status"] = "processing"
            
            # Process the task using the framework
            result = await self.framework.process_task_with_pipeline_async(
                pipeline_id, task_type, task_content, task_context)
            
            # Update task status
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["completed_at"] = time.time()
            self.active_tasks[task_id]["result"] = result
            
            # Check if we need to convert output data format
            if output_data_format and "content" in result:
                framework_format = DataFormat.LLM_JSON
                
                if output_data_format != framework_format:
                    # Convert output data
                    converted_result, success = self.data_bridge.convert(
                        result["content"], framework_format, output_data_format)
                    
                    if success:
                        # Store original content
                        result["original_content"] = result["content"]
                        result["content"] = converted_result
                        result["output_data_format"] = output_data_format.value
                    elif self.config["data_conversion_strict"]:
                        self.logger.warning(
                            f"Failed to convert output data from {framework_format.value} "
                            f"to {output_data_format.value}")
            
            # Generate explanation if requested
            if self.config["enable_explanation"] and explanation_level:
                explanation = self.xai_module.explain(
                    component=ExplanationComponent.PIPELINE,
                    data={
                        "pipeline_id": pipeline_id,
                        "task_type": task_type,
                        "input": task_content,
                        "output": result
                    },
                    context=task_context,
                    level=explanation_level
                )
                
                # Add explanation to result
                result["explanation"] = {
                    "explanation_id": explanation.explanation_id,
                    "content": explanation.content,
                    "level": explanation.level.name
                }
            
            # Release resources if allocated
            if "allocation_id" in self.active_tasks[task_id]:
                allocation_id = self.active_tasks[task_id]["allocation_id"]
                self.resource_optimizer.release_resources(allocation_id, "completed")
            
            return result
            
        except Exception as e:
            # Handle exception
            self.logger.error(f"Error processing async pipeline task {task_id}: {str(e)}")
            
            # Update task status
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["error"] = str(e)
            
            # Release resources if allocated
            if "allocation_id" in self.active_tasks[task_id]:
                allocation_id = self.active_tasks[task_id]["allocation_id"]
                self.resource_optimizer.release_resources(allocation_id, "failed")
            
            # Return error result
            return {
                "status": "error",
                "task_id": task_id,
                "error": str(e)
            }
    
    def select_integration_for_task(
        self, 
        task_type: str, 
        task_content: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        input_data_format: Optional[DataFormat] = None,
        required_capabilities: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Select the most appropriate integration for a task.
        
        Args:
            task_type: Type of task
            task_content: Task content
            context: Optional context
            input_data_format: Format of input data
            required_capabilities: Required capabilities
            
        Returns:
            ID of the selected integration, or None if none found
        """
        # Convert capabilities to the format expected by the framework
        framework_capabilities = None
        
        if required_capabilities:
            try:
                # This expects an enum import from your framework
                from .framework import IntegrationCapabilityTag
                framework_capabilities = [
                    getattr(IntegrationCapabilityTag, cap.upper())
                    for cap in required_capabilities
                    if hasattr(IntegrationCapabilityTag, cap.upper())
                ]
            except ImportError:
                self.logger.warning("Failed to import IntegrationCapabilityTag")
        
        # Convert task content if needed
        task_content_to_use = task_content
        
        if input_data_format and input_data_format != DataFormat.LLM_JSON and "content" in task_content:
            original_content = task_content["content"]
            converted_content, success = self.data_bridge.convert(
                original_content, input_data_format, DataFormat.LLM_JSON)
            
            if success:
                task_content_to_use = task_content.copy()
                task_content_to_use["content"] = converted_content
        
        # Use the framework to select an integration
        return self.framework.select_integration_for_task(
            task_type, task_content_to_use, context, framework_capabilities)
    
    def create_task_pipeline(
        self, 
        task_type: str,
        task_content: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        input_data_format: Optional[DataFormat] = None,
        required_capabilities: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Create a pipeline specifically for a task.
        
        Args:
            task_type: Type of task
            task_content: Task content
            context: Optional context
            input_data_format: Format of input data
            required_capabilities: Required capabilities
            
        Returns:
            ID of the created pipeline, or None if creation failed
        """
        # Convert capabilities to the format expected by the framework
        framework_capabilities = None
        
        if required_capabilities:
            try:
                from .framework import IntegrationCapabilityTag
                framework_capabilities = [
                    getattr(IntegrationCapabilityTag, cap.upper())
                    for cap in required_capabilities
                    if hasattr(IntegrationCapabilityTag, cap.upper())
                ]
            except ImportError:
                self.logger.warning("Failed to import IntegrationCapabilityTag")
        
        # Convert task content if needed
        task_content_to_use = task_content
        
        if input_data_format and input_data_format != DataFormat.LLM_JSON and "content" in task_content:
            original_content = task_content["content"]
            converted_content, success = self.data_bridge.convert(
                original_content, input_data_format, DataFormat.LLM_JSON)
            
            if success:
                task_content_to_use = task_content.copy()
                task_content_to_use["content"] = converted_content
        
        # Use the framework to create a task pipeline
        return self.framework.create_task_pipeline(
            task_type, task_content_to_use, context, framework_capabilities)
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dictionary with task status information
        """
        if task_id not in self.active_tasks:
            return {"error": "Task not found", "task_id": task_id}
        
        task = self.active_tasks[task_id]
        
        # Create a copy to avoid modifying the original
        status = task.copy()
        
        # Remove potentially large content
        if "content" in status and len(json.dumps(status["content"])) > 1000:
            status["content"] = {
                "truncated": True, 
                "size": len(json.dumps(task["content"]))
            }
        
        if "result" in status and len(json.dumps(status["result"])) > 1000:
            status["result"] = {
                "truncated": True, 
                "size": len(json.dumps(task["result"]))
            }
        
        # Add resource allocation status if available
        if "allocation_id" in task:
            allocation_id = task["allocation_id"]
            allocation = self.resource_optimizer.get_allocation(allocation_id)
            if allocation:
                status["resource_allocation"] = allocation
        
        # Add context information
        context_info = self.context_manager.find_contexts(
            scope=ContextScope.TASK,
            scope_id=task_id,
            limit=10
        )
        
        if context_info:
            status["context_items"] = len(context_info)
        
        return status
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if the task was cancelled successfully
        """
        if task_id not in self.active_tasks:
            self.logger.warning(f"Task {task_id} not found for cancellation")
            return False
        
        task = self.active_tasks[task_id]
        
        # Check if task can be cancelled
        if task["status"] in ["completed", "failed", "cancelled"]:
            self.logger.warning(f"Task {task_id} already {task['status']}, cannot cancel")
            return False
        
        # Release resources if allocated
        if "allocation_id" in task:
            allocation_id = task["allocation_id"]
            self.resource_optimizer.release_resources(allocation_id, "cancelled")
        
        # Cancel task in framework
        if "integration_id" in task:
            self.framework.cancel_task(task_id)
            
        elif "pipeline_id" in task:
            # No specific method for cancelling pipeline tasks in the framework,
            # but we could implement one if needed
            pass
        
        # Update status
        task["status"] = "cancelled"
        task["cancelled_at"] = time.time()
        
        # Emit event if available
        self.event_bus.publish(Event(
            "integrator.task_cancelled",
            {"task_id": task_id}
        ))
        
        self.logger.info(f"Task {task_id} cancelled")
        return True
    
    def get_explanation(self, explanation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an explanation by ID.
        
        Args:
            explanation_id: ID of the explanation
            
        Returns:
            Explanation dictionary, or None if not found
        """
        explanation = self.xai_module.get_explanation(explanation_id)
        if explanation:
            return explanation.to_dict()
        return None
    
    def get_integrator_status(self) -> Dict[str, Any]:
        """
        Get the current status of the integrator.
        
        Returns:
            Dictionary with status information
        """
        # Get status from each component
        framework_status = self.framework.get_enhanced_status()
        resource_status = self.resource_optimizer.get_resource_usage_stats()
        context_status = self.context_manager.get_memory_usage()
        
        # Count tasks by status
        task_counts = {
            "total": len(self.active_tasks),
            "created": sum(1 for t in self.active_tasks.values() if t.get("status") == "created"),
            "processing": sum(1 for t in self.active_tasks.values() if t.get("status") == "processing"),
            "completed": sum(1 for t in self.active_tasks.values() if t.get("status") == "completed"),
            "failed": sum(1 for t in self.active_tasks.values() if t.get("status") == "failed"),
            "cancelled": sum(1 for t in self.active_tasks.values() if t.get("status") == "cancelled")
        }
        
        return {
            "integrator_id": self.integrator_id,
            "status": self.status.value,
            "framework_status": framework_status,
            "resource_status": resource_status,
            "context_status": context_status,
            "task_counts": task_counts,
            "platform_capabilities": self.platform_capabilities,
            "config": self.config,
            "timestamp": time.time()
        }
    
    def set_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration settings.
        
        Args:
            config_updates: Dictionary with configuration updates
            
        Returns:
            Dictionary with the updated configuration
        """
        # Update config
        self.config.update(config_updates)
        
        # Apply relevant changes to components
        if "auto_resource_optimization" in config_updates:
            if self.config["auto_resource_optimization"]:
                self.resource_optimizer.start_monitoring()
            else:
                self.resource_optimizer.stop_monitoring()
        
        if "context_preservation_level" in config_updates:
            self._configure_context_manager()
        
        self.logger.info(f"Updated configuration: {config_updates}")
        return self.config
    
    def _register_event_handlers(self) -> None:
        """Register event handlers for component events."""
        # Framework events
        self.event_bus.subscribe(
            "integration.task_completed", self._handle_framework_task_completed)
        self.event_bus.subscribe(
            "integration.task_failed", self._handle_framework_task_failed)
        
        # Resource events
        self.event_bus.subscribe(
            "resource.device_updated", self._handle_resource_device_updated)
        self.event_bus.subscribe(
            "resource.allocation_optimized", self._handle_resource_allocation_optimized)
        
        # Context events
        self.event_bus.subscribe(
            "context.updated", self._handle_context_updated)
        self.event_bus.subscribe(
            "context.summary_created", self._handle_context_summary_created)
    
    def _handle_framework_task_completed(self, event: Event) -> None:
        """
        Handle framework task completion event.
        
        Args:
            event: The event data
        """
        if "task_id" not in event.data:
            return
            
        task_id = event.data["task_id"]
        
        if task_id in self.active_tasks:
            self.logger.debug(f"Task {task_id} completed in framework")
    
    def _handle_framework_task_failed(self, event: Event) -> None:
        """
        Handle framework task failure event.
        
        Args:
            event: The event data
        """
        if "task_id" not in event.data:
            return
            
        task_id = event.data["task_id"]
        
        if task_id in self.active_tasks:
            self.logger.warning(f"Task {task_id} failed in framework: {event.data.get('error', 'Unknown error')}")
    
    def _handle_resource_device_updated(self, event: Event) -> None:
        """
        Handle resource device update event.
        
        Args:
            event: The event data
        """
        if "device_id" not in event.data:
            return
            
        device_id = event.data["device_id"]
        online = event.data.get("online")
        
        if online is False:
            self.logger.warning(f"Device {device_id} is now offline")
    
    def _handle_resource_allocation_optimized(self, event: Event) -> None:
        """
        Handle resource allocation optimization event.
        
        Args:
            event: The event data
        """
        if "task_id" not in event.data:
            return
            
        task_id = event.data["task_id"]
        old_device_id = event.data.get("old_device_id")
        new_device_id = event.data.get("new_device_id")
        
        if task_id in self.active_tasks:
            self.logger.info(
                f"Task {task_id} resources reallocated from device {old_device_id} "
                f"to device {new_device_id}"
            )
            
            # Update task allocation ID
            if "new_allocation_id" in event.data:
                self.active_tasks[task_id]["allocation_id"] = event.data["new_allocation_id"]
    
    def _handle_context_updated(self, event: Event) -> None:
        """
        Handle context update event.
        
        Args:
            event: The event data
        """
        if self.config["debug_mode"]:
            if "context_id" in event.data and "key" in event.data:
                self.logger.debug(
                    f"Context updated: {event.data['context_id']} "
                    f"(key: {event.data['key']})"
                )
    
    def _handle_context_summary_created(self, event: Event) -> None:
        """
        Handle context summary creation event.
        
        Args:
            event: The event data
        """
        if "summary_id" in event.data and "context_count" in event.data:
            self.logger.info(
                f"Context summary created: {event.data['summary_id']} "
                f"({event.data['context_count']} contexts)"
            )
    
    def _configure_context_manager(self) -> None:
        """Configure context manager based on settings."""
        level = self.config["context_preservation_level"]
        
        if level == "minimal":
            self.context_manager.config.update({
                "auto_summarize": True,
                "summarization_threshold": 5,
                "auto_cleanup_expired": True,
                "cleanup_interval": 60
            })
        elif level == "medium":
            self.context_manager.config.update({
                "auto_summarize": True,
                "summarization_threshold": 10,
                "auto_cleanup_expired": True,
                "cleanup_interval": 300
            })
        elif level == "full":
            self.context_manager.config.update({
                "auto_summarize": False,
                "auto_cleanup_expired": False
            })
    
    def _estimate_resource_requirements(
        self, 
        integration_id: str,
        task_type: str,
        task_content: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Estimate resource requirements for a task.
        
        Args:
            integration_id: ID of the integration
            task_type: Type of task
            task_content: Task content
            
        Returns:
            Dictionary mapping resource types to required amounts
        """
        from .resource_optimizer import ResourceType
        
        # This is a simple implementation - in a real system, this would be
        # more sophisticated, possibly using machine learning to predict
        # resource requirements based on historical data
        
        # Default requirements
        requirements = {
            ResourceType.CPU: 10.0,  # 10% CPU
            ResourceType.MEMORY: 0.2,  # 0.2 GB
            ResourceType.API_RATE: 1.0  # 1 API call
        }
        
        # Adjust based on task type
        if "code_generation" in task_type:
            # Code generation typically needs more resources
            requirements[ResourceType.CPU] *= 1.5
            requirements[ResourceType.MEMORY] *= 1.5
            requirements[ResourceType.API_RATE] *= 1.5
            
        elif "analysis" in task_type:
            # Analysis typically needs more memory
            requirements[ResourceType.MEMORY] *= 2.0
        
        # Adjust based on task content size
        content_size = len(json.dumps(task_content))
        if content_size > 10000:
            # Large content typically needs more resources
            requirements[ResourceType.CPU] *= 1.2
            requirements[ResourceType.MEMORY] *= 1.5
        
        return requirements
    
    def _estimate_pipeline_resource_requirements(
        self, 
        pipeline,
        task_type: str,
        task_content: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Estimate resource requirements for a pipeline task.
        
        Args:
            pipeline: Pipeline object
            task_type: Type of task
            task_content: Task content
            
        Returns:
            Dictionary mapping resource types to required amounts
        """
        from .framework import IntegrationMethod
        from .resource_optimizer import ResourceType
        
        # Get method type
        method = pipeline.method
        
        # Calculate total requirements based on individual integrations
        total_requirements = {}
        
        for integration in pipeline.integrations:
            # Estimate requirements for this integration
            requirements = self._estimate_resource_requirements(
                integration.integration_id, task_type, task_content)
            
            # Add to total requirements based on method
            for res_type, amount in requirements.items():
                if res_type not in total_requirements:
                    total_requirements[res_type] = amount
                elif method == IntegrationMethod.PARALLEL:
                    # For parallel pipelines, add up resources
                    total_requirements[res_type] += amount
                else:
                    # For sequential pipelines, take the max
                    total_requirements[res_type] = max(
                        total_requirements[res_type], amount)
        
        # Add pipeline overhead
        for res_type in total_requirements:
            total_requirements[res_type] *= 1.1  # 10% overhead
        
        return total_requirements
