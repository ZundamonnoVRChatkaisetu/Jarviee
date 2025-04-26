"""
Enhanced AI Technology Integration Framework for Jarviee

This module extends the base integration framework with advanced features for
AI technology integration, including dynamic technology selection, resource optimization,
and context continuity management.

It serves as a unified integration layer that coordinates multiple AI technologies,
enabling them to work together synergistically to provide capabilities beyond
what any single technology could achieve.
"""

import asyncio
import json
import logging
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from ..utils.event_bus import Event, EventBus
from ..utils.logger import Logger
from .base import AIComponent, ComponentType, IntegrationMessage
from .coordinator.context_manager import ContextLifetime, ContextManager, ContextScope
from .coordinator.dynamic_selector import DynamicTechnologySelector, SelectionCriterion
from .coordinator.resource_manager import ResourceManager, ResourceType
from .framework import (AITechnologyIntegration, IntegrationCapabilityTag,
                     IntegrationFramework, IntegrationMethod,
                     IntegrationPipeline, IntegrationPriority,
                     TechnologyIntegrationType)


class IntegrationPerformanceMetrics:
    """Performance metrics for technology integrations."""
    
    def __init__(self):
        """Initialize the performance metrics."""
        self.start_time = time.time()
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.avg_response_time_ms = 0.0
        self.avg_task_complexity = 0.0
        self.avg_resource_usage = {}
        self.integration_stats = {}
        self.task_type_stats = {}
    
    def update_task_metrics(
        self, 
        integration_id: str,
        task_type: str,
        success: bool,
        response_time_ms: float,
        complexity: float = 0.5,
        resource_usage: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update metrics with task results.
        
        Args:
            integration_id: ID of the integration
            task_type: Type of task processed
            success: Whether the task was successful
            response_time_ms: Response time in milliseconds
            complexity: Task complexity (0.0-1.0)
            resource_usage: Resource usage metrics
        """
        # Update global metrics
        self.total_tasks += 1
        
        if success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1
        
        # Update average response time (exponential moving average)
        alpha = 0.1  # Weight for new value
        self.avg_response_time_ms = (
            (1 - alpha) * self.avg_response_time_ms + alpha * response_time_ms
        )
        
        # Update average task complexity
        self.avg_task_complexity = (
            (self.avg_task_complexity * (self.total_tasks - 1) + complexity) /
            self.total_tasks
        )
        
        # Update average resource usage
        resource_usage = resource_usage or {}
        for resource_type, usage in resource_usage.items():
            if resource_type not in self.avg_resource_usage:
                self.avg_resource_usage[resource_type] = usage
            else:
                self.avg_resource_usage[resource_type] = (
                    (self.avg_resource_usage[resource_type] * (self.total_tasks - 1) + usage) /
                    self.total_tasks
                )
        
        # Update integration-specific metrics
        if integration_id not in self.integration_stats:
            self.integration_stats[integration_id] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "avg_response_time_ms": 0.0,
                "task_types": set()
            }
        
        integration_stats = self.integration_stats[integration_id]
        integration_stats["total_tasks"] += 1
        
        if success:
            integration_stats["successful_tasks"] += 1
        else:
            integration_stats["failed_tasks"] += 1
        
        integration_stats["avg_response_time_ms"] = (
            (integration_stats["avg_response_time_ms"] * (integration_stats["total_tasks"] - 1) +
             response_time_ms) / integration_stats["total_tasks"]
        )
        
        integration_stats["task_types"].add(task_type)
        
        # Update task type-specific metrics
        if task_type not in self.task_type_stats:
            self.task_type_stats[task_type] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "avg_response_time_ms": 0.0,
                "integrations": set()
            }
        
        task_stats = self.task_type_stats[task_type]
        task_stats["total_tasks"] += 1
        
        if success:
            task_stats["successful_tasks"] += 1
        else:
            task_stats["failed_tasks"] += 1
        
        task_stats["avg_response_time_ms"] = (
            (task_stats["avg_response_time_ms"] * (task_stats["total_tasks"] - 1) +
             response_time_ms) / task_stats["total_tasks"]
        )
        
        task_stats["integrations"].add(integration_id)
    
    def get_success_rate(self) -> float:
        """Get the overall success rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks
    
    def get_integration_success_rate(self, integration_id: str) -> float:
        """Get the success rate for a specific integration."""
        if integration_id not in self.integration_stats:
            return 0.0
        
        stats = self.integration_stats[integration_id]
        total = stats["total_tasks"]
        
        if total == 0:
            return 0.0
        
        return stats["successful_tasks"] / total
    
    def get_task_type_success_rate(self, task_type: str) -> float:
        """Get the success rate for a specific task type."""
        if task_type not in self.task_type_stats:
            return 0.0
        
        stats = self.task_type_stats[task_type]
        total = stats["total_tasks"]
        
        if total == 0:
            return 0.0
        
        return stats["successful_tasks"] / total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get the overall statistics."""
        uptime_seconds = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime_seconds,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.get_success_rate(),
            "avg_response_time_ms": self.avg_response_time_ms,
            "avg_task_complexity": self.avg_task_complexity,
            "avg_resource_usage": self.avg_resource_usage,
            "integration_count": len(self.integration_stats),
            "task_type_count": len(self.task_type_stats)
        }


class EnhancedIntegrationFramework(IntegrationFramework):
    """
    Enhanced framework for managing AI technology integrations.
    
    This class extends the base IntegrationFramework with advanced features:
    - Dynamic technology selection
    - Resource optimization
    - Context continuity management
    - Performance monitoring and optimization
    - Advanced pipeline execution strategies
    """
    
    def __init__(self):
        """Initialize the enhanced integration framework."""
        super().__init__()
        
        self.logger = Logger().get_logger("jarviee.integration.enhanced_framework")
        
        # Enhanced components
        self.resource_manager = ResourceManager()
        self.context_manager = ContextManager(self.event_bus)
        self.technology_selector = DynamicTechnologySelector(
            self.resource_manager, self.event_bus)
        
        # Performance metrics
        self.performance_metrics = IntegrationPerformanceMetrics()
        
        # Active tasks
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Enhanced settings
        self.config = {
            "enable_resource_optimization": True,
            "enable_context_continuity": True,
            "enable_dynamic_selection": True,
            "enable_performance_monitoring": True,
            "auto_create_pipelines": True,
            "cache_task_results": True,
            "cache_lifetime_seconds": 3600,  # 1 hour
            "max_concurrent_tasks": 10,
            "task_timeout_seconds": 300,  # 5 minutes
            "log_level": "info"
        }
        
        # Register additional event handlers
        self._register_enhanced_event_handlers()
        
        self.logger.info("Enhanced Integration Framework initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the framework and all components.
        
        Returns:
            True if initialization was successful
        """
        # Initialize base framework
        if not super().initialize():
            return False
        
        try:
            # Initialize enhanced components
            self.resource_manager.start_monitoring()
            
            # Set up default integrations
            if self.config["auto_create_pipelines"]:
                self._create_default_pipelines()
            
            self.logger.info("Enhanced Integration Framework initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Enhanced Integration Framework: {str(e)}")
            return False
    
    def shutdown(self) -> bool:
        """
        Shutdown the framework and all components.
        
        Returns:
            True if shutdown was successful
        """
        try:
            # Stop resource monitoring
            self.resource_manager.stop_monitoring()
            
            # Clean up active tasks
            for task_id in list(self.active_tasks.keys()):
                self.cancel_task(task_id)
            
            # Base shutdown
            if not super().shutdown():
                return False
            
            self.logger.info("Enhanced Integration Framework shut down successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error shutting down Enhanced Integration Framework: {str(e)}")
            return False
    
    def register_integration(self, integration: AITechnologyIntegration) -> None:
        """
        Register an integration with enhanced tracking.
        
        Args:
            integration: The integration to register
        """
        # Use base implementation
        super().register_integration(integration)
        
        # Register with the dynamic selector
        if self.config["enable_dynamic_selection"]:
            # The selector already monitors events, so no need to call it directly
            pass
        
        # Set up task context for the integration
        if self.config["enable_context_continuity"]:
            self.context_manager.set_context(
                key="integration_registered",
                value=True,
                scope=ContextScope.INTEGRATION,
                lifetime=ContextLifetime.PERMANENT,
                integration_id=integration.integration_id,
                metadata={
                    "integration_type": integration.integration_type.name,
                    "priority": integration.priority.name,
                    "method": integration.method.name,
                    "capabilities": [cap.name for cap in integration.capabilities]
                }
            )
    
    def process_task(
        self, 
        integration_id: str,
        task_type: str, 
        task_content: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a task with enhanced resource management and context tracking.
        
        Args:
            integration_id: ID of the integration to use
            task_type: Type of task to process
            task_content: Content of the task
            context: Optional context information
            
        Returns:
            Dictionary containing the result
        """
        # Create task ID and task context
        task_id = str(uuid.uuid4())
        task_context = context or {}
        task_context["task_id"] = task_id
        task_context["start_time"] = time.time()
        
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
        
        # Resource management
        if self.config["enable_resource_optimization"]:
            integration = self.get_integration(integration_id)
            if not integration:
                raise ValueError(f"Integration with ID {integration_id} not found")
            
            # Get resource requirements based on integration type and task
            resource_requirements = self._estimate_resource_requirements(
                integration, task_type, task_content)
            
            # Check if resources are available
            if not self.resource_manager.check_resources_available(resource_requirements):
                # Resources not available - return info about unavailability
                availability = self.resource_manager.get_availability_estimate(
                    resource_requirements)
                
                result = {
                    "status": "resource_unavailable",
                    "task_id": task_id,
                    "message": "Required resources are not available",
                    "availability_estimate": availability
                }
                
                self.active_tasks[task_id]["status"] = "resource_unavailable"
                self.active_tasks[task_id]["result"] = result
                
                return result
            
            # Allocate resources
            self.resource_manager.allocate_resources(
                component_id=integration_id,
                requests=resource_requirements,
                priority=integration.priority.value,
                duration=self.config["task_timeout_seconds"]
            )
        
        # Set up task context
        if self.config["enable_context_continuity"]:
            self.context_manager.create_task_context(
                task_id=task_id,
                context_data={
                    "task_type": task_type,
                    "integration_id": integration_id,
                    "timestamp": time.time(),
                    **task_context
                }
            )
        
        # Check for cached result if enabled
        if self.config["cache_task_results"]:
            # Generate cache key based on task type, content, and integration
            cache_key = self._generate_cache_key(task_type, task_content, integration_id)
            
            # Check if result is cached
            cached_result = self.context_manager.get_context(
                key=cache_key,
                scope=ContextScope.GLOBAL,
                default=None
            )
            
            if cached_result is not None:
                # Use cached result
                self.logger.debug(f"Using cached result for task {task_id}")
                
                result = cached_result.copy()
                result["cached"] = True
                result["task_id"] = task_id
                
                self.active_tasks[task_id]["status"] = "completed"
                self.active_tasks[task_id]["result"] = result
                self.active_tasks[task_id]["completed_at"] = time.time()
                
                # Track metrics
                if self.config["enable_performance_monitoring"]:
                    # Calculate response time (from cache)
                    response_time_ms = 50.0  # Default cached response time
                    
                    # Update metrics
                    self.performance_metrics.update_task_metrics(
                        integration_id=integration_id,
                        task_type=task_type,
                        success=True,
                        response_time_ms=response_time_ms,
                        complexity=0.3  # Lower complexity since cached
                    )
                
                return result
        
        try:
            # Process task
            self.active_tasks[task_id]["status"] = "processing"
            
            # Start measuring response time
            start_time = time.time()
            
            # Actual processing using the base implementation
            result = super().process_task(integration_id, task_type, task_content, task_context)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Update task status
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["completed_at"] = time.time()
            self.active_tasks[task_id]["result"] = result
            
            # Add task ID to result
            result["task_id"] = task_id
            
            # Track metrics
            if self.config["enable_performance_monitoring"]:
                # Estimate task complexity based on task type and content
                complexity = self._estimate_task_complexity(task_type, task_content)
                
                # Get resource usage
                resource_usage = {}
                if self.config["enable_resource_optimization"]:
                    # Get actual resource usage (placeholder - would be more sophisticated)
                    resource_usage = {
                        "cpu": 0.5,  # Placeholder values
                        "memory": 0.3,
                        "api_calls": 10
                    }
                
                # Update metrics
                self.performance_metrics.update_task_metrics(
                    integration_id=integration_id,
                    task_type=task_type,
                    success=True,
                    response_time_ms=response_time_ms,
                    complexity=complexity,
                    resource_usage=resource_usage
                )
            
            # Cache result if enabled
            if self.config["cache_task_results"]:
                # Generate cache key
                cache_key = self._generate_cache_key(task_type, task_content, integration_id)
                
                # Cache the result
                self.context_manager.set_context(
                    key=cache_key,
                    value=result.copy(),
                    scope=ContextScope.GLOBAL,
                    lifetime=ContextLifetime.SHORT,  # Use short lifetime for cache
                    metadata={
                        "task_type": task_type,
                        "integration_id": integration_id,
                        "timestamp": time.time()
                    }
                )
            
            # Emit task completion event
            self.event_bus.publish(Event(
                "integration.task_completed",
                {
                    "task_id": task_id,
                    "task_type": task_type,
                    "integration_id": integration_id,
                    "response_time_ms": response_time_ms
                }
            ))
            
            return result
            
        except Exception as e:
            # Handle exception
            self.logger.error(f"Error processing task {task_id}: {str(e)}")
            
            # Update task status
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["error"] = str(e)
            
            # Track metrics for failed task
            if self.config["enable_performance_monitoring"]:
                # Calculate response time
                response_time_ms = (time.time() - start_time) * 1000
                
                # Update metrics
                self.performance_metrics.update_task_metrics(
                    integration_id=integration_id,
                    task_type=task_type,
                    success=False,
                    response_time_ms=response_time_ms
                )
            
            # Emit task failure event
            self.event_bus.publish(Event(
                "integration.task_failed",
                {
                    "task_id": task_id,
                    "task_type": task_type,
                    "integration_id": integration_id,
                    "error": str(e)
                }
            ))
            
            # Return error result
            return {
                "status": "error",
                "task_id": task_id,
                "error": str(e)
            }
            
        finally:
            # Release resources
            if self.config["enable_resource_optimization"]:
                self.resource_manager.release_resources(integration_id)
    
    async def process_task_async(
        self, 
        integration_id: str,
        task_type: str, 
        task_content: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a task asynchronously with enhanced features.
        
        Args:
            integration_id: ID of the integration to use
            task_type: Type of task to process
            task_content: Content of the task
            context: Optional context information
            
        Returns:
            Dictionary containing the result
        """
        # Much of this code is similar to the synchronous version
        # Create task ID and task context
        task_id = str(uuid.uuid4())
        task_context = context or {}
        task_context["task_id"] = task_id
        task_context["start_time"] = time.time()
        
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
        
        # Resource management
        if self.config["enable_resource_optimization"]:
            integration = self.get_integration(integration_id)
            if not integration:
                raise ValueError(f"Integration with ID {integration_id} not found")
            
            # Get resource requirements based on integration type and task
            resource_requirements = self._estimate_resource_requirements(
                integration, task_type, task_content)
            
            # Check if resources are available
            if not self.resource_manager.check_resources_available(resource_requirements):
                # Resources not available - return info about unavailability
                availability = self.resource_manager.get_availability_estimate(
                    resource_requirements)
                
                result = {
                    "status": "resource_unavailable",
                    "task_id": task_id,
                    "message": "Required resources are not available",
                    "availability_estimate": availability
                }
                
                self.active_tasks[task_id]["status"] = "resource_unavailable"
                self.active_tasks[task_id]["result"] = result
                
                return result
            
            # Allocate resources
            self.resource_manager.allocate_resources(
                component_id=integration_id,
                requests=resource_requirements,
                priority=integration.priority.value,
                duration=self.config["task_timeout_seconds"]
            )
        
        # Set up task context
        if self.config["enable_context_continuity"]:
            self.context_manager.create_task_context(
                task_id=task_id,
                context_data={
                    "task_type": task_type,
                    "integration_id": integration_id,
                    "timestamp": time.time(),
                    **task_context
                }
            )
        
        # Check for cached result if enabled
        if self.config["cache_task_results"]:
            # Generate cache key based on task type, content, and integration
            cache_key = self._generate_cache_key(task_type, task_content, integration_id)
            
            # Check if result is cached
            cached_result = self.context_manager.get_context(
                key=cache_key,
                scope=ContextScope.GLOBAL,
                default=None
            )
            
            if cached_result is not None:
                # Use cached result
                self.logger.debug(f"Using cached result for task {task_id}")
                
                result = cached_result.copy()
                result["cached"] = True
                result["task_id"] = task_id
                
                self.active_tasks[task_id]["status"] = "completed"
                self.active_tasks[task_id]["result"] = result
                self.active_tasks[task_id]["completed_at"] = time.time()
                
                # Track metrics
                if self.config["enable_performance_monitoring"]:
                    # Calculate response time (from cache)
                    response_time_ms = 50.0  # Default cached response time
                    
                    # Update metrics
                    self.performance_metrics.update_task_metrics(
                        integration_id=integration_id,
                        task_type=task_type,
                        success=True,
                        response_time_ms=response_time_ms,
                        complexity=0.3  # Lower complexity since cached
                    )
                
                return result
        
        try:
            # Process task
            self.active_tasks[task_id]["status"] = "processing"
            
            # Start measuring response time
            start_time = time.time()
            
            # Actual async processing using the base implementation
            result = await super().process_task_async(
                integration_id, task_type, task_content, task_context)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Update task status
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["completed_at"] = time.time()
            self.active_tasks[task_id]["result"] = result
            
            # Add task ID to result
            result["task_id"] = task_id
            
            # Track metrics
            if self.config["enable_performance_monitoring"]:
                # Estimate task complexity based on task type and content
                complexity = self._estimate_task_complexity(task_type, task_content)
                
                # Get resource usage
                resource_usage = {}
                if self.config["enable_resource_optimization"]:
                    # Get actual resource usage (placeholder - would be more sophisticated)
                    resource_usage = {
                        "cpu": 0.5,  # Placeholder values
                        "memory": 0.3,
                        "api_calls": 10
                    }
                
                # Update metrics
                self.performance_metrics.update_task_metrics(
                    integration_id=integration_id,
                    task_type=task_type,
                    success=True,
                    response_time_ms=response_time_ms,
                    complexity=complexity,
                    resource_usage=resource_usage
                )
            
            # Cache result if enabled
            if self.config["cache_task_results"]:
                # Generate cache key
                cache_key = self._generate_cache_key(task_type, task_content, integration_id)
                
                # Cache the result
                self.context_manager.set_context(
                    key=cache_key,
                    value=result.copy(),
                    scope=ContextScope.GLOBAL,
                    lifetime=ContextLifetime.SHORT,  # Use short lifetime for cache
                    metadata={
                        "task_type": task_type,
                        "integration_id": integration_id,
                        "timestamp": time.time()
                    }
                )
            
            # Emit task completion event
            self.event_bus.publish(Event(
                "integration.task_completed",
                {
                    "task_id": task_id,
                    "task_type": task_type,
                    "integration_id": integration_id,
                    "response_time_ms": response_time_ms
                }
            ))
            
            return result
            
        except Exception as e:
            # Handle exception
            self.logger.error(f"Error processing async task {task_id}: {str(e)}")
            
            # Update task status
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["error"] = str(e)
            
            # Track metrics for failed task
            if self.config["enable_performance_monitoring"]:
                # Calculate response time
                response_time_ms = (time.time() - start_time) * 1000
                
                # Update metrics
                self.performance_metrics.update_task_metrics(
                    integration_id=integration_id,
                    task_type=task_type,
                    success=False,
                    response_time_ms=response_time_ms
                )
            
            # Emit task failure event
            self.event_bus.publish(Event(
                "integration.task_failed",
                {
                    "task_id": task_id,
                    "task_type": task_type,
                    "integration_id": integration_id,
                    "error": str(e)
                }
            ))
            
            # Return error result
            return {
                "status": "error",
                "task_id": task_id,
                "error": str(e)
            }
            
        finally:
            # Release resources
            if self.config["enable_resource_optimization"]:
                self.resource_manager.release_resources(integration_id)
    
    def process_task_with_pipeline(
        self, 
        pipeline_id: str,
        task_type: str, 
        task_content: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a task with a pipeline with enhanced features.
        
        Args:
            pipeline_id: ID of the pipeline to use
            task_type: Type of task to process
            task_content: Content of the task
            context: Optional context information
            
        Returns:
            Dictionary containing the result
        """
        # Create task ID and task context
        task_id = str(uuid.uuid4())
        task_context = context or {}
        task_context["task_id"] = task_id
        task_context["start_time"] = time.time()
        
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
        
        # Get the pipeline
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline with ID {pipeline_id} not found")
        
        # Resource management
        if self.config["enable_resource_optimization"]:
            # For pipelines, we need to estimate the total resource requirements
            # This is a simple approach - in reality, this would be more sophisticated
            total_requirements = {}
            
            for integration in pipeline.integrations:
                # Get requirements for this integration
                requirements = self._estimate_resource_requirements(
                    integration, task_type, task_content)
                
                # Add to total requirements
                for resource_type, amount in requirements.items():
                    if resource_type not in total_requirements:
                        total_requirements[resource_type] = amount
                    else:
                        # For sequential pipelines, take the max
                        # For parallel pipelines, add them up
                        if pipeline.method == IntegrationMethod.PARALLEL:
                            total_requirements[resource_type] += amount
                        else:
                            total_requirements[resource_type] = max(
                                total_requirements[resource_type], amount)
            
            # Check if resources are available
            if not self.resource_manager.check_resources_available(total_requirements):
                # Resources not available - return info about unavailability
                availability = self.resource_manager.get_availability_estimate(
                    total_requirements)
                
                result = {
                    "status": "resource_unavailable",
                    "task_id": task_id,
                    "message": "Required resources are not available",
                    "availability_estimate": availability
                }
                
                self.active_tasks[task_id]["status"] = "resource_unavailable"
                self.active_tasks[task_id]["result"] = result
                
                return result
            
            # Allocate resources
            self.resource_manager.allocate_resources(
                component_id=f"pipeline_{pipeline_id}",
                requests=total_requirements,
                priority=3,  # High priority for pipelines
                duration=self.config["task_timeout_seconds"]
            )
        
        # Set up task context
        if self.config["enable_context_continuity"]:
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
            # Process task
            self.active_tasks[task_id]["status"] = "processing"
            
            # Start measuring response time
            start_time = time.time()
            
            # Actual processing using the base implementation
            result = super().process_task_with_pipeline(
                pipeline_id, task_type, task_content, task_context)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Update task status
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["completed_at"] = time.time()
            self.active_tasks[task_id]["result"] = result
            
            # Add task ID to result
            result["task_id"] = task_id
            
            # Track metrics for each integration in the pipeline
            if self.config["enable_performance_monitoring"]:
                # Estimate task complexity based on task type and content
                complexity = self._estimate_task_complexity(task_type, task_content)
                
                # Track metrics for each integration
                for integration in pipeline.integrations:
                    self.performance_metrics.update_task_metrics(
                        integration_id=integration.integration_id,
                        task_type=task_type,
                        success=True,
                        response_time_ms=response_time_ms / len(pipeline.integrations),
                        complexity=complexity
                    )
            
            # Emit task completion event
            self.event_bus.publish(Event(
                "integration.pipeline_task_completed",
                {
                    "task_id": task_id,
                    "task_type": task_type,
                    "pipeline_id": pipeline_id,
                    "response_time_ms": response_time_ms
                }
            ))
            
            return result
            
        except Exception as e:
            # Handle exception
            self.logger.error(f"Error processing pipeline task {task_id}: {str(e)}")
            
            # Update task status
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["error"] = str(e)
            
            # Track metrics for failed task
            if self.config["enable_performance_monitoring"]:
                # Calculate response time
                response_time_ms = (time.time() - start_time) * 1000
                
                # Track metrics for each integration
                for integration in pipeline.integrations:
                    self.performance_metrics.update_task_metrics(
                        integration_id=integration.integration_id,
                        task_type=task_type,
                        success=False,
                        response_time_ms=response_time_ms / len(pipeline.integrations)
                    )
            
            # Emit task failure event
            self.event_bus.publish(Event(
                "integration.pipeline_task_failed",
                {
                    "task_id": task_id,
                    "task_type": task_type,
                    "pipeline_id": pipeline_id,
                    "error": str(e)
                }
            ))
            
            # Return error result
            return {
                "status": "error",
                "task_id": task_id,
                "error": str(e)
            }
            
        finally:
            # Release resources
            if self.config["enable_resource_optimization"]:
                self.resource_manager.release_resources(f"pipeline_{pipeline_id}")
    
    async def process_task_with_pipeline_async(
        self, 
        pipeline_id: str,
        task_type: str, 
        task_content: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a task with a pipeline asynchronously with enhanced features.
        
        Args:
            pipeline_id: ID of the pipeline to use
            task_type: Type of task to process
            task_content: Content of the task
            context: Optional context information
            
        Returns:
            Dictionary containing the result
        """
        # This implementation follows the same pattern as process_task_with_pipeline
        # but using the async version of the processing
        
        # Create task ID and task context
        task_id = str(uuid.uuid4())
        task_context = context or {}
        task_context["task_id"] = task_id
        task_context["start_time"] = time.time()
        
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
        
        # Get the pipeline
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline with ID {pipeline_id} not found")
        
        # Resource management
        if self.config["enable_resource_optimization"]:
            # For pipelines, we need to estimate the total resource requirements
            total_requirements = {}
            
            for integration in pipeline.integrations:
                # Get requirements for this integration
                requirements = self._estimate_resource_requirements(
                    integration, task_type, task_content)
                
                # Add to total requirements
                for resource_type, amount in requirements.items():
                    if resource_type not in total_requirements:
                        total_requirements[resource_type] = amount
                    else:
                        # For sequential pipelines, take the max
                        # For parallel pipelines, add them up
                        if pipeline.method == IntegrationMethod.PARALLEL:
                            total_requirements[resource_type] += amount
                        else:
                            total_requirements[resource_type] = max(
                                total_requirements[resource_type], amount)
            
            # Check if resources are available
            if not self.resource_manager.check_resources_available(total_requirements):
                # Resources not available - return info about unavailability
                availability = self.resource_manager.get_availability_estimate(
                    total_requirements)
                
                result = {
                    "status": "resource_unavailable",
                    "task_id": task_id,
                    "message": "Required resources are not available",
                    "availability_estimate": availability
                }
                
                self.active_tasks[task_id]["status"] = "resource_unavailable"
                self.active_tasks[task_id]["result"] = result
                
                return result
            
            # Allocate resources
            self.resource_manager.allocate_resources(
                component_id=f"pipeline_{pipeline_id}",
                requests=total_requirements,
                priority=3,  # High priority for pipelines
                duration=self.config["task_timeout_seconds"]
            )
        
        # Set up task context
        if self.config["enable_context_continuity"]:
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
            # Process task
            self.active_tasks[task_id]["status"] = "processing"
            
            # Start measuring response time
            start_time = time.time()
            
            # Actual async processing using the base implementation
            result = await super().process_task_with_pipeline_async(
                pipeline_id, task_type, task_content, task_context)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Update task status
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["completed_at"] = time.time()
            self.active_tasks[task_id]["result"] = result
            
            # Add task ID to result
            result["task_id"] = task_id
            
            # Track metrics for each integration in the pipeline
            if self.config["enable_performance_monitoring"]:
                # Estimate task complexity based on task type and content
                complexity = self._estimate_task_complexity(task_type, task_content)
                
                # Track metrics for each integration
                for integration in pipeline.integrations:
                    self.performance_metrics.update_task_metrics(
                        integration_id=integration.integration_id,
                        task_type=task_type,
                        success=True,
                        response_time_ms=response_time_ms / len(pipeline.integrations),
                        complexity=complexity
                    )
            
            # Emit task completion event
            self.event_bus.publish(Event(
                "integration.pipeline_task_completed",
                {
                    "task_id": task_id,
                    "task_type": task_type,
                    "pipeline_id": pipeline_id,
                    "response_time_ms": response_time_ms
                }
            ))
            
            return result
            
        except Exception as e:
            # Handle exception
            self.logger.error(f"Error processing async pipeline task {task_id}: {str(e)}")
            
            # Update task status
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["error"] = str(e)
            
            # Track metrics for failed task
            if self.config["enable_performance_monitoring"]:
                # Calculate response time
                response_time_ms = (time.time() - start_time) * 1000
                
                # Track metrics for each integration
                for integration in pipeline.integrations:
                    self.performance_metrics.update_task_metrics(
                        integration_id=integration.integration_id,
                        task_type=task_type,
                        success=False,
                        response_time_ms=response_time_ms / len(pipeline.integrations)
                    )
            
            # Emit task failure event
            self.event_bus.publish(Event(
                "integration.pipeline_task_failed",
                {
                    "task_id": task_id,
                    "task_type": task_type,
                    "pipeline_id": pipeline_id,
                    "error": str(e)
                }
            ))
            
            # Return error result
            return {
                "status": "error",
                "task_id": task_id,
                "error": str(e)
            }
            
        finally:
            # Release resources
            if self.config["enable_resource_optimization"]:
                self.resource_manager.release_resources(f"pipeline_{pipeline_id}")
    
    def select_integration_for_task(
        self, 
        task_type: str, 
        task_content: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        required_capabilities: Optional[List[IntegrationCapabilityTag]] = None
    ) -> Optional[str]:
        """
        Select the most appropriate integration for a task.
        
        Args:
            task_type: Type of task to process
            task_content: Content of the task
            context: Optional context information
            required_capabilities: List of capabilities that the integration must have
            
        Returns:
            ID of the selected integration, or None if no suitable integration was found
        """
        if not self.config["enable_dynamic_selection"]:
            # Fall back to base implementation
            return super().select_integration_for_task(
                task_type, task_content, context, required_capabilities)
        
        # Get all active integrations
        active_integrations = [
            integration for integration in self.integrations.values()
            if integration.active
        ]
        
        if not active_integrations:
            return None
        
        # Use the dynamic selector
        selected_integration = self.technology_selector.select_technology(
            task_type=task_type,
            task_content=task_content,
            context=context or {},
            available_integrations=active_integrations,
            required_capabilities=required_capabilities
        )
        
        if selected_integration:
            return selected_integration.integration_id
        
        return None
    
    def create_task_pipeline(
        self, 
        task_type: str,
        task_content: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        required_capabilities: Optional[List[IntegrationCapabilityTag]] = None
    ) -> Optional[str]:
        """
        Create a pipeline specifically for a task.
        
        Args:
            task_type: Type of task to process
            task_content: Content of the task
            context: Optional context information
            required_capabilities: List of capabilities that the pipeline must provide
            
        Returns:
            ID of the created pipeline, or None if no suitable pipeline could be created
        """
        if not self.config["enable_dynamic_selection"]:
            # Fall back to base implementation
            return super().create_task_pipeline(
                task_type, task_content, context, required_capabilities)
        
        # Get all active integrations
        active_integrations = [
            integration for integration in self.integrations.values()
            if integration.active
        ]
        
        if not active_integrations:
            return None
        
        # Create pipeline using the dynamic selector
        selected_integrations, method = self.technology_selector.design_integration_pipeline(
            task_type=task_type,
            task_content=task_content,
            context=context or {},
            available_integrations=active_integrations,
            required_capabilities=required_capabilities
        )
        
        if not selected_integrations:
            return None
        
        # Create a new pipeline
        pipeline_id = f"task_pipeline_{task_type}_{int(time.time())}"
        
        # Create the pipeline
        pipeline = IntegrationPipeline(
            pipeline_id=pipeline_id,
            method=method,
            resource_manager=self.resource_manager
        )
        
        # Add the selected integrations
        for integration in selected_integrations:
            pipeline.add_integration(integration)
        
        # Register the pipeline
        self.register_pipeline(pipeline)
        
        self.logger.info(
            f"Created task pipeline {pipeline_id} with {len(selected_integrations)} "
            f"integrations using {method.name} method"
        )
        
        return pipeline_id
    
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
        if task["status"] in ["completed", "failed"]:
            self.logger.warning(f"Task {task_id} already {task['status']}, cannot cancel")
            return False
        
        # Release resources
        if self.config["enable_resource_optimization"]:
            if "integration_id" in task:
                self.resource_manager.release_resources(task["integration_id"])
            elif "pipeline_id" in task:
                self.resource_manager.release_resources(f"pipeline_{task['pipeline_id']}")
        
        # Update status
        task["status"] = "cancelled"
        task["cancelled_at"] = time.time()
        
        # Clean up context
        if self.config["enable_context_continuity"]:
            self.context_manager.clear_context(task_id=task_id)
        
        self.logger.info(f"Task {task_id} cancelled")
        
        # Emit task cancellation event
        self.event_bus.publish(Event(
            "integration.task_cancelled",
            {"task_id": task_id}
        ))
        
        return True
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """
        Get enhanced status information.
        
        Returns:
            Dictionary with enhanced status information
        """
        base_status = self.get_framework_status()
        
        # Add enhanced status information
        enhanced_status = {
            "base_status": base_status,
            "resource_status": (
                self.resource_manager.get_resource_usage_stats()
                if self.config["enable_resource_optimization"] else None
            ),
            "context_status": (
                self.context_manager.get_memory_usage()
                if self.config["enable_context_continuity"] else None
            ),
            "performance_metrics": (
                self.performance_metrics.get_stats()
                if self.config["enable_performance_monitoring"] else None
            ),
            "active_tasks": len(self.active_tasks),
            "config": self.config
        }
        
        return enhanced_status
    
    def get_performance_metrics(
        self, 
        integration_id: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Args:
            integration_id: Optional integration ID to filter by
            task_type: Optional task type to filter by
            
        Returns:
            Dictionary with performance metrics
        """
        if not self.config["enable_performance_monitoring"]:
            return {"error": "Performance monitoring is disabled"}
        
        # Get metrics from the selector
        metrics = self.technology_selector.get_performance_stats(
            integration_id, task_type)
        
        return metrics
    
    def set_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration settings.
        
        Args:
            config_updates: Dictionary with configuration updates
            
        Returns:
            Dictionary with the updated configuration
        """
        # Update config
        for key, value in config_updates.items():
            if key in self.config:
                self.config[key] = value
            else:
                self.logger.warning(f"Unknown config key: {key}")
        
        # Apply changes based on config settings
        if self.config["enable_resource_optimization"]:
            if not self.resource_manager._monitoring_active:
                self.resource_manager.start_monitoring()
        else:
            if self.resource_manager._monitoring_active:
                self.resource_manager.stop_monitoring()
        
        # Set log level
        log_level = self.config.get("log_level", "info").upper()
        log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR
        }
        
        if log_level in log_level_map:
            self.logger.setLevel(log_level_map[log_level])
        
        return self.config
    
    def set_selection_weights(
        self, 
        weights: Dict[str, float]
    ) -> Dict[SelectionCriterion, float]:
        """
        Set weights for the technology selector.
        
        Args:
            weights: Dictionary mapping selection criteria to weights
            
        Returns:
            Dictionary with the updated weights
        """
        if not self.config["enable_dynamic_selection"]:
            return {}
        
        # Update weights
        for criterion_name, weight in weights.items():
            try:
                criterion = SelectionCriterion(criterion_name)
                self.technology_selector.set_selection_weight(criterion, weight)
            except ValueError:
                self.logger.warning(f"Unknown selection criterion: {criterion_name}")
        
        # Return current weights
        return {
            criterion.value: self.technology_selector.get_selection_weight(criterion)
            for criterion in SelectionCriterion
        }
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get resource usage information.
        
        Returns:
            Dictionary with resource usage information
        """
        if not self.config["enable_resource_optimization"]:
            return {"error": "Resource optimization is disabled"}
        
        return self.resource_manager.get_resource_usage_stats()
    
    def optimize_resources(self) -> Dict[str, Any]:
        """
        Optimize resource usage.
        
        Returns:
            Dictionary with optimization results
        """
        if not self.config["enable_resource_optimization"]:
            return {"error": "Resource optimization is disabled"}
        
        changes_made = self.resource_manager.optimize_allocations()
        
        return {
            "optimized": changes_made,
            "current_usage": self.resource_manager.get_resource_usage_stats()
        }
    
    def get_context_stats(self) -> Dict[str, Any]:
        """
        Get context management statistics.
        
        Returns:
            Dictionary with context statistics
        """
        if not self.config["enable_context_continuity"]:
            return {"error": "Context continuity is disabled"}
        
        return self.context_manager.get_memory_usage()
    
    def _register_enhanced_event_handlers(self) -> None:
        """Register event handlers for enhanced features."""
        # Task events
        self.event_bus.subscribe(
            "integration.task_completed", self._handle_task_completed)
        self.event_bus.subscribe(
            "integration.task_failed", self._handle_task_failed)
        self.event_bus.subscribe(
            "integration.pipeline_task_completed", self._handle_pipeline_task_completed)
        self.event_bus.subscribe(
            "integration.pipeline_task_failed", self._handle_pipeline_task_failed)
        
        # Resource events
        self.event_bus.subscribe(
            "resource.availability_updated", self._handle_resource_availability)
        
        # Context events
        self.event_bus.subscribe(
            "context.updated", self._handle_context_updated)
    
    def _handle_task_completed(self, event: Event) -> None:
        """
        Handle task completion event.
        
        Args:
            event: The event data
        """
        # This is handled in the technology selector
        pass
    
    def _handle_task_failed(self, event: Event) -> None:
        """
        Handle task failure event.
        
        Args:
            event: The event data
        """
        # This is handled in the technology selector
        pass
    
    def _handle_pipeline_task_completed(self, event: Event) -> None:
        """
        Handle pipeline task completion event.
        
        Args:
            event: The event data
        """
        # Pipeline-specific handling
        pass
    
    def _handle_pipeline_task_failed(self, event: Event) -> None:
        """
        Handle pipeline task failure event.
        
        Args:
            event: The event data
        """
        # Pipeline-specific handling
        pass
    
    def _handle_resource_availability(self, event: Event) -> None:
        """
        Handle resource availability update event.
        
        Args:
            event: The event data
        """
        # Currently, no special handling needed
        pass
    
    def _handle_context_updated(self, event: Event) -> None:
        """
        Handle context update event.
        
        Args:
            event: The event data
        """
        # Currently, no special handling needed
        pass
    
    def _estimate_resource_requirements(
        self, 
        integration: AITechnologyIntegration,
        task_type: str,
        task_content: Dict[str, Any]
    ) -> Dict[ResourceType, float]:
        """
        Estimate resource requirements for a task.
        
        Args:
            integration: The integration to use
            task_type: Type of task to process
            task_content: Content of the task
            
        Returns:
            Dictionary mapping resource types to required amounts
        """
        # This is a simple implementation - in a real system, this would be
        # more sophisticated, possibly using machine learning to predict
        # resource requirements based on historical data
        
        # Default requirements
        requirements = {
            ResourceType.CPU: 10.0,  # 10% CPU
            ResourceType.MEMORY: 0.2,  # 0.2 GB
            ResourceType.API_RATE: 1.0  # 1 API call
        }
        
        # Adjust based on integration type
        if integration.integration_type == TechnologyIntegrationType.LLM_RL:
            # RL typically needs more CPU
            requirements[ResourceType.CPU] = 20.0
            requirements[ResourceType.MEMORY] = 0.5
            
        elif integration.integration_type == TechnologyIntegrationType.LLM_MULTIMODAL:
            # Multimodal typically needs more memory
            requirements[ResourceType.MEMORY] = 1.0
            
        elif integration.integration_type == TechnologyIntegrationType.LLM_AGENT:
            # Agents might make more API calls
            requirements[ResourceType.API_RATE] = 5.0
        
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
    
    def _estimate_task_complexity(
        self, 
        task_type: str,
        task_content: Dict[str, Any]
    ) -> float:
        """
        Estimate the complexity of a task.
        
        Args:
            task_type: Type of task
            task_content: Content of the task
            
        Returns:
            Complexity score (0.0-1.0)
        """
        # This is a simple implementation - in a real system, this would be
        # more sophisticated, possibly using machine learning
        
        # Base complexity
        complexity = 0.5
        
        # Adjust based on task type
        if "code_generation" in task_type:
            complexity += 0.2
        elif "analysis" in task_type:
            complexity += 0.1
        elif "simple" in task_type:
            complexity -= 0.2
        
        # Adjust based on content size
        content_size = len(json.dumps(task_content))
        if content_size > 10000:
            complexity += 0.2
        elif content_size < 1000:
            complexity -= 0.1
        
        # Ensure complexity is in valid range
        return max(0.0, min(1.0, complexity))
    
    def _generate_cache_key(
        self, 
        task_type: str,
        task_content: Dict[str, Any],
        integration_id: str
    ) -> str:
        """
        Generate a cache key for a task.
        
        Args:
            task_type: Type of task
            task_content: Content of the task
            integration_id: ID of the integration
            
        Returns:
            Cache key
        """
        # This is a simple implementation - in a real system, this would be
        # more sophisticated, handling things like input normalization
        
        # Create a cache key
        key_parts = [
            f"type:{task_type}",
            f"integration:{integration_id}"
        ]
        
        # Add a hash of the content
        content_str = json.dumps(task_content, sort_keys=True)
        content_hash = hash(content_str) % 10000000  # Simple hash
        key_parts.append(f"content_hash:{content_hash}")
        
        return "cache:" + ":".join(key_parts)
    
    def _create_default_pipelines(self) -> None:
        """Create default integration pipelines."""
        # This is a placeholder implementation
        
        # We'll only create default pipelines if we have enough integrations
        if len(self.integrations) < 2:
            return
        
        # Pick some integrations
        active_integrations = [
            integration for integration in self.integrations.values()
            if integration.active
        ]
        
        integration_ids = [
            integration.integration_id for integration in active_integrations
        ]
        
        if len(integration_ids) < 2:
            return
        
        # Create a sample pipeline
        pipeline_id = "default_pipeline"
        
        # Check if pipeline already exists
        if pipeline_id in self.pipelines:
            return
        
        # Create the pipeline
        self.create_pipeline(
            pipeline_id=pipeline_id,
            integration_ids=integration_ids[:2],  # Use first two integrations
            method=IntegrationMethod.SEQUENTIAL
        )
        
        self.logger.info(f"Created default pipeline {pipeline_id}")
