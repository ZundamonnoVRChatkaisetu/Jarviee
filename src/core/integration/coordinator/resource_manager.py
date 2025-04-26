"""
Resource Manager for AI Technology Integrations.

This module provides a resource management system for AI technology integrations,
allowing for efficient allocation and optimization of computational resources
across different AI technologies.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import psutil

from ...utils.event_bus import Event, EventBus
from ...utils.logger import Logger


class ResourceType(Enum):
    """Types of resources that can be managed."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    API_RATE = "api_rate"
    DATABASE = "database"
    NETWORK = "network"


@dataclass
class ResourceAllocation:
    """Represents a resource allocation for a component."""
    component_id: str
    resource_type: ResourceType
    amount: float  # Percentage or absolute value
    priority: int
    timestamp: float
    expiration: Optional[float] = None  # None means no expiration


@dataclass
class ResourceAvailability:
    """Represents the availability of a resource."""
    resource_type: ResourceType
    total: float
    available: float
    reserved: float


class ResourceManager:
    """
    Manages and optimizes resource allocation across AI technology integrations.
    
    This class provides functionality for tracking, allocating, and releasing
    computational resources such as CPU, memory, and GPU, as well as managing
    rate limits for external APIs and other shared resources.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the resource manager.
        
        Args:
            event_bus: Optional event bus for communication
        """
        self.logger = Logger().get_logger("jarviee.resource_manager")
        self.event_bus = event_bus or EventBus()
        
        # Resource tracking
        self.allocations: Dict[str, List[ResourceAllocation]] = {}
        self.availability: Dict[ResourceType, ResourceAvailability] = {}
        
        # Resource limits
        self.resource_limits: Dict[ResourceType, float] = {}
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 5.0  # seconds
        
        # Initialize with default values
        self._initialize_resources()
        
        self.logger.info("Resource Manager initialized")
    
    def _initialize_resources(self) -> None:
        """Initialize resource tracking with system information."""
        # CPU
        cpu_count = psutil.cpu_count(logical=True)
        self.resource_limits[ResourceType.CPU] = 100.0  # Percentage
        self.availability[ResourceType.CPU] = ResourceAvailability(
            resource_type=ResourceType.CPU,
            total=100.0,  # Percentage
            available=100.0,
            reserved=0.0
        )
        
        # Memory
        total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
        self.resource_limits[ResourceType.MEMORY] = total_memory
        self.availability[ResourceType.MEMORY] = ResourceAvailability(
            resource_type=ResourceType.MEMORY,
            total=total_memory,
            available=total_memory,
            reserved=0.0
        )
        
        # GPU - Use a placeholder if not available
        try:
            # This is a placeholder for GPU detection
            # In a real implementation, we would use a library like pynvml
            gpu_count = 0
            self.resource_limits[ResourceType.GPU] = gpu_count * 100.0
            self.availability[ResourceType.GPU] = ResourceAvailability(
                resource_type=ResourceType.GPU,
                total=gpu_count * 100.0,
                available=gpu_count * 100.0,
                reserved=0.0
            )
        except Exception as e:
            self.logger.warning(f"GPU detection failed: {e}")
            self.resource_limits[ResourceType.GPU] = 0.0
            self.availability[ResourceType.GPU] = ResourceAvailability(
                resource_type=ResourceType.GPU,
                total=0.0,
                available=0.0,
                reserved=0.0
            )
        
        # API rate limits - These would be configured based on external APIs
        self.resource_limits[ResourceType.API_RATE] = 100.0  # Requests per minute
        self.availability[ResourceType.API_RATE] = ResourceAvailability(
            resource_type=ResourceType.API_RATE,
            total=100.0,
            available=100.0,
            reserved=0.0
        )
        
        # Database connections
        self.resource_limits[ResourceType.DATABASE] = 100.0  # Max connections
        self.availability[ResourceType.DATABASE] = ResourceAvailability(
            resource_type=ResourceType.DATABASE,
            total=100.0,
            available=100.0,
            reserved=0.0
        )
        
        # Network bandwidth (in Mbps)
        self.resource_limits[ResourceType.NETWORK] = 1000.0  # Mbps
        self.availability[ResourceType.NETWORK] = ResourceAvailability(
            resource_type=ResourceType.NETWORK,
            total=1000.0,
            available=1000.0,
            reserved=0.0
        )
    
    def start_monitoring(self) -> bool:
        """
        Start resource usage monitoring.
        
        Returns:
            True if monitoring was started successfully
        """
        if self.monitoring_active:
            return True
        
        try:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            self.logger.info("Resource monitoring started")
            return True
        except Exception as e:
            self.logger.error(f"Error starting resource monitoring: {e}")
            self.monitoring_active = False
            return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop resource usage monitoring.
        
        Returns:
            True if monitoring was stopped successfully
        """
        if not self.monitoring_active:
            return True
        
        try:
            self.monitoring_active = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            self.logger.info("Resource monitoring stopped")
            return True
        except Exception as e:
            self.logger.error(f"Error stopping resource monitoring: {e}")
            return False
    
    def _monitoring_loop(self) -> None:
        """Background thread for monitoring resource usage."""
        while self.monitoring_active:
            try:
                self._update_resource_availability()
                self._cleanup_expired_allocations()
                
                # Emit resource update event
                if self.event_bus:
                    self.event_bus.publish(Event(
                        "resource.availability_updated",
                        {"availability": self.get_resource_availability()}
                    ))
                
                # Sleep for the monitoring interval
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in resource monitoring loop: {e}")
                time.sleep(5.0)  # Sleep a bit longer on error
    
    def _update_resource_availability(self) -> None:
        """Update resource availability based on system metrics."""
        with self.lock:
            # Update CPU availability
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.availability[ResourceType.CPU].available = max(
                0.0, 100.0 - cpu_percent - self.availability[ResourceType.CPU].reserved
            )
            
            # Update memory availability
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024 * 1024 * 1024)
            memory_total_gb = memory.total / (1024 * 1024 * 1024)
            self.availability[ResourceType.MEMORY].available = max(
                0.0, memory_total_gb - memory_used_gb - self.availability[ResourceType.MEMORY].reserved
            )
            
            # Update GPU availability (placeholder)
            # In a real implementation, we would use a library like pynvml
            
            # Other resources would be updated based on their specific metrics
    
    def _cleanup_expired_allocations(self) -> None:
        """Remove expired resource allocations."""
        current_time = time.time()
        
        with self.lock:
            for component_id, allocations in list(self.allocations.items()):
                # Filter out expired allocations
                valid_allocations = []
                for allocation in allocations:
                    if allocation.expiration is not None and allocation.expiration <= current_time:
                        # Release the resource
                        self._release_resource(allocation)
                        self.logger.debug(
                            f"Released expired {allocation.resource_type.value} "
                            f"allocation for {component_id}"
                        )
                    else:
                        valid_allocations.append(allocation)
                
                # Update the allocations list
                if valid_allocations:
                    self.allocations[component_id] = valid_allocations
                else:
                    del self.allocations[component_id]
    
    def allocate_resources(
        self, 
        component_id: str,
        requests: Dict[ResourceType, float],
        priority: int = 1,
        duration: Optional[float] = None
    ) -> bool:
        """
        Allocate resources to a component.
        
        Args:
            component_id: ID of the component requesting resources
            requests: Dictionary mapping resource types to requested amounts
            priority: Priority level of the allocation (higher is more important)
            duration: Optional duration in seconds for the allocation
            
        Returns:
            True if allocation was successful, False otherwise
        """
        with self.lock:
            # Check if all requested resources are available
            for resource_type, amount in requests.items():
                if resource_type not in self.availability:
                    self.logger.error(f"Unknown resource type: {resource_type}")
                    return False
                
                if amount > self.availability[resource_type].available:
                    self.logger.warning(
                        f"Insufficient {resource_type.value} resources: "
                        f"requested {amount}, available {self.availability[resource_type].available}"
                    )
                    return False
            
            # Allocate resources
            allocation_time = time.time()
            expiration = allocation_time + duration if duration is not None else None
            
            # Create allocations
            for resource_type, amount in requests.items():
                allocation = ResourceAllocation(
                    component_id=component_id,
                    resource_type=resource_type,
                    amount=amount,
                    priority=priority,
                    timestamp=allocation_time,
                    expiration=expiration
                )
                
                # Add to allocations
                if component_id not in self.allocations:
                    self.allocations[component_id] = []
                self.allocations[component_id].append(allocation)
                
                # Update availability
                self.availability[resource_type].available -= amount
                self.availability[resource_type].reserved += amount
            
            self.logger.debug(f"Resources allocated for {component_id}")
            
            # Emit resource allocation event
            if self.event_bus:
                self.event_bus.publish(Event(
                    "resource.allocation_created",
                    {
                        "component_id": component_id,
                        "resources": requests,
                        "priority": priority,
                        "duration": duration
                    }
                ))
            
            return True
    
    def release_resources(self, component_id: str) -> bool:
        """
        Release all resources allocated to a component.
        
        Args:
            component_id: ID of the component
            
        Returns:
            True if resources were released, False if component not found
        """
        with self.lock:
            if component_id not in self.allocations:
                self.logger.warning(f"No allocations found for component {component_id}")
                return False
            
            # Release each allocation
            for allocation in self.allocations[component_id]:
                self._release_resource(allocation)
            
            # Remove the component's allocations
            del self.allocations[component_id]
            
            self.logger.debug(f"Resources released for {component_id}")
            
            # Emit resource release event
            if self.event_bus:
                self.event_bus.publish(Event(
                    "resource.allocation_released",
                    {"component_id": component_id}
                ))
            
            return True
    
    def _release_resource(self, allocation: ResourceAllocation) -> None:
        """Release a specific resource allocation."""
        resource_type = allocation.resource_type
        amount = allocation.amount
        
        # Update availability
        self.availability[resource_type].available += amount
        self.availability[resource_type].reserved -= amount
    
    def get_resource_availability(self) -> Dict[str, Dict[str, float]]:
        """
        Get the current availability of all resources.
        
        Returns:
            Dictionary mapping resource types to availability information
        """
        with self.lock:
            availability = {}
            for resource_type, resource_avail in self.availability.items():
                availability[resource_type.value] = {
                    "total": resource_avail.total,
                    "available": resource_avail.available,
                    "reserved": resource_avail.reserved
                }
            return availability
    
    def get_component_allocations(self, component_id: str) -> List[Dict[str, Any]]:
        """
        Get all resource allocations for a component.
        
        Args:
            component_id: ID of the component
            
        Returns:
            List of allocation information
        """
        with self.lock:
            if component_id not in self.allocations:
                return []
            
            allocations = []
            for allocation in self.allocations[component_id]:
                allocations.append({
                    "resource_type": allocation.resource_type.value,
                    "amount": allocation.amount,
                    "priority": allocation.priority,
                    "timestamp": allocation.timestamp,
                    "expiration": allocation.expiration
                })
            return allocations
    
    def get_all_allocations(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all resource allocations.
        
        Returns:
            Dictionary mapping component IDs to allocation information
        """
        with self.lock:
            all_allocations = {}
            for component_id, allocations in self.allocations.items():
                component_allocations = []
                for allocation in allocations:
                    component_allocations.append({
                        "resource_type": allocation.resource_type.value,
                        "amount": allocation.amount,
                        "priority": allocation.priority,
                        "timestamp": allocation.timestamp,
                        "expiration": allocation.expiration
                    })
                all_allocations[component_id] = component_allocations
            return all_allocations
    
    def set_resource_limit(self, resource_type: ResourceType, limit: float) -> None:
        """
        Set the limit for a resource.
        
        Args:
            resource_type: Type of resource
            limit: Maximum amount of the resource
        """
        with self.lock:
            old_limit = self.resource_limits.get(resource_type, 0.0)
            self.resource_limits[resource_type] = limit
            
            # Update availability
            if resource_type in self.availability:
                diff = limit - old_limit
                self.availability[resource_type].total = limit
                self.availability[resource_type].available += diff
            else:
                self.availability[resource_type] = ResourceAvailability(
                    resource_type=resource_type,
                    total=limit,
                    available=limit,
                    reserved=0.0
                )
            
            self.logger.info(f"Set {resource_type.value} limit to {limit}")
    
    def get_resource_limit(self, resource_type: ResourceType) -> float:
        """
        Get the limit for a resource.
        
        Args:
            resource_type: Type of resource
            
        Returns:
            The resource limit
        """
        with self.lock:
            return self.resource_limits.get(resource_type, 0.0)
    
    def get_all_resource_limits(self) -> Dict[str, float]:
        """
        Get all resource limits.
        
        Returns:
            Dictionary mapping resource types to limits
        """
        with self.lock:
            return {rt.value: limit for rt, limit in self.resource_limits.items()}
    
    def check_resources_available(
        self, 
        requests: Dict[ResourceType, float]
    ) -> bool:
        """
        Check if requested resources are available.
        
        Args:
            requests: Dictionary mapping resource types to requested amounts
            
        Returns:
            True if all requested resources are available
        """
        with self.lock:
            for resource_type, amount in requests.items():
                if resource_type not in self.availability:
                    return False
                
                if amount > self.availability[resource_type].available:
                    return False
            
            return True
    
    def get_availability_estimate(
        self, 
        requests: Dict[ResourceType, float]
    ) -> Dict[str, Any]:
        """
        Get an estimate of when requested resources will be available.
        
        Args:
            requests: Dictionary mapping resource types to requested amounts
            
        Returns:
            Dictionary with availability information
        """
        with self.lock:
            result = {
                "all_available_now": True,
                "unavailable_resources": {},
                "estimated_wait_time": 0.0
            }
            
            for resource_type, amount in requests.items():
                if resource_type not in self.availability:
                    result["all_available_now"] = False
                    result["unavailable_resources"][resource_type.value] = {
                        "requested": amount,
                        "available": 0.0,
                        "reason": "Unknown resource type"
                    }
                    continue
                
                if amount > self.availability[resource_type].available:
                    result["all_available_now"] = False
                    result["unavailable_resources"][resource_type.value] = {
                        "requested": amount,
                        "available": self.availability[resource_type].available,
                        "reason": "Insufficient resources"
                    }
                    
                    # Estimate wait time based on expiring allocations
                    wait_time = self._estimate_wait_time(resource_type, amount)
                    result["estimated_wait_time"] = max(
                        result["estimated_wait_time"], wait_time)
            
            return result
    
    def _estimate_wait_time(
        self, 
        resource_type: ResourceType,
        amount_needed: float
    ) -> float:
        """
        Estimate the wait time for a resource to become available.
        
        Args:
            resource_type: Type of resource
            amount_needed: Amount of the resource needed
            
        Returns:
            Estimated wait time in seconds
        """
        current_time = time.time()
        available_amount = self.availability[resource_type].available
        
        if amount_needed <= available_amount:
            return 0.0
        
        # Find all allocations for this resource type
        allocations_for_type = []
        for component_allocations in self.allocations.values():
            for allocation in component_allocations:
                if allocation.resource_type == resource_type:
                    allocations_for_type.append(allocation)
        
        # Sort by expiration time (None values last)
        allocations_for_type.sort(
            key=lambda a: a.expiration if a.expiration is not None else float('inf'))
        
        # Calculate how long until we have enough resources
        amount_accumulated = available_amount
        last_expiration = current_time
        
        for allocation in allocations_for_type:
            if allocation.expiration is None:
                continue
            
            amount_accumulated += allocation.amount
            last_expiration = allocation.expiration
            
            if amount_accumulated >= amount_needed:
                return max(0.0, last_expiration - current_time)
        
        # If we can't accumulate enough resources, return a large value
        return 3600.0  # 1 hour as a placeholder
    
    def get_resource_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get resource usage statistics.
        
        Returns:
            Dictionary with usage statistics for each resource type
        """
        with self.lock:
            stats = {}
            for resource_type, avail in self.availability.items():
                usage_percent = (avail.total - avail.available) / avail.total * 100.0 if avail.total > 0 else 0.0
                reserved_percent = avail.reserved / avail.total * 100.0 if avail.total > 0 else 0.0
                
                stats[resource_type.value] = {
                    "total": avail.total,
                    "available": avail.available,
                    "used": avail.total - avail.available,
                    "reserved": avail.reserved,
                    "usage_percent": usage_percent,
                    "reserved_percent": reserved_percent
                }
            
            return stats
    
    def optimize_allocations(self) -> bool:
        """
        Optimize resource allocations to maximize efficiency.
        
        This method attempts to redistribute resources based on component
        priorities and resource usage patterns.
        
        Returns:
            True if optimizations were made
        """
        with self.lock:
            # This is a placeholder for a more sophisticated optimization algorithm
            # In a real implementation, this would analyze usage patterns and
            # redistribute resources accordingly
            
            # For now, just check for and release any unused allocations
            changes_made = False
            
            for component_id, allocations in list(self.allocations.items()):
                for allocation in list(allocations):
                    # Check if the allocation is old and might be unused
                    if time.time() - allocation.timestamp > 300.0:  # 5 minutes
                        # This is just a simplistic example
                        # In a real implementation, we would check actual usage
                        
                        # For CPU resources, check actual usage
                        if allocation.resource_type == ResourceType.CPU:
                            try:
                                # This is a placeholder for component-specific CPU tracking
                                # In reality, this would be more sophisticated
                                unused = True
                                
                                if unused:
                                    allocations.remove(allocation)
                                    self._release_resource(allocation)
                                    changes_made = True
                                    
                                    self.logger.info(
                                        f"Released unused {allocation.resource_type.value} "
                                        f"allocation for {component_id}"
                                    )
                            except Exception as e:
                                self.logger.error(f"Error optimizing allocation: {e}")
            
            return changes_made
