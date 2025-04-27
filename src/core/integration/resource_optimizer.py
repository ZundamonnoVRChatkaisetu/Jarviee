"""
Resource Optimization Module for AI Technology Integration in Jarviee.

This module implements resource optimization strategies for AI technology
integrations, enabling efficient distribution of workloads between edge devices
and cloud resources. It addresses the computational cost challenges in integrating
multiple AI technologies.
"""

import json
import math
import threading
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..utils.event_bus import Event, EventBus
from ..utils.logger import Logger


class ResourceType(Enum):
    """Types of resources that can be managed."""
    CPU = "cpu"  # CPU cores or percentage
    GPU = "gpu"  # GPU cores or percentage
    MEMORY = "memory"  # RAM in GB
    STORAGE = "storage"  # Storage in GB
    NETWORK = "network"  # Network bandwidth in Mbps
    API_RATE = "api_rate"  # API calls per minute
    TOKENS = "tokens"  # LLM tokens per minute
    POWER = "power"  # Power consumption in watts
    COST = "cost"  # Financial cost in currency units


class ComputeLocation(Enum):
    """Locations where computation can be performed."""
    EDGE = "edge"  # Edge device (local)
    FOG = "fog"  # Fog computing (near edge)
    CLOUD = "cloud"  # Cloud computing (remote)
    HYBRID = "hybrid"  # Hybrid computing (both edge and cloud)


class OptimizationStrategy(Enum):
    """Strategies for resource optimization."""
    PERFORMANCE = "performance"  # Optimize for speed/performance
    EFFICIENCY = "efficiency"  # Optimize for resource efficiency
    COST = "cost"  # Optimize for financial cost
    RELIABILITY = "reliability"  # Optimize for reliability/redundancy
    POWER = "power"  # Optimize for power consumption
    PRIVACY = "privacy"  # Optimize for data privacy/security
    BALANCED = "balanced"  # Balanced approach


class DeviceProfile:
    """
    Profile of a computing device (edge, fog, cloud instance).
    """
    
    def __init__(self, 
                 device_id: str,
                 name: str,
                 location: ComputeLocation,
                 resources: Dict[ResourceType, float],
                 cost_per_hour: float,
                 capabilities: List[str],
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a device profile.
        
        Args:
            device_id: Unique identifier for this device
            name: Human-readable device name
            location: Compute location type
            resources: Available resources by type
            cost_per_hour: Financial cost per hour
            capabilities: List of capability identifiers
            metadata: Additional metadata
        """
        self.device_id = device_id
        self.name = name
        self.location = location
        self.resources = resources
        self.cost_per_hour = cost_per_hour
        self.capabilities = capabilities
        self.metadata = metadata or {}
        self.current_load: Dict[ResourceType, float] = {
            res_type: 0.0 for res_type in resources
        }
        self.online = True
        self.last_updated = time.time()
    
    def get_available_resources(self) -> Dict[ResourceType, float]:
        """
        Get available (remaining) resources.
        
        Returns:
            Dictionary mapping resource types to available amounts
        """
        return {
            res_type: max(0.0, amount - self.current_load.get(res_type, 0.0))
            for res_type, amount in self.resources.items()
        }
    
    def can_accommodate(self, requirements: Dict[ResourceType, float]) -> bool:
        """
        Check if this device can accommodate resource requirements.
        
        Args:
            requirements: Resource requirements
            
        Returns:
            True if requirements can be accommodated
        """
        available = self.get_available_resources()
        
        for res_type, amount in requirements.items():
            if res_type not in available or available[res_type] < amount:
                return False
        
        return True
    
    def allocate_resources(self, requirements: Dict[ResourceType, float]) -> bool:
        """
        Allocate resources on this device.
        
        Args:
            requirements: Resource requirements
            
        Returns:
            True if allocation was successful
        """
        if not self.can_accommodate(requirements):
            return False
        
        for res_type, amount in requirements.items():
            if res_type in self.current_load:
                self.current_load[res_type] += amount
            else:
                self.current_load[res_type] = amount
        
        self.last_updated = time.time()
        return True
    
    def release_resources(self, resources: Dict[ResourceType, float]) -> None:
        """
        Release allocated resources.
        
        Args:
            resources: Resources to release
        """
        for res_type, amount in resources.items():
            if res_type in self.current_load:
                self.current_load[res_type] = max(0.0, self.current_load[res_type] - amount)
        
        self.last_updated = time.time()
    
    def get_utilization(self) -> Dict[ResourceType, float]:
        """
        Get resource utilization ratios.
        
        Returns:
            Dictionary mapping resource types to utilization ratios (0.0-1.0)
        """
        return {
            res_type: min(1.0, self.current_load.get(res_type, 0.0) / amount)
            for res_type, amount in self.resources.items()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "location": self.location.value,
            "resources": {
                res_type.value: amount
                for res_type, amount in self.resources.items()
            },
            "current_load": {
                res_type.value: amount
                for res_type, amount in self.current_load.items()
            },
            "available_resources": {
                res_type.value: amount
                for res_type, amount in self.get_available_resources().items()
            },
            "utilization": {
                res_type.value: ratio
                for res_type, ratio in self.get_utilization().items()
            },
            "cost_per_hour": self.cost_per_hour,
            "capabilities": self.capabilities,
            "online": self.online,
            "last_updated": self.last_updated,
            "metadata": self.metadata
        }


class TaskResourceProfile:
    """
    Resource profile for a computational task.
    """
    
    def __init__(self, 
                 task_id: str,
                 task_type: str,
                 resource_requirements: Dict[ResourceType, float],
                 required_capabilities: List[str],
                 estimated_duration: float,
                 priority: int = 1,
                 deadline: Optional[float] = None,
                 data_transfer: Dict[str, float] = None,
                 location_constraints: Optional[List[ComputeLocation]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a task resource profile.
        
        Args:
            task_id: Unique identifier for this task
            task_type: Type of computational task
            resource_requirements: Resources required by type
            required_capabilities: Required device capabilities
            estimated_duration: Estimated duration in seconds
            priority: Task priority (higher values = higher priority)
            deadline: Optional deadline timestamp
            data_transfer: Data transfer requirements (in/out in MB)
            location_constraints: Allowable compute locations
            metadata: Additional metadata
        """
        self.task_id = task_id
        self.task_type = task_type
        self.resource_requirements = resource_requirements
        self.required_capabilities = required_capabilities
        self.estimated_duration = estimated_duration
        self.priority = priority
        self.deadline = deadline
        self.data_transfer = data_transfer or {"in": 0.0, "out": 0.0}
        self.location_constraints = location_constraints
        self.metadata = metadata or {}
        self.created_at = time.time()
    
    def is_critical(self) -> bool:
        """
        Check if this task is time-critical.
        
        Returns:
            True if task is critical
        """
        return (
            self.priority >= 3 or
            (self.deadline is not None and time.time() > self.deadline - 60)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "resource_requirements": {
                res_type.value: amount
                for res_type, amount in self.resource_requirements.items()
            },
            "required_capabilities": self.required_capabilities,
            "estimated_duration": self.estimated_duration,
            "priority": self.priority,
            "deadline": self.deadline,
            "data_transfer": self.data_transfer,
            "location_constraints": [
                loc.value for loc in self.location_constraints
            ] if self.location_constraints else None,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "is_critical": self.is_critical()
        }


class ResourceAllocation:
    """
    Allocation of resources to a task on a device.
    """
    
    def __init__(self, 
                 allocation_id: str,
                 task_id: str,
                 device_id: str,
                 resources: Dict[ResourceType, float],
                 start_time: float,
                 planned_end_time: float,
                 actual_end_time: Optional[float] = None,
                 status: str = "allocated"):
        """
        Initialize a resource allocation.
        
        Args:
            allocation_id: Unique identifier for this allocation
            task_id: ID of the task
            device_id: ID of the device
            resources: Resources allocated
            start_time: Allocation start time
            planned_end_time: Planned end time
            actual_end_time: Actual end time (when completed)
            status: Allocation status
        """
        self.allocation_id = allocation_id
        self.task_id = task_id
        self.device_id = device_id
        self.resources = resources
        self.start_time = start_time
        self.planned_end_time = planned_end_time
        self.actual_end_time = actual_end_time
        self.status = status  # allocated, running, completed, failed, canceled
    
    def complete(self) -> None:
        """Mark allocation as completed."""
        self.status = "completed"
        self.actual_end_time = time.time()
    
    def cancel(self) -> None:
        """Mark allocation as canceled."""
        self.status = "canceled"
        self.actual_end_time = time.time()
    
    def fail(self) -> None:
        """Mark allocation as failed."""
        self.status = "failed"
        self.actual_end_time = time.time()
    
    def is_active(self) -> bool:
        """
        Check if this allocation is still active.
        
        Returns:
            True if allocation is active
        """
        return self.status in ["allocated", "running"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "allocation_id": self.allocation_id,
            "task_id": self.task_id,
            "device_id": self.device_id,
            "resources": {
                res_type.value: amount
                for res_type, amount in self.resources.items()
            },
            "start_time": self.start_time,
            "planned_end_time": self.planned_end_time,
            "actual_end_time": self.actual_end_time,
            "status": self.status,
            "is_active": self.is_active()
        }


class ResourceOptimizer:
    """
    Resource optimization system for AI technology integrations.
    
    This class manages resource allocation and optimization across edge
    and cloud resources, enabling efficient distributed computing.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the resource optimizer.
        
        Args:
            event_bus: Optional event bus for communication
        """
        self.logger = Logger().get_logger("jarviee.integration.resource_optimizer")
        self.event_bus = event_bus
        
        # Device registry
        self.devices: Dict[str, DeviceProfile] = {}
        
        # Task profiles
        self.task_profiles: Dict[str, TaskResourceProfile] = {}
        
        # Resource allocations
        self.allocations: Dict[str, ResourceAllocation] = {}
        
        # Allocation history
        self.allocation_history: List[ResourceAllocation] = []
        self.max_history_size = 1000
        
        # Optimization strategy
        self.strategy = OptimizationStrategy.BALANCED
        
        # Resource utilization statistics
        self.utilization_stats: Dict[ComputeLocation, Dict[ResourceType, List[float]]] = {
            location: {resource: [] for resource in ResourceType}
            for location in ComputeLocation
        }
        self.max_stats_samples = 100
        
        # Configuration
        self.config = {
            "enable_automatic_optimization": True,
            "optimization_interval": 60,  # seconds
            "enable_load_balancing": True,
            "edge_preference_factor": 0.8,  # Preference for edge vs. cloud
            "cost_importance": 0.5,  # Importance of cost in decisions
            "privacy_importance": 0.5,  # Importance of privacy in decisions
            "performance_importance": 0.7,  # Importance of performance in decisions
            "power_importance": 0.4,  # Importance of power in decisions
            "task_timeout_margin": 300,  # seconds, margin for task timeout
            "task_resource_buffer": 0.2,  # 20% buffer for task resources
            "device_max_load": 0.9  # 90% max load per device
        }
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self.logger.info("Resource Optimizer initialized")
    
    def start_monitoring(self) -> bool:
        """
        Start resource monitoring thread.
        
        Returns:
            True if monitoring started successfully
        """
        if self.monitoring_active:
            return True
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("Resource monitoring started")
        return True
    
    def stop_monitoring(self) -> bool:
        """
        Stop resource monitoring thread.
        
        Returns:
            True if monitoring stopped successfully
        """
        if not self.monitoring_active:
            return True
        
        self.monitoring_active = False
        if self.monitoring_thread:
            try:
                self.monitoring_thread.join(timeout=2.0)
            except:
                pass
            
        self.logger.info("Resource monitoring stopped")
        return True
    
    def register_device(self, device: DeviceProfile) -> bool:
        """
        Register a computing device.
        
        Args:
            device: Device profile to register
            
        Returns:
            True if registration was successful
        """
        with self.lock:
            if device.device_id in self.devices:
                self.logger.warning(f"Device {device.device_id} already registered")
                return False
            
            self.devices[device.device_id] = device
            self.logger.info(f"Registered device {device.device_id} ({device.name})")
            
            # Emit event if available
            if self.event_bus:
                self.event_bus.publish(Event(
                    "resource.device_registered",
                    {"device_id": device.device_id, "location": device.location.value}
                ))
            
            return True
    
    def unregister_device(self, device_id: str) -> bool:
        """
        Unregister a computing device.
        
        Args:
            device_id: ID of device to unregister
            
        Returns:
            True if unregistration was successful
        """
        with self.lock:
            if device_id not in self.devices:
                self.logger.warning(f"Device {device_id} not registered")
                return False
            
            # Check for active allocations
            active_allocations = [
                a for a in self.allocations.values()
                if a.device_id == device_id and a.is_active()
            ]
            
            if active_allocations:
                self.logger.warning(
                    f"Cannot unregister device {device_id} with {len(active_allocations)} "
                    "active allocations"
                )
                return False
            
            # Remove device
            device = self.devices.pop(device_id)
            self.logger.info(f"Unregistered device {device_id} ({device.name})")
            
            # Emit event if available
            if self.event_bus:
                self.event_bus.publish(Event(
                    "resource.device_unregistered",
                    {"device_id": device_id, "location": device.location.value}
                ))
            
            return True
    
    def update_device_status(
        self, 
        device_id: str,
        online: Optional[bool] = None,
        current_load: Optional[Dict[ResourceType, float]] = None,
        resources: Optional[Dict[ResourceType, float]] = None
    ) -> bool:
        """
        Update device status.
        
        Args:
            device_id: ID of device to update
            online: Whether device is online
            current_load: Current resource load
            resources: Available resources
            
        Returns:
            True if update was successful
        """
        with self.lock:
            if device_id not in self.devices:
                self.logger.warning(f"Device {device_id} not registered")
                return False
            
            device = self.devices[device_id]
            
            # Update online status
            if online is not None:
                was_online = device.online
                device.online = online
                
                if was_online and not online:
                    # Device went offline, handle active allocations
                    self._handle_device_offline(device_id)
            
            # Update current load
            if current_load:
                for res_type, amount in current_load.items():
                    device.current_load[res_type] = amount
            
            # Update available resources
            if resources:
                for res_type, amount in resources.items():
                    device.resources[res_type] = amount
            
            device.last_updated = time.time()
            
            # Emit event if available
            if self.event_bus:
                self.event_bus.publish(Event(
                    "resource.device_updated",
                    {"device_id": device_id, "online": device.online}
                ))
            
            return True
    
    def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        """
        Get device information.
        
        Args:
            device_id: ID of device to get
            
        Returns:
            Device dictionary, or None if not found
        """
        with self.lock:
            if device_id not in self.devices:
                return None
            
            return self.devices[device_id].to_dict()
    
    def list_devices(
        self, 
        location: Optional[ComputeLocation] = None,
        online_only: bool = True,
        with_capability: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List registered devices.
        
        Args:
            location: Filter by compute location
            online_only: Only include online devices
            with_capability: Filter by capability
            
        Returns:
            List of device dictionaries
        """
        with self.lock:
            results = []
            
            for device in self.devices.values():
                # Apply filters
                if online_only and not device.online:
                    continue
                    
                if location and device.location != location:
                    continue
                    
                if with_capability and with_capability not in device.capabilities:
                    continue
                
                # Add to results
                results.append(device.to_dict())
            
            return results
    
    def register_task(self, task: TaskResourceProfile) -> bool:
        """
        Register a computational task.
        
        Args:
            task: Task profile to register
            
        Returns:
            True if registration was successful
        """
        with self.lock:
            if task.task_id in self.task_profiles:
                self.logger.warning(f"Task {task.task_id} already registered")
                return False
            
            self.task_profiles[task.task_id] = task
            self.logger.info(
                f"Registered task {task.task_id} (type: {task.task_type}, "
                f"priority: {task.priority})"
            )
            
            # Emit event if available
            if self.event_bus:
                self.event_bus.publish(Event(
                    "resource.task_registered",
                    {"task_id": task.task_id, "task_type": task.task_type}
                ))
            
            return True
    
    def unregister_task(self, task_id: str) -> bool:
        """
        Unregister a computational task.
        
        Args:
            task_id: ID of task to unregister
            
        Returns:
            True if unregistration was successful
        """
        with self.lock:
            if task_id not in self.task_profiles:
                self.logger.warning(f"Task {task_id} not registered")
                return False
            
            # Check for active allocations
            active_allocations = [
                a for a in self.allocations.values()
                if a.task_id == task_id and a.is_active()
            ]
            
            if active_allocations:
                self.logger.warning(
                    f"Cannot unregister task {task_id} with {len(active_allocations)} "
                    "active allocations"
                )
                return False
            
            # Remove task
            task = self.task_profiles.pop(task_id)
            self.logger.info(f"Unregistered task {task_id} (type: {task.task_type})")
            
            # Emit event if available
            if self.event_bus:
                self.event_bus.publish(Event(
                    "resource.task_unregistered",
                    {"task_id": task_id}
                ))
            
            return True
    
    def allocate_resources(
        self, 
        task_id: str,
        preferred_location: Optional[ComputeLocation] = None,
        preferred_device_id: Optional[str] = None,
        duration: Optional[float] = None
    ) -> Optional[str]:
        """
        Allocate resources for a task.
        
        Args:
            task_id: ID of task to allocate resources for
            preferred_location: Preferred compute location
            preferred_device_id: Preferred device ID
            duration: Allocation duration in seconds
            
        Returns:
            Allocation ID if successful, or None if allocation failed
        """
        with self.lock:
            if task_id not in self.task_profiles:
                self.logger.warning(f"Task {task_id} not registered")
                return None
            
            task = self.task_profiles[task_id]
            
            # Check for existing allocations
            existing_allocations = [
                a for a in self.allocations.values()
                if a.task_id == task_id and a.is_active()
            ]
            
            if existing_allocations:
                self.logger.warning(
                    f"Task {task_id} already has {len(existing_allocations)} "
                    "active allocations"
                )
                return existing_allocations[0].allocation_id
            
            # Set duration if not provided
            if duration is None:
                duration = task.estimated_duration + self.config["task_timeout_margin"]
            
            # Apply resource buffer
            buffered_requirements = {
                res_type: amount * (1 + self.config["task_resource_buffer"])
                for res_type, amount in task.resource_requirements.items()
            }
            
            # Find suitable device
            device_id = None
            
            # Try preferred device first
            if preferred_device_id and preferred_device_id in self.devices:
                device = self.devices[preferred_device_id]
                if (device.online and
                    device.can_accommodate(buffered_requirements) and
                    all(cap in device.capabilities for cap in task.required_capabilities) and
                    (not task.location_constraints or device.location in task.location_constraints)):
                    device_id = preferred_device_id
            
            # If no preferred device or it's not suitable, find the best device
            if not device_id:
                device_id = self._find_optimal_device(
                    task, preferred_location or ComputeLocation.EDGE)
            
            if not device_id:
                self.logger.warning(
                    f"No suitable device found for task {task_id}")
                return None
            
            # Get device
            device = self.devices[device_id]
            
            # Create allocation
            allocation_id = str(uuid.uuid4())
            allocation = ResourceAllocation(
                allocation_id=allocation_id,
                task_id=task_id,
                device_id=device_id,
                resources=buffered_requirements,
                start_time=time.time(),
                planned_end_time=time.time() + duration
            )
            
            # Allocate resources on the device
            if not device.allocate_resources(buffered_requirements):
                self.logger.warning(
                    f"Failed to allocate resources on device {device_id} for task {task_id}")
                return None
            
            # Store allocation
            self.allocations[allocation_id] = allocation
            
            self.logger.info(
                f"Allocated resources on device {device_id} ({device.name}) "
                f"for task {task_id} (allocation {allocation_id})"
            )
            
            # Emit event if available
            if self.event_bus:
                self.event_bus.publish(Event(
                    "resource.resources_allocated",
                    {
                        "allocation_id": allocation_id,
                        "task_id": task_id,
                        "device_id": device_id,
                        "location": device.location.value
                    }
                ))
            
            # Update utilization statistics
            self._update_utilization_stats()
            
            return allocation_id
    
    def release_resources(self, allocation_id: str, status: str = "completed") -> bool:
        """
        Release allocated resources.
        
        Args:
            allocation_id: ID of allocation to release
            status: Completion status (completed, failed, canceled)
            
        Returns:
            True if release was successful
        """
        with self.lock:
            if allocation_id not in self.allocations:
                self.logger.warning(f"Allocation {allocation_id} not found")
                return False
            
            allocation = self.allocations[allocation_id]
            
            if not allocation.is_active():
                self.logger.warning(f"Allocation {allocation_id} is not active")
                return False
            
            # Get device
            device_id = allocation.device_id
            if device_id not in self.devices:
                self.logger.warning(f"Device {device_id} not found for allocation {allocation_id}")
                return False
            
            device = self.devices[device_id]
            
            # Release resources
            device.release_resources(allocation.resources)
            
            # Update allocation status
            if status == "completed":
                allocation.complete()
            elif status == "failed":
                allocation.fail()
            else:
                allocation.cancel()
            
            # Move to history
            self.allocation_history.append(allocation)
            if len(self.allocation_history) > self.max_history_size:
                self.allocation_history = self.allocation_history[-self.max_history_size:]
            
            self.logger.info(
                f"Released resources on device {device_id} ({device.name}) "
                f"for task {allocation.task_id} (allocation {allocation_id})"
            )
            
            # Emit event if available
            if self.event_bus:
                self.event_bus.publish(Event(
                    "resource.resources_released",
                    {
                        "allocation_id": allocation_id,
                        "task_id": allocation.task_id,
                        "device_id": device_id,
                        "status": status
                    }
                ))
            
            # Update utilization statistics
            self._update_utilization_stats()
            
            return True
    
    def get_allocation(self, allocation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get allocation information.
        
        Args:
            allocation_id: ID of allocation to get
            
        Returns:
            Allocation dictionary, or None if not found
        """
        with self.lock:
            if allocation_id in self.allocations:
                return self.allocations[allocation_id].to_dict()
            
            # Check history
            for allocation in self.allocation_history:
                if allocation.allocation_id == allocation_id:
                    return allocation.to_dict()
            
            return None
    
    def get_task_allocations(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get allocations for a task.
        
        Args:
            task_id: ID of task to get allocations for
            
        Returns:
            List of allocation dictionaries
        """
        with self.lock:
            # Find active allocations
            active_allocations = [
                a.to_dict() for a in self.allocations.values()
                if a.task_id == task_id
            ]
            
            # Find historical allocations
            historical_allocations = [
                a.to_dict() for a in self.allocation_history
                if a.task_id == task_id
            ]
            
            return active_allocations + historical_allocations
    
    def get_device_allocations(self, device_id: str) -> List[Dict[str, Any]]:
        """
        Get allocations for a device.
        
        Args:
            device_id: ID of device to get allocations for
            
        Returns:
            List of allocation dictionaries
        """
        with self.lock:
            # Find active allocations
            active_allocations = [
                a.to_dict() for a in self.allocations.values()
                if a.device_id == device_id
            ]
            
            # Find historical allocations
            historical_allocations = [
                a.to_dict() for a in self.allocation_history
                if a.device_id == device_id
            ]
            
            return active_allocations + historical_allocations
    
    def get_resource_usage_stats(self) -> Dict[str, Any]:
        """
        Get resource usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        with self.lock:
            # Calculate total resources by location
            total_resources = {
                location.value: {res_type.value: 0.0 for res_type in ResourceType}
                for location in ComputeLocation
            }
            
            # Calculate used resources by location
            used_resources = {
                location.value: {res_type.value: 0.0 for res_type in ResourceType}
                for location in ComputeLocation
            }
            
            # Process each device
            for device in self.devices.values():
                if not device.online:
                    continue
                
                location = device.location.value
                
                for res_type, amount in device.resources.items():
                    res_key = res_type.value
                    total_resources[location][res_key] += amount
                    used_resources[location][res_key] += device.current_load.get(res_type, 0.0)
            
            # Calculate utilization ratios
            utilization_ratios = {}
            for location in ComputeLocation:
                loc_key = location.value
                utilization_ratios[loc_key] = {}
                
                for res_type in ResourceType:
                    res_key = res_type.value
                    total = total_resources[loc_key][res_key]
                    used = used_resources[loc_key][res_key]
                    
                    if total > 0:
                        utilization_ratios[loc_key][res_key] = used / total
                    else:
                        utilization_ratios[loc_key][res_key] = 0.0
            
            # Count devices by location
            device_counts = {
                location.value: sum(1 for d in self.devices.values() 
                                  if d.location == location and d.online)
                for location in ComputeLocation
            }
            
            # Count active allocations by location
            allocation_counts = {
                location.value: sum(1 for a in self.allocations.values()
                                   if a.is_active() and
                                   self.devices.get(a.device_id) and
                                   self.devices[a.device_id].location == location)
                for location in ComputeLocation
            }
            
            # Get historical trends
            utilization_trends = {
                location.value: {
                    res_type.value: self.utilization_stats[location][res_type][-10:]
                    for res_type in ResourceType
                    if self.utilization_stats[location][res_type]
                }
                for location in ComputeLocation
            }
            
            return {
                "total_resources": total_resources,
                "used_resources": used_resources,
                "utilization_ratios": utilization_ratios,
                "device_counts": device_counts,
                "allocation_counts": allocation_counts,
                "active_allocations": len(self.allocations),
                "active_tasks": len(set(a.task_id for a in self.allocations.values() if a.is_active())),
                "utilization_trends": utilization_trends,
                "timestamp": time.time()
            }
    
    def optimize_allocations(self) -> bool:
        """
        Optimize current resource allocations.
        
        Returns:
            True if optimization was performed
        """
        with self.lock:
            # Get active allocations
            active_allocations = [a for a in self.allocations.values() if a.is_active()]
            
            if not active_allocations:
                self.logger.info("No active allocations to optimize")
                return False
            
            # Group allocations by device
            device_allocations = {}
            for allocation in active_allocations:
                device_id = allocation.device_id
                if device_id not in device_allocations:
                    device_allocations[device_id] = []
                device_allocations[device_id].append(allocation)
            
            # Check for overloaded devices
            overloaded_devices = []
            for device_id, allocations in device_allocations.items():
                if device_id not in self.devices:
                    continue
                
                device = self.devices[device_id]
                utilization = device.get_utilization()
                
                # Check if any resource is overutilized
                for res_type, ratio in utilization.items():
                    if ratio > self.config["device_max_load"]:
                        overloaded_devices.append(device_id)
                        break
            
            if not overloaded_devices:
                self.logger.info("No overloaded devices to optimize")
                return False
            
            # Get tasks for allocations on overloaded devices
            tasks_to_reallocate = []
            for device_id in overloaded_devices:
                allocations = device_allocations.get(device_id, [])
                for allocation in allocations:
                    task_id = allocation.task_id
                    if task_id in self.task_profiles:
                        task = self.task_profiles[task_id]
                        tasks_to_reallocate.append((task, allocation))
            
            if not tasks_to_reallocate:
                self.logger.info("No tasks to reallocate")
                return False
            
            # Sort tasks by priority (lowest first, to move non-critical tasks)
            tasks_to_reallocate.sort(key=lambda x: x[0].priority)
            
            # Attempt to reallocate a few tasks
            changes_made = False
            
            for task, allocation in tasks_to_reallocate[:3]:  # Limit to 3 tasks per round
                # Find a better device
                current_device_id = allocation.device_id
                current_device = self.devices.get(current_device_id)
                
                if not current_device:
                    continue
                
                # Try to find a less loaded device
                new_device_id = self._find_optimal_device(
                    task, current_device.location, exclude_device_id=current_device_id)
                
                if not new_device_id:
                    continue
                
                # Create new allocation
                new_allocation_id = str(uuid.uuid4())
                new_allocation = ResourceAllocation(
                    allocation_id=new_allocation_id,
                    task_id=task.task_id,
                    device_id=new_device_id,
                    resources=allocation.resources,
                    start_time=time.time(),
                    planned_end_time=allocation.planned_end_time
                )
                
                # Get new device
                new_device = self.devices[new_device_id]
                
                # Allocate resources on new device
                if not new_device.allocate_resources(allocation.resources):
                    continue
                
                # Store new allocation
                self.allocations[new_allocation_id] = new_allocation
                
                # Release resources on old device
                current_device.release_resources(allocation.resources)
                
                # Cancel old allocation
                allocation.cancel()
                
                # Move to history
                self.allocation_history.append(allocation)
                
                self.logger.info(
                    f"Reallocated task {task.task_id} from device {current_device_id} "
                    f"to {new_device_id} (old allocation: {allocation.allocation_id}, "
                    f"new allocation: {new_allocation_id})"
                )
                
                # Emit event if available
                if self.event_bus:
                    self.event_bus.publish(Event(
                        "resource.allocation_optimized",
                        {
                            "task_id": task.task_id,
                            "old_device_id": current_device_id,
                            "new_device_id": new_device_id,
                            "old_allocation_id": allocation.allocation_id,
                            "new_allocation_id": new_allocation_id
                        }
                    ))
                
                changes_made = True
            
            # Update utilization statistics
            if changes_made:
                self._update_utilization_stats()
            
            return changes_made
    
    def set_optimization_strategy(self, strategy: OptimizationStrategy) -> None:
        """
        Set the optimization strategy.
        
        Args:
            strategy: New optimization strategy
        """
        self.strategy = strategy
        self.logger.info(f"Set optimization strategy to {strategy.value}")
        
        # Adjust configuration based on strategy
        if strategy == OptimizationStrategy.PERFORMANCE:
            self.config["edge_preference_factor"] = 0.3
            self.config["cost_importance"] = 0.3
            self.config["performance_importance"] = 0.9
            self.config["power_importance"] = 0.2
            
        elif strategy == OptimizationStrategy.EFFICIENCY:
            self.config["edge_preference_factor"] = 0.7
            self.config["cost_importance"] = 0.7
            self.config["performance_importance"] = 0.5
            self.config["power_importance"] = 0.8
            
        elif strategy == OptimizationStrategy.COST:
            self.config["edge_preference_factor"] = 0.5
            self.config["cost_importance"] = 0.9
            self.config["performance_importance"] = 0.4
            self.config["power_importance"] = 0.6
            
        elif strategy == OptimizationStrategy.PRIVACY:
            self.config["edge_preference_factor"] = 0.9
            self.config["privacy_importance"] = 0.9
            
        elif strategy == OptimizationStrategy.BALANCED:
            self.config["edge_preference_factor"] = 0.6
            self.config["cost_importance"] = 0.5
            self.config["privacy_importance"] = 0.5
            self.config["performance_importance"] = 0.6
            self.config["power_importance"] = 0.5
    
    def set_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update configuration settings.
        
        Args:
            config_updates: Dictionary with configuration updates
        """
        self.config.update(config_updates)
        self.logger.info(f"Updated configuration: {config_updates}")
    
    def _find_optimal_device(
        self, 
        task: TaskResourceProfile,
        preferred_location: ComputeLocation,
        exclude_device_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Find the optimal device for a task.
        
        Args:
            task: Task resource profile
            preferred_location: Preferred compute location
            exclude_device_id: Device ID to exclude
            
        Returns:
            ID of the optimal device, or None if no suitable device found
        """
        # Filter devices by capabilities and resources
        candidates = []
        
        for device_id, device in self.devices.items():
            # Skip offline devices
            if not device.online:
                continue
                
            # Skip excluded device
            if exclude_device_id and device_id == exclude_device_id:
                continue
                
            # Check location constraints
            if (task.location_constraints and 
                device.location not in task.location_constraints):
                continue
                
            # Check capabilities
            if not all(cap in device.capabilities for cap in task.required_capabilities):
                continue
                
            # Check resources
            if not device.can_accommodate(task.resource_requirements):
                continue
                
            # Add to candidates
            candidates.append(device)
        
        if not candidates:
            return None
        
        # Score candidates based on strategy
        scores = {}
        
        for device in candidates:
            # Base score
            score = 0.0
            
            # Location preference
            if device.location == preferred_location:
                score += self.config["edge_preference_factor"]
            
            # Resource availability
            available = device.get_available_resources()
            for res_type, req_amount in task.resource_requirements.items():
                if res_type in available:
                    avail_amount = available[res_type]
                    if avail_amount > 0:
                        # Higher score for more available resources
                        ratio = min(1.0, (avail_amount - req_amount) / avail_amount)
                        score += ratio * 0.5
            
            # Cost factor
            if self.strategy in [OptimizationStrategy.COST, OptimizationStrategy.BALANCED]:
                if device.cost_per_hour > 0:
                    # Lower score for higher cost
                    cost_factor = self.config["cost_importance"] * (1.0 - min(1.0, device.cost_per_hour / 10.0))
                    score += cost_factor
            
            # Privacy factor
            if self.strategy in [OptimizationStrategy.PRIVACY, OptimizationStrategy.BALANCED]:
                # Higher score for edge (more private)
                if device.location == ComputeLocation.EDGE:
                    score += self.config["privacy_importance"]
                elif device.location == ComputeLocation.FOG:
                    score += self.config["privacy_importance"] * 0.7
                elif device.location == ComputeLocation.CLOUD:
                    score += self.config["privacy_importance"] * 0.3
            
            # Performance factor
            if self.strategy in [OptimizationStrategy.PERFORMANCE, OptimizationStrategy.BALANCED]:
                # This would use device performance metrics in a real implementation
                if device.location == ComputeLocation.CLOUD:
                    score += self.config["performance_importance"] * 0.8
                elif device.location == ComputeLocation.FOG:
                    score += self.config["performance_importance"] * 0.5
                elif device.location == ComputeLocation.EDGE:
                    score += self.config["performance_importance"] * 0.3
            
            # Store score
            scores[device.device_id] = score
        
        # Find device with highest score
        if not scores:
            return None
            
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _handle_device_offline(self, device_id: str) -> None:
        """
        Handle a device going offline.
        
        Args:
            device_id: ID of device that went offline
        """
        # Find active allocations on this device
        device_allocations = [
            a for a in self.allocations.values()
            if a.device_id == device_id and a.is_active()
        ]
        
        if not device_allocations:
            return
            
        self.logger.info(
            f"Device {device_id} went offline with {len(device_allocations)} "
            "active allocations, attempting to reallocate"
        )
        
        # Try to reallocate each task
        for allocation in device_allocations:
            task_id = allocation.task_id
            
            if task_id not in self.task_profiles:
                # Can't reallocate without task profile
                allocation.fail()
                continue
                
            task = self.task_profiles[task_id]
            
            # Try to find another device
            new_device_id = self._find_optimal_device(
                task, ComputeLocation.EDGE, exclude_device_id=device_id)
                
            if not new_device_id:
                # No suitable device found
                allocation.fail()
                continue
                
            # Create new allocation
            new_allocation_id = str(uuid.uuid4())
            new_allocation = ResourceAllocation(
                allocation_id=new_allocation_id,
                task_id=task_id,
                device_id=new_device_id,
                resources=allocation.resources,
                start_time=time.time(),
                planned_end_time=allocation.planned_end_time
            )
            
            # Get new device
            new_device = self.devices[new_device_id]
            
            # Allocate resources on new device
            if not new_device.allocate_resources(allocation.resources):
                allocation.fail()
                continue
                
            # Store new allocation
            self.allocations[new_allocation_id] = new_allocation
            
            # Mark old allocation as failed
            allocation.fail()
            
            # Move to history
            self.allocation_history.append(allocation)
            
            self.logger.info(
                f"Reallocated task {task_id} from offline device {device_id} "
                f"to {new_device_id} (old allocation: {allocation.allocation_id}, "
                f"new allocation: {new_allocation_id})"
            )
            
            # Emit event if available
            if self.event_bus:
                self.event_bus.publish(Event(
                    "resource.task_reallocated",
                    {
                        "task_id": task_id,
                        "old_device_id": device_id,
                        "new_device_id": new_device_id,
                        "old_allocation_id": allocation.allocation_id,
                        "new_allocation_id": new_allocation_id,
                        "reason": "device_offline"
                    }
                ))
    
    def _update_utilization_stats(self) -> None:
        """Update resource utilization statistics."""
        # Calculate utilization by location and resource type
        for location in ComputeLocation:
            for resource in ResourceType:
                # Get total and used resources
                total = 0.0
                used = 0.0
                
                for device in self.devices.values():
                    if device.location == location and device.online:
                        if resource in device.resources:
                            total += device.resources[resource]
                            used += device.current_load.get(resource, 0.0)
                
                # Calculate utilization ratio
                if total > 0:
                    ratio = used / total
                else:
                    ratio = 0.0
                
                # Add to stats
                self.utilization_stats[location][resource].append(ratio)
                
                # Trim if needed
                if len(self.utilization_stats[location][resource]) > self.max_stats_samples:
                    self.utilization_stats[location][resource] = \
                        self.utilization_stats[location][resource][-self.max_stats_samples:]
    
    def _monitoring_loop(self) -> None:
        """Resource monitoring thread loop."""
        last_optimization_time = time.time()
        
        while self.monitoring_active:
            try:
                # Check for expired allocations
                with self.lock:
                    now = time.time()
                    
                    expired_allocations = [
                        a for a in self.allocations.values()
                        if a.is_active() and a.planned_end_time < now
                    ]
                    
                    for allocation in expired_allocations:
                        self.logger.warning(
                            f"Allocation {allocation.allocation_id} for task "
                            f"{allocation.task_id} expired, releasing resources"
                        )
                        self.release_resources(allocation.allocation_id, "expired")
                    
                    # Check for automatic optimization
                    if (self.config["enable_automatic_optimization"] and
                        now - last_optimization_time > self.config["optimization_interval"]):
                        self.optimize_allocations()
                        last_optimization_time = now
                    
                    # Update utilization statistics
                    self._update_utilization_stats()
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
            
            # Sleep
            time.sleep(10)
