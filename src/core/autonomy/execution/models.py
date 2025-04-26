"""
Plan Execution Models for Autonomous Action Engine.

This module defines the data models for plan execution in the autonomy system.
It includes classes for representing execution state, resource allocation,
and execution context.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Union, Any, Tuple, Callable
import uuid

from ..planning.models import Plan, PlanStep, StepResult


class ExecutionStatus(Enum):
    """Status values for execution state."""
    CREATED = "created"          # Execution has been created but not started
    PREPARING = "preparing"      # Preparing for execution
    RUNNING = "running"          # Execution is in progress
    PAUSED = "paused"            # Execution is temporarily paused
    CANCELING = "canceling"      # Execution is being canceled
    CANCELED = "canceled"        # Execution was canceled
    COMPLETED = "completed"      # Execution completed successfully
    FAILED = "failed"            # Execution failed


class ExecutionMode(Enum):
    """Modes for plan execution."""
    AUTOMATIC = "automatic"      # Fully automated execution
    SEMI_AUTO = "semi_auto"      # Semi-automated with confirmation points
    MANUAL = "manual"            # Manual execution with step-by-step approval
    SIMULATION = "simulation"    # Simulation mode (no real actions)


class StepOutcome(Enum):
    """Outcome values for step execution."""
    SUCCESS = "success"          # Step completed successfully
    FAILURE = "failure"          # Step failed but in an expected way
    ERROR = "error"              # Unexpected error occurred
    TIMEOUT = "timeout"          # Step timed out
    CANCELED = "canceled"        # Step was canceled
    SKIPPED = "skipped"          # Step was skipped


class ResourceAllocationStatus(Enum):
    """Status values for resource allocation."""
    REQUESTED = "requested"      # Resource has been requested
    PENDING = "pending"          # Resource request is being processed
    ALLOCATED = "allocated"      # Resource has been allocated
    FAILED = "failed"            # Resource allocation failed
    RELEASED = "released"        # Resource has been released


@dataclass
class ResourceRequirement:
    """Requirement for a specific resource."""
    resource_type: str
    identifier: Optional[str] = None
    quantity: Optional[float] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    is_critical: bool = False
    alternative_resources: List[str] = field(default_factory=list)


@dataclass
class ResourceAllocation:
    """Representation of an allocated resource."""
    allocation_id: str
    resource_type: str
    requester_id: str
    status: ResourceAllocationStatus
    resource_id: Optional[str] = None
    quantity: Optional[float] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    allocation_time: Optional[datetime] = None
    release_time: Optional[datetime] = None
    
    @classmethod
    def create(cls, resource_type: str, requester_id: str, 
               quantity: Optional[float] = None, constraints: Dict[str, Any] = None,
               status: ResourceAllocationStatus = ResourceAllocationStatus.REQUESTED) -> 'ResourceAllocation':
        """
        Create a new resource allocation with a generated ID.
        
        Args:
            resource_type: Type of resource to allocate
            requester_id: ID of the entity requesting the resource
            quantity: Optional quantity to allocate
            constraints: Optional constraints on the allocation
            status: Initial status of the allocation
            
        Returns:
            A new ResourceAllocation instance
        """
        allocation_id = str(uuid.uuid4())
        
        return cls(
            allocation_id=allocation_id,
            resource_type=resource_type,
            requester_id=requester_id,
            status=status,
            quantity=quantity,
            constraints=constraints or {}
        )


@dataclass
class LogEntry:
    """Entry in an execution log."""
    timestamp: datetime
    level: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Context information for execution."""
    variables: Dict[str, Any] = field(default_factory=dict)
    env_variables: Dict[str, str] = field(default_factory=dict)
    credentials: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None  # Global timeout in seconds
    retry_limit: int = 3
    available_executors: Dict[str, str] = field(default_factory=dict)  # action_type -> executor_id
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepExecution:
    """
    Representation of a step execution.
    """
    execution_id: str
    step: PlanStep
    status: ExecutionStatus = ExecutionStatus.CREATED
    outcome: Optional[StepOutcome] = None
    progress: float = 0.0  # 0.0-1.0
    result: Optional[StepResult] = None
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Execution details
    executor_id: Optional[str] = None
    resource_allocation_ids: List[str] = field(default_factory=list)
    logs: List[LogEntry] = field(default_factory=list)
    
    # Control
    cancellation_requested: bool = False
    retry_count: int = 0
    
    # Context
    context: ExecutionContext = field(default_factory=ExecutionContext)
    
    @classmethod
    def create(cls, step: PlanStep, 
               context: Optional[ExecutionContext] = None) -> 'StepExecution':
        """
        Create a new step execution with a generated ID.
        
        Args:
            step: The step to execute
            context: Optional execution context
            
        Returns:
            A new StepExecution instance
        """
        execution_id = str(uuid.uuid4())
        
        return cls(
            execution_id=execution_id,
            step=step,
            context=context or ExecutionContext()
        )
    
    def add_log(self, level: str, message: str, details: Dict[str, Any] = None) -> None:
        """
        Add a log entry to this execution.
        
        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: Log message
            details: Optional additional details
        """
        self.logs.append(LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            details=details or {}
        ))


@dataclass
class PlanExecution:
    """
    Representation of a plan execution.
    """
    execution_id: str
    plan: Plan
    mode: ExecutionMode
    status: ExecutionStatus = ExecutionStatus.CREATED
    progress: float = 0.0  # 0.0-1.0
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Execution details
    step_executions: Dict[str, StepExecution] = field(default_factory=dict)  # step_id -> execution
    logs: List[LogEntry] = field(default_factory=list)
    evaluation_result: Optional[Dict[str, Any]] = None
    
    # Control
    cancellation_requested: bool = False
    paused_at_step_id: Optional[str] = None
    
    # Context
    execution_context: ExecutionContext = field(default_factory=ExecutionContext)
    
    @classmethod
    def create(cls, plan: Plan, 
               mode: ExecutionMode = ExecutionMode.AUTOMATIC,
               context: Optional[ExecutionContext] = None) -> 'PlanExecution':
        """
        Create a new plan execution with a generated ID.
        
        Args:
            plan: The plan to execute
            mode: Execution mode
            context: Optional execution context
            
        Returns:
            A new PlanExecution instance
        """
        execution_id = str(uuid.uuid4())
        
        return cls(
            execution_id=execution_id,
            plan=plan,
            mode=mode,
            execution_context=context or ExecutionContext()
        )
    
    def initialize_step_executions(self) -> None:
        """Initialize step executions for all steps in the plan."""
        self.step_executions = {}
        
        for step_id, step in self.plan.steps.items():
            # Create step execution
            step_execution = StepExecution.create(
                step=step,
                context=self.execution_context
            )
            
            # Store in step executions
            self.step_executions[step_id] = step_execution
    
    def add_log(self, level: str, message: str, details: Dict[str, Any] = None) -> None:
        """
        Add a log entry to this execution.
        
        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: Log message
            details: Optional additional details
        """
        self.logs.append(LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            details=details or {}
        ))
    
    def update_progress(self) -> float:
        """
        Update the progress of this execution based on step progress.
        
        Returns:
            The updated progress value (0.0-1.0)
        """
        if not self.step_executions:
            return 0.0
        
        # Calculate total and completed weight
        total_weight = 0
        completed_weight = 0
        
        for step_id, step_execution in self.step_executions.items():
            weight = step_execution.step.importance
            total_weight += weight
            completed_weight += weight * step_execution.progress
        
        # Calculate overall progress
        if total_weight > 0:
            self.progress = completed_weight / total_weight
        else:
            self.progress = 0.0
            
        return self.progress
    
    def get_next_executable_steps(self) -> List[StepExecution]:
        """
        Get steps that are ready to be executed.
        
        Returns:
            List of executable step executions
        """
        executable_steps = []
        
        # Collect completed step IDs
        completed_steps = set()
        for step_id, step_execution in self.step_executions.items():
            if step_execution.status == ExecutionStatus.COMPLETED:
                completed_steps.add(step_id)
        
        # Check each step
        for step_id, step_execution in self.step_executions.items():
            # Skip steps that are not in created status
            if step_execution.status != ExecutionStatus.CREATED:
                continue
                
            # Check if all dependencies are satisfied
            dependencies_met = True
            for dep_id in step_execution.step.dependencies:
                if dep_id not in completed_steps:
                    dependencies_met = False
                    break
                    
            if dependencies_met:
                executable_steps.append(step_execution)
        
        return executable_steps


@dataclass
class ActionExecutor:
    """
    Interface for step executors that can execute specific action types.
    """
    executor_id: str
    name: str
    description: str
    supported_action_types: List[str]
    version: str
    
    def can_execute(self, step: PlanStep) -> bool:
        """
        Check if this executor can execute a specific step.
        
        Args:
            step: The step to check
            
        Returns:
            True if this executor can execute the step
        """
        # Basic check: action type is supported
        return step.action_type in self.supported_action_types
    
    def get_required_resources(self, step: PlanStep) -> List[ResourceRequirement]:
        """
        Get resource requirements for executing a step.
        
        Args:
            step: The step to execute
            
        Returns:
            List of resource requirements
        """
        # By default, return the step's resource requirements
        return step.resource_requirements
    
    async def execute(self, step: PlanStep, context: ExecutionContext) -> StepResult:
        """
        Execute a step.
        
        Args:
            step: The step to execute
            context: Execution context
            
        Returns:
            Result of the execution
        """
        # This is an abstract method - subclasses should override
        raise NotImplementedError("Subclasses must implement execute method")


@dataclass
class ExecutionCallback:
    """
    Callbacks for execution events.
    """
    on_start: Optional[Callable[[StepExecution], None]] = None
    on_progress: Optional[Callable[[StepExecution, float], None]] = None
    on_complete: Optional[Callable[[StepExecution, StepResult], None]] = None
    on_error: Optional[Callable[[StepExecution, Exception], None]] = None
    on_cancel: Optional[Callable[[StepExecution], None]] = None
