"""
Planning Models for Autonomous Action Engine.

This module defines the core data models for planning in the autonomy system.
Plans are structured representations of steps to achieve goals, including
constraints, dependencies, resources, and evaluation criteria.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Union, Any
import uuid


class PlanStatus(Enum):
    """Status values for a plan's lifecycle."""
    DRAFT = "draft"           # Plan is being created
    READY = "ready"           # Plan is ready for execution
    IN_PROGRESS = "in_progress"  # Plan is currently being executed
    COMPLETED = "completed"   # Plan was successfully completed
    FAILED = "failed"         # Plan execution failed
    CANCELLED = "cancelled"   # Plan was cancelled before completion
    PAUSED = "paused"         # Plan execution is temporarily paused


class ExecutionStrategy(Enum):
    """Strategies for executing a plan."""
    SEQUENTIAL = "sequential"  # Execute steps one after another
    PARALLEL = "parallel"      # Execute steps in parallel where possible
    ADAPTIVE = "adaptive"      # Dynamically decide execution order


class EvaluationMethod(Enum):
    """Methods for evaluating plan success."""
    BINARY = "binary"           # Success or failure only
    MULTI_CRITERIA = "multi_criteria"  # Multiple criteria with weights
    FUZZY = "fuzzy"             # Degree of success (0-1)
    CUSTOM = "custom"           # Custom evaluation method


@dataclass
class ResourceRequirement:
    """Representation of a resource requirement for a step or plan."""
    resource_type: str
    identifier: Optional[str] = None
    quantity: Optional[float] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    is_critical: bool = False
    alternative_resources: List[str] = field(default_factory=list)


@dataclass
class StepConstraint:
    """Constraint on a plan step's execution."""
    constraint_type: str  # "time", "resource", "precondition", "quality", etc.
    description: str
    evaluation_function: Optional[str] = None  # Reference to function for checking constraint
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result data for a plan step."""
    success: bool
    completion_time: Optional[datetime] = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class PlanStep:
    """
    A step in a plan, representing a discrete action to be performed.
    """
    step_id: str
    description: str
    action_type: str  # Type of action to be performed
    parameters: Dict[str, Any] = field(default_factory=dict)  # Parameters for the action
    
    # Status tracking
    status: PlanStatus = PlanStatus.DRAFT
    progress: float = 0.0  # 0.0-1.0
    result: Optional[StepResult] = None
    
    # Dependencies and constraints
    dependencies: List[str] = field(default_factory=list)  # IDs of steps this depends on
    constraints: List[StepConstraint] = field(default_factory=list)
    
    # Resources and timing
    estimated_duration: Optional[int] = None  # In seconds
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    
    # Metadata
    importance: int = 50  # 0-100, with higher values being more important
    fallback_step: Optional[str] = None  # ID of step to execute if this fails
    retry_strategy: Optional[Dict[str, Any]] = None  # Strategy for retrying if fail
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, description: str, action_type: str, 
               parameters: Dict[str, Any] = None, **kwargs) -> 'PlanStep':
        """
        Create a new plan step with a generated ID.
        
        Args:
            description: Description of the step
            action_type: Type of action to perform
            parameters: Parameters for the action
            **kwargs: Additional properties for the step
            
        Returns:
            A new PlanStep instance
        """
        step_id = str(uuid.uuid4())
        
        return cls(
            step_id=step_id,
            description=description,
            action_type=action_type,
            parameters=parameters or {},
            **kwargs
        )


@dataclass
class PlanEvaluation:
    """
    Evaluation criteria and results for a plan.
    """
    method: EvaluationMethod
    criteria: List[Dict[str, Any]] = field(default_factory=list)
    weights: Optional[Dict[str, float]] = None
    thresholds: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create_binary(cls) -> 'PlanEvaluation':
        """Create a simple binary evaluation."""
        return cls(
            method=EvaluationMethod.BINARY,
            criteria=[{"id": "completion", "description": "Plan is fully completed"}]
        )
    
    @classmethod
    def create_multi_criteria(cls, criteria: List[Dict[str, Any]], 
                              weights: Dict[str, float]) -> 'PlanEvaluation':
        """Create a multi-criteria evaluation."""
        return cls(
            method=EvaluationMethod.MULTI_CRITERIA,
            criteria=criteria,
            weights=weights
        )


@dataclass
class ExecutionConfig:
    """
    Configuration for plan execution.
    """
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    max_parallel_steps: int = 1
    timeout: Optional[int] = None  # In seconds
    retry_limit: int = 3
    pause_on_error: bool = True
    resource_conflict_strategy: str = "wait"  # "wait", "reorder", "fail"
    monitoring_interval: int = 5  # In seconds
    execution_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """
    A structured plan for achieving a goal, consisting of ordered steps.
    """
    plan_id: str
    goal_id: str  # Reference to the goal this plan is for
    name: str
    description: str
    
    # Plan steps
    steps: Dict[str, PlanStep] = field(default_factory=dict)
    
    # Status tracking
    status: PlanStatus = PlanStatus.DRAFT
    progress: float = 0.0  # 0.0-1.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Execution configuration
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    # Evaluation
    evaluation: PlanEvaluation = field(default_factory=PlanEvaluation.create_binary)
    
    # Resources
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    
    # Metadata
    version: int = 1
    created_by: str = "planner"  # Component that created the plan
    is_template: bool = False
    template_id: Optional[str] = None  # Reference to template if derived
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, goal_id: str, name: str, description: str, **kwargs) -> 'Plan':
        """
        Create a new plan with a generated ID.
        
        Args:
            goal_id: ID of the goal this plan is for
            name: Name of the plan
            description: Description of the plan
            **kwargs: Additional properties for the plan
            
        Returns:
            A new Plan instance
        """
        plan_id = str(uuid.uuid4())
        
        return cls(
            plan_id=plan_id,
            goal_id=goal_id,
            name=name,
            description=description,
            **kwargs
        )
    
    def add_step(self, step: PlanStep) -> None:
        """
        Add a step to the plan.
        
        Args:
            step: The step to add
        """
        self.steps[step.step_id] = step
    
    def get_ordered_steps(self) -> List[PlanStep]:
        """
        Get steps in execution order, respecting dependencies.
        
        Returns:
            List of steps in execution order
        """
        # Implementation of topological sort to order steps by dependencies
        result = []
        visited = set()
        temp_visited = set()
        
        def visit(step_id):
            if step_id in temp_visited:
                raise ValueError(f"Circular dependency detected in plan {self.plan_id}")
            
            if step_id not in visited:
                temp_visited.add(step_id)
                
                step = self.steps.get(step_id)
                if step:
                    for dep_id in step.dependencies:
                        visit(dep_id)
                    
                temp_visited.remove(step_id)
                visited.add(step_id)
                result.append(step)
        
        # Visit all steps
        for step_id in self.steps:
            if step_id not in visited:
                visit(step_id)
        
        return result
    
    def update_progress(self) -> float:
        """
        Update the plan's progress based on steps' progress.
        
        Returns:
            The updated progress value (0.0-1.0)
        """
        if not self.steps:
            return 0.0
        
        # Calculate weighted progress of all steps
        total_importance = sum(step.importance for step in self.steps.values())
        weighted_progress = sum(
            step.progress * (step.importance / total_importance)
            for step in self.steps.values()
        )
        
        self.progress = min(1.0, max(0.0, weighted_progress))
        return self.progress
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the plan's success based on its evaluation criteria.
        
        Returns:
            Evaluation result
        """
        result = {}
        
        if self.evaluation.method == EvaluationMethod.BINARY:
            # Simple binary evaluation
            all_completed = all(
                step.status == PlanStatus.COMPLETED
                for step in self.steps.values()
            )
            result = {"success": all_completed}
        
        elif self.evaluation.method == EvaluationMethod.MULTI_CRITERIA:
            # Multi-criteria evaluation
            criteria_results = {}
            overall_score = 0.0
            
            for criterion in self.evaluation.criteria:
                criterion_id = criterion["id"]
                # Implementation would depend on the specific criteria
                # This is a placeholder
                criterion_score = 0.0
                criteria_results[criterion_id] = criterion_score
                
                if self.evaluation.weights:
                    weight = self.evaluation.weights.get(criterion_id, 1.0)
                    overall_score += criterion_score * weight
            
            result = {
                "criteria_results": criteria_results,
                "overall_score": overall_score,
                "success": overall_score >= self.evaluation.thresholds.get("minimum_score", 0.5)
            }
        
        # Store the evaluation result
        self.evaluation.result = result
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the plan to a dictionary representation.
        
        Returns:
            Dict representation of the plan
        """
        return {
            "plan_id": self.plan_id,
            "goal_id": self.goal_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_config": {
                "strategy": self.execution_config.strategy.value,
                "max_parallel_steps": self.execution_config.max_parallel_steps,
                "timeout": self.execution_config.timeout,
                "retry_limit": self.execution_config.retry_limit,
                "pause_on_error": self.execution_config.pause_on_error,
                "resource_conflict_strategy": self.execution_config.resource_conflict_strategy,
                "monitoring_interval": self.execution_config.monitoring_interval,
                "execution_parameters": self.execution_config.execution_parameters
            },
            "evaluation": {
                "method": self.evaluation.method.value,
                "criteria": self.evaluation.criteria,
                "weights": self.evaluation.weights,
                "thresholds": self.evaluation.thresholds,
                "result": self.evaluation.result
            },
            "resource_requirements": [
                {
                    "resource_type": req.resource_type,
                    "identifier": req.identifier,
                    "quantity": req.quantity,
                    "properties": req.properties,
                    "is_critical": req.is_critical,
                    "alternative_resources": req.alternative_resources
                }
                for req in self.resource_requirements
            ],
            "steps": {
                step_id: {
                    "step_id": step.step_id,
                    "description": step.description,
                    "action_type": step.action_type,
                    "parameters": step.parameters,
                    "status": step.status.value,
                    "progress": step.progress,
                    "result": {
                        "success": step.result.success,
                        "completion_time": step.result.completion_time.isoformat() if step.result and step.result.completion_time else None,
                        "outputs": step.result.outputs,
                        "metrics": step.result.metrics,
                        "error": step.result.error,
                        "notes": step.result.notes
                    } if step.result else None,
                    "dependencies": step.dependencies,
                    "constraints": [
                        {
                            "constraint_type": constraint.constraint_type,
                            "description": constraint.description,
                            "evaluation_function": constraint.evaluation_function,
                            "parameters": constraint.parameters
                        }
                        for constraint in step.constraints
                    ],
                    "estimated_duration": step.estimated_duration,
                    "resource_requirements": [
                        {
                            "resource_type": req.resource_type,
                            "identifier": req.identifier,
                            "quantity": req.quantity,
                            "properties": req.properties,
                            "is_critical": req.is_critical,
                            "alternative_resources": req.alternative_resources
                        }
                        for req in step.resource_requirements
                    ],
                    "importance": step.importance,
                    "fallback_step": step.fallback_step,
                    "retry_strategy": step.retry_strategy,
                    "metadata": step.metadata
                }
                for step_id, step in self.steps.items()
            },
            "version": self.version,
            "created_by": self.created_by,
            "is_template": self.is_template,
            "template_id": self.template_id,
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Plan':
        """
        Create a Plan instance from a dictionary.
        
        Args:
            data: Dictionary representation of a plan
            
        Returns:
            Plan instance
        """
        # Process dates
        created_at = datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now()
        started_at = datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
        completed_at = datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        
        # Process enums
        status = PlanStatus(data["status"]) if "status" in data else PlanStatus.DRAFT
        
        # Process execution config
        execution_config_data = data.get("execution_config", {})
        execution_config = ExecutionConfig(
            strategy=ExecutionStrategy(execution_config_data.get("strategy", "sequential")),
            max_parallel_steps=execution_config_data.get("max_parallel_steps", 1),
            timeout=execution_config_data.get("timeout"),
            retry_limit=execution_config_data.get("retry_limit", 3),
            pause_on_error=execution_config_data.get("pause_on_error", True),
            resource_conflict_strategy=execution_config_data.get("resource_conflict_strategy", "wait"),
            monitoring_interval=execution_config_data.get("monitoring_interval", 5),
            execution_parameters=execution_config_data.get("execution_parameters", {})
        )
        
        # Process evaluation
        evaluation_data = data.get("evaluation", {})
        evaluation = PlanEvaluation(
            method=EvaluationMethod(evaluation_data.get("method", "binary")),
            criteria=evaluation_data.get("criteria", []),
            weights=evaluation_data.get("weights"),
            thresholds=evaluation_data.get("thresholds", {}),
            result=evaluation_data.get("result")
        )
        
        # Process resource requirements
        resource_requirements = []
        for req_data in data.get("resource_requirements", []):
            resource_requirements.append(
                ResourceRequirement(
                    resource_type=req_data["resource_type"],
                    identifier=req_data.get("identifier"),
                    quantity=req_data.get("quantity"),
                    properties=req_data.get("properties", {}),
                    is_critical=req_data.get("is_critical", False),
                    alternative_resources=req_data.get("alternative_resources", [])
                )
            )
        
        # Create the plan without steps
        plan = cls(
            plan_id=data["plan_id"],
            goal_id=data["goal_id"],
            name=data["name"],
            description=data["description"],
            status=status,
            progress=data.get("progress", 0.0),
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            execution_config=execution_config,
            evaluation=evaluation,
            resource_requirements=resource_requirements,
            version=data.get("version", 1),
            created_by=data.get("created_by", "planner"),
            is_template=data.get("is_template", False),
            template_id=data.get("template_id"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )
        
        # Process steps
        steps_data = data.get("steps", {})
        for step_id, step_data in steps_data.items():
            # Process step result
            result_data = step_data.get("result")
            result = None
            if result_data:
                completion_time = (datetime.fromisoformat(result_data["completion_time"]) 
                                 if result_data.get("completion_time") else None)
                result = StepResult(
                    success=result_data["success"],
                    completion_time=completion_time,
                    outputs=result_data.get("outputs", {}),
                    metrics=result_data.get("metrics", {}),
                    error=result_data.get("error"),
                    notes=result_data.get("notes")
                )
            
            # Process step constraints
            constraints = []
            for constraint_data in step_data.get("constraints", []):
                constraints.append(
                    StepConstraint(
                        constraint_type=constraint_data["constraint_type"],
                        description=constraint_data["description"],
                        evaluation_function=constraint_data.get("evaluation_function"),
                        parameters=constraint_data.get("parameters", {})
                    )
                )
            
            # Process step resource requirements
            step_resource_requirements = []
            for req_data in step_data.get("resource_requirements", []):
                step_resource_requirements.append(
                    ResourceRequirement(
                        resource_type=req_data["resource_type"],
                        identifier=req_data.get("identifier"),
                        quantity=req_data.get("quantity"),
                        properties=req_data.get("properties", {}),
                        is_critical=req_data.get("is_critical", False),
                        alternative_resources=req_data.get("alternative_resources", [])
                    )
                )
            
            # Create the step
            step = PlanStep(
                step_id=step_data["step_id"],
                description=step_data["description"],
                action_type=step_data["action_type"],
                parameters=step_data.get("parameters", {}),
                status=PlanStatus(step_data.get("status", "draft")),
                progress=step_data.get("progress", 0.0),
                result=result,
                dependencies=step_data.get("dependencies", []),
                constraints=constraints,
                estimated_duration=step_data.get("estimated_duration"),
                resource_requirements=step_resource_requirements,
                importance=step_data.get("importance", 50),
                fallback_step=step_data.get("fallback_step"),
                retry_strategy=step_data.get("retry_strategy"),
                metadata=step_data.get("metadata", {})
            )
            
            # Add step to plan
            plan.steps[step_id] = step
        
        return plan
