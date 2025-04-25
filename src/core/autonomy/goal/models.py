"""
Goal Management System - Data Models

This module defines the data models for representing goals in the autonomy engine.
Goals are the central concept that drive the planning and execution of autonomous actions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union
import uuid


class GoalStatus(Enum):
    """Status values for a goal's lifecycle."""
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GoalType(Enum):
    """Types of goals that the system can manage."""
    ACHIEVEMENT = "achievement"  # Goal is to reach a specific state
    MAINTENANCE = "maintenance"  # Goal is to maintain a state or condition
    AVOIDANCE = "avoidance"      # Goal is to prevent a certain state


class GoalSource(Enum):
    """Source of a goal."""
    USER = "user"            # Created based on user input
    SYSTEM = "system"        # Created by the system autonomously
    DERIVED = "derived"      # Derived from another goal


class GoalPriority(Enum):
    """Priority levels for goals."""
    CRITICAL = 100
    HIGH = 75
    MEDIUM = 50
    LOW = 25
    BACKGROUND = 10


@dataclass
class SuccessCriteria:
    """Criteria for determining if a goal has been successfully achieved."""
    description: str
    validation_method: str  # "manual", "automatic", "hybrid"
    validation_params: Dict = field(default_factory=dict)


@dataclass
class GoalMetrics:
    """Metrics for tracking goal progress and quality."""
    completion_method: str  # How to measure completion (percentage, binary, etc.)
    quality_metrics: List[Dict] = field(default_factory=list)  # Quality assessment metrics
    measurement_frequency: str = "on_update"  # How often to measure progress


@dataclass
class GoalResource:
    """Resource required or optional for goal achievement."""
    name: str
    type: str  # "computation", "data", "external", etc.
    required: bool = True
    estimated_quantity: Optional[Union[int, float]] = None
    constraints: Dict = field(default_factory=dict)


@dataclass
class GoalContext:
    """Contextual information about a goal."""
    importance: str = ""  # Description of why this goal is important
    background: str = ""  # Background information relevant to this goal
    constraints: List[str] = field(default_factory=list)  # Constraints on goal achievement
    tags: List[str] = field(default_factory=list)  # Categorization tags


@dataclass
class Goal:
    """
    Represents a goal that the system aims to achieve.
    
    A goal is a desired state or outcome that the system works towards.
    Goals drive the planning and execution of actions.
    """
    # Core attributes
    goal_id: str
    description: str  # Natural language description
    
    # Structured representation
    goal_type: GoalType
    success_criteria: List[SuccessCriteria]
    
    # Status tracking
    status: GoalStatus = GoalStatus.CREATED
    progress: float = 0.0  # 0.0-1.0
    
    # Metadata
    priority: int = 50  # 0-100, with higher values indicating higher priority
    source: GoalSource = GoalSource.USER
    deadline: Optional[datetime] = None
    
    # Relationships
    parent_goal_id: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)  # IDs of child goals
    dependencies: List[str] = field(default_factory=list)  # IDs of goals this depends on
    
    # Details
    context: GoalContext = field(default_factory=GoalContext)
    metrics: GoalMetrics = field(default_factory=lambda: GoalMetrics(completion_method="percentage"))
    resources: List[GoalResource] = field(default_factory=list)
    
    # Timing information
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[int] = None  # In seconds
    
    # Extended properties
    structured_representation: Dict = field(default_factory=dict)
    estimated_difficulty: int = 50  # 0-100
    metadata: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    @classmethod
    def create(cls, description: str, goal_type: GoalType = GoalType.ACHIEVEMENT, 
               priority: int = 50, deadline: Optional[datetime] = None, 
               source: GoalSource = GoalSource.USER, **kwargs) -> 'Goal':
        """
        Create a new goal with a generated ID.
        
        Args:
            description: Natural language description of the goal
            goal_type: Type of goal (achievement, maintenance, avoidance)
            priority: Priority level (0-100)
            deadline: Optional deadline for goal completion
            source: Source of the goal (user, system, derived)
            **kwargs: Additional properties for the goal
            
        Returns:
            A new Goal instance
        """
        goal_id = str(uuid.uuid4())
        
        # Create default success criteria if not provided
        success_criteria = kwargs.get('success_criteria', [])
        if not success_criteria:
            success_criteria = [SuccessCriteria(
                description=f"Complete the goal: {description}",
                validation_method="automatic"
            )]
        
        return cls(
            goal_id=goal_id,
            description=description,
            goal_type=goal_type,
            success_criteria=success_criteria,
            priority=priority,
            deadline=deadline,
            source=source,
            **kwargs
        )
    
    def to_dict(self) -> Dict:
        """
        Convert the goal to a dictionary representation.
        
        Returns:
            Dict representation of the goal
        """
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "goal_type": self.goal_type.value,
            "success_criteria": [
                {
                    "description": criterion.description,
                    "validation_method": criterion.validation_method,
                    "validation_params": criterion.validation_params
                }
                for criterion in self.success_criteria
            ],
            "status": self.status.value,
            "progress": self.progress,
            "priority": self.priority,
            "source": self.source.value,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "parent_goal_id": self.parent_goal_id,
            "sub_goals": self.sub_goals,
            "dependencies": self.dependencies,
            "context": {
                "importance": self.context.importance,
                "background": self.context.background,
                "constraints": self.context.constraints,
                "tags": self.context.tags
            },
            "metrics": {
                "completion_method": self.metrics.completion_method,
                "quality_metrics": self.metrics.quality_metrics,
                "measurement_frequency": self.metrics.measurement_frequency
            },
            "resources": [
                {
                    "name": resource.name,
                    "type": resource.type,
                    "required": resource.required,
                    "estimated_quantity": resource.estimated_quantity,
                    "constraints": resource.constraints
                }
                for resource in self.resources
            ],
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "estimated_duration": self.estimated_duration,
            "structured_representation": self.structured_representation,
            "estimated_difficulty": self.estimated_difficulty,
            "metadata": self.metadata,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Goal':
        """
        Create a Goal instance from a dictionary.
        
        Args:
            data: Dictionary representation of a goal
            
        Returns:
            Goal instance
        """
        # Process dates
        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
        started_at = datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
        completed_at = datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        deadline = datetime.fromisoformat(data["deadline"]) if data.get("deadline") else None
        
        # Process enums
        goal_type = GoalType(data["goal_type"]) if data.get("goal_type") else GoalType.ACHIEVEMENT
        status = GoalStatus(data["status"]) if data.get("status") else GoalStatus.CREATED
        source = GoalSource(data["source"]) if data.get("source") else GoalSource.USER
        
        # Process complex objects
        success_criteria = [
            SuccessCriteria(
                description=criterion["description"],
                validation_method=criterion["validation_method"],
                validation_params=criterion.get("validation_params", {})
            )
            for criterion in data.get("success_criteria", [])
        ]
        
        context = GoalContext(
            importance=data.get("context", {}).get("importance", ""),
            background=data.get("context", {}).get("background", ""),
            constraints=data.get("context", {}).get("constraints", []),
            tags=data.get("context", {}).get("tags", [])
        )
        
        metrics_data = data.get("metrics", {})
        metrics = GoalMetrics(
            completion_method=metrics_data.get("completion_method", "percentage"),
            quality_metrics=metrics_data.get("quality_metrics", []),
            measurement_frequency=metrics_data.get("measurement_frequency", "on_update")
        )
        
        resources = [
            GoalResource(
                name=resource["name"],
                type=resource["type"],
                required=resource.get("required", True),
                estimated_quantity=resource.get("estimated_quantity"),
                constraints=resource.get("constraints", {})
            )
            for resource in data.get("resources", [])
        ]
        
        return cls(
            goal_id=data["goal_id"],
            description=data["description"],
            goal_type=goal_type,
            success_criteria=success_criteria,
            status=status,
            progress=data.get("progress", 0.0),
            priority=data.get("priority", 50),
            source=source,
            deadline=deadline,
            parent_goal_id=data.get("parent_goal_id"),
            sub_goals=data.get("sub_goals", []),
            dependencies=data.get("dependencies", []),
            context=context,
            metrics=metrics,
            resources=resources,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            estimated_duration=data.get("estimated_duration"),
            structured_representation=data.get("structured_representation", {}),
            estimated_difficulty=data.get("estimated_difficulty", 50),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", [])
        )
    
    def update_status(self, new_status: GoalStatus, progress: Optional[float] = None) -> None:
        """
        Update the status of this goal.
        
        Args:
            new_status: The new status to set
            progress: Optional progress value to update (0.0-1.0)
        """
        self.status = new_status
        
        # Update timing information based on status changes
        if new_status == GoalStatus.ACTIVE and not self.started_at:
            self.started_at = datetime.now()
        elif new_status in (GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.CANCELLED):
            self.completed_at = datetime.now()
        
        # Update progress if provided
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))  # Ensure between 0 and 1
            
            # Automatically complete goal if progress reaches 100%
            if self.progress >= 1.0 and new_status != GoalStatus.COMPLETED:
                self.status = GoalStatus.COMPLETED
                self.completed_at = datetime.now()
    
    def add_sub_goal(self, sub_goal_id: str) -> None:
        """
        Add a sub-goal to this goal.
        
        Args:
            sub_goal_id: ID of the sub-goal to add
        """
        if sub_goal_id not in self.sub_goals:
            self.sub_goals.append(sub_goal_id)
    
    def add_dependency(self, dependency_id: str) -> None:
        """
        Add a dependency to this goal.
        
        Args:
            dependency_id: ID of the goal this goal depends on
        """
        if dependency_id not in self.dependencies:
            self.dependencies.append(dependency_id)
