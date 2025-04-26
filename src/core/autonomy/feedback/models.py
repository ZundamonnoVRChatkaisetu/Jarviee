"""
Models related to the feedback and learning system.

This module defines data structures for capturing feedback, recording learning
experiences, and managing improvement data.
"""

import uuid
import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field


class FeedbackSource(Enum):
    """Sources of feedback for actions and plans."""
    HUMAN = "human"
    SYSTEM = "system"
    EXECUTION = "execution"
    SELF_EVALUATION = "self_evaluation"
    EXTERNAL = "external"


class FeedbackType(Enum):
    """Types of feedback that can be provided."""
    SUCCESS = "success"  # Full success
    PARTIAL = "partial"  # Partial success
    FAILURE = "failure"  # Failure
    SUGGESTION = "suggestion"  # Suggestion for improvement
    CORRECTION = "correction"  # Correction of error
    PREFERENCE = "preference"  # Preference (no right/wrong)
    VALIDATION = "validation"  # Validation checkpoint


class FeedbackSeverity(Enum):
    """Severity levels for feedback."""
    INFO = "info"  # Informational
    LOW = "low"  # Minor
    MEDIUM = "medium"  # Significant
    HIGH = "high"  # Critical
    BLOCKER = "blocker"  # Completely blocks progress


class LearningStrategy(Enum):
    """Strategies for incorporating feedback."""
    IMMEDIATE = "immediate"  # Apply immediately
    DELIBERATE = "deliberate"  # Analyze before applying
    AGGREGATE = "aggregate"  # Apply after collecting multiple instances
    CONTEXTUAL = "contextual"  # Apply in similar contexts only
    EXPERIMENTAL = "experimental"  # Test in limited scenarios first


@dataclass
class Feedback:
    """Representation of feedback on an action, step, or plan."""
    
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    source: FeedbackSource = FeedbackSource.SYSTEM
    feedback_type: FeedbackType = FeedbackType.SUGGESTION
    severity: FeedbackSeverity = FeedbackSeverity.MEDIUM
    
    # Target information
    target_type: str = "action"  # "action", "step", "plan", "goal", etc.
    target_id: str = ""
    
    # Content
    content: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Tags for categorization and analysis
    tags: List[str] = field(default_factory=list)
    
    @classmethod
    def create(cls, source: FeedbackSource, feedback_type: FeedbackType, 
              content: str, target_type: str, target_id: str, **kwargs) -> 'Feedback':
        """
        Create a new feedback instance.
        
        Args:
            source: Source of the feedback
            feedback_type: Type of feedback
            content: Feedback content
            target_type: Type of target
            target_id: ID of target
            **kwargs: Additional fields
            
        Returns:
            New Feedback instance
        """
        return cls(
            source=source,
            feedback_type=feedback_type,
            content=content,
            target_type=target_type,
            target_id=target_id,
            **kwargs
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "feedback_id": self.feedback_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source.value,
            "feedback_type": self.feedback_type.value,
            "severity": self.severity.value,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "content": self.content,
            "details": self.details,
            "context": self.context,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Feedback':
        """
        Create from a dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            Feedback instance
        """
        feedback = cls(
            feedback_id=data.get("feedback_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data 
                      else datetime.now(),
            source=FeedbackSource(data["source"]) if "source" in data 
                   else FeedbackSource.SYSTEM,
            feedback_type=FeedbackType(data["feedback_type"]) if "feedback_type" in data 
                         else FeedbackType.SUGGESTION,
            severity=FeedbackSeverity(data["severity"]) if "severity" in data 
                    else FeedbackSeverity.MEDIUM,
            target_type=data.get("target_type", "action"),
            target_id=data.get("target_id", "")
        )
        
        feedback.content = data.get("content", "")
        feedback.details = data.get("details", {})
        feedback.context = data.get("context", {})
        feedback.tags = data.get("tags", [])
        
        return feedback


@dataclass
class LearningExperience:
    """Representation of a learning experience derived from feedback."""
    
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Source feedback
    feedback_ids: List[str] = field(default_factory=list)
    
    # What was learned
    insight: str = ""
    learned_pattern: Dict[str, Any] = field(default_factory=dict)
    
    # How to apply the learning
    application_strategy: LearningStrategy = LearningStrategy.DELIBERATE
    application_context: Dict[str, Any] = field(default_factory=dict)
    
    # Implementation details
    implementation: Optional[Dict[str, Any]] = None
    
    # Verification
    verified: bool = False
    verification_details: Optional[Dict[str, Any]] = None
    
    # Tags for categorization and analysis
    tags: List[str] = field(default_factory=list)
    
    @classmethod
    def create_from_feedback(cls, feedback: Union[Feedback, List[Feedback]], 
                           insight: str, strategy: LearningStrategy, **kwargs) -> 'LearningExperience':
        """
        Create a learning experience from feedback.
        
        Args:
            feedback: Feedback instance(s)
            insight: Insight derived from feedback
            strategy: Application strategy
            **kwargs: Additional fields
            
        Returns:
            New LearningExperience instance
        """
        if isinstance(feedback, Feedback):
            feedback_ids = [feedback.feedback_id]
            tags = feedback.tags.copy()
        else:
            feedback_ids = [f.feedback_id for f in feedback]
            # Combine tags from all feedback
            tags = list(set().union(*[f.tags for f in feedback]))
        
        return cls(
            feedback_ids=feedback_ids,
            insight=insight,
            application_strategy=strategy,
            tags=tags,
            **kwargs
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "experience_id": self.experience_id,
            "timestamp": self.timestamp.isoformat(),
            "feedback_ids": self.feedback_ids,
            "insight": self.insight,
            "learned_pattern": self.learned_pattern,
            "application_strategy": self.application_strategy.value,
            "application_context": self.application_context,
            "implementation": self.implementation,
            "verified": self.verified,
            "verification_details": self.verification_details,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningExperience':
        """
        Create from a dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            LearningExperience instance
        """
        experience = cls(
            experience_id=data.get("experience_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data 
                      else datetime.now(),
            feedback_ids=data.get("feedback_ids", []),
            insight=data.get("insight", ""),
            application_strategy=LearningStrategy(data["application_strategy"]) 
                                if "application_strategy" in data 
                                else LearningStrategy.DELIBERATE
        )
        
        experience.learned_pattern = data.get("learned_pattern", {})
        experience.application_context = data.get("application_context", {})
        experience.implementation = data.get("implementation")
        experience.verified = data.get("verified", False)
        experience.verification_details = data.get("verification_details")
        experience.tags = data.get("tags", [])
        
        return experience


@dataclass
class PerformanceMetric:
    """Representation of a performance metric for evaluation and learning."""
    
    metric_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Metric type and units
    metric_type: str = "numeric"  # "numeric", "boolean", "categorical", "distribution"
    unit: Optional[str] = None
    
    # Thresholds for evaluation
    thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Target improvement direction
    improvement_direction: str = "increase"  # "increase", "decrease", "target"
    target_value: Optional[float] = None
    
    # Additional configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, name: str, description: str, metric_type: str = "numeric", 
              improvement_direction: str = "increase", **kwargs) -> 'PerformanceMetric':
        """
        Create a new performance metric.
        
        Args:
            name: Metric name
            description: Metric description
            metric_type: Type of metric
            improvement_direction: Direction of improvement
            **kwargs: Additional fields
            
        Returns:
            New PerformanceMetric instance
        """
        return cls(
            name=name,
            description=description,
            metric_type=metric_type,
            improvement_direction=improvement_direction,
            **kwargs
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "description": self.description,
            "metric_type": self.metric_type,
            "unit": self.unit,
            "thresholds": self.thresholds,
            "improvement_direction": self.improvement_direction,
            "target_value": self.target_value,
            "config": self.config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetric':
        """
        Create from a dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            PerformanceMetric instance
        """
        metric = cls(
            metric_id=data.get("metric_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            metric_type=data.get("metric_type", "numeric"),
            improvement_direction=data.get("improvement_direction", "increase")
        )
        
        metric.unit = data.get("unit")
        metric.thresholds = data.get("thresholds", {})
        metric.target_value = data.get("target_value")
        metric.config = data.get("config", {})
        
        return metric


@dataclass
class MetricMeasurement:
    """Representation of a measurement of a performance metric."""
    
    measurement_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Metric information
    metric_id: str = ""
    metric_name: str = ""
    
    # Target information
    target_type: str = ""  # "action", "plan", "goal", "system", etc.
    target_id: str = ""
    
    # Measurement value
    value: Any = None
    confidence: Optional[float] = None
    
    # Context of measurement
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation against thresholds
    evaluation: Optional[str] = None  # "excellent", "good", "satisfactory", "poor", "unacceptable"
    
    @classmethod
    def create(cls, metric_id: str, metric_name: str, value: Any, 
              target_type: str, target_id: str, **kwargs) -> 'MetricMeasurement':
        """
        Create a new metric measurement.
        
        Args:
            metric_id: ID of the metric
            metric_name: Name of the metric
            value: Measured value
            target_type: Type of target
            target_id: ID of target
            **kwargs: Additional fields
            
        Returns:
            New MetricMeasurement instance
        """
        return cls(
            metric_id=metric_id,
            metric_name=metric_name,
            value=value,
            target_type=target_type,
            target_id=target_id,
            **kwargs
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "measurement_id": self.measurement_id,
            "timestamp": self.timestamp.isoformat(),
            "metric_id": self.metric_id,
            "metric_name": self.metric_name,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "value": self.value,
            "confidence": self.confidence,
            "context": self.context,
            "evaluation": self.evaluation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricMeasurement':
        """
        Create from a dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            MetricMeasurement instance
        """
        measurement = cls(
            measurement_id=data.get("measurement_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data 
                      else datetime.now(),
            metric_id=data.get("metric_id", ""),
            metric_name=data.get("metric_name", ""),
            target_type=data.get("target_type", ""),
            target_id=data.get("target_id", ""),
            value=data.get("value")
        )
        
        measurement.confidence = data.get("confidence")
        measurement.context = data.get("context", {})
        measurement.evaluation = data.get("evaluation")
        
        return measurement


@dataclass
class ImprovementPlan:
    """Representation of a plan for system improvement based on feedback and learning."""
    
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Plan information
    name: str = ""
    description: str = ""
    priority: int = 3  # 1 (highest) to 5 (lowest)
    
    # Source information
    learning_experience_ids: List[str] = field(default_factory=list)
    feedback_ids: List[str] = field(default_factory=list)
    
    # Improvement details
    target_components: List[str] = field(default_factory=list)
    implementation_steps: List[Dict[str, Any]] = field(default_factory=list)
    success_criteria: List[Dict[str, Any]] = field(default_factory=list)
    
    # Plan status
    status: str = "draft"  # "draft", "approved", "in_progress", "completed", "rejected"
    progress: float = 0.0  # 0.0 to 1.0
    
    # Results
    results: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, name: str, description: str, priority: int = 3, **kwargs) -> 'ImprovementPlan':
        """
        Create a new improvement plan.
        
        Args:
            name: Plan name
            description: Plan description
            priority: Priority level
            **kwargs: Additional fields
            
        Returns:
            New ImprovementPlan instance
        """
        return cls(
            name=name,
            description=description,
            priority=priority,
            **kwargs
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "plan_id": self.plan_id,
            "timestamp": self.timestamp.isoformat(),
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "learning_experience_ids": self.learning_experience_ids,
            "feedback_ids": self.feedback_ids,
            "target_components": self.target_components,
            "implementation_steps": self.implementation_steps,
            "success_criteria": self.success_criteria,
            "status": self.status,
            "progress": self.progress,
            "results": self.results
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImprovementPlan':
        """
        Create from a dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            ImprovementPlan instance
        """
        plan = cls(
            plan_id=data.get("plan_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data 
                      else datetime.now(),
            name=data.get("name", ""),
            description=data.get("description", ""),
            priority=data.get("priority", 3)
        )
        
        plan.learning_experience_ids = data.get("learning_experience_ids", [])
        plan.feedback_ids = data.get("feedback_ids", [])
        plan.target_components = data.get("target_components", [])
        plan.implementation_steps = data.get("implementation_steps", [])
        plan.success_criteria = data.get("success_criteria", [])
        plan.status = data.get("status", "draft")
        plan.progress = data.get("progress", 0.0)
        plan.results = data.get("results")
        
        return plan
