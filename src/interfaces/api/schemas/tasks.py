"""
Task-related schema definitions for Jarviee API.

This module defines schemas for task requests and responses.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, root_validator


class TaskContent(BaseModel):
    """Base model for task content.
    
    This is a generic model that serves as a base for different task types.
    Specific task types may have additional validation.
    """
    
    class Config:
        extra = "allow"  # Allow additional fields based on task type


class CodeAnalysisContent(TaskContent):
    """Content model for code analysis tasks."""
    
    code: str = Field(
        ...,
        description="Code to be analyzed"
    )
    language: str = Field(
        ...,
        description="Programming language of the code"
    )
    analysis_type: str = Field(
        default="general",
        description="Type of analysis to perform (e.g., performance, security, quality)"
    )
    improvement_goal: Optional[str] = Field(
        default=None,
        description="Specific improvement goal (if any)"
    )


class CreativeProblemContent(TaskContent):
    """Content model for creative problem-solving tasks."""
    
    problem_statement: str = Field(
        ...,
        description="Description of the problem to solve"
    )
    constraints: List[str] = Field(
        default=[],
        description="Constraints to consider in the solution"
    )
    performance_criteria: List[str] = Field(
        default=[],
        description="Criteria for evaluating solution performance"
    )
    visualization_required: bool = Field(
        default=False,
        description="Whether visualization is required in the solution"
    )


class MultimodalAnalysisContent(TaskContent):
    """Content model for multimodal analysis tasks."""
    
    text_data: Optional[str] = Field(
        default=None,
        description="Text data for analysis"
    )
    image_data: Optional[str] = Field(
        default=None,
        description="Path or URL to image data"
    )
    audio_data: Optional[str] = Field(
        default=None,
        description="Path or URL to audio data"
    )
    analysis_goal: str = Field(
        ...,
        description="Goal of the multimodal analysis"
    )
    required_outputs: List[str] = Field(
        default=[],
        description="Specific outputs required from the analysis"
    )
    
    @root_validator
    def check_data_presence(cls, values):
        """Validate that at least one data type is provided."""
        if not any([values.get("text_data"), values.get("image_data"), values.get("audio_data")]):
            raise ValueError("At least one data type (text, image, or audio) must be provided")
        return values


class TaskRequest(BaseModel):
    """Request model for task execution."""
    
    task_type: str = Field(
        ..., 
        description="Type of task to process"
    )
    content: Dict[str, Any] = Field(
        ...,
        description="Task content, structure depends on task type"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context information for task processing"
    )


class StageResult(BaseModel):
    """Result of a single stage in a pipeline."""
    
    integration_id: str = Field(
        ...,
        description="ID of the integration that processed this stage"
    )
    status: str = Field(
        ...,
        description="Status of the stage processing"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message (if status is 'error')"
    )


class TaskResult(BaseModel):
    """Response model for task execution results."""
    
    status: str = Field(
        ...,
        description="Status of the task processing (success, error, etc.)"
    )
    task_type: str = Field(
        ...,
        description="Type of task that was processed"
    )
    processing_time_ms: Optional[float] = Field(
        default=None,
        description="Processing time in milliseconds"
    )
    integration: Optional[str] = Field(
        default=None,
        description="ID of the integration used (if direct integration processing)"
    )
    pipeline: Optional[str] = Field(
        default=None,
        description="ID of the pipeline used (if pipeline processing)"
    )
    stages: Optional[List[StageResult]] = Field(
        default=None,
        description="Results of individual pipeline stages (if pipeline processing)"
    )
    content: Dict[str, Any] = Field(
        default={},
        description="Task result content"
    )
