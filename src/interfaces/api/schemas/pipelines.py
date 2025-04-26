"""
Pipeline-related schema definitions for Jarviee API.

This module defines schemas for pipeline-related requests and responses.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class PipelineMethod(str, Enum):
    """Processing methods for pipelines."""
    
    SEQUENTIAL = "SEQUENTIAL"
    PARALLEL = "PARALLEL"
    HYBRID = "HYBRID"
    ADAPTIVE = "ADAPTIVE"


class PipelineRequest(BaseModel):
    """Request model for pipeline creation."""
    
    pipeline_id: str = Field(
        ...,
        description="ID for the new pipeline"
    )
    integration_ids: List[str] = Field(
        ...,
        description="List of integration IDs to include in the pipeline"
    )
    method: PipelineMethod = Field(
        default=PipelineMethod.SEQUENTIAL,
        description="Processing method for the pipeline"
    )


class PipelineInfo(BaseModel):
    """Detailed information about a pipeline."""
    
    pipeline_id: str = Field(
        ...,
        description="ID of the pipeline"
    )
    method: str = Field(
        ...,
        description="Processing method used by the pipeline"
    )
    integrations: List[str] = Field(
        ...,
        description="List of integration IDs included in the pipeline"
    )


class PipelineList(BaseModel):
    """List of pipelines with count information."""
    
    pipelines: Dict[str, PipelineInfo] = Field(
        ...,
        description="Map of pipeline IDs to pipeline information"
    )
    count: int = Field(
        ...,
        description="Total number of pipelines"
    )
