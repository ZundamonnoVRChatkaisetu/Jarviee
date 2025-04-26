"""
Integration-related schema definitions for Jarviee API.

This module defines schemas for integration-related requests and responses.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class IntegrationMetrics(BaseModel):
    """Metrics for an integration."""
    
    requests: int = Field(
        default=0,
        description="Number of requests processed"
    )
    successful_integrations: int = Field(
        default=0,
        description="Number of successful integrations"
    )
    failed_integrations: int = Field(
        default=0,
        description="Number of failed integrations"
    )
    avg_response_time_ms: float = Field(
        default=0.0,
        description="Average response time in milliseconds"
    )
    last_used_timestamp: Optional[float] = Field(
        default=None,
        description="Timestamp of last usage"
    )


class IntegrationInfo(BaseModel):
    """Detailed information about an integration."""
    
    integration_id: str = Field(
        ...,
        description="Unique identifier for the integration"
    )
    integration_type: str = Field(
        ...,
        description="Type of AI technology integration"
    )
    llm_component_id: str = Field(
        ...,
        description="ID of the LLM component"
    )
    technology_component_id: str = Field(
        ...,
        description="ID of the other AI technology component"
    )
    priority: str = Field(
        ...,
        description="Priority level for this integration"
    )
    method: str = Field(
        ...,
        description="Method used for this integration"
    )
    capabilities: List[str] = Field(
        default=[],
        description="List of capabilities provided by this integration"
    )
    active: bool = Field(
        ...,
        description="Whether the integration is active"
    )
    metrics: IntegrationMetrics = Field(
        default_factory=IntegrationMetrics,
        description="Integration metrics"
    )


class IntegrationSummary(BaseModel):
    """Summary information about an integration."""
    
    integration_id: str = Field(
        ...,
        description="Unique identifier for the integration"
    )
    integration_type: str = Field(
        ...,
        description="Type of AI technology integration"
    )
    capabilities: List[str] = Field(
        default=[],
        description="List of capabilities provided by this integration"
    )
    active: bool = Field(
        ...,
        description="Whether the integration is active"
    )


class IntegrationList(BaseModel):
    """List of integrations with count information."""
    
    integrations: Dict[str, IntegrationSummary] = Field(
        ...,
        description="Map of integration IDs to integration information"
    )
    count: int = Field(
        ...,
        description="Total number of integrations"
    )
    active_count: int = Field(
        ...,
        description="Number of active integrations"
    )
