"""
Common schema definitions for Jarviee API.

This module defines common schemas used across the API.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error: str = Field(
        ...,
        description="Error message"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )


class SuccessResponse(BaseModel):
    """Schema for simple success responses."""
    
    status: str = Field(
        default="success",
        description="Status of the operation"
    )
    message: str = Field(
        ...,
        description="Success message"
    )
