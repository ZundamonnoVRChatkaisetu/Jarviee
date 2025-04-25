"""
AI Technology Adapters for Jarviee System.

This package contains adapter modules that enable the integration of various
AI technologies with the LLM core of the Jarviee system. Each adapter provides
a standardized interface for a specific AI technology to communicate with the
rest of the system.
"""

from .base import TechnologyAdapter
from .registry import AdapterRegistry

__all__ = ["TechnologyAdapter", "AdapterRegistry"]
