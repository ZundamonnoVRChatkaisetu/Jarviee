"""
Integration Module for Jarviee System.

This package contains interface definitions and base classes for integrating
different AI technologies with the LLM core. It provides standardized communication
protocols and data conversion utilities to enable seamless interoperability.
"""

from .base import AIComponent, ComponentType, IntegrationMessage
from .registry import ComponentRegistry

__all__ = ["AIComponent", "ComponentType", "IntegrationMessage", "ComponentRegistry"]
