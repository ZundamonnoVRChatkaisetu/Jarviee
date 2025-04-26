"""
API Interface Module for Jarviee System.

This module provides a RESTful API for interacting with the Jarviee system,
allowing external applications to access and control various AI technology
integrations through HTTP endpoints.
"""

from .server import create_app

__all__ = ["create_app"]
