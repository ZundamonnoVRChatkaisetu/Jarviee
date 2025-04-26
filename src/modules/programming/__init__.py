"""
Programming Module for Jarviee

This module provides specialized programming support capabilities including:
- Code understanding and analysis
- Code generation with optimization
- Debugging assistance
- Integration with development environments
- Programming knowledge representation and application
"""

from .code_analyzer import CodeAnalyzer
from .code_generator import CodeGenerator
from .debugger import DebuggingEngine
from .ide_connector import IDEConnector
from .code_repository import CodeRepository
from .knowledge_assistant import ProgrammingKnowledgeAssistant

__all__ = [
    'CodeAnalyzer',
    'CodeGenerator',
    'DebuggingEngine',
    'IDEConnector',
    'CodeRepository',
    'ProgrammingKnowledgeAssistant'
]
