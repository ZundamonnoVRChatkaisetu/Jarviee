"""
Programming Languages Support Module

This module provides language-specific support for programming
activities including analysis, generation, and optimization.
"""

from .base import ProgrammingLanguage, LanguageFeature, SyntaxElement
from .python import PythonLanguage
from .javascript import JavaScriptLanguage
from .typescript import TypeScriptLanguage
from .registry import LanguageRegistry

__all__ = [
    'ProgrammingLanguage',
    'LanguageFeature',
    'SyntaxElement',
    'PythonLanguage',
    'JavaScriptLanguage',
    'TypeScriptLanguage',
    'LanguageRegistry'
]
