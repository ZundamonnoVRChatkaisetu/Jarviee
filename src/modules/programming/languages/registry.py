"""
Programming Language Registry

This module implements a registry for programming languages supported by
Jarviee, allowing for dynamic registration and lookup of language handlers.
"""

import logging
from typing import Dict, List, Optional, Type, Union

from .base import ProgrammingLanguage


class LanguageRegistry:
    """
    Registry for programming languages supported by Jarviee.
    
    This class provides a centralized registry for programming language
    implementations, allowing for lookup by name or file extension.
    """
    
    _instance = None
    
    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(LanguageRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the language registry."""
        if self._initialized:
            return
        
        self.logger = logging.getLogger("language_registry")
        self.languages: Dict[str, ProgrammingLanguage] = {}
        self.extension_map: Dict[str, str] = {}  # Maps file extensions to language IDs
        self._initialized = True
    
    def register_language(self, language: ProgrammingLanguage) -> None:
        """
        Register a programming language.
        
        Args:
            language: The programming language to register
        """
        language_id = language.id
        
        if language_id in self.languages:
            self.logger.warning(
                f"Overwriting existing language with ID '{language_id}'")
        
        self.languages[language_id] = language
        
        # Register file extensions
        for ext in language.file_extensions:
            if ext in self.extension_map and self.extension_map[ext] != language_id:
                self.logger.warning(
                    f"Extension '{ext}' was previously mapped to "
                    f"'{self.extension_map[ext]}', now mapped to '{language_id}'")
            
            self.extension_map[ext] = language_id
        
        self.logger.info(
            f"Registered language '{language.name}' with extensions: "
            f"{', '.join(language.file_extensions)}")
    
    def get_language(self, language_id: str) -> Optional[ProgrammingLanguage]:
        """
        Get a programming language by ID.
        
        Args:
            language_id: The ID of the language to retrieve
            
        Returns:
            The language if found, None otherwise
        """
        return self.languages.get(language_id.lower())
    
    def get_language_by_name(self, name: str) -> Optional[ProgrammingLanguage]:
        """
        Get a programming language by name.
        
        Args:
            name: The name of the language to retrieve
            
        Returns:
            The language if found, None otherwise
        """
        name_lower = name.lower()
        
        # Exact ID match
        if name_lower in self.languages:
            return self.languages[name_lower]
        
        # Exact name match
        for language in self.languages.values():
            if language.name.lower() == name_lower:
                return language
        
        # Partial name match
        for language in self.languages.values():
            if name_lower in language.name.lower():
                return language
        
        return None
    
    def get_language_for_file(self, filename: str) -> Optional[ProgrammingLanguage]:
        """
        Get the appropriate language for a file.
        
        Args:
            filename: The name of the file
            
        Returns:
            The language if found, None otherwise
        """
        # Extract extension (handle cases with multiple dots)
        parts = filename.split('.')
        if len(parts) > 1:
            ext = parts[-1].lower()
            
            # Check for compound extensions like .d.ts
            if len(parts) > 2 and parts[-2].lower() + '.' + ext in self.extension_map:
                compound_ext = parts[-2].lower() + '.' + ext
                language_id = self.extension_map.get(compound_ext)
                if language_id:
                    return self.languages.get(language_id)
            
            # Check for simple extension
            if ext in self.extension_map:
                language_id = self.extension_map[ext]
                return self.languages.get(language_id)
        
        return None
    
    def get_language_for_code(self, code: str) -> Optional[ProgrammingLanguage]:
        """
        Attempt to detect the language for a code snippet.
        
        Args:
            code: The code to analyze
            
        Returns:
            The detected language if found, None otherwise
        """
        # This is a placeholder for more sophisticated language detection.
        # In a real implementation, this would use statistical analysis,
        # keyword detection, and other heuristics.
        
        # Simple keyword-based detection as a fallback
        code_lower = code.lower()
        
        # Python detection
        if ('def ' in code_lower or 'import ' in code_lower) and ':' in code and '#' in code:
            return self.get_language_by_name('python')
        
        # JavaScript/TypeScript detection
        if ('function' in code_lower or 'const ' in code_lower or 'let ' in code_lower) and '{' in code and '}' in code:
            if 'interface ' in code_lower or ': ' in code_lower and ('string' in code_lower or 'number' in code_lower):
                return self.get_language_by_name('typescript')
            return self.get_language_by_name('javascript')
        
        # Further language detection logic would go here
        
        return None
    
    def list_languages(self) -> List[ProgrammingLanguage]:
        """
        Get a list of all registered languages.
        
        Returns:
            List of all registered programming languages
        """
        return list(self.languages.values())
    
    def clear(self) -> None:
        """Clear all registered languages."""
        self.languages.clear()
        self.extension_map.clear()
        self.logger.info("Cleared all registered languages")
