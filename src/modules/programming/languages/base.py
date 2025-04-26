"""
Base classes for programming language support

This module defines the foundation for Jarviee's programming language
support system, providing abstractions for language features, syntax,
and common operations required for code analysis and generation.
"""

import abc
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union


class LanguageFeature(Enum):
    """Features that a programming language might support."""
    STATIC_TYPING = auto()
    DYNAMIC_TYPING = auto()
    FUNCTIONAL = auto()
    OBJECT_ORIENTED = auto()
    PROCEDURAL = auto()
    CONCURRENT = auto()
    REFLECTIVE = auto()
    META_PROGRAMMING = auto()
    MEMORY_MANAGEMENT = auto()
    GARBAGE_COLLECTION = auto()
    EXCEPTION_HANDLING = auto()
    GENERICS = auto()
    CLOSURES = auto()
    FIRST_CLASS_FUNCTIONS = auto()
    TYPE_INFERENCE = auto()
    OPERATOR_OVERLOADING = auto()
    MODULE_SYSTEM = auto()
    PACKAGE_MANAGER = auto()


class SyntaxElement(Enum):
    """Common syntax elements in programming languages."""
    COMMENT = auto()
    FUNCTION_DEFINITION = auto()
    CLASS_DEFINITION = auto()
    IMPORT_STATEMENT = auto()
    VARIABLE_DECLARATION = auto()
    CONTROL_FLOW = auto()
    LOOP = auto()
    CONDITIONAL = auto()
    EXCEPTION_HANDLING = auto()
    STRING_LITERAL = auto()
    NUMBER_LITERAL = auto()
    BOOLEAN_LITERAL = auto()
    NULL_LITERAL = auto()
    ARRAY_LITERAL = auto()
    OBJECT_LITERAL = auto()
    FUNCTION_CALL = auto()
    METHOD_CALL = auto()
    PROPERTY_ACCESS = auto()
    OPERATOR = auto()
    ASSIGNMENT = auto()


class ProgrammingLanguage(abc.ABC):
    """
    Base class representing a programming language.
    
    This abstract class defines the interface that all specific programming
    language implementations must provide.
    """
    
    def __init__(self, name: str, file_extensions: List[str]):
        """
        Initialize a programming language.
        
        Args:
            name: The name of the programming language
            file_extensions: List of file extensions associated with this language
        """
        self.name = name
        self.file_extensions = file_extensions
        self.features: Set[LanguageFeature] = set()
        self.syntax_patterns: Dict[SyntaxElement, str] = {}
    
    @property
    def id(self) -> str:
        """Get a unique identifier for this language."""
        return self.name.lower().replace(' ', '_')
    
    def has_feature(self, feature: LanguageFeature) -> bool:
        """Check if this language supports a specific feature."""
        return feature in self.features
    
    def get_syntax_pattern(self, element: SyntaxElement) -> Optional[str]:
        """
        Get the regex pattern for a specific syntax element.
        
        Args:
            element: The syntax element to get a pattern for
            
        Returns:
            A regex pattern string if available, None otherwise
        """
        return self.syntax_patterns.get(element)
    
    @abc.abstractmethod
    def parse_code(self, code: str) -> Any:
        """
        Parse code into an abstract syntax tree.
        
        Args:
            code: The code to parse
            
        Returns:
            A language-specific AST representation
        """
        pass
    
    @abc.abstractmethod
    def format_code(self, code: str) -> str:
        """
        Format code according to language conventions.
        
        Args:
            code: The code to format
            
        Returns:
            Formatted code
        """
        pass
    
    @abc.abstractmethod
    def analyze_quality(self, code: str) -> Dict[str, Any]:
        """
        Analyze code quality.
        
        Args:
            code: The code to analyze
            
        Returns:
            A dictionary with quality metrics
        """
        pass
    
    @abc.abstractmethod
    def detect_issues(self, code: str) -> List[Dict[str, Any]]:
        """
        Detect issues in code.
        
        Args:
            code: The code to analyze
            
        Returns:
            A list of issues, each represented as a dictionary
        """
        pass
    
    @abc.abstractmethod
    def get_documentation_template(self, code_element: str) -> str:
        """
        Get a documentation template for a code element.
        
        Args:
            code_element: The code element to document
            
        Returns:
            A documentation template string
        """
        pass
    
    @abc.abstractmethod
    def extract_imports(self, code: str) -> List[str]:
        """
        Extract import statements from code.
        
        Args:
            code: The code to analyze
            
        Returns:
            A list of import statements
        """
        pass
    
    @abc.abstractmethod
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract function definitions from code.
        
        Args:
            code: The code to analyze
            
        Returns:
            A list of function information dictionaries
        """
        pass
    
    @abc.abstractmethod
    def extract_classes(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract class definitions from code.
        
        Args:
            code: The code to analyze
            
        Returns:
            A list of class information dictionaries
        """
        pass
    
    @abc.abstractmethod
    def get_code_completion(
        self, 
        code: str, 
        cursor_position: int
    ) -> List[Dict[str, Any]]:
        """
        Get code completion suggestions.
        
        Args:
            code: The code context
            cursor_position: Position of the cursor in the code
            
        Returns:
            A list of completion suggestions
        """
        pass
    
    @abc.abstractmethod
    def generate_test(self, code: str, function_name: Optional[str] = None) -> str:
        """
        Generate a test for code.
        
        Args:
            code: The code to test
            function_name: Optional name of a specific function to test
            
        Returns:
            Generated test code
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the language."""
        return f"{self.name} ({', '.join(self.file_extensions)})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the language."""
        features = ', '.join(f.name for f in self.features)
        return f"{self.name} ({', '.join(self.file_extensions)}) [{features}]"
