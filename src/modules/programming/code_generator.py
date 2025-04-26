"""
Code Generation Engine for Jarviee

This module provides advanced code generation capabilities, including:
- Natural language to code translation
- Code completion and suggestion
- Refactoring and optimization
- Test generation
- Documentation generation
- Boilerplate generation
"""

import logging
import re
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from ..reasoning.symbolic.kb_interface import KnowledgeBaseInterface
from ...core.llm.engine import LLMEngine
from ...core.knowledge.query_engine import QueryEngine
from .languages.base import ProgrammingLanguage
from .languages.registry import LanguageRegistry


class CodeGeneration(Enum):
    """Types of code generation."""
    IMPLEMENTATION = auto()
    FUNCTION = auto()
    CLASS = auto()
    TEST = auto()
    DOCUMENTATION = auto()
    REFACTORING = auto()
    OPTIMIZATION = auto()
    COMPLETION = auto()
    BOILERPLATE = auto()


class GenerationSettings:
    """Settings for code generation."""
    
    def __init__(
        self,
        style: str = "standard",
        quality: str = "production",
        complexity: str = "medium",
        verbosity: str = "medium",
        documentation: bool = True,
        comments: bool = True,
        type_hints: bool = True,
        error_handling: bool = True,
        tests: bool = False,
        examples: bool = False
    ):
        """
        Initialize generation settings.
        
        Args:
            style: Coding style ('standard', 'concise', 'verbose')
            quality: Code quality target ('production', 'prototype', 'example')
            complexity: Complexity level ('simple', 'medium', 'advanced')
            verbosity: Comment verbosity ('minimal', 'medium', 'detailed')
            documentation: Whether to include docstrings
            comments: Whether to include inline comments
            type_hints: Whether to include type hints
            error_handling: Whether to include error handling
            tests: Whether to include tests
            examples: Whether to include usage examples
        """
        self.style = style
        self.quality = quality
        self.complexity = complexity
        self.verbosity = verbosity
        self.documentation = documentation
        self.comments = comments
        self.type_hints = type_hints
        self.error_handling = error_handling
        self.tests = tests
        self.examples = examples
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "style": self.style,
            "quality": self.quality,
            "complexity": self.complexity,
            "verbosity": self.verbosity,
            "documentation": self.documentation,
            "comments": self.comments,
            "type_hints": self.type_hints,
            "error_handling": self.error_handling,
            "tests": self.tests,
            "examples": self.examples
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationSettings':
        """Create settings from dictionary."""
        return cls(
            style=data.get("style", "standard"),
            quality=data.get("quality", "production"),
            complexity=data.get("complexity", "medium"),
            verbosity=data.get("verbosity", "medium"),
            documentation=data.get("documentation", True),
            comments=data.get("comments", True),
            type_hints=data.get("type_hints", True),
            error_handling=data.get("error_handling", True),
            tests=data.get("tests", False),
            examples=data.get("examples", False)
        )


class CodeGenerator:
    """
    Code generation engine for Jarviee.
    
    This class provides various code generation capabilities powered by
    the LLM engine and enhanced with programming-specific knowledge.
    """
    
    def __init__(
        self,
        llm_engine: Optional[LLMEngine] = None,
        query_engine: Optional[QueryEngine] = None,
        kb_interface: Optional[KnowledgeBaseInterface] = None
    ):
        """
        Initialize the code generator.
        
        Args:
            llm_engine: LLM engine for code generation
            query_engine: Query engine for knowledge retrieval
            kb_interface: Knowledge base interface for symbolic reasoning
        """
        self.logger = logging.getLogger("code_generator")
        self.language_registry = LanguageRegistry()
        self.llm_engine = llm_engine
        self.query_engine = query_engine
        self.kb_interface = kb_interface
        
        # Default generation settings
        self.default_settings = GenerationSettings()
    
    def set_llm_engine(self, llm_engine: LLMEngine) -> None:
        """
        Set the LLM engine.
        
        Args:
            llm_engine: LLM engine to use
        """
        self.llm_engine = llm_engine
    
    def set_query_engine(self, query_engine: QueryEngine) -> None:
        """
        Set the query engine.
        
        Args:
            query_engine: Query engine to use
        """
        self.query_engine = query_engine
    
    def set_kb_interface(self, kb_interface: KnowledgeBaseInterface) -> None:
        """
        Set the knowledge base interface.
        
        Args:
            kb_interface: Knowledge base interface to use
        """
        self.kb_interface = kb_interface
    
    def generate_from_description(
        self,
        description: str,
        language: str,
        generation_type: CodeGeneration = CodeGeneration.IMPLEMENTATION,
        settings: Optional[GenerationSettings] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate code from a natural language description.
        
        Args:
            description: Natural language description of the code to generate
            language: Target programming language
            generation_type: Type of code to generate
            settings: Code generation settings
            additional_context: Additional context for generation
            
        Returns:
            Generated code and metadata
        """
        if self.llm_engine is None:
            raise ValueError("LLM engine is required for code generation")
        
        # Use default settings if none provided
        if settings is None:
            settings = self.default_settings
        
        # Get the language handler
        language_handler = self.language_registry.get_language_by_name(language)
        if language_handler is None:
            return {
                "success": False,
                "error": f"Unsupported language: {language}",
                "code": ""
            }
        
        # Retrieve relevant knowledge if query engine is available
        knowledge = []
        if self.query_engine is not None:
            try:
                # Query for programming knowledge related to the task
                query_result = self.query_engine.query(
                    f"{description} {language} programming",
                    limit=5,
                    context=additional_context
                )
                knowledge = query_result.get("results", [])
            except Exception as e:
                self.logger.warning(f"Error retrieving knowledge: {str(e)}")
        
        # Prepare the context for LLM
        context = {
            "description": description,
            "language": language,
            "generation_type": generation_type.name,
            "settings": settings.to_dict(),
            "knowledge": knowledge,
            "additional_context": additional_context or {}
        }
        
        # Generate the code
        result = self._generate_code(context, language_handler)
        
        # Format the code if generation was successful
        if result.get("success", False) and result.get("code"):
            try:
                result["code"] = language_handler.format_code(result["code"])
            except Exception as e:
                self.logger.warning(f"Error formatting code: {str(e)}")
        
        return result
    
    def complete_code(
        self,
        partial_code: str,
        language: Optional[str] = None,
        cursor_position: Optional[int] = None,
        settings: Optional[GenerationSettings] = None
    ) -> Dict[str, Any]:
        """
        Complete partial code.
        
        Args:
            partial_code: Partial code to complete
            language: Programming language (if None, will be auto-detected)
            cursor_position: Position of the cursor in the code
            settings: Code generation settings
            
        Returns:
            Completed code and metadata
        """
        if self.llm_engine is None:
            raise ValueError("LLM engine is required for code completion")
        
        # Use default settings if none provided
        if settings is None:
            settings = self.default_settings
        
        # Auto-detect language if not provided
        language_handler = None
        if language:
            language_handler = self.language_registry.get_language_by_name(language)
        else:
            language_handler = self.language_registry.get_language_for_code(partial_code)
        
        if language_handler is None:
            return {
                "success": False,
                "error": "Could not determine the programming language",
                "code": partial_code
            }
        
        # Set cursor position to the end if not provided
        if cursor_position is None:
            cursor_position = len(partial_code)
        
        # Extract the code before and after the cursor
        code_before = partial_code[:cursor_position]
        code_after = partial_code[cursor_position:]
        
        # Analyze the code structure to provide context
        try:
            functions = language_handler.extract_functions(partial_code)
            classes = language_handler.extract_classes(partial_code)
            imports = language_handler.extract_imports(partial_code)
        except Exception as e:
            self.logger.warning(f"Error analyzing code structure: {str(e)}")
            functions = []
            classes = []
            imports = []
        
        # Prepare the context for LLM
        context = {
            "partial_code": partial_code,
            "code_before": code_before,
            "code_after": code_after,
            "language": language_handler.name,
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "settings": settings.to_dict(),
            "generation_type": CodeGeneration.COMPLETION.name
        }
        
        # Generate the completion
        result = self._generate_completion(context, language_handler)
        
        return result
    
    def refactor_code(
        self,
        code: str,
        refactoring_description: str,
        language: Optional[str] = None,
        settings: Optional[GenerationSettings] = None
    ) -> Dict[str, Any]:
        """
        Refactor code based on description.
        
        Args:
            code: Code to refactor
            refactoring_description: Description of the refactoring to perform
            language: Programming language (if None, will be auto-detected)
            settings: Code generation settings
            
        Returns:
            Refactored code and metadata
        """
        if self.llm_engine is None:
            raise ValueError("LLM engine is required for code refactoring")
        
        # Use default settings if none provided
        if settings is None:
            settings = self.default_settings
        
        # Auto-detect language if not provided
        language_handler = None
        if language:
            language_handler = self.language_registry.get_language_by_name(language)
        else:
            language_handler = self.language_registry.get_language_for_code(code)
        
        if language_handler is None:
            return {
                "success": False,
                "error": "Could not determine the programming language",
                "code": code
            }
        
        # Analyze the code structure to provide context
        try:
            functions = language_handler.extract_functions(code)
            classes = language_handler.extract_classes(code)
        except Exception as e:
            self.logger.warning(f"Error analyzing code structure: {str(e)}")
            functions = []
            classes = []
        
        # Prepare the context for LLM
        context = {
            "original_code": code,
            "refactoring_description": refactoring_description,
            "language": language_handler.name,
            "functions": functions,
            "classes": classes,
            "settings": settings.to_dict(),
            "generation_type": CodeGeneration.REFACTORING.name
        }
        
        # Generate the refactored code
        result = self._generate_refactoring(context, language_handler)
        
        # Format the code if refactoring was successful
        if result.get("success", False) and result.get("code"):
            try:
                result["code"] = language_handler.format_code(result["code"])
            except Exception as e:
                self.logger.warning(f"Error formatting code: {str(e)}")
        
        return result
    
    def optimize_code(
        self,
        code: str,
        optimization_targets: List[str],
        language: Optional[str] = None,
        settings: Optional[GenerationSettings] = None
    ) -> Dict[str, Any]:
        """
        Optimize code for specific targets.
        
        Args:
            code: Code to optimize
            optimization_targets: List of optimization targets
                (e.g., 'performance', 'memory', 'readability')
            language: Programming language (if None, will be auto-detected)
            settings: Code generation settings
            
        Returns:
            Optimized code and metadata
        """
        if self.llm_engine is None:
            raise ValueError("LLM engine is required for code optimization")
        
        # Use default settings if none provided
        if settings is None:
            settings = self.default_settings
        
        # Auto-detect language if not provided
        language_handler = None
        if language:
            language_handler = self.language_registry.get_language_by_name(language)
        else:
            language_handler = self.language_registry.get_language_for_code(code)
        
        if language_handler is None:
            return {
                "success": False,
                "error": "Could not determine the programming language",
                "code": code
            }
        
        # Prepare the context for LLM
        context = {
            "original_code": code,
            "optimization_targets": optimization_targets,
            "language": language_handler.name,
            "settings": settings.to_dict(),
            "generation_type": CodeGeneration.OPTIMIZATION.name
        }
        
        # Generate the optimized code
        result = self._generate_optimization(context, language_handler)
        
        # Format the code if optimization was successful
        if result.get("success", False) and result.get("code"):
            try:
                result["code"] = language_handler.format_code(result["code"])
            except Exception as e:
                self.logger.warning(f"Error formatting code: {str(e)}")
        
        return result
    
    def generate_tests(
        self,
        code: str,
        test_framework: Optional[str] = None,
        language: Optional[str] = None,
        function_name: Optional[str] = None,
        settings: Optional[GenerationSettings] = None
    ) -> Dict[str, Any]:
        """
        Generate tests for code.
        
        Args:
            code: Code to generate tests for
            test_framework: Test framework to use (e.g., 'unittest', 'pytest')
            language: Programming language (if None, will be auto-detected)
            function_name: Name of the function to test (if None, all functions)
            settings: Code generation settings
            
        Returns:
            Generated tests and metadata
        """
        if self.llm_engine is None:
            raise ValueError("LLM engine is required for test generation")
        
        # Use default settings if none provided
        if settings is None:
            settings = self.default_settings
        
        # Auto-detect language if not provided
        language_handler = None
        if language:
            language_handler = self.language_registry.get_language_by_name(language)
        else:
            language_handler = self.language_registry.get_language_for_code(code)
        
        if language_handler is None:
            return {
                "success": False,
                "error": "Could not determine the programming language",
                "code": ""
            }
        
        # Try to use the language's built-in test generator if available
        try:
            test_code = language_handler.generate_test(code, function_name)
            
            return {
                "success": True,
                "code": test_code,
                "message": f"Generated tests using {language_handler.name}'s built-in test generator"
            }
        except Exception as e:
            self.logger.warning(f"Error using built-in test generator: {str(e)}")
        
        # Fall back to LLM-based test generation
        context = {
            "code": code,
            "test_framework": test_framework,
            "language": language_handler.name,
            "function_name": function_name,
            "settings": settings.to_dict(),
            "generation_type": CodeGeneration.TEST.name
        }
        
        # Generate the tests
        result = self._generate_tests(context, language_handler)
        
        # Format the code if generation was successful
        if result.get("success", False) and result.get("code"):
            try:
                result["code"] = language_handler.format_code(result["code"])
            except Exception as e:
                self.logger.warning(f"Error formatting code: {str(e)}")
        
        return result
    
    def generate_documentation(
        self,
        code: str,
        doc_style: Optional[str] = None,
        language: Optional[str] = None,
        settings: Optional[GenerationSettings] = None
    ) -> Dict[str, Any]:
        """
        Generate documentation for code.
        
        Args:
            code: Code to document
            doc_style: Documentation style to use (e.g., 'jsdoc', 'sphinx')
            language: Programming language (if None, will be auto-detected)
            settings: Code generation settings
            
        Returns:
            Documented code and metadata
        """
        if self.llm_engine is None:
            raise ValueError("LLM engine is required for documentation generation")
        
        # Use default settings if none provided
        if settings is None:
            settings = self.default_settings
        
        # Auto-detect language if not provided
        language_handler = None
        if language:
            language_handler = self.language_registry.get_language_by_name(language)
        else:
            language_handler = self.language_registry.get_language_for_code(code)
        
        if language_handler is None:
            return {
                "success": False,
                "error": "Could not determine the programming language",
                "code": code
            }
        
        # Prepare the context for LLM
        context = {
            "code": code,
            "doc_style": doc_style,
            "language": language_handler.name,
            "settings": settings.to_dict(),
            "generation_type": CodeGeneration.DOCUMENTATION.name
        }
        
        # Generate the documentation
        result = self._generate_documentation(context, language_handler)
        
        # Format the code if generation was successful
        if result.get("success", False) and result.get("code"):
            try:
                result["code"] = language_handler.format_code(result["code"])
            except Exception as e:
                self.logger.warning(f"Error formatting code: {str(e)}")
        
        return result
    
    def generate_boilerplate(
        self,
        project_type: str,
        language: str,
        features: List[str],
        settings: Optional[GenerationSettings] = None
    ) -> Dict[str, Any]:
        """
        Generate project boilerplate code.
        
        Args:
            project_type: Type of project (e.g., 'web', 'cli', 'library')
            language: Target programming language
            features: List of features to include
            settings: Code generation settings
            
        Returns:
            Dictionary of generated files and metadata
        """
        if self.llm_engine is None:
            raise ValueError("LLM engine is required for boilerplate generation")
        
        # Use default settings if none provided
        if settings is None:
            settings = self.default_settings
        
        # Get the language handler
        language_handler = self.language_registry.get_language_by_name(language)
        if language_handler is None:
            return {
                "success": False,
                "error": f"Unsupported language: {language}",
                "files": {}
            }
        
        # Prepare the context for LLM
        context = {
            "project_type": project_type,
            "language": language_handler.name,
            "features": features,
            "settings": settings.to_dict(),
            "generation_type": CodeGeneration.BOILERPLATE.name
        }
        
        # Generate the boilerplate
        result = self._generate_boilerplate(context, language_handler)
        
        # Format the code in each file if generation was successful
        if result.get("success", False) and result.get("files"):
            try:
                for filename, code in result["files"].items():
                    result["files"][filename] = language_handler.format_code(code)
            except Exception as e:
                self.logger.warning(f"Error formatting code: {str(e)}")
        
        return result
    
    def _generate_code(
        self, 
        context: Dict[str, Any],
        language_handler: ProgrammingLanguage
    ) -> Dict[str, Any]:
        """
        Generate code using LLM.
        
        Args:
            context: Context for code generation
            language_handler: Language handler
            
        Returns:
            Generated code and metadata
        """
        if self.llm_engine is None:
            raise ValueError("LLM engine is required for code generation")
        
        # Build the prompt
        prompt = self._build_code_generation_prompt(context, language_handler)
        
        try:
            # Generate code using LLM
            response = self.llm_engine.generate(prompt)
            
            # Extract the code from the response
            code = self._extract_code_from_response(response, language_handler.name)
            
            # Check if code was successfully extracted
            if code:
                return {
                    "success": True,
                    "code": code,
                    "language": language_handler.name
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to extract code from LLM response",
                    "code": ""
                }
        
        except Exception as e:
            self.logger.error(f"Error generating code: {str(e)}")
            return {
                "success": False,
                "error": f"Error generating code: {str(e)}",
                "code": ""
            }
    
    def _generate_completion(
        self, 
        context: Dict[str, Any],
        language_handler: ProgrammingLanguage
    ) -> Dict[str, Any]:
        """
        Generate code completion using LLM.
        
        Args:
            context: Context for code completion
            language_handler: Language handler
            
        Returns:
            Completed code and metadata
        """
        if self.llm_engine is None:
            raise ValueError("LLM engine is required for code completion")
        
        # Build the prompt
        prompt = self._build_completion_prompt(context, language_handler)
        
        try:
            # Generate completion using LLM
            response = self.llm_engine.generate(prompt)
            
            # Extract the completion from the response
            completion = self._extract_completion_from_response(response, context)
            
            if completion:
                # Combine the code before the cursor, the completion, and the code after
                code_before = context.get("code_before", "")
                code_after = context.get("code_after", "")
                completed_code = code_before + completion + code_after
                
                return {
                    "success": True,
                    "code": completed_code,
                    "completion": completion,
                    "language": language_handler.name
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to extract completion from LLM response",
                    "code": context.get("partial_code", ""),
                    "completion": ""
                }
        
        except Exception as e:
            self.logger.error(f"Error generating completion: {str(e)}")
            return {
                "success": False,
                "error": f"Error generating completion: {str(e)}",
                "code": context.get("partial_code", ""),
                "completion": ""
            }
    
    def _generate_refactoring(
        self, 
        context: Dict[str, Any],
        language_handler: ProgrammingLanguage
    ) -> Dict[str, Any]:
        """
        Generate code refactoring using LLM.
        
        Args:
            context: Context for code refactoring
            language_handler: Language handler
            
        Returns:
            Refactored code and metadata
        """
        if self.llm_engine is None:
            raise ValueError("LLM engine is required for code refactoring")
        
        # Build the prompt
        prompt = self._build_refactoring_prompt(context, language_handler)
        
        try:
            # Generate refactoring using LLM
            response = self.llm_engine.generate(prompt)
            
            # Extract the refactored code from the response
            refactored_code = self._extract_code_from_response(response, language_handler.name)
            
            # Check if refactored code was successfully extracted
            if refactored_code:
                return {
                    "success": True,
                    "code": refactored_code,
                    "original_code": context.get("original_code", ""),
                    "language": language_handler.name
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to extract refactored code from LLM response",
                    "code": context.get("original_code", "")
                }
        
        except Exception as e:
            self.logger.error(f"Error generating refactoring: {str(e)}")
            return {
                "success": False,
                "error": f"Error generating refactoring: {str(e)}",
                "code": context.get("original_code", "")
            }
    
    def _generate_optimization(
        self, 
        context: Dict[str, Any],
        language_handler: ProgrammingLanguage
    ) -> Dict[str, Any]:
        """
        Generate code optimization using LLM.
        
        Args:
            context: Context for code optimization
            language_handler: Language handler
            
        Returns:
            Optimized code and metadata
        """
        if self.llm_engine is None:
            raise ValueError("LLM engine is required for code optimization")
        
        # Build the prompt
        prompt = self._build_optimization_prompt(context, language_handler)
        
        try:
            # Generate optimization using LLM
            response = self.llm_engine.generate(prompt)
            
            # Extract the optimized code from the response
            optimized_code = self._extract_code_from_response(response, language_handler.name)
            
            # Check if optimized code was successfully extracted
            if optimized_code:
                return {
                    "success": True,
                    "code": optimized_code,
                    "original_code": context.get("original_code", ""),
                    "optimization_targets": context.get("optimization_targets", []),
                    "language": language_handler.name
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to extract optimized code from LLM response",
                    "code": context.get("original_code", "")
                }
        
        except Exception as e:
            self.logger.error(f"Error generating optimization: {str(e)}")
            return {
                "success": False,
                "error": f"Error generating optimization: {str(e)}",
                "code": context.get("original_code", "")
            }
    
    def _generate_tests(
        self, 
        context: Dict[str, Any],
        language_handler: ProgrammingLanguage
    ) -> Dict[str, Any]:
        """
        Generate tests using LLM.
        
        Args:
            context: Context for test generation
            language_handler: Language handler
            
        Returns:
            Generated tests and metadata
        """
        if self.llm_engine is None:
            raise ValueError("LLM engine is required for test generation")
        
        # Build the prompt
        prompt = self._build_test_prompt(context, language_handler)
        
        try:
            # Generate tests using LLM
            response = self.llm_engine.generate(prompt)
            
            # Extract the tests from the response
            tests = self._extract_code_from_response(response, language_handler.name)
            
            # Check if tests were successfully extracted
            if tests:
                return {
                    "success": True,
                    "code": tests,
                    "test_framework": context.get("test_framework"),
                    "language": language_handler.name
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to extract tests from LLM response",
                    "code": ""
                }
        
        except Exception as e:
            self.logger.error(f"Error generating tests: {str(e)}")
            return {
                "success": False,
                "error": f"Error generating tests: {str(e)}",
                "code": ""
            }
    
    def _generate_documentation(
        self, 
        context: Dict[str, Any],
        language_handler: ProgrammingLanguage
    ) -> Dict[str, Any]:
        """
        Generate documentation using LLM.
        
        Args:
            context: Context for documentation generation
            language_handler: Language handler
            
        Returns:
            Documented code and metadata
        """
        if self.llm_engine is None:
            raise ValueError("LLM engine is required for documentation generation")
        
        # Build the prompt
        prompt = self._build_documentation_prompt(context, language_handler)
        
        try:
            # Generate documentation using LLM
            response = self.llm_engine.generate(prompt)
            
            # Extract the documented code from the response
            documented_code = self._extract_code_from_response(response, language_handler.name)
            
            # Check if documented code was successfully extracted
            if documented_code:
                return {
                    "success": True,
                    "code": documented_code,
                    "original_code": context.get("code", ""),
                    "doc_style": context.get("doc_style"),
                    "language": language_handler.name
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to extract documented code from LLM response",
                    "code": context.get("code", "")
                }
        
        except Exception as e:
            self.logger.error(f"Error generating documentation: {str(e)}")
            return {
                "success": False,
                "error": f"Error generating documentation: {str(e)}",
                "code": context.get("code", "")
            }
    
    def _generate_boilerplate(
        self, 
        context: Dict[str, Any],
        language_handler: ProgrammingLanguage
    ) -> Dict[str, Any]:
        """
        Generate project boilerplate using LLM.
        
        Args:
            context: Context for boilerplate generation
            language_handler: Language handler
            
        Returns:
            Dictionary of generated files and metadata
        """
        if self.llm_engine is None:
            raise ValueError("LLM engine is required for boilerplate generation")
        
        # Build the prompt
        prompt = self._build_boilerplate_prompt(context, language_handler)
        
        try:
            # Generate boilerplate using LLM
            response = self.llm_engine.generate(prompt)
            
            # Extract the files from the response
            files = self._extract_files_from_response(response, language_handler.name)
            
            # Check if files were successfully extracted
            if files:
                return {
                    "success": True,
                    "files": files,
                    "project_type": context.get("project_type"),
                    "language": language_handler.name,
                    "features": context.get("features", [])
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to extract files from LLM response",
                    "files": {}
                }
        
        except Exception as e:
            self.logger.error(f"Error generating boilerplate: {str(e)}")
            return {
                "success": False,
                "error": f"Error generating boilerplate: {str(e)}",
                "files": {}
            }
    
    def _build_code_generation_prompt(
        self, 
        context: Dict[str, Any],
        language_handler: ProgrammingLanguage
    ) -> str:
        """
        Build prompt for code generation.
        
        Args:
            context: Context for code generation
            language_handler: Language handler
            
        Returns:
            Prompt string
        """
        description = context.get("description", "")
        language = context.get("language", "")
        generation_type = context.get("generation_type", "")
        settings = context.get("settings", {})
        knowledge = context.get("knowledge", [])
        additional_context = context.get("additional_context", {})
        
        # Construct the prompt
        prompt = f"Generate {language} code for the following description:\n\n"
        prompt += f"{description}\n\n"
        
        # Add information about the type of code to generate
        prompt += f"Type of code to generate: {generation_type}\n\n"
        
        # Add settings
        prompt += "Settings:\n"
        for key, value in settings.items():
            prompt += f"- {key}: {value}\n"
        prompt += "\n"
        
        # Add relevant knowledge if available
        if knowledge:
            prompt += "Relevant knowledge:\n"
            for item in knowledge:
                prompt += f"- {item.get('content', '')}\n"
            prompt += "\n"
        
        # Add additional context if available
        if additional_context:
            prompt += "Additional context:\n"
            for key, value in additional_context.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n"
        
        # Add instructions for the response format
        prompt += f"Please provide only the {language} code in your response, without explanations. "
        prompt += f"The code should be well-structured, follow {language} best practices, "
        prompt += "and include appropriate comments and documentation."
        
        return prompt
    
    def _build_completion_prompt(
        self, 
        context: Dict[str, Any],
        language_handler: ProgrammingLanguage
    ) -> str:
        """
        Build prompt for code completion.
        
        Args:
            context: Context for code completion
            language_handler: Language handler
            
        Returns:
            Prompt string
        """
        partial_code = context.get("partial_code", "")
        code_before = context.get("code_before", "")
        code_after = context.get("code_after", "")
        language = context.get("language", "")
        functions = context.get("functions", [])
        classes = context.get("classes", [])
        imports = context.get("imports", [])
        
        # Construct the prompt
        prompt = f"Complete the following {language} code:\n\n"
        
        # Add the partial code
        prompt += f"```{language}\n{partial_code}\n```\n\n"
        
        # Add information about cursor position
        prompt += "Cursor position is indicated by [CURSOR] in the code below:\n\n"
        prompt += f"```{language}\n{code_before}[CURSOR]{code_after}\n```\n\n"
        
        # Add information about functions and classes if available
        if functions:
            prompt += "Functions in the code:\n"
            for func in functions:
                prompt += f"- {func.get('name', '')}\n"
            prompt += "\n"
        
        if classes:
            prompt += "Classes in the code:\n"
            for cls in classes:
                prompt += f"- {cls.get('name', '')}\n"
            prompt += "\n"
        
        if imports:
            prompt += "Imports in the code:\n"
            for imp in imports:
                prompt += f"- {imp}\n"
            prompt += "\n"
        
        # Add instructions for the response format
        prompt += "Please provide only the code to insert at the cursor position, "
        prompt += "without any additional explanations or formatting."
        
        return prompt
    
    def _build_refactoring_prompt(
        self, 
        context: Dict[str, Any],
        language_handler: ProgrammingLanguage
    ) -> str:
        """
        Build prompt for code refactoring.
        
        Args:
            context: Context for code refactoring
            language_handler: Language handler
            
        Returns:
            Prompt string
        """
        original_code = context.get("original_code", "")
        refactoring_description = context.get("refactoring_description", "")
        language = context.get("language", "")
        functions = context.get("functions", [])
        classes = context.get("classes", [])
        
        # Construct the prompt
        prompt = f"Refactor the following {language} code according to this description:\n\n"
        prompt += f"{refactoring_description}\n\n"
        
        # Add the original code
        prompt += f"Original code:\n```{language}\n{original_code}\n```\n\n"
        
        # Add information about functions and classes if available
        if functions:
            prompt += "Functions in the code:\n"
            for func in functions:
                prompt += f"- {func.get('name', '')}\n"
            prompt += "\n"
        
        if classes:
            prompt += "Classes in the code:\n"
            for cls in classes:
                prompt += f"- {cls.get('name', '')}\n"
            prompt += "\n"
        
        # Add instructions for the response format
        prompt += f"Please provide the refactored {language} code in your response, "
        prompt += "without explanations. The code should maintain the same functionality "
        prompt += "while addressing the refactoring requirements."
        
        return prompt
    
    def _build_optimization_prompt(
        self, 
        context: Dict[str, Any],
        language_handler: ProgrammingLanguage
    ) -> str:
        """
        Build prompt for code optimization.
        
        Args:
            context: Context for code optimization
            language_handler: Language handler
            
        Returns:
            Prompt string
        """
        original_code = context.get("original_code", "")
        optimization_targets = context.get("optimization_targets", [])
        language = context.get("language", "")
        
        # Construct the prompt
        prompt = f"Optimize the following {language} code for these targets:\n\n"
        
        for target in optimization_targets:
            prompt += f"- {target}\n"
        prompt += "\n"
        
        # Add the original code
        prompt += f"Original code:\n```{language}\n{original_code}\n```\n\n"
        
        # Add instructions for the response format
        prompt += f"Please provide the optimized {language} code in your response, "
        prompt += "followed by brief comments explaining the optimizations made. "
        prompt += "The code should maintain the same functionality while improving "
        prompt += "the specified optimization targets."
        
        return prompt
    
    def _build_test_prompt(
        self, 
        context: Dict[str, Any],
        language_handler: ProgrammingLanguage
    ) -> str:
        """
        Build prompt for test generation.
        
        Args:
            context: Context for test generation
            language_handler: Language handler
            
        Returns:
            Prompt string
        """
        code = context.get("code", "")
        test_framework = context.get("test_framework")
        language = context.get("language", "")
        function_name = context.get("function_name")
        
        # Construct the prompt
        prompt = f"Generate tests for the following {language} code:\n\n"
        prompt += f"```{language}\n{code}\n```\n\n"
        
        # Add information about the test framework if specified
        if test_framework:
            prompt += f"Use the {test_framework} testing framework.\n\n"
        
        # Add information about the function to test if specified
        if function_name:
            prompt += f"Generate tests specifically for the '{function_name}' function.\n\n"
        
        # Add instructions for the response format
        prompt += f"Please provide only the test code in {language}, without explanations. "
        prompt += "The tests should be comprehensive and cover edge cases."
        
        return prompt
    
    def _build_documentation_prompt(
        self, 
        context: Dict[str, Any],
        language_handler: ProgrammingLanguage
    ) -> str:
        """
        Build prompt for documentation generation.
        
        Args:
            context: Context for documentation generation
            language_handler: Language handler
            
        Returns:
            Prompt string
        """
        code = context.get("code", "")
        doc_style = context.get("doc_style")
        language = context.get("language", "")
        
        # Construct the prompt
        prompt = f"Add documentation to the following {language} code:\n\n"
        prompt += f"```{language}\n{code}\n```\n\n"
        
        # Add information about the documentation style if specified
        if doc_style:
            prompt += f"Use the {doc_style} documentation style.\n\n"
        
        # Add instructions for the response format
        prompt += f"Please provide the documented {language} code in your response, "
        prompt += "without explanations. The documentation should be comprehensive "
        prompt += "and follow best practices for the specified language and style."
        
        return prompt
    
    def _build_boilerplate_prompt(
        self, 
        context: Dict[str, Any],
        language_handler: ProgrammingLanguage
    ) -> str:
        """
        Build prompt for boilerplate generation.
        
        Args:
            context: Context for boilerplate generation
            language_handler: Language handler
            
        Returns:
            Prompt string
        """
        project_type = context.get("project_type", "")
        language = context.get("language", "")
        features = context.get("features", [])
        
        # Construct the prompt
        prompt = f"Generate boilerplate code for a {project_type} project in {language} "
        prompt += "with the following features:\n\n"
        
        for feature in features:
            prompt += f"- {feature}\n"
        prompt += "\n"
        
        # Add instructions for the response format
        prompt += "Please provide the code for each file in the following format:\n\n"
        prompt += "```filename: path/to/file.ext\n"
        prompt += "file content goes here\n"
        prompt += "```\n\n"
        
        prompt += "Include all necessary files for a basic project structure, "
        prompt += "such as source files, configuration files, and README."
        
        return prompt
    
    def _extract_code_from_response(self, response: str, language: str) -> str:
        """
        Extract code from LLM response.
        
        Args:
            response: LLM response
            language: Programming language
            
        Returns:
            Extracted code
        """
        # Try to extract code from markdown code blocks
        pattern = r"```(?:" + re.escape(language) + r"|" + language.lower() + r")?\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks found, try to extract the entire response
        # (minus any explanatory text at the beginning or end)
        lines = response.strip().split("\n")
        
        # Skip initial explanatory text
        start_idx = 0
        while start_idx < len(lines) and (
            lines[start_idx].startswith("#") or
            lines[start_idx].startswith("Here") or
            lines[start_idx].startswith("Sure") or
            lines[start_idx].startswith("I'll")
        ):
            start_idx += 1
        
        # Skip final explanatory text
        end_idx = len(lines)
        while end_idx > start_idx and (
            lines[end_idx-1].startswith("Note") or
            lines[end_idx-1].startswith("This") or
            lines[end_idx-1].startswith("You") or
            lines[end_idx-1].startswith("The")
        ):
            end_idx -= 1
        
        return "\n".join(lines[start_idx:end_idx]).strip()
    
    def _extract_completion_from_response(
        self, 
        response: str, 
        context: Dict[str, Any]
    ) -> str:
        """
        Extract code completion from LLM response.
        
        Args:
            response: LLM response
            context: Context for code completion
            
        Returns:
            Extracted completion
        """
        # Strip any explanatory text
        lines = response.strip().split("\n")
        
        # Skip initial explanatory text
        start_idx = 0
        while start_idx < len(lines) and (
            lines[start_idx].startswith("#") or
            lines[start_idx].startswith("Here") or
            lines[start_idx].startswith("Sure") or
            lines[start_idx].startswith("I'll") or
            not lines[start_idx].strip()
        ):
            start_idx += 1
        
        # Skip final explanatory text
        end_idx = len(lines)
        while end_idx > start_idx and (
            lines[end_idx-1].startswith("Note") or
            lines[end_idx-1].startswith("This") or
            lines[end_idx-1].startswith("You") or
            lines[end_idx-1].startswith("The") or
            not lines[end_idx-1].strip()
        ):
            end_idx -= 1
        
        # Extract the completion
        completion = "\n".join(lines[start_idx:end_idx]).strip()
        
        # Remove any code block markers
        completion = re.sub(r"```.*?\n", "", completion)
        completion = re.sub(r"```$", "", completion)
        
        return completion
    
    def _extract_files_from_response(self, response: str, language: str) -> Dict[str, str]:
        """
        Extract files from LLM response for boilerplate generation.
        
        Args:
            response: LLM response
            language: Programming language
            
        Returns:
            Dictionary mapping file paths to content
        """
        files = {}
        
        # Extract file blocks
        pattern = r"```filename:\s*(.*?)\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        
        for filename, content in matches:
            files[filename.strip()] = content.strip()
        
        # If no file blocks found, try to find individual code blocks
        if not files:
            # Try to extract code blocks with filenames in headers
            headers = re.findall(r"#+\s*(.*?\.(?:py|js|ts|html|css|json|md|txt))\s*\n", response)
            code_blocks = re.findall(r"```.*?\n(.*?)```", response, re.DOTALL)
            
            if len(headers) == len(code_blocks):
                for i in range(len(headers)):
                    files[headers[i]] = code_blocks[i].strip()
        
        return files


# Utility functions

def generate_code_from_description(
    description: str,
    language: str,
    llm_engine: Optional[LLMEngine] = None
) -> str:
    """
    Generate code from a description.
    
    Args:
        description: Description of the code to generate
        language: Target programming language
        llm_engine: Optional LLM engine to use
        
    Returns:
        Generated code
    """
    generator = CodeGenerator(llm_engine=llm_engine)
    
    result = generator.generate_from_description(
        description=description,
        language=language
    )
    
    return result.get("code", "")


def generate_docs_for_code(
    code: str,
    language: Optional[str] = None,
    llm_engine: Optional[LLMEngine] = None
) -> str:
    """
    Generate documentation for code.
    
    Args:
        code: Code to document
        language: Programming language (if None, will be auto-detected)
        llm_engine: Optional LLM engine to use
        
    Returns:
        Documented code
    """
    generator = CodeGenerator(llm_engine=llm_engine)
    
    result = generator.generate_documentation(
        code=code,
        language=language
    )
    
    return result.get("code", code)
