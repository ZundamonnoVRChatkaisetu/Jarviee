"""
Code Analysis Engine for Jarviee

This module provides comprehensive code analysis capabilities, including:
- Structure analysis (AST-based)
- Dependency analysis
- Quality assessment
- Security vulnerability detection
- Pattern recognition
- Performance insights
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from .languages.base import ProgrammingLanguage, SyntaxElement
from .languages.registry import LanguageRegistry


class CodeAnalysisResult:
    """Container for code analysis results."""
    
    def __init__(self, file_path: Optional[str] = None, code: Optional[str] = None):
        """
        Initialize a code analysis result.
        
        Args:
            file_path: Path to the analyzed file if applicable
            code: Analyzed code if not from a file
        """
        self.file_path = file_path
        self.code_sample = code[:200] + "..." if code and len(code) > 200 else code
        self.language: Optional[str] = None
        self.structure: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.issues: List[Dict[str, Any]] = []
        self.dependencies: List[Dict[str, Any]] = []
        self.security: Dict[str, Any] = {}
        self.patterns: List[Dict[str, Any]] = []
        self.performance: Dict[str, Any] = {}
        self.timestamp = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the analysis result to a dictionary."""
        return {
            "file_path": self.file_path,
            "code_sample": self.code_sample,
            "language": self.language,
            "structure": self.structure,
            "metrics": self.metrics,
            "issues": self.issues,
            "dependencies": self.dependencies,
            "security": self.security,
            "patterns": self.patterns,
            "performance": self.performance,
            "timestamp": self.timestamp
        }


class CodeAnalyzer:
    """
    Core code analysis engine for Jarviee.
    
    This class provides comprehensive analysis of code, supporting multiple
    programming languages and providing various types of insights.
    """
    
    def __init__(self):
        """Initialize the code analyzer."""
        self.logger = logging.getLogger("code_analyzer")
        self.language_registry = LanguageRegistry()
        self.analysis_cache: Dict[str, CodeAnalysisResult] = {}
    
    def analyze_file(
        self, 
        file_path: str,
        analysis_types: Optional[List[str]] = None
    ) -> CodeAnalysisResult:
        """
        Analyze code from a file.
        
        Args:
            file_path: Path to the file to analyze
            analysis_types: Types of analysis to perform (default: all)
            
        Returns:
            Analysis result object
        """
        try:
            # Check if file exists
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Perform analysis
            result = self.analyze_code(code, analysis_types, file_path)
            
            # Cache the result
            self.analysis_cache[file_path] = result
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {str(e)}")
            result = CodeAnalysisResult(file_path)
            result.issues.append({
                "type": "error",
                "message": f"Failed to analyze file: {str(e)}",
                "severity": "error"
            })
            return result
    
    def analyze_code(
        self, 
        code: str,
        analysis_types: Optional[List[str]] = None,
        file_path: Optional[str] = None
    ) -> CodeAnalysisResult:
        """
        Analyze code string.
        
        Args:
            code: Code string to analyze
            analysis_types: Types of analysis to perform (default: all)
            file_path: Optional path to the source file
            
        Returns:
            Analysis result object
        """
        result = CodeAnalysisResult(file_path, code)
        
        try:
            # Determine the language
            language = self._detect_language(code, file_path)
            if not language:
                raise ValueError("Could not determine the programming language")
            
            result.language = language.name
            
            # Determine which analysis types to perform
            if analysis_types is None:
                analysis_types = [
                    "structure", "metrics", "issues", "dependencies", 
                    "security", "patterns", "performance"
                ]
            
            # Perform requested analyses
            if "structure" in analysis_types:
                result.structure = self._analyze_structure(code, language)
            
            if "metrics" in analysis_types:
                result.metrics = self._analyze_metrics(code, language)
            
            if "issues" in analysis_types:
                result.issues = self._analyze_issues(code, language)
            
            if "dependencies" in analysis_types:
                result.dependencies = self._analyze_dependencies(code, language)
            
            if "security" in analysis_types:
                result.security = self._analyze_security(code, language)
            
            if "patterns" in analysis_types:
                result.patterns = self._analyze_patterns(code, language)
            
            if "performance" in analysis_types:
                result.performance = self._analyze_performance(code, language)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error analyzing code: {str(e)}")
            result.issues.append({
                "type": "error",
                "message": f"Failed to analyze code: {str(e)}",
                "severity": "error"
            })
            return result
    
    def analyze_directory(
        self, 
        directory_path: str,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
        analysis_types: Optional[List[str]] = None,
        max_files: int = 100
    ) -> Dict[str, CodeAnalysisResult]:
        """
        Analyze all code files in a directory.
        
        Args:
            directory_path: Path to the directory to analyze
            file_extensions: File extensions to include (default: all supported)
            recursive: Whether to analyze subdirectories
            analysis_types: Types of analysis to perform
            max_files: Maximum number of files to analyze
            
        Returns:
            Dictionary mapping file paths to analysis results
        """
        results = {}
        files_analyzed = 0
        
        try:
            # Check if directory exists
            if not os.path.isdir(directory_path):
                raise NotADirectoryError(f"Directory not found: {directory_path}")
            
            # Get all supported file extensions if none specified
            if file_extensions is None:
                all_languages = self.language_registry.list_languages()
                file_extensions = []
                for language in all_languages:
                    file_extensions.extend(language.file_extensions)
            
            # Walk the directory
            for root, _, files in os.walk(directory_path):
                if files_analyzed >= max_files:
                    break
                
                # Skip if not recursive and not the root directory
                if not recursive and root != directory_path:
                    continue
                
                for file in files:
                    if files_analyzed >= max_files:
                        break
                    
                    # Check file extension
                    _, ext = os.path.splitext(file)
                    if file_extensions and ext not in file_extensions:
                        continue
                    
                    # Analyze file
                    file_path = os.path.join(root, file)
                    self.logger.info(f"Analyzing file: {file_path}")
                    result = self.analyze_file(file_path, analysis_types)
                    results[file_path] = result
                    files_analyzed += 1
        
        except Exception as e:
            self.logger.error(f"Error analyzing directory {directory_path}: {str(e)}")
        
        return results
    
    def get_cached_analysis(self, file_path: str) -> Optional[CodeAnalysisResult]:
        """
        Get a cached analysis result for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Cached analysis result if available, None otherwise
        """
        return self.analysis_cache.get(file_path)
    
    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self.analysis_cache.clear()
    
    def _detect_language(
        self, 
        code: str, 
        file_path: Optional[str] = None
    ) -> Optional[ProgrammingLanguage]:
        """
        Detect the programming language of code.
        
        Args:
            code: Code to analyze
            file_path: Optional path to the source file
            
        Returns:
            Detected programming language, or None if unknown
        """
        # Try to detect by file extension first
        if file_path:
            language = self.language_registry.get_language_for_file(file_path)
            if language:
                return language
        
        # Fall back to content-based detection
        return self.language_registry.get_language_for_code(code)
    
    def _analyze_structure(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> Dict[str, Any]:
        """
        Analyze the structure of code.
        
        Args:
            code: Code to analyze
            language: Programming language
            
        Returns:
            Structure analysis result
        """
        try:
            # Parse the code
            ast = language.parse_code(code)
            
            # Extract functions and classes
            functions = language.extract_functions(code)
            classes = language.extract_classes(code)
            
            return {
                "functions": functions,
                "classes": classes,
                "ast": ast
            }
        
        except Exception as e:
            self.logger.error(f"Error analyzing structure: {str(e)}")
            return {
                "error": str(e)
            }
    
    def _analyze_metrics(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> Dict[str, Any]:
        """
        Analyze metrics of code.
        
        Args:
            code: Code to analyze
            language: Programming language
            
        Returns:
            Metrics analysis result
        """
        try:
            # Get language-specific metrics
            quality = language.analyze_quality(code)
            
            # Add general metrics
            lines = code.split('\n')
            
            quality["metrics"]["chars_total"] = len(code)
            quality["metrics"]["avg_line_length"] = sum(len(line) for line in lines) / len(lines) if lines else 0
            
            return quality
        
        except Exception as e:
            self.logger.error(f"Error analyzing metrics: {str(e)}")
            return {
                "error": str(e)
            }
    
    def _analyze_issues(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> List[Dict[str, Any]]:
        """
        Analyze issues in code.
        
        Args:
            code: Code to analyze
            language: Programming language
            
        Returns:
            List of issues
        """
        try:
            # Get language-specific issues
            return language.detect_issues(code)
        
        except Exception as e:
            self.logger.error(f"Error analyzing issues: {str(e)}")
            return [{
                "type": "error",
                "message": f"Failed to analyze issues: {str(e)}",
                "severity": "error"
            }]
    
    def _analyze_dependencies(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> List[Dict[str, Any]]:
        """
        Analyze dependencies in code.
        
        Args:
            code: Code to analyze
            language: Programming language
            
        Returns:
            Dependencies analysis result
        """
        try:
            # Extract imports
            imports = language.extract_imports(code)
            
            # Transform imports into dependency information
            dependencies = []
            for imp in imports:
                dependencies.append({
                    "statement": imp,
                    "type": "import"
                })
            
            return dependencies
        
        except Exception as e:
            self.logger.error(f"Error analyzing dependencies: {str(e)}")
            return []
    
    def _analyze_security(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> Dict[str, Any]:
        """
        Analyze security vulnerabilities in code.
        
        Args:
            code: Code to analyze
            language: Programming language
            
        Returns:
            Security analysis result
        """
        # This is a stub implementation
        # A real implementation would use security scanning tools
        
        result = {
            "vulnerabilities": [],
            "risk_level": "low"
        }
        
        # Perform basic checks
        if "eval(" in code:
            result["vulnerabilities"].append({
                "type": "code_injection",
                "description": "Possible code injection vulnerability (eval)",
                "severity": "high",
                "cwe": "CWE-95"
            })
            result["risk_level"] = "high"
        
        if "exec(" in code:
            result["vulnerabilities"].append({
                "type": "code_injection",
                "description": "Possible code injection vulnerability (exec)",
                "severity": "high",
                "cwe": "CWE-95"
            })
            result["risk_level"] = "high"
        
        # Check for SQL injection (very basic)
        sql_patterns = ["SELECT", "INSERT", "UPDATE", "DELETE", "DROP"]
        for pattern in sql_patterns:
            if pattern in code and "?" not in code and "%s" not in code:
                result["vulnerabilities"].append({
                    "type": "sql_injection",
                    "description": "Possible SQL injection vulnerability",
                    "severity": "high",
                    "cwe": "CWE-89"
                })
                result["risk_level"] = "high"
                break
        
        # Check for hardcoded secrets (very basic)
        secret_patterns = ["password", "secret", "token", "key", "apikey", "api_key"]
        for pattern in secret_patterns:
            if f"{pattern}=" in code.lower():
                result["vulnerabilities"].append({
                    "type": "hardcoded_secret",
                    "description": "Possible hardcoded secret found",
                    "severity": "medium",
                    "cwe": "CWE-798"
                })
                result["risk_level"] = max(result["risk_level"], "medium")
        
        return result
    
    def _analyze_patterns(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> List[Dict[str, Any]]:
        """
        Analyze patterns in code.
        
        Args:
            code: Code to analyze
            language: Programming language
            
        Returns:
            Patterns analysis result
        """
        # This is a simplified implementation
        patterns = []
        
        # Check for common design patterns (very basic detection)
        
        # Singleton pattern
        if ("class" in code and 
                "instance" in code and 
                "if" in code and 
                "return" in code and 
                "None" in code):
            patterns.append({
                "name": "Singleton",
                "confidence": "medium",
                "description": "Possible Singleton pattern detected"
            })
        
        # Factory pattern
        if ("class" in code and 
                "create" in code and 
                "return" in code and 
                "if" in code):
            patterns.append({
                "name": "Factory",
                "confidence": "medium",
                "description": "Possible Factory pattern detected"
            })
        
        # Observer pattern
        if ("subscribe" in code or "register" in code) and ("notify" in code or "update" in code):
            patterns.append({
                "name": "Observer",
                "confidence": "medium",
                "description": "Possible Observer pattern detected"
            })
        
        # Strategy pattern
        if "strategy" in code and "interface" in code:
            patterns.append({
                "name": "Strategy",
                "confidence": "medium",
                "description": "Possible Strategy pattern detected"
            })
        
        return patterns
    
    def _analyze_performance(
        self, 
        code: str, 
        language: ProgrammingLanguage
    ) -> Dict[str, Any]:
        """
        Analyze performance considerations in code.
        
        Args:
            code: Code to analyze
            language: Programming language
            
        Returns:
            Performance analysis result
        """
        # This is a simplified implementation
        result = {
            "issues": [],
            "recommendations": []
        }
        
        # Check for nested loops (potential performance issue)
        if "for" in code and code.count("for") > 1:
            # Basic check for nested loops (very simplified)
            lines = code.split('\n')
            indent_levels = []
            
            for line in lines:
                if line.strip().startswith("for "):
                    indent_level = len(line) - len(line.lstrip())
                    indent_levels.append(indent_level)
            
            indent_levels.sort()
            
            if len(indent_levels) >= 2:
                for i in range(len(indent_levels) - 1):
                    if indent_levels[i+1] > indent_levels[i]:
                        result["issues"].append({
                            "type": "nested_loop",
                            "description": "Nested loops detected, may cause performance issues for large datasets",
                            "severity": "medium"
                        })
                        
                        result["recommendations"].append({
                            "description": "Consider optimizing nested loops or using more efficient data structures",
                            "priority": "medium"
                        })
                        
                        break
        
        # Check for large string concatenation (potential performance issue)
        if "+" in code and "str" in code:
            result["recommendations"].append({
                "description": "If concatenating many strings, consider using join() or string builders for better performance",
                "priority": "low"
            })
        
        return result


# Utility functions

def format_code(code: str, language: Optional[str] = None) -> str:
    """
    Format code using the appropriate formatter.
    
    Args:
        code: Code to format
        language: Programming language (if None, will be auto-detected)
        
    Returns:
        Formatted code
    """
    registry = LanguageRegistry()
    
    if language:
        lang = registry.get_language_by_name(language)
    else:
        lang = registry.get_language_for_code(code)
    
    if lang:
        return lang.format_code(code)
    else:
        return code  # Return unformatted code if language not supported


def extract_code_structure(code: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract the structure of code (functions, classes, etc.)
    
    Args:
        code: Code to analyze
        language: Programming language (if None, will be auto-detected)
        
    Returns:
        Code structure
    """
    analyzer = CodeAnalyzer()
    
    if language:
        lang = analyzer.language_registry.get_language_by_name(language)
    else:
        lang = analyzer.language_registry.get_language_for_code(code)
    
    if not lang:
        return {"error": "Could not determine the programming language"}
    
    return analyzer._analyze_structure(code, lang)
