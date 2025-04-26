"""
Python Language Support

This module implements Python language support for Jarviee's programming
capabilities, providing parsing, analysis, and generation features
specifically tailored for Python code.
"""

import ast
import re
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import black

from .base import ProgrammingLanguage, LanguageFeature, SyntaxElement


class PythonLanguage(ProgrammingLanguage):
    """
    Python language implementation.
    
    This class provides Python-specific functionality for code analysis,
    generation, and manipulation.
    """
    
    def __init__(self):
        """Initialize Python language support."""
        super().__init__(
            name="Python",
            file_extensions=[".py", ".pyi", ".pyx", ".ipynb"]
        )
        
        # Set Python features
        self.features.update([
            LanguageFeature.DYNAMIC_TYPING,
            LanguageFeature.OBJECT_ORIENTED,
            LanguageFeature.FUNCTIONAL,
            LanguageFeature.PROCEDURAL,
            LanguageFeature.REFLECTIVE,
            LanguageFeature.META_PROGRAMMING,
            LanguageFeature.GARBAGE_COLLECTION,
            LanguageFeature.EXCEPTION_HANDLING,
            LanguageFeature.FIRST_CLASS_FUNCTIONS,
            LanguageFeature.CLOSURES,
            LanguageFeature.MODULE_SYSTEM,
            LanguageFeature.PACKAGE_MANAGER,
        ])
        
        # Define syntax patterns
        self.syntax_patterns.update({
            SyntaxElement.COMMENT: r'#.*$',
            SyntaxElement.FUNCTION_DEFINITION: r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?\s*:',
            SyntaxElement.CLASS_DEFINITION: r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*\(([^)]*)\))?\s*:',
            SyntaxElement.IMPORT_STATEMENT: r'(?:from\s+([a-zA-Z0-9_.]+)\s+)?import\s+([a-zA-Z0-9_.*]+(?:\s*,\s*[a-zA-Z0-9_.*]+)*)',
            SyntaxElement.VARIABLE_DECLARATION: r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(?::([^=]+))?\s*=\s*([^#\n]+)',
            SyntaxElement.FUNCTION_CALL: r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)',
            SyntaxElement.STRING_LITERAL: r'(?:[bfru]|[bfru]{2})?("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|"""(?:\\.|[^"\\])*"""|\'\'\'(?:\\.|[^\'\\])*\'\'\')',
        })
    
    def parse_code(self, code: str) -> ast.AST:
        """
        Parse Python code into an abstract syntax tree.
        
        Args:
            code: The Python code to parse
            
        Returns:
            An AST representation of the code
            
        Raises:
            SyntaxError: If the code has syntax errors
        """
        try:
            return ast.parse(code)
        except SyntaxError as e:
            # Add more context to the syntax error
            line_num = e.lineno if hasattr(e, 'lineno') else 0
            col_num = e.offset if hasattr(e, 'offset') else 0
            
            if line_num > 0 and col_num > 0:
                lines = code.split('\n')
                if 0 <= line_num - 1 < len(lines):
                    line = lines[line_num - 1]
                    pointer = ' ' * (col_num - 1) + '^'
                    context = f"\n{line}\n{pointer}"
                    raise SyntaxError(f"{e.msg} at line {line_num}, column {col_num}: {context}") from e
            
            raise
    
    def format_code(self, code: str) -> str:
        """
        Format Python code using Black.
        
        Args:
            code: The code to format
            
        Returns:
            Formatted code
            
        Note:
            If Black fails to format the code, the original code is returned.
        """
        try:
            return black.format_str(code, mode=black.Mode())
        except Exception:
            # If Black fails, return the original code
            return code
    
    def analyze_quality(self, code: str) -> Dict[str, Any]:
        """
        Analyze Python code quality.
        
        Args:
            code: The code to analyze
            
        Returns:
            A dictionary with quality metrics
        """
        result = {
            "complexity": {},
            "style": {},
            "metrics": {},
        }
        
        try:
            tree = self.parse_code(code)
            
            # Count basic metrics
            lines = code.split('\n')
            result["metrics"]["lines_total"] = len(lines)
            result["metrics"]["lines_code"] = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
            result["metrics"]["lines_comment"] = sum(1 for line in lines if line.strip().startswith('#'))
            result["metrics"]["lines_empty"] = sum(1 for line in lines if not line.strip())
            
            # Count functions and classes
            function_visitor = _FunctionVisitor()
            function_visitor.visit(tree)
            result["metrics"]["functions"] = function_visitor.function_count
            result["metrics"]["classes"] = function_visitor.class_count
            
            # Calculate complexity
            complexity_visitor = _ComplexityVisitor()
            complexity_visitor.visit(tree)
            result["complexity"]["cyclomatic"] = complexity_visitor.complexity
            result["complexity"]["max_nesting"] = complexity_visitor.max_nesting
            
            # Style checks
            result["style"]["line_length"] = {
                "max": max((len(line) for line in lines), default=0),
                "violations": sum(1 for line in lines if len(line) > 88)  # PEP 8 recommends 79, but Black uses 88
            }
            
            # Import analysis
            import_visitor = _ImportVisitor()
            import_visitor.visit(tree)
            result["metrics"]["imports"] = {
                "count": import_visitor.import_count,
                "modules": import_visitor.imported_modules
            }
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def detect_issues(self, code: str) -> List[Dict[str, Any]]:
        """
        Detect issues in Python code.
        
        Args:
            code: The code to analyze
            
        Returns:
            A list of issues, each represented as a dictionary
        """
        issues = []
        
        try:
            tree = self.parse_code(code)
            
            # Check for unused imports
            import_visitor = _ImportVisitor()
            import_visitor.visit(tree)
            
            name_visitor = _NameVisitor()
            name_visitor.visit(tree)
            
            for module, names in import_visitor.imported_names.items():
                for name in names:
                    if name not in name_visitor.used_names:
                        issues.append({
                            "type": "unused_import",
                            "message": f"Unused import: {name} from {module}",
                            "line": import_visitor.import_lines.get(name, 0),
                            "severity": "warning"
                        })
            
            # Check for potential bugs
            bug_visitor = _BugVisitor()
            bug_visitor.visit(tree)
            issues.extend(bug_visitor.issues)
            
            # Check for style issues
            style_visitor = _StyleVisitor()
            style_visitor.visit(tree)
            issues.extend(style_visitor.issues)
            
        except SyntaxError as e:
            # Report syntax errors
            line_num = e.lineno if hasattr(e, 'lineno') else 0
            col_num = e.offset if hasattr(e, 'offset') else 0
            
            issues.append({
                "type": "syntax_error",
                "message": str(e),
                "line": line_num,
                "column": col_num,
                "severity": "error"
            })
        except Exception as e:
            # Report other errors
            issues.append({
                "type": "error",
                "message": f"Error analyzing code: {str(e)}",
                "line": 0,
                "severity": "error"
            })
        
        return issues
    
    def get_documentation_template(self, code_element: str) -> str:
        """
        Get a documentation template for a Python code element.
        
        Args:
            code_element: The code element to document
            
        Returns:
            A documentation template string in Sphinx format
        """
        try:
            # Try to parse the code element
            tree = self.parse_code(code_element)
            
            for node in ast.walk(tree):
                # Function definition
                if isinstance(node, ast.FunctionDef):
                    return self._generate_function_docstring(node)
                
                # Class definition
                elif isinstance(node, ast.ClassDef):
                    return self._generate_class_docstring(node)
            
            # If no specific element was found, return a generic template
            return '"""\nDescription\n\nDetailed description\n"""'
            
        except Exception:
            # If parsing fails, return a generic template
            return '"""\nDescription\n\nDetailed description\n"""'
    
    def _generate_function_docstring(self, func_node: ast.FunctionDef) -> str:
        """Generate a docstring template for a function."""
        args = []
        returns = None
        
        # Extract arguments
        for arg in func_node.args.args:
            if arg.arg != 'self' and arg.arg != 'cls':
                arg_type = ""
                if hasattr(arg, 'annotation') and arg.annotation is not None:
                    if isinstance(arg.annotation, ast.Name):
                        arg_type = f": {arg.annotation.id}"
                    elif isinstance(arg.annotation, ast.Constant) and isinstance(arg.annotation.value, str):
                        arg_type = f": {arg.annotation.value}"
                
                args.append(f"{arg.arg}{arg_type}")
        
        # Extract return type
        if hasattr(func_node, 'returns') and func_node.returns is not None:
            if isinstance(func_node.returns, ast.Name):
                returns = func_node.returns.id
            elif isinstance(func_node.returns, ast.Constant) and isinstance(func_node.returns.value, str):
                returns = func_node.returns.value
        
        # Build docstring
        docstring = f'"""\n{func_node.name}\n\nDescription of {func_node.name}\n'
        
        # Add parameters
        if args:
            docstring += "\nParameters\n----------\n"
            for arg in args:
                name = arg.split(':')[0].strip()
                docstring += f"{name}\n    Description of {name}\n"
        
        # Add return value
        if returns:
            docstring += "\nReturns\n-------\n"
            docstring += f"{returns}\n    Description of return value\n"
        
        docstring += '"""'
        return docstring
    
    def _generate_class_docstring(self, class_node: ast.ClassDef) -> str:
        """Generate a docstring template for a class."""
        # Build docstring
        docstring = f'"""\n{class_node.name}\n\nDescription of {class_node.name}\n'
        
        # Add attributes section if the class has instance variables
        has_attributes = False
        for node in ast.walk(class_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                        has_attributes = True
                        break
            if has_attributes:
                break
        
        if has_attributes:
            docstring += "\nAttributes\n----------\nattr_name : type\n    Description of attribute\n"
        
        # Add methods section if the class has methods
        has_methods = False
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name != '__init__':
                has_methods = True
                break
        
        if has_methods:
            docstring += "\nMethods\n-------\nmethod_name(param1, param2)\n    Description of method\n"
        
        docstring += '"""'
        return docstring
    
    def extract_imports(self, code: str) -> List[str]:
        """
        Extract import statements from Python code.
        
        Args:
            code: The code to analyze
            
        Returns:
            A list of import statements
        """
        imports = []
        
        try:
            tree = self.parse_code(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.asname:
                            imports.append(f"import {name.name} as {name.asname}")
                        else:
                            imports.append(f"import {name.name}")
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names_str = ", ".join(
                        f"{name.name} as {name.asname}" if name.asname else name.name
                        for name in node.names
                    )
                    imports.append(f"from {module} import {names_str}")
        
        except Exception:
            pass
        
        return imports
    
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract function definitions from Python code.
        
        Args:
            code: The code to analyze
            
        Returns:
            A list of function information dictionaries
        """
        functions = []
        
        try:
            tree = self.parse_code(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function name
                    function_info = {
                        "name": node.name,
                        "line": node.lineno,
                        "end_line": node.end_lineno if hasattr(node, 'end_lineno') else None,
                        "args": [],
                        "returns": None,
                        "decorators": [],
                        "docstring": None
                    }
                    
                    # Extract arguments
                    for arg in node.args.args:
                        arg_info = {"name": arg.arg}
                        
                        # Extract argument type annotation
                        if hasattr(arg, 'annotation') and arg.annotation is not None:
                            if isinstance(arg.annotation, ast.Name):
                                arg_info["type"] = arg.annotation.id
                            elif isinstance(arg.annotation, ast.Constant) and isinstance(arg.annotation.value, str):
                                arg_info["type"] = arg.annotation.value
                        
                        function_info["args"].append(arg_info)
                    
                    # Extract return type annotation
                    if hasattr(node, 'returns') and node.returns is not None:
                        if isinstance(node.returns, ast.Name):
                            function_info["returns"] = node.returns.id
                        elif isinstance(node.returns, ast.Constant) and isinstance(node.returns.value, str):
                            function_info["returns"] = node.returns.value
                    
                    # Extract decorators
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name):
                            function_info["decorators"].append(decorator.id)
                        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                            function_info["decorators"].append(decorator.func.id)
                    
                    # Extract docstring
                    if (node.body and isinstance(node.body[0], ast.Expr) and 
                            isinstance(node.body[0].value, ast.Constant) and 
                            isinstance(node.body[0].value.value, str)):
                        function_info["docstring"] = node.body[0].value.value
                    
                    functions.append(function_info)
        
        except Exception:
            pass
        
        return functions
    
    def extract_classes(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract class definitions from Python code.
        
        Args:
            code: The code to analyze
            
        Returns:
            A list of class information dictionaries
        """
        classes = []
        
        try:
            tree = self.parse_code(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Extract class name and bases
                    class_info = {
                        "name": node.name,
                        "line": node.lineno,
                        "end_line": node.end_lineno if hasattr(node, 'end_lineno') else None,
                        "bases": [],
                        "methods": [],
                        "attributes": [],
                        "decorators": [],
                        "docstring": None
                    }
                    
                    # Extract base classes
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            class_info["bases"].append(base.id)
                        elif isinstance(base, ast.Attribute):
                            # Handle cases like module.BaseClass
                            base_name = []
                            current = base
                            while isinstance(current, ast.Attribute):
                                base_name.insert(0, current.attr)
                                current = current.value
                            if isinstance(current, ast.Name):
                                base_name.insert(0, current.id)
                            class_info["bases"].append(".".join(base_name))
                    
                    # Extract decorators
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name):
                            class_info["decorators"].append(decorator.id)
                        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                            class_info["decorators"].append(decorator.func.id)
                    
                    # Extract docstring
                    if (node.body and isinstance(node.body[0], ast.Expr) and 
                            isinstance(node.body[0].value, ast.Constant) and 
                            isinstance(node.body[0].value.value, str)):
                        class_info["docstring"] = node.body[0].value.value
                    
                    # Extract methods and attributes
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            # Method
                            method_info = {
                                "name": item.name,
                                "line": item.lineno,
                                "is_static": any(d.id == 'staticmethod' for d in item.decorator_list if isinstance(d, ast.Name)),
                                "is_class": any(d.id == 'classmethod' for d in item.decorator_list if isinstance(d, ast.Name))
                            }
                            class_info["methods"].append(method_info)
                        
                        elif isinstance(item, ast.Assign):
                            # Class attribute
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    attr_info = {
                                        "name": target.id,
                                        "line": item.lineno
                                    }
                                    class_info["attributes"].append(attr_info)
                    
                    # Look for instance attributes in __init__
                    init_method = next((m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == '__init__'), None)
                    if init_method:
                        for init_item in init_method.body:
                            if isinstance(init_item, ast.Assign):
                                for target in init_item.targets:
                                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                                        attr_info = {
                                            "name": target.attr,
                                            "line": init_item.lineno,
                                            "is_instance": True
                                        }
                                        class_info["attributes"].append(attr_info)
                    
                    classes.append(class_info)
        
        except Exception:
            pass
        
        return classes
    
    def get_code_completion(
        self, 
        code: str, 
        cursor_position: int
    ) -> List[Dict[str, Any]]:
        """
        Get code completion suggestions for Python code.
        
        Args:
            code: The code context
            cursor_position: Position of the cursor in the code
            
        Returns:
            A list of completion suggestions
        """
        # This is a simplified implementation
        # A real implementation would use a language server or more sophisticated analysis
        suggestions = []
        
        try:
            # Extract the current line and word being completed
            lines = code[:cursor_position].split('\n')
            current_line = lines[-1] if lines else ""
            
            # Extract the partial word being completed
            match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)$', current_line)
            prefix = match.group(1) if match else ""
            
            # Parse the code up to the cursor
            tree = self.parse_code(code[:cursor_position])
            
            # Extract all names in the code
            name_visitor = _NameVisitor()
            name_visitor.visit(tree)
            
            # Add suggestions based on used names
            for name in name_visitor.used_names:
                if prefix and not name.startswith(prefix):
                    continue
                
                suggestions.append({
                    "text": name,
                    "type": "variable"  # This is a simplification
                })
            
            # Add built-in functions and keywords
            builtins = [
                "abs", "all", "any", "bin", "bool", "chr", "dict", "dir", "enumerate", 
                "filter", "float", "format", "frozenset", "getattr", "hasattr", "input", 
                "int", "isinstance", "len", "list", "map", "max", "min", "open", "ord", 
                "pow", "print", "range", "repr", "reversed", "round", "set", "sorted", 
                "str", "sum", "super", "tuple", "type", "zip"
            ]
            
            keywords = [
                "and", "as", "assert", "async", "await", "break", "class", "continue", 
                "def", "del", "elif", "else", "except", "False", "finally", "for", 
                "from", "global", "if", "import", "in", "is", "lambda", "None", 
                "nonlocal", "not", "or", "pass", "raise", "return", "True", "try", 
                "while", "with", "yield"
            ]
            
            for builtin in builtins:
                if not prefix or builtin.startswith(prefix):
                    suggestions.append({
                        "text": builtin,
                        "type": "builtin"
                    })
            
            for keyword in keywords:
                if not prefix or keyword.startswith(prefix):
                    suggestions.append({
                        "text": keyword,
                        "type": "keyword"
                    })
        
        except Exception:
            pass
        
        return suggestions
    
    def generate_test(self, code: str, function_name: Optional[str] = None) -> str:
        """
        Generate a unittest test for Python code.
        
        Args:
            code: The code to test
            function_name: Optional name of a specific function to test
            
        Returns:
            Generated test code
        """
        try:
            tree = self.parse_code(code)
            
            # Extract module name from code if available
            module_name = "module"  # Default
            
            # Find all functions or a specific function
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if function_name is None or node.name == function_name:
                        functions.append(node)
            
            if not functions:
                return f"# No {'function named ' + function_name if function_name else 'functions'} found in the code"
            
            # Generate test code
            test_code = "import unittest\n"
            
            # Import the module/function to test
            test_code += f"# Import the code to test\n"
            test_code += f"# from {module_name} import {', '.join(f.name for f in functions)}\n\n"
            
            # Generate the test class
            test_code += f"class Test{module_name.capitalize()}(unittest.TestCase):\n"
            
            # Generate test methods for each function
            for func in functions:
                test_code += f"    def test_{func.name}(self):\n"
                test_code += f"        # TODO: Add test cases for {func.name}\n"
                
                # Generate placeholder assertions based on parameters
                args = []
                for arg in func.args.args:
                    if arg.arg != 'self' and arg.arg != 'cls':
                        args.append(arg.arg)
                
                if args:
                    test_code += f"        # Example: result = {func.name}({', '.join(args)})\n"
                    test_code += f"        # self.assertEqual(expected_result, result)\n"
                else:
                    test_code += f"        # Example: result = {func.name}()\n"
                    test_code += f"        # self.assertEqual(expected_result, result)\n"
                
                test_code += "\n"
            
            # Add the main block
            test_code += "if __name__ == '__main__':\n"
            test_code += "    unittest.main()\n"
            
            return test_code
        
        except Exception as e:
            return f"# Error generating test: {str(e)}\n# Please ensure the code is syntactically correct."


# Helper AST visitor classes for code analysis

class _FunctionVisitor(ast.NodeVisitor):
    """AST visitor to count functions and classes."""
    
    def __init__(self):
        self.function_count = 0
        self.class_count = 0
    
    def visit_FunctionDef(self, node):
        self.function_count += 1
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        self.function_count += 1
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.class_count += 1
        self.generic_visit(node)


class _ComplexityVisitor(ast.NodeVisitor):
    """AST visitor to measure code complexity."""
    
    def __init__(self):
        self.complexity = 1  # Base complexity
        self.current_nesting = 0
        self.max_nesting = 0
    
    def visit_If(self, node):
        self.complexity += 1
        self.current_nesting += 1
        self.max_nesting = max(self.max_nesting, self.current_nesting)
        self.generic_visit(node)
        self.current_nesting -= 1
    
    def visit_For(self, node):
        self.complexity += 1
        self.current_nesting += 1
        self.max_nesting = max(self.max_nesting, self.current_nesting)
        self.generic_visit(node)
        self.current_nesting -= 1
    
    def visit_AsyncFor(self, node):
        self.complexity += 1
        self.current_nesting += 1
        self.max_nesting = max(self.max_nesting, self.current_nesting)
        self.generic_visit(node)
        self.current_nesting -= 1
    
    def visit_While(self, node):
        self.complexity += 1
        self.current_nesting += 1
        self.max_nesting = max(self.max_nesting, self.current_nesting)
        self.generic_visit(node)
        self.current_nesting -= 1
    
    def visit_Try(self, node):
        self.complexity += len(node.handlers) + len(node.finalbody) + (1 if node.orelse else 0)
        self.current_nesting += 1
        self.max_nesting = max(self.max_nesting, self.current_nesting)
        self.generic_visit(node)
        self.current_nesting -= 1
    
    def visit_With(self, node):
        self.current_nesting += 1
        self.max_nesting = max(self.max_nesting, self.current_nesting)
        self.generic_visit(node)
        self.current_nesting -= 1
    
    def visit_AsyncWith(self, node):
        self.current_nesting += 1
        self.max_nesting = max(self.max_nesting, self.current_nesting)
        self.generic_visit(node)
        self.current_nesting -= 1
    
    def visit_BoolOp(self, node):
        self.complexity += len(node.values) - 1
        self.generic_visit(node)


class _ImportVisitor(ast.NodeVisitor):
    """AST visitor to analyze imports."""
    
    def __init__(self):
        self.import_count = 0
        self.imported_modules = []
        self.imported_names = {}  # Map from module to names
        self.import_lines = {}  # Map from name to line number
    
    def visit_Import(self, node):
        for name in node.names:
            self.import_count += 1
            module_name = name.name
            
            # Record the imported module
            self.imported_modules.append(module_name)
            
            # Record the name
            imported_name = name.asname or name.name
            if module_name not in self.imported_names:
                self.imported_names[module_name] = []
            self.imported_names[module_name].append(imported_name)
            
            # Record the line number
            self.import_lines[imported_name] = node.lineno
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        module_name = node.module or ""
        for name in node.names:
            self.import_count += 1
            
            # Record the imported module
            if module_name:
                self.imported_modules.append(module_name)
            
            # Record the name
            imported_name = name.asname or name.name
            if module_name not in self.imported_names:
                self.imported_names[module_name] = []
            self.imported_names[module_name].append(imported_name)
            
            # Record the line number
            self.import_lines[imported_name] = node.lineno
        
        self.generic_visit(node)


class _NameVisitor(ast.NodeVisitor):
    """AST visitor to collect used names."""
    
    def __init__(self):
        self.used_names = set()
    
    def visit_Name(self, node):
        self.used_names.add(node.id)
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        self.used_names.add(node.attr)
        self.generic_visit(node)


class _BugVisitor(ast.NodeVisitor):
    """AST visitor to detect potential bugs."""
    
    def __init__(self):
        self.issues = []
    
    def visit_BinOp(self, node):
        # Check for division by zero
        if isinstance(node.op, ast.Div) and isinstance(node.right, ast.Constant) and node.right.value == 0:
            self.issues.append({
                "type": "division_by_zero",
                "message": "Division by zero detected",
                "line": node.lineno if hasattr(node, 'lineno') else 0,
                "severity": "error"
            })
        
        self.generic_visit(node)
    
    def visit_Compare(self, node):
        # Check for is/is not comparison with literals
        for i, op in enumerate(node.ops):
            if isinstance(op, (ast.Is, ast.IsNot)):
                comparator = node.comparators[i]
                if isinstance(comparator, ast.Constant) and not isinstance(comparator.value, (type(None), bool)):
                    op_name = "is" if isinstance(op, ast.Is) else "is not"
                    self.issues.append({
                        "type": "identity_comparison",
                        "message": f"Use == instead of {op_name} for literal comparison",
                        "line": node.lineno if hasattr(node, 'lineno') else 0,
                        "severity": "warning"
                    })
        
        self.generic_visit(node)
    
    def visit_Except(self, node):
        # Check for bare except
        if node.type is None:
            self.issues.append({
                "type": "bare_except",
                "message": "Bare except clause detected - specify exceptions to catch",
                "line": node.lineno if hasattr(node, 'lineno') else 0,
                "severity": "warning"
            })
        
        self.generic_visit(node)


class _StyleVisitor(ast.NodeVisitor):
    """AST visitor to detect style issues."""
    
    def __init__(self):
        self.issues = []
        self.function_names = set()
        self.class_names = set()
        self.variable_names = set()
    
    def visit_FunctionDef(self, node):
        # Check function naming convention (snake_case)
        self.function_names.add(node.name)
        if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
            self.issues.append({
                "type": "function_naming",
                "message": f"Function '{node.name}' does not follow snake_case convention",
                "line": node.lineno,
                "severity": "style"
            })
        
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        # Check function naming convention (snake_case)
        self.function_names.add(node.name)
        if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
            self.issues.append({
                "type": "function_naming",
                "message": f"Async function '{node.name}' does not follow snake_case convention",
                "line": node.lineno,
                "severity": "style"
            })
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        # Check class naming convention (PascalCase)
        self.class_names.add(node.name)
        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
            self.issues.append({
                "type": "class_naming",
                "message": f"Class '{node.name}' does not follow PascalCase convention",
                "line": node.lineno,
                "severity": "style"
            })
        
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        # Check variable naming convention (snake_case)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variable_names.add(target.id)
                
                # Skip constants (all uppercase)
                if re.match(r'^[A-Z_][A-Z0-9_]*$', target.id):
                    continue
                
                if not re.match(r'^[a-z_][a-z0-9_]*$', target.id):
                    self.issues.append({
                        "type": "variable_naming",
                        "message": f"Variable '{target.id}' does not follow snake_case convention",
                        "line": node.lineno,
                        "severity": "style"
                    })
        
        self.generic_visit(node)
