"""
TypeScript Language Support

This module implements TypeScript language support for Jarviee's programming
capabilities, providing parsing, analysis, and generation features
specifically tailored for TypeScript code.
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import ProgrammingLanguage, LanguageFeature, SyntaxElement


class TypeScriptLanguage(ProgrammingLanguage):
    """
    TypeScript language implementation.
    
    This class provides TypeScript-specific functionality for code analysis,
    generation, and manipulation.
    """
    
    def __init__(self):
        """Initialize TypeScript language support."""
        super().__init__(
            name="TypeScript",
            file_extensions=[".ts", ".tsx", ".d.ts"]
        )
        
        # Set TypeScript features
        self.features.update([
            LanguageFeature.STATIC_TYPING,
            LanguageFeature.OBJECT_ORIENTED,
            LanguageFeature.FUNCTIONAL,
            LanguageFeature.PROCEDURAL,
            LanguageFeature.REFLECTIVE,
            LanguageFeature.EXCEPTION_HANDLING,
            LanguageFeature.GENERICS,
            LanguageFeature.FIRST_CLASS_FUNCTIONS,
            LanguageFeature.CLOSURES,
            LanguageFeature.TYPE_INFERENCE,
            LanguageFeature.MODULE_SYSTEM,
            LanguageFeature.PACKAGE_MANAGER,
        ])
        
        # Define syntax patterns
        self.syntax_patterns.update({
            SyntaxElement.COMMENT: r'(?://.*$|/\*[\s\S]*?\*/)',
            SyntaxElement.FUNCTION_DEFINITION: r'(?:function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(([^)]*)\)(?:\s*:\s*([^{]*))?|(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:\([^)]*\)|)(?:\s*:\s*[^=]*?)?\s*=>\s*)',
            SyntaxElement.CLASS_DEFINITION: r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)(?:\s+extends\s+([a-zA-Z_$][a-zA-Z0-9_$.]*))?(?:\s+implements\s+([a-zA-Z_$][a-zA-Z0-9_$.]*(?:\s*,\s*[a-zA-Z_$][a-zA-Z0-9_$.]*)*))?\s*{',
            SyntaxElement.IMPORT_STATEMENT: r'import\s+(?:{[^}]*}|[a-zA-Z_$][a-zA-Z0-9_$]*|\*\s+as\s+[a-zA-Z_$][a-zA-Z0-9_$]*)\s+from\s+[\'"]([^\'"]+)[\'"]',
            SyntaxElement.VARIABLE_DECLARATION: r'(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)(?:\s*:\s*([^=]*))?(?:\s*=\s*([^;]*))?',
            SyntaxElement.INTERFACE_DEFINITION: r'interface\s+([a-zA-Z_$][a-zA-Z0-9_$]*)(?:\s+extends\s+([a-zA-Z_$][a-zA-Z0-9_$.]*(?:\s*,\s*[a-zA-Z_$][a-zA-Z0-9_$.]*)*))?\s*{',
            SyntaxElement.TYPE_DEFINITION: r'type\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(?:<[^>]*>)?\s*=\s*',
        })
    
    def parse_code(self, code: str) -> Any:
        """
        Parse TypeScript code using TypeScript compiler API.
        
        Args:
            code: The TypeScript code to parse
            
        Returns:
            An AST-like representation of the code
            
        Note:
            This implementation uses a simplified representation as a fallback
            when the TypeScript compiler is not available.
        """
        try:
            # Try to use TypeScript compiler if available
            return self._parse_with_tsc(code)
        except Exception:
            # Fall back to simplified parsing
            return self._parse_simple(code)
    
    def _parse_with_tsc(self, code: str) -> Dict[str, Any]:
        """Parse TypeScript code using the TypeScript compiler."""
        # This would normally use the TypeScript compiler API
        # For simplicity, we'll use a mock implementation that parses basic structures
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as temp:
            temp_path = temp.name
            temp.write(code.encode('utf-8'))
        
        try:
            # Run TypeScript compiler in AST mode (requires TypeScript to be installed)
            result = subprocess.run(
                [
                    "npx", "typescript", "--noEmit", "--pretty", "false", 
                    "--target", "ES2020", temp_path
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            # In a real implementation, we would parse the output and build an AST
            # For now, we'll just use our simplified parser
            return self._parse_simple(code)
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
    
    def _parse_simple(self, code: str) -> Dict[str, Any]:
        """Simplified TypeScript parser implementation."""
        result = {
            "type": "Program",
            "imports": [],
            "exports": [],
            "declarations": [],
            "interfaces": [],
            "classes": [],
            "functions": [],
            "variables": [],
        }
        
        # Extract imports
        import_pattern = self.syntax_patterns[SyntaxElement.IMPORT_STATEMENT]
        for match in re.finditer(import_pattern, code):
            result["imports"].append({
                "module": match.group(1),
                "start": match.start(),
                "end": match.end(),
            })
        
        # Extract exports
        export_pattern = r'export\s+(?:{[^}]*}|default\s+|(?:class|interface|function|const|let|var|type))'
        for match in re.finditer(export_pattern, code):
            result["exports"].append({
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
            })
        
        # Extract interfaces
        interface_pattern = self.syntax_patterns[SyntaxElement.INTERFACE_DEFINITION]
        for match in re.finditer(interface_pattern, code):
            name = match.group(1)
            extends = match.group(2)
            
            # Find the interface body
            start_pos = match.end()
            brace_count = 1
            end_pos = start_pos
            
            for i in range(start_pos, len(code)):
                if code[i] == '{':
                    brace_count += 1
                elif code[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            result["interfaces"].append({
                "name": name,
                "extends": extends.split(',') if extends else [],
                "start": match.start(),
                "end": end_pos,
                "body": code[start_pos:end_pos],
            })
        
        # Extract classes
        class_pattern = self.syntax_patterns[SyntaxElement.CLASS_DEFINITION]
        for match in re.finditer(class_pattern, code):
            name = match.group(1)
            extends = match.group(2)
            implements = match.group(3)
            
            # Find the class body
            start_pos = match.end()
            brace_count = 1
            end_pos = start_pos
            
            for i in range(start_pos, len(code)):
                if code[i] == '{':
                    brace_count += 1
                elif code[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            result["classes"].append({
                "name": name,
                "extends": extends if extends else None,
                "implements": implements.split(',') if implements else [],
                "start": match.start(),
                "end": end_pos,
                "body": code[start_pos:end_pos],
            })
        
        # Extract functions
        function_pattern = self.syntax_patterns[SyntaxElement.FUNCTION_DEFINITION]
        for match in re.finditer(function_pattern, code):
            if match.group(1):  # Named function declaration
                name = match.group(1)
                params = match.group(2)
                return_type = match.group(3)
                
                result["functions"].append({
                    "name": name,
                    "params": params,
                    "return_type": return_type.strip() if return_type else None,
                    "start": match.start(),
                    "end": match.end(),
                })
            elif match.group(4):  # Arrow function
                name = match.group(4)
                
                result["functions"].append({
                    "name": name,
                    "type": "arrow",
                    "start": match.start(),
                    "end": match.end(),
                })
        
        # Extract variables
        variable_pattern = self.syntax_patterns[SyntaxElement.VARIABLE_DECLARATION]
        for match in re.finditer(variable_pattern, code):
            name = match.group(1)
            type_annotation = match.group(2)
            initializer = match.group(3)
            
            result["variables"].append({
                "name": name,
                "type": type_annotation.strip() if type_annotation else None,
                "initializer": initializer.strip() if initializer else None,
                "start": match.start(),
                "end": match.end(),
            })
        
        return result
    
    def format_code(self, code: str) -> str:
        """
        Format TypeScript code using Prettier.
        
        Args:
            code: The code to format
            
        Returns:
            Formatted code
            
        Note:
            This implementation attempts to use Prettier if available,
            otherwise it returns the original code with some basic formatting.
        """
        try:
            # Try to use Prettier if available
            with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as temp:
                temp_path = temp.name
                temp.write(code.encode('utf-8'))
            
            try:
                # Run Prettier
                result = subprocess.run(
                    ["npx", "prettier", "--write", temp_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Read the formatted code
                with open(temp_path, 'r', encoding='utf-8') as f:
                    formatted_code = f.read()
                
                return formatted_code
            finally:
                # Clean up
                Path(temp_path).unlink(missing_ok=True)
        except Exception:
            # Fall back to basic formatting
            return self._basic_format(code)
    
    def _basic_format(self, code: str) -> str:
        """Apply basic formatting to TypeScript code."""
        # This is a very simplified formatter
        lines = code.split('\n')
        
        # Track indentation level
        indent_level = 0
        formatted_lines = []
        
        for line in lines:
            # Strip leading/trailing whitespace
            stripped = line.strip()
            
            if not stripped:
                # Preserve empty lines
                formatted_lines.append('')
                continue
            
            # Adjust indentation for this line
            if stripped.startswith('}') or stripped.startswith(')'):
                indent_level = max(0, indent_level - 1)
            
            # Add the line with proper indentation
            formatted_lines.append('  ' * indent_level + stripped)
            
            # Adjust indentation for the next line
            if stripped.endswith('{') or stripped.endswith('('):
                indent_level += 1
        
        return '\n'.join(formatted_lines)
    
    def analyze_quality(self, code: str) -> Dict[str, Any]:
        """
        Analyze TypeScript code quality.
        
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
            # Parse the code
            parsed = self.parse_code(code)
            
            # Count basic metrics
            lines = code.split('\n')
            result["metrics"]["lines_total"] = len(lines)
            result["metrics"]["lines_code"] = sum(1 for line in lines if line.strip() and not re.match(r'^\s*//.*$', line))
            result["metrics"]["lines_comment"] = sum(1 for line in lines if re.match(r'^\s*//.*$', line))
            result["metrics"]["lines_empty"] = sum(1 for line in lines if not line.strip())
            
            # Count declarations
            result["metrics"]["classes"] = len(parsed["classes"])
            result["metrics"]["interfaces"] = len(parsed["interfaces"])
            result["metrics"]["functions"] = len(parsed["functions"])
            result["metrics"]["variables"] = len(parsed["variables"])
            
            # Calculate complexity
            result["complexity"]["cyclomatic"] = self._calculate_cyclomatic_complexity(code)
            result["complexity"]["nesting"] = self._calculate_max_nesting(code)
            
            # Style checks
            result["style"]["line_length"] = {
                "max": max((len(line) for line in lines), default=0),
                "violations": sum(1 for line in lines if len(line) > 100)
            }
            
            # Import analysis
            result["metrics"]["imports"] = {
                "count": len(parsed["imports"]),
                "modules": [imp["module"] for imp in parsed["imports"]]
            }
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity of TypeScript code."""
        # Count decision points
        decision_points = 0
        
        # Count if statements
        decision_points += len(re.findall(r'\bif\s*\(', code))
        
        # Count else if statements
        decision_points += len(re.findall(r'\belse\s+if\s*\(', code))
        
        # Count switches
        decision_points += len(re.findall(r'\bswitch\s*\(', code))
        
        # Count case statements
        decision_points += len(re.findall(r'\bcase\s+[^:]+:', code))
        
        # Count loops (for, while, do-while)
        decision_points += len(re.findall(r'\bfor\s*\(', code))
        decision_points += len(re.findall(r'\bwhile\s*\(', code))
        decision_points += len(re.findall(r'\bdo\s*{', code))
        
        # Count conditional expressions (ternary operators)
        decision_points += len(re.findall(r'\?.*:', code))
        
        # Count logical operators (&&, ||)
        decision_points += len(re.findall(r'&&|\|\|', code))
        
        # Count catch blocks
        decision_points += len(re.findall(r'\bcatch\s*\(', code))
        
        # Base complexity is 1, add decision points
        return 1 + decision_points
    
    def _calculate_max_nesting(self, code: str) -> int:
        """Calculate maximum nesting level in TypeScript code."""
        lines = code.split('\n')
        max_nesting = 0
        current_nesting = 0
        
        for line in lines:
            # Count opening braces
            open_braces = line.count('{')
            
            # Count closing braces
            close_braces = line.count('}')
            
            # Update nesting level
            current_nesting += open_braces - close_braces
            
            # Update maximum nesting
            max_nesting = max(max_nesting, current_nesting)
        
        return max_nesting
    
    def detect_issues(self, code: str) -> List[Dict[str, Any]]:
        """
        Detect issues in TypeScript code.
        
        Args:
            code: The code to analyze
            
        Returns:
            A list of issues, each represented as a dictionary
        """
        issues = []
        
        try:
            # Try to use ESLint if available
            issues.extend(self._eslint_check(code))
        except Exception:
            # Fall back to basic checks
            issues.extend(self._basic_checks(code))
        
        return issues
    
    def _eslint_check(self, code: str) -> List[Dict[str, Any]]:
        """Check TypeScript code with ESLint."""
        issues = []
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as temp:
            temp_path = temp.name
            temp.write(code.encode('utf-8'))
        
        try:
            # Run ESLint
            result = subprocess.run(
                [
                    "npx", "eslint", "--format", "json",
                    "--no-eslintrc", "--rule", "semi:error", temp_path
                ],
                capture_output=True,
                text=True
            )
            
            # Parse the output
            if result.stdout:
                try:
                    eslint_results = json.loads(result.stdout)
                    
                    for file_result in eslint_results:
                        for message in file_result.get("messages", []):
                            issues.append({
                                "type": message.get("ruleId", "unknown"),
                                "message": message.get("message", ""),
                                "line": message.get("line", 0),
                                "column": message.get("column", 0),
                                "severity": "error" if message.get("severity") == 2 else "warning",
                            })
                except json.JSONDecodeError:
                    pass
        
        except Exception:
            pass
        
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
        
        return issues
    
    def _basic_checks(self, code: str) -> List[Dict[str, Any]]:
        """Perform basic checks on TypeScript code."""
        issues = []
        
        # Check for common issues
        
        # Missing semicolons
        lines = code.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines, comments, and lines that end with specific characters
            if (not stripped or 
                stripped.startswith('//') or 
                stripped.startswith('/*') or 
                stripped.endswith(';') or 
                stripped.endswith('{') or 
                stripped.endswith('}') or 
                stripped.endswith(',') or 
                stripped.endswith('(') or
                stripped.endswith(')')):
                continue
            
            # Check if the line should end with a semicolon
            if re.match(r'^(?!function|class|interface|type|import|export|if|for|while|switch|try|catch|else|do|return).*[a-zA-Z0-9_$\'")]', stripped):
                issues.append({
                    "type": "missing_semicolon",
                    "message": "Missing semicolon",
                    "line": i + 1,
                    "severity": "warning"
                })
        
        # TODO: Add more basic checks
        
        return issues
    
    def get_documentation_template(self, code_element: str) -> str:
        """
        Get a documentation template for a TypeScript code element.
        
        Args:
            code_element: The code element to document
            
        Returns:
            A documentation template string in JSDoc format
        """
        # Try to determine the type of element
        code_element = code_element.strip()
        
        if code_element.startswith('function '):
            # Function declaration
            return self._generate_function_jsdoc(code_element)
        
        elif code_element.startswith('class '):
            # Class declaration
            return self._generate_class_jsdoc(code_element)
        
        elif code_element.startswith('interface '):
            # Interface declaration
            return self._generate_interface_jsdoc(code_element)
        
        elif re.match(r'^(const|let|var)\s+.*?=\s*function\s*\(', code_element):
            # Function expression
            return self._generate_function_jsdoc(code_element)
        
        elif re.match(r'^(const|let|var)\s+.*?=\s*\(.*?\)\s*=>', code_element):
            # Arrow function
            return self._generate_function_jsdoc(code_element)
        
        elif re.match(r'^(const|let|var)\s+', code_element):
            # Variable declaration
            return self._generate_variable_jsdoc(code_element)
        
        else:
            # Generic template
            return '/**\n * Description\n *\n * @remarks\n * Additional details about this code element\n */\n'
    
    def _generate_function_jsdoc(self, function_code: str) -> str:
        """Generate JSDoc for a function."""
        # Extract function name and parameters
        func_match = re.search(r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(([^)]*)\)', function_code)
        arrow_match = re.search(r'(const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:\(([^)]*)\)|([a-zA-Z_$][a-zA-Z0-9_$]*))\s*=>', function_code)
        
        if func_match:
            name = func_match.group(1)
            params_str = func_match.group(2)
        elif arrow_match:
            name = arrow_match.group(2)
            params_str = arrow_match.group(3) or arrow_match.group(4) or ""
        else:
            name = "function"
            params_str = ""
        
        # Parse parameters
        params = []
        if params_str:
            for param in params_str.split(','):
                param = param.strip()
                if param:
                    # Handle parameter with type annotation
                    param_match = re.match(r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(?::\s*([^=]*))?', param)
                    if param_match:
                        param_name = param_match.group(1)
                        param_type = param_match.group(2).strip() if param_match.group(2) else "any"
                        params.append((param_name, param_type))
                    else:
                        params.append((param, "any"))
        
        # Extract return type
        return_type = "void"
        return_match = re.search(r'function\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*\([^)]*\)\s*:\s*([^{]+)', function_code)
        if return_match:
            return_type = return_match.group(1).strip()
        
        # Build JSDoc
        jsdoc = f'/**\n * {name}\n *\n * @description\n * Description of {name}\n *\n'
        
        # Add parameters
        for param_name, param_type in params:
            jsdoc += f' * @param {param_name} - Description of {param_name}\n'
        
        # Add return
        if return_type != "void":
            jsdoc += f' * @returns Description of return value\n'
        
        jsdoc += ' */\n'
        return jsdoc
    
    def _generate_class_jsdoc(self, class_code: str) -> str:
        """Generate JSDoc for a class."""
        # Extract class name
        class_match = re.search(r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', class_code)
        name = class_match.group(1) if class_match else "Class"
        
        # Build JSDoc
        jsdoc = f'/**\n * {name}\n *\n * @description\n * Description of {name}\n *\n'
        
        # Check if the class extends another class
        extends_match = re.search(r'class\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s+extends\s+([a-zA-Z_$][a-zA-Z0-9_$.]*)', class_code)
        if extends_match:
            jsdoc += f' * @extends {extends_match.group(1)}\n'
        
        # Check if the class implements interfaces
        implements_match = re.search(r'class\s+[a-zA-Z_$][a-zA-Z0-9_$]*(?:\s+extends\s+[a-zA-Z_$][a-zA-Z0-9_$.]*)?(?:\s+implements\s+([a-zA-Z_$][a-zA-Z0-9_$.]*(?:\s*,\s*[a-zA-Z_$][a-zA-Z0-9_$.]*)*))?\s*{', class_code)
        if implements_match and implements_match.group(1):
            implements = [intf.strip() for intf in implements_match.group(1).split(',')]
            for intf in implements:
                jsdoc += f' * @implements {intf}\n'
        
        jsdoc += ' */\n'
        return jsdoc
    
    def _generate_interface_jsdoc(self, interface_code: str) -> str:
        """Generate JSDoc for an interface."""
        # Extract interface name
        interface_match = re.search(r'interface\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', interface_code)
        name = interface_match.group(1) if interface_match else "Interface"
        
        # Build JSDoc
        jsdoc = f'/**\n * {name} interface\n *\n * @description\n * Description of {name}\n *\n'
        
        # Check if the interface extends other interfaces
        extends_match = re.search(r'interface\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s+extends\s+([a-zA-Z_$][a-zA-Z0-9_$.]*(?:\s*,\s*[a-zA-Z_$][a-zA-Z0-9_$.]*)*)', interface_code)
        if extends_match:
            extends = [ext.strip() for ext in extends_match.group(1).split(',')]
            for ext in extends:
                jsdoc += f' * @extends {ext}\n'
        
        jsdoc += ' */\n'
        return jsdoc
    
    def _generate_variable_jsdoc(self, variable_code: str) -> str:
        """Generate JSDoc for a variable."""
        # Extract variable name and type
        var_match = re.search(r'(const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)(?:\s*:\s*([^=]*))?', variable_code)
        
        if var_match:
            var_kind = var_match.group(1)
            name = var_match.group(2)
            var_type = var_match.group(3).strip() if var_match.group(3) else "any"
        else:
            var_kind = "const"
            name = "variable"
            var_type = "any"
        
        # Build JSDoc
        jsdoc = f'/**\n * {name}\n *\n * @description\n * Description of {name}\n *\n'
        
        # Add type
        jsdoc += f' * @type {{{var_type}}}\n'
        
        jsdoc += ' */\n'
        return jsdoc
    
    def extract_imports(self, code: str) -> List[str]:
        """
        Extract import statements from TypeScript code.
        
        Args:
            code: The code to analyze
            
        Returns:
            A list of import statements
        """
        imports = []
        
        # Match import statements
        import_pattern = r'import\s+(?:{[^}]*}|[a-zA-Z_$][a-zA-Z0-9_$]*|\*\s+as\s+[a-zA-Z_$][a-zA-Z0-9_$]*)\s+from\s+[\'"]([^\'"]+)[\'"]'
        for match in re.finditer(import_pattern, code):
            # Extract the entire import statement
            start = match.start()
            end = match.end()
            import_stmt = code[start:end]
            imports.append(import_stmt)
        
        return imports
    
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract function definitions from TypeScript code.
        
        Args:
            code: The code to analyze
            
        Returns:
            A list of function information dictionaries
        """
        functions = []
        
        try:
            # Parse the code
            parsed = self.parse_code(code)
            
            # Return the functions from the parsed code
            return parsed["functions"]
        
        except Exception:
            # Fall back to regex-based extraction
            
            # Match function declarations
            func_pattern = r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?\s*{'
            for match in re.finditer(func_pattern, code):
                name = match.group(1)
                params = match.group(2)
                return_type = match.group(3)
                
                functions.append({
                    "name": name,
                    "params": params.strip(),
                    "return_type": return_type.strip() if return_type else None,
                    "start": match.start(),
                    "end": match.end(),
                })
            
            # Match arrow functions
            arrow_pattern = r'(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:\(([^)]*)\)|([a-zA-Z_$][a-zA-Z0-9_$]*))\s*(?::\s*([^=]*))?=>'
            for match in re.finditer(arrow_pattern, code):
                name = match.group(1)
                params = match.group(2) or match.group(3) or ""
                return_type = match.group(4)
                
                functions.append({
                    "name": name,
                    "params": params.strip(),
                    "return_type": return_type.strip() if return_type else None,
                    "type": "arrow",
                    "start": match.start(),
                    "end": match.end(),
                })
            
            # Match method declarations
            method_pattern = r'(?:public|private|protected)?\s*(?:static)?\s*(?:async)?\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?\s*{'
            for match in re.finditer(method_pattern, code):
                name = match.group(1)
                params = match.group(2)
                return_type = match.group(3)
                
                # Skip constructor
                if name == 'constructor':
                    continue
                
                functions.append({
                    "name": name,
                    "params": params.strip(),
                    "return_type": return_type.strip() if return_type else None,
                    "type": "method",
                    "start": match.start(),
                    "end": match.end(),
                })
        
        return functions
    
    def extract_classes(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract class definitions from TypeScript code.
        
        Args:
            code: The code to analyze
            
        Returns:
            A list of class information dictionaries
        """
        try:
            # Parse the code
            parsed = self.parse_code(code)
            
            # Return the classes from the parsed code
            return parsed["classes"]
        
        except Exception:
            # Fall back to regex-based extraction
            classes = []
            
            # Match class declarations
            class_pattern = r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)(?:\s+extends\s+([a-zA-Z_$][a-zA-Z0-9_$.]*))?\s*(?:implements\s+([a-zA-Z_$][a-zA-Z0-9_$.]*(?:\s*,\s*[a-zA-Z_$][a-zA-Z0-9_$.]*)*))?\s*{'
            for match in re.finditer(class_pattern, code):
                name = match.group(1)
                extends = match.group(2)
                implements = match.group(3)
                
                # Find methods and properties
                # This is a simplified approach
                methods = []
                properties = []
                
                # Extract the class body
                start_pos = match.end()
                brace_count = 1
                end_pos = start_pos
                
                for i in range(start_pos, len(code)):
                    if code[i] == '{':
                        brace_count += 1
                    elif code[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                
                class_body = code[start_pos:end_pos-1]
                
                # Extract methods
                method_pattern = r'(?:public|private|protected)?\s*(?:static)?\s*(?:async)?\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?\s*{'
                for m_match in re.finditer(method_pattern, class_body):
                    method_name = m_match.group(1)
                    methods.append({
                        "name": method_name,
                        "start": m_match.start() + start_pos,
                        "end": m_match.end() + start_pos,
                    })
                
                # Extract properties
                property_pattern = r'(?:public|private|protected)?\s*(?:readonly)?\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(?::\s*([^;=]+))?(?:\s*=\s*([^;]+))?;'
                for p_match in re.finditer(property_pattern, class_body):
                    property_name = p_match.group(1)
                    property_type = p_match.group(2)
                    properties.append({
                        "name": property_name,
                        "type": property_type.strip() if property_type else None,
                        "start": p_match.start() + start_pos,
                        "end": p_match.end() + start_pos,
                    })
                
                classes.append({
                    "name": name,
                    "extends": extends,
                    "implements": implements.split(',') if implements else [],
                    "methods": methods,
                    "properties": properties,
                    "start": match.start(),
                    "end": end_pos,
                })
        
        return classes
    
    def get_code_completion(
        self, 
        code: str, 
        cursor_position: int
    ) -> List[Dict[str, Any]]:
        """
        Get code completion suggestions for TypeScript code.
        
        Args:
            code: The code context
            cursor_position: Position of the cursor in the code
            
        Returns:
            A list of completion suggestions
        """
        # This is a simplified implementation
        # A real implementation would use TypeScript Language Service
        suggestions = []
        
        try:
            # Extract the current line and word being completed
            lines = code[:cursor_position].split('\n')
            current_line = lines[-1] if lines else ""
            
            # Extract the partial word being completed
            match = re.search(r'([a-zA-Z_$][a-zA-Z0-9_$]*)$', current_line)
            prefix = match.group(1) if match else ""
            
            # Parse the code to find declarations
            parsed = self.parse_code(code[:cursor_position])
            
            # Add variables
            for variable in parsed["variables"]:
                if not prefix or variable["name"].startswith(prefix):
                    suggestions.append({
                        "text": variable["name"],
                        "type": "variable",
                        "detail": f"({variable['type'] or 'any'}) Variable"
                    })
            
            # Add functions
            for function in parsed["functions"]:
                if not prefix or function["name"].startswith(prefix):
                    suggestions.append({
                        "text": function["name"],
                        "type": "function",
                        "detail": "Function"
                    })
            
            # Add classes
            for cls in parsed["classes"]:
                if not prefix or cls["name"].startswith(prefix):
                    suggestions.append({
                        "text": cls["name"],
                        "type": "class",
                        "detail": "Class"
                    })
            
            # Add interfaces
            for interface in parsed["interfaces"]:
                if not prefix or interface["name"].startswith(prefix):
                    suggestions.append({
                        "text": interface["name"],
                        "type": "interface",
                        "detail": "Interface"
                    })
            
            # Add TypeScript keywords
            keywords = [
                "as", "async", "await", "break", "case", "catch", "class", "const",
                "continue", "debugger", "default", "delete", "do", "else", "enum",
                "export", "extends", "false", "finally", "for", "from", "function",
                "if", "implements", "import", "in", "instanceof", "interface", "let",
                "new", "null", "package", "private", "protected", "public", "return",
                "super", "switch", "this", "throw", "true", "try", "type", "typeof",
                "var", "void", "while", "with", "yield"
            ]
            
            for keyword in keywords:
                if not prefix or keyword.startswith(prefix):
                    suggestions.append({
                        "text": keyword,
                        "type": "keyword",
                        "detail": "Keyword"
                    })
            
            # Add TypeScript types
            types = [
                "any", "boolean", "never", "null", "number", "object", "string",
                "symbol", "undefined", "unknown", "Array", "Date", "Error", "Map",
                "Promise", "RegExp", "Set", "WeakMap", "WeakSet"
            ]
            
            for type_name in types:
                if not prefix or type_name.startswith(prefix):
                    suggestions.append({
                        "text": type_name,
                        "type": "type",
                        "detail": "Type"
                    })
        
        except Exception:
            pass
        
        return suggestions
    
    def generate_test(self, code: str, function_name: Optional[str] = None) -> str:
        """
        Generate a Jest test for TypeScript code.
        
        Args:
            code: The code to test
            function_name: Optional name of a specific function to test
            
        Returns:
            Generated test code
        """
        try:
            # Parse the code
            parsed = self.parse_code(code)
            
            # Extract functions to test
            functions_to_test = []
            
            if function_name:
                # Test a specific function
                for func in parsed["functions"]:
                    if func["name"] == function_name:
                        functions_to_test.append(func)
                        break
            else:
                # Test all functions
                functions_to_test = parsed["functions"]
            
            if not functions_to_test:
                return f"// No {'function named ' + function_name if function_name else 'functions'} found in the code"
            
            # Generate test code
            test_code = "import { describe, expect, it } from 'jest';\n"
            
            # Import the module/function to test
            test_code += "// Import the code to test\n"
            test_code += f"// import {{ {', '.join(f['name'] for f in functions_to_test)} }} from './module';\n\n"
            
            # Generate test suite
            test_code += "describe('Module tests', () => {\n"
            
            # Generate test cases for each function
            for func in functions_to_test:
                test_code += f"  describe('{func['name']}', () => {{\n"
                test_code += f"    it('should work correctly', () => {{\n"
                test_code += f"      // TODO: Add test cases for {func['name']}\n"
                
                # Generate placeholder assertions based on parameters
                params = func.get("params", "")
                param_list = []
                
                if params:
                    for param in params.split(','):
                        param = param.strip()
                        if param:
                            # Extract parameter name
                            param_match = re.match(r'([a-zA-Z_$][a-zA-Z0-9_$]*)', param)
                            if param_match:
                                param_list.append(param_match.group(1))
                
                if param_list:
                    test_code += f"      // Example: const result = {func['name']}({', '.join(param_list)});\n"
                    test_code += f"      // expect(result).toBe(expectedResult);\n"
                else:
                    test_code += f"      // Example: const result = {func['name']}();\n"
                    test_code += f"      // expect(result).toBe(expectedResult);\n"
                
                test_code += "    });\n"
                test_code += "  });\n"
            
            test_code += "});\n"
            
            return test_code
        
        except Exception as e:
            return f"// Error generating test: {str(e)}\n// Please ensure the code is syntactically correct."
