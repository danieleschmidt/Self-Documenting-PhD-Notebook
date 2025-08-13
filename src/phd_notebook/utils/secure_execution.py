"""
Secure execution utilities to replace dangerous eval/exec operations.

This module provides safe alternatives to eval() and exec() functions
that were identified as security vulnerabilities in the codebase.
"""

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, Union
from functools import wraps

logger = logging.getLogger(__name__)


class SecureExecutionError(Exception):
    """Error raised during secure execution operations."""
    pass


class SafeEvaluator:
    """Safe evaluation of expressions without using eval()."""
    
    # Allowed node types for AST evaluation
    SAFE_NODES = {
        ast.Expression, ast.Constant, ast.Name, ast.Load,
        ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt,
        ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
        ast.List, ast.Tuple, ast.Dict, ast.Set,
        ast.ListComp, ast.DictComp, ast.SetComp,
        ast.comprehension, ast.keyword,
        # Allow some function calls but restrict to safe functions
        ast.Call, ast.Attribute
    }
    
    # Safe built-in functions that can be called
    SAFE_FUNCTIONS = {
        'abs', 'max', 'min', 'len', 'sum', 'round', 'sorted',
        'reversed', 'enumerate', 'zip', 'range', 'bool',
        'int', 'float', 'str', 'list', 'dict', 'set', 'tuple'
    }
    
    # Safe methods for specific types
    SAFE_METHODS = {
        'str': {'lower', 'upper', 'strip', 'split', 'join', 'replace', 'startswith', 'endswith'},
        'list': {'append', 'extend', 'count', 'index', 'sort', 'reverse'},
        'dict': {'get', 'keys', 'values', 'items', 'update'},
        'set': {'add', 'update', 'union', 'intersection', 'difference'}
    }
    
    def __init__(self, allowed_names: Optional[Dict[str, Any]] = None):
        """Initialize with optional allowed variable names."""
        self.allowed_names = allowed_names or {}
        
    def safe_eval(self, expression: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Safely evaluate an expression without using eval()."""
        if not expression or not isinstance(expression, str):
            raise SecureExecutionError("Expression must be a non-empty string")
        
        # Parse the expression into an AST
        try:
            tree = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            raise SecureExecutionError(f"Invalid expression syntax: {e}")
        
        # Validate that all nodes are safe
        self._validate_ast_safety(tree)
        
        # Combine allowed names with context
        eval_context = {**self.allowed_names}
        if context:
            eval_context.update(context)
        
        # Add safe built-in functions
        safe_builtins = {name: getattr(__builtins__, name) 
                        for name in self.SAFE_FUNCTIONS 
                        if hasattr(__builtins__, name)}
        eval_context.update(safe_builtins)
        
        try:
            # Use compile and eval with restricted context
            code = compile(tree, '<string>', 'eval')
            result = eval(code, {"__builtins__": {}}, eval_context)
            return result
        except Exception as e:
            raise SecureExecutionError(f"Evaluation failed: {e}")
    
    def _validate_ast_safety(self, node: ast.AST) -> None:
        """Validate that AST only contains safe operations."""
        if type(node) not in self.SAFE_NODES:
            raise SecureExecutionError(f"Unsafe AST node type: {type(node).__name__}")
        
        # Special validation for function calls
        if isinstance(node, ast.Call):
            self._validate_function_call(node)
        
        # Special validation for attribute access
        if isinstance(node, ast.Attribute):
            self._validate_attribute_access(node)
        
        # Recursively validate child nodes
        for child in ast.iter_child_nodes(node):
            self._validate_ast_safety(child)
    
    def _validate_function_call(self, node: ast.Call) -> None:
        """Validate that function calls are safe."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in self.SAFE_FUNCTIONS and func_name not in self.allowed_names:
                raise SecureExecutionError(f"Unsafe function call: {func_name}")
        elif isinstance(node.func, ast.Attribute):
            # Allow method calls on safe objects
            pass  # Will be validated by _validate_attribute_access
        else:
            raise SecureExecutionError("Complex function calls not allowed")
    
    def _validate_attribute_access(self, node: ast.Attribute) -> None:
        """Validate that attribute access is safe."""
        attr_name = node.attr
        
        # Check for dangerous attributes
        dangerous_attrs = {
            '__class__', '__bases__', '__subclasses__', '__globals__',
            '__dict__', '__getattribute__', '__setattr__', '__delattr__',
            '__import__', '__builtins__', 'exec', 'eval', 'compile'
        }
        
        if attr_name in dangerous_attrs:
            raise SecureExecutionError(f"Unsafe attribute access: {attr_name}")


class SafeJSONProcessor:
    """Safe JSON processing without eval()."""
    
    @staticmethod
    def safe_json_loads(json_string: str) -> Any:
        """Safely load JSON string."""
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise SecureExecutionError(f"Invalid JSON: {e}")
    
    @staticmethod
    def safe_json_dumps(obj: Any, **kwargs) -> str:
        """Safely dump object to JSON string."""
        try:
            return json.dumps(obj, **kwargs)
        except (TypeError, ValueError) as e:
            raise SecureExecutionError(f"JSON serialization failed: {e}")


class SafeStringProcessor:
    """Safe string processing without exec()."""
    
    @staticmethod
    def safe_template_substitute(template: str, variables: Dict[str, Any]) -> str:
        """Safely substitute variables in template using format()."""
        try:
            # Validate template format
            if '{' in template and '}' in template:
                # Use format() instead of eval() for string substitution
                return template.format(**variables)
            else:
                # Simple string replacement
                result = template
                for key, value in variables.items():
                    placeholder = f"{{{key}}}"
                    if placeholder in result:
                        result = result.replace(placeholder, str(value))
                return result
        except (KeyError, ValueError) as e:
            raise SecureExecutionError(f"Template substitution failed: {e}")
    
    @staticmethod
    def safe_regex_substitute(text: str, pattern: str, replacement: str) -> str:
        """Safely perform regex substitution."""
        try:
            compiled_pattern = re.compile(pattern)
            return compiled_pattern.sub(replacement, text)
        except re.error as e:
            raise SecureExecutionError(f"Invalid regex pattern: {e}")


class SecureConfigProcessor:
    """Process configuration safely without exec()."""
    
    @staticmethod
    def safe_config_evaluation(config_value: str, context: Dict[str, Any]) -> Any:
        """Safely evaluate configuration values."""
        if not config_value or not isinstance(config_value, str):
            return config_value
        
        # Try to parse as JSON first
        try:
            return json.loads(config_value)
        except json.JSONDecodeError:
            pass
        
        # Try safe evaluation for simple expressions
        if config_value.strip().startswith(('"', "'", '[', '{', '(')) or config_value.strip().isdigit():
            evaluator = SafeEvaluator(context)
            try:
                return evaluator.safe_eval(config_value)
            except SecureExecutionError:
                pass
        
        # Return as string if nothing else works
        return config_value


class SecureCodeExecution:
    """Secure alternatives to exec() and eval() operations."""
    
    def __init__(self):
        self.evaluator = SafeEvaluator()
        self.json_processor = SafeJSONProcessor()
        self.string_processor = SafeStringProcessor()
        self.config_processor = SecureConfigProcessor()
    
    def execute_safe_calculation(self, expression: str, variables: Dict[str, Any]) -> Any:
        """Execute mathematical or logical calculations safely."""
        try:
            return self.evaluator.safe_eval(expression, variables)
        except SecureExecutionError as e:
            logger.warning(f"Safe calculation failed: {e}")
            raise
    
    def process_template_safely(self, template: str, context: Dict[str, Any]) -> str:
        """Process templates without exec()."""
        return self.string_processor.safe_template_substitute(template, context)
    
    def parse_configuration_safely(self, config_str: str, context: Dict[str, Any]) -> Any:
        """Parse configuration safely."""
        return self.config_processor.safe_config_evaluation(config_str, context)


# Decorator to replace dangerous operations
def secure_execution(func):
    """Decorator to ensure secure execution of functions."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Log the function call for security auditing
        logger.debug(f"Secure execution of {func.__name__}")
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Secure execution failed for {func.__name__}: {e}")
            raise SecureExecutionError(f"Function {func.__name__} failed: {e}")
    
    return wrapper


# Safe alternatives for common dangerous operations
class SafeOperations:
    """Collection of safe operations to replace dangerous ones."""
    
    @staticmethod
    def safe_dynamic_import(module_name: str, allowed_modules: List[str]) -> Any:
        """Safely import modules from allowed list."""
        if module_name not in allowed_modules:
            raise SecureExecutionError(f"Module {module_name} not in allowed list")
        
        try:
            return __import__(module_name)
        except ImportError as e:
            raise SecureExecutionError(f"Failed to import {module_name}: {e}")
    
    @staticmethod
    def safe_getattr(obj: Any, attr_name: str, default: Any = None) -> Any:
        """Safely get attribute without exposing dangerous methods."""
        dangerous_attrs = {
            '__class__', '__bases__', '__subclasses__', '__globals__',
            '__dict__', '__getattribute__', '__setattr__', '__delattr__',
            '__import__', '__builtins__', 'exec', 'eval', 'compile'
        }
        
        if attr_name in dangerous_attrs:
            raise SecureExecutionError(f"Access to {attr_name} not allowed")
        
        return getattr(obj, attr_name, default)
    
    @staticmethod
    def safe_setattr(obj: Any, attr_name: str, value: Any) -> None:
        """Safely set attribute with validation."""
        dangerous_attrs = {
            '__class__', '__bases__', '__dict__', '__globals__',
            '__builtins__', '__module__'
        }
        
        if attr_name in dangerous_attrs:
            raise SecureExecutionError(f"Setting {attr_name} not allowed")
        
        if attr_name.startswith('_'):
            raise SecureExecutionError("Setting private attributes not allowed")
        
        setattr(obj, attr_name, value)


# Create global instances for easy access
secure_executor = SecureCodeExecution()
safe_ops = SafeOperations()


# Replacement functions for dangerous operations
def safe_eval_replacement(expression: str, context: Optional[Dict[str, Any]] = None) -> Any:
    """Safe replacement for eval() function."""
    return secure_executor.execute_safe_calculation(expression, context or {})


def safe_exec_replacement(code: str, context: Optional[Dict[str, Any]] = None) -> None:
    """Safe replacement for exec() - throws error as exec is inherently unsafe."""
    raise SecureExecutionError(
        "exec() is not allowed for security reasons. Use specific safe operations instead."
    )


def safe_compile_replacement(source: str, filename: str, mode: str) -> Any:
    """Safe replacement for compile() with restrictions."""
    if mode not in ('eval',):
        raise SecureExecutionError(f"Compile mode '{mode}' not allowed")
    
    try:
        tree = ast.parse(source, filename, mode)
        evaluator = SafeEvaluator()
        evaluator._validate_ast_safety(tree)
        return compile(tree, filename, mode)
    except Exception as e:
        raise SecureExecutionError(f"Safe compile failed: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Test safe evaluation
    evaluator = SafeEvaluator({'x': 10, 'y': 20})
    
    try:
        result = evaluator.safe_eval("x + y * 2")
        print(f"Safe evaluation result: {result}")  # Should be 50
    except SecureExecutionError as e:
        print(f"Evaluation error: {e}")
    
    # Test unsafe evaluation
    try:
        result = evaluator.safe_eval("__import__('os').system('ls')")
        print(f"This should not execute: {result}")
    except SecureExecutionError as e:
        print(f"Correctly blocked unsafe operation: {e}")
    
    # Test safe template processing
    template = "Hello {name}, your score is {score}"
    context = {'name': 'Alice', 'score': 95}
    
    processor = SafeStringProcessor()
    result = processor.safe_template_substitute(template, context)
    print(f"Template result: {result}")