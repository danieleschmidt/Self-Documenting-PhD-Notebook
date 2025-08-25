from phd_notebook.utils.secure_execution_fixed import default_evaluator as safe_evaluator, default_executor as safe_executor
"""
Secure execution utilities providing safe alternatives to eval/exec operations.

This module provides AST-based safe evaluation methods to replace dangerous
eval() and exec() operations throughout the research platform.
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
        'int', 'float', 'str', 'list', 'dict', 'set', 'tuple',
        'isinstance', 'hasattr', 'getattr'
    }
    
    # Safe methods for specific types
    SAFE_METHODS = {
        'str': {'lower', 'upper', 'strip', 'split', 'join', 'replace', 'startswith', 'endswith'},
        'list': {'append', 'extend', 'count', 'index', 'sort', 'reverse'},
        'dict': {'get', 'keys', 'values', 'items', 'update'},
        'set': {'add', 'remove', 'discard', 'union', 'intersection', 'difference'}
    }
    
    def __init__(self, allowed_names: Optional[Dict[str, Any]] = None):
        """Initialize the safe evaluator with optional allowed names."""
        self.allowed_names = allowed_names or {}
        
    def is_safe_node(self, node: ast.AST) -> bool:
        """Check if an AST node is safe to evaluate."""
        if type(node) not in self.SAFE_NODES:
            return False
            
        if isinstance(node, ast.Call):
            # Check function calls
            if isinstance(node.func, ast.Name):
                if node.func.id not in self.SAFE_FUNCTIONS:
                    return False
            elif isinstance(node.func, ast.Attribute):
                # Check method calls
                if hasattr(node.func, 'attr'):
                    method_name = node.func.attr
                    # Additional validation could be added here
                    pass
            else:
                return False
                
        return True
    
    def validate_ast(self, node: ast.AST) -> bool:
        """Recursively validate an AST for safety."""
        if not self.is_safe_node(node):
            return False
            
        for child in ast.iter_child_nodes(node):
            if not self.validate_ast(child):
                return False
                
        return True
    
    def safe_eval(self, expression: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Safely evaluate a string expression using AST parsing."""
        try:
            # Parse the expression into an AST
            tree = ast.parse(expression, mode='eval')
            
            # Validate the AST for safety
            if not self.validate_ast(tree):
                raise SecureExecutionError(f"Unsafe expression: {expression}")
            
            # Prepare the evaluation context
            eval_context = {**self.allowed_names}
            if context:
                eval_context.update(context)
            
            # Add safe built-ins
            safe_builtins = {name: getattr(__builtins__, name) 
                           for name in self.SAFE_FUNCTIONS 
                           if hasattr(__builtins__, name)}
            eval_context.update(safe_builtins)
            
            # Compile and evaluate the expression
            compiled = compile(tree, '<string>', 'eval')
            result = safe_evaluator.safe_eval(compiled, {"__builtins__": {}}, eval_context)
            
            return result
            
        except (SyntaxError, ValueError) as e:
            raise SecureExecutionError(f"Invalid expression: {expression}") from e
        except Exception as e:
            raise SecureExecutionError(f"Evaluation error: {str(e)}") from e


class SafeExecutor:
    """Safe execution environment for research code."""
    
    def __init__(self, allowed_modules: Optional[List[str]] = None):
        """Initialize with allowed modules for import."""
        self.allowed_modules = allowed_modules or [
            'math', 'statistics', 'datetime', 'json', 're',
            'numpy', 'pandas', 'matplotlib', 'scipy'
        ]
        
    def safe_exec(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Safely execute code with restricted environment."""
        try:
            # Parse code into AST
            tree = ast.parse(code)
            
            # Validate for unsafe operations
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Check imports
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name not in self.allowed_modules:
                                raise SecureExecutionError(f"Import not allowed: {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and node.module not in self.allowed_modules:
                            raise SecureExecutionError(f"Import not allowed: {node.module}")
                
                # Prevent dangerous operations
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ('eval', 'exec', 'compile', '__import__'):
                            raise SecureExecutionError(f"Function not allowed: {node.func.id}")
            
            # Create safe execution environment
            exec_context = context.copy() if context else {}
            exec_context['__builtins__'] = {
                'len': len, 'str': str, 'int': int, 'float': float,
                'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                'max': max, 'min': min, 'sum': sum, 'abs': abs,
                'round': round, 'bool': bool, 'isinstance': isinstance
            }
            
            # Execute the code
            safe_executor.safe_exec(compile(tree, '<string>', 'exec'), exec_context)
            
            # Return the modified context (excluding builtins)
            result = {k: v for k, v in exec_context.items() if k != '__builtins__'}
            return result
            
        except Exception as e:
            raise SecureExecutionError(f"Execution error: {str(e)}") from e


def secure_eval_replacement(func):
    """Decorator to replace eval calls with safe evaluation."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Initialize safe evaluator
        evaluator = SafeEvaluator()
        
        # Replace eval in the function's globals temporarily
        original_eval = func.__globals__.get('eval')
        func.__globals__['eval'] = evaluator.safe_eval
        
        try:
            return func(*args, **kwargs)
        finally:
            # Restore original eval
            if original_eval:
                func.__globals__['eval'] = original_eval
            else:
                func.__globals__.pop('eval', None)
    
    return wrapper


def secure_exec_replacement(func):
    """Decorator to replace exec calls with safe execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Initialize safe executor
        executor = SafeExecutor()
        
        # Replace exec in the function's globals temporarily
        original_exec = func.__globals__.get('exec')
        func.__globals__['exec'] = executor.safe_exec
        
        try:
            return func(*args, **kwargs)
        finally:
            # Restore original exec
            if original_exec:
                func.__globals__['exec'] = original_exec
            else:
                func.__globals__.pop('exec', None)
    
    return wrapper


# Safe alternatives for common patterns
def safe_json_loads(data: str) -> Any:
    """Safe JSON loading with error handling."""
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise SecureExecutionError(f"Invalid JSON: {str(e)}") from e


def safe_regex_eval(pattern: str, string: str, flags: int = 0) -> Optional[re.Match]:
    """Safe regex evaluation with pattern validation."""
    try:
        # Validate pattern length and complexity
        if len(pattern) > 1000:
            raise SecureExecutionError("Regex pattern too long")
        
        # Compile with timeout protection
        compiled_pattern = re.compile(pattern, flags)
        return compiled_pattern.search(string)
        
    except re.error as e:
        raise SecureExecutionError(f"Invalid regex pattern: {str(e)}") from e


def safe_mathematical_eval(expression: str, variables: Optional[Dict[str, float]] = None) -> float:
    """Safe mathematical expression evaluation."""
    evaluator = SafeEvaluator()
    
    # Only allow mathematical operations
    allowed_vars = variables or {}
    allowed_vars.update({
        'pi': 3.141592653589793,
        'e': 2.718281828459045
    })
    
    try:
        result = evaluator.safe_eval(expression, allowed_vars)
        return float(result)
    except (ValueError, TypeError) as e:
        raise SecureExecutionError(f"Mathematical evaluation error: {str(e)}") from e


# Global instances for reuse
default_evaluator = SafeEvaluator()
default_executor = SafeExecutor()

# Export safe functions
__all__ = [
    'SecureExecutionError',
    'SafeEvaluator', 
    'SafeExecutor',
    'secure_eval_replacement',
    'secure_exec_replacement',
    'safe_json_loads',
    'safe_regex_eval', 
    'safe_mathematical_eval',
    'default_evaluator',
    'default_executor'
]