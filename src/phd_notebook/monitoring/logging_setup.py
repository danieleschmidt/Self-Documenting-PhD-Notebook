"""
Comprehensive logging setup for the PhD notebook system.
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add custom fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class ContextFilter(logging.Filter):
    """Add context information to log records."""
    
    def __init__(self, notebook_name: str = "phd_notebook"):
        super().__init__()
        self.notebook_name = notebook_name
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.notebook_name = self.notebook_name
        record.session_id = getattr(self, 'session_id', 'unknown')
        return True


def setup_logging(
    log_dir: Path = None,
    level: str = "INFO",
    notebook_name: str = "phd_notebook",
    structured_logging: bool = True,
    console_output: bool = True
) -> Dict[str, logging.Logger]:
    """
    Setup comprehensive logging for the PhD notebook system.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        notebook_name: Name of notebook for context
        structured_logging: Use JSON formatting
        console_output: Enable console output
    
    Returns:
        Dictionary of configured loggers
    """
    
    # Create log directory
    if log_dir is None:
        log_dir = Path.home() / ".phd_notebook" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Setup formatters
    if structured_logging:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add context filter
    context_filter = ContextFilter(notebook_name)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(context_filter)
        root_logger.addHandler(console_handler)
    
    # File handlers
    handlers = {}
    
    # Main application log (rotating)
    main_handler = logging.handlers.RotatingFileHandler(
        log_dir / "phd_notebook.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    main_handler.setLevel(logging.DEBUG)
    main_handler.setFormatter(formatter)
    main_handler.addFilter(context_filter)
    root_logger.addHandler(main_handler)
    handlers['main'] = main_handler
    
    # Error log (errors and critical only)
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    error_handler.addFilter(context_filter)
    root_logger.addHandler(error_handler)
    handlers['error'] = error_handler
    
    # Performance log (for slow operations)
    perf_logger = logging.getLogger("performance")
    perf_handler = logging.handlers.RotatingFileHandler(
        log_dir / "performance.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(formatter)
    perf_handler.addFilter(context_filter)
    perf_logger.addHandler(perf_handler)
    handlers['performance'] = perf_handler
    
    # Audit log (for tracking user actions)
    audit_logger = logging.getLogger("audit")
    audit_handler = logging.handlers.RotatingFileHandler(
        log_dir / "audit.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10  # Keep more audit logs
    )
    audit_handler.setLevel(logging.INFO)
    audit_handler.setFormatter(formatter)
    audit_handler.addFilter(context_filter)
    audit_logger.addHandler(audit_handler)
    handlers['audit'] = audit_handler
    
    # AI/Agent activity log
    ai_logger = logging.getLogger("ai_activity")
    ai_handler = logging.handlers.RotatingFileHandler(
        log_dir / "ai_activity.log",
        maxBytes=20 * 1024 * 1024,  # 20MB (AI logs can be verbose)
        backupCount=5
    )
    ai_handler.setLevel(logging.INFO)
    ai_handler.setFormatter(formatter)
    ai_handler.addFilter(context_filter)
    ai_logger.addHandler(ai_handler)
    handlers['ai'] = ai_handler
    
    # Workflow execution log
    workflow_logger = logging.getLogger("workflows")
    workflow_handler = logging.handlers.RotatingFileHandler(
        log_dir / "workflows.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3
    )
    workflow_handler.setLevel(logging.INFO)
    workflow_handler.setFormatter(formatter)
    workflow_handler.addFilter(context_filter)
    workflow_logger.addHandler(workflow_handler)
    handlers['workflow'] = workflow_handler
    
    # Create logger instances
    loggers = {
        'main': logging.getLogger(),
        'performance': perf_logger,
        'audit': audit_logger,
        'ai': ai_logger,
        'workflow': workflow_logger
    }
    
    # Log the setup completion
    main_logger = loggers['main']
    main_logger.info(f"Logging setup completed for {notebook_name}")
    main_logger.info(f"Log directory: {log_dir}")
    main_logger.info(f"Log level: {level}")
    main_logger.info(f"Structured logging: {structured_logging}")
    
    return loggers


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance with consistent configuration."""
    if name is None:
        return logging.getLogger()
    
    logger = logging.getLogger(name)
    
    # Ensure logger has proper level if not set
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    
    return logger


class LogContext:
    """Context manager for adding context to logs."""
    
    def __init__(self, **context):
        self.context = context
        self.original_factory = logging.getLogRecordFactory()
    
    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.original_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.original_factory)


def log_function_call(logger: logging.Logger = None, level: str = "DEBUG"):
    """Decorator to log function calls."""
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        def wrapper(*args, **kwargs):
            log_level = getattr(logging, level.upper())
            
            # Log function entry
            logger.log(log_level, f"Calling {func.__name__}", extra={
                'function': func.__name__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            })
            
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                duration = (datetime.now() - start_time).total_seconds()
                logger.log(log_level, f"Completed {func.__name__}", extra={
                    'function': func.__name__,
                    'duration_seconds': duration,
                    'status': 'success'
                })
                
                return result
                
            except Exception as e:
                # Log error
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"Error in {func.__name__}: {e}", extra={
                    'function': func.__name__,
                    'duration_seconds': duration,
                    'status': 'error',
                    'error_type': type(e).__name__
                })
                raise
        
        return wrapper
    return decorator


# Performance logging utilities
def log_performance_metrics(metrics: Dict[str, Any], operation: str = "unknown"):
    """Log performance metrics."""
    perf_logger = get_logger("performance")
    perf_logger.info(f"Performance metrics for {operation}", extra={
        'operation': operation,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    })


def log_user_action(action: str, details: Dict[str, Any] = None, user_id: str = "unknown"):
    """Log user actions for audit trail."""
    audit_logger = get_logger("audit")
    audit_logger.info(f"User action: {action}", extra={
        'action': action,
        'user_id': user_id,
        'details': details or {},
        'timestamp': datetime.now().isoformat()
    })


def log_ai_interaction(
    agent_name: str,
    operation: str,
    tokens_used: int = 0,
    cost: float = 0.0,
    success: bool = True,
    details: Dict[str, Any] = None
):
    """Log AI/Agent interactions."""
    ai_logger = get_logger("ai_activity")
    ai_logger.info(f"AI interaction: {agent_name} - {operation}", extra={
        'agent_name': agent_name,
        'operation': operation,
        'tokens_used': tokens_used,
        'cost': cost,
        'success': success,
        'details': details or {},
        'timestamp': datetime.now().isoformat()
    })