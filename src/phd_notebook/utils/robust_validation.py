"""
Robust Validation and Error Handling
====================================

Comprehensive validation framework ensuring data integrity, type safety,
and graceful error handling across all research operations.

Features:
- Schema-based validation
- Type checking and coercion
- Custom validation rules
- Error recovery mechanisms
- Data sanitization
- Input normalization
"""

import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"          # All validations must pass
    MODERATE = "moderate"      # Critical validations must pass, warnings allowed
    LENIENT = "lenient"        # Best-effort validation, coercion attempted


class ErrorSeverity(Enum):
    """Error severity classification."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationError:
    """Individual validation error."""
    field: str
    message: str
    severity: ErrorSeverity
    code: str
    value: Any = None
    expected_type: Optional[Type] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'field': self.field,
            'message': self.message,
            'severity': self.severity.value,
            'code': self.code,
            'value': str(self.value) if self.value is not None else None,
            'expected_type': self.expected_type.__name__ if self.expected_type else None
        }


@dataclass
class ValidationResult:
    """Result of validation operation."""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    sanitized_data: Optional[Dict[str, Any]] = None
    validation_time: Optional[float] = None
    
    def add_error(self, field: str, message: str, code: str, value: Any = None):
        """Add validation error."""
        error = ValidationError(field, message, ErrorSeverity.ERROR, code, value)
        self.errors.append(error)
        self.valid = False
    
    def add_warning(self, field: str, message: str, code: str, value: Any = None):
        """Add validation warning."""
        warning = ValidationError(field, message, ErrorSeverity.WARNING, code, value)
        self.warnings.append(warning)
    
    def has_critical_errors(self) -> bool:
        """Check if there are critical errors."""
        return any(error.severity == ErrorSeverity.CRITICAL for error in self.errors)
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts by severity."""
        summary = {severity.value: 0 for severity in ErrorSeverity}
        
        for error in self.errors + self.warnings:
            summary[error.severity.value] += 1
        
        return summary


class BaseValidator(ABC):
    """Abstract base class for validators."""
    
    @abstractmethod
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate a value and return result."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the validation schema."""
        pass


class TypeValidator(BaseValidator):
    """Validates and coerces data types."""
    
    def __init__(self, expected_type: Type, coerce: bool = True):
        self.expected_type = expected_type
        self.coerce = coerce
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate type and optionally coerce."""
        result = ValidationResult(valid=True)
        
        if value is None:
            result.add_error("type", f"Value cannot be None", "NULL_VALUE")
            return result
        
        if isinstance(value, self.expected_type):
            result.sanitized_data = {'value': value}
            return result
        
        # Attempt coercion if enabled
        if self.coerce:
            try:
                coerced_value = self._coerce_value(value)
                result.sanitized_data = {'value': coerced_value}
                result.add_warning("type", f"Value coerced from {type(value).__name__} to {self.expected_type.__name__}", "TYPE_COERCION")
                return result
            except (ValueError, TypeError) as e:
                result.add_error("type", f"Cannot coerce {type(value).__name__} to {self.expected_type.__name__}: {e}", "COERCION_FAILED", value)
        else:
            result.add_error("type", f"Expected {self.expected_type.__name__}, got {type(value).__name__}", "TYPE_MISMATCH", value)
        
        return result
    
    def _coerce_value(self, value: Any) -> Any:
        """Attempt to coerce value to expected type."""
        if self.expected_type == str:
            return str(value)
        elif self.expected_type == int:
            if isinstance(value, float) and value.is_integer():
                return int(value)
            return int(value)
        elif self.expected_type == float:
            return float(value)
        elif self.expected_type == bool:
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        elif self.expected_type == list and not isinstance(value, list):
            return [value]
        else:
            return self.expected_type(value)
    
    def get_schema(self) -> Dict[str, Any]:
        """Get type validation schema."""
        return {
            'type': 'type_validator',
            'expected_type': self.expected_type.__name__,
            'coerce': self.coerce
        }


class RangeValidator(BaseValidator):
    """Validates numeric ranges."""
    
    def __init__(self, min_value: Optional[Union[int, float]] = None,
                 max_value: Optional[Union[int, float]] = None,
                 inclusive: bool = True):
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate numeric range."""
        result = ValidationResult(valid=True)
        
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            result.add_error("range", f"Value must be numeric, got {type(value).__name__}", "NOT_NUMERIC", value)
            return result
        
        # Check minimum
        if self.min_value is not None:
            if self.inclusive and numeric_value < self.min_value:
                result.add_error("range", f"Value {numeric_value} is below minimum {self.min_value}", "BELOW_MIN", value)
            elif not self.inclusive and numeric_value <= self.min_value:
                result.add_error("range", f"Value {numeric_value} must be greater than {self.min_value}", "BELOW_MIN_EXCLUSIVE", value)
        
        # Check maximum
        if self.max_value is not None:
            if self.inclusive and numeric_value > self.max_value:
                result.add_error("range", f"Value {numeric_value} exceeds maximum {self.max_value}", "ABOVE_MAX", value)
            elif not self.inclusive and numeric_value >= self.max_value:
                result.add_error("range", f"Value {numeric_value} must be less than {self.max_value}", "ABOVE_MAX_EXCLUSIVE", value)
        
        if result.valid:
            result.sanitized_data = {'value': numeric_value}
        
        return result
    
    def get_schema(self) -> Dict[str, Any]:
        """Get range validation schema."""
        return {
            'type': 'range_validator',
            'min_value': self.min_value,
            'max_value': self.max_value,
            'inclusive': self.inclusive
        }


class RegexValidator(BaseValidator):
    """Validates strings against regular expressions."""
    
    def __init__(self, pattern: str, flags: int = 0, message: Optional[str] = None):
        self.pattern = pattern
        self.flags = flags
        self.compiled_pattern = re.compile(pattern, flags)
        self.message = message or f"Value does not match pattern: {pattern}"
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate against regex pattern."""
        result = ValidationResult(valid=True)
        
        # Convert to string
        str_value = str(value) if value is not None else ""
        
        if not self.compiled_pattern.match(str_value):
            result.add_error("regex", self.message, "PATTERN_MISMATCH", value)
        else:
            result.sanitized_data = {'value': str_value}
        
        return result
    
    def get_schema(self) -> Dict[str, Any]:
        """Get regex validation schema."""
        return {
            'type': 'regex_validator',
            'pattern': self.pattern,
            'flags': self.flags,
            'message': self.message
        }


class LengthValidator(BaseValidator):
    """Validates length of strings, lists, or other sized objects."""
    
    def __init__(self, min_length: Optional[int] = None,
                 max_length: Optional[int] = None):
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate length constraints."""
        result = ValidationResult(valid=True)
        
        try:
            length = len(value)
        except TypeError:
            result.add_error("length", f"Value of type {type(value).__name__} has no length", "NO_LENGTH", value)
            return result
        
        # Check minimum length
        if self.min_length is not None and length < self.min_length:
            result.add_error("length", f"Length {length} is below minimum {self.min_length}", "TOO_SHORT", value)
        
        # Check maximum length
        if self.max_length is not None and length > self.max_length:
            result.add_error("length", f"Length {length} exceeds maximum {self.max_length}", "TOO_LONG", value)
        
        if result.valid:
            result.sanitized_data = {'value': value}
        
        return result
    
    def get_schema(self) -> Dict[str, Any]:
        """Get length validation schema."""
        return {
            'type': 'length_validator',
            'min_length': self.min_length,
            'max_length': self.max_length
        }


class ChoiceValidator(BaseValidator):
    """Validates that value is from a set of allowed choices."""
    
    def __init__(self, choices: List[Any], case_sensitive: bool = True):
        self.choices = choices
        self.case_sensitive = case_sensitive
        
        if not case_sensitive:
            self.normalized_choices = [str(choice).lower() for choice in choices]
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate choice selection."""
        result = ValidationResult(valid=True)
        
        if self.case_sensitive:
            if value not in self.choices:
                result.add_error("choice", f"Value '{value}' not in allowed choices: {self.choices}", "INVALID_CHOICE", value)
        else:
            str_value = str(value).lower()
            if str_value not in self.normalized_choices:
                result.add_error("choice", f"Value '{value}' not in allowed choices: {self.choices}", "INVALID_CHOICE", value)
            else:
                # Find the original choice
                for i, normalized in enumerate(self.normalized_choices):
                    if str_value == normalized:
                        result.sanitized_data = {'value': self.choices[i]}
                        break
        
        if result.valid and result.sanitized_data is None:
            result.sanitized_data = {'value': value}
        
        return result
    
    def get_schema(self) -> Dict[str, Any]:
        """Get choice validation schema."""
        return {
            'type': 'choice_validator',
            'choices': self.choices,
            'case_sensitive': self.case_sensitive
        }


class DateTimeValidator(BaseValidator):
    """Validates and parses datetime values."""
    
    def __init__(self, formats: List[str] = None, timezone_aware: bool = False):
        self.formats = formats or [
            '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%fZ'
        ]
        self.timezone_aware = timezone_aware
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate and parse datetime."""
        result = ValidationResult(valid=True)
        
        if isinstance(value, datetime):
            result.sanitized_data = {'value': value}
            return result
        
        str_value = str(value)
        parsed_datetime = None
        
        # Try each format
        for fmt in self.formats:
            try:
                parsed_datetime = datetime.strptime(str_value, fmt)
                break
            except ValueError:
                continue
        
        if parsed_datetime is None:
            result.add_error("datetime", f"Cannot parse datetime '{str_value}' with any of the formats: {self.formats}", "PARSE_FAILED", value)
        else:
            result.sanitized_data = {'value': parsed_datetime}
        
        return result
    
    def get_schema(self) -> Dict[str, Any]:
        """Get datetime validation schema."""
        return {
            'type': 'datetime_validator',
            'formats': self.formats,
            'timezone_aware': self.timezone_aware
        }


class CompositeValidator(BaseValidator):
    """Combines multiple validators with logical operations."""
    
    def __init__(self, validators: List[BaseValidator], operation: str = "AND"):
        self.validators = validators
        self.operation = operation.upper()
        
        if self.operation not in ["AND", "OR"]:
            raise ValueError("Operation must be 'AND' or 'OR'")
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate using composite logic."""
        results = [validator.validate(value, context) for validator in self.validators]
        
        if self.operation == "AND":
            return self._combine_and(results)
        else:
            return self._combine_or(results)
    
    def _combine_and(self, results: List[ValidationResult]) -> ValidationResult:
        """Combine results with AND logic (all must be valid)."""
        combined = ValidationResult(valid=True)
        
        for result in results:
            if not result.valid:
                combined.valid = False
            
            combined.errors.extend(result.errors)
            combined.warnings.extend(result.warnings)
            
            # Use last valid sanitized data
            if result.sanitized_data is not None:
                combined.sanitized_data = result.sanitized_data
        
        return combined
    
    def _combine_or(self, results: List[ValidationResult]) -> ValidationResult:
        """Combine results with OR logic (at least one must be valid)."""
        combined = ValidationResult(valid=False)
        
        # If any result is valid, the whole thing is valid
        for result in results:
            if result.valid:
                combined.valid = True
                combined.sanitized_data = result.sanitized_data
                combined.warnings.extend(result.warnings)
                return combined
        
        # All failed - combine all errors
        for result in results:
            combined.errors.extend(result.errors)
            combined.warnings.extend(result.warnings)
        
        return combined
    
    def get_schema(self) -> Dict[str, Any]:
        """Get composite validation schema."""
        return {
            'type': 'composite_validator',
            'operation': self.operation,
            'validators': [validator.get_schema() for validator in self.validators]
        }


class SchemaValidator:
    """Validates complex nested data structures against schemas."""
    
    def __init__(self, schema: Dict[str, Any], level: ValidationLevel = ValidationLevel.MODERATE):
        self.schema = schema
        self.level = level
        self.validators: Dict[str, BaseValidator] = {}
        self._build_validators()
    
    def _build_validators(self):
        """Build validators from schema definition."""
        for field, field_schema in self.schema.get('fields', {}).items():
            self.validators[field] = self._create_validator(field_schema)
    
    def _create_validator(self, field_schema: Dict[str, Any]) -> BaseValidator:
        """Create validator from schema definition."""
        validator_type = field_schema.get('type')
        
        if validator_type == 'type':
            type_name = field_schema.get('expected_type', 'str')
            type_map = {
                'str': str, 'string': str,
                'int': int, 'integer': int,
                'float': float, 'number': float,
                'bool': bool, 'boolean': bool,
                'list': list, 'array': list,
                'dict': dict, 'object': dict
            }
            expected_type = type_map.get(type_name, str)
            return TypeValidator(expected_type, field_schema.get('coerce', True))
        
        elif validator_type == 'range':
            return RangeValidator(
                field_schema.get('min_value'),
                field_schema.get('max_value'),
                field_schema.get('inclusive', True)
            )
        
        elif validator_type == 'regex':
            return RegexValidator(
                field_schema['pattern'],
                field_schema.get('flags', 0),
                field_schema.get('message')
            )
        
        elif validator_type == 'length':
            return LengthValidator(
                field_schema.get('min_length'),
                field_schema.get('max_length')
            )
        
        elif validator_type == 'choice':
            return ChoiceValidator(
                field_schema['choices'],
                field_schema.get('case_sensitive', True)
            )
        
        elif validator_type == 'datetime':
            return DateTimeValidator(
                field_schema.get('formats'),
                field_schema.get('timezone_aware', False)
            )
        
        elif validator_type == 'composite':
            sub_validators = [
                self._create_validator(sub_schema) 
                for sub_schema in field_schema.get('validators', [])
            ]
            return CompositeValidator(sub_validators, field_schema.get('operation', 'AND'))
        
        else:
            # Default to type validator
            return TypeValidator(str)
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data against schema."""
        start_time = datetime.now()
        result = ValidationResult(valid=True, sanitized_data={})
        
        # Check required fields
        required_fields = self.schema.get('required', [])
        for field in required_fields:
            if field not in data:
                result.add_error(field, f"Required field '{field}' is missing", "MISSING_REQUIRED")
        
        # Validate each field
        for field, value in data.items():
            if field in self.validators:
                field_result = self.validators[field].validate(value)
                
                if not field_result.valid:
                    result.valid = False
                    result.errors.extend(field_result.errors)
                
                result.warnings.extend(field_result.warnings)
                
                # Add sanitized data
                if field_result.sanitized_data:
                    result.sanitized_data[field] = field_result.sanitized_data.get('value', value)
                else:
                    result.sanitized_data[field] = value
            else:
                # Unknown field
                if self.level == ValidationLevel.STRICT:
                    result.add_error(field, f"Unknown field '{field}' not allowed in strict mode", "UNKNOWN_FIELD")
                elif self.level == ValidationLevel.MODERATE:
                    result.add_warning(field, f"Unknown field '{field}' will be ignored", "UNKNOWN_FIELD")
                # LENIENT mode - just include the field
                result.sanitized_data[field] = value
        
        result.validation_time = (datetime.now() - start_time).total_seconds()
        
        return result


class ErrorRecoveryManager:
    """Manages error recovery and fallback strategies."""
    
    def __init__(self):
        self.recovery_strategies: Dict[str, Callable] = {}
        self.fallback_values: Dict[str, Any] = {}
    
    def register_recovery_strategy(self, error_code: str, strategy: Callable[[Any, ValidationError], Any]):
        """Register a recovery strategy for specific error codes."""
        self.recovery_strategies[error_code] = strategy
        logger.info(f"Registered recovery strategy for error code: {error_code}")
    
    def set_fallback_value(self, field: str, value: Any):
        """Set fallback value for a field."""
        self.fallback_values[field] = value
    
    def attempt_recovery(self, data: Dict[str, Any], validation_result: ValidationResult) -> Dict[str, Any]:
        """Attempt to recover from validation errors."""
        recovered_data = data.copy()
        recovery_log = []
        
        for error in validation_result.errors:
            recovery_attempted = False
            
            # Try specific recovery strategy
            if error.code in self.recovery_strategies:
                try:
                    strategy = self.recovery_strategies[error.code]
                    recovered_value = strategy(error.value, error)
                    recovered_data[error.field] = recovered_value
                    recovery_log.append(f"Recovered {error.field} using strategy for {error.code}")
                    recovery_attempted = True
                except Exception as e:
                    logger.warning(f"Recovery strategy failed for {error.code}: {e}")
            
            # Try fallback value
            if not recovery_attempted and error.field in self.fallback_values:
                recovered_data[error.field] = self.fallback_values[error.field]
                recovery_log.append(f"Used fallback value for {error.field}")
                recovery_attempted = True
            
            # Generic recovery attempts
            if not recovery_attempted:
                recovered_value = self._generic_recovery(error.value, error)
                if recovered_value is not None:
                    recovered_data[error.field] = recovered_value
                    recovery_log.append(f"Applied generic recovery to {error.field}")
        
        if recovery_log:
            logger.info(f"Recovery attempts: {recovery_log}")
        
        return recovered_data
    
    def _generic_recovery(self, value: Any, error: ValidationError) -> Any:
        """Apply generic recovery strategies."""
        if error.code == "TYPE_MISMATCH":
            # Try basic type conversion
            if error.expected_type == str:
                return str(value) if value is not None else ""
            elif error.expected_type in [int, float]:
                try:
                    return error.expected_type(value)
                except (ValueError, TypeError):
                    return 0
            elif error.expected_type == bool:
                return bool(value)
            elif error.expected_type == list:
                return [value] if not isinstance(value, list) else value
        
        elif error.code in ["TOO_SHORT", "TOO_LONG"]:
            # String truncation or padding
            if isinstance(value, str):
                if error.code == "TOO_LONG":
                    # Find max_length from error message
                    match = re.search(r'maximum (\d+)', error.message)
                    if match:
                        max_length = int(match.group(1))
                        return value[:max_length]
                elif error.code == "TOO_SHORT":
                    # Find min_length and pad
                    match = re.search(r'minimum (\d+)', error.message)
                    if match:
                        min_length = int(match.group(1))
                        return value.ljust(min_length)
        
        return None


class RobustValidationFramework:
    """Main framework coordinating all validation components."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.error_recovery = ErrorRecoveryManager()
        self.schemas: Dict[str, SchemaValidator] = {}
        
        self._setup_default_recovery_strategies()
    
    def _setup_default_recovery_strategies(self):
        """Setup default recovery strategies."""
        
        def coercion_recovery(value: Any, error: ValidationError) -> Any:
            """Generic type coercion recovery."""
            if error.expected_type == str:
                return str(value) if value is not None else ""
            elif error.expected_type == int:
                try:
                    return int(float(str(value)))
                except (ValueError, TypeError):
                    return 0
            elif error.expected_type == float:
                try:
                    return float(str(value))
                except (ValueError, TypeError):
                    return 0.0
            return value
        
        def date_recovery(value: Any, error: ValidationError) -> Any:
            """Date parsing recovery."""
            str_val = str(value)
            # Try ISO format as fallback
            try:
                return datetime.fromisoformat(str_val.replace('Z', '+00:00'))
            except ValueError:
                return datetime.now()
        
        self.error_recovery.register_recovery_strategy("TYPE_MISMATCH", coercion_recovery)
        self.error_recovery.register_recovery_strategy("COERCION_FAILED", coercion_recovery)
        self.error_recovery.register_recovery_strategy("PARSE_FAILED", date_recovery)
    
    def register_schema(self, name: str, schema: Dict[str, Any]):
        """Register a validation schema."""
        self.schemas[name] = SchemaValidator(schema, self.validation_level)
        logger.info(f"Registered validation schema: {name}")
    
    def validate_data(self, schema_name: str, data: Dict[str, Any], 
                     attempt_recovery: bool = True) -> ValidationResult:
        """Validate data against a registered schema."""
        if schema_name not in self.schemas:
            raise ValueError(f"Schema '{schema_name}' not found")
        
        validator = self.schemas[schema_name]
        result = validator.validate(data)
        
        # Attempt recovery if validation failed and recovery is enabled
        if not result.valid and attempt_recovery:
            logger.info(f"Attempting error recovery for {len(result.errors)} errors")
            
            recovered_data = self.error_recovery.attempt_recovery(data, result)
            
            # Re-validate recovered data
            recovery_result = validator.validate(recovered_data)
            
            if recovery_result.valid or len(recovery_result.errors) < len(result.errors):
                logger.info("Error recovery successful")
                recovery_result.sanitized_data = recovery_result.sanitized_data or recovered_data
                return recovery_result
            else:
                logger.warning("Error recovery did not improve validation results")
        
        return result
    
    def sanitize_data(self, data: Dict[str, Any], schema_name: Optional[str] = None) -> Dict[str, Any]:
        """Sanitize data with optional schema validation."""
        sanitized = data.copy()
        
        # Basic sanitization
        sanitized = self._basic_sanitization(sanitized)
        
        # Schema-based sanitization if schema provided
        if schema_name and schema_name in self.schemas:
            result = self.validate_data(schema_name, sanitized, attempt_recovery=True)
            if result.sanitized_data:
                sanitized = result.sanitized_data
        
        return sanitized
    
    def _basic_sanitization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply basic sanitization rules."""
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize keys (remove special characters, make safe)
            safe_key = re.sub(r'[^\w\-_]', '_', str(key))
            
            # Sanitize values based on type
            if isinstance(value, str):
                # Remove null bytes and control characters
                sanitized_value = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)
                # Trim whitespace
                sanitized_value = sanitized_value.strip()
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized_value = self._basic_sanitization(value)
            elif isinstance(value, list):
                # Sanitize list elements
                sanitized_value = [
                    self._basic_sanitization(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized_value = value
            
            sanitized[safe_key] = sanitized_value
        
        return sanitized


# Pre-defined common schemas
RESEARCH_NOTE_SCHEMA = {
    'fields': {
        'title': {
            'type': 'composite',
            'operation': 'AND',
            'validators': [
                {'type': 'type', 'expected_type': 'str'},
                {'type': 'length', 'min_length': 1, 'max_length': 200}
            ]
        },
        'content': {
            'type': 'type',
            'expected_type': 'str'
        },
        'tags': {
            'type': 'type',
            'expected_type': 'list'
        },
        'created_at': {
            'type': 'datetime',
            'formats': ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f']
        },
        'priority': {
            'type': 'range',
            'min_value': 1,
            'max_value': 5
        },
        'status': {
            'type': 'choice',
            'choices': ['draft', 'in_progress', 'completed', 'archived'],
            'case_sensitive': False
        }
    },
    'required': ['title', 'content']
}

EXPERIMENT_SCHEMA = {
    'fields': {
        'hypothesis': {
            'type': 'composite',
            'operation': 'AND',
            'validators': [
                {'type': 'type', 'expected_type': 'str'},
                {'type': 'length', 'min_length': 10}
            ]
        },
        'methodology': {
            'type': 'type',
            'expected_type': 'str'
        },
        'sample_size': {
            'type': 'composite',
            'operation': 'AND',
            'validators': [
                {'type': 'type', 'expected_type': 'int'},
                {'type': 'range', 'min_value': 1, 'max_value': 100000}
            ]
        },
        'duration_weeks': {
            'type': 'composite',
            'operation': 'AND',
            'validators': [
                {'type': 'type', 'expected_type': 'int'},
                {'type': 'range', 'min_value': 1, 'max_value': 520}  # Max 10 years
            ]
        },
        'ethical_approval': {
            'type': 'type',
            'expected_type': 'bool'
        },
        'start_date': {
            'type': 'datetime'
        }
    },
    'required': ['hypothesis', 'methodology', 'sample_size']
}


# Example usage and testing
if __name__ == "__main__":
    # Create validation framework
    framework = RobustValidationFramework(ValidationLevel.MODERATE)
    
    # Register schemas
    framework.register_schema('research_note', RESEARCH_NOTE_SCHEMA)
    framework.register_schema('experiment', EXPERIMENT_SCHEMA)
    
    # Test data with various issues
    test_note = {
        'title': 'Test Note',
        'content': 'This is a test note for validation',
        'tags': ['test', 'validation'],
        'created_at': '2023-12-01T10:30:00',
        'priority': '3',  # String instead of int
        'status': 'DRAFT',  # Different case
        'unknown_field': 'should be handled based on validation level'
    }
    
    print("🔍 Testing Robust Validation Framework")
    
    # Validate note
    result = framework.validate_data('research_note', test_note)
    
    print(f"Validation Result: {'PASS' if result.valid else 'FAIL'}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    
    if result.errors:
        for error in result.errors[:3]:  # Show first 3 errors
            print(f"  Error: {error.field} - {error.message}")
    
    if result.warnings:
        for warning in result.warnings[:3]:  # Show first 3 warnings
            print(f"  Warning: {warning.field} - {warning.message}")
    
    # Show sanitized data
    if result.sanitized_data:
        print(f"Sanitized data keys: {list(result.sanitized_data.keys())}")
    
    print("✅ Validation framework test completed")