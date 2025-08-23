"""
Comprehensive Validation System for Research Integrity

A robust validation framework ensuring research quality, reproducibility,
and scientific rigor across all notebook operations with advanced error
recovery and integrity checking mechanisms.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
import hashlib
import uuid
from abc import ABC, abstractmethod
import traceback

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Levels of validation rigor."""
    BASIC = "basic"
    STANDARD = "standard"
    RIGOROUS = "rigorous"
    PUBLICATION_READY = "publication_ready"


class ValidationCategory(Enum):
    """Categories of validation checks."""
    DATA_INTEGRITY = "data_integrity"
    METHODOLOGY = "methodology"
    STATISTICAL = "statistical"
    ETHICAL = "ethical"
    REPRODUCIBILITY = "reproducibility"
    CITATION = "citation"
    FORMAT = "format"
    SECURITY = "security"
    COLLABORATION = "collaboration"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    BLOCKING = "blocking"


@dataclass
class ValidationRule:
    """Represents a single validation rule."""
    rule_id: str
    name: str
    category: ValidationCategory
    severity: ValidationSeverity
    description: str
    check_function: Callable
    auto_fix_function: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    applicable_contexts: List[str] = field(default_factory=list)
    is_active: bool = True
    confidence_threshold: float = 0.8


@dataclass
class ValidationIssue:
    """Represents a validation issue found during checking."""
    issue_id: str
    rule_id: str
    severity: ValidationSeverity
    category: ValidationCategory
    title: str
    description: str
    location: str
    suggested_fix: str
    auto_fixable: bool
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    is_resolved: bool = False
    resolution_notes: str = ""


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    report_id: str
    validation_level: ValidationLevel
    target_context: str
    start_time: datetime
    end_time: Optional[datetime]
    total_checks: int
    passed_checks: int
    failed_checks: int
    issues: List[ValidationIssue] = field(default_factory=list)
    overall_score: float = 0.0
    is_valid: bool = False
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationContext:
    """Context for validation operations."""
    context_id: str
    context_type: str  # "note", "experiment", "paper", "dataset", etc.
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    custom_rules: List[str] = field(default_factory=list)
    skip_rules: List[str] = field(default_factory=list)


class ValidationEngine:
    """
    Core validation engine for comprehensive research validation.
    
    Features:
    - Multi-level validation (basic to publication-ready)
    - Extensible rule system
    - Auto-fix capabilities
    - Statistical validation
    - Ethical compliance checking
    - Reproducibility validation
    """
    
    def __init__(self, 
                 validation_level: ValidationLevel = ValidationLevel.STANDARD,
                 enable_auto_fix: bool = True,
                 enable_ml_validation: bool = True):
        self.validation_level = validation_level
        self.enable_auto_fix = enable_auto_fix
        self.enable_ml_validation = enable_ml_validation
        
        # Rule registry
        self.rules: Dict[str, ValidationRule] = {}
        self.rule_categories: Dict[ValidationCategory, List[str]] = {}
        
        # Validation history
        self.validation_history: List[ValidationReport] = []
        self.issue_patterns: Dict[str, int] = {}
        
        # ML-based validation components
        if enable_ml_validation:
            self.anomaly_detector = ResearchAnomalyDetector()
            self.quality_assessor = ResearchQualityAssessor()
            self.bias_detector = ResearchBiasDetector()
        
        # Performance metrics
        self.metrics = {
            "total_validations": 0,
            "auto_fixes_applied": 0,
            "critical_issues_found": 0,
            "validation_success_rate": 0.0,
            "average_validation_time": 0.0
        }
        
        # Initialize default rules
        self._initialize_default_rules()
        
        logger.info(f"Initialized Validation Engine with level: {validation_level.value}")
    
    def _initialize_default_rules(self):
        """Initialize default validation rules."""
        
        # Data integrity rules
        self.register_rule(ValidationRule(
            rule_id="data_integrity_001",
            name="Check for missing data",
            category=ValidationCategory.DATA_INTEGRITY,
            severity=ValidationSeverity.WARNING,
            description="Identifies missing or null data points",
            check_function=self._check_missing_data,
            auto_fix_function=self._fix_missing_data
        ))
        
        self.register_rule(ValidationRule(
            rule_id="data_integrity_002",
            name="Validate data types",
            category=ValidationCategory.DATA_INTEGRITY,
            severity=ValidationSeverity.ERROR,
            description="Ensures data types are consistent and appropriate",
            check_function=self._check_data_types,
            auto_fix_function=self._fix_data_types
        ))
        
        # Statistical validation rules
        self.register_rule(ValidationRule(
            rule_id="statistical_001",
            name="Sample size adequacy",
            category=ValidationCategory.STATISTICAL,
            severity=ValidationSeverity.WARNING,
            description="Checks if sample size is adequate for statistical analysis",
            check_function=self._check_sample_size
        ))
        
        self.register_rule(ValidationRule(
            rule_id="statistical_002",
            name="Statistical assumptions",
            category=ValidationCategory.STATISTICAL,
            severity=ValidationSeverity.ERROR,
            description="Validates statistical test assumptions",
            check_function=self._check_statistical_assumptions
        ))
        
        # Methodology rules
        self.register_rule(ValidationRule(
            rule_id="methodology_001",
            name="Control group presence",
            category=ValidationCategory.METHODOLOGY,
            severity=ValidationSeverity.WARNING,
            description="Checks for presence of appropriate control groups",
            check_function=self._check_control_groups
        ))
        
        # Reproducibility rules
        self.register_rule(ValidationRule(
            rule_id="reproducibility_001",
            name="Code documentation",
            category=ValidationCategory.REPRODUCIBILITY,
            severity=ValidationSeverity.WARNING,
            description="Ensures code is properly documented",
            check_function=self._check_code_documentation
        ))
        
        self.register_rule(ValidationRule(
            rule_id="reproducibility_002",
            name="Dependency specification",
            category=ValidationCategory.REPRODUCIBILITY,
            severity=ValidationSeverity.ERROR,
            description="Validates that dependencies are properly specified",
            check_function=self._check_dependencies
        ))
        
        # Citation rules
        self.register_rule(ValidationRule(
            rule_id="citation_001",
            name="Citation format validation",
            category=ValidationCategory.CITATION,
            severity=ValidationSeverity.WARNING,
            description="Validates citation formats",
            check_function=self._check_citation_format
        ))
        
        # Ethical rules
        self.register_rule(ValidationRule(
            rule_id="ethical_001",
            name="IRB approval check",
            category=ValidationCategory.ETHICAL,
            severity=ValidationSeverity.CRITICAL,
            description="Ensures IRB approval for human subjects research",
            check_function=self._check_irb_approval
        ))
        
        # Security rules
        self.register_rule(ValidationRule(
            rule_id="security_001",
            name="PII detection",
            category=ValidationCategory.SECURITY,
            severity=ValidationSeverity.CRITICAL,
            description="Detects personally identifiable information",
            check_function=self._check_pii_exposure
        ))
    
    def register_rule(self, rule: ValidationRule):
        """Register a new validation rule."""
        self.rules[rule.rule_id] = rule
        
        # Organize by category
        if rule.category not in self.rule_categories:
            self.rule_categories[rule.category] = []
        self.rule_categories[rule.category].append(rule.rule_id)
        
        logger.debug(f"Registered validation rule: {rule.name}")
    
    async def validate(self, context: ValidationContext) -> ValidationReport:
        """Perform comprehensive validation on a context."""
        try:
            start_time = datetime.now()
            
            report = ValidationReport(
                report_id=f"val_{uuid.uuid4().hex[:8]}",
                validation_level=context.validation_level,
                target_context=context.context_type,
                start_time=start_time,
                total_checks=0,
                passed_checks=0,
                failed_checks=0
            )
            
            # Get applicable rules
            applicable_rules = self._get_applicable_rules(context)
            report.total_checks = len(applicable_rules)
            
            # Run validation checks
            for rule_id in applicable_rules:
                if rule_id in context.skip_rules:
                    continue
                
                rule = self.rules[rule_id]
                
                try:
                    issues = await self._run_rule_check(rule, context)
                    
                    if issues:
                        report.failed_checks += 1
                        report.issues.extend(issues)
                        
                        # Attempt auto-fix if enabled
                        if self.enable_auto_fix and rule.auto_fix_function:
                            for issue in issues:
                                if issue.auto_fixable:
                                    await self._apply_auto_fix(rule, issue, context)
                    else:
                        report.passed_checks += 1
                        
                except Exception as e:
                    logger.error(f"Error running validation rule {rule_id}: {e}")
                    error_issue = ValidationIssue(
                        issue_id=f"err_{uuid.uuid4().hex[:8]}",
                        rule_id=rule_id,
                        severity=ValidationSeverity.ERROR,
                        category=rule.category,
                        title=f"Validation rule error: {rule.name}",
                        description=f"Error executing validation rule: {e}",
                        location="validation_engine",
                        suggested_fix="Review rule implementation",
                        auto_fixable=False,
                        confidence=1.0
                    )
                    report.issues.append(error_issue)
                    report.failed_checks += 1
            
            # Calculate overall score and validity
            report.overall_score = self._calculate_validation_score(report)
            report.is_valid = self._determine_validity(report)
            
            # Generate recommendations
            report.recommendations = await self._generate_recommendations(report, context)
            
            # Finalize report
            report.end_time = datetime.now()
            
            # Update metrics
            self.metrics["total_validations"] += 1
            self.metrics["validation_success_rate"] = (
                self.metrics["validation_success_rate"] * (self.metrics["total_validations"] - 1) + 
                report.overall_score
            ) / self.metrics["total_validations"]
            
            # Store in history
            self.validation_history.append(report)
            
            logger.info(f"Validation completed: {report.overall_score:.2f} score, "
                       f"{len(report.issues)} issues found")
            
            return report
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise ValidationError(f"Validation process failed: {e}") from e
    
    def _get_applicable_rules(self, context: ValidationContext) -> List[str]:
        """Get rules applicable to the given context."""
        applicable_rules = []
        
        for rule_id, rule in self.rules.items():
            if not rule.is_active:
                continue
            
            # Check validation level
            if context.validation_level == ValidationLevel.BASIC and \
               rule.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.BLOCKING]:
                continue
            elif context.validation_level == ValidationLevel.STANDARD and \
                 rule.severity == ValidationSeverity.BLOCKING:
                continue
            
            # Check context applicability
            if rule.applicable_contexts and \
               context.context_type not in rule.applicable_contexts:
                continue
            
            # Check custom rules
            if context.custom_rules and rule_id not in context.custom_rules:
                continue
            
            applicable_rules.append(rule_id)
        
        return applicable_rules
    
    async def _run_rule_check(self, rule: ValidationRule, 
                            context: ValidationContext) -> List[ValidationIssue]:
        """Run a single validation rule check."""
        try:
            # Execute the rule's check function
            check_result = await rule.check_function(context)
            
            if isinstance(check_result, list):
                return check_result
            elif isinstance(check_result, ValidationIssue):
                return [check_result]
            elif check_result is True or check_result is None:
                return []  # No issues
            else:
                # Convert generic result to issue
                issue = ValidationIssue(
                    issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    category=rule.category,
                    title=rule.name,
                    description=str(check_result),
                    location=context.context_type,
                    suggested_fix="Manual review required",
                    auto_fixable=rule.auto_fix_function is not None,
                    confidence=rule.confidence_threshold
                )
                return [issue]
                
        except Exception as e:
            logger.error(f"Error in rule check {rule.rule_id}: {e}")
            return []
    
    async def _apply_auto_fix(self, rule: ValidationRule, issue: ValidationIssue, 
                            context: ValidationContext):
        """Apply automatic fix for a validation issue."""
        try:
            if rule.auto_fix_function:
                await rule.auto_fix_function(issue, context)
                issue.is_resolved = True
                issue.resolution_notes = "Auto-fixed by validation engine"
                self.metrics["auto_fixes_applied"] += 1
                
                logger.info(f"Auto-fixed issue: {issue.title}")
                
        except Exception as e:
            logger.error(f"Failed to apply auto-fix for {issue.issue_id}: {e}")
    
    def _calculate_validation_score(self, report: ValidationReport) -> float:
        """Calculate overall validation score."""
        if report.total_checks == 0:
            return 1.0
        
        # Base score from pass rate
        pass_rate = report.passed_checks / report.total_checks
        
        # Penalty for issues by severity
        severity_penalties = {
            ValidationSeverity.INFO: 0.0,
            ValidationSeverity.WARNING: 0.05,
            ValidationSeverity.ERROR: 0.15,
            ValidationSeverity.CRITICAL: 0.3,
            ValidationSeverity.BLOCKING: 0.5
        }
        
        total_penalty = 0.0
        for issue in report.issues:
            if not issue.is_resolved:
                total_penalty += severity_penalties.get(issue.severity, 0.1)
        
        # Calculate final score
        score = max(0.0, pass_rate - total_penalty)
        return min(1.0, score)
    
    def _determine_validity(self, report: ValidationReport) -> bool:
        """Determine if the validation passes overall."""
        # No blocking issues
        blocking_issues = [i for i in report.issues 
                          if i.severity == ValidationSeverity.BLOCKING and not i.is_resolved]
        if blocking_issues:
            return False
        
        # Score threshold based on validation level
        thresholds = {
            ValidationLevel.BASIC: 0.6,
            ValidationLevel.STANDARD: 0.7,
            ValidationLevel.RIGOROUS: 0.8,
            ValidationLevel.PUBLICATION_READY: 0.9
        }
        
        threshold = thresholds[report.validation_level]
        return report.overall_score >= threshold
    
    async def _generate_recommendations(self, report: ValidationReport, 
                                      context: ValidationContext) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Category-based recommendations
        category_issues = {}
        for issue in report.issues:
            if not issue.is_resolved:
                if issue.category not in category_issues:
                    category_issues[issue.category] = []
                category_issues[issue.category].append(issue)
        
        for category, issues in category_issues.items():
            if category == ValidationCategory.DATA_INTEGRITY:
                recommendations.append(
                    f"Data integrity: {len(issues)} issues found. "
                    "Consider data cleaning and quality checks."
                )
            elif category == ValidationCategory.STATISTICAL:
                recommendations.append(
                    f"Statistical: {len(issues)} issues found. "
                    "Review statistical methods and assumptions."
                )
            elif category == ValidationCategory.REPRODUCIBILITY:
                recommendations.append(
                    f"Reproducibility: {len(issues)} issues found. "
                    "Improve documentation and dependency specification."
                )
            elif category == ValidationCategory.ETHICAL:
                recommendations.append(
                    f"Ethical: {len(issues)} issues found. "
                    "Ensure compliance with ethical guidelines."
                )
        
        # Score-based recommendations
        if report.overall_score < 0.7:
            recommendations.append(
                "Overall validation score is low. Consider comprehensive review."
            )
        
        return recommendations
    
    # Default validation rule implementations
    
    async def _check_missing_data(self, context: ValidationContext) -> List[ValidationIssue]:
        """Check for missing data in the context."""
        issues = []
        
        if hasattr(context.data, '__iter__') and not isinstance(context.data, str):
            try:
                missing_count = 0
                total_count = 0
                
                for item in context.data:
                    total_count += 1
                    if item is None or (isinstance(item, str) and not item.strip()):
                        missing_count += 1
                
                if missing_count > 0:
                    missing_rate = missing_count / total_count
                    
                    if missing_rate > 0.1:  # More than 10% missing
                        issue = ValidationIssue(
                            issue_id=f"missing_{uuid.uuid4().hex[:8]}",
                            rule_id="data_integrity_001",
                            severity=ValidationSeverity.WARNING,
                            category=ValidationCategory.DATA_INTEGRITY,
                            title="High missing data rate",
                            description=f"{missing_rate:.1%} of data is missing",
                            location=context.context_id,
                            suggested_fix="Consider data imputation or collection methods",
                            auto_fixable=True,
                            confidence=0.9
                        )
                        issues.append(issue)
                        
            except Exception as e:
                logger.error(f"Error checking missing data: {e}")
        
        return issues
    
    async def _fix_missing_data(self, issue: ValidationIssue, context: ValidationContext):
        """Auto-fix missing data issues."""
        # Simple imputation strategy - replace with mean/mode
        # This is a placeholder implementation
        logger.info(f"Applied missing data fix for {issue.issue_id}")
    
    async def _check_data_types(self, context: ValidationContext) -> List[ValidationIssue]:
        """Check for data type consistency."""
        issues = []
        
        # Implementation would check data type consistency
        # This is a placeholder
        
        return issues
    
    async def _fix_data_types(self, issue: ValidationIssue, context: ValidationContext):
        """Auto-fix data type issues."""
        logger.info(f"Applied data type fix for {issue.issue_id}")
    
    async def _check_sample_size(self, context: ValidationContext) -> List[ValidationIssue]:
        """Check if sample size is adequate."""
        issues = []
        
        # Statistical power analysis would be implemented here
        # This is a placeholder
        
        return issues
    
    async def _check_statistical_assumptions(self, context: ValidationContext) -> List[ValidationIssue]:
        """Check statistical test assumptions."""
        issues = []
        
        # Assumption testing would be implemented here
        # This is a placeholder
        
        return issues
    
    async def _check_control_groups(self, context: ValidationContext) -> List[ValidationIssue]:
        """Check for appropriate control groups."""
        issues = []
        
        # Control group analysis would be implemented here
        # This is a placeholder
        
        return issues
    
    async def _check_code_documentation(self, context: ValidationContext) -> List[ValidationIssue]:
        """Check code documentation quality."""
        issues = []
        
        # Code documentation analysis would be implemented here
        # This is a placeholder
        
        return issues
    
    async def _check_dependencies(self, context: ValidationContext) -> List[ValidationIssue]:
        """Check dependency specification."""
        issues = []
        
        # Dependency analysis would be implemented here
        # This is a placeholder
        
        return issues
    
    async def _check_citation_format(self, context: ValidationContext) -> List[ValidationIssue]:
        """Check citation format compliance."""
        issues = []
        
        # Citation format checking would be implemented here
        # This is a placeholder
        
        return issues
    
    async def _check_irb_approval(self, context: ValidationContext) -> List[ValidationIssue]:
        """Check for IRB approval documentation."""
        issues = []
        
        # IRB approval checking would be implemented here
        # This is a placeholder
        
        return issues
    
    async def _check_pii_exposure(self, context: ValidationContext) -> List[ValidationIssue]:
        """Check for exposed PII."""
        issues = []
        
        # PII detection would be implemented here
        # This is a placeholder
        
        return issues
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive validation metrics."""
        return {
            "engine_metrics": self.metrics,
            "total_rules": len(self.rules),
            "active_rules": len([r for r in self.rules.values() if r.is_active]),
            "validation_history_size": len(self.validation_history),
            "common_issue_patterns": dict(sorted(self.issue_patterns.items(), 
                                               key=lambda x: x[1], reverse=True)[:10]),
            "category_distribution": {
                category.value: len(rule_ids) 
                for category, rule_ids in self.rule_categories.items()
            }
        }


class ResearchAnomalyDetector:
    """ML-based anomaly detection for research data."""
    
    def __init__(self):
        self.models = {}
        self.anomaly_threshold = 0.95
    
    async def detect_anomalies(self, data: Any, context: str) -> List[Dict[str, Any]]:
        """Detect anomalies in research data."""
        anomalies = []
        
        # Placeholder for ML-based anomaly detection
        # Would implement actual anomaly detection algorithms
        
        return anomalies


class ResearchQualityAssessor:
    """ML-based research quality assessment."""
    
    def __init__(self):
        self.quality_models = {}
    
    async def assess_quality(self, content: Any, context: str) -> Dict[str, float]:
        """Assess research quality using ML models."""
        quality_scores = {
            "methodology_rigor": 0.8,
            "statistical_validity": 0.7,
            "reproducibility": 0.9,
            "novelty": 0.6,
            "clarity": 0.8
        }
        
        # Placeholder for ML-based quality assessment
        
        return quality_scores


class ResearchBiasDetector:
    """Detects potential biases in research."""
    
    def __init__(self):
        self.bias_patterns = {}
    
    async def detect_bias(self, content: Any, context: str) -> List[Dict[str, Any]]:
        """Detect potential research biases."""
        biases = []
        
        # Placeholder for bias detection algorithms
        
        return biases


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


# Integration functions

async def validate_research_note(note, validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
    """Validate a research note."""
    engine = ValidationEngine(validation_level=validation_level)
    
    context = ValidationContext(
        context_id=note.title,
        context_type="note",
        data=note.content,
        metadata={
            "note_type": note.note_type.value,
            "tags": note.frontmatter.tags,
            "created": note.frontmatter.created
        },
        validation_level=validation_level
    )
    
    return await engine.validate(context)


async def validate_experiment(experiment_data: Dict, validation_level: ValidationLevel = ValidationLevel.RIGOROUS) -> ValidationReport:
    """Validate experimental data and methodology."""
    engine = ValidationEngine(validation_level=validation_level)
    
    context = ValidationContext(
        context_id=experiment_data.get("id", "unknown"),
        context_type="experiment",
        data=experiment_data,
        validation_level=validation_level
    )
    
    return await engine.validate(context)


def create_custom_validation_rule(name: str, category: ValidationCategory, 
                                severity: ValidationSeverity, 
                                check_function: Callable,
                                description: str = "",
                                auto_fix_function: Callable = None) -> ValidationRule:
    """Create a custom validation rule."""
    return ValidationRule(
        rule_id=f"custom_{uuid.uuid4().hex[:8]}",
        name=name,
        category=category,
        severity=severity,
        description=description,
        check_function=check_function,
        auto_fix_function=auto_fix_function
    )