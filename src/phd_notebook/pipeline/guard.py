"""Main pipeline guard orchestrator."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .monitor import PipelineMonitor
from .detector import FailureDetector
from .healer import SelfHealer
from .resilience import ResilienceManager, CircuitBreakerOpenError
from .validation import InputValidator, SecurityAuditor
from .scaling import PerformanceOptimizer, ScalingConfig, AdaptiveController, ResourceMonitor
from .advanced_monitor import MLPredictor, AnomalyDetector, TrendAnalyzer
from ..utils.config import NotebookConfig
from ..utils.logging import get_logger


@dataclass
class GuardConfig:
    """Configuration for pipeline guard."""
    check_interval: int = 30  # seconds
    heal_timeout: int = 300  # seconds
    max_heal_attempts: int = 3
    notification_webhooks: List[str] = None
    enable_security_audit: bool = True
    enable_resilience: bool = True
    circuit_breaker_threshold: int = 5
    enable_ml_prediction: bool = True
    enable_performance_optimization: bool = True
    enable_adaptive_scaling: bool = True
    
    def __post_init__(self):
        if self.notification_webhooks is None:
            self.notification_webhooks = []


class PipelineGuard:
    """Self-healing pipeline guardian."""
    
    def __init__(self, config: GuardConfig = None):
        self.config = config or GuardConfig()
        self.logger = get_logger(__name__)
        
        self.monitor = PipelineMonitor()
        self.detector = FailureDetector()
        self.healer = SelfHealer()
        
        # Enhanced features
        if self.config.enable_resilience:
            self.resilience = ResilienceManager()
            # Register pipeline monitoring as a service
            self.resilience.register_service(
                "pipeline_monitor",
                failure_threshold=self.config.circuit_breaker_threshold
            )
        else:
            self.resilience = None
        
        if self.config.enable_security_audit:
            self.validator = InputValidator()
            self.auditor = SecurityAuditor()
        else:
            self.validator = None
            self.auditor = None
        
        # Advanced features
        if self.config.enable_performance_optimization:
            scaling_config = ScalingConfig(
                max_concurrent_heals=self.config.max_heal_attempts,
                adaptive_intervals=self.config.enable_adaptive_scaling
            )
            self.performance_optimizer = PerformanceOptimizer(scaling_config)
            self.adaptive_controller = AdaptiveController(scaling_config) if self.config.enable_adaptive_scaling else None
            self.resource_monitor = ResourceMonitor()
        else:
            self.performance_optimizer = None
            self.adaptive_controller = None
            self.resource_monitor = None
        
        if self.config.enable_ml_prediction:
            self.ml_predictor = MLPredictor()
            self.anomaly_detector = AnomalyDetector()
            self.trend_analyzer = TrendAnalyzer()
        else:
            self.ml_predictor = None
            self.anomaly_detector = None
            self.trend_analyzer = None
        
        self._is_running = False
        self._heal_history: Dict[str, List[datetime]] = {}
        self._execution_metrics = {
            "total_checks": 0,
            "successful_heals": 0,
            "failed_heals": 0,
            "security_violations": 0,
            "predictions_made": 0,
            "anomalies_detected": 0
        }
        
    async def start_monitoring(self) -> None:
        """Start continuous pipeline monitoring."""
        self._is_running = True
        self.logger.info("Pipeline guard started monitoring")
        
        while self._is_running:
            try:
                await self._check_and_heal_cycle()
                
                # Use adaptive interval if available
                if self.adaptive_controller:
                    interval = self.adaptive_controller.get_optimal_check_interval()
                else:
                    interval = self.config.check_interval
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(self.config.check_interval)
    
    async def stop_monitoring(self) -> None:
        """Stop pipeline monitoring."""
        self._is_running = False
        self.logger.info("Pipeline guard stopped monitoring")
    
    async def _check_and_heal_cycle(self) -> None:
        """Single check and heal cycle."""
        self._execution_metrics["total_checks"] += 1
        
        try:
            # Get pipeline status with performance optimization
            if self.performance_optimizer:
                # Use optimized pipeline checking
                pipeline_ids = await self._get_monitored_pipeline_ids()
                pipelines = await self.performance_optimizer.optimized_pipeline_check(
                    self.monitor.get_pipeline_status,
                    pipeline_ids
                )
            elif self.resilience:
                pipelines = await self.resilience.call_with_circuit_breaker(
                    "pipeline_monitor",
                    self.monitor.get_pipeline_status
                )
            else:
                pipelines = await self.monitor.get_pipeline_status()
            
            # Update adaptive controller with metrics
            if self.adaptive_controller:
                failed_count = sum(1 for status in pipelines.values() if status.state == "failed")
                self.adaptive_controller.update_load_metrics(
                    active_pipelines=len(pipelines),
                    failed_pipelines=failed_count,
                    healing_queue_size=len(self._heal_history),
                    avg_response_time=1.0  # Would calculate actual response time
                )
            
            # Process pipelines
            failed_pipelines = []
            for pipeline_id, status in pipelines.items():
                # Update ML models and trend analysis
                if self.ml_predictor:
                    pipeline_data = self._extract_pipeline_data(pipeline_id, status, pipelines)
                    
                    # Make prediction
                    failure_prob = self.ml_predictor.predict_failure_probability(pipeline_id, pipeline_data)
                    self._execution_metrics["predictions_made"] += 1
                    
                    # Add training data if pipeline actually failed
                    if status.state == "failed":
                        self.ml_predictor.add_training_data(pipeline_id, pipeline_data, True)
                    
                    # Proactive healing for high-risk pipelines
                    if failure_prob > 0.8 and status.state in ["running", "pending"]:
                        self.logger.warning(f"High failure probability ({failure_prob:.2f}) for {pipeline_id} - considering proactive measures")
                
                # Anomaly detection
                if self.anomaly_detector:
                    metrics = self._extract_metrics_for_anomaly_detection(status)
                    anomalies = self.anomaly_detector.detect_anomalies(pipeline_id, metrics)
                    if anomalies:
                        self._execution_metrics["anomalies_detected"] += len(anomalies)
                        self.logger.warning(f"Anomalies detected for {pipeline_id}: {len(anomalies)} issues")
                
                # Trend analysis
                if self.trend_analyzer:
                    self._update_trend_data(pipeline_id, status)
                
                # Handle failures
                if status.state == "failed":
                    failed_pipelines.append((pipeline_id, status))
            
            # Batch process failures for better performance
            if failed_pipelines:
                await self._handle_multiple_pipeline_failures(failed_pipelines)
        
        except CircuitBreakerOpenError:
            self.logger.warning("Pipeline monitoring circuit breaker is open - skipping this cycle")
        except Exception as e:
            self.logger.error(f"Error in check and heal cycle: {e}")
            raise
    
    async def _handle_pipeline_failure(self, pipeline_id: str, status: Any) -> None:
        """Handle a failed pipeline."""
        self.logger.warning(f"Pipeline {pipeline_id} failed: {status.error}")
        
        # Check if we should attempt healing
        if not self._should_attempt_heal(pipeline_id):
            self.logger.info(f"Skipping heal for {pipeline_id} - too many recent attempts")
            return
        
        # Detect failure type
        failure_type = await self.detector.analyze_failure(pipeline_id, status)
        
        # Attempt healing with security validation
        try:
            # Security audit before healing
            if self.auditor:
                audit_result = self.auditor.audit_pipeline_execution(
                    pipeline_id=pipeline_id,
                    commands=[],  # Would collect actual commands from healing process
                    files_accessed=[],  # Would collect from healing process
                    network_calls=[]   # Would collect from healing process
                )
                
                if audit_result["security_score"] < 50:
                    self.logger.error(f"Security audit failed for pipeline {pipeline_id} - blocking healing")
                    self._execution_metrics["security_violations"] += 1
                    return
            
            # Execute healing with resilience
            if self.resilience:
                success = await self.resilience.retry_with_backoff(
                    self.healer.heal_pipeline,
                    f"heal_{pipeline_id}",
                    pipeline_id,
                    failure_type
                )
            else:
                success = await self.healer.heal_pipeline(pipeline_id, failure_type)
            
            if success:
                self.logger.info(f"Successfully healed pipeline {pipeline_id}")
                self._execution_metrics["successful_heals"] += 1
                await self._notify_heal_success(pipeline_id, failure_type)
            else:
                self.logger.warning(f"Failed to heal pipeline {pipeline_id}")
                self._execution_metrics["failed_heals"] += 1
                await self._notify_heal_failure(pipeline_id, failure_type)
                
        except Exception as e:
            self.logger.error(f"Error during healing: {e}")
            self._execution_metrics["failed_heals"] += 1
            await self._notify_heal_error(pipeline_id, str(e))
        
        # Record heal attempt
        self._record_heal_attempt(pipeline_id)
    
    def _should_attempt_heal(self, pipeline_id: str) -> bool:
        """Check if we should attempt healing for this pipeline."""
        if pipeline_id not in self._heal_history:
            return True
        
        # Check recent heal attempts
        recent_attempts = [
            attempt for attempt in self._heal_history[pipeline_id]
            if attempt > datetime.now() - timedelta(hours=1)
        ]
        
        return len(recent_attempts) < self.config.max_heal_attempts
    
    def _record_heal_attempt(self, pipeline_id: str) -> None:
        """Record a heal attempt."""
        if pipeline_id not in self._heal_history:
            self._heal_history[pipeline_id] = []
        
        self._heal_history[pipeline_id].append(datetime.now())
        
        # Keep only recent attempts
        cutoff = datetime.now() - timedelta(days=1)
        self._heal_history[pipeline_id] = [
            attempt for attempt in self._heal_history[pipeline_id]
            if attempt > cutoff
        ]
    
    async def _notify_heal_success(self, pipeline_id: str, failure_type: str) -> None:
        """Send notification of successful healing."""
        message = f"✅ Pipeline {pipeline_id} healed successfully (failure: {failure_type})"
        await self._send_notifications(message)
    
    async def _notify_heal_failure(self, pipeline_id: str, failure_type: str) -> None:
        """Send notification of failed healing."""
        message = f"❌ Failed to heal pipeline {pipeline_id} (failure: {failure_type})"
        await self._send_notifications(message)
    
    async def _notify_heal_error(self, pipeline_id: str, error: str) -> None:
        """Send notification of healing error."""
        message = f"⚠️ Error healing pipeline {pipeline_id}: {error}"
        await self._send_notifications(message)
    
    async def _send_notifications(self, message: str) -> None:
        """Send notifications to configured webhooks."""
        for webhook in self.config.notification_webhooks:
            try:
                # Simple webhook notification implementation
                import httpx
                async with httpx.AsyncClient() as client:
                    await client.post(webhook, json={"text": message})
            except Exception as e:
                self.logger.error(f"Failed to send notification to {webhook}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current guard status."""
        status = {
            "is_running": self._is_running,
            "config": {
                "check_interval": self.config.check_interval,
                "heal_timeout": self.config.heal_timeout,
                "max_heal_attempts": self.config.max_heal_attempts,
                "security_audit_enabled": self.config.enable_security_audit,
                "resilience_enabled": self.config.enable_resilience,
            },
            "execution_metrics": self._execution_metrics,
            "heal_history": {
                pipeline_id: len(attempts)
                for pipeline_id, attempts in self._heal_history.items()
            }
        }
        
        # Add resilience metrics if available
        if self.resilience:
            status["resilience_metrics"] = self.resilience.get_service_metrics()
        
        # Add security summary if available
        if self.auditor:
            status["security_summary"] = self.auditor.get_security_summary()
        
        return status
    
    async def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive status report including health checks."""
        basic_status = self.get_status()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "guard_status": basic_status
        }
        
        # Add resilience report
        if self.resilience:
            report["resilience_report"] = await self.resilience.create_resilience_report()
        
        # Add current pipeline status
        try:
            pipelines = await self.monitor.get_pipeline_status()
            report["current_pipelines"] = {
                pipeline_id: {
                    "state": status.state,
                    "started_at": status.started_at.isoformat(),
                    "error": status.error
                }
                for pipeline_id, status in pipelines.items()
            }
        except Exception as e:
            report["current_pipelines"] = {"error": str(e)}
        
        return report
    
    async def _get_monitored_pipeline_ids(self) -> List[str]:
        """Get list of pipeline IDs to monitor."""
        # This would query the actual pipelines being monitored
        # For now, return a simple list
        try:
            pipelines = await self.monitor.get_pipeline_status()
            return list(pipelines.keys())
        except Exception:
            return []
    
    def _extract_pipeline_data(self, pipeline_id: str, status: Any, all_pipelines: Dict) -> Dict[str, Any]:
        """Extract data for ML prediction."""
        # Calculate metrics for this pipeline
        data = {
            "pipeline_id": pipeline_id,
            "current_state": status.state,
            "consecutive_failures": self._heal_history.get(pipeline_id, []),
            "failure_rate_24h": 0.0,  # Would calculate from history
            "avg_duration": 0.0,  # Would calculate from history
            "system_cpu": 0.0,  # Would get from resource monitor
            "system_memory": 0.0,  # Would get from resource monitor
            "active_pipelines": len(all_pipelines),
            "success_rate_7d": 1.0,  # Would calculate from history
            "mean_time_between_failures": 24.0,  # Would calculate from history
        }
        
        return data
    
    def _extract_metrics_for_anomaly_detection(self, status: Any) -> Dict[str, float]:
        """Extract metrics for anomaly detection."""
        return {
            "duration": (datetime.now() - status.started_at).total_seconds() if status.started_at else 0,
            "cpu_usage": 0.0,  # Would get from monitoring
            "memory_usage": 0.0,  # Would get from monitoring
            "failure_rate": 1.0 if status.state == "failed" else 0.0
        }
    
    def _update_trend_data(self, pipeline_id: str, status: Any):
        """Update trend analysis data."""
        duration = (datetime.now() - status.started_at).total_seconds() if status.started_at else 0
        
        self.trend_analyzer.add_data_point(pipeline_id, "duration", duration)
        self.trend_analyzer.add_data_point(pipeline_id, "success_rate", 0.0 if status.state == "failed" else 1.0)
    
    async def _handle_multiple_pipeline_failures(self, failed_pipelines: List[Tuple[str, Any]]):
        """Handle multiple pipeline failures efficiently."""
        healing_requests = []
        
        for pipeline_id, status in failed_pipelines:
            # Check if we should attempt healing
            if not self._should_attempt_heal(pipeline_id):
                continue
            
            # Detect failure type
            failure_type = await self.detector.analyze_failure(pipeline_id, status)
            healing_requests.append((pipeline_id, failure_type))
        
        if not healing_requests:
            return
        
        # Use optimized healing if available
        if self.performance_optimizer:
            results = await self.performance_optimizer.optimized_healing(
                self._heal_single_pipeline,
                healing_requests
            )
            
            # Process results
            for pipeline_id, success in results.items():
                if success:
                    self._execution_metrics["successful_heals"] += 1
                    await self._notify_heal_success(pipeline_id, "batch_healing")
                else:
                    self._execution_metrics["failed_heals"] += 1
                    await self._notify_heal_failure(pipeline_id, "batch_healing")
                
                self._record_heal_attempt(pipeline_id)
        else:
            # Fallback to individual healing
            for pipeline_id, failure_type in healing_requests:
                await self._handle_pipeline_failure(pipeline_id, status)
    
    async def _heal_single_pipeline(self, pipeline_id: str, failure_analysis: Any) -> bool:
        """Heal a single pipeline (for use in batch operations)."""
        try:
            return await self.healer.heal_pipeline(pipeline_id, failure_analysis)
        except Exception as e:
            self.logger.error(f"Error healing pipeline {pipeline_id}: {e}")
            return False