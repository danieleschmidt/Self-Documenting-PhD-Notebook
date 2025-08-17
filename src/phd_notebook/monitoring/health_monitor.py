"""
Advanced health monitoring and system diagnostics for the PhD notebook.
Implements real-time health checks, performance monitoring, and automated alerting.
"""

import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import aiofiles
import json

from ..utils.exceptions import HealthCheckError, MonitoringError
from .metrics import MetricsCollector


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check configuration."""
    name: str
    check_function: Callable
    interval_seconds: int = 60
    timeout_seconds: int = 10
    critical: bool = False
    enabled: bool = True
    last_run: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_error: Optional[str] = None
    consecutive_failures: int = 0


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    process_count: int
    uptime_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AlertRule:
    """Alert configuration for health monitoring."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: str = "warning"
    cooldown_minutes: int = 15
    last_triggered: Optional[datetime] = None
    enabled: bool = True


class HealthMonitor:
    """
    Comprehensive system health monitoring and alerting.
    """
    
    def __init__(self, metrics_collector: MetricsCollector = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.logger = logging.getLogger("health_monitor")
        
        self.health_checks: Dict[str, HealthCheck] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.system_metrics_history: List[SystemMetrics] = []
        
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.start_time = datetime.now()
        
        # Initialize default health checks
        self._register_default_health_checks()
        self._register_default_alerts()
    
    def _register_default_health_checks(self):
        """Register default system health checks."""
        
        # System resource checks
        self.register_health_check(
            "cpu_usage",
            self._check_cpu_usage,
            interval_seconds=30,
            critical=True
        )
        
        self.register_health_check(
            "memory_usage", 
            self._check_memory_usage,
            interval_seconds=30,
            critical=True
        )
        
        self.register_health_check(
            "disk_space",
            self._check_disk_space,
            interval_seconds=300,  # 5 minutes
            critical=True
        )
        
        # Application-specific checks
        self.register_health_check(
            "database_connection",
            self._check_database_connection,
            interval_seconds=60
        )
        
        self.register_health_check(
            "external_apis",
            self._check_external_apis,
            interval_seconds=120
        )
        
        self.register_health_check(
            "file_system_access",
            self._check_file_system_access,
            interval_seconds=300
        )
    
    def _register_default_alerts(self):
        """Register default alert rules."""
        
        # High CPU usage alert
        self.register_alert_rule(
            "high_cpu_usage",
            lambda metrics: metrics.get("system", {}).get("cpu_percent", 0) > 85,
            severity="warning",
            cooldown_minutes=10
        )
        
        # High memory usage alert
        self.register_alert_rule(
            "high_memory_usage",
            lambda metrics: metrics.get("system", {}).get("memory_percent", 0) > 90,
            severity="critical",
            cooldown_minutes=5
        )
        
        # Low disk space alert
        self.register_alert_rule(
            "low_disk_space",
            lambda metrics: metrics.get("system", {}).get("disk_percent", 0) > 95,
            severity="critical",
            cooldown_minutes=30
        )
        
        # Multiple health check failures
        self.register_alert_rule(
            "multiple_health_failures",
            lambda metrics: len([
                check for check in metrics.get("health_checks", {}).values()
                if check.get("status") != "healthy"
            ]) >= 3,
            severity="critical",
            cooldown_minutes=15
        )
    
    def register_health_check(
        self,
        name: str,
        check_function: Callable,
        interval_seconds: int = 60,
        timeout_seconds: int = 10,
        critical: bool = False,
        enabled: bool = True
    ):
        """Register a new health check."""
        health_check = HealthCheck(
            name=name,
            check_function=check_function,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
            critical=critical,
            enabled=enabled
        )
        
        self.health_checks[name] = health_check
        self.logger.info(f"Registered health check: {name}")
    
    def register_alert_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        severity: str = "warning",
        cooldown_minutes: int = 15
    ):
        """Register a new alert rule."""
        alert_rule = AlertRule(
            name=name,
            condition=condition,
            severity=severity,
            cooldown_minutes=cooldown_minutes
        )
        
        self.alert_rules[name] = alert_rule
        self.logger.info(f"Registered alert rule: {name}")
    
    async def start_monitoring(self):
        """Start the health monitoring system."""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop the health monitoring system."""
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Keep only last 1000 entries (about 16 hours at 1-minute intervals)
                if len(self.system_metrics_history) > 1000:
                    self.system_metrics_history = self.system_metrics_history[-1000:]
                
                # Run health checks
                await self._run_health_checks()
                
                # Check alert rules
                await self._check_alert_rules()
                
                # Emit metrics
                await self._emit_monitoring_metrics()
                
                await asyncio.sleep(10)  # Main loop runs every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            network_io={
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv
            },
            process_count=len(psutil.pids()),
            uptime_seconds=uptime
        )
        
        return metrics
    
    async def _run_health_checks(self):
        """Execute all enabled health checks."""
        current_time = datetime.now()
        
        for name, health_check in self.health_checks.items():
            if not health_check.enabled:
                continue
            
            # Check if it's time to run this health check
            if (health_check.last_run and 
                (current_time - health_check.last_run).total_seconds() < health_check.interval_seconds):
                continue
            
            try:
                # Run health check with timeout
                result = await asyncio.wait_for(
                    self._execute_health_check(health_check),
                    timeout=health_check.timeout_seconds
                )
                
                # Update health check status
                health_check.last_run = current_time
                health_check.last_status = HealthStatus.HEALTHY if result else HealthStatus.WARNING
                health_check.last_error = None
                health_check.consecutive_failures = 0 if result else health_check.consecutive_failures + 1
                
            except asyncio.TimeoutError:
                self._handle_health_check_failure(health_check, "Health check timed out")
            except Exception as e:
                self._handle_health_check_failure(health_check, str(e))
    
    async def _execute_health_check(self, health_check: HealthCheck) -> bool:
        """Execute a single health check."""
        if asyncio.iscoroutinefunction(health_check.check_function):
            return await health_check.check_function()
        else:
            return health_check.check_function()
    
    def _handle_health_check_failure(self, health_check: HealthCheck, error: str):
        """Handle health check failure."""
        health_check.last_run = datetime.now()
        health_check.last_status = HealthStatus.CRITICAL if health_check.critical else HealthStatus.WARNING
        health_check.last_error = error
        health_check.consecutive_failures += 1
        
        self.logger.error(
            f"Health check '{health_check.name}' failed: {error} "
            f"(consecutive failures: {health_check.consecutive_failures})"
        )
    
    async def _check_alert_rules(self):
        """Evaluate all alert rules and trigger alerts if necessary."""
        current_metrics = self.get_current_health_status()
        current_time = datetime.now()
        
        for name, alert_rule in self.alert_rules.items():
            if not alert_rule.enabled:
                continue
            
            # Check cooldown period
            if (alert_rule.last_triggered and
                (current_time - alert_rule.last_triggered).total_seconds() < alert_rule.cooldown_minutes * 60):
                continue
            
            try:
                if alert_rule.condition(current_metrics):
                    await self._trigger_alert(alert_rule, current_metrics)
                    alert_rule.last_triggered = current_time
                    
            except Exception as e:
                self.logger.error(f"Error evaluating alert rule '{name}': {e}")
    
    async def _trigger_alert(self, alert_rule: AlertRule, metrics: Dict[str, Any]):
        """Trigger an alert."""
        alert_data = {
            "alert_name": alert_rule.name,
            "severity": alert_rule.severity,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        self.logger.warning(f"ALERT TRIGGERED: {alert_rule.name} ({alert_rule.severity})")
        
        # Emit alert metric
        self.metrics_collector.record_counter(f"alerts.{alert_rule.name}.triggered")
        
        # Here you could add integrations with external alerting systems
        # (e.g., email, Slack, PagerDuty, etc.)
    
    async def _emit_monitoring_metrics(self):
        """Emit monitoring metrics to the metrics collector."""
        if not self.system_metrics_history:
            return
        
        latest_metrics = self.system_metrics_history[-1]
        
        # Emit system metrics
        self.metrics_collector.record_gauge("system.cpu_percent", latest_metrics.cpu_percent)
        self.metrics_collector.record_gauge("system.memory_percent", latest_metrics.memory_percent)
        self.metrics_collector.record_gauge("system.disk_percent", latest_metrics.disk_percent)
        self.metrics_collector.record_gauge("system.process_count", latest_metrics.process_count)
        self.metrics_collector.record_gauge("system.uptime_seconds", latest_metrics.uptime_seconds)
        
        # Emit health check metrics
        for name, health_check in self.health_checks.items():
            status_value = {
                HealthStatus.HEALTHY: 1,
                HealthStatus.WARNING: 0.5,
                HealthStatus.CRITICAL: 0,
                HealthStatus.UNKNOWN: -1
            }.get(health_check.last_status, -1)
            
            self.metrics_collector.record_gauge(f"health_check.{name}.status", status_value)
            self.metrics_collector.record_gauge(f"health_check.{name}.consecutive_failures", health_check.consecutive_failures)
    
    def get_current_health_status(self) -> Dict[str, Any]:
        """Get current comprehensive health status."""
        overall_status = HealthStatus.HEALTHY
        critical_issues = []
        warning_issues = []
        
        # Check health checks
        health_check_details = {}
        for name, health_check in self.health_checks.items():
            status_info = {
                "status": health_check.last_status.value,
                "last_run": health_check.last_run.isoformat() if health_check.last_run else None,
                "last_error": health_check.last_error,
                "consecutive_failures": health_check.consecutive_failures,
                "critical": health_check.critical
            }
            health_check_details[name] = status_info
            
            if health_check.last_status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                critical_issues.append(f"Health check '{name}' is critical: {health_check.last_error}")
            elif health_check.last_status == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.WARNING
                warning_issues.append(f"Health check '{name}' has warnings: {health_check.last_error}")
        
        # Get latest system metrics
        system_info = {}
        if self.system_metrics_history:
            latest = self.system_metrics_history[-1]
            system_info = {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "disk_percent": latest.disk_percent,
                "process_count": latest.process_count,
                "uptime_seconds": latest.uptime_seconds,
                "timestamp": latest.timestamp.isoformat()
            }
        
        return {
            "overall_status": overall_status.value,
            "critical_issues": critical_issues,
            "warning_issues": warning_issues,
            "health_checks": health_check_details,
            "system": system_info,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "monitoring_active": self.is_running
        }
    
    async def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        current_status = self.get_current_health_status()
        
        # Calculate trends
        trends = {}
        if len(self.system_metrics_history) >= 2:
            recent = self.system_metrics_history[-10:]  # Last 10 data points
            older = self.system_metrics_history[-20:-10] if len(self.system_metrics_history) >= 20 else []
            
            if older:
                recent_avg_cpu = sum(m.cpu_percent for m in recent) / len(recent)
                older_avg_cpu = sum(m.cpu_percent for m in older) / len(older)
                trends["cpu_trend"] = "increasing" if recent_avg_cpu > older_avg_cpu else "decreasing"
                
                recent_avg_memory = sum(m.memory_percent for m in recent) / len(recent)
                older_avg_memory = sum(m.memory_percent for m in older) / len(older)
                trends["memory_trend"] = "increasing" if recent_avg_memory > older_avg_memory else "decreasing"
        
        return {
            **current_status,
            "trends": trends,
            "metrics_history_length": len(self.system_metrics_history),
            "registered_health_checks": len(self.health_checks),
            "registered_alert_rules": len(self.alert_rules)
        }
    
    # Default health check implementations
    async def _check_cpu_usage(self) -> bool:
        """Check if CPU usage is within acceptable limits."""
        cpu_percent = psutil.cpu_percent(interval=1)
        return cpu_percent < 95
    
    async def _check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits."""
        memory = psutil.virtual_memory()
        return memory.percent < 95
    
    async def _check_disk_space(self) -> bool:
        """Check if disk space is sufficient."""
        disk = psutil.disk_usage('/')
        return disk.percent < 90
    
    async def _check_database_connection(self) -> bool:
        """Check database connectivity (placeholder)."""
        # This would contain actual database connection logic
        return True
    
    async def _check_external_apis(self) -> bool:
        """Check external API availability (placeholder)."""
        # This would contain actual API health checks
        return True
    
    async def _check_file_system_access(self) -> bool:
        """Check file system read/write access."""
        try:
            test_file = Path("/tmp/health_check_test")
            async with aiofiles.open(test_file, "w") as f:
                await f.write("health_check")
            
            async with aiofiles.open(test_file, "r") as f:
                content = await f.read()
            
            test_file.unlink()
            return content == "health_check"
            
        except Exception:
            return False