"""
Advanced Research Monitoring and Health Checks.
"""

import asyncio
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

from ..core.notebook import ResearchNotebook
from ..utils.exceptions import ResearchError


class AlertLevel(Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """System health metric."""
    
    name: str
    value: float
    unit: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    status: str = "normal"  # normal, warning, critical
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ResearchAlert:
    """Research system alert."""
    
    id: str
    level: AlertLevel
    component: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class ResearchMonitor:
    """Comprehensive monitoring system for research activities."""
    
    def __init__(self, notebook: ResearchNotebook):
        self.notebook = notebook
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.is_monitoring = False
        self.metrics: Dict[str, HealthMetric] = {}
        self.alerts: Dict[str, ResearchAlert] = {}
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring intervals
        self.system_check_interval = 60  # seconds
        self.research_check_interval = 300  # 5 minutes
        self.backup_check_interval = 3600  # 1 hour
        
        # Performance baselines
        self.performance_baselines: Dict[str, float] = {}
        self.productivity_trends: List[Tuple[datetime, float]] = []
        
        # Initialize health metrics
        self._initialize_health_metrics()
    
    def _initialize_health_metrics(self):
        """Initialize system health metrics."""
        
        metrics = [
            HealthMetric("cpu_usage", 0.0, "%", 80.0, 95.0),
            HealthMetric("memory_usage", 0.0, "%", 85.0, 95.0),
            HealthMetric("disk_usage", 0.0, "%", 80.0, 90.0),
            HealthMetric("note_creation_rate", 0.0, "notes/day", None, None),
            HealthMetric("experiment_success_rate", 0.0, "%", 60.0, 40.0),
            HealthMetric("writing_velocity", 0.0, "words/day", None, None),
            HealthMetric("backup_health", 100.0, "%", 90.0, 80.0),
            HealthMetric("ai_response_time", 0.0, "ms", 5000.0, 10000.0),
            HealthMetric("workflow_efficiency", 100.0, "%", 80.0, 60.0)
        ]
        
        for metric in metrics:
            self.metrics[metric.name] = metric
    
    async def start_monitoring(self):
        """Start comprehensive monitoring."""
        
        if self.is_monitoring:
            self.logger.warning("Monitoring already running")
            return
        
        self.is_monitoring = True
        self.logger.info("Starting research monitoring system")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._system_health_monitor()),
            asyncio.create_task(self._research_progress_monitor()),
            asyncio.create_task(self._productivity_monitor()),
            asyncio.create_task(self._backup_monitor()),
            asyncio.create_task(self._ai_performance_monitor())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
            await self.stop_monitoring()
    
    async def stop_monitoring(self):
        """Stop monitoring system."""
        
        self.is_monitoring = False
        self.logger.info("Stopping research monitoring system")
    
    async def _system_health_monitor(self):
        """Monitor system resource health."""
        
        while self.is_monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                await self._update_metric("cpu_usage", cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                await self._update_metric("memory_usage", memory_percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                await self._update_metric("disk_usage", disk_percent)
                
                # Check for critical resource usage
                await self._check_resource_alerts()
                
                await asyncio.sleep(self.system_check_interval)
                
            except Exception as e:
                self.logger.error(f"System health monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _research_progress_monitor(self):
        """Monitor research progress and productivity."""
        
        while self.is_monitoring:
            try:
                # Calculate note creation rate
                notes = self.notebook.list_notes()
                recent_notes = len([n for n in notes 
                                  if self._is_recent(getattr(n.frontmatter, 'created', datetime.now()))])
                await self._update_metric("note_creation_rate", recent_notes)
                
                # Calculate experiment success rate
                experiments = self.notebook.get_experiments()
                if experiments:
                    successful = len([e for e in experiments 
                                    if getattr(e.frontmatter, 'status', '') == 'completed'])
                    success_rate = (successful / len(experiments)) * 100
                    await self._update_metric("experiment_success_rate", success_rate)
                
                # Monitor workflow efficiency
                workflow_status = self.notebook.get_workflow_status()
                efficiency = self._calculate_workflow_efficiency(workflow_status)
                await self._update_metric("workflow_efficiency", efficiency)
                
                # Check for research progress alerts
                await self._check_progress_alerts()
                
                await asyncio.sleep(self.research_check_interval)
                
            except Exception as e:
                self.logger.error(f"Research progress monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _productivity_monitor(self):
        """Monitor productivity trends and patterns."""
        
        while self.is_monitoring:
            try:
                # Calculate writing velocity
                writing_velocity = await self._calculate_writing_velocity()
                await self._update_metric("writing_velocity", writing_velocity)
                
                # Track productivity trend
                productivity_score = self._calculate_productivity_score()
                self.productivity_trends.append((datetime.now(), productivity_score))
                
                # Keep only last 30 days of trends
                cutoff_date = datetime.now() - timedelta(days=30)
                self.productivity_trends = [(date, score) for date, score in self.productivity_trends 
                                          if date > cutoff_date]
                
                # Analyze productivity patterns
                await self._analyze_productivity_patterns()
                
                await asyncio.sleep(self.research_check_interval * 2)  # Less frequent
                
            except Exception as e:
                self.logger.error(f"Productivity monitoring error: {e}")
                await asyncio.sleep(120)
    
    async def _backup_monitor(self):
        """Monitor backup health and integrity."""
        
        while self.is_monitoring:
            try:
                # Check vault backup status
                backup_health = await self._check_backup_health()
                await self._update_metric("backup_health", backup_health)
                
                # Verify backup integrity
                integrity_score = await self._verify_backup_integrity()
                
                if integrity_score < 80:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        "backup",
                        "Backup integrity check failed",
                        {"integrity_score": integrity_score}
                    )
                
                await asyncio.sleep(self.backup_check_interval)
                
            except Exception as e:
                self.logger.error(f"Backup monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _ai_performance_monitor(self):
        """Monitor AI client performance and response times."""
        
        while self.is_monitoring:
            try:
                # Test AI response time
                start_time = time.time()
                
                # Simple AI health check
                test_prompt = "Health check: respond with 'OK'"
                
                try:
                    from ..ai.client_factory import AIClientFactory
                    ai_client = AIClientFactory.get_client()
                    await ai_client.generate_text(test_prompt, max_tokens=5)
                    response_time = (time.time() - start_time) * 1000  # ms
                    
                    await self._update_metric("ai_response_time", response_time)
                    
                except Exception as ai_error:
                    await self._create_alert(
                        AlertLevel.ERROR,
                        "ai_client",
                        "AI client health check failed",
                        {"error": str(ai_error)}
                    )
                
                await asyncio.sleep(300)  # 5 minute intervals
                
            except Exception as e:
                self.logger.error(f"AI performance monitoring error: {e}")
                await asyncio.sleep(120)
    
    async def _update_metric(self, name: str, value: float):
        """Update a health metric and check thresholds."""
        
        if name not in self.metrics:
            return
        
        metric = self.metrics[name]
        metric.value = value
        metric.last_updated = datetime.now()
        
        # Check thresholds
        old_status = metric.status
        
        if metric.threshold_critical and value >= metric.threshold_critical:
            metric.status = "critical"
        elif metric.threshold_warning and value >= metric.threshold_warning:
            metric.status = "warning"
        else:
            metric.status = "normal"
        
        # Create alert if status changed to warning/critical
        if old_status != metric.status and metric.status in ["warning", "critical"]:
            level = AlertLevel.CRITICAL if metric.status == "critical" else AlertLevel.WARNING
            await self._create_alert(
                level,
                "metrics",
                f"{name} threshold exceeded",
                {
                    "metric": name,
                    "value": value,
                    "threshold": metric.threshold_critical if metric.status == "critical" else metric.threshold_warning,
                    "unit": metric.unit
                }
            )
    
    async def _check_resource_alerts(self):
        """Check for system resource alerts."""
        
        cpu_metric = self.metrics["cpu_usage"]
        memory_metric = self.metrics["memory_usage"]
        disk_metric = self.metrics["disk_usage"]
        
        # Check for sustained high resource usage
        if (cpu_metric.status == "critical" and 
            memory_metric.status in ["warning", "critical"]):
            
            await self._create_alert(
                AlertLevel.CRITICAL,
                "system",
                "High system resource usage detected",
                {
                    "cpu_usage": cpu_metric.value,
                    "memory_usage": memory_metric.value,
                    "recommendation": "Consider closing applications or upgrading hardware"
                }
            )
    
    async def _check_progress_alerts(self):
        """Check for research progress alerts."""
        
        note_rate = self.metrics["note_creation_rate"].value
        experiment_rate = self.metrics["experiment_success_rate"].value
        
        # Check for low productivity
        if note_rate < 1.0:  # Less than 1 note per day
            await self._create_alert(
                AlertLevel.WARNING,
                "productivity",
                "Low note creation rate detected",
                {
                    "current_rate": note_rate,
                    "recommendation": "Consider increasing daily research activities"
                }
            )
        
        # Check for low experiment success
        if experiment_rate < 50.0 and experiment_rate > 0:
            await self._create_alert(
                AlertLevel.WARNING,
                "research",
                "Low experiment success rate",
                {
                    "success_rate": experiment_rate,
                    "recommendation": "Review experimental methodology and design"
                }
            )
    
    def _is_recent(self, date: datetime, days: int = 7) -> bool:
        """Check if date is within recent timeframe."""
        return (datetime.now() - date).days <= days
    
    async def _calculate_writing_velocity(self) -> float:
        """Calculate recent writing velocity in words per day."""
        
        try:
            notes = self.notebook.list_notes()
            recent_notes = [n for n in notes 
                          if self._is_recent(getattr(n.frontmatter, 'created', datetime.now()))]
            
            if not recent_notes:
                return 0.0
            
            total_words = sum(len(note.content.split()) for note in recent_notes)
            return total_words / 7.0  # Average per day over last week
            
        except Exception as e:
            self.logger.error(f"Writing velocity calculation error: {e}")
            return 0.0
    
    def _calculate_workflow_efficiency(self, workflow_status: Dict[str, Any]) -> float:
        """Calculate workflow efficiency percentage."""
        
        if not workflow_status:
            return 100.0
        
        try:
            total_workflows = len(workflow_status)
            active_workflows = len([w for w in workflow_status.values() if w.get('status') == 'active'])
            
            if total_workflows == 0:
                return 100.0
            
            return (active_workflows / total_workflows) * 100
            
        except Exception:
            return 100.0
    
    def _calculate_productivity_score(self) -> float:
        """Calculate overall productivity score (0-100)."""
        
        # Combine multiple productivity indicators
        factors = []
        
        # Note creation rate (normalized to 0-100)
        note_rate = min(self.metrics["note_creation_rate"].value * 10, 100)
        factors.append(note_rate * 0.3)
        
        # Writing velocity (normalized)
        writing_score = min(self.metrics["writing_velocity"].value / 10, 100)
        factors.append(writing_score * 0.4)
        
        # Experiment success rate
        exp_success = self.metrics["experiment_success_rate"].value
        factors.append(exp_success * 0.2)
        
        # Workflow efficiency
        workflow_eff = self.metrics["workflow_efficiency"].value
        factors.append(workflow_eff * 0.1)
        
        return sum(factors) if factors else 50.0
    
    async def _analyze_productivity_patterns(self):
        """Analyze productivity patterns and generate insights."""
        
        if len(self.productivity_trends) < 7:  # Need at least a week of data
            return
        
        recent_scores = [score for _, score in self.productivity_trends[-7:]]
        avg_recent = sum(recent_scores) / len(recent_scores)
        
        older_scores = [score for _, score in self.productivity_trends[-14:-7]]
        if older_scores:
            avg_older = sum(older_scores) / len(older_scores)
            
            # Check for declining productivity
            if avg_recent < avg_older * 0.8:  # 20% decline
                await self._create_alert(
                    AlertLevel.WARNING,
                    "productivity",
                    "Declining productivity trend detected",
                    {
                        "recent_average": avg_recent,
                        "previous_average": avg_older,
                        "decline_percentage": ((avg_older - avg_recent) / avg_older) * 100,
                        "recommendation": "Consider reviewing research goals and methods"
                    }
                )
    
    async def _check_backup_health(self) -> float:
        """Check backup system health."""
        
        try:
            # Check if vault exists and is accessible
            vault_path = self.notebook.vault_path
            if not vault_path.exists():
                return 0.0
            
            # Check recent backup activity (simplified)
            # In real implementation, would check actual backup systems
            return 100.0
            
        except Exception as e:
            self.logger.error(f"Backup health check error: {e}")
            return 50.0
    
    async def _verify_backup_integrity(self) -> float:
        """Verify backup integrity."""
        
        try:
            # Simplified integrity check
            # In real implementation, would verify backup checksums, etc.
            return 100.0
            
        except Exception as e:
            self.logger.error(f"Backup integrity verification error: {e}")
            return 0.0
    
    async def _create_alert(
        self, 
        level: AlertLevel, 
        component: str, 
        message: str, 
        details: Dict[str, Any]
    ):
        """Create and process a new alert."""
        
        alert = ResearchAlert(
            id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            level=level,
            component=component,
            message=message,
            details=details
        )
        
        self.alerts[alert.id] = alert
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }[level]
        
        self.logger.log(log_level, f"[{component}] {message}: {details}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
        
        # Auto-resolve info alerts after 1 hour
        if level == AlertLevel.INFO:
            asyncio.create_task(self._auto_resolve_alert(alert.id, 3600))
    
    async def _auto_resolve_alert(self, alert_id: str, delay_seconds: int):
        """Auto-resolve alert after delay."""
        
        await asyncio.sleep(delay_seconds)
        
        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolution_time = datetime.now()
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        
        status = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "active_alerts": len([a for a in self.alerts.values() if not a.resolved]),
            "critical_alerts": len([a for a in self.alerts.values() 
                                  if not a.resolved and a.level == AlertLevel.CRITICAL])
        }
        
        # Add metric summaries
        for name, metric in self.metrics.items():
            status["metrics"][name] = {
                "value": metric.value,
                "unit": metric.unit,
                "status": metric.status,
                "last_updated": metric.last_updated.isoformat()
            }
        
        # Determine overall status
        if status["critical_alerts"] > 0:
            status["overall_status"] = "critical"
        elif any(m["status"] == "warning" for m in status["metrics"].values()):
            status["overall_status"] = "warning"
        elif not self.is_monitoring:
            status["overall_status"] = "unknown"
        
        return status
    
    def get_alerts(self, resolved: bool = False, level: AlertLevel = None) -> List[ResearchAlert]:
        """Get alerts, optionally filtered."""
        
        alerts = list(self.alerts.values())
        
        # Filter by resolution status
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        # Filter by level
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert."""
        
        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolution_time = datetime.now()
            self.logger.info(f"Alert {alert_id} resolved manually")
            return True
        
        return False
    
    def get_productivity_trend(self, days: int = 30) -> Dict[str, Any]:
        """Get productivity trend analysis."""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trends = [(date, score) for date, score in self.productivity_trends 
                        if date > cutoff_date]
        
        if len(recent_trends) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        scores = [score for _, score in recent_trends]
        
        # Calculate trend statistics
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        # Simple trend direction
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        
        trend_direction = "stable"
        if second_half and first_half:
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            
            if avg_second > avg_first * 1.1:
                trend_direction = "improving"
            elif avg_second < avg_first * 0.9:
                trend_direction = "declining"
        
        return {
            "period_days": days,
            "data_points": len(recent_trends),
            "average_score": avg_score,
            "min_score": min_score,
            "max_score": max_score,
            "trend_direction": trend_direction,
            "current_score": scores[-1] if scores else 0
        }
    
    async def generate_health_report(self) -> str:
        """Generate comprehensive health report."""
        
        health_status = self.get_health_status()
        productivity_trend = self.get_productivity_trend()
        active_alerts = self.get_alerts(resolved=False)
        
        report = f"""
# Research System Health Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Status: {health_status['overall_status'].upper()}

## System Metrics
{self._format_metrics_table(health_status['metrics'])}

## Productivity Analysis
- **Current Score**: {productivity_trend.get('current_score', 0):.1f}/100
- **30-Day Average**: {productivity_trend.get('average_score', 0):.1f}/100
- **Trend Direction**: {productivity_trend.get('trend_direction', 'unknown').title()}
- **Data Points**: {productivity_trend.get('data_points', 0)}

## Active Alerts ({len(active_alerts)})
{self._format_alerts_table(active_alerts)}

## Recommendations
{await self._generate_health_recommendations(health_status, active_alerts)}

---
*Generated by ResearchMonitor*
        """
        
        # Create health report note
        if self.notebook:
            health_note = self.notebook.create_note(
                title=f"Health Report - {datetime.now().strftime('%Y-%m-%d')}",
                content=report,
                note_type=self.notebook.NoteType.PROJECT if hasattr(self.notebook, 'NoteType') else "project",
                tags=["#health_report", "#monitoring", "#system_status"]
            )
        
        return report
    
    def _format_metrics_table(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as a table."""
        
        table = "| Metric | Value | Status |\n|--------|-------|--------|\n"
        
        for name, metric in metrics.items():
            status_icon = {"normal": "✅", "warning": "⚠️", "critical": "❌"}.get(metric["status"], "❓")
            table += f"| {name.replace('_', ' ').title()} | {metric['value']:.1f}{metric['unit']} | {status_icon} {metric['status'].title()} |\n"
        
        return table
    
    def _format_alerts_table(self, alerts: List[ResearchAlert]) -> str:
        """Format alerts as a table."""
        
        if not alerts:
            return "No active alerts."
        
        table = "| Level | Component | Message | Time |\n|-------|-----------|---------|------|\n"
        
        for alert in alerts[:10]:  # Limit to 10 most recent
            level_icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}[alert.level.value]
            table += f"| {level_icon} {alert.level.value.title()} | {alert.component} | {alert.message} | {alert.timestamp.strftime('%H:%M')} |\n"
        
        if len(alerts) > 10:
            table += f"| ... | ... | +{len(alerts) - 10} more alerts | ... |\n"
        
        return table
    
    async def _generate_health_recommendations(
        self, 
        health_status: Dict[str, Any], 
        active_alerts: List[ResearchAlert]
    ) -> str:
        """Generate health improvement recommendations."""
        
        recommendations = []
        
        # System resource recommendations
        cpu_status = health_status["metrics"].get("cpu_usage", {}).get("status")
        memory_status = health_status["metrics"].get("memory_usage", {}).get("status")
        
        if cpu_status in ["warning", "critical"]:
            recommendations.append("🖥️ High CPU usage detected - consider closing unnecessary applications")
        
        if memory_status in ["warning", "critical"]:
            recommendations.append("💾 High memory usage - restart applications or increase system memory")
        
        # Research productivity recommendations
        note_rate = health_status["metrics"].get("note_creation_rate", {}).get("value", 0)
        if note_rate < 1.0:
            recommendations.append("📝 Low note creation rate - consider daily research journaling")
        
        writing_velocity = health_status["metrics"].get("writing_velocity", {}).get("value", 0)
        if writing_velocity < 50:
            recommendations.append("✍️ Low writing velocity - set daily writing goals")
        
        # Alert-based recommendations
        critical_alerts = [a for a in active_alerts if a.level == AlertLevel.CRITICAL]
        if critical_alerts:
            recommendations.append(f"🚨 {len(critical_alerts)} critical alerts require immediate attention")
        
        # Default recommendation
        if not recommendations:
            recommendations.append("👍 System health looks good - maintain current practices")
        
        return "\n".join([f"- {rec}" for rec in recommendations])
    
    def export_monitoring_data(self, days: int = 30) -> Dict[str, Any]:
        """Export monitoring data for external analysis."""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        return {
            "export_date": datetime.now().isoformat(),
            "period_days": days,
            "current_metrics": {
                name: {
                    "value": metric.value,
                    "unit": metric.unit,
                    "status": metric.status,
                    "thresholds": {
                        "warning": metric.threshold_warning,
                        "critical": metric.threshold_critical
                    }
                }
                for name, metric in self.metrics.items()
            },
            "productivity_trends": [
                {"timestamp": timestamp.isoformat(), "score": score}
                for timestamp, score in self.productivity_trends
                if timestamp > cutoff_date
            ],
            "recent_alerts": [
                {
                    "id": alert.id,
                    "level": alert.level.value,
                    "component": alert.component,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved,
                    "details": alert.details
                }
                for alert in self.alerts.values()
                if alert.timestamp > cutoff_date
            ]
        }