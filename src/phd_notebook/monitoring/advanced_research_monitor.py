"""
Advanced Research Monitoring System
Comprehensive monitoring for research activities, hypothesis tracking, and collaboration metrics.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, Counter
import statistics
import psutil
import time

from ..core.note import Note, NoteType
from ..research.hypothesis_testing_engine import HypothesisTestingEngine, HypothesisStatus
from ..research.collaboration_engine import CollaborationEngine, CollaborationStatus
from ..research.intelligent_paper_generator import IntelligentPaperGenerator, PaperStatus
from ..utils.logging import setup_logger


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    RESEARCH_PROGRESS = "research_progress"
    COLLABORATION_HEALTH = "collaboration_health"
    PAPER_GENERATION = "paper_generation"
    SYSTEM_PERFORMANCE = "system_performance"
    DATA_QUALITY = "data_quality"


@dataclass
class Alert:
    """Research monitoring alert."""
    alert_id: str
    level: AlertLevel
    metric_type: MetricType
    title: str
    description: str
    timestamp: datetime
    
    # Context
    affected_components: List[str]
    suggested_actions: List[str]
    threshold_breached: Optional[Dict[str, float]] = None
    
    # Resolution
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['level'] = self.level.value
        result['metric_type'] = self.metric_type.value
        result['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            result['resolved_at'] = self.resolved_at.isoformat()
        return result


@dataclass
class ResearchMetrics:
    """Comprehensive research performance metrics."""
    timestamp: datetime
    
    # Research Progress Metrics
    active_hypotheses: int
    validated_hypotheses: int
    rejected_hypotheses: int
    avg_hypothesis_confidence: float
    research_velocity: float  # hypotheses per day
    
    # Collaboration Metrics
    active_collaborations: int
    collaboration_response_rate: float
    avg_collaboration_quality: float
    network_growth_rate: float
    
    # Paper Generation Metrics
    papers_in_progress: int
    completed_papers: int
    avg_paper_quality: float
    generation_success_rate: float
    
    # System Performance
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    response_time_ms: float
    
    # Data Quality Metrics
    content_quality_score: float
    citation_accuracy_score: float
    duplicate_detection_score: float
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class AdvancedResearchMonitor:
    """
    Comprehensive monitoring system for research activities with intelligent
    alerting, performance tracking, and automated quality assurance.
    """
    
    def __init__(self, notebook_path: Path):
        self.logger = setup_logger("monitoring.research_monitor")
        self.notebook_path = notebook_path
        
        # Core components (will be injected)
        self.hypothesis_engine: Optional[HypothesisTestingEngine] = None
        self.collaboration_engine: Optional[CollaborationEngine] = None
        self.paper_generator: Optional[IntelligentPaperGenerator] = None
        
        # Monitoring data
        self.metrics_history: List[ResearchMetrics] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Configuration
        self.monitoring_interval = 300  # 5 minutes
        self.alert_thresholds = self._initialize_alert_thresholds()
        self.is_monitoring = False
        
        # Create directories
        self.monitoring_dir = notebook_path / "monitoring"
        self.metrics_dir = self.monitoring_dir / "metrics"
        self.alerts_dir = self.monitoring_dir / "alerts"
        self.reports_dir = self.monitoring_dir / "reports"
        
        for dir_path in [self.monitoring_dir, self.metrics_dir, self.alerts_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self._load_existing_data()
        
        # Start monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
    
    def _initialize_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize default alert thresholds."""
        return {
            "research_progress": {
                "min_hypothesis_confidence": 0.3,
                "min_research_velocity": 0.05,  # hypotheses per day
                "max_rejection_rate": 0.8
            },
            "collaboration_health": {
                "min_response_rate": 0.4,
                "min_collaboration_quality": 0.5,
                "max_inactive_days": 30
            },
            "paper_generation": {
                "min_quality_score": 0.6,
                "max_generation_failure_rate": 0.3,
                "min_completion_rate": 0.5
            },
            "system_performance": {
                "max_cpu_usage": 80.0,
                "max_memory_usage": 85.0,
                "max_response_time_ms": 5000
            },
            "data_quality": {
                "min_content_quality": 0.6,
                "min_citation_accuracy": 0.7,
                "max_duplicate_rate": 0.1
            }
        }
    
    def _load_existing_data(self):
        """Load existing monitoring data."""
        try:
            # Load metrics history
            for metric_file in self.metrics_dir.glob("*.json"):
                with open(metric_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert timestamp
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    metrics = ResearchMetrics(**data)
                    self.metrics_history.append(metrics)
            
            # Load alerts
            for alert_file in self.alerts_dir.glob("*.json"):
                with open(alert_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    alert = self._dict_to_alert(data)
                    if not alert.resolved:
                        self.active_alerts[alert.alert_id] = alert
                    self.alert_history.append(alert)
            
            # Sort by timestamp
            self.metrics_history.sort(key=lambda x: x.timestamp)
            self.alert_history.sort(key=lambda x: x.timestamp)
            
            self.logger.info(f"Loaded {len(self.metrics_history)} metrics, "
                           f"{len(self.active_alerts)} active alerts")
                           
        except Exception as e:
            self.logger.error(f"Error loading monitoring data: {e}")
    
    def _dict_to_alert(self, data: Dict) -> Alert:
        """Convert dictionary to Alert object."""
        data['level'] = AlertLevel(data['level'])
        data['metric_type'] = MetricType(data['metric_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('resolved_at'):
            data['resolved_at'] = datetime.fromisoformat(data['resolved_at'])
        return Alert(**data)
    
    def set_components(
        self,
        hypothesis_engine: HypothesisTestingEngine = None,
        collaboration_engine: CollaborationEngine = None,
        paper_generator: IntelligentPaperGenerator = None
    ):
        """Set the research components to monitor."""
        if hypothesis_engine:
            self.hypothesis_engine = hypothesis_engine
        if collaboration_engine:
            self.collaboration_engine = collaboration_engine
        if paper_generator:
            self.paper_generator = paper_generator
    
    async def start_monitoring(self):
        """Start the continuous monitoring process."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already running")
            return
        
        self.is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Research monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring process."""
        self.is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Research monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                self._save_metrics(metrics)
                
                # Check for alerts
                await self._check_alerts(metrics)
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Wait for next cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _collect_metrics(self) -> ResearchMetrics:
        """Collect comprehensive research metrics."""
        timestamp = datetime.now()
        
        # Research Progress Metrics
        research_metrics = self._collect_research_metrics()
        
        # Collaboration Metrics  
        collaboration_metrics = self._collect_collaboration_metrics()
        
        # Paper Generation Metrics
        paper_metrics = self._collect_paper_metrics()
        
        # System Performance Metrics
        system_metrics = self._collect_system_metrics()
        
        # Data Quality Metrics
        quality_metrics = self._collect_quality_metrics()
        
        return ResearchMetrics(
            timestamp=timestamp,
            **research_metrics,
            **collaboration_metrics,
            **paper_metrics,
            **system_metrics,
            **quality_metrics
        )
    
    def _collect_research_metrics(self) -> Dict[str, Any]:
        """Collect research progress metrics."""
        metrics = {
            "active_hypotheses": 0,
            "validated_hypotheses": 0,
            "rejected_hypotheses": 0,
            "avg_hypothesis_confidence": 0.0,
            "research_velocity": 0.0
        }
        
        if not self.hypothesis_engine:
            return metrics
        
        try:
            hypotheses = self.hypothesis_engine.hypotheses
            
            # Count by status
            active_count = len([h for h in hypotheses.values() 
                              if h.status in [HypothesisStatus.ACTIVE, HypothesisStatus.TESTING]])
            validated_count = len([h for h in hypotheses.values() 
                                 if h.status == HypothesisStatus.VALIDATED])
            rejected_count = len([h for h in hypotheses.values() 
                                if h.status == HypothesisStatus.REJECTED])
            
            # Average confidence
            if hypotheses:
                avg_confidence = sum(h.confidence_score for h in hypotheses.values()) / len(hypotheses)
            else:
                avg_confidence = 0.0
            
            # Research velocity (hypotheses created per day over last 30 days)
            cutoff_date = datetime.now() - timedelta(days=30)
            recent_hypotheses = [h for h in hypotheses.values() if h.created_at >= cutoff_date]
            velocity = len(recent_hypotheses) / 30.0
            
            metrics.update({
                "active_hypotheses": active_count,
                "validated_hypotheses": validated_count,
                "rejected_hypotheses": rejected_count,
                "avg_hypothesis_confidence": avg_confidence,
                "research_velocity": velocity
            })
            
        except Exception as e:
            self.logger.error(f"Error collecting research metrics: {e}")
        
        return metrics
    
    def _collect_collaboration_metrics(self) -> Dict[str, Any]:
        """Collect collaboration health metrics."""
        metrics = {
            "active_collaborations": 0,
            "collaboration_response_rate": 0.0,
            "avg_collaboration_quality": 0.0,
            "network_growth_rate": 0.0
        }
        
        if not self.collaboration_engine:
            return metrics
        
        try:
            collaborations = self.collaboration_engine.collaborations
            collaborators = self.collaboration_engine.collaborators
            interactions = self.collaboration_engine.interactions
            
            # Active collaborations
            active_count = len([c for c in collaborations.values() 
                              if c.status == CollaborationStatus.ACTIVE])
            
            # Response rate (interactions per collaboration)
            if collaborations:
                total_interactions = len(interactions)
                response_rate = total_interactions / len(collaborations)
            else:
                response_rate = 0.0
            
            # Average collaboration quality
            if collaborators:
                avg_quality = sum(c.collaboration_quality for c in collaborators.values()) / len(collaborators)
            else:
                avg_quality = 0.0
            
            # Network growth rate (new collaborators per month)
            cutoff_date = datetime.now() - timedelta(days=30)
            recent_collaborators = [c for c in collaborators.values() 
                                  if c.first_contact and c.first_contact >= cutoff_date]
            growth_rate = len(recent_collaborators) / 30.0 * 30  # per month
            
            metrics.update({
                "active_collaborations": active_count,
                "collaboration_response_rate": response_rate,
                "avg_collaboration_quality": avg_quality,
                "network_growth_rate": growth_rate
            })
            
        except Exception as e:
            self.logger.error(f"Error collecting collaboration metrics: {e}")
        
        return metrics
    
    def _collect_paper_metrics(self) -> Dict[str, Any]:
        """Collect paper generation metrics."""
        metrics = {
            "papers_in_progress": 0,
            "completed_papers": 0,
            "avg_paper_quality": 0.0,
            "generation_success_rate": 0.0
        }
        
        if not self.paper_generator:
            return metrics
        
        try:
            papers = self.paper_generator.generated_papers
            
            # Count by status
            in_progress_count = len([p for p in papers.values() 
                                   if p.status in [PaperStatus.DRAFT, PaperStatus.UNDER_REVIEW, 
                                                  PaperStatus.REVISING]])
            completed_count = len([p for p in papers.values() 
                                 if p.status in [PaperStatus.PUBLISHED, PaperStatus.READY_FOR_SUBMISSION]])
            
            # Average quality (composite of all quality scores)
            if papers:
                quality_scores = []
                for paper in papers.values():
                    composite_quality = (
                        paper.coherence_score + 
                        paper.novelty_score + 
                        paper.completeness_score + 
                        paper.citation_quality_score
                    ) / 4.0
                    quality_scores.append(composite_quality)
                avg_quality = sum(quality_scores) / len(quality_scores)
            else:
                avg_quality = 0.0
            
            # Success rate (non-rejected papers)
            if papers:
                successful_papers = len([p for p in papers.values() if p.status != PaperStatus.REJECTED])
                success_rate = successful_papers / len(papers)
            else:
                success_rate = 0.0
            
            metrics.update({
                "papers_in_progress": in_progress_count,
                "completed_papers": completed_count,
                "avg_paper_quality": avg_quality,
                "generation_success_rate": success_rate
            })
            
        except Exception as e:
            self.logger.error(f"Error collecting paper metrics: {e}")
        
        return metrics
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage(str(self.notebook_path))
            disk_usage = disk.percent
            
            # Response time (measure time to access research data)
            start_time = time.time()
            # Simple operation to test responsiveness
            list(self.notebook_path.glob("*.md"))
            response_time = (time.time() - start_time) * 1000  # ms
            
            return {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "disk_usage": disk_usage,
                "response_time_ms": response_time
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "disk_usage": 0.0,
                "response_time_ms": 0.0
            }
    
    def _collect_quality_metrics(self) -> Dict[str, Any]:
        """Collect data quality metrics."""
        metrics = {
            "content_quality_score": 0.0,
            "citation_accuracy_score": 0.0,
            "duplicate_detection_score": 0.0
        }
        
        try:
            # Content quality (based on available research content)
            if self.paper_generator:
                content_items = self.paper_generator.research_content.values()
                if content_items:
                    avg_content_quality = sum(c.quality_score for c in content_items) / len(content_items)
                    metrics["content_quality_score"] = avg_content_quality
            
            # Citation accuracy (based on paper generation)
            if self.paper_generator:
                papers = self.paper_generator.generated_papers.values()
                if papers:
                    avg_citation_quality = sum(p.citation_quality_score for p in papers) / len(papers)
                    metrics["citation_accuracy_score"] = avg_citation_quality
            
            # Duplicate detection (simplified - based on content titles)
            if self.paper_generator:
                content_items = list(self.paper_generator.research_content.values())
                if content_items:
                    titles = [c.title.lower() for c in content_items]
                    unique_titles = set(titles)
                    duplicate_rate = 1.0 - (len(unique_titles) / len(titles))
                    metrics["duplicate_detection_score"] = 1.0 - duplicate_rate  # Higher is better
            
        except Exception as e:
            self.logger.error(f"Error collecting quality metrics: {e}")
        
        return metrics
    
    async def _check_alerts(self, metrics: ResearchMetrics):
        """Check metrics against thresholds and generate alerts."""
        new_alerts = []
        
        # Research Progress Alerts
        research_alerts = self._check_research_alerts(metrics)
        new_alerts.extend(research_alerts)
        
        # Collaboration Alerts
        collaboration_alerts = self._check_collaboration_alerts(metrics)
        new_alerts.extend(collaboration_alerts)
        
        # Paper Generation Alerts
        paper_alerts = self._check_paper_alerts(metrics)
        new_alerts.extend(paper_alerts)
        
        # System Performance Alerts
        system_alerts = self._check_system_alerts(metrics)
        new_alerts.extend(system_alerts)
        
        # Data Quality Alerts
        quality_alerts = self._check_quality_alerts(metrics)
        new_alerts.extend(quality_alerts)
        
        # Process new alerts
        for alert in new_alerts:
            await self._process_alert(alert)
    
    def _check_research_alerts(self, metrics: ResearchMetrics) -> List[Alert]:
        """Check research progress metrics for alerts."""
        alerts = []
        thresholds = self.alert_thresholds["research_progress"]
        
        # Low hypothesis confidence
        if metrics.avg_hypothesis_confidence < thresholds["min_hypothesis_confidence"]:
            alerts.append(Alert(
                alert_id=f"research_confidence_{int(metrics.timestamp.timestamp())}",
                level=AlertLevel.WARNING,
                metric_type=MetricType.RESEARCH_PROGRESS,
                title="Low Hypothesis Confidence",
                description=f"Average hypothesis confidence is {metrics.avg_hypothesis_confidence:.1%}, "
                           f"below threshold of {thresholds['min_hypothesis_confidence']:.1%}",
                timestamp=metrics.timestamp,
                affected_components=["hypothesis_engine"],
                suggested_actions=[
                    "Review evidence quality for active hypotheses",
                    "Consider collecting additional supporting evidence",
                    "Re-evaluate hypothesis formulation"
                ],
                threshold_breached={"avg_hypothesis_confidence": metrics.avg_hypothesis_confidence}
            ))
        
        # Low research velocity
        if metrics.research_velocity < thresholds["min_research_velocity"]:
            alerts.append(Alert(
                alert_id=f"research_velocity_{int(metrics.timestamp.timestamp())}",
                level=AlertLevel.INFO,
                metric_type=MetricType.RESEARCH_PROGRESS,
                title="Low Research Velocity",
                description=f"Research velocity is {metrics.research_velocity:.3f} hypotheses/day, "
                           f"below threshold of {thresholds['min_research_velocity']:.3f}",
                timestamp=metrics.timestamp,
                affected_components=["research_planning"],
                suggested_actions=[
                    "Schedule dedicated research time",
                    "Break down complex research questions",
                    "Consider collaborative hypothesis development"
                ]
            ))
        
        # High rejection rate
        if metrics.rejected_hypotheses > 0:
            total_hypotheses = metrics.active_hypotheses + metrics.validated_hypotheses + metrics.rejected_hypotheses
            rejection_rate = metrics.rejected_hypotheses / max(total_hypotheses, 1)
            
            if rejection_rate > thresholds["max_rejection_rate"]:
                alerts.append(Alert(
                    alert_id=f"high_rejection_{int(metrics.timestamp.timestamp())}",
                    level=AlertLevel.WARNING,
                    metric_type=MetricType.RESEARCH_PROGRESS,
                    title="High Hypothesis Rejection Rate",
                    description=f"Hypothesis rejection rate is {rejection_rate:.1%}, "
                               f"above threshold of {thresholds['max_rejection_rate']:.1%}",
                    timestamp=metrics.timestamp,
                    affected_components=["hypothesis_formulation"],
                    suggested_actions=[
                        "Review hypothesis formulation methodology",
                        "Consider pilot studies before formal hypothesis testing",
                        "Seek expert feedback on research design"
                    ]
                ))
        
        return alerts
    
    def _check_collaboration_alerts(self, metrics: ResearchMetrics) -> List[Alert]:
        """Check collaboration metrics for alerts."""
        alerts = []
        thresholds = self.alert_thresholds["collaboration_health"]
        
        # Low response rate
        if metrics.collaboration_response_rate < thresholds["min_response_rate"]:
            alerts.append(Alert(
                alert_id=f"collaboration_response_{int(metrics.timestamp.timestamp())}",
                level=AlertLevel.WARNING,
                metric_type=MetricType.COLLABORATION_HEALTH,
                title="Low Collaboration Response Rate",
                description=f"Collaboration response rate is {metrics.collaboration_response_rate:.2f}, "
                           f"below threshold of {thresholds['min_response_rate']:.2f}",
                timestamp=metrics.timestamp,
                affected_components=["collaboration_engine"],
                suggested_actions=[
                    "Follow up with inactive collaborators",
                    "Review communication strategies",
                    "Consider scheduling regular check-ins"
                ]
            ))
        
        # Low collaboration quality
        if metrics.avg_collaboration_quality < thresholds["min_collaboration_quality"]:
            alerts.append(Alert(
                alert_id=f"collaboration_quality_{int(metrics.timestamp.timestamp())}",
                level=AlertLevel.INFO,
                metric_type=MetricType.COLLABORATION_HEALTH,
                title="Low Collaboration Quality",
                description=f"Average collaboration quality is {metrics.avg_collaboration_quality:.1%}, "
                           f"below threshold of {thresholds['min_collaboration_quality']:.1%}",
                timestamp=metrics.timestamp,
                affected_components=["collaboration_management"],
                suggested_actions=[
                    "Evaluate collaboration effectiveness",
                    "Improve meeting structure and outcomes",
                    "Set clear collaboration goals and expectations"
                ]
            ))
        
        return alerts
    
    def _check_paper_alerts(self, metrics: ResearchMetrics) -> List[Alert]:
        """Check paper generation metrics for alerts."""
        alerts = []
        thresholds = self.alert_thresholds["paper_generation"]
        
        # Low paper quality
        if metrics.avg_paper_quality < thresholds["min_quality_score"]:
            alerts.append(Alert(
                alert_id=f"paper_quality_{int(metrics.timestamp.timestamp())}",
                level=AlertLevel.WARNING,
                metric_type=MetricType.PAPER_GENERATION,
                title="Low Paper Quality Score",
                description=f"Average paper quality is {metrics.avg_paper_quality:.1%}, "
                           f"below threshold of {thresholds['min_quality_score']:.1%}",
                timestamp=metrics.timestamp,
                affected_components=["paper_generator"],
                suggested_actions=[
                    "Review and improve content quality",
                    "Enhance research methodology documentation",
                    "Strengthen literature review and citations"
                ]
            ))
        
        # Low generation success rate
        if metrics.generation_success_rate < thresholds["min_completion_rate"]:
            alerts.append(Alert(
                alert_id=f"generation_success_{int(metrics.timestamp.timestamp())}",
                level=AlertLevel.INFO,
                metric_type=MetricType.PAPER_GENERATION,
                title="Low Paper Generation Success Rate",
                description=f"Paper generation success rate is {metrics.generation_success_rate:.1%}, "
                           f"below threshold of {thresholds['min_completion_rate']:.1%}",
                timestamp=metrics.timestamp,
                affected_components=["paper_workflow"],
                suggested_actions=[
                    "Review paper generation process",
                    "Improve content preparation and organization",
                    "Consider template optimization"
                ]
            ))
        
        return alerts
    
    def _check_system_alerts(self, metrics: ResearchMetrics) -> List[Alert]:
        """Check system performance metrics for alerts."""
        alerts = []
        thresholds = self.alert_thresholds["system_performance"]
        
        # High CPU usage
        if metrics.cpu_usage > thresholds["max_cpu_usage"]:
            alerts.append(Alert(
                alert_id=f"high_cpu_{int(metrics.timestamp.timestamp())}",
                level=AlertLevel.WARNING,
                metric_type=MetricType.SYSTEM_PERFORMANCE,
                title="High CPU Usage",
                description=f"CPU usage is {metrics.cpu_usage:.1f}%, "
                           f"above threshold of {thresholds['max_cpu_usage']:.1f}%",
                timestamp=metrics.timestamp,
                affected_components=["system_resources"],
                suggested_actions=[
                    "Check for resource-intensive processes",
                    "Consider optimizing research workflows",
                    "Monitor for background tasks"
                ]
            ))
        
        # High memory usage
        if metrics.memory_usage > thresholds["max_memory_usage"]:
            alerts.append(Alert(
                alert_id=f"high_memory_{int(metrics.timestamp.timestamp())}",
                level=AlertLevel.WARNING,
                metric_type=MetricType.SYSTEM_PERFORMANCE,
                title="High Memory Usage",
                description=f"Memory usage is {metrics.memory_usage:.1f}%, "
                           f"above threshold of {thresholds['max_memory_usage']:.1f}%",
                timestamp=metrics.timestamp,
                affected_components=["system_resources"],
                suggested_actions=[
                    "Clear cache and temporary files",
                    "Optimize data processing workflows",
                    "Consider increasing system memory"
                ]
            ))
        
        # High response time
        if metrics.response_time_ms > thresholds["max_response_time_ms"]:
            alerts.append(Alert(
                alert_id=f"slow_response_{int(metrics.timestamp.timestamp())}",
                level=AlertLevel.INFO,
                metric_type=MetricType.SYSTEM_PERFORMANCE,
                title="Slow System Response",
                description=f"System response time is {metrics.response_time_ms:.0f}ms, "
                           f"above threshold of {thresholds['max_response_time_ms']:.0f}ms",
                timestamp=metrics.timestamp,
                affected_components=["system_performance"],
                suggested_actions=[
                    "Check disk I/O performance",
                    "Optimize file organization",
                    "Consider SSD storage upgrade"
                ]
            ))
        
        return alerts
    
    def _check_quality_alerts(self, metrics: ResearchMetrics) -> List[Alert]:
        """Check data quality metrics for alerts."""
        alerts = []
        thresholds = self.alert_thresholds["data_quality"]
        
        # Low content quality
        if metrics.content_quality_score < thresholds["min_content_quality"]:
            alerts.append(Alert(
                alert_id=f"content_quality_{int(metrics.timestamp.timestamp())}",
                level=AlertLevel.INFO,
                metric_type=MetricType.DATA_QUALITY,
                title="Low Content Quality Score",
                description=f"Content quality score is {metrics.content_quality_score:.1%}, "
                           f"below threshold of {thresholds['min_content_quality']:.1%}",
                timestamp=metrics.timestamp,
                affected_components=["content_processing"],
                suggested_actions=[
                    "Review content creation process",
                    "Improve data collection quality",
                    "Enhance content validation procedures"
                ]
            ))
        
        # Low citation accuracy
        if metrics.citation_accuracy_score < thresholds["min_citation_accuracy"]:
            alerts.append(Alert(
                alert_id=f"citation_accuracy_{int(metrics.timestamp.timestamp())}",
                level=AlertLevel.WARNING,
                metric_type=MetricType.DATA_QUALITY,
                title="Low Citation Accuracy",
                description=f"Citation accuracy score is {metrics.citation_accuracy_score:.1%}, "
                           f"below threshold of {thresholds['min_citation_accuracy']:.1%}",
                timestamp=metrics.timestamp,
                affected_components=["citation_management"],
                suggested_actions=[
                    "Verify citation sources and formatting",
                    "Update reference management system",
                    "Review automated citation processes"
                ]
            ))
        
        return alerts
    
    async def _process_alert(self, alert: Alert):
        """Process a new alert."""
        # Check if similar alert already exists
        similar_alert = self._find_similar_alert(alert)
        if similar_alert:
            self.logger.debug(f"Similar alert already exists: {similar_alert.alert_id}")
            return
        
        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Save alert
        self._save_alert(alert)
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }.get(alert.level, logging.INFO)
        
        self.logger.log(log_level, f"ALERT: {alert.title} - {alert.description}")
        
        # Create alert note
        self._create_alert_note(alert)
    
    def _find_similar_alert(self, alert: Alert) -> Optional[Alert]:
        """Find similar active alert to avoid duplicates."""
        for existing_alert in self.active_alerts.values():
            if (existing_alert.metric_type == alert.metric_type and
                existing_alert.title == alert.title and
                (alert.timestamp - existing_alert.timestamp).seconds < 3600):  # Within 1 hour
                return existing_alert
        return None
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """Manually resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            alert.resolution_notes = resolution_notes
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            # Save updated alert
            self._save_alert(alert)
            
            self.logger.info(f"Alert resolved: {alert_id}")
        else:
            self.logger.warning(f"Alert not found: {alert_id}")
    
    def get_monitoring_dashboard(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive monitoring dashboard."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter recent data
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_date]
        recent_alerts = [a for a in self.alert_history if a.timestamp >= cutoff_date]
        
        dashboard = {
            "period": f"Last {days} days",
            "generated_at": datetime.now().isoformat(),
            "overview": self._generate_overview(recent_metrics),
            "alerts_summary": self._generate_alerts_summary(recent_alerts),
            "trends": self._generate_trends(recent_metrics),
            "recommendations": self._generate_recommendations(),
            "active_alerts": len(self.active_alerts),
            "system_health": self._assess_system_health(recent_metrics)
        }
        
        return dashboard
    
    def _generate_overview(self, metrics: List[ResearchMetrics]) -> Dict[str, Any]:
        """Generate overview statistics."""
        if not metrics:
            return {"error": "No metrics available"}
        
        latest = metrics[-1]
        
        return {
            "current_status": {
                "active_hypotheses": latest.active_hypotheses,
                "active_collaborations": latest.active_collaborations,
                "papers_in_progress": latest.papers_in_progress,
                "avg_research_confidence": latest.avg_hypothesis_confidence,
                "system_performance": "good" if latest.cpu_usage < 70 else "warning"
            },
            "totals": {
                "validated_hypotheses": latest.validated_hypotheses,
                "completed_papers": latest.completed_papers,
                "total_collaborations": latest.active_collaborations + 
                                       (latest.completed_papers if hasattr(latest, 'completed_collaborations') else 0)
            }
        }
    
    def _generate_alerts_summary(self, alerts: List[Alert]) -> Dict[str, Any]:
        """Generate alerts summary."""
        alert_counts = Counter(alert.level.value for alert in alerts)
        alert_types = Counter(alert.metric_type.value for alert in alerts)
        
        return {
            "total_alerts": len(alerts),
            "by_level": dict(alert_counts),
            "by_type": dict(alert_types),
            "resolution_rate": len([a for a in alerts if a.resolved]) / max(len(alerts), 1)
        }
    
    def _generate_trends(self, metrics: List[ResearchMetrics]) -> Dict[str, Any]:
        """Generate trend analysis."""
        if len(metrics) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Calculate trends
        def calculate_trend(values):
            if len(values) < 2:
                return 0.0
            return (values[-1] - values[0]) / max(abs(values[0]), 0.001)
        
        research_velocity_trend = calculate_trend([m.research_velocity for m in metrics])
        confidence_trend = calculate_trend([m.avg_hypothesis_confidence for m in metrics])
        collaboration_trend = calculate_trend([m.active_collaborations for m in metrics])
        
        return {
            "research_velocity": "improving" if research_velocity_trend > 0.1 else "stable" if abs(research_velocity_trend) < 0.1 else "declining",
            "hypothesis_confidence": "improving" if confidence_trend > 0.05 else "stable" if abs(confidence_trend) < 0.05 else "declining",
            "collaboration_activity": "growing" if collaboration_trend > 0.1 else "stable" if abs(collaboration_trend) < 0.1 else "declining"
        }
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate system recommendations based on monitoring data."""
        recommendations = []
        
        # Analyze recent alerts for patterns
        recent_alerts = [a for a in self.alert_history[-10:]]  # Last 10 alerts
        alert_types = Counter(a.metric_type.value for a in recent_alerts)
        
        # Research progress recommendations
        if alert_types.get("research_progress", 0) >= 2:
            recommendations.append({
                "category": "Research Progress",
                "priority": "high",
                "recommendation": "Focus on hypothesis development and validation",
                "action": "Schedule dedicated research planning sessions"
            })
        
        # Collaboration recommendations
        if alert_types.get("collaboration_health", 0) >= 2:
            recommendations.append({
                "category": "Collaboration",
                "priority": "medium",
                "recommendation": "Improve collaboration management",
                "action": "Review and optimize communication strategies"
            })
        
        # System performance recommendations
        if alert_types.get("system_performance", 0) >= 2:
            recommendations.append({
                "category": "System Performance",
                "priority": "high",
                "recommendation": "Optimize system resources",
                "action": "Review and cleanup system processes"
            })
        
        # Data quality recommendations
        if alert_types.get("data_quality", 0) >= 2:
            recommendations.append({
                "category": "Data Quality",
                "priority": "medium",
                "recommendation": "Enhance data validation processes",
                "action": "Implement stricter quality control measures"
            })
        
        return recommendations
    
    def _assess_system_health(self, metrics: List[ResearchMetrics]) -> str:
        """Assess overall system health."""
        if not metrics:
            return "unknown"
        
        latest = metrics[-1]
        health_score = 0
        
        # Research health (25%)
        if latest.avg_hypothesis_confidence > 0.6:
            health_score += 25
        elif latest.avg_hypothesis_confidence > 0.4:
            health_score += 15
        
        # Collaboration health (25%)
        if latest.collaboration_response_rate > 0.7:
            health_score += 25
        elif latest.collaboration_response_rate > 0.5:
            health_score += 15
        
        # System performance (25%)
        if latest.cpu_usage < 60 and latest.memory_usage < 70:
            health_score += 25
        elif latest.cpu_usage < 80 and latest.memory_usage < 85:
            health_score += 15
        
        # Data quality (25%)
        if latest.content_quality_score > 0.7:
            health_score += 25
        elif latest.content_quality_score > 0.5:
            health_score += 15
        
        if health_score >= 80:
            return "excellent"
        elif health_score >= 60:
            return "good"
        elif health_score >= 40:
            return "fair"
        else:
            return "poor"
    
    def _cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old monitoring data."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Remove old metrics
        self.metrics_history = [m for m in self.metrics_history if m.timestamp >= cutoff_date]
        
        # Remove old resolved alerts
        self.alert_history = [a for a in self.alert_history 
                             if a.timestamp >= cutoff_date or not a.resolved]
    
    def _create_alert_note(self, alert: Alert):
        """Create an Obsidian note for the alert."""
        note_path = self.alerts_dir / f"{alert.alert_id}.md"
        
        content = f"""# Alert: {alert.title}

## Alert Details
- **Level**: {alert.level.value.upper()}
- **Type**: {alert.metric_type.value.replace('_', ' ').title()}
- **Timestamp**: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- **Status**: {'Resolved' if alert.resolved else 'Active'}

## Description
{alert.description}

## Affected Components
{chr(10).join(f"- {comp}" for comp in alert.affected_components)}

## Suggested Actions
{chr(10).join(f"- [ ] {action}" for action in alert.suggested_actions)}

## Threshold Information
{json.dumps(alert.threshold_breached, indent=2) if alert.threshold_breached else 'N/A'}

## Resolution
{f"**Resolved At**: {alert.resolved_at.strftime('%Y-%m-%d %H:%M:%S')}" if alert.resolved_at else "Not yet resolved"}
{f"**Resolution Notes**: {alert.resolution_notes}" if alert.resolution_notes else ""}

---
*Alert ID: {alert.alert_id}*
"""
        
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _save_metrics(self, metrics: ResearchMetrics):
        """Save metrics to JSON storage."""
        timestamp_str = metrics.timestamp.strftime('%Y%m%d_%H%M%S')
        file_path = self.metrics_dir / f"metrics_{timestamp_str}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _save_alert(self, alert: Alert):
        """Save alert to JSON storage."""
        file_path = self.alerts_dir / f"{alert.alert_id}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(alert.to_dict(), f, indent=2, ensure_ascii=False)