"""
Advanced Research Intelligence - Generation 2 Enhancement
Intelligent monitoring and analytics for research processes with predictive capabilities.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
from enum import Enum
from collections import defaultdict, deque
import statistics

# Import numpy with fallback
try:
    import numpy as np
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.fallbacks import np
from utils.fallbacks import sklearn
import pickle
from sklearn.base import BaseEstimator
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class IntelligenceType(Enum):
    """Types of research intelligence."""
    PREDICTIVE = "predictive"
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PRESCRIPTIVE = "prescriptive"
    ADAPTIVE = "adaptive"


class MetricCategory(Enum):
    """Categories of research metrics."""
    PRODUCTIVITY = "productivity"
    QUALITY = "quality"
    IMPACT = "impact"
    EFFICIENCY = "efficiency"
    COLLABORATION = "collaboration"
    INNOVATION = "innovation"
    RESOURCE_UTILIZATION = "resource_utilization"
    TIMELINE = "timeline"


class AlertPriority(Enum):
    """Alert priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ResearchMetric:
    """Research performance metric."""
    metric_id: str
    name: str
    category: MetricCategory
    value: float
    target_value: Optional[float]
    unit: str
    timestamp: datetime
    source: str
    context: Dict[str, Any]
    trend_direction: str = "stable"  # "increasing", "decreasing", "stable"
    confidence: float = 1.0
    
    
@dataclass
class IntelligenceInsight:
    """AI-generated research insight."""
    insight_id: str
    title: str
    description: str
    intelligence_type: IntelligenceType
    confidence: float
    impact_level: str  # "high", "medium", "low"
    evidence: List[str]
    recommendations: List[str]
    affected_areas: List[str]
    timestamp: datetime
    validity_period: timedelta
    actionable: bool = True


@dataclass
class PredictiveModel:
    """Predictive model for research outcomes."""
    model_id: str
    name: str
    model_type: str
    target_variable: str
    features: List[str]
    accuracy: float
    last_trained: datetime
    predictions: Dict[str, Any]
    confidence_intervals: Dict[str, Tuple[float, float]]
    feature_importance: Dict[str, float]


@dataclass
class ResearchAlert:
    """Research monitoring alert."""
    alert_id: str
    title: str
    description: str
    priority: AlertPriority
    category: str
    triggered_by: str
    threshold: float
    actual_value: float
    timestamp: datetime
    resolved: bool = False
    resolution_actions: List[str] = None
    
    def __post_init__(self):
        if self.resolution_actions is None:
            self.resolution_actions = []


class AdvancedResearchIntelligence:
    """
    Advanced AI-driven research intelligence system.
    
    Features:
    - Predictive analytics for research outcomes
    - Real-time performance monitoring
    - Intelligent alerting system
    - Adaptive learning from research patterns
    - Cross-project insight generation
    - Resource optimization recommendations
    """
    
    def __init__(self, notebook_context=None):
        self.intelligence_id = f"ari_{uuid.uuid4().hex[:8]}"
        self.notebook_context = notebook_context
        
        # Intelligence components
        self.metrics_store: Dict[str, List[ResearchMetric]] = defaultdict(list)
        self.insights: Dict[str, IntelligenceInsight] = {}
        self.predictive_models: Dict[str, PredictiveModel] = {}
        self.active_alerts: Dict[str, ResearchAlert] = {}
        
        # AI engines
        self.predictive_engine = PredictiveAnalyticsEngine()
        self.pattern_detector = PatternDetectionEngine()
        self.insight_generator = InsightGenerationEngine()
        self.anomaly_detector = AnomalyDetectionEngine()
        self.recommendation_engine = RecommendationEngine()
        
        # Monitoring configuration
        self.monitoring_config = {
            "collection_interval": 300,  # 5 minutes
            "prediction_horizon": 30,    # days
            "alert_thresholds": {},
            "learning_rate": 0.1,
            "confidence_threshold": 0.7
        }
        
        # Intelligence metrics
        self.intelligence_metrics = {
            "predictions_made": 0,
            "prediction_accuracy": 0.0,
            "insights_generated": 0,
            "alerts_triggered": 0,
            "patterns_detected": 0,
            "recommendations_provided": 0,
            "learning_iterations": 0
        }
        
        # Initialize default models
        self._initialize_predictive_models()
        
        logger.info(f"Initialized Advanced Research Intelligence: {self.intelligence_id}")
    
    async def start_intelligent_monitoring(self) -> None:
        """Start intelligent research monitoring."""
        logger.info("Starting intelligent research monitoring")
        
        monitoring_tasks = [
            self._metrics_collection_loop(),
            self._predictive_analysis_loop(),
            self._pattern_detection_loop(),
            self._insight_generation_loop(),
            self._anomaly_monitoring_loop(),
            self._adaptive_learning_loop()
        ]
        
        await asyncio.gather(*monitoring_tasks, return_exceptions=True)
    
    async def generate_predictive_insights(self, 
                                         target_outcomes: List[str],
                                         time_horizon: int = 30) -> Dict[str, Any]:
        """Generate predictive insights for research outcomes."""
        try:
            predictions = {}
            
            for outcome in target_outcomes:
                # Get relevant model
                model = self.predictive_models.get(f"{outcome}_model")
                if not model:
                    model = await self._create_predictive_model(outcome)
                
                # Generate predictions
                prediction = await self.predictive_engine.predict_outcome(
                    model, time_horizon, self.metrics_store
                )
                
                predictions[outcome] = {
                    "predicted_value": prediction["value"],
                    "confidence_interval": prediction["confidence_interval"],
                    "probability_of_success": prediction["success_probability"],
                    "key_factors": prediction["influential_factors"],
                    "recommended_actions": await self._get_recommendations_for_prediction(prediction),
                    "timeline_forecast": prediction["timeline_forecast"]
                }
                
                self.intelligence_metrics["predictions_made"] += 1
            
            # Generate meta-insights about predictions
            meta_insights = await self._generate_meta_prediction_insights(predictions)
            
            return {
                "predictions": predictions,
                "meta_insights": meta_insights,
                "prediction_confidence": self._calculate_overall_confidence(predictions),
                "generated_at": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate predictive insights: {e}")
            return {}
    
    async def detect_research_patterns(self, 
                                     analysis_window: int = 90) -> List[Dict[str, Any]]:
        """Detect patterns in research data."""
        try:
            patterns = await self.pattern_detector.detect_patterns(
                self.metrics_store, analysis_window
            )
            
            validated_patterns = []
            for pattern in patterns:
                # Validate pattern significance
                validation = await self._validate_pattern_significance(pattern)
                
                if validation["significant"]:
                    pattern_insight = {
                        "pattern_id": f"pattern_{uuid.uuid4().hex[:8]}",
                        "type": pattern["type"],
                        "description": pattern["description"],
                        "strength": pattern["strength"],
                        "confidence": validation["confidence"],
                        "affected_metrics": pattern["affected_metrics"],
                        "temporal_characteristics": pattern["temporal_info"],
                        "implications": await self._analyze_pattern_implications(pattern),
                        "actionable_insights": await self._generate_pattern_recommendations(pattern)
                    }
                    validated_patterns.append(pattern_insight)
            
            self.intelligence_metrics["patterns_detected"] += len(validated_patterns)
            
            return validated_patterns
            
        except Exception as e:
            logger.error(f"Failed to detect research patterns: {e}")
            return []
    
    async def generate_adaptive_recommendations(self, 
                                              current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate adaptive recommendations based on current research context."""
        try:
            recommendations = await self.recommendation_engine.generate_recommendations(
                current_context, self.metrics_store, self.insights
            )
            
            prioritized_recommendations = []
            for rec in recommendations:
                # Calculate recommendation priority and feasibility
                priority_score = await self._calculate_recommendation_priority(rec)
                feasibility_score = await self._assess_recommendation_feasibility(rec)
                
                if priority_score > 0.6 and feasibility_score > 0.5:
                    enhanced_rec = {
                        "recommendation_id": f"rec_{uuid.uuid4().hex[:8]}",
                        "title": rec["title"],
                        "description": rec["description"],
                        "category": rec["category"],
                        "priority_score": priority_score,
                        "feasibility_score": feasibility_score,
                        "expected_impact": rec["expected_impact"],
                        "implementation_steps": rec["implementation_steps"],
                        "resource_requirements": rec["resource_requirements"],
                        "timeline": rec["timeline"],
                        "success_metrics": rec["success_metrics"],
                        "risk_factors": rec.get("risk_factors", [])
                    }
                    prioritized_recommendations.append(enhanced_rec)
            
            # Sort by combined priority and feasibility
            prioritized_recommendations.sort(
                key=lambda x: x["priority_score"] * x["feasibility_score"], 
                reverse=True
            )
            
            self.intelligence_metrics["recommendations_provided"] += len(prioritized_recommendations)
            
            return prioritized_recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate adaptive recommendations: {e}")
            return []
    
    async def monitor_research_health(self) -> Dict[str, Any]:
        """Monitor overall research health and generate alerts."""
        try:
            health_assessment = {
                "overall_score": 0.0,
                "category_scores": {},
                "active_alerts": [],
                "health_trends": {},
                "risk_factors": [],
                "improvement_opportunities": []
            }
            
            # Assess health by category
            category_scores = {}
            for category in MetricCategory:
                category_metrics = await self._get_category_metrics(category)
                if category_metrics:
                    score = await self._calculate_category_health_score(category_metrics)
                    category_scores[category.value] = score
            
            health_assessment["category_scores"] = category_scores
            health_assessment["overall_score"] = statistics.mean(category_scores.values()) if category_scores else 0.0
            
            # Check for alerts
            new_alerts = await self._check_alert_conditions(category_scores)
            for alert in new_alerts:
                self.active_alerts[alert.alert_id] = alert
                health_assessment["active_alerts"].append(asdict(alert))
            
            # Analyze health trends
            health_trends = await self._analyze_health_trends()
            health_assessment["health_trends"] = health_trends
            
            # Identify risk factors
            risk_factors = await self._identify_health_risk_factors(category_scores)
            health_assessment["risk_factors"] = risk_factors
            
            # Generate improvement opportunities
            improvements = await self._identify_improvement_opportunities(category_scores)
            health_assessment["improvement_opportunities"] = improvements
            
            return health_assessment
            
        except Exception as e:
            logger.error(f"Failed to monitor research health: {e}")
            return {}
    
    async def adaptive_threshold_learning(self) -> None:
        """Adaptively learn and adjust monitoring thresholds."""
        try:
            for metric_name, metric_history in self.metrics_store.items():
                if len(metric_history) >= 30:  # Need sufficient data
                    # Analyze metric distribution
                    values = [m.value for m in metric_history[-30:]]  # Last 30 measurements
                    
                    # Calculate adaptive thresholds
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values) if len(values) > 1 else 0
                    
                    # Set dynamic thresholds based on statistical properties
                    lower_threshold = mean_val - 2 * std_val
                    upper_threshold = mean_val + 2 * std_val
                    
                    # Update alert thresholds
                    self.monitoring_config["alert_thresholds"][metric_name] = {
                        "lower": lower_threshold,
                        "upper": upper_threshold,
                        "last_updated": datetime.now(),
                        "confidence": min(1.0, len(values) / 30.0)
                    }
            
            logger.info("Updated adaptive thresholds for monitoring")
            
        except Exception as e:
            logger.error(f"Failed adaptive threshold learning: {e}")
    
    async def cross_project_intelligence(self, 
                                       project_contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate intelligence across multiple research projects."""
        try:
            cross_project_insights = {
                "comparative_analysis": {},
                "shared_patterns": [],
                "resource_sharing_opportunities": [],
                "collaboration_recommendations": [],
                "best_practices": [],
                "risk_correlations": []
            }
            
            # Comparative analysis
            comparative = await self._perform_comparative_analysis(project_contexts)
            cross_project_insights["comparative_analysis"] = comparative
            
            # Identify shared patterns across projects
            shared_patterns = await self._find_shared_patterns(project_contexts)
            cross_project_insights["shared_patterns"] = shared_patterns
            
            # Resource sharing opportunities
            sharing_opportunities = await self._identify_resource_sharing_opportunities(project_contexts)
            cross_project_insights["resource_sharing_opportunities"] = sharing_opportunities
            
            # Collaboration recommendations
            collaborations = await self._recommend_cross_project_collaborations(project_contexts)
            cross_project_insights["collaboration_recommendations"] = collaborations
            
            # Extract best practices
            best_practices = await self._extract_cross_project_best_practices(project_contexts)
            cross_project_insights["best_practices"] = best_practices
            
            return cross_project_insights
            
        except Exception as e:
            logger.error(f"Failed cross-project intelligence analysis: {e}")
            return {}
    
    async def _metrics_collection_loop(self) -> None:
        """Continuous metrics collection loop."""
        while True:
            try:
                # Collect current metrics
                current_metrics = await self._collect_current_metrics()
                
                # Store metrics with timestamp
                timestamp = datetime.now()
                for metric_name, metric_value in current_metrics.items():
                    metric = ResearchMetric(
                        metric_id=f"metric_{uuid.uuid4().hex[:8]}",
                        name=metric_name,
                        category=self._categorize_metric(metric_name),
                        value=metric_value,
                        target_value=None,
                        unit=self._get_metric_unit(metric_name),
                        timestamp=timestamp,
                        source="intelligent_monitoring",
                        context={"collection_method": "automated"}
                    )
                    
                    self.metrics_store[metric_name].append(metric)
                    
                    # Keep only recent metrics (sliding window)
                    if len(self.metrics_store[metric_name]) > 1000:
                        self.metrics_store[metric_name] = self.metrics_store[metric_name][-1000:]
                
                await asyncio.sleep(self.monitoring_config["collection_interval"])
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)
    
    async def _predictive_analysis_loop(self) -> None:
        """Continuous predictive analysis loop."""
        while True:
            try:
                # Run predictive analysis every hour
                target_outcomes = ["research_productivity", "project_success", "collaboration_effectiveness"]
                predictions = await self.generate_predictive_insights(target_outcomes)
                
                # Update models with recent data
                for model_name, model in self.predictive_models.items():
                    await self._update_predictive_model(model)
                
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in predictive analysis loop: {e}")
                await asyncio.sleep(1800)  # 30 minutes on error
    
    async def _pattern_detection_loop(self) -> None:
        """Continuous pattern detection loop."""
        while True:
            try:
                # Detect patterns every 30 minutes
                patterns = await self.detect_research_patterns()
                
                # Generate insights from patterns
                for pattern in patterns:
                    insight = await self._convert_pattern_to_insight(pattern)
                    self.insights[insight.insight_id] = insight
                
                await asyncio.sleep(1800)  # 30 minutes
                
            except Exception as e:
                logger.error(f"Error in pattern detection loop: {e}")
                await asyncio.sleep(900)  # 15 minutes on error
    
    async def _insight_generation_loop(self) -> None:
        """Continuous insight generation loop."""
        while True:
            try:
                # Generate insights every 2 hours
                new_insights = await self.insight_generator.generate_insights(
                    self.metrics_store, self.predictive_models
                )
                
                for insight in new_insights:
                    self.insights[insight.insight_id] = insight
                
                # Clean up old insights
                await self._cleanup_expired_insights()
                
                self.intelligence_metrics["insights_generated"] += len(new_insights)
                
                await asyncio.sleep(7200)  # 2 hours
                
            except Exception as e:
                logger.error(f"Error in insight generation loop: {e}")
                await asyncio.sleep(3600)  # 1 hour on error
    
    async def _anomaly_monitoring_loop(self) -> None:
        """Continuous anomaly monitoring loop."""
        while True:
            try:
                # Check for anomalies every 10 minutes
                anomalies = await self.anomaly_detector.detect_anomalies(self.metrics_store)
                
                # Generate alerts for significant anomalies
                for anomaly in anomalies:
                    if anomaly["severity"] >= 0.7:  # High severity threshold
                        alert = ResearchAlert(
                            alert_id=f"alert_{uuid.uuid4().hex[:8]}",
                            title=f"Anomaly detected: {anomaly['metric']}",
                            description=anomaly["description"],
                            priority=AlertPriority.HIGH if anomaly["severity"] > 0.8 else AlertPriority.MEDIUM,
                            category="anomaly",
                            triggered_by=anomaly["metric"],
                            threshold=anomaly["expected_value"],
                            actual_value=anomaly["actual_value"],
                            timestamp=datetime.now()
                        )
                        
                        self.active_alerts[alert.alert_id] = alert
                        self.intelligence_metrics["alerts_triggered"] += 1
                
                await asyncio.sleep(600)  # 10 minutes
                
            except Exception as e:
                logger.error(f"Error in anomaly monitoring loop: {e}")
                await asyncio.sleep(300)  # 5 minutes on error
    
    async def _adaptive_learning_loop(self) -> None:
        """Continuous adaptive learning loop."""
        while True:
            try:
                # Adaptive learning every 6 hours
                await self.adaptive_threshold_learning()
                
                # Update model performance metrics
                await self._update_model_performance_metrics()
                
                # Retrain models if necessary
                await self._adaptive_model_retraining()
                
                self.intelligence_metrics["learning_iterations"] += 1
                
                await asyncio.sleep(21600)  # 6 hours
                
            except Exception as e:
                logger.error(f"Error in adaptive learning loop: {e}")
                await asyncio.sleep(10800)  # 3 hours on error
    
    def _initialize_predictive_models(self) -> None:
        """Initialize default predictive models."""
        # Research productivity model
        productivity_model = PredictiveModel(
            model_id="productivity_model",
            name="Research Productivity Predictor",
            model_type="ensemble",
            target_variable="productivity_score",
            features=["hours_worked", "papers_written", "experiments_completed", "collaborations"],
            accuracy=0.75,  # Placeholder
            last_trained=datetime.now(),
            predictions={},
            confidence_intervals={},
            feature_importance={"hours_worked": 0.3, "papers_written": 0.4, "experiments_completed": 0.2, "collaborations": 0.1}
        )
        
        # Project success model
        success_model = PredictiveModel(
            model_id="success_model",
            name="Project Success Predictor",
            model_type="classification",
            target_variable="project_success",
            features=["resource_allocation", "team_size", "timeline_adherence", "quality_metrics"],
            accuracy=0.82,  # Placeholder
            last_trained=datetime.now(),
            predictions={},
            confidence_intervals={},
            feature_importance={"resource_allocation": 0.25, "team_size": 0.2, "timeline_adherence": 0.3, "quality_metrics": 0.25}
        )
        
        self.predictive_models = {
            "research_productivity_model": productivity_model,
            "project_success_model": success_model
        }
    
    def get_intelligence_metrics(self) -> Dict[str, Any]:
        """Get comprehensive intelligence metrics."""
        return {
            "intelligence_metrics": self.intelligence_metrics,
            "active_metrics": len(self.metrics_store),
            "stored_insights": len(self.insights),
            "predictive_models": len(self.predictive_models),
            "active_alerts": len([a for a in self.active_alerts.values() if not a.resolved]),
            "model_accuracy": {
                model_name: model.accuracy 
                for model_name, model in self.predictive_models.items()
            },
            "monitoring_health": self._calculate_monitoring_health()
        }
    
    def _calculate_monitoring_health(self) -> str:
        """Calculate overall monitoring system health."""
        health_factors = [
            len(self.metrics_store) > 0,  # Has metrics
            self.intelligence_metrics["predictions_made"] > 0,  # Making predictions
            self.intelligence_metrics["insights_generated"] > 0,  # Generating insights
            self.intelligence_metrics["prediction_accuracy"] > 0.6  # Good accuracy
        ]
        
        health_score = sum(health_factors) / len(health_factors)
        
        if health_score >= 0.8:
            return "excellent"
        elif health_score >= 0.6:
            return "good"
        elif health_score >= 0.4:
            return "fair"
        else:
            return "poor"


# Supporting intelligence engines

class PredictiveAnalyticsEngine:
    """Advanced predictive analytics engine."""
    
    def __init__(self):
        self.models = {}
        self.feature_encoders = {}
    
    async def predict_outcome(self, 
                            model: PredictiveModel, 
                            time_horizon: int,
                            metrics_store: Dict[str, List[ResearchMetric]]) -> Dict[str, Any]:
        """Predict research outcomes using the specified model."""
        # Simplified prediction - would use actual ML models
        
        # Extract features from recent metrics
        features = await self._extract_features_for_prediction(model.features, metrics_store)
        
        # Generate prediction (placeholder logic)
        base_prediction = sum(features.values()) / len(features) if features else 0.5
        
        # Add time-based adjustment
        time_factor = min(1.0, time_horizon / 30.0)  # Normalize by 30 days
        adjusted_prediction = base_prediction * time_factor
        
        # Calculate confidence interval
        uncertainty = 0.1 * (1 + time_horizon / 30.0)  # Uncertainty increases with time
        confidence_interval = (
            max(0, adjusted_prediction - uncertainty),
            min(1, adjusted_prediction + uncertainty)
        )
        
        # Generate timeline forecast
        timeline_forecast = []
        for day in range(0, time_horizon, 7):  # Weekly forecast
            day_prediction = adjusted_prediction * (1 - day / (time_horizon * 2))  # Decay over time
            timeline_forecast.append({
                "day": day,
                "predicted_value": max(0, day_prediction),
                "confidence": max(0.5, 1 - day / time_horizon)
            })
        
        return {
            "value": adjusted_prediction,
            "confidence_interval": confidence_interval,
            "success_probability": adjusted_prediction,
            "influential_factors": list(model.feature_importance.keys())[:3],
            "timeline_forecast": timeline_forecast,
            "model_accuracy": model.accuracy
        }
    
    async def _extract_features_for_prediction(self, 
                                             feature_names: List[str],
                                             metrics_store: Dict[str, List[ResearchMetric]]) -> Dict[str, float]:
        """Extract feature values from metrics store."""
        features = {}
        
        for feature_name in feature_names:
            if feature_name in metrics_store and metrics_store[feature_name]:
                # Use most recent value
                recent_metric = metrics_store[feature_name][-1]
                features[feature_name] = recent_metric.value
            else:
                # Default value if metric not available
                features[feature_name] = 0.5
        
        return features


class PatternDetectionEngine:
    """Pattern detection in research data."""
    
    def __init__(self):
        self.pattern_detectors = {
            "trend": self._detect_trend_patterns,
            "cyclical": self._detect_cyclical_patterns,
            "correlation": self._detect_correlation_patterns,
            "anomaly_cluster": self._detect_anomaly_clusters
        }
    
    async def detect_patterns(self, 
                            metrics_store: Dict[str, List[ResearchMetric]], 
                            window_days: int) -> List[Dict[str, Any]]:
        """Detect patterns in research metrics."""
        patterns = []
        
        for pattern_type, detector in self.pattern_detectors.items():
            try:
                type_patterns = await detector(metrics_store, window_days)
                patterns.extend(type_patterns)
            except Exception as e:
                logger.error(f"Error detecting {pattern_type} patterns: {e}")
        
        return patterns
    
    async def _detect_trend_patterns(self, 
                                   metrics_store: Dict[str, List[ResearchMetric]], 
                                   window_days: int) -> List[Dict[str, Any]]:
        """Detect trend patterns in metrics."""
        patterns = []
        cutoff_date = datetime.now() - timedelta(days=window_days)
        
        for metric_name, metric_history in metrics_store.items():
            # Filter to time window
            windowed_metrics = [m for m in metric_history if m.timestamp >= cutoff_date]
            
            if len(windowed_metrics) >= 5:  # Need minimum data points
                values = [m.value for m in windowed_metrics]
                
                # Simple trend detection using linear correlation with time
                if len(values) > 1:
                    # Calculate trend strength (simplified)
                    first_half = values[:len(values)//2]
                    second_half = values[len(values)//2:]
                    
                    first_avg = statistics.mean(first_half)
                    second_avg = statistics.mean(second_half)
                    
                    if abs(second_avg - first_avg) > 0.1:  # Threshold for significant change
                        trend_direction = "increasing" if second_avg > first_avg else "decreasing"
                        trend_strength = abs(second_avg - first_avg) / first_avg if first_avg != 0 else 0
                        
                        pattern = {
                            "type": "trend",
                            "metric": metric_name,
                            "direction": trend_direction,
                            "strength": min(1.0, trend_strength),
                            "description": f"{metric_name} shows {trend_direction} trend",
                            "affected_metrics": [metric_name],
                            "temporal_info": {"duration_days": window_days, "data_points": len(windowed_metrics)}
                        }
                        patterns.append(pattern)
        
        return patterns
    
    async def _detect_cyclical_patterns(self, 
                                      metrics_store: Dict[str, List[ResearchMetric]], 
                                      window_days: int) -> List[Dict[str, Any]]:
        """Detect cyclical patterns in metrics."""
        # Simplified cyclical detection - would use FFT or similar in practice
        patterns = []
        
        for metric_name, metric_history in metrics_store.items():
            if len(metric_history) >= 14:  # Need at least 2 weeks of data
                values = [m.value for m in metric_history[-14:]]  # Last 2 weeks
                
                # Simple weekly pattern detection
                weekly_avg = [values[i::7] for i in range(7)]  # Group by day of week
                
                day_averages = []
                for day_values in weekly_avg:
                    if day_values:
                        day_averages.append(statistics.mean(day_values))
                    else:
                        day_averages.append(0)
                
                if day_averages and len(day_averages) == 7:
                    max_avg = max(day_averages)
                    min_avg = min(day_averages)
                    
                    if max_avg > 0 and (max_avg - min_avg) / max_avg > 0.2:  # 20% variation
                        pattern = {
                            "type": "cyclical",
                            "metric": metric_name,
                            "cycle_type": "weekly",
                            "strength": (max_avg - min_avg) / max_avg,
                            "description": f"{metric_name} shows weekly cyclical pattern",
                            "affected_metrics": [metric_name],
                            "temporal_info": {"cycle_length": "7 days", "variation": max_avg - min_avg}
                        }
                        patterns.append(pattern)
        
        return patterns
    
    async def _detect_correlation_patterns(self, 
                                         metrics_store: Dict[str, List[ResearchMetric]], 
                                         window_days: int) -> List[Dict[str, Any]]:
        """Detect correlation patterns between metrics."""
        patterns = []
        
        metric_names = list(metrics_store.keys())
        
        # Check all pairs of metrics for correlation
        for i, metric_a in enumerate(metric_names):
            for metric_b in metric_names[i+1:]:
                correlation = await self._calculate_metric_correlation(
                    metrics_store[metric_a], metrics_store[metric_b], window_days
                )
                
                if abs(correlation) > 0.7:  # Strong correlation threshold
                    correlation_type = "positive" if correlation > 0 else "negative"
                    
                    pattern = {
                        "type": "correlation",
                        "metrics": [metric_a, metric_b],
                        "correlation_strength": abs(correlation),
                        "correlation_type": correlation_type,
                        "description": f"Strong {correlation_type} correlation between {metric_a} and {metric_b}",
                        "affected_metrics": [metric_a, metric_b],
                        "temporal_info": {"analysis_window": window_days}
                    }
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_anomaly_clusters(self, 
                                     metrics_store: Dict[str, List[ResearchMetric]], 
                                     window_days: int) -> List[Dict[str, Any]]:
        """Detect clusters of anomalies."""
        patterns = []
        
        # Simplified anomaly clustering
        for metric_name, metric_history in metrics_store.items():
            if len(metric_history) >= 20:  # Need sufficient data
                values = [m.value for m in metric_history[-20:]]
                
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                
                if std_val > 0:
                    anomalies = []
                    for i, value in enumerate(values):
                        if abs(value - mean_val) > 2 * std_val:  # 2 sigma threshold
                            anomalies.append(i)
                    
                    if len(anomalies) >= 3:  # Multiple anomalies
                        pattern = {
                            "type": "anomaly_cluster",
                            "metric": metric_name,
                            "anomaly_count": len(anomalies),
                            "strength": len(anomalies) / len(values),
                            "description": f"Cluster of {len(anomalies)} anomalies in {metric_name}",
                            "affected_metrics": [metric_name],
                            "temporal_info": {"anomaly_positions": anomalies}
                        }
                        patterns.append(pattern)
        
        return patterns
    
    async def _calculate_metric_correlation(self, 
                                          metrics_a: List[ResearchMetric], 
                                          metrics_b: List[ResearchMetric], 
                                          window_days: int) -> float:
        """Calculate correlation between two metrics."""
        cutoff_date = datetime.now() - timedelta(days=window_days)
        
        # Filter both metrics to same time window
        values_a = [m.value for m in metrics_a if m.timestamp >= cutoff_date]
        values_b = [m.value for m in metrics_b if m.timestamp >= cutoff_date]
        
        # Align to same length (take minimum)
        min_length = min(len(values_a), len(values_b))
        if min_length < 5:  # Need minimum data points
            return 0.0
        
        values_a = values_a[-min_length:]
        values_b = values_b[-min_length:]
        
        # Calculate Pearson correlation coefficient
        if len(values_a) > 1 and len(values_b) > 1:
            try:
                mean_a = statistics.mean(values_a)
                mean_b = statistics.mean(values_b)
                
                numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b))
                
                sum_sq_a = sum((a - mean_a) ** 2 for a in values_a)
                sum_sq_b = sum((b - mean_b) ** 2 for b in values_b)
                
                denominator = (sum_sq_a * sum_sq_b) ** 0.5
                
                if denominator > 0:
                    return numerator / denominator
            except Exception as e:
                logger.error(f"Error calculating correlation: {e}")
        
        return 0.0


class InsightGenerationEngine:
    """Generate actionable insights from research data."""
    
    async def generate_insights(self, 
                              metrics_store: Dict[str, List[ResearchMetric]],
                              models: Dict[str, PredictiveModel]) -> List[IntelligenceInsight]:
        """Generate actionable insights."""
        insights = []
        
        # Performance insights
        performance_insights = await self._generate_performance_insights(metrics_store)
        insights.extend(performance_insights)
        
        # Predictive insights
        predictive_insights = await self._generate_predictive_insights(models)
        insights.extend(predictive_insights)
        
        # Resource optimization insights
        resource_insights = await self._generate_resource_insights(metrics_store)
        insights.extend(resource_insights)
        
        return insights
    
    async def _generate_performance_insights(self, 
                                           metrics_store: Dict[str, List[ResearchMetric]]) -> List[IntelligenceInsight]:
        """Generate performance-related insights."""
        insights = []
        
        # Example: Productivity insight
        if "productivity_score" in metrics_store:
            recent_productivity = metrics_store["productivity_score"][-7:]  # Last week
            if recent_productivity:
                avg_productivity = statistics.mean([m.value for m in recent_productivity])
                
                if avg_productivity > 0.8:
                    insight = IntelligenceInsight(
                        insight_id=f"insight_{uuid.uuid4().hex[:8]}",
                        title="High Productivity Period",
                        description=f"Research productivity is exceptionally high (average: {avg_productivity:.2f})",
                        intelligence_type=IntelligenceType.DESCRIPTIVE,
                        confidence=0.9,
                        impact_level="high",
                        evidence=[f"Average productivity: {avg_productivity:.2f}", "Consistent high performance over 7 days"],
                        recommendations=["Maintain current practices", "Document successful strategies"],
                        affected_areas=["productivity", "research_output"],
                        timestamp=datetime.now(),
                        validity_period=timedelta(days=7)
                    )
                    insights.append(insight)
        
        return insights
    
    async def _generate_predictive_insights(self, 
                                          models: Dict[str, PredictiveModel]) -> List[IntelligenceInsight]:
        """Generate predictive insights from models."""
        insights = []
        
        for model_name, model in models.items():
            if model.predictions:
                # Generate insight based on model predictions
                insight = IntelligenceInsight(
                    insight_id=f"insight_{uuid.uuid4().hex[:8]}",
                    title=f"Predictive Insight: {model.name}",
                    description=f"Model predicts {model.target_variable} with {model.accuracy:.2f} accuracy",
                    intelligence_type=IntelligenceType.PREDICTIVE,
                    confidence=model.accuracy,
                    impact_level="medium",
                    evidence=[f"Model accuracy: {model.accuracy:.2f}", f"Features: {', '.join(model.features)}"],
                    recommendations=["Monitor key features", "Validate predictions with actual outcomes"],
                    affected_areas=[model.target_variable],
                    timestamp=datetime.now(),
                    validity_period=timedelta(days=30)
                )
                insights.append(insight)
        
        return insights
    
    async def _generate_resource_insights(self, 
                                        metrics_store: Dict[str, List[ResearchMetric]]) -> List[IntelligenceInsight]:
        """Generate resource optimization insights."""
        insights = []
        
        # Example: Resource utilization insight
        resource_metrics = [name for name in metrics_store.keys() if "resource" in name.lower()]
        
        for metric_name in resource_metrics:
            recent_values = [m.value for m in metrics_store[metric_name][-10:]]  # Last 10 measurements
            if recent_values:
                avg_utilization = statistics.mean(recent_values)
                
                if avg_utilization < 0.5:  # Low utilization
                    insight = IntelligenceInsight(
                        insight_id=f"insight_{uuid.uuid4().hex[:8]}",
                        title="Low Resource Utilization",
                        description=f"{metric_name} utilization is low (average: {avg_utilization:.2f})",
                        intelligence_type=IntelligenceType.DIAGNOSTIC,
                        confidence=0.8,
                        impact_level="medium",
                        evidence=[f"Average utilization: {avg_utilization:.2f}", "Below optimal threshold (0.7)"],
                        recommendations=["Investigate underutilization causes", "Reallocate resources if possible"],
                        affected_areas=["resource_efficiency", "cost_optimization"],
                        timestamp=datetime.now(),
                        validity_period=timedelta(days=14)
                    )
                    insights.append(insight)
        
        return insights


class AnomalyDetectionEngine:
    """Detect anomalies in research metrics."""
    
    def __init__(self):
        self.baseline_models = {}
        self.anomaly_history = []
    
    async def detect_anomalies(self, 
                             metrics_store: Dict[str, List[ResearchMetric]]) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics."""
        anomalies = []
        
        for metric_name, metric_history in metrics_store.items():
            if len(metric_history) >= 10:  # Need sufficient history
                recent_values = [m.value for m in metric_history[-10:]]
                latest_value = recent_values[-1]
                
                # Statistical anomaly detection
                historical_values = recent_values[:-1]  # Exclude latest
                
                if len(historical_values) > 1:
                    mean_val = statistics.mean(historical_values)
                    std_val = statistics.stdev(historical_values)
                    
                    if std_val > 0:
                        z_score = abs(latest_value - mean_val) / std_val
                        
                        if z_score > 2.5:  # Anomaly threshold
                            anomaly = {
                                "metric": metric_name,
                                "actual_value": latest_value,
                                "expected_value": mean_val,
                                "z_score": z_score,
                                "severity": min(1.0, z_score / 5.0),  # Normalize severity
                                "description": f"{metric_name} value {latest_value:.3f} is {z_score:.2f} standard deviations from mean",
                                "timestamp": datetime.now()
                            }
                            anomalies.append(anomaly)
        
        return anomalies


class RecommendationEngine:
    """Generate actionable recommendations."""
    
    async def generate_recommendations(self, 
                                     context: Dict[str, Any],
                                     metrics_store: Dict[str, List[ResearchMetric]],
                                     insights: Dict[str, IntelligenceInsight]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Performance-based recommendations
        performance_recs = await self._generate_performance_recommendations(metrics_store)
        recommendations.extend(performance_recs)
        
        # Resource optimization recommendations
        resource_recs = await self._generate_resource_recommendations(metrics_store)
        recommendations.extend(resource_recs)
        
        # Insight-based recommendations
        insight_recs = await self._generate_insight_recommendations(insights)
        recommendations.extend(insight_recs)
        
        return recommendations
    
    async def _generate_performance_recommendations(self, 
                                                  metrics_store: Dict[str, List[ResearchMetric]]) -> List[Dict[str, Any]]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # Example: Low productivity recommendation
        if "productivity_score" in metrics_store:
            recent_productivity = [m.value for m in metrics_store["productivity_score"][-5:]]
            if recent_productivity and statistics.mean(recent_productivity) < 0.6:
                rec = {
                    "title": "Improve Research Productivity",
                    "description": "Recent productivity metrics indicate below-optimal performance",
                    "category": "performance",
                    "expected_impact": "high",
                    "implementation_steps": [
                        "Analyze time allocation patterns",
                        "Identify productivity bottlenecks",
                        "Implement time management strategies",
                        "Set clear daily goals"
                    ],
                    "resource_requirements": {"time": "2 hours", "tools": "productivity tracking"},
                    "timeline": "2 weeks",
                    "success_metrics": ["increased_productivity_score", "better_time_utilization"]
                }
                recommendations.append(rec)
        
        return recommendations
    
    async def _generate_resource_recommendations(self, 
                                               metrics_store: Dict[str, List[ResearchMetric]]) -> List[Dict[str, Any]]:
        """Generate resource optimization recommendations."""
        recommendations = []
        
        # Look for resource utilization patterns
        for metric_name, metrics in metrics_store.items():
            if "resource" in metric_name.lower() or "utilization" in metric_name.lower():
                recent_values = [m.value for m in metrics[-7:]]
                if recent_values:
                    avg_utilization = statistics.mean(recent_values)
                    
                    if avg_utilization > 0.9:  # Over-utilization
                        rec = {
                            "title": f"Optimize {metric_name}",
                            "description": f"High utilization detected in {metric_name}",
                            "category": "resource_optimization",
                            "expected_impact": "medium",
                            "implementation_steps": [
                                "Analyze resource usage patterns",
                                "Identify peak usage times",
                                "Consider scaling or load balancing"
                            ],
                            "resource_requirements": {"analysis_time": "1 day"},
                            "timeline": "1 week",
                            "success_metrics": ["reduced_utilization", "improved_efficiency"]
                        }
                        recommendations.append(rec)
        
        return recommendations
    
    async def _generate_insight_recommendations(self, 
                                              insights: Dict[str, IntelligenceInsight]) -> List[Dict[str, Any]]:
        """Generate recommendations based on insights."""
        recommendations = []
        
        for insight in insights.values():
            # Convert insight recommendations to structured format
            if insight.recommendations:
                rec = {
                    "title": f"Act on Insight: {insight.title}",
                    "description": f"Based on insight: {insight.description}",
                    "category": "insight_based",
                    "expected_impact": insight.impact_level,
                    "implementation_steps": insight.recommendations,
                    "resource_requirements": {"time": "varies"},
                    "timeline": "based on insight",
                    "success_metrics": ["insight_validation", "improved_outcomes"]
                }
                recommendations.append(rec)
        
        return recommendations