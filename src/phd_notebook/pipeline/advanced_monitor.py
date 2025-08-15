"""Advanced monitoring with machine learning and predictive capabilities."""

import asyncio
import json
import pickle
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import hashlib

from ..utils.logging import get_logger


@dataclass
class PredictionModel:
    """Simple prediction model for pipeline failures."""
    feature_weights: Dict[str, float]
    threshold: float = 0.7
    accuracy: float = 0.0
    last_trained: Optional[datetime] = None


class MLPredictor:
    """Machine learning predictor for pipeline failures."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.models: Dict[str, PredictionModel] = {}
        self.training_data = deque(maxlen=10000)  # Keep last 10k data points
        self.feature_extractors = self._build_feature_extractors()
    
    def extract_features(self, pipeline_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from pipeline data for ML prediction."""
        features = {}
        
        # Time-based features
        now = datetime.now()
        if pipeline_data.get("last_success"):
            time_since_success = (now - pipeline_data["last_success"]).total_seconds() / 3600  # hours
            features["hours_since_success"] = min(time_since_success, 168)  # Cap at 1 week
        else:
            features["hours_since_success"] = 168  # Default to max if no success
        
        # Failure pattern features
        features["consecutive_failures"] = min(pipeline_data.get("consecutive_failures", 0), 20)
        features["failure_rate_24h"] = pipeline_data.get("failure_rate_24h", 0.0)
        features["avg_duration"] = min(pipeline_data.get("avg_duration", 0), 3600)  # Cap at 1 hour
        
        # External factors
        features["hour_of_day"] = now.hour / 24.0
        features["day_of_week"] = now.weekday() / 7.0
        
        # System load features
        features["system_cpu"] = pipeline_data.get("system_cpu", 0.0) / 100.0
        features["system_memory"] = pipeline_data.get("system_memory", 0.0) / 100.0
        features["active_pipelines"] = min(pipeline_data.get("active_pipelines", 0), 100) / 100.0
        
        # Historical features
        features["success_rate_7d"] = pipeline_data.get("success_rate_7d", 1.0)
        features["mean_time_between_failures"] = min(
            pipeline_data.get("mean_time_between_failures", 24), 168
        ) / 168.0
        
        return features
    
    def predict_failure_probability(self, 
                                   pipeline_id: str,
                                   pipeline_data: Dict[str, Any]) -> float:
        """Predict probability of pipeline failure."""
        features = self.extract_features(pipeline_data)
        
        # Use pipeline-specific model if available, otherwise use global model
        model_key = pipeline_id if pipeline_id in self.models else "global"
        
        if model_key not in self.models:
            # Return default probability based on simple heuristics
            if pipeline_data.get("consecutive_failures", 0) > 3:
                return 0.8
            elif pipeline_data.get("failure_rate_24h", 0) > 0.5:
                return 0.6
            else:
                return 0.2
        
        model = self.models[model_key]
        
        # Simple linear model prediction
        score = 0.0
        for feature, value in features.items():
            weight = model.feature_weights.get(feature, 0.0)
            score += weight * value
        
        # Apply sigmoid function to get probability
        probability = 1 / (1 + np.exp(-score))
        
        return probability
    
    def add_training_data(self, 
                         pipeline_id: str,
                         pipeline_data: Dict[str, Any],
                         actual_failure: bool):
        """Add training data for model improvement."""
        features = self.extract_features(pipeline_data)
        
        training_point = {
            "timestamp": datetime.now(),
            "pipeline_id": pipeline_id,
            "features": features,
            "label": actual_failure
        }
        
        self.training_data.append(training_point)
    
    def train_models(self, min_data_points: int = 100) -> Dict[str, float]:
        """Train prediction models using collected data."""
        if len(self.training_data) < min_data_points:
            self.logger.warning(f"Insufficient training data: {len(self.training_data)} < {min_data_points}")
            return {}
        
        # Separate data by pipeline
        pipeline_data = defaultdict(list)
        global_data = list(self.training_data)
        
        for point in self.training_data:
            pipeline_data[point["pipeline_id"]].append(point)
        
        trained_models = {}
        
        # Train global model
        if len(global_data) >= min_data_points:
            accuracy = self._train_single_model("global", global_data)
            trained_models["global"] = accuracy
        
        # Train pipeline-specific models
        for pipeline_id, data in pipeline_data.items():
            if len(data) >= min_data_points:
                accuracy = self._train_single_model(pipeline_id, data)
                trained_models[pipeline_id] = accuracy
        
        return trained_models
    
    def _train_single_model(self, model_key: str, training_data: List[Dict]) -> float:
        """Train a single prediction model."""
        # Extract features and labels
        feature_names = set()
        for point in training_data:
            feature_names.update(point["features"].keys())
        
        feature_names = sorted(feature_names)
        
        X = []
        y = []
        
        for point in training_data:
            feature_vector = [point["features"].get(name, 0.0) for name in feature_names]
            X.append(feature_vector)
            y.append(1.0 if point["label"] else 0.0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Simple logistic regression (manually implemented)
        weights = self._fit_logistic_regression(X, y)
        
        # Calculate accuracy
        predictions = self._predict_with_weights(X, weights)
        accuracy = np.mean((predictions > 0.5) == (y > 0.5))
        
        # Store model
        model = PredictionModel(
            feature_weights=dict(zip(feature_names, weights)),
            accuracy=accuracy,
            last_trained=datetime.now()
        )
        
        self.models[model_key] = model
        
        self.logger.info(f"Trained model {model_key} with accuracy: {accuracy:.3f}")
        
        return accuracy
    
    def _fit_logistic_regression(self, X: np.ndarray, y: np.ndarray, 
                                learning_rate: float = 0.01, 
                                max_iterations: int = 1000) -> np.ndarray:
        """Simple logistic regression implementation."""
        n_features = X.shape[1]
        weights = np.zeros(n_features)
        
        for _ in range(max_iterations):
            # Forward pass
            linear_pred = X.dot(weights)
            predictions = 1 / (1 + np.exp(-np.clip(linear_pred, -250, 250)))
            
            # Compute gradient
            gradient = X.T.dot(predictions - y) / len(y)
            
            # Update weights
            weights -= learning_rate * gradient
            
            # Check for convergence (simplified)
            if np.linalg.norm(gradient) < 1e-6:
                break
        
        return weights
    
    def _predict_with_weights(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Make predictions using weights."""
        linear_pred = X.dot(weights)
        return 1 / (1 + np.exp(-np.clip(linear_pred, -250, 250)))
    
    def _build_feature_extractors(self) -> Dict[str, callable]:
        """Build feature extraction functions."""
        return {
            "time_features": self._extract_time_features,
            "failure_pattern": self._extract_failure_patterns,
            "system_load": self._extract_system_load,
            "historical": self._extract_historical_features
        }
    
    def _extract_time_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract time-based features."""
        now = datetime.now()
        return {
            "hour_sin": np.sin(2 * np.pi * now.hour / 24),
            "hour_cos": np.cos(2 * np.pi * now.hour / 24),
            "day_sin": np.sin(2 * np.pi * now.weekday() / 7),
            "day_cos": np.cos(2 * np.pi * now.weekday() / 7),
        }
    
    def _extract_failure_patterns(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract failure pattern features."""
        return {
            "consecutive_failures": data.get("consecutive_failures", 0),
            "failures_last_24h": data.get("failures_last_24h", 0),
            "avg_failure_duration": data.get("avg_failure_duration", 0),
        }
    
    def _extract_system_load(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract system load features."""
        return {
            "cpu_usage": data.get("cpu_usage", 0) / 100,
            "memory_usage": data.get("memory_usage", 0) / 100,
            "active_processes": min(data.get("active_processes", 0), 1000) / 1000,
        }
    
    def _extract_historical_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract historical performance features."""
        return {
            "success_rate_7d": data.get("success_rate_7d", 1.0),
            "avg_duration_7d": min(data.get("avg_duration_7d", 0), 3600) / 3600,
            "reliability_score": data.get("reliability_score", 1.0),
        }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models."""
        performance = {}
        
        for model_key, model in self.models.items():
            performance[model_key] = {
                "accuracy": model.accuracy,
                "last_trained": model.last_trained.isoformat() if model.last_trained else None,
                "feature_count": len(model.feature_weights),
                "most_important_features": sorted(
                    model.feature_weights.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:5]
            }
        
        return performance


class AnomalyDetector:
    """Detects anomalies in pipeline behavior."""
    
    def __init__(self, window_size: int = 100):
        self.logger = get_logger(__name__)
        self.window_size = window_size
        self.baseline_metrics = {}
        self.anomaly_thresholds = {
            "duration": 3.0,  # 3 standard deviations
            "failure_rate": 2.0,  # 2 standard deviations
            "resource_usage": 2.5  # 2.5 standard deviations
        }
    
    def update_baseline(self, pipeline_id: str, metrics: Dict[str, float]):
        """Update baseline metrics for anomaly detection."""
        if pipeline_id not in self.baseline_metrics:
            self.baseline_metrics[pipeline_id] = {
                "duration": deque(maxlen=self.window_size),
                "cpu_usage": deque(maxlen=self.window_size),
                "memory_usage": deque(maxlen=self.window_size),
                "failure_rate": deque(maxlen=self.window_size)
            }
        
        baseline = self.baseline_metrics[pipeline_id]
        
        for metric_name, value in metrics.items():
            if metric_name in baseline and value is not None:
                baseline[metric_name].append(value)
    
    def detect_anomalies(self, pipeline_id: str, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect anomalies in current metrics compared to baseline."""
        if pipeline_id not in self.baseline_metrics:
            return []  # No baseline yet
        
        anomalies = []
        baseline = self.baseline_metrics[pipeline_id]
        
        for metric_name, current_value in current_metrics.items():
            if metric_name not in baseline or len(baseline[metric_name]) < 10:
                continue  # Not enough baseline data
            
            baseline_values = list(baseline[metric_name])
            mean_value = np.mean(baseline_values)
            std_value = np.std(baseline_values)
            
            if std_value == 0:
                continue  # No variation in baseline
            
            # Calculate z-score
            z_score = abs(current_value - mean_value) / std_value
            threshold = self.anomaly_thresholds.get(metric_name, 2.0)
            
            if z_score > threshold:
                anomalies.append({
                    "metric": metric_name,
                    "current_value": current_value,
                    "baseline_mean": mean_value,
                    "baseline_std": std_value,
                    "z_score": z_score,
                    "severity": "high" if z_score > threshold * 1.5 else "medium",
                    "description": f"{metric_name} is {z_score:.1f} standard deviations from normal"
                })
        
        return anomalies
    
    def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of anomalies detected in recent time period."""
        # This would require storing anomaly history
        # Simplified implementation
        return {
            "total_pipelines_monitored": len(self.baseline_metrics),
            "baseline_window_size": self.window_size,
            "detection_thresholds": self.anomaly_thresholds
        }


class TrendAnalyzer:
    """Analyzes trends in pipeline performance and health."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.trend_data = defaultdict(lambda: defaultdict(deque))
    
    def add_data_point(self, pipeline_id: str, metric: str, value: float, timestamp: datetime = None):
        """Add a data point for trend analysis."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.trend_data[pipeline_id][metric].append({
            "timestamp": timestamp,
            "value": value
        })
        
        # Keep only recent data (last 1000 points per metric)
        if len(self.trend_data[pipeline_id][metric]) > 1000:
            self.trend_data[pipeline_id][metric].popleft()
    
    def analyze_trend(self, pipeline_id: str, metric: str, days: int = 7) -> Dict[str, Any]:
        """Analyze trend for a specific metric over time period."""
        if (pipeline_id not in self.trend_data or 
            metric not in self.trend_data[pipeline_id]):
            return {"error": "No data available"}
        
        cutoff_time = datetime.now() - timedelta(days=days)
        data_points = [
            point for point in self.trend_data[pipeline_id][metric]
            if point["timestamp"] > cutoff_time
        ]
        
        if len(data_points) < 2:
            return {"error": "Insufficient data points"}
        
        # Extract values and timestamps
        values = [point["value"] for point in data_points]
        timestamps = [point["timestamp"] for point in data_points]
        
        # Convert timestamps to hours since start
        start_time = timestamps[0]
        time_hours = [(ts - start_time).total_seconds() / 3600 for ts in timestamps]
        
        # Simple linear regression
        slope, intercept = self._linear_regression(time_hours, values)
        
        # Calculate trend strength (correlation coefficient)
        correlation = self._correlation_coefficient(time_hours, values)
        
        # Determine trend direction and strength
        if abs(correlation) < 0.3:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        trend_strength = abs(correlation)
        
        return {
            "metric": metric,
            "period_days": days,
            "data_points": len(data_points),
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "slope": slope,
            "correlation": correlation,
            "current_value": values[-1],
            "start_value": values[0],
            "mean_value": np.mean(values),
            "std_value": np.std(values)
        }
    
    def _linear_regression(self, x: List[float], y: List[float]) -> Tuple[float, float]:
        """Simple linear regression."""
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        return slope, intercept
    
    def _correlation_coefficient(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def get_system_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get overall system trends across all pipelines."""
        system_trends = {}
        
        # Aggregate metrics across all pipelines
        all_metrics = set()
        for pipeline_data in self.trend_data.values():
            all_metrics.update(pipeline_data.keys())
        
        for metric in all_metrics:
            all_values = []
            all_timestamps = []
            
            cutoff_time = datetime.now() - timedelta(days=days)
            
            for pipeline_data in self.trend_data.values():
                if metric in pipeline_data:
                    for point in pipeline_data[metric]:
                        if point["timestamp"] > cutoff_time:
                            all_values.append(point["value"])
                            all_timestamps.append(point["timestamp"])
            
            if len(all_values) >= 2:
                # Sort by timestamp
                sorted_data = sorted(zip(all_timestamps, all_values))
                timestamps = [item[0] for item in sorted_data]
                values = [item[1] for item in sorted_data]
                
                # Analyze trend
                start_time = timestamps[0]
                time_hours = [(ts - start_time).total_seconds() / 3600 for ts in timestamps]
                slope, _ = self._linear_regression(time_hours, values)
                correlation = self._correlation_coefficient(time_hours, values)
                
                system_trends[metric] = {
                    "slope": slope,
                    "correlation": correlation,
                    "data_points": len(values),
                    "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                }
        
        return system_trends