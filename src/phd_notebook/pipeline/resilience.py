"""Resilience and reliability features for pipeline guard."""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from ..utils.logging import get_logger
from ..utils.security import InputSanitizer


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker for service resilience."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_calls: int = 3
    
    # Internal state
    failure_count: int = field(default=0, init=False)
    last_failure_time: Optional[datetime] = field(default=None, init=False)
    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    half_open_attempts: int = field(default=0, init=False)


class ResilienceManager:
    """Manages resilience patterns for pipeline operations."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.sanitizer = InputSanitizer()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_configs: Dict[str, Dict[str, Any]] = {}
        self.health_checks: Dict[str, Callable] = {}
        self._service_status: Dict[str, Dict[str, Any]] = {}
    
    def register_service(self, 
                        service_name: str,
                        failure_threshold: int = 5,
                        recovery_timeout: int = 60) -> None:
        """Register a service for resilience management."""
        self.circuit_breakers[service_name] = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        
        self.retry_configs[service_name] = {
            "max_attempts": 3,
            "base_delay": 1.0,
            "max_delay": 30.0,
            "exponential_base": 2,
            "jitter": True
        }
        
        self._service_status[service_name] = {
            "last_check": None,
            "consecutive_failures": 0,
            "total_calls": 0,
            "successful_calls": 0,
            "last_error": None
        }
        
        self.logger.info(f"Registered service for resilience: {service_name}")
    
    async def call_with_circuit_breaker(self,
                                       service_name: str,
                                       operation: Callable,
                                       *args, **kwargs) -> Any:
        """Execute operation with circuit breaker protection."""
        if service_name not in self.circuit_breakers:
            self.register_service(service_name)
        
        circuit = self.circuit_breakers[service_name]
        status = self._service_status[service_name]
        
        # Check circuit state
        if circuit.state == CircuitState.OPEN:
            if self._should_attempt_reset(circuit):
                circuit.state = CircuitState.HALF_OPEN
                circuit.half_open_attempts = 0
                self.logger.info(f"Circuit breaker for {service_name} moved to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker open for {service_name}")
        
        if circuit.state == CircuitState.HALF_OPEN:
            if circuit.half_open_attempts >= circuit.half_open_max_calls:
                raise CircuitBreakerOpenError(f"Circuit breaker half-open limit reached for {service_name}")
        
        # Execute operation
        status["total_calls"] += 1
        status["last_check"] = datetime.now()
        
        try:
            result = await operation(*args, **kwargs)
            
            # Success - reset failure count
            if circuit.state == CircuitState.HALF_OPEN:
                circuit.half_open_attempts += 1
                if circuit.half_open_attempts >= circuit.half_open_max_calls:
                    circuit.state = CircuitState.CLOSED
                    circuit.failure_count = 0
                    self.logger.info(f"Circuit breaker for {service_name} reset to CLOSED")
            
            circuit.failure_count = 0
            status["successful_calls"] += 1
            status["consecutive_failures"] = 0
            status["last_error"] = None
            
            return result
            
        except Exception as e:
            # Failure - increment counts
            circuit.failure_count += 1
            circuit.last_failure_time = datetime.now()
            status["consecutive_failures"] += 1
            status["last_error"] = str(e)
            
            if circuit.state == CircuitState.HALF_OPEN:
                circuit.state = CircuitState.OPEN
                self.logger.warning(f"Circuit breaker for {service_name} opened after half-open failure")
            elif circuit.failure_count >= circuit.failure_threshold:
                circuit.state = CircuitState.OPEN
                self.logger.warning(f"Circuit breaker opened for {service_name} after {circuit.failure_count} failures")
            
            raise
    
    async def retry_with_backoff(self,
                                operation: Callable,
                                service_name: str = "default",
                                *args, **kwargs) -> Any:
        """Execute operation with exponential backoff retry."""
        if service_name not in self.retry_configs:
            self.register_service(service_name)
        
        config = self.retry_configs[service_name]
        
        for attempt in range(config["max_attempts"]):
            try:
                return await self.call_with_circuit_breaker(
                    service_name, operation, *args, **kwargs
                )
                
            except CircuitBreakerOpenError:
                raise  # Don't retry if circuit breaker is open
                
            except Exception as e:
                if attempt == config["max_attempts"] - 1:
                    self.logger.error(f"All retry attempts failed for {service_name}: {e}")
                    raise
                
                # Calculate delay with exponential backoff and jitter
                delay = min(
                    config["base_delay"] * (config["exponential_base"] ** attempt),
                    config["max_delay"]
                )
                
                if config["jitter"]:
                    import random
                    delay = delay * (0.5 + random.random() * 0.5)
                
                self.logger.warning(f"Attempt {attempt + 1} failed for {service_name}, retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
    
    def _should_attempt_reset(self, circuit: CircuitBreaker) -> bool:
        """Check if circuit breaker should attempt reset."""
        if circuit.last_failure_time is None:
            return False
        
        time_since_failure = datetime.now() - circuit.last_failure_time
        return time_since_failure.total_seconds() >= circuit.recovery_timeout
    
    def register_health_check(self, service_name: str, health_check: Callable) -> None:
        """Register a health check for a service."""
        self.health_checks[service_name] = health_check
        self.logger.info(f"Registered health check for {service_name}")
    
    async def run_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks."""
        results = {}
        
        for service_name, health_check in self.health_checks.items():
            try:
                start_time = datetime.now()
                
                # Run health check with timeout
                health_result = await asyncio.wait_for(
                    health_check(),
                    timeout=30.0  # 30 second timeout
                )
                
                response_time = (datetime.now() - start_time).total_seconds()
                
                results[service_name] = {
                    "status": "healthy" if health_result else "unhealthy",
                    "response_time": response_time,
                    "timestamp": datetime.now().isoformat(),
                    "details": health_result if isinstance(health_result, dict) else {}
                }
                
            except asyncio.TimeoutError:
                results[service_name] = {
                    "status": "timeout",
                    "response_time": 30.0,
                    "timestamp": datetime.now().isoformat(),
                    "error": "Health check timed out"
                }
                
            except Exception as e:
                results[service_name] = {
                    "status": "error",
                    "response_time": None,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
        
        return results
    
    def get_service_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all monitored services."""
        metrics = {}
        
        for service_name in self.circuit_breakers:
            circuit = self.circuit_breakers[service_name]
            status = self._service_status[service_name]
            
            success_rate = 0.0
            if status["total_calls"] > 0:
                success_rate = status["successful_calls"] / status["total_calls"]
            
            metrics[service_name] = {
                "circuit_state": circuit.state.value,
                "failure_count": circuit.failure_count,
                "total_calls": status["total_calls"],
                "successful_calls": status["successful_calls"],
                "success_rate": success_rate,
                "consecutive_failures": status["consecutive_failures"],
                "last_check": status["last_check"].isoformat() if status["last_check"] else None,
                "last_error": status["last_error"]
            }
        
        return metrics
    
    async def create_resilience_report(self) -> Dict[str, Any]:
        """Create comprehensive resilience report."""
        health_results = await self.run_health_checks()
        service_metrics = self.get_service_metrics()
        
        # Calculate overall system health
        healthy_services = sum(1 for result in health_results.values() if result["status"] == "healthy")
        total_services = len(health_results)
        overall_health = (healthy_services / total_services * 100) if total_services > 0 else 100
        
        # Identify problematic services
        problematic_services = []
        for service_name, metrics in service_metrics.items():
            if (metrics["success_rate"] < 0.8 or 
                metrics["consecutive_failures"] > 3 or
                metrics["circuit_state"] == "open"):
                problematic_services.append({
                    "service": service_name,
                    "issues": [
                        f"Success rate: {metrics['success_rate']:.1%}" if metrics["success_rate"] < 0.8 else None,
                        f"Consecutive failures: {metrics['consecutive_failures']}" if metrics["consecutive_failures"] > 3 else None,
                        f"Circuit breaker: {metrics['circuit_state']}" if metrics["circuit_state"] == "open" else None
                    ]
                })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health_percentage": overall_health,
            "total_services": total_services,
            "healthy_services": healthy_services,
            "problematic_services": problematic_services,
            "health_checks": health_results,
            "service_metrics": service_metrics,
            "recommendations": self._generate_recommendations(service_metrics, health_results)
        }
    
    def _generate_recommendations(self, 
                                 service_metrics: Dict[str, Any],
                                 health_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on service health."""
        recommendations = []
        
        for service_name, metrics in service_metrics.items():
            if metrics["success_rate"] < 0.5:
                recommendations.append(f"Service {service_name} has very low success rate - investigate immediately")
            elif metrics["success_rate"] < 0.8:
                recommendations.append(f"Service {service_name} success rate below 80% - monitor closely")
            
            if metrics["circuit_state"] == "open":
                recommendations.append(f"Circuit breaker open for {service_name} - service may be down")
            
            if metrics["consecutive_failures"] > 5:
                recommendations.append(f"Service {service_name} has {metrics['consecutive_failures']} consecutive failures")
        
        for service_name, health in health_results.items():
            if health["status"] == "timeout":
                recommendations.append(f"Health check timeout for {service_name} - check service responsiveness")
            elif health["status"] == "error":
                recommendations.append(f"Health check error for {service_name}: {health.get('error', 'Unknown error')}")
        
        return recommendations


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation."""
    max_concurrent_calls: int = 10
    queue_size: int = 100
    timeout: float = 30.0


class Bulkhead:
    """Implements bulkhead isolation pattern."""
    
    def __init__(self, config: BulkheadConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_calls)
        self.queue = asyncio.Queue(maxsize=config.queue_size)
        self.active_calls = 0
        self.total_calls = 0
        self.rejected_calls = 0
        self.logger = get_logger(__name__)
    
    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation within bulkhead constraints."""
        self.total_calls += 1
        
        try:
            # Try to acquire semaphore with timeout
            await asyncio.wait_for(
                self.semaphore.acquire(),
                timeout=self.config.timeout
            )
        except asyncio.TimeoutError:
            self.rejected_calls += 1
            raise BulkheadCapacityExceededError("Bulkhead capacity exceeded")
        
        try:
            self.active_calls += 1
            result = await asyncio.wait_for(
                operation(*args, **kwargs),
                timeout=self.config.timeout
            )
            return result
        finally:
            self.active_calls -= 1
            self.semaphore.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "active_calls": self.active_calls,
            "total_calls": self.total_calls,
            "rejected_calls": self.rejected_calls,
            "rejection_rate": self.rejected_calls / self.total_calls if self.total_calls > 0 else 0,
            "available_capacity": self.semaphore._value,
            "max_capacity": self.config.max_concurrent_calls
        }


class BulkheadCapacityExceededError(Exception):
    """Raised when bulkhead capacity is exceeded."""
    pass