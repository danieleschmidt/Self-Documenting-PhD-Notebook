"""
Enhanced resilience and recovery mechanisms for the PhD notebook system.
Implements circuit breakers, retry logic, graceful degradation, and self-healing capabilities.
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from functools import wraps
import random

from ..utils.exceptions import ResilienceError, CircuitBreakerError
from ..monitoring.metrics import MetricsCollector


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    timeout_seconds: int = 60
    success_threshold: int = 3
    max_retry_attempts: int = 3
    exponential_backoff: bool = True
    jitter: bool = True


@dataclass
class OperationMetrics:
    """Metrics for tracked operations."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    average_response_time: float = 0.0
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for resilient operations.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = OperationMetrics()
        self.last_failure_time = None
        self.logger = logging.getLogger(f"circuit_breaker.{name}")
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        return wrapper
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        self.metrics.total_calls += 1
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")
        
        start_time = time.time()
        try:
            result = await self._execute_with_retry(func, *args, **kwargs)
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_failure(execution_time, e)
            raise
    
    async def _execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retry_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retry_attempts:
                    delay = self._calculate_backoff_delay(attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed for {self.name}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                break
        
        raise last_exception
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with jitter."""
        if not self.config.exponential_backoff:
            return 1.0
        
        base_delay = 2 ** attempt
        if self.config.jitter:
            jitter = random.uniform(0.1, 0.3) * base_delay
            return base_delay + jitter
        return base_delay
    
    def _record_success(self, execution_time: float):
        """Record successful operation."""
        self.metrics.successful_calls += 1
        self.metrics.consecutive_successes += 1
        self.metrics.consecutive_failures = 0
        
        # Update average response time
        total_successful = self.metrics.successful_calls
        current_avg = self.metrics.average_response_time
        self.metrics.average_response_time = (
            (current_avg * (total_successful - 1) + execution_time) / total_successful
        )
        
        if self.state == CircuitState.HALF_OPEN:
            if self.metrics.consecutive_successes >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.logger.info(f"Circuit breaker {self.name} reset to CLOSED")
    
    def _record_failure(self, execution_time: float, exception: Exception):
        """Record failed operation."""
        self.metrics.failed_calls += 1
        self.metrics.consecutive_failures += 1
        self.metrics.consecutive_successes = 0
        self.metrics.last_failure_time = datetime.now()
        self.last_failure_time = datetime.now()
        
        self.logger.error(f"Operation failed in {self.name}: {exception}")
        
        if (self.state == CircuitState.CLOSED and 
            self.metrics.consecutive_failures >= self.config.failure_threshold):
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker {self.name} opened due to failures")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker {self.name} reopened after half-open failure")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.timeout_seconds
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics."""
        success_rate = 0.0
        if self.metrics.total_calls > 0:
            success_rate = self.metrics.successful_calls / self.metrics.total_calls
        
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.metrics.total_calls,
            "successful_calls": self.metrics.successful_calls,
            "failed_calls": self.metrics.failed_calls,
            "success_rate": success_rate,
            "average_response_time": self.metrics.average_response_time,
            "consecutive_failures": self.metrics.consecutive_failures,
            "consecutive_successes": self.metrics.consecutive_successes,
            "last_failure_time": self.metrics.last_failure_time
        }


class ResilienceManager:
    """
    Central manager for resilience patterns and recovery mechanisms.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.metrics_collector = MetricsCollector()
        self.logger = logging.getLogger("resilience_manager")
        self._health_check_tasks: Dict[str, asyncio.Task] = {}
        
    def create_circuit_breaker(
        self, 
        name: str, 
        config: CircuitBreakerConfig = None
    ) -> CircuitBreaker:
        """Create and register a new circuit breaker."""
        if name in self.circuit_breakers:
            return self.circuit_breakers[name]
        
        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        
        # Start health check for this circuit breaker
        self._start_health_check(name)
        
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get existing circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    async def execute_with_resilience(
        self,
        operation_name: str,
        func: Callable,
        *args,
        config: CircuitBreakerConfig = None,
        **kwargs
    ) -> Any:
        """Execute operation with full resilience patterns."""
        circuit_breaker = self.create_circuit_breaker(operation_name, config)
        return await circuit_breaker.call(func, *args, **kwargs)
    
    def _start_health_check(self, circuit_breaker_name: str):
        """Start background health check for a circuit breaker."""
        async def health_check():
            while circuit_breaker_name in self.circuit_breakers:
                try:
                    circuit_breaker = self.circuit_breakers[circuit_breaker_name]
                    metrics = circuit_breaker.get_metrics()
                    
                    # Emit metrics
                    self.metrics_collector.record_gauge(
                        f"circuit_breaker.{circuit_breaker_name}.success_rate",
                        metrics["success_rate"]
                    )
                    self.metrics_collector.record_gauge(
                        f"circuit_breaker.{circuit_breaker_name}.response_time",
                        metrics["average_response_time"]
                    )
                    
                    # Check for degraded performance
                    if metrics["success_rate"] < 0.95 and metrics["total_calls"] > 10:
                        self.logger.warning(
                            f"Circuit breaker {circuit_breaker_name} showing degraded performance: "
                            f"{metrics['success_rate']:.2%} success rate"
                        )
                    
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Health check failed for {circuit_breaker_name}: {e}")
                    await asyncio.sleep(60)  # Longer wait on error
        
        task = asyncio.create_task(health_check())
        self._health_check_tasks[circuit_breaker_name] = task
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system resilience health."""
        total_circuits = len(self.circuit_breakers)
        open_circuits = sum(
            1 for cb in self.circuit_breakers.values() 
            if cb.state == CircuitState.OPEN
        )
        half_open_circuits = sum(
            1 for cb in self.circuit_breakers.values() 
            if cb.state == CircuitState.HALF_OPEN
        )
        
        circuit_metrics = {
            name: cb.get_metrics() 
            for name, cb in self.circuit_breakers.items()
        }
        
        overall_success_rate = 0.0
        total_calls = sum(metrics["total_calls"] for metrics in circuit_metrics.values())
        
        if total_calls > 0:
            total_successful = sum(
                metrics["successful_calls"] for metrics in circuit_metrics.values()
            )
            overall_success_rate = total_successful / total_calls
        
        return {
            "total_circuit_breakers": total_circuits,
            "open_circuits": open_circuits,
            "half_open_circuits": half_open_circuits,
            "closed_circuits": total_circuits - open_circuits - half_open_circuits,
            "overall_success_rate": overall_success_rate,
            "circuit_details": circuit_metrics,
            "system_status": "healthy" if open_circuits == 0 else "degraded"
        }
    
    async def shutdown(self):
        """Gracefully shutdown resilience manager."""
        self.logger.info("Shutting down resilience manager")
        
        # Cancel all health check tasks
        for task_name, task in self._health_check_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._health_check_tasks.clear()
        self.circuit_breakers.clear()


# Global resilience manager instance
resilience_manager = ResilienceManager()


# Convenience decorators
def with_circuit_breaker(
    name: str = None, 
    config: CircuitBreakerConfig = None
):
    """Decorator to apply circuit breaker pattern to a function."""
    def decorator(func: Callable) -> Callable:
        circuit_name = name or f"{func.__module__}.{func.__name__}"
        circuit_breaker = resilience_manager.create_circuit_breaker(circuit_name, config)
        return circuit_breaker(func)
    return decorator


def with_retry(
    max_attempts: int = 3,
    exponential_backoff: bool = True,
    jitter: bool = True
):
    """Decorator to add retry logic to a function."""
    def decorator(func: Callable) -> Callable:
        config = CircuitBreakerConfig(
            max_retry_attempts=max_attempts,
            exponential_backoff=exponential_backoff,
            jitter=jitter,
            failure_threshold=999999  # Effectively disable circuit breaking
        )
        return with_circuit_breaker(config=config)(func)
    return decorator


async def execute_with_graceful_degradation(
    primary_func: Callable,
    fallback_func: Callable,
    *args,
    **kwargs
) -> Any:
    """Execute primary function with fallback on failure."""
    try:
        return await resilience_manager.execute_with_resilience(
            f"{primary_func.__name__}_primary",
            primary_func,
            *args,
            **kwargs
        )
    except Exception as e:
        resilience_manager.logger.warning(
            f"Primary function {primary_func.__name__} failed: {e}. "
            f"Falling back to {fallback_func.__name__}"
        )
        
        try:
            return await resilience_manager.execute_with_resilience(
                f"{fallback_func.__name__}_fallback",
                fallback_func,
                *args,
                **kwargs
            )
        except Exception as fallback_error:
            resilience_manager.logger.error(
                f"Both primary and fallback functions failed. "
                f"Primary: {e}, Fallback: {fallback_error}"
            )
            raise