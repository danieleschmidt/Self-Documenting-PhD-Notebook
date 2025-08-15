"""Tests for the pipeline guard system."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.phd_notebook.pipeline import (
    PipelineGuard, GuardConfig, PipelineMonitor, PipelineStatus,
    FailureType, FailureAnalysis, SelfHealer
)


@pytest.fixture
def guard_config():
    """Create test guard configuration."""
    return GuardConfig(
        check_interval=1,  # Fast for testing
        heal_timeout=10,
        max_heal_attempts=2,
        enable_security_audit=False,  # Disable for simpler testing
        enable_resilience=False,
        enable_ml_prediction=False,
        enable_performance_optimization=False
    )


@pytest.fixture
def mock_pipeline_status():
    """Create mock pipeline status."""
    return PipelineStatus(
        pipeline_id="test_pipeline",
        state="failed",
        started_at=datetime.now() - timedelta(minutes=5),
        error="Test error"
    )


@pytest.fixture
def pipeline_guard(guard_config):
    """Create pipeline guard instance."""
    return PipelineGuard(guard_config)


@pytest.mark.asyncio
async def test_pipeline_guard_initialization(guard_config):
    """Test pipeline guard initializes correctly."""
    guard = PipelineGuard(guard_config)
    
    assert guard.config == guard_config
    assert not guard._is_running
    assert guard.monitor is not None
    assert guard.detector is not None
    assert guard.healer is not None


@pytest.mark.asyncio
async def test_pipeline_guard_start_stop(pipeline_guard):
    """Test starting and stopping pipeline guard."""
    # Mock the monitoring method to avoid infinite loop
    pipeline_guard._check_and_heal_cycle = AsyncMock()
    
    # Start monitoring in background
    monitor_task = asyncio.create_task(pipeline_guard.start_monitoring())
    
    # Give it a moment to start
    await asyncio.sleep(0.1)
    assert pipeline_guard._is_running
    
    # Stop monitoring
    await pipeline_guard.stop_monitoring()
    assert not pipeline_guard._is_running
    
    # Cancel the monitoring task
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_check_and_heal_cycle(pipeline_guard, mock_pipeline_status):
    """Test check and heal cycle functionality."""
    # Mock the monitor to return a failed pipeline
    pipeline_guard.monitor.get_pipeline_status = AsyncMock(
        return_value={"test_pipeline": mock_pipeline_status}
    )
    
    # Mock the failure handler
    pipeline_guard._handle_pipeline_failure = AsyncMock()
    
    # Run one cycle
    await pipeline_guard._check_and_heal_cycle()
    
    # Verify monitor was called
    pipeline_guard.monitor.get_pipeline_status.assert_called_once()
    
    # Verify failure handler was called for failed pipeline
    pipeline_guard._handle_pipeline_failure.assert_called_once_with(
        "test_pipeline", mock_pipeline_status
    )


@pytest.mark.asyncio
async def test_should_attempt_heal(pipeline_guard):
    """Test heal attempt logic."""
    pipeline_id = "test_pipeline"
    
    # Should attempt heal for new pipeline
    assert pipeline_guard._should_attempt_heal(pipeline_id)
    
    # Record heal attempts
    pipeline_guard._record_heal_attempt(pipeline_id)
    pipeline_guard._record_heal_attempt(pipeline_id)
    
    # Should still attempt (under limit)
    assert pipeline_guard._should_attempt_heal(pipeline_id)
    
    # Add one more attempt to reach limit
    pipeline_guard._record_heal_attempt(pipeline_id)
    
    # Should not attempt (over limit)
    assert not pipeline_guard._should_attempt_heal(pipeline_id)


@pytest.mark.asyncio
async def test_handle_pipeline_failure(pipeline_guard, mock_pipeline_status):
    """Test handling pipeline failure."""
    pipeline_id = "test_pipeline"
    
    # Mock dependencies
    mock_analysis = FailureAnalysis(
        failure_type=FailureType.TEST_FAILURE,
        confidence=0.9,
        description="Test failure",
        suggested_fixes=["Fix tests"],
        error_patterns=["test.*failed"],
        metadata={}
    )
    
    pipeline_guard.detector.analyze_failure = AsyncMock(return_value=mock_analysis)
    pipeline_guard.healer.heal_pipeline = AsyncMock(return_value=True)
    pipeline_guard._notify_heal_success = AsyncMock()
    
    # Handle failure
    await pipeline_guard._handle_pipeline_failure(pipeline_id, mock_pipeline_status)
    
    # Verify detector was called
    pipeline_guard.detector.analyze_failure.assert_called_once()
    
    # Verify healer was called
    pipeline_guard.healer.heal_pipeline.assert_called_once_with(pipeline_id, mock_analysis)
    
    # Verify success notification
    pipeline_guard._notify_heal_success.assert_called_once()


@pytest.mark.asyncio
async def test_get_status(pipeline_guard):
    """Test getting guard status."""
    status = pipeline_guard.get_status()
    
    assert "is_running" in status
    assert "config" in status
    assert "execution_metrics" in status
    assert "heal_history" in status
    
    assert status["is_running"] == False
    assert status["config"]["check_interval"] == 1
    assert status["execution_metrics"]["total_checks"] == 0


@pytest.mark.asyncio
async def test_comprehensive_report(pipeline_guard):
    """Test comprehensive status report."""
    # Mock pipeline status
    pipeline_guard.monitor.get_pipeline_status = AsyncMock(
        return_value={"test": PipelineStatus("test", "running", datetime.now())}
    )
    
    report = await pipeline_guard.get_comprehensive_report()
    
    assert "timestamp" in report
    assert "guard_status" in report
    assert "current_pipelines" in report


@pytest.mark.asyncio
async def test_notification_webhooks(pipeline_guard):
    """Test webhook notifications."""
    # Configure webhook
    pipeline_guard.config.notification_webhooks = ["https://example.com/webhook"]
    
    # Mock httpx
    with patch('httpx.AsyncClient') as mock_client:
        mock_response = AsyncMock()
        mock_client.return_value.__aenter__.return_value.post = mock_response
        
        await pipeline_guard._send_notifications("Test message")
        
        # Verify webhook was called
        mock_response.assert_called_once()


class TestPipelineMonitor:
    """Test pipeline monitor functionality."""
    
    @pytest.mark.asyncio
    async def test_get_pipeline_status(self):
        """Test getting pipeline status."""
        monitor = PipelineMonitor()
        
        # Mock subprocess for git commands
        with patch('asyncio.create_subprocess_shell') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"clean\n", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            status = await monitor.get_pipeline_status()
            
            # Should return at least local git status
            assert isinstance(status, dict)


class TestFailureDetector:
    """Test failure detector functionality."""
    
    def test_analyze_error_text(self):
        """Test error text analysis."""
        from src.phd_notebook.pipeline.detector import FailureDetector
        
        detector = FailureDetector()
        mock_status = MagicMock()
        mock_status.pipeline_id = "test"
        
        # Test dependency failure detection
        error_text = "npm install failed with exit code 1"
        analysis = detector._analyze_error_text(error_text, mock_status)
        
        assert analysis.failure_type == FailureType.DEPENDENCY_FAILURE
        assert analysis.confidence > 0.8
    
    def test_failure_statistics(self):
        """Test failure statistics calculation."""
        from src.phd_notebook.pipeline.detector import FailureDetector, FailureAnalysis, FailureType
        
        detector = FailureDetector()
        
        failures = [
            FailureAnalysis(FailureType.TEST_FAILURE, 0.9, "test", [], [], {}),
            FailureAnalysis(FailureType.TEST_FAILURE, 0.8, "test", [], [], {}),
            FailureAnalysis(FailureType.BUILD_FAILURE, 0.7, "build", [], [], {}),
        ]
        
        stats = detector.get_failure_statistics(failures)
        
        assert stats["total_failures"] == 3
        assert stats["failure_types"]["test_failure"] == 2
        assert stats["failure_types"]["build_failure"] == 1
        assert stats["most_common_type"] == "test_failure"


class TestSelfHealer:
    """Test self-healer functionality."""
    
    @pytest.mark.asyncio
    async def test_heal_pipeline(self):
        """Test pipeline healing."""
        healer = SelfHealer()
        
        mock_analysis = FailureAnalysis(
            failure_type=FailureType.DEPENDENCY_FAILURE,
            confidence=0.9,
            description="Package install failed",
            suggested_fixes=["Clear cache"],
            error_patterns=["npm.*failed"],
            metadata={}
        )
        
        # Mock subprocess for healing commands
        with patch('asyncio.create_subprocess_shell') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"Success\n", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            result = await healer.heal_pipeline("test_pipeline", mock_analysis)
            
            # Should attempt healing
            assert isinstance(result, bool)
    
    def test_healing_statistics(self):
        """Test healing statistics."""
        healer = SelfHealer()
        
        stats = healer.get_healing_statistics()
        
        assert "total_attempts" in stats
        assert stats["total_attempts"] == 0


@pytest.mark.asyncio
async def test_integration_full_cycle():
    """Integration test for full pipeline guard cycle."""
    config = GuardConfig(
        check_interval=1,
        max_heal_attempts=1,
        enable_security_audit=False,
        enable_resilience=False,
        enable_ml_prediction=False,
        enable_performance_optimization=False
    )
    
    guard = PipelineGuard(config)
    
    # Mock a failed pipeline
    failed_status = PipelineStatus(
        pipeline_id="integration_test",
        state="failed",
        started_at=datetime.now(),
        error="Integration test failure"
    )
    
    # Mock all external dependencies
    guard.monitor.get_pipeline_status = AsyncMock(
        return_value={"integration_test": failed_status}
    )
    
    mock_analysis = FailureAnalysis(
        failure_type=FailureType.TEST_FAILURE,
        confidence=0.9,
        description="Integration test",
        suggested_fixes=["Rerun tests"],
        error_patterns=["test.*failed"],
        metadata={}
    )
    
    guard.detector.analyze_failure = AsyncMock(return_value=mock_analysis)
    guard.healer.heal_pipeline = AsyncMock(return_value=True)
    guard._send_notifications = AsyncMock()
    
    # Run one cycle
    await guard._check_and_heal_cycle()
    
    # Verify the full flow
    assert guard._execution_metrics["total_checks"] == 1
    assert guard._execution_metrics["successful_heals"] == 1
    assert len(guard._heal_history["integration_test"]) == 1
    
    # Verify healer was called with correct parameters
    guard.healer.heal_pipeline.assert_called_once_with("integration_test", mock_analysis)


if __name__ == "__main__":
    pytest.main([__file__])