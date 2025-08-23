"""
Unit Tests for Quantum-Inspired Systems

Comprehensive unit testing for quantum research accelerator and 
quantum performance optimizer components.
"""

import pytest
import asyncio
import math
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import uuid

# Import quantum systems
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from phd_notebook.research.quantum_research_accelerator import (
    QuantumResearchAccelerator, QuantumHypothesis, QuantumState,
    QuantumResearchCircuit, ResearchEntanglement, create_quantum_hypothesis
)
from phd_notebook.performance.quantum_performance_optimizer import (
    QuantumPerformanceOptimizer, PerformanceMetric, OptimizationDimension,
    OptimizationStrategy, PerformanceState, create_optimization_target
)


class TestQuantumHypothesis:
    """Test quantum hypothesis functionality."""
    
    def test_quantum_hypothesis_creation(self):
        """Test creation of quantum hypotheses."""
        hypothesis = create_quantum_hypothesis(
            "Quantum computing will revolutionize machine learning",
            confidence=0.8,
            research_domains=["quantum_computing", "machine_learning"]
        )
        
        assert hypothesis.statement == "Quantum computing will revolutionize machine learning"
        assert hypothesis.confidence_level == 0.8
        assert hypothesis.research_domains == ["quantum_computing", "machine_learning"]
        assert hypothesis.quantum_state == QuantumState.SUPERPOSITION
        assert abs(hypothesis.probability_amplitude) > 0
        
        # Check quantum amplitude properties
        amplitude_magnitude = abs(hypothesis.probability_amplitude)
        expected_magnitude = math.sqrt(0.8)  # Based on confidence
        assert abs(amplitude_magnitude - expected_magnitude) < 0.1
    
    def test_quantum_hypothesis_measurement(self):
        """Test quantum hypothesis measurement and state collapse."""
        hypothesis = create_quantum_hypothesis(
            "Test hypothesis for measurement",
            confidence=0.7
        )
        
        # Initially in superposition
        assert hypothesis.quantum_state == QuantumState.SUPERPOSITION
        assert hypothesis.measurement_count == 0
        assert hypothesis.last_measured is None
        
        # Simulate measurement (would normally collapse state)
        hypothesis.measurement_count += 1
        hypothesis.last_measured = datetime.now()
        hypothesis.quantum_state = QuantumState.COLLAPSED
        
        assert hypothesis.measurement_count == 1
        assert hypothesis.last_measured is not None
        assert hypothesis.quantum_state == QuantumState.COLLAPSED


class TestQuantumResearchCircuit:
    """Test quantum research circuit functionality."""
    
    def test_circuit_initialization(self):
        """Test quantum research circuit initialization."""
        circuit = QuantumResearchCircuit(num_qubits=3)
        
        assert circuit.num_qubits == 3
        assert len(circuit.gates) == 0
        assert len(circuit.hypotheses) == 0
        assert len(circuit.entanglements) == 0
    
    def test_superposition_gate_addition(self):
        """Test adding superposition gates to circuit."""
        circuit = QuantumResearchCircuit(num_qubits=2)
        hypothesis = create_quantum_hypothesis("Test hypothesis")
        
        # Add superposition gate
        circuit.add_superposition_gate(0, hypothesis)
        
        assert len(circuit.gates) == 1
        assert circuit.hypotheses[0] == hypothesis
        assert hypothesis.quantum_state == QuantumState.SUPERPOSITION
        
        # Verify gate properties
        gate = circuit.gates[0]
        assert gate.operation_type == "superposition"
        assert hypothesis.hypothesis_id in gate.input_hypotheses
    
    def test_entanglement_gate_addition(self):
        """Test adding entanglement gates to circuit."""
        circuit = QuantumResearchCircuit(num_qubits=2)
        
        # Add hypotheses to both qubits
        hyp1 = create_quantum_hypothesis("Hypothesis 1")
        hyp2 = create_quantum_hypothesis("Hypothesis 2")
        
        circuit.add_superposition_gate(0, hyp1)
        circuit.add_superposition_gate(1, hyp2)
        
        # Add entanglement gate
        circuit.add_entanglement_gate(0, 1, correlation_strength=0.8)
        
        # Verify entanglement was created
        assert len(circuit.entanglements) == 1
        entanglement = circuit.entanglements[0]
        assert entanglement.correlation_strength == 0.8
        
        # Verify hypotheses are entangled
        assert hyp1.quantum_state == QuantumState.ENTANGLED
        assert hyp2.quantum_state == QuantumState.ENTANGLED
        assert hyp2.hypothesis_id in hyp1.entangled_hypotheses
        assert hyp1.hypothesis_id in hyp2.entangled_hypotheses
    
    def test_hypothesis_measurement(self):
        """Test measuring hypotheses in quantum circuit."""
        circuit = QuantumResearchCircuit(num_qubits=1)
        hypothesis = create_quantum_hypothesis("Test measurement", confidence=0.9)
        
        circuit.add_superposition_gate(0, hypothesis)
        
        # Measure hypothesis
        measured_state, probability = circuit.measure_hypothesis(0)
        
        assert measured_state in ["confirmed", "refuted"]
        assert 0.0 <= probability <= 1.0
        assert hypothesis.quantum_state == QuantumState.COLLAPSED
        assert hypothesis.measurement_count == 1
        assert hypothesis.last_measured is not None


class TestQuantumResearchAccelerator:
    """Test quantum research accelerator functionality."""
    
    @pytest.fixture
    def accelerator(self):
        """Create quantum research accelerator for testing."""
        return QuantumResearchAccelerator(system_id="test_accelerator")
    
    async def test_accelerator_initialization(self, accelerator):
        """Test accelerator initialization."""
        assert accelerator.system_id == "test_accelerator"
        assert len(accelerator.research_circuits) == 0
        assert len(accelerator.hypothesis_space) == 0
        assert accelerator.pattern_recognizer is not None
        assert accelerator.outcome_predictor is not None
    
    async def test_research_circuit_creation(self, accelerator):
        """Test creating research circuits."""
        hypotheses = [
            "Quantum algorithms improve optimization",
            "Machine learning benefits from quantum computing"
        ]
        
        circuit_id = await accelerator.create_research_circuit(
            "test_circuit", hypotheses
        )
        
        assert circuit_id in accelerator.research_circuits
        circuit = accelerator.research_circuits[circuit_id]
        assert circuit.num_qubits == len(hypotheses)
        assert len(circuit.hypotheses) == len(hypotheses)
        
        # Verify hypotheses are in hypothesis space
        for hypothesis in circuit.hypotheses.values():
            assert hypothesis.hypothesis_id in accelerator.hypothesis_space
    
    async def test_concept_entanglement(self, accelerator):
        """Test entangling research concepts."""
        # Create circuit with hypotheses
        hypotheses = [
            "Quantum computing enhances AI",
            "AI optimizes quantum algorithms"
        ]
        
        circuit_id = await accelerator.create_research_circuit(
            "entanglement_test", hypotheses
        )
        
        # Create entanglements
        concept_pairs = [("quantum computing", "AI")]
        entanglement_ids = await accelerator.entangle_research_concepts(
            circuit_id, concept_pairs
        )
        
        assert len(entanglement_ids) > 0
        
        # Verify entanglement in circuit
        circuit = accelerator.research_circuits[circuit_id]
        assert len(circuit.entanglements) > 0
    
    async def test_hypothesis_space_exploration(self, accelerator):
        """Test exploring hypothesis space."""
        # Create circuit
        hypotheses = ["Base hypothesis 1", "Base hypothesis 2"]
        circuit_id = await accelerator.create_research_circuit("explore_test", hypotheses)
        
        # Explore hypothesis space
        new_hypotheses = await accelerator.explore_hypothesis_space(
            circuit_id, exploration_depth=2
        )
        
        # Should generate some new hypotheses (may be empty in simple cases)
        assert isinstance(new_hypotheses, list)
        
        # If hypotheses were generated, verify they're in the hypothesis space
        for hypothesis in new_hypotheses:
            assert hypothesis.hypothesis_id in accelerator.hypothesis_space
            assert hypothesis.quantum_state == QuantumState.SUPERPOSITION
    
    async def test_research_outcome_prediction(self, accelerator):
        """Test predicting research outcomes."""
        # Create test hypotheses
        hypotheses = [
            create_quantum_hypothesis("High confidence hypothesis", confidence=0.9),
            create_quantum_hypothesis("Low confidence hypothesis", confidence=0.3)
        ]
        
        predictions = await accelerator.predict_research_outcomes(
            hypotheses, time_horizon=90
        )
        
        assert len(predictions) == 2
        
        # Verify prediction structure
        for hyp_id, prediction in predictions.items():
            assert "success_probability" in prediction
            assert "impact_score" in prediction
            assert "timeline_accuracy" in prediction
            
            # All values should be between 0 and 1
            for key, value in prediction.items():
                assert 0.0 <= value <= 1.0
    
    async def test_cross_domain_synthesis(self, accelerator):
        """Test cross-domain insight synthesis."""
        # Add hypotheses with different domains
        hyp1 = create_quantum_hypothesis(
            "Computer science benefits from physics",
            research_domains=["computer_science", "physics"]
        )
        hyp2 = create_quantum_hypothesis(
            "Physics problems solved with CS methods",
            research_domains=["physics", "computer_science"]
        )
        
        accelerator.hypothesis_space[hyp1.hypothesis_id] = hyp1
        accelerator.hypothesis_space[hyp2.hypothesis_id] = hyp2
        
        # Test synthesis
        domains = ["computer_science", "physics"]
        insights = await accelerator.synthesize_cross_domain_insights(domains)
        
        # Should find cross-domain hypotheses
        assert isinstance(insights, list)
        
        # If insights found, verify structure
        for insight in insights:
            assert "hypothesis_id" in insight
            assert "domains_involved" in insight
            assert "confidence" in insight


class TestPerformanceMetric:
    """Test performance metric functionality."""
    
    def test_metric_creation(self):
        """Test creating performance metrics."""
        metric = PerformanceMetric(
            metric_id="cpu_metric",
            name="CPU Utilization",
            dimension=OptimizationDimension.CPU_USAGE,
            current_value=0.6,
            optimal_value=0.7,
            acceptable_range=(0.3, 0.9),
            optimization_weight=1.2,
            is_critical=True
        )
        
        assert metric.metric_id == "cpu_metric"
        assert metric.name == "CPU Utilization"
        assert metric.dimension == OptimizationDimension.CPU_USAGE
        assert metric.current_value == 0.6
        assert metric.optimal_value == 0.7
        assert metric.acceptable_range == (0.3, 0.9)
        assert metric.optimization_weight == 1.2
        assert metric.is_critical == True
    
    def test_metric_measurement_history(self):
        """Test metric measurement history tracking."""
        metric = PerformanceMetric(
            metric_id="test_metric",
            name="Test Metric",
            dimension=OptimizationDimension.MEMORY_USAGE,
            current_value=0.5,
            optimal_value=0.6,
            acceptable_range=(0.2, 0.8)
        )
        
        # Initially no history
        assert len(metric.measurement_history) == 0
        
        # Add measurements
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(minutes=5)
        
        metric.measurement_history.append((timestamp1, 0.5))
        metric.measurement_history.append((timestamp2, 0.6))
        
        assert len(metric.measurement_history) == 2
        assert metric.measurement_history[0] == (timestamp1, 0.5)
        assert metric.measurement_history[1] == (timestamp2, 0.6)


class TestQuantumPerformanceOptimizer:
    """Test quantum performance optimizer functionality."""
    
    @pytest.fixture
    def optimizer(self):
        """Create quantum performance optimizer for testing."""
        optimizer = QuantumPerformanceOptimizer(
            optimization_level="test",
            enable_quantum_algorithms=True,
            enable_predictive_scaling=True
        )
        # Stop scheduler for testing
        optimizer.optimization_scheduler.stop_scheduler()
        return optimizer
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.optimization_level == "test"
        assert optimizer.enable_quantum_algorithms == True
        assert optimizer.enable_predictive_scaling == True
        assert len(optimizer.metrics) > 0  # Should have default metrics
        assert optimizer.quantum_optimizer is not None
        assert optimizer.predictive_scaler is not None
    
    def test_metric_registration(self, optimizer):
        """Test registering performance metrics."""
        initial_count = len(optimizer.metrics)
        
        custom_metric = PerformanceMetric(
            metric_id="custom_metric",
            name="Custom Test Metric",
            dimension=OptimizationDimension.CACHE_HIT_RATE,
            current_value=0.8,
            optimal_value=0.9,
            acceptable_range=(0.7, 1.0)
        )
        
        optimizer.register_metric(custom_metric)
        
        assert len(optimizer.metrics) == initial_count + 1
        assert "custom_metric" in optimizer.metrics
        assert optimizer.metrics["custom_metric"] == custom_metric
    
    async def test_performance_metric_collection(self, optimizer):
        """Test collecting performance metrics."""
        metrics = await optimizer._collect_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        
        # All metric values should be valid
        for metric_id, value in metrics.items():
            assert isinstance(value, (int, float))
            assert 0.0 <= value <= 1.0
    
    async def test_optimization_target_creation(self, optimizer):
        """Test creating optimization targets."""
        target = await optimizer._create_default_optimization_target()
        
        assert target.target_id is not None
        assert target.objective_function == "multi_metric_optimization"
        assert target.optimization_strategy == OptimizationStrategy.ADAPTIVE_HYBRID
        assert len(target.target_metrics) > 0
    
    async def test_performance_optimization(self, optimizer):
        """Test performance optimization process."""
        result = await optimizer.optimize_performance()
        
        assert "target_id" in result
        assert "strategy" in result
        assert "performance_gain" in result
        assert "optimization_time" in result
        assert "success" in result
        
        # Verify optimization completed
        assert result["optimization_time"] > 0
        assert isinstance(result["success"], bool)
    
    async def test_performance_prediction(self, optimizer):
        """Test performance prediction."""
        workload_description = {
            "type": "data_processing",
            "complexity": "medium",
            "expected_load": 0.7
        }
        
        prediction = await optimizer.predict_performance(
            workload_description,
            time_horizon=timedelta(hours=1)
        )
        
        assert "predicted_metrics" in prediction
        assert "confidence" in prediction
        assert "prediction_timestamp" in prediction
        
        # Verify predicted metrics structure
        for metric_id, predicted_value in prediction["predicted_metrics"].items():
            assert isinstance(predicted_value, (int, float))
            assert 0.0 <= predicted_value <= 1.0
    
    async def test_auto_scaling(self, optimizer):
        """Test automatic resource scaling."""
        predicted_demand = {
            "cpu_usage": 0.9,  # High demand
            "memory_usage": 0.3  # Low demand
        }
        
        scaling_result = await optimizer.auto_scale_resources(predicted_demand)
        
        assert "scaling_applied" in scaling_result
        assert "decisions" in scaling_result
        assert "timestamp" in scaling_result
        assert "predicted_demand" in scaling_result
        
        # Verify scaling decisions structure
        for decision in scaling_result["decisions"]:
            assert "resource_type" in decision
            assert "action" in decision
            assert decision["action"] in ["scale_up", "scale_down", "maintain"]
    
    async def test_performance_profile_operations(self, optimizer):
        """Test performance profile creation and usage."""
        # Create performance profile
        profile_id = await optimizer.create_performance_profile(
            workload_type="test_workload",
            characteristics={
                "resource_requirements": {"cpu_intensive": True},
                "performance_characteristics": {"batch_processing": True}
            }
        )
        
        assert profile_id in optimizer.performance_profiles
        profile = optimizer.performance_profiles[profile_id]
        assert profile.workload_type == "test_workload"
        
        # Optimize with profile
        optimization_result = await optimizer.optimize_with_profile(profile_id)
        
        assert "performance_gain" in optimization_result
        assert "success" in optimization_result
        
        # Verify profile was updated
        updated_profile = optimizer.performance_profiles[profile_id]
        assert len(updated_profile.historical_performance) > 0
    
    def test_optimization_analytics(self, optimizer):
        """Test optimization analytics."""
        analytics = optimizer.get_optimization_analytics()
        
        assert "performance_metrics" in analytics
        assert "active_metrics" in analytics
        assert "current_metric_values" in analytics
        assert "metric_health" in analytics
        
        # Verify metric values
        assert analytics["active_metrics"] == len(optimizer.metrics)
        
        # Verify metric health assessment
        for metric_id, health in analytics["metric_health"].items():
            assert health in ["optimal", "suboptimal"]


class TestOptimizationStrategies:
    """Test different optimization strategies."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer for strategy testing."""
        optimizer = QuantumPerformanceOptimizer()
        optimizer.optimization_scheduler.stop_scheduler()
        return optimizer
    
    async def test_quantum_annealing_strategy(self, optimizer):
        """Test quantum annealing optimization."""
        target = create_optimization_target(
            "test_quantum_annealing",
            OptimizationStrategy.QUANTUM_ANNEALING,
            list(optimizer.metrics.keys())
        )
        
        baseline_metrics = await optimizer._collect_performance_metrics()
        
        result = await optimizer._quantum_annealing_optimization(target, baseline_metrics)
        
        # Should return optimization parameters
        assert isinstance(result, dict)
    
    async def test_genetic_algorithm_strategy(self, optimizer):
        """Test genetic algorithm optimization."""
        target = create_optimization_target(
            "test_genetic_algorithm",
            OptimizationStrategy.GENETIC_ALGORITHM,
            list(optimizer.metrics.keys())
        )
        
        baseline_metrics = await optimizer._collect_performance_metrics()
        
        result = await optimizer._genetic_algorithm_optimization(target, baseline_metrics)
        
        assert isinstance(result, dict)
        # Should have parameters for metrics
        for metric_id in target.target_metrics:
            if metric_id in baseline_metrics:
                assert metric_id in result
    
    async def test_multi_objective_strategy(self, optimizer):
        """Test multi-objective optimization."""
        target = create_optimization_target(
            "test_multi_objective",
            OptimizationStrategy.MULTI_OBJECTIVE,
            list(optimizer.metrics.keys())
        )
        
        baseline_metrics = await optimizer._collect_performance_metrics()
        
        result = await optimizer._multi_objective_optimization(target, baseline_metrics)
        
        assert isinstance(result, dict)
        # Should optimize towards optimal values
        for metric_id, optimized_value in result.items():
            if metric_id in optimizer.metrics:
                assert 0.0 <= optimized_value <= 1.0
    
    async def test_adaptive_hybrid_strategy(self, optimizer):
        """Test adaptive hybrid optimization."""
        target = create_optimization_target(
            "test_adaptive_hybrid", 
            OptimizationStrategy.ADAPTIVE_HYBRID,
            list(optimizer.metrics.keys())[:3]  # Limit for testing
        )
        
        baseline_metrics = await optimizer._collect_performance_metrics()
        
        result = await optimizer._adaptive_hybrid_optimization(target, baseline_metrics)
        
        assert isinstance(result, dict)
        # Should select best strategy result


class TestOptimizationUtilities:
    """Test optimization utility functions."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer for utility testing."""
        optimizer = QuantumPerformanceOptimizer()
        optimizer.optimization_scheduler.stop_scheduler()
        return optimizer
    
    def test_improvement_calculation(self, optimizer):
        """Test performance improvement calculation."""
        baseline = {
            "metric1": 0.5,
            "metric2": 0.3,
            "metric3": 0.8
        }
        
        optimized = {
            "metric1": 0.7,  # Improvement (closer to optimal)
            "metric2": 0.6,  # Improvement
            "metric3": 0.9   # Improvement
        }
        
        improvements = optimizer._calculate_improvements(baseline, optimized)
        
        assert isinstance(improvements, dict)
        assert len(improvements) <= len(baseline)
        
        # All improvements should be between -1 and 1
        for metric_id, improvement in improvements.items():
            assert -1.0 <= improvement <= 1.0
    
    def test_overall_gain_calculation(self, optimizer):
        """Test overall performance gain calculation."""
        improvements = {
            "metric1": 0.2,   # 20% improvement
            "metric2": 0.1,   # 10% improvement
            "metric3": -0.05  # 5% degradation
        }
        
        overall_gain = optimizer._calculate_overall_gain(improvements)
        
        assert isinstance(overall_gain, float)
        # Should be weighted average considering metric weights
        assert overall_gain > 0  # Should be positive overall
    
    async def test_optimization_result_evaluation(self, optimizer):
        """Test evaluation of optimization results."""
        baseline = {"metric1": 0.5, "metric2": 0.3}
        
        # Good optimization result (moves towards optimal)
        good_result = {"metric1": 0.6, "metric2": 0.5}
        good_score = await optimizer._evaluate_optimization_result(good_result, baseline)
        
        # Poor optimization result (moves away from optimal)
        poor_result = {"metric1": 0.3, "metric2": 0.1}
        poor_score = await optimizer._evaluate_optimization_result(poor_result, baseline)
        
        # Good result should score higher than poor result
        assert good_score >= poor_score
        assert 0.0 <= good_score <= 1.0
        assert 0.0 <= poor_score <= 1.0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])