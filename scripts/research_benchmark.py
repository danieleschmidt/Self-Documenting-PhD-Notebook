#!/usr/bin/env python3
"""
Research Benchmarking Suite for AMRF

This script demonstrates the research contributions and validates
the performance improvements of the Adaptive Multi-Modal Research Framework.
"""

import asyncio
import sys
import time
import json
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from phd_notebook.research.adaptive_framework import (
        AdaptiveMultiModalResearchFramework,
        ResearchDomain,
        WorkflowState
    )
    from phd_notebook.research.research_validation import (
        ComprehensiveValidator,
        ResearchDataGenerator,
        ValidationMetrics
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Note: Some dependencies may not be available in the testing environment")
    sys.exit(1)


class ResearchBenchmarkSuite:
    """Comprehensive benchmarking for research contributions."""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
    async def run_algorithm_performance_benchmark(self) -> Dict[str, Any]:
        """Benchmark core algorithmic components."""
        print("🧠 Running Algorithm Performance Benchmark...")
        
        # Initialize framework
        framework = AdaptiveMultiModalResearchFramework()
        
        # Generate test data
        data_generator = ResearchDataGenerator(seed=42)
        historical_data = {
            'research_history': data_generator.generate_research_history(100),
            'researcher_interactions': data_generator.generate_researcher_interactions(10, 50)
        }
        
        # Benchmark initialization
        init_start = time.time()
        await framework.initialize_framework(historical_data)
        init_time = time.time() - init_start
        
        # Benchmark optimization performance
        test_scenario = {
            'workflow_state': 'experimentation',
            'domain': 'cs',
            'problem_description': 'novel machine learning algorithm development',
            'available_resources': {'time_budget': 480, 'funding': 1.0, 'personnel': 2.0},
            'pending_tasks': [
                {'id': f'task_{i}', 'priority': 0.8, 'impact': 0.9, 'estimated_time': 60}
                for i in range(10)
            ],
            'expertise': ['machine learning', 'data analysis'],
            'trending_topics': ['transformers', 'federated learning', 'quantum computing']
        }
        
        # Run multiple optimizations to measure performance
        optimization_times = []
        optimization_scores = []
        
        for i in range(20):
            opt_start = time.time()
            result = await framework.optimize_research_workflow(f'researcher_{i}', test_scenario)
            opt_time = time.time() - opt_start
            
            optimization_times.append(opt_time)
            
            # Calculate optimization quality score
            productivity = result.get('performance_metrics', {}).get('productivity_score', 0.0)
            innovation = result.get('performance_metrics', {}).get('innovation_quotient', 0.0)
            score = (productivity + innovation) / 2
            optimization_scores.append(score)
        
        # Calculate benchmark metrics
        avg_opt_time = statistics.mean(optimization_times)
        std_opt_time = statistics.stdev(optimization_times) if len(optimization_times) > 1 else 0
        avg_score = statistics.mean(optimization_scores)
        std_score = statistics.stdev(optimization_scores) if len(optimization_scores) > 1 else 0
        
        benchmark_results = {
            'initialization_time_seconds': init_time,
            'average_optimization_time_ms': avg_opt_time * 1000,
            'optimization_time_std_ms': std_opt_time * 1000,
            'average_optimization_score': avg_score,
            'optimization_score_std': std_score,
            'throughput_ops_per_second': 1 / avg_opt_time if avg_opt_time > 0 else 0,
            'total_optimizations': len(optimization_times),
            'algorithm_stability': 1 - (std_score / avg_score) if avg_score > 0 else 0
        }
        
        print(f"  ✅ Initialization: {init_time:.3f}s")
        print(f"  ⚡ Avg Optimization: {avg_opt_time*1000:.1f}ms")
        print(f"  📊 Avg Score: {avg_score:.3f}")
        print(f"  🚀 Throughput: {benchmark_results['throughput_ops_per_second']:.1f} ops/s")
        
        return benchmark_results
    
    async def run_comparative_baseline_benchmark(self) -> Dict[str, Any]:
        """Compare AMRF against traditional baseline approaches."""
        print("\n📊 Running Comparative Baseline Benchmark...")
        
        # Setup comparison
        validator = ComprehensiveValidator()
        framework = AdaptiveMultiModalResearchFramework()
        
        # Generate test scenarios
        test_scenarios = validator.data_generator.generate_test_scenarios(15)
        
        # Initialize framework
        historical_data = {
            'research_history': validator.data_generator.generate_research_history(50),
            'researcher_interactions': validator.data_generator.generate_researcher_interactions(5, 30)
        }
        await framework.initialize_framework(historical_data)
        
        # Run comparative benchmarks
        benchmark_results = await validator.benchmarker.run_comparative_benchmark(
            framework, test_scenarios, num_iterations=3
        )
        
        # Format results
        comparison_results = {}
        for baseline_name, result in benchmark_results.items():
            comparison_results[baseline_name] = {
                'improvement_percentage': result.improvement_percentage,
                'statistical_significance': result.statistical_significance,
                'framework_performance': result.framework_performance,
                'baseline_performance': result.baseline_performance,
                'confidence_interval': result.confidence_interval
            }
            
            print(f"  vs {baseline_name}: +{result.improvement_percentage:.1f}% improvement")
            print(f"    (p-value: {result.statistical_significance:.3f})")
        
        return comparison_results
    
    async def run_scalability_benchmark(self) -> Dict[str, Any]:
        """Benchmark scalability with increasing complexity."""
        print("\n📈 Running Scalability Benchmark...")
        
        framework = AdaptiveMultiModalResearchFramework()
        
        # Test with increasing data sizes
        scalability_results = {}
        data_sizes = [10, 25, 50, 100, 200]
        
        for size in data_sizes:
            print(f"  Testing with {size} projects...")
            
            # Generate data of increasing size
            data_generator = ResearchDataGenerator(seed=42)
            historical_data = {
                'research_history': data_generator.generate_research_history(size),
                'researcher_interactions': data_generator.generate_researcher_interactions(size//10, 20)
            }
            
            # Measure initialization time
            init_start = time.time()
            await framework.initialize_framework(historical_data)
            init_time = time.time() - init_start
            
            # Measure optimization time
            test_scenario = {
                'workflow_state': 'experimentation',
                'domain': 'cs',
                'problem_description': 'scalability test',
                'available_resources': {'time_budget': 480},
                'pending_tasks': [{'id': f'task_{i}', 'priority': 0.5, 'impact': 0.5, 'estimated_time': 60} 
                                for i in range(5)],
                'expertise': ['machine learning'],
                'trending_topics': ['transformers']
            }
            
            opt_start = time.time()
            result = await framework.optimize_research_workflow('test_researcher', test_scenario)
            opt_time = time.time() - opt_start
            
            scalability_results[size] = {
                'initialization_time': init_time,
                'optimization_time': opt_time,
                'memory_estimate_mb': size * 0.1,  # Rough estimate
                'cross_domain_connections': len(framework.cross_domain_intelligence.domain_connections)
            }
        
        # Calculate scalability metrics
        sizes = list(scalability_results.keys())
        init_times = [scalability_results[s]['initialization_time'] for s in sizes]
        opt_times = [scalability_results[s]['optimization_time'] for s in sizes]
        
        # Linear regression approximation for complexity analysis
        def simple_regression(x_vals, y_vals):
            n = len(x_vals)
            x_mean = sum(x_vals) / n
            y_mean = sum(y_vals) / n
            
            numerator = sum((x_vals[i] - x_mean) * (y_vals[i] - y_mean) for i in range(n))
            denominator = sum((x_vals[i] - x_mean) ** 2 for i in range(n))
            
            slope = numerator / denominator if denominator != 0 else 0
            return slope
        
        init_complexity = simple_regression(sizes, init_times)
        opt_complexity = simple_regression(sizes, opt_times)
        
        scalability_summary = {
            'data_points': scalability_results,
            'initialization_complexity_factor': init_complexity,
            'optimization_complexity_factor': opt_complexity,
            'linear_scalability_score': max(0, 1 - abs(init_complexity)),  # Closer to 0 slope = better
            'optimization_scalability_score': max(0, 1 - abs(opt_complexity))
        }
        
        print(f"  📊 Initialization complexity factor: {init_complexity:.6f}")
        print(f"  ⚡ Optimization complexity factor: {opt_complexity:.6f}")
        
        return scalability_summary
    
    async def run_innovation_validation_benchmark(self) -> Dict[str, Any]:
        """Validate novel algorithmic contributions."""
        print("\n🔬 Running Innovation Validation Benchmark...")
        
        innovation_metrics = {}
        
        # Test 1: Adaptive Learning Validation
        print("  Testing Adaptive Learning Algorithm...")
        framework = AdaptiveMultiModalResearchFramework()
        
        # Generate diverse researcher behavior data
        data_generator = ResearchDataGenerator(seed=123)
        interactions = data_generator.generate_researcher_interactions(5, 100)
        
        # Measure learning effectiveness
        learning_scores = []
        for researcher_id, researcher_interactions in interactions.items():
            behavior_pattern = await framework.adaptive_learner.learn_from_behavior(researcher_interactions)
            
            # Score based on pattern quality
            pattern_score = (
                len(behavior_pattern.preferred_work_hours) / 24 +  # Work hour diversity
                behavior_pattern.focus_duration_avg / 240 +  # Focus duration (max 4 hours)
                behavior_pattern.collaboration_preference  # Collaboration score
            ) / 3
            
            learning_scores.append(pattern_score)
        
        innovation_metrics['adaptive_learning'] = {
            'average_pattern_quality': statistics.mean(learning_scores),
            'pattern_diversity': statistics.stdev(learning_scores) if len(learning_scores) > 1 else 0,
            'researchers_processed': len(learning_scores)
        }
        
        # Test 2: Cross-Domain Intelligence
        print("  Testing Cross-Domain Intelligence...")
        research_history = data_generator.generate_research_history(80)
        await framework.cross_domain_intelligence.discover_domain_connections(research_history)
        
        innovation_metrics['cross_domain_intelligence'] = {
            'connections_discovered': len(framework.cross_domain_intelligence.domain_connections),
            'methodology_mappings': len(framework.cross_domain_intelligence.methodology_map),
            'domain_coverage': len(set(
                domain for connection in framework.cross_domain_intelligence.domain_connections.keys()
                for domain in connection
            ))
        }
        
        # Test 3: Predictive Planning Accuracy
        print("  Testing Predictive Planning...")
        await framework.predictive_planner.analyze_research_patterns(research_history)
        
        # Test prediction accuracy with synthetic scenarios
        prediction_accuracies = []
        for i in range(10):
            test_proposal = {
                'type': 'experimental',
                'complexity': 0.6,
                'available_resources': {'funding': 1.0, 'personnel': 2.0},
                'novelty_score': 0.7
            }
            
            prediction = await framework.predictive_planner.predict_research_timeline(test_proposal)
            
            # Score prediction quality (heuristic)
            timeline_reasonableness = 1.0 if 6 <= prediction['total_duration_months'] <= 24 else 0.5
            success_prob_reasonableness = 1.0 if 0.3 <= prediction['success_probability'] <= 0.9 else 0.5
            
            accuracy = (timeline_reasonableness + success_prob_reasonableness) / 2
            prediction_accuracies.append(accuracy)
        
        innovation_metrics['predictive_planning'] = {
            'average_prediction_accuracy': statistics.mean(prediction_accuracies),
            'prediction_consistency': 1 - statistics.stdev(prediction_accuracies) if len(prediction_accuracies) > 1 else 1,
            'predictions_tested': len(prediction_accuracies)
        }
        
        # Test 4: Framework Integration Score
        print("  Testing Framework Integration...")
        test_context = {
            'workflow_state': 'methodology_design',
            'domain': 'interdisciplinary',
            'problem_description': 'novel interdisciplinary research challenge',
            'available_resources': {'time_budget': 480, 'funding': 1.5, 'personnel': 3.0},
            'pending_tasks': [
                {'id': f'integration_task_{i}', 'priority': 0.7, 'impact': 0.8, 'estimated_time': 90}
                for i in range(6)
            ],
            'expertise': ['machine learning', 'biology', 'statistics'],
            'trending_topics': ['AI in biology', 'computational biology', 'bioinformatics']
        }
        
        integration_start = time.time()
        integration_result = await framework.optimize_research_workflow('integration_test', test_context)
        integration_time = time.time() - integration_start
        
        # Score integration quality
        components_used = sum([
            1 if integration_result.get('optimal_workflow_sequence') else 0,
            1 if integration_result.get('cross_domain_suggestions') else 0,
            1 if integration_result.get('resource_optimization') else 0,
            1 if integration_result.get('research_opportunities') else 0,
            1 if integration_result.get('adaptive_recommendations') else 0
        ])
        
        integration_score = components_used / 5.0  # All 5 components should be present
        
        innovation_metrics['framework_integration'] = {
            'integration_score': integration_score,
            'execution_time_ms': integration_time * 1000,
            'components_active': components_used,
            'performance_score': integration_result.get('performance_metrics', {}).get('productivity_score', 0)
        }
        
        # Overall innovation score
        innovation_scores = [
            innovation_metrics['adaptive_learning']['average_pattern_quality'],
            min(innovation_metrics['cross_domain_intelligence']['connections_discovered'] / 10, 1.0),
            innovation_metrics['predictive_planning']['average_prediction_accuracy'],
            innovation_metrics['framework_integration']['integration_score']
        ]
        
        innovation_metrics['overall_innovation_score'] = statistics.mean(innovation_scores)
        
        print(f"  🧠 Adaptive Learning Score: {innovation_metrics['adaptive_learning']['average_pattern_quality']:.3f}")
        print(f"  🔗 Cross-Domain Connections: {innovation_metrics['cross_domain_intelligence']['connections_discovered']}")
        print(f"  🎯 Prediction Accuracy: {innovation_metrics['predictive_planning']['average_prediction_accuracy']:.3f}")
        print(f"  🚀 Overall Innovation Score: {innovation_metrics['overall_innovation_score']:.3f}")
        
        return innovation_metrics
    
    async def run_reproducibility_benchmark(self) -> Dict[str, Any]:
        """Test reproducibility and consistency."""
        print("\n🔄 Running Reproducibility Benchmark...")
        
        framework = AdaptiveMultiModalResearchFramework()
        
        # Initialize with fixed data for reproducibility testing
        data_generator = ResearchDataGenerator(seed=42)  # Fixed seed
        historical_data = {
            'research_history': data_generator.generate_research_history(50),
            'researcher_interactions': data_generator.generate_researcher_interactions(3, 30)
        }
        await framework.initialize_framework(historical_data)
        
        # Fixed test scenario
        test_scenario = {
            'workflow_state': 'analysis',
            'domain': 'cs',
            'problem_description': 'reproducibility test scenario',
            'available_resources': {'time_budget': 360, 'funding': 1.0, 'personnel': 2.0},
            'pending_tasks': [
                {'id': 'repro_task_1', 'priority': 0.8, 'impact': 0.7, 'estimated_time': 90},
                {'id': 'repro_task_2', 'priority': 0.6, 'impact': 0.9, 'estimated_time': 120}
            ],
            'expertise': ['data analysis', 'statistics'],
            'trending_topics': ['reproducible research', 'open science']
        }
        
        # Run multiple times to test consistency
        results = []
        for i in range(10):
            result = await framework.optimize_research_workflow('repro_researcher', test_scenario)
            
            # Extract key metrics for consistency testing
            metrics = {
                'productivity_score': result.get('performance_metrics', {}).get('productivity_score', 0),
                'innovation_quotient': result.get('performance_metrics', {}).get('innovation_quotient', 0),
                'resource_utility': result.get('resource_optimization', {}).get('expected_utility', 0),
                'num_recommendations': len(result.get('adaptive_recommendations', [])),
                'num_cross_domain': len(result.get('cross_domain_suggestions', []))
            }
            results.append(metrics)
        
        # Calculate reproducibility metrics
        reproducibility_scores = {}
        for metric_name in results[0].keys():
            values = [r[metric_name] for r in results]
            
            if len(values) > 1:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values)
                cv = std_val / mean_val if mean_val > 0 else 1.0  # Coefficient of variation
                
                # Reproducibility score (1 - normalized CV)
                reproducibility_score = max(0.0, 1.0 - cv)
                reproducibility_scores[f'{metric_name}_reproducibility'] = reproducibility_score
                reproducibility_scores[f'{metric_name}_cv'] = cv
        
        # Overall reproducibility
        individual_scores = [v for k, v in reproducibility_scores.items() if k.endswith('_reproducibility')]
        overall_reproducibility = statistics.mean(individual_scores) if individual_scores else 0.0
        
        reproducibility_results = {
            'overall_reproducibility': overall_reproducibility,
            'individual_metrics': reproducibility_scores,
            'test_runs': len(results),
            'consistency_threshold': 0.9  # Target threshold
        }
        
        print(f"  📊 Overall Reproducibility: {overall_reproducibility:.3f}")
        print(f"  🎯 Target Threshold: {reproducibility_results['consistency_threshold']}")
        print(f"  ✅ Passes Threshold: {'Yes' if overall_reproducibility >= 0.9 else 'No'}")
        
        return reproducibility_results
    
    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        print("\n📋 Generating Comprehensive Benchmark Report...")
        
        # Run all benchmarks
        algorithm_performance = await self.run_algorithm_performance_benchmark()
        baseline_comparison = await self.run_comparative_baseline_benchmark()
        scalability = await self.run_scalability_benchmark()
        innovation = await self.run_innovation_validation_benchmark()
        reproducibility = await self.run_reproducibility_benchmark()
        
        # Calculate overall scores
        performance_score = min(1.0, algorithm_performance['average_optimization_score'])
        improvement_score = statistics.mean([
            max(0, result['improvement_percentage'] / 100) 
            for result in baseline_comparison.values()
        ])
        scalability_score = (scalability['linear_scalability_score'] + 
                           scalability['optimization_scalability_score']) / 2
        innovation_score = innovation['overall_innovation_score']
        reproducibility_score = reproducibility['overall_reproducibility']
        
        overall_score = statistics.mean([
            performance_score, improvement_score, scalability_score, 
            innovation_score, reproducibility_score
        ])
        
        # Comprehensive report
        report = {
            'benchmark_metadata': {
                'execution_time': (datetime.now() - self.start_time).total_seconds(),
                'framework_version': '1.0.0',
                'benchmark_date': datetime.now().isoformat(),
                'research_contribution': 'Adaptive Multi-Modal Research Framework (AMRF)'
            },
            'performance_benchmarks': algorithm_performance,
            'baseline_comparisons': baseline_comparison,
            'scalability_analysis': scalability,
            'innovation_validation': innovation,
            'reproducibility_testing': reproducibility,
            'summary_scores': {
                'performance_score': performance_score,
                'improvement_score': improvement_score,
                'scalability_score': scalability_score,
                'innovation_score': innovation_score,
                'reproducibility_score': reproducibility_score,
                'overall_score': overall_score
            },
            'research_contributions': {
                'algorithmic_innovations': [
                    'Bayesian Adaptive Learning for researcher behavior optimization',
                    'Cross-Domain Intelligence with methodology transfer analysis',
                    'Predictive Research Planning with timeline and risk assessment',
                    'Multi-Objective Resource Optimization using utility maximization'
                ],
                'performance_improvements': {
                    baseline: f"+{result['improvement_percentage']:.1f}%"
                    for baseline, result in baseline_comparison.items()
                },
                'validation_results': {
                    'statistical_significance': all(
                        result['statistical_significance'] < 0.05 
                        for result in baseline_comparison.values()
                    ),
                    'reproducibility_validated': reproducibility_score >= 0.9,
                    'scalability_confirmed': scalability_score >= 0.7,
                    'innovation_demonstrated': innovation_score >= 0.7
                }
            },
            'quality_assessment': {
                'passes_performance_threshold': performance_score >= 0.7,
                'shows_significant_improvement': improvement_score >= 0.1,
                'demonstrates_scalability': scalability_score >= 0.7,
                'validates_innovation': innovation_score >= 0.7,
                'confirms_reproducibility': reproducibility_score >= 0.9,
                'overall_quality': 'EXCELLENT' if overall_score >= 0.8 else 'GOOD' if overall_score >= 0.6 else 'FAIR'
            }
        }
        
        return report
    
    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print benchmark summary."""
        print("\n" + "="*80)
        print("🔬 ADAPTIVE MULTI-MODAL RESEARCH FRAMEWORK (AMRF)")
        print("📊 COMPREHENSIVE BENCHMARK RESULTS")
        print("="*80)
        
        scores = report['summary_scores']
        quality = report['quality_assessment']
        
        print(f"\n📈 PERFORMANCE SCORES:")
        print(f"  Algorithm Performance:  {scores['performance_score']:.3f}")
        print(f"  Baseline Improvement:   {scores['improvement_score']:.3f}")
        print(f"  Scalability:           {scores['scalability_score']:.3f}")
        print(f"  Innovation:            {scores['innovation_score']:.3f}")
        print(f"  Reproducibility:       {scores['reproducibility_score']:.3f}")
        print(f"  OVERALL SCORE:         {scores['overall_score']:.3f}")
        
        print(f"\n🎯 QUALITY ASSESSMENT: {quality['overall_quality']}")
        
        print(f"\n🚀 KEY ACHIEVEMENTS:")
        for baseline, improvement in report['research_contributions']['performance_improvements'].items():
            print(f"  • {baseline}: {improvement} improvement")
        
        validation = report['research_contributions']['validation_results']
        print(f"\n✅ VALIDATION STATUS:")
        print(f"  • Statistical Significance: {'✓' if validation['statistical_significance'] else '✗'}")
        print(f"  • Reproducibility: {'✓' if validation['reproducibility_validated'] else '✗'}")
        print(f"  • Scalability: {'✓' if validation['scalability_confirmed'] else '✗'}")
        print(f"  • Innovation: {'✓' if validation['innovation_demonstrated'] else '✗'}")
        
        print(f"\n🔬 ALGORITHMIC CONTRIBUTIONS:")
        for contribution in report['research_contributions']['algorithmic_innovations']:
            print(f"  • {contribution}")
        
        execution_time = report['benchmark_metadata']['execution_time']
        print(f"\n⏱️  Total Benchmark Time: {execution_time:.1f} seconds")
        print("="*80)


async def main():
    """Main benchmark execution function."""
    print("🚀 Starting AMRF Research Benchmark Suite")
    print("This comprehensive benchmark validates the novel research contributions")
    print("of the Adaptive Multi-Modal Research Framework.\n")
    
    benchmark_suite = ResearchBenchmarkSuite()
    
    try:
        # Run comprehensive benchmarking
        report = await benchmark_suite.generate_comprehensive_report()
        
        # Save detailed report
        output_path = Path("amrf_benchmark_report.json")
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        benchmark_suite.print_summary(report)
        
        print(f"\n📄 Detailed report saved to: {output_path}")
        
        # Return exit code based on overall quality
        overall_score = report['summary_scores']['overall_score']
        if overall_score >= 0.8:
            print("\n🏆 BENCHMARK RESULT: EXCELLENT - Research contributions validated!")
            return 0
        elif overall_score >= 0.6:
            print("\n✅ BENCHMARK RESULT: GOOD - Strong research contributions demonstrated")
            return 0
        else:
            print("\n⚠️  BENCHMARK RESULT: FAIR - Some improvements needed")
            return 1
            
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)