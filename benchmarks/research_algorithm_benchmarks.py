#!/usr/bin/env python3
"""
Research Algorithm Benchmarks
============================

Comprehensive benchmarking suite for novel research algorithms.
Tests performance, accuracy, and scalability of advanced research features.
"""

import asyncio
import time
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phd_notebook.research.advanced_algorithms import (
    KnowledgeGraphNeuralNetwork,
    AutomatedLiteratureSynthesis,
    ExperimentalDesignOptimizer,
    MultiObjectiveResearchPlanner,
    ResearchOpportunity,
    ResearchVector,
    ResearchPhase,
    enhance_research_workflow
)


class ResearchBenchmarkSuite:
    """Comprehensive benchmark suite for research algorithms."""
    
    def __init__(self):
        self.results = {}
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self) -> Dict[str, Any]:
        """Generate comprehensive test data for benchmarking."""
        
        # Generate test papers
        papers = []
        research_topics = [
            "machine learning", "neural networks", "deep learning", "natural language processing",
            "computer vision", "reinforcement learning", "optimization", "algorithms",
            "data mining", "artificial intelligence", "robotics", "quantum computing"
        ]
        
        for i in range(200):  # 200 test papers
            topic = random.choice(research_topics)
            paper = {
                "id": f"paper_{i}",
                "title": f"Study on {topic} method {i}",
                "content": f"This paper presents a novel approach to {topic}. " * random.randint(10, 50),
                "type": "literature",
                "created": (datetime.now() - timedelta(days=random.randint(1, 1000))).isoformat(),
                "tags": [f"#{topic.replace(' ', '_')}", "#research", "#academic"],
                "related_papers": [f"paper_{j}" for j in random.sample(range(200), random.randint(0, 5))],
                "collaborators": [f"researcher_{j}" for j in random.sample(range(20), random.randint(0, 3))]
            }
            papers.append(paper)
        
        # Generate research opportunities
        opportunities = []
        for i in range(50):  # 50 research opportunities
            vector = ResearchVector(
                novelty=random.uniform(0.3, 1.0),
                feasibility=random.uniform(0.2, 0.9),
                impact=random.uniform(0.4, 1.0),
                resources=random.uniform(0.3, 0.8),
                timeline=random.uniform(0.4, 0.9),
                collaboration=random.uniform(0.2, 0.8)
            )
            
            opportunity = ResearchOpportunity(
                id=f"opp_{i}",
                title=f"Research Opportunity {i}",
                description=f"Investigation of novel approach to {random.choice(research_topics)}",
                vector=vector,
                phase=random.choice(list(ResearchPhase)),
                estimated_duration=random.randint(6, 24),
                confidence=random.uniform(0.5, 0.9)
            )
            opportunities.append(opportunity)
        
        return {
            "papers": papers,
            "opportunities": [
                {
                    "id": opp.id,
                    "title": opp.title,
                    "description": opp.description,
                    "vector": {
                        "novelty": opp.vector.novelty,
                        "feasibility": opp.vector.feasibility,
                        "impact": opp.vector.impact,
                        "resources": opp.vector.resources,
                        "timeline": opp.vector.timeline,
                        "collaboration": opp.vector.collaboration
                    },
                    "phase": opp.phase.value,
                    "estimated_duration": opp.estimated_duration,
                    "confidence": opp.confidence
                } for opp in opportunities
            ],
            "experiment_constraints": {
                "min_sample_size": 20,
                "max_sample_size": 500,
                "min_duration": 4,
                "max_duration": 26,
                "max_conditions": 5,
                "budget": 50000
            },
            "experiment_objectives": ["power", "efficiency", "validity", "feasibility"],
            "planning_constraints": {
                "max_concurrent_projects": 3,
                "max_total_months": 36,
                "budget": 100000,
                "objective_weights": {
                    "novelty": 0.25,
                    "impact": 0.30,
                    "feasibility": 0.20,
                    "efficiency": 0.15,
                    "collaboration": 0.10
                }
            }
        }
    
    async def run_knowledge_graph_benchmark(self) -> Dict[str, Any]:
        """Benchmark knowledge graph neural network performance."""
        print("🧠 Benchmarking Knowledge Graph Neural Network...")
        
        kg_nn = KnowledgeGraphNeuralNetwork(embedding_dim=256, num_heads=8)
        results = {}
        
        # Test embedding generation performance
        start_time = time.time()
        embeddings = []
        
        for paper in self.test_data["papers"][:100]:  # Test with 100 papers
            embedding = kg_nn.create_research_embedding(paper)
            embeddings.append(embedding)
        
        embedding_time = time.time() - start_time
        
        # Test attention computation performance
        start_time = time.time()
        source_embedding = embeddings[0]
        target_embeddings = embeddings[1:11]  # 10 targets
        
        attention_weights = kg_nn.compute_attention_weights(source_embedding, target_embeddings)
        attention_time = time.time() - start_time
        
        # Validate embedding properties
        embedding_dims = [len(emb) for emb in embeddings]
        avg_magnitude = sum(sum(x**2 for x in emb)**0.5 for emb in embeddings) / len(embeddings)
        
        results = {
            "embedding_generation": {
                "time_per_paper_ms": (embedding_time / 100) * 1000,
                "total_time_ms": embedding_time * 1000,
                "papers_processed": 100,
                "embedding_dimension": embedding_dims[0] if embedding_dims else 0,
                "avg_magnitude": avg_magnitude
            },
            "attention_computation": {
                "time_ms": attention_time * 1000,
                "source_targets": 10,
                "weights_sum": sum(attention_weights) if attention_weights else 0,
                "max_weight": max(attention_weights) if attention_weights else 0
            },
            "validation": {
                "consistent_dimensions": len(set(embedding_dims)) == 1,
                "non_zero_embeddings": all(any(x != 0 for x in emb) for emb in embeddings[:10]),
                "valid_attention_weights": abs(sum(attention_weights) - 1.0) < 0.01 if attention_weights else False
            }
        }
        
        return results
    
    async def run_literature_synthesis_benchmark(self) -> Dict[str, Any]:
        """Benchmark automated literature synthesis performance."""
        print("📚 Benchmarking Automated Literature Synthesis...")
        
        kg_nn = KnowledgeGraphNeuralNetwork()
        lit_synth = AutomatedLiteratureSynthesis(kg_nn)
        results = {}
        
        # Test with different paper set sizes
        test_sizes = [10, 25, 50, 100]
        
        for size in test_sizes:
            print(f"  Testing with {size} papers...")
            test_papers = self.test_data["papers"][:size]
            
            start_time = time.time()
            synthesis_result = await lit_synth.synthesize_literature(test_papers)
            processing_time = time.time() - start_time
            
            results[f"size_{size}"] = {
                "processing_time_ms": processing_time * 1000,
                "papers_per_second": size / processing_time if processing_time > 0 else 0,
                "clusters_found": len(synthesis_result.get("clusters", [])),
                "gaps_identified": len(synthesis_result.get("gaps", [])),
                "synthesis_length": len(synthesis_result.get("synthesis", "")),
                "quality_metrics": {
                    "has_synthesis": bool(synthesis_result.get("synthesis")),
                    "has_clusters": len(synthesis_result.get("clusters", [])) > 0,
                    "has_gaps": len(synthesis_result.get("gaps", [])) > 0,
                    "reasonable_clusters": 1 <= len(synthesis_result.get("clusters", [])) <= size // 2
                }
            }
        
        # Scalability analysis
        sizes = [r for r in results.keys() if r.startswith("size_")]
        processing_times = [results[s]["processing_time_ms"] for s in sizes]
        
        # Calculate scaling factor (linear, quadratic, etc.)
        if len(processing_times) >= 2:
            scaling_ratio = processing_times[-1] / processing_times[0]
            size_ratio = test_sizes[-1] / test_sizes[0]
            scaling_factor = scaling_ratio / size_ratio
        else:
            scaling_factor = 1.0
        
        results["scalability"] = {
            "scaling_factor": scaling_factor,
            "is_linear": 0.8 <= scaling_factor <= 1.5,
            "performance_degradation": scaling_factor > 2.0
        }
        
        return results
    
    async def run_experimental_design_benchmark(self) -> Dict[str, Any]:
        """Benchmark experimental design optimizer."""
        print("⚗️ Benchmarking Experimental Design Optimizer...")
        
        optimizer = ExperimentalDesignOptimizer(population_size=50, generations=50)
        results = {}
        
        constraints = self.test_data["experiment_constraints"]
        objectives = self.test_data["experiment_objectives"]
        
        # Test optimization performance
        start_time = time.time()
        optimization_result = await optimizer.optimize_experimental_design(constraints, objectives)
        optimization_time = time.time() - start_time
        
        # Validate optimization result
        optimal_design = optimization_result.get("optimal_design", {})
        fitness_score = optimization_result.get("fitness_score", 0)
        
        # Check constraint satisfaction
        constraints_satisfied = True
        if optimal_design:
            sample_size = optimal_design.get("sample_size", 0)
            duration = optimal_design.get("duration_weeks", 0)
            
            constraints_satisfied = (
                constraints["min_sample_size"] <= sample_size <= constraints["max_sample_size"] and
                constraints["min_duration"] <= duration <= constraints["max_duration"]
            )
        
        results = {
            "optimization_performance": {
                "time_ms": optimization_time * 1000,
                "generations": optimizer.generations,
                "population_size": optimizer.population_size,
                "convergence_achieved": fitness_score > 0.5
            },
            "solution_quality": {
                "fitness_score": fitness_score,
                "constraints_satisfied": constraints_satisfied,
                "has_recommendations": len(optimization_result.get("recommendations", [])) > 0,
                "design_completeness": len(optimal_design) >= 8
            },
            "optimal_design": optimal_design
        }
        
        return results
    
    async def run_research_planning_benchmark(self) -> Dict[str, Any]:
        """Benchmark multi-objective research planner."""
        print("📋 Benchmarking Multi-Objective Research Planner...")
        
        planner = MultiObjectiveResearchPlanner()
        results = {}
        
        # Convert opportunity data to ResearchOpportunity objects
        opportunities = []
        for opp_data in self.test_data["opportunities"]:
            vector = ResearchVector(**opp_data["vector"])
            opp = ResearchOpportunity(
                id=opp_data["id"],
                title=opp_data["title"],
                description=opp_data["description"],
                vector=vector,
                phase=ResearchPhase(opp_data["phase"]),
                estimated_duration=opp_data["estimated_duration"],
                confidence=opp_data["confidence"]
            )
            opportunities.append(opp)
        
        constraints = self.test_data["planning_constraints"]
        
        # Test planning performance
        start_time = time.time()
        planning_result = await planner.optimize_research_plan(opportunities, constraints)
        planning_time = time.time() - start_time
        
        # Analyze planning result
        optimal_plan = planning_result.get("optimal_plan", [])
        pareto_front = planning_result.get("pareto_front", [])
        analysis = planning_result.get("analysis", "")
        
        # Validate plan constraints
        total_duration = sum(opp.estimated_duration for opp in optimal_plan)
        constraints_satisfied = (
            len(optimal_plan) <= constraints["max_concurrent_projects"] and
            total_duration <= constraints["max_total_months"]
        )
        
        results = {
            "planning_performance": {
                "time_ms": planning_time * 1000,
                "opportunities_evaluated": len(opportunities),
                "pareto_solutions": len(pareto_front),
                "optimization_iterations": planner.optimization_iterations
            },
            "solution_quality": {
                "plan_size": len(optimal_plan),
                "total_duration_months": total_duration,
                "constraints_satisfied": constraints_satisfied,
                "has_analysis": len(analysis) > 0,
                "pareto_front_diversity": len(set(len(sol["plan"]) for sol in pareto_front))
            },
            "plan_metrics": {
                "avg_novelty": sum(opp.vector.novelty for opp in optimal_plan) / len(optimal_plan) if optimal_plan else 0,
                "avg_impact": sum(opp.vector.impact for opp in optimal_plan) / len(optimal_plan) if optimal_plan else 0,
                "avg_feasibility": sum(opp.vector.feasibility for opp in optimal_plan) / len(optimal_plan) if optimal_plan else 0
            }
        }
        
        return results
    
    async def run_integration_benchmark(self) -> Dict[str, Any]:
        """Benchmark integrated research workflow enhancement."""
        print("🔗 Benchmarking Integrated Research Workflow...")
        
        start_time = time.time()
        
        # Run complete workflow enhancement
        enhancement_result = await enhance_research_workflow(
            self.test_data,
            config={
                "kg_config": {"embedding_dim": 128, "num_heads": 4},
                "exp_config": {"population_size": 25, "generations": 25}
            }
        )
        
        total_time = time.time() - start_time
        
        # Analyze integration results
        has_lit_synthesis = "literature_synthesis" in enhancement_result
        has_exp_design = "experimental_design" in enhancement_result
        has_research_plan = "research_plan" in enhancement_result
        has_recommendations = "recommendations" in enhancement_result
        
        results = {
            "integration_performance": {
                "total_time_ms": total_time * 1000,
                "components_executed": sum([has_lit_synthesis, has_exp_design, has_research_plan]),
                "avg_time_per_component_ms": (total_time * 1000) / max(sum([has_lit_synthesis, has_exp_design, has_research_plan]), 1)
            },
            "workflow_completeness": {
                "literature_synthesis": has_lit_synthesis,
                "experimental_design": has_exp_design,
                "research_planning": has_research_plan,
                "recommendations": has_recommendations,
                "complete_workflow": all([has_lit_synthesis, has_exp_design, has_research_plan, has_recommendations])
            },
            "output_quality": {
                "recommendations_count": len(enhancement_result.get("recommendations", [])),
                "synthesis_clusters": len(enhancement_result.get("literature_synthesis", {}).get("clusters", [])),
                "identified_gaps": len(enhancement_result.get("literature_synthesis", {}).get("gaps", [])),
                "optimal_plan_size": len(enhancement_result.get("research_plan", {}).get("optimal_plan", []))
            }
        }
        
        return results
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("🚀 Starting Research Algorithm Benchmark Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all benchmarks
        self.results["knowledge_graph"] = await self.run_knowledge_graph_benchmark()
        self.results["literature_synthesis"] = await self.run_literature_synthesis_benchmark()
        self.results["experimental_design"] = await self.run_experimental_design_benchmark()
        self.results["research_planning"] = await self.run_research_planning_benchmark()
        self.results["integration"] = await self.run_integration_benchmark()
        
        total_time = time.time() - start_time
        
        # Generate summary
        self.results["benchmark_summary"] = {
            "total_execution_time_ms": total_time * 1000,
            "timestamp": datetime.now().isoformat(),
            "test_data_size": {
                "papers": len(self.test_data["papers"]),
                "opportunities": len(self.test_data["opportunities"])
            },
            "overall_performance": self._calculate_overall_performance()
        }
        
        return self.results
    
    def _calculate_overall_performance(self) -> Dict[str, Any]:
        """Calculate overall performance metrics."""
        # Define performance targets (ms)
        targets = {
            "kg_embedding_per_paper": 10.0,  # 10ms per paper
            "literature_synthesis_100_papers": 5000.0,  # 5s for 100 papers
            "experimental_optimization": 2000.0,  # 2s for optimization
            "research_planning": 3000.0,  # 3s for planning
            "integration_workflow": 10000.0  # 10s for complete workflow
        }
        
        # Extract actual performance
        actual = {}
        
        kg_results = self.results.get("knowledge_graph", {})
        actual["kg_embedding_per_paper"] = kg_results.get("embedding_generation", {}).get("time_per_paper_ms", float('inf'))
        
        lit_results = self.results.get("literature_synthesis", {})
        actual["literature_synthesis_100_papers"] = lit_results.get("size_100", {}).get("processing_time_ms", float('inf'))
        
        exp_results = self.results.get("experimental_design", {})
        actual["experimental_optimization"] = exp_results.get("optimization_performance", {}).get("time_ms", float('inf'))
        
        plan_results = self.results.get("research_planning", {})
        actual["research_planning"] = plan_results.get("planning_performance", {}).get("time_ms", float('inf'))
        
        int_results = self.results.get("integration", {})
        actual["integration_workflow"] = int_results.get("integration_performance", {}).get("total_time_ms", float('inf'))
        
        # Calculate performance ratios
        performance_ratios = {}
        for metric in targets:
            if actual.get(metric, float('inf')) != float('inf'):
                performance_ratios[metric] = targets[metric] / actual[metric]
            else:
                performance_ratios[metric] = 0.0
        
        # Overall performance score (geometric mean)
        valid_ratios = [r for r in performance_ratios.values() if r > 0]
        if valid_ratios:
            overall_score = (1.0 / len(valid_ratios)) * sum(valid_ratios)
        else:
            overall_score = 0.0
        
        return {
            "overall_score": overall_score,
            "performance_ratios": performance_ratios,
            "meets_targets": overall_score >= 1.0,
            "target_achievements": sum(1 for r in performance_ratios.values() if r >= 1.0),
            "total_targets": len(targets)
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return "No benchmark results available."
        
        report_lines = []
        report_lines.append("🔬 RESEARCH ALGORITHM BENCHMARK REPORT")
        report_lines.append("=" * 60)
        
        summary = self.results.get("benchmark_summary", {})
        report_lines.append(f"Execution Time: {summary.get('total_execution_time_ms', 0):.0f}ms")
        report_lines.append(f"Timestamp: {summary.get('timestamp', 'unknown')}")
        
        # Overall performance
        perf = summary.get("overall_performance", {})
        score = perf.get("overall_score", 0)
        report_lines.append(f"\\nOverall Performance Score: {score:.2f}")
        report_lines.append(f"Targets Met: {perf.get('target_achievements', 0)}/{perf.get('total_targets', 0)}")
        
        # Individual component results
        components = [
            ("Knowledge Graph NN", "knowledge_graph"),
            ("Literature Synthesis", "literature_synthesis"),
            ("Experimental Design", "experimental_design"),
            ("Research Planning", "research_planning"),
            ("Integration Workflow", "integration")
        ]
        
        for name, key in components:
            if key in self.results:
                report_lines.append(f"\\n{name}:")
                self._add_component_summary(report_lines, key)
        
        # Recommendations
        report_lines.append("\\n🎯 RECOMMENDATIONS:")
        if score >= 1.0:
            report_lines.append("✅ Excellent performance - algorithms meet all targets")
        elif score >= 0.8:
            report_lines.append("✅ Good performance - minor optimizations recommended")
        elif score >= 0.5:
            report_lines.append("⚠️ Acceptable performance - some optimization needed")
        else:
            report_lines.append("❌ Below target performance - significant optimization required")
        
        return "\\n".join(report_lines)
    
    def _add_component_summary(self, report_lines: List[str], component_key: str):
        """Add component-specific summary to report."""
        data = self.results[component_key]
        
        if component_key == "knowledge_graph":
            embedding_time = data.get("embedding_generation", {}).get("time_per_paper_ms", 0)
            report_lines.append(f"  - Embedding generation: {embedding_time:.2f}ms per paper")
            
        elif component_key == "literature_synthesis":
            if "size_100" in data:
                time_100 = data["size_100"].get("processing_time_ms", 0)
                clusters = data["size_100"].get("clusters_found", 0)
                report_lines.append(f"  - 100 papers processed in {time_100:.0f}ms")
                report_lines.append(f"  - {clusters} clusters identified")
                
        elif component_key == "experimental_design":
            opt_time = data.get("optimization_performance", {}).get("time_ms", 0)
            fitness = data.get("solution_quality", {}).get("fitness_score", 0)
            report_lines.append(f"  - Optimization completed in {opt_time:.0f}ms")
            report_lines.append(f"  - Solution fitness: {fitness:.2f}")
            
        elif component_key == "research_planning":
            plan_time = data.get("planning_performance", {}).get("time_ms", 0)
            plan_size = data.get("solution_quality", {}).get("plan_size", 0)
            report_lines.append(f"  - Planning completed in {plan_time:.0f}ms")
            report_lines.append(f"  - Optimal plan: {plan_size} projects")
            
        elif component_key == "integration":
            total_time = data.get("integration_performance", {}).get("total_time_ms", 0)
            components = data.get("integration_performance", {}).get("components_executed", 0)
            report_lines.append(f"  - Full workflow: {total_time:.0f}ms")
            report_lines.append(f"  - Components executed: {components}")


async def main():
    """Main benchmark execution."""
    print("🧪 PhD Notebook Research Algorithm Benchmarks")
    print("Testing novel algorithms for academic research enhancement")
    print("=" * 60)
    
    # Initialize benchmark suite
    benchmark = ResearchBenchmarkSuite()
    
    try:
        # Run all benchmarks
        results = await benchmark.run_all_benchmarks()
        
        # Generate and display report
        print("\\n" + "=" * 60)
        print(benchmark.generate_report())
        
        # Save results
        output_file = Path(__file__).parent / "benchmark_results.json"
        with open(output_file, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            json_results = json.loads(json.dumps(results, default=str))
            json.dump(json_results, f, indent=2)
        
        print(f"\\n📊 Detailed results saved to: {output_file}")
        
        # Return success code
        return 0 if results["benchmark_summary"]["overall_performance"]["meets_targets"] else 1
        
    except Exception as e:
        print(f"\\n❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))