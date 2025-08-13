#!/usr/bin/env python3
"""
Publication Pipeline Benchmarks
===============================

Comprehensive benchmarking for the advanced publication pipeline.
Tests arXiv submission, citation management, and academic publishing workflows.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phd_notebook.publication.arxiv_publisher import ArxivPublisher, ArxivSubmission
from phd_notebook.publication.citation_manager import CitationManager, Citation, CitationStyle


class PublicationPipelineBenchmarks:
    """Comprehensive benchmarks for publication pipeline."""
    
    def __init__(self):
        self.results = {}
        
    def run_arxiv_publisher_benchmarks(self) -> dict:
        """Benchmark arXiv publisher functionality."""
        print("📄 Benchmarking ArXiv Publisher...")
        
        publisher = ArxivPublisher()
        results = {}
        
        # Test submission preparation
        test_paper = {
            "title": "Novel Approaches to Machine Learning Optimization in Research Environments",
            "authors": ["Dr. Jane Smith", "Prof. John Doe", "Dr. Alice Johnson"],
            "abstract": "This paper presents groundbreaking research in machine learning optimization techniques specifically designed for academic research environments. Our methodology introduces three novel algorithms that significantly improve computational efficiency while maintaining accuracy. Through extensive experimentation on diverse datasets, we demonstrate performance improvements of up to 300% over existing methods. The proposed techniques are particularly effective for resource-constrained academic computing environments. Our results show significant implications for democratizing access to advanced machine learning capabilities in educational settings.",
            "content": "Introduction\\n\\nMachine learning optimization has become crucial...\\n\\n",
            "page_count": 12,
            "figure_count": 8,
            "conference": "ICML 2025"
        }
        
        start_time = time.time()
        submission = publisher.prepare_submission(test_paper, ["cs.LG", "stat.ML"])
        prep_time = time.time() - start_time
        
        # Test validation
        start_time = time.time()
        is_valid, errors = publisher.validate_submission(submission)
        validation_time = time.time() - start_time
        
        # Test simulation
        start_time = time.time()
        sim_result = publisher.simulate_submission(submission)
        submission_time = time.time() - start_time
        
        # Test metadata generation
        start_time = time.time()
        metadata = publisher.generate_arxiv_metadata(submission)
        metadata_time = time.time() - start_time
        
        results = {
            "preparation": {
                "time_ms": prep_time * 1000,
                "success": isinstance(submission, ArxivSubmission),
                "categories_set": len(submission.categories) == 2
            },
            "validation": {
                "time_ms": validation_time * 1000,
                "is_valid": is_valid,
                "error_count": len(errors),
                "validation_comprehensive": len(errors) == 0
            },
            "submission_simulation": {
                "time_ms": submission_time * 1000,
                "success": sim_result.get("success", False),
                "arxiv_id_generated": "submission_id" in sim_result,
                "proper_response": all(k in sim_result for k in ["arxiv_url", "pdf_url"])
            },
            "metadata_generation": {
                "time_ms": metadata_time * 1000,
                "metadata_length": len(metadata),
                "contains_required_fields": all(field in metadata for field in ["Title:", "Authors:", "Abstract:"])
            }
        }
        
        return results
    
    def run_citation_manager_benchmarks(self) -> dict:
        """Benchmark citation manager functionality."""
        print("📚 Benchmarking Citation Manager...")
        
        manager = CitationManager()
        results = {}
        
        # Create test citations
        test_citations = [
            Citation(
                id="smith2023",
                title="Deep Learning for Academic Research: A Comprehensive Survey",
                authors=["Jane Smith", "John Doe"],
                year=2023,
                publication_type="journal",
                venue="Journal of Machine Learning Research",
                volume="24",
                pages="1-45",
                doi="10.1234/jmlr.2023.001"
            ),
            Citation(
                id="doe2022",
                title="Optimization Techniques in Neural Networks",
                authors=["John Doe", "Alice Johnson", "Bob Wilson"],
                year=2022,
                publication_type="conference",
                venue="International Conference on Machine Learning",
                pages="123-134"
            ),
            Citation(
                id="johnson2021",
                title="Machine Learning in Education: Practical Applications",
                authors=["Alice Johnson"],
                year=2021,
                publication_type="book",
                venue="Academic Press",
                publisher="Elsevier"
            )
        ]
        
        # Test citation addition
        start_time = time.time()
        for citation in test_citations:
            manager.add_citation(citation)
        addition_time = time.time() - start_time
        
        # Test formatting in different styles
        styles_to_test = [CitationStyle.APA, CitationStyle.IEEE, CitationStyle.NATURE]
        formatting_results = {}
        
        for style in styles_to_test:
            start_time = time.time()
            formatted_citations = []
            for citation in test_citations:
                formatted = manager.format_citation(citation.id, style)
                formatted_citations.append(formatted)
            
            formatting_time = time.time() - start_time
            formatting_results[style.value] = {
                "time_ms": formatting_time * 1000,
                "citations_formatted": len(formatted_citations),
                "avg_length": sum(len(c) for c in formatted_citations) / len(formatted_citations),
                "all_formatted": all(len(c) > 10 for c in formatted_citations)
            }
        
        # Test bibliography generation
        start_time = time.time()
        bibliography = manager.generate_bibliography(CitationStyle.APA, sort_by="author")
        bibliography_time = time.time() - start_time
        
        # Test search functionality
        start_time = time.time()
        search_results = manager.search_citations("machine learning", ["title", "venue"])
        search_time = time.time() - start_time
        
        # Test BibTeX parsing
        test_bibtex = """@article{test2023,
  title = {Test Article for Parsing},
  author = {Test Author and Another Author},
  year = {2023},
  journal = {Test Journal},
  volume = {10},
  pages = {1--10}
}"""
        
        start_time = time.time()
        try:
            parsed_citation = manager.parse_bibtex_entry(test_bibtex)
            parsing_success = True
        except:
            parsed_citation = None
            parsing_success = False
        parsing_time = time.time() - start_time
        
        # Test export functionality
        start_time = time.time()
        exported_bibtex = manager.export_citations("bibtex")
        export_time = time.time() - start_time
        
        results = {
            "citation_addition": {
                "time_ms": addition_time * 1000,
                "citations_added": len(test_citations),
                "avg_time_per_citation": (addition_time * 1000) / len(test_citations)
            },
            "formatting_performance": formatting_results,
            "bibliography_generation": {
                "time_ms": bibliography_time * 1000,
                "entries_generated": len(bibliography),
                "proper_formatting": all("(" in entry and ")" in entry for entry in bibliography)
            },
            "search_functionality": {
                "time_ms": search_time * 1000,
                "results_found": len(search_results),
                "search_accuracy": len([r for r in search_results if "machine learning" in r.title.lower() or "machine learning" in r.venue.lower()])
            },
            "bibtex_parsing": {
                "time_ms": parsing_time * 1000,
                "parsing_success": parsing_success,
                "parsed_correctly": parsed_citation.title == "Test Article for Parsing" if parsed_citation else False
            },
            "export_functionality": {
                "time_ms": export_time * 1000,
                "export_length": len(exported_bibtex),
                "contains_bibtex": "@" in exported_bibtex and "{" in exported_bibtex
            }
        }
        
        return results
    
    def run_integration_benchmarks(self) -> dict:
        """Test integration between publication components."""
        print("🔗 Benchmarking Publication Pipeline Integration...")
        
        start_time = time.time()
        
        # Create integrated workflow
        publisher = ArxivPublisher()
        citation_manager = CitationManager()
        
        # Test complete publication workflow
        paper_data = {
            "title": "Automated Research Pipeline: From Literature Review to Publication",
            "authors": ["Research Team", "PhD Candidate", "Supervisor"],
            "abstract": "This paper presents an automated research pipeline that streamlines the academic publication process from initial literature review through final submission. Our system integrates advanced citation management, automated formatting, and submission workflows to reduce the time from research completion to publication by 75%. The pipeline supports multiple citation styles, automated compliance checking, and direct integration with academic databases. Experimental validation shows significant improvements in publication efficiency and accuracy.",
            "content": "Full paper content here...",
            "references": [
                "smith2023: Deep Learning for Academic Research",
                "doe2022: Optimization Techniques", 
                "johnson2021: ML in Education"
            ]
        }
        
        # Test workflow steps
        submission = publisher.prepare_submission(paper_data)
        is_valid, errors = publisher.validate_submission(submission)
        
        # Add citations referenced in paper
        for ref in paper_data["references"]:
            ref_id, ref_title = ref.split(": ", 1)
            citation = Citation(
                id=ref_id,
                title=ref_title,
                authors=["Test Author"],
                year=2023,
                publication_type="journal",
                venue="Test Venue"
            )
            citation_manager.add_citation(citation, ref_id)
        
        # Generate bibliography
        bibliography = citation_manager.generate_bibliography()
        
        # Simulate submission
        submission_result = publisher.simulate_submission(submission)
        
        total_time = time.time() - start_time
        
        return {
            "workflow_performance": {
                "total_time_ms": total_time * 1000,
                "submission_prepared": isinstance(submission, ArxivSubmission),
                "validation_passed": is_valid,
                "bibliography_generated": len(bibliography) > 0,
                "submission_successful": submission_result.get("success", False)
            },
            "integration_quality": {
                "all_components_working": all([
                    isinstance(submission, ArxivSubmission),
                    is_valid,
                    len(bibliography) > 0,
                    submission_result.get("success", False)
                ]),
                "references_processed": len(paper_data["references"]),
                "citations_in_manager": len(citation_manager.citations),
                "workflow_complete": submission_result.get("arxiv_url") is not None
            }
        }
    
    def run_performance_stress_tests(self) -> dict:
        """Run stress tests for publication components."""
        print("⚡ Running Publication Pipeline Stress Tests...")
        
        results = {}
        
        # Stress test citation manager with large dataset
        citation_manager = CitationManager()
        
        # Generate many test citations
        start_time = time.time()
        num_citations = 1000
        
        for i in range(num_citations):
            citation = Citation(
                id=f"test{i}",
                title=f"Test Paper {i}: Research in Various Fields",
                authors=[f"Author {i}", f"Coauthor {i}"],
                year=2020 + (i % 4),
                publication_type="journal" if i % 2 == 0 else "conference",
                venue=f"Test Venue {i % 10}",
                volume=str((i % 20) + 1),
                pages=f"{i}-{i+10}"
            )
            citation_manager.add_citation(citation)
        
        addition_time = time.time() - start_time
        
        # Test large bibliography generation
        start_time = time.time()
        large_bibliography = citation_manager.generate_bibliography()
        bibliography_time = time.time() - start_time
        
        # Test search performance with large dataset
        start_time = time.time()
        search_results = citation_manager.search_citations("Research")
        search_time = time.time() - start_time
        
        # Test export performance
        start_time = time.time()
        exported_data = citation_manager.export_citations("json")
        export_time = time.time() - start_time
        
        results["stress_tests"] = {
            "large_dataset_addition": {
                "citations_added": num_citations,
                "total_time_ms": addition_time * 1000,
                "avg_time_per_citation_ms": (addition_time * 1000) / num_citations,
                "citations_per_second": num_citations / addition_time
            },
            "large_bibliography_generation": {
                "entries_generated": len(large_bibliography),
                "time_ms": bibliography_time * 1000,
                "entries_per_second": len(large_bibliography) / bibliography_time
            },
            "large_dataset_search": {
                "search_time_ms": search_time * 1000,
                "results_found": len(search_results),
                "search_throughput": len(citation_manager.citations) / search_time
            },
            "large_export": {
                "export_time_ms": export_time * 1000,
                "export_size_kb": len(exported_data) / 1024,
                "export_rate_kb_per_sec": (len(exported_data) / 1024) / export_time
            }
        }
        
        return results
    
    def run_all_benchmarks(self) -> dict:
        """Run complete publication pipeline benchmark suite."""
        print("🚀 Starting Publication Pipeline Benchmark Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run individual component benchmarks
        self.results["arxiv_publisher"] = self.run_arxiv_publisher_benchmarks()
        self.results["citation_manager"] = self.run_citation_manager_benchmarks()
        self.results["integration"] = self.run_integration_benchmarks()
        self.results["stress_tests"] = self.run_performance_stress_tests()
        
        total_time = time.time() - start_time
        
        # Calculate overall performance metrics
        self.results["benchmark_summary"] = {
            "total_execution_time_ms": total_time * 1000,
            "timestamp": datetime.now().isoformat(),
            "components_tested": 4,
            "overall_performance": self._calculate_overall_performance()
        }
        
        return self.results
    
    def _calculate_overall_performance(self) -> dict:
        """Calculate overall performance metrics."""
        # Define performance targets
        targets = {
            "arxiv_preparation": 100.0,  # 100ms
            "citation_formatting": 10.0,  # 10ms per citation
            "bibliography_generation": 500.0,  # 500ms for 3 citations
            "large_dataset_search": 1000.0,  # 1s for 1000 citations
            "integration_workflow": 2000.0  # 2s for complete workflow
        }
        
        # Extract actual performance
        actual = {}
        
        arxiv_results = self.results.get("arxiv_publisher", {})
        actual["arxiv_preparation"] = arxiv_results.get("preparation", {}).get("time_ms", float('inf'))
        
        citation_results = self.results.get("citation_manager", {})
        actual["citation_formatting"] = citation_results.get("formatting_performance", {}).get("apa", {}).get("time_ms", float('inf'))
        actual["bibliography_generation"] = citation_results.get("bibliography_generation", {}).get("time_ms", float('inf'))
        
        stress_results = self.results.get("stress_tests", {})
        actual["large_dataset_search"] = stress_results.get("large_dataset_search", {}).get("search_time_ms", float('inf'))
        
        integration_results = self.results.get("integration", {})
        actual["integration_workflow"] = integration_results.get("workflow_performance", {}).get("total_time_ms", float('inf'))
        
        # Calculate performance ratios
        performance_ratios = {}
        for metric in targets:
            if actual.get(metric, float('inf')) != float('inf'):
                performance_ratios[metric] = targets[metric] / actual[metric]
            else:
                performance_ratios[metric] = 0.0
        
        # Overall score
        valid_ratios = [r for r in performance_ratios.values() if r > 0]
        overall_score = sum(valid_ratios) / len(valid_ratios) if valid_ratios else 0.0
        
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
        report_lines.append("📚 PUBLICATION PIPELINE BENCHMARK REPORT")
        report_lines.append("=" * 60)
        
        summary = self.results.get("benchmark_summary", {})
        report_lines.append(f"Execution Time: {summary.get('total_execution_time_ms', 0):.0f}ms")
        report_lines.append(f"Timestamp: {summary.get('timestamp', 'unknown')}")
        
        # Overall performance
        perf = summary.get("overall_performance", {})
        score = perf.get("overall_score", 0)
        report_lines.append(f"\\nOverall Performance Score: {score:.2f}")
        report_lines.append(f"Targets Met: {perf.get('target_achievements', 0)}/{perf.get('total_targets', 0)}")
        
        # Component results
        components = [
            ("ArXiv Publisher", "arxiv_publisher"),
            ("Citation Manager", "citation_manager"),
            ("Integration Tests", "integration"),
            ("Stress Tests", "stress_tests")
        ]
        
        for name, key in components:
            if key in self.results:
                report_lines.append(f"\\n{name}:")
                self._add_component_report(report_lines, key)
        
        # Performance recommendations
        report_lines.append("\\n🎯 RECOMMENDATIONS:")
        if score >= 1.0:
            report_lines.append("✅ Excellent performance - publication pipeline ready for production")
        elif score >= 0.8:
            report_lines.append("✅ Good performance - minor optimizations recommended")
        else:
            report_lines.append("⚠️ Performance optimization needed for production use")
        
        return "\\n".join(report_lines)
    
    def _add_component_report(self, report_lines: list, component_key: str):
        """Add component-specific report section."""
        data = self.results[component_key]
        
        if component_key == "arxiv_publisher":
            prep_time = data.get("preparation", {}).get("time_ms", 0)
            validation_time = data.get("validation", {}).get("time_ms", 0)
            report_lines.append(f"  - Submission preparation: {prep_time:.1f}ms")
            report_lines.append(f"  - Validation: {validation_time:.1f}ms")
            
        elif component_key == "citation_manager":
            if "formatting_performance" in data:
                apa_time = data["formatting_performance"].get("apa", {}).get("time_ms", 0)
                report_lines.append(f"  - Citation formatting (APA): {apa_time:.1f}ms")
            
            bib_time = data.get("bibliography_generation", {}).get("time_ms", 0)
            report_lines.append(f"  - Bibliography generation: {bib_time:.1f}ms")
            
        elif component_key == "integration":
            total_time = data.get("workflow_performance", {}).get("total_time_ms", 0)
            components_working = data.get("integration_quality", {}).get("all_components_working", False)
            report_lines.append(f"  - Complete workflow: {total_time:.0f}ms")
            report_lines.append(f"  - All components working: {'✅' if components_working else '❌'}")
            
        elif component_key == "stress_tests":
            citations_per_sec = data.get("large_dataset_addition", {}).get("citations_per_second", 0)
            search_time = data.get("large_dataset_search", {}).get("search_time_ms", 0)
            report_lines.append(f"  - Citation processing: {citations_per_sec:.0f} citations/sec")
            report_lines.append(f"  - Large dataset search: {search_time:.0f}ms")


async def main():
    """Main benchmark execution."""
    print("📚 PhD Notebook Publication Pipeline Benchmarks")
    print("Testing advanced publication and citation management features")
    print("=" * 60)
    
    # Initialize benchmark suite
    benchmark = PublicationPipelineBenchmarks()
    
    try:
        # Run all benchmarks
        results = benchmark.run_all_benchmarks()
        
        # Generate and display report
        print("\\n" + "=" * 60)
        print(benchmark.generate_report())
        
        # Save results
        output_file = Path(__file__).parent / "publication_benchmark_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n📊 Detailed results saved to: {output_file}")
        
        # Return success code
        overall_performance = results["benchmark_summary"]["overall_performance"]
        return 0 if overall_performance["meets_targets"] else 1
        
    except Exception as e:
        print(f"\\n❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    import asyncio
    exit(asyncio.run(main()))