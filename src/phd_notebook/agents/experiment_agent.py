"""
Experiment Agent for managing research experiments and data analysis.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import re

from .base import BaseAgent
from ..core.note import Note, NoteType


class ExperimentAgent(BaseAgent):
    """
    AI agent for experiment design and analysis.
    
    Capabilities:
    - Experiment planning and design
    - Data analysis and visualization suggestions
    - Results interpretation
    - Methodology validation
    """
    
    def __init__(self):
        super().__init__(
            name="ExperimentAgent",
            capabilities=[
                'experiment_design',
                'data_analysis', 
                'results_interpretation',
                'methodology_validation',
                'protocol_generation'
            ]
        )
    
    def process(self, input_data: Any, **kwargs) -> Any:
        """Process experiment-related data."""
        if isinstance(input_data, dict):
            if 'experiment_type' in input_data:
                return self.design_experiment(input_data)
            elif 'results_data' in input_data:
                return self.analyze_results(input_data['results_data'])
        
        return input_data
    
    def design_experiment(self, experiment_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Design an experiment based on specifications."""
        self.log_activity("Designing experiment")
        
        exp_type = experiment_spec.get('experiment_type', 'general')
        hypothesis = experiment_spec.get('hypothesis', '')
        variables = experiment_spec.get('variables', {})
        
        design = {
            'experimental_design': self._suggest_experimental_design(exp_type, variables),
            'methodology': self._generate_methodology(exp_type, hypothesis),
            'materials_needed': self._suggest_materials(exp_type),
            'data_collection': self._suggest_data_collection(exp_type, variables),
            'analysis_plan': self._suggest_analysis_plan(exp_type, variables),
            'expected_timeline': self._estimate_timeline(exp_type),
            'potential_challenges': self._identify_challenges(exp_type),
            'success_metrics': self._define_success_metrics(hypothesis, variables)
        }
        
        return design
    
    def create_experiment_note(
        self, 
        title: str, 
        hypothesis: str,
        experiment_type: str = "general",
        variables: Dict[str, Any] = None
    ) -> Note:
        """Create a structured experiment note."""
        if not self.notebook:
            raise RuntimeError("Agent must be registered with a notebook")
        
        variables = variables or {}
        
        # Design the experiment
        experiment_spec = {
            'experiment_type': experiment_type,
            'hypothesis': hypothesis,
            'variables': variables
        }
        design = self.design_experiment(experiment_spec)
        
        # Build experiment content
        content = self._build_experiment_content(title, hypothesis, design)
        
        # Create note
        experiment_id = f"{datetime.now().strftime('%Y%m%d')}_{title.lower().replace(' ', '_')}"
        
        note = self.notebook.create_note(
            title=title,
            content=content,
            note_type=NoteType.EXPERIMENT,
            tags=['#experiment', '#active', f'#{experiment_type}']
        )
        
        # Add metadata
        note.frontmatter.metadata.update({
            'experiment_id': experiment_id,
            'experiment_type': experiment_type,
            'hypothesis': hypothesis,
            'variables': variables,
            'design_date': datetime.now().isoformat(),
            'status': 'planned'
        })
        
        if note.file_path:
            note.save()
        
        self.log_activity(f"Created experiment note: {title}")
        return note
    
    def analyze_results(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experiment results."""
        self.log_activity("Analyzing experiment results")
        
        analysis = {
            'summary_statistics': self._calculate_summary_stats(results_data),
            'key_observations': self._extract_key_observations(results_data),
            'hypothesis_evaluation': self._evaluate_hypothesis(results_data),
            'significance_test': self._suggest_significance_tests(results_data),
            'visualization_suggestions': self._suggest_visualizations(results_data),
            'interpretation': self._interpret_results(results_data),
            'limitations': self._identify_result_limitations(results_data),
            'next_steps': self._suggest_next_steps(results_data)
        }
        
        return analysis
    
    def update_experiment_results(self, experiment_note: Note, results_data: Dict[str, Any]) -> Note:
        """Update an experiment note with results and analysis."""
        analysis = self.analyze_results(results_data)
        
        # Add results section
        results_content = self._build_results_content(analysis)
        experiment_note.add_section("Results", results_content)
        
        # Add analysis section
        analysis_content = self._build_analysis_content(analysis)
        experiment_note.add_section("Analysis", analysis_content)
        
        # Update metadata
        experiment_note.frontmatter.metadata.update({
            'results_added': datetime.now().isoformat(),
            'status': 'completed'
        })
        
        # Update tags
        if '#active' in experiment_note.frontmatter.tags:
            experiment_note.frontmatter.tags.remove('#active')
        experiment_note.add_tag('#completed')
        
        if experiment_note.file_path:
            experiment_note.save()
        
        self.log_activity(f"Updated experiment results: {experiment_note.title}")
        return experiment_note
    
    def suggest_follow_up_experiments(self, experiment_note: Note) -> List[Dict[str, Any]]:
        """Suggest follow-up experiments based on results."""
        if not self.notebook:
            return []
        
        suggestions = []
        
        # Extract key information from the experiment
        hypothesis = experiment_note.frontmatter.metadata.get('hypothesis', '')
        exp_type = experiment_note.frontmatter.metadata.get('experiment_type', 'general')
        
        # Analyze the content for potential follow-ups
        content = experiment_note.content.lower()
        
        # Rule-based suggestions
        if 'unexpected' in content or 'surprising' in content:
            suggestions.append({
                'type': 'replication',
                'title': f"Replication of {experiment_note.title}",
                'rationale': 'Verify unexpected results through replication',
                'priority': 'high'
            })
        
        if 'limited' in content or 'small sample' in content:
            suggestions.append({
                'type': 'scale_up',
                'title': f"Scaled-up {experiment_note.title}",
                'rationale': 'Increase sample size for more robust results',
                'priority': 'medium'
            })
        
        if 'control' in content or 'baseline' in content:
            suggestions.append({
                'type': 'variation',
                'title': f"Variation of {experiment_note.title}",
                'rationale': 'Test different conditions or parameters',
                'priority': 'medium'
            })
        
        # Generic follow-up suggestions
        if not suggestions:
            suggestions.append({
                'type': 'extension',
                'title': f"Extended {experiment_note.title}",
                'rationale': 'Explore additional aspects or parameters',
                'priority': 'low'
            })
        
        return suggestions
    
    def _suggest_experimental_design(self, exp_type: str, variables: Dict[str, Any]) -> str:
        """Suggest appropriate experimental design."""
        designs = {
            'controlled': 'Randomized controlled trial with treatment and control groups',
            'comparative': 'Comparative study testing multiple conditions',
            'longitudinal': 'Longitudinal study tracking changes over time',
            'cross_sectional': 'Cross-sectional analysis at a single time point',
            'case_study': 'In-depth case study analysis',
            'general': 'Controlled experiment with systematic variable manipulation'
        }
        
        design = designs.get(exp_type, designs['general'])
        
        if len(variables) > 2:
            design += " with factorial design to test multiple variables"
        
        return design
    
    def _generate_methodology(self, exp_type: str, hypothesis: str) -> List[str]:
        """Generate methodology steps."""
        base_steps = [
            "Define clear research question and hypothesis",
            "Identify independent and dependent variables",
            "Design experimental protocol",
            "Prepare materials and setup",
            "Conduct pilot test if necessary",
            "Execute main experiment",
            "Collect and record data systematically",
            "Analyze results using appropriate statistical methods"
        ]
        
        # Customize based on experiment type
        if exp_type == 'controlled':
            base_steps.insert(4, "Randomize assignment to treatment/control groups")
        elif exp_type == 'longitudinal':
            base_steps.insert(6, "Schedule multiple data collection time points")
        
        return base_steps
    
    def _suggest_materials(self, exp_type: str) -> List[str]:
        """Suggest materials needed for the experiment."""
        common_materials = [
            "Data collection instruments/tools",
            "Recording equipment (digital/paper)",
            "Statistical analysis software",
            "Backup storage for data"
        ]
        
        specific_materials = {
            'controlled': ["Randomization tools", "Control materials"],
            'comparative': ["Comparison materials/conditions"],
            'longitudinal': ["Scheduling system", "Participant tracking tools"],
            'case_study': ["Interview recording equipment", "Documentation tools"]
        }
        
        materials = common_materials + specific_materials.get(exp_type, [])
        return materials
    
    def _suggest_data_collection(self, exp_type: str, variables: Dict[str, Any]) -> Dict[str, str]:
        """Suggest data collection methods."""
        return {
            'method': 'Systematic measurement and recording',
            'frequency': 'Regular intervals as appropriate',
            'tools': 'Standardized instruments when possible',
            'quality_control': 'Regular calibration and validation checks'
        }
    
    def _suggest_analysis_plan(self, exp_type: str, variables: Dict[str, Any]) -> List[str]:
        """Suggest data analysis plan."""
        analysis_steps = [
            "Data cleaning and preprocessing",
            "Descriptive statistics calculation",
            "Exploratory data analysis",
            "Statistical significance testing",
            "Effect size calculation",
            "Results interpretation and discussion"
        ]
        
        return analysis_steps
    
    def _estimate_timeline(self, exp_type: str) -> Dict[str, str]:
        """Estimate experiment timeline."""
        timelines = {
            'controlled': {'planning': '1-2 weeks', 'execution': '2-4 weeks', 'analysis': '1-2 weeks'},
            'longitudinal': {'planning': '2-3 weeks', 'execution': '8-12 weeks', 'analysis': '2-3 weeks'},
            'comparative': {'planning': '1-2 weeks', 'execution': '2-3 weeks', 'analysis': '1-2 weeks'},
            'case_study': {'planning': '1 week', 'execution': '3-6 weeks', 'analysis': '2-3 weeks'},
            'general': {'planning': '1-2 weeks', 'execution': '2-4 weeks', 'analysis': '1-2 weeks'}
        }
        
        return timelines.get(exp_type, timelines['general'])
    
    def _identify_challenges(self, exp_type: str) -> List[str]:
        """Identify potential challenges."""
        common_challenges = [
            "Sample size limitations",
            "Measurement accuracy",
            "External factors/confounding variables",
            "Time and resource constraints"
        ]
        
        specific_challenges = {
            'controlled': ["Blinding difficulties", "Control group maintenance"],
            'longitudinal': ["Participant dropout", "Long-term commitment"],
            'comparative': ["Fair comparison conditions", "Baseline differences"],
            'case_study': ["Generalizability limitations", "Researcher bias"]
        }
        
        challenges = common_challenges + specific_challenges.get(exp_type, [])
        return challenges
    
    def _define_success_metrics(self, hypothesis: str, variables: Dict[str, Any]) -> List[str]:
        """Define success metrics for the experiment."""
        metrics = [
            "Statistical significance (p < 0.05)",
            "Practical significance (effect size)",
            "Reproducibility of results",
            "Achievement of research objectives"
        ]
        
        return metrics
    
    def _calculate_summary_stats(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics (simplified)."""
        return {
            'data_points': len(results_data.get('measurements', [])),
            'categories': len(results_data.get('categories', [])),
            'completeness': 'High' if results_data else 'Low'
        }
    
    def _extract_key_observations(self, results_data: Dict[str, Any]) -> List[str]:
        """Extract key observations from results."""
        observations = ["Key patterns identified in the data"]
        
        # Add specific observations based on data structure
        if 'measurements' in results_data:
            observations.append(f"Collected {len(results_data['measurements'])} measurements")
        
        return observations
    
    def _evaluate_hypothesis(self, results_data: Dict[str, Any]) -> str:
        """Evaluate hypothesis based on results."""
        return "Hypothesis evaluation requires detailed statistical analysis"
    
    def _suggest_significance_tests(self, results_data: Dict[str, Any]) -> List[str]:
        """Suggest appropriate significance tests."""
        return [
            "T-test for group comparisons",
            "ANOVA for multiple groups",
            "Chi-square for categorical data",
            "Correlation analysis for relationships"
        ]
    
    def _suggest_visualizations(self, results_data: Dict[str, Any]) -> List[str]:
        """Suggest data visualizations."""
        return [
            "Bar charts for categorical comparisons",
            "Line plots for trends over time",
            "Scatter plots for relationships",
            "Box plots for distribution analysis",
            "Heatmaps for correlation matrices"
        ]
    
    def _interpret_results(self, results_data: Dict[str, Any]) -> str:
        """Provide basic results interpretation."""
        return "Results show [describe main findings]. This suggests [implications for hypothesis]."
    
    def _identify_result_limitations(self, results_data: Dict[str, Any]) -> List[str]:
        """Identify limitations in the results."""
        return [
            "Sample size may limit generalizability",
            "Measurement precision could be improved",
            "External validity needs consideration"
        ]
    
    def _suggest_next_steps(self, results_data: Dict[str, Any]) -> List[str]:
        """Suggest next steps based on results."""
        return [
            "Replicate with larger sample size",
            "Investigate unexpected findings",
            "Consider alternative explanations",
            "Plan follow-up experiments"
        ]
    
    def _build_experiment_content(self, title: str, hypothesis: str, design: Dict[str, Any]) -> str:
        """Build structured content for experiment note."""
        content = f"# {title}\n\n"
        
        # Hypothesis
        content += "## Hypothesis\n"
        content += f"{hypothesis}\n\n"
        
        # Experimental Design
        content += "## Experimental Design\n"
        content += f"{design['experimental_design']}\n\n"
        
        # Methodology
        content += "## Methodology\n"
        for i, step in enumerate(design['methodology'], 1):
            content += f"{i}. {step}\n"
        content += "\n"
        
        # Materials
        content += "## Materials Needed\n"
        for material in design['materials_needed']:
            content += f"- {material}\n"
        content += "\n"
        
        # Data Collection Plan
        content += "## Data Collection Plan\n"
        dc_plan = design['data_collection']
        content += f"**Method**: {dc_plan['method']}\n"
        content += f"**Frequency**: {dc_plan['frequency']}\n"
        content += f"**Tools**: {dc_plan['tools']}\n"
        content += f"**Quality Control**: {dc_plan['quality_control']}\n\n"
        
        # Timeline
        content += "## Timeline\n"
        timeline = design['expected_timeline']
        content += f"- **Planning**: {timeline['planning']}\n"
        content += f"- **Execution**: {timeline['execution']}\n"
        content += f"- **Analysis**: {timeline['analysis']}\n\n"
        
        # Potential Challenges
        content += "## Potential Challenges\n"
        for challenge in design['potential_challenges']:
            content += f"- {challenge}\n"
        content += "\n"
        
        # Success Metrics
        content += "## Success Metrics\n"
        for metric in design['success_metrics']:
            content += f"- {metric}\n"
        content += "\n"
        
        # Placeholders for later sections
        content += "## Results\n"
        content += "*[Results will be added here during execution]*\n\n"
        
        content += "## Analysis\n"
        content += "*[Analysis will be added here after data collection]*\n\n"
        
        content += "## Conclusions\n"
        content += "*[Conclusions will be added here after analysis]*\n\n"
        
        return content
    
    def _build_results_content(self, analysis: Dict[str, Any]) -> str:
        """Build results section content."""
        content = ""
        
        # Summary Statistics
        stats = analysis['summary_statistics']
        content += f"**Data Summary**: {stats['data_points']} data points collected, {stats['completeness']} completeness\n\n"
        
        # Key Observations
        content += "**Key Observations**:\n"
        for obs in analysis['key_observations']:
            content += f"- {obs}\n"
        content += "\n"
        
        return content
    
    def _build_analysis_content(self, analysis: Dict[str, Any]) -> str:
        """Build analysis section content."""
        content = ""
        
        # Hypothesis Evaluation
        content += f"**Hypothesis Evaluation**: {analysis['hypothesis_evaluation']}\n\n"
        
        # Interpretation
        content += f"**Interpretation**: {analysis['interpretation']}\n\n"
        
        # Limitations
        content += "**Limitations**:\n"
        for limitation in analysis['limitations']:
            content += f"- {limitation}\n"
        content += "\n"
        
        # Next Steps
        content += "**Next Steps**:\n"
        for step in analysis['next_steps']:
            content += f"- {step}\n"
        content += "\n"
        
        return content