"""
Automated Experiment Design and Protocol Generation.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import json
from dataclasses import dataclass, field
from enum import Enum

from ..ai.client_factory import AIClientFactory
from ..core.note import Note, NoteType
from ..utils.exceptions import ResearchError


class ExperimentType(Enum):
    """Types of experiments supported."""
    
    OBSERVATIONAL = "observational"
    CONTROLLED = "controlled" 
    COMPARATIVE = "comparative"
    LONGITUDINAL = "longitudinal"
    CROSS_SECTIONAL = "cross_sectional"
    CASE_STUDY = "case_study"
    META_ANALYSIS = "meta_analysis"


@dataclass
class ExperimentProtocol:
    """Structured experiment protocol."""
    
    id: str
    title: str
    experiment_type: ExperimentType
    hypothesis: str
    objectives: List[str]
    variables: Dict[str, Dict[str, str]]  # independent, dependent, control
    methodology: Dict[str, Any]
    sample_size: int
    duration_days: int
    resources_needed: List[str]
    safety_considerations: List[str]
    data_collection_plan: Dict[str, Any]
    analysis_plan: Dict[str, Any]
    success_criteria: List[str]
    risks_and_mitigation: Dict[str, str]
    ethical_considerations: List[str]
    budget_estimate: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "designed"  # designed, approved, running, completed, cancelled


class ExperimentDesigner:
    """AI-powered experiment design and protocol generation."""
    
    def __init__(self, notebook=None, ai_provider="auto"):
        self.notebook = notebook
        self.ai_client = AIClientFactory.get_client(ai_provider)
        self.protocols: Dict[str, ExperimentProtocol] = {}
        
        # Design templates for different fields
        self.field_templates = {
            "computer_science": {
                "common_metrics": ["accuracy", "precision", "recall", "f1_score", "runtime", "memory_usage"],
                "typical_datasets": ["training", "validation", "test"],
                "standard_methods": ["cross_validation", "statistical_significance_testing"]
            },
            "psychology": {
                "common_metrics": ["response_time", "accuracy", "likert_scales", "behavioral_measures"],
                "ethical_requirements": ["informed_consent", "debriefing", "confidentiality"],
                "standard_methods": ["randomization", "counterbalancing", "blinding"]
            },
            "biology": {
                "common_metrics": ["cell_viability", "protein_expression", "growth_rate"],
                "safety_requirements": ["biosafety_protocols", "waste_disposal", "sterile_technique"],
                "standard_methods": ["replication", "controls", "statistical_power_analysis"]
            },
            "physics": {
                "common_metrics": ["measurement_precision", "signal_to_noise", "calibration"],
                "equipment_considerations": ["instrument_calibration", "environmental_controls"],
                "standard_methods": ["error_propagation", "uncertainty_analysis"]
            }
        }
    
    async def design_experiment(
        self,
        hypothesis: str,
        research_context: str,
        experiment_type: ExperimentType = ExperimentType.CONTROLLED,
        field: str = "general",
        constraints: Dict[str, Any] = None
    ) -> ExperimentProtocol:
        """Design a comprehensive experiment protocol."""
        
        constraints = constraints or {}
        field_template = self.field_templates.get(field.lower(), {})
        
        design_prompt = f"""
        Design a comprehensive {experiment_type.value} experiment to test this hypothesis:
        
        **Hypothesis**: {hypothesis}
        **Research Context**: {research_context}
        **Field**: {field}
        **Constraints**: {json.dumps(constraints, indent=2)}
        
        Provide a detailed experimental protocol including:
        
        1. **Objectives** (3-5 specific, measurable goals)
        2. **Variables**:
           - Independent variables (what you manipulate)
           - Dependent variables (what you measure)
           - Control variables (what you keep constant)
        3. **Methodology** (step-by-step procedure)
        4. **Sample Size** (with justification)
        5. **Duration** (realistic timeline)
        6. **Resources Needed** (equipment, materials, personnel)
        7. **Safety Considerations**
        8. **Data Collection Plan**
        9. **Analysis Plan** 
        10. **Success Criteria** (how to determine if hypothesis is supported)
        11. **Potential Risks and Mitigation Strategies**
        12. **Ethical Considerations**
        
        Field-specific considerations for {field}:
        {json.dumps(field_template, indent=2)}
        
        Format as detailed structured text.
        """
        
        try:
            response = await self.ai_client.generate_text(
                design_prompt,
                max_tokens=2000,
                temperature=0.7
            )
            
            # Parse response into structured protocol
            protocol_data = self._parse_protocol_response(response)
            
            # Create protocol object
            protocol = ExperimentProtocol(
                id=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                title=protocol_data.get('title', f"Experiment for: {hypothesis[:50]}..."),
                experiment_type=experiment_type,
                hypothesis=hypothesis,
                objectives=protocol_data.get('objectives', []),
                variables=protocol_data.get('variables', {}),
                methodology=protocol_data.get('methodology', {}),
                sample_size=protocol_data.get('sample_size', 30),
                duration_days=protocol_data.get('duration_days', 30),
                resources_needed=protocol_data.get('resources_needed', []),
                safety_considerations=protocol_data.get('safety_considerations', []),
                data_collection_plan=protocol_data.get('data_collection_plan', {}),
                analysis_plan=protocol_data.get('analysis_plan', {}),
                success_criteria=protocol_data.get('success_criteria', []),
                risks_and_mitigation=protocol_data.get('risks_and_mitigation', {}),
                ethical_considerations=protocol_data.get('ethical_considerations', [])
            )
            
            # Store protocol
            self.protocols[protocol.id] = protocol
            
            # Create protocol note if notebook available
            if self.notebook:
                self._create_protocol_note(protocol)
            
            return protocol
            
        except Exception as e:
            raise ResearchError(f"Experiment design failed: {e}")
    
    def _parse_protocol_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into structured protocol data."""
        
        protocol_data = {}
        lines = response.strip().split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            
            if any(keyword in line.lower() for keyword in ['objective', 'variable', 'methodology', 'sample', 'duration', 'resource', 'safety', 'data collection', 'analysis', 'success', 'risk', 'ethical']):
                
                # Save previous section
                if current_section:
                    protocol_data[current_section] = self._process_section_content(current_section, current_content)
                
                # Start new section
                if 'objective' in line.lower():
                    current_section = 'objectives'
                elif 'variable' in line.lower():
                    current_section = 'variables'
                elif 'methodology' in line.lower():
                    current_section = 'methodology'
                elif 'sample' in line.lower():
                    current_section = 'sample_size'
                elif 'duration' in line.lower():
                    current_section = 'duration_days'
                elif 'resource' in line.lower():
                    current_section = 'resources_needed'
                elif 'safety' in line.lower():
                    current_section = 'safety_considerations'
                elif 'data collection' in line.lower():
                    current_section = 'data_collection_plan'
                elif 'analysis' in line.lower():
                    current_section = 'analysis_plan'
                elif 'success' in line.lower():
                    current_section = 'success_criteria'
                elif 'risk' in line.lower():
                    current_section = 'risks_and_mitigation'
                elif 'ethical' in line.lower():
                    current_section = 'ethical_considerations'
                
                current_content = []
            else:
                if line and current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section:
            protocol_data[current_section] = self._process_section_content(current_section, current_content)
        
        return protocol_data
    
    def _process_section_content(self, section: str, content: List[str]) -> Any:
        """Process section content based on section type."""
        
        if section in ['objectives', 'resources_needed', 'safety_considerations', 'success_criteria', 'ethical_considerations']:
            # List sections
            items = []
            for line in content:
                if line.startswith(('-', '•', '*')) or line.startswith(tuple('123456789')):
                    items.append(line.lstrip('-•*123456789. '))
                elif line:
                    items.append(line)
            return items
        
        elif section == 'variables':
            # Parse variables structure
            variables = {"independent": {}, "dependent": {}, "control": {}}
            current_type = "independent"
            
            for line in content:
                if 'independent' in line.lower():
                    current_type = "independent"
                elif 'dependent' in line.lower():
                    current_type = "dependent"
                elif 'control' in line.lower():
                    current_type = "control"
                elif ':' in line:
                    parts = line.split(':', 1)
                    variables[current_type][parts[0].strip()] = parts[1].strip()
            
            return variables
        
        elif section in ['sample_size', 'duration_days']:
            # Numeric sections
            for line in content:
                try:
                    return int(''.join(filter(str.isdigit, line)))
                except ValueError:
                    pass
            return 30 if section == 'duration_days' else 30
        
        elif section == 'risks_and_mitigation':
            # Dictionary section
            risks = {}
            for line in content:
                if ':' in line:
                    parts = line.split(':', 1)
                    risks[parts[0].strip()] = parts[1].strip()
            return risks
        
        else:
            # Default: dictionary or string
            if len(content) == 1:
                return content[0]
            return {"description": '\n'.join(content)}
    
    def _create_protocol_note(self, protocol: ExperimentProtocol) -> Note:
        """Create a detailed protocol note."""
        
        note_content = f"""
# Experiment Protocol: {protocol.title}

## Overview
- **Protocol ID**: {protocol.id}
- **Type**: {protocol.experiment_type.value.title()}
- **Created**: {protocol.created_at.strftime('%Y-%m-%d %H:%M:%S')}
- **Status**: {protocol.status}

## Hypothesis
{protocol.hypothesis}

## Objectives
{chr(10).join([f"- {obj}" for obj in protocol.objectives])}

## Variables
### Independent Variables
{json.dumps(protocol.variables.get('independent', {}), indent=2)}

### Dependent Variables  
{json.dumps(protocol.variables.get('dependent', {}), indent=2)}

### Control Variables
{json.dumps(protocol.variables.get('control', {}), indent=2)}

## Methodology
{json.dumps(protocol.methodology, indent=2)}

## Sample & Timeline
- **Sample Size**: {protocol.sample_size}
- **Duration**: {protocol.duration_days} days

## Resources Required
{chr(10).join([f"- {resource}" for resource in protocol.resources_needed])}

## Safety Considerations
{chr(10).join([f"- {safety}" for safety in protocol.safety_considerations])}

## Data Collection Plan
{json.dumps(protocol.data_collection_plan, indent=2)}

## Analysis Plan
{json.dumps(protocol.analysis_plan, indent=2)}

## Success Criteria
{chr(10).join([f"- {criteria}" for criteria in protocol.success_criteria])}

## Risk Mitigation
{chr(10).join([f"- **{risk}**: {mitigation}" for risk, mitigation in protocol.risks_and_mitigation.items()])}

## Ethical Considerations
{chr(10).join([f"- {ethical}" for ethical in protocol.ethical_considerations])}

---
*Generated by ExperimentDesigner on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
        """
        
        protocol_note = self.notebook.create_note(
            title=f"Protocol - {protocol.title}",
            content=note_content,
            note_type=NoteType.EXPERIMENT,
            tags=["#protocol", f"#{protocol.experiment_type.value}", "#experiment_design"]
        )
        
        return protocol_note
    
    async def optimize_protocol(
        self,
        protocol_id: str,
        optimization_goals: List[str] = None
    ) -> ExperimentProtocol:
        """Optimize an existing protocol for specific goals."""
        
        if protocol_id not in self.protocols:
            raise ResearchError(f"Protocol {protocol_id} not found")
        
        protocol = self.protocols[protocol_id]
        optimization_goals = optimization_goals or ["reduce_cost", "improve_efficiency", "increase_power"]
        
        optimization_prompt = f"""
        Optimize this experimental protocol for the following goals:
        {', '.join(optimization_goals)}
        
        Current Protocol:
        - Type: {protocol.experiment_type.value}
        - Sample Size: {protocol.sample_size}
        - Duration: {protocol.duration_days} days
        - Resources: {protocol.resources_needed}
        - Methodology: {json.dumps(protocol.methodology, indent=2)}
        
        Provide specific recommendations for:
        1. Sample size optimization
        2. Timeline optimization  
        3. Resource optimization
        4. Methodology improvements
        5. Cost reduction strategies
        
        Maintain scientific rigor and statistical power.
        """
        
        try:
            optimization_response = await self.ai_client.generate_text(
                optimization_prompt,
                max_tokens=1000,
                temperature=0.6
            )
            
            # Create optimized protocol
            optimized_protocol = ExperimentProtocol(
                id=f"{protocol.id}_opt",
                title=f"{protocol.title} (Optimized)",
                experiment_type=protocol.experiment_type,
                hypothesis=protocol.hypothesis,
                objectives=protocol.objectives,
                variables=protocol.variables,
                methodology=protocol.methodology,
                sample_size=protocol.sample_size,
                duration_days=protocol.duration_days,
                resources_needed=protocol.resources_needed,
                safety_considerations=protocol.safety_considerations,
                data_collection_plan=protocol.data_collection_plan,
                analysis_plan=protocol.analysis_plan,
                success_criteria=protocol.success_criteria,
                risks_and_mitigation=protocol.risks_and_mitigation,
                ethical_considerations=protocol.ethical_considerations,
                status="optimized"
            )
            
            # Apply optimizations (simplified parsing)
            self._apply_optimizations(optimized_protocol, optimization_response)
            
            # Store optimized protocol
            self.protocols[optimized_protocol.id] = optimized_protocol
            
            return optimized_protocol
            
        except Exception as e:
            raise ResearchError(f"Protocol optimization failed: {e}")
    
    def _apply_optimizations(self, protocol: ExperimentProtocol, optimization_text: str):
        """Apply optimization suggestions to protocol."""
        
        lines = optimization_text.lower().split('\n')
        
        for line in lines:
            if 'sample size' in line and any(char.isdigit() for char in line):
                numbers = [int(s) for s in line.split() if s.isdigit()]
                if numbers:
                    protocol.sample_size = numbers[0]
            
            elif 'duration' in line and ('day' in line or 'week' in line):
                numbers = [int(s) for s in line.split() if s.isdigit()]
                if numbers:
                    if 'week' in line:
                        protocol.duration_days = numbers[0] * 7
                    else:
                        protocol.duration_days = numbers[0]
    
    async def generate_power_analysis(self, protocol_id: str) -> Dict[str, Any]:
        """Generate statistical power analysis for a protocol."""
        
        if protocol_id not in self.protocols:
            raise ResearchError(f"Protocol {protocol_id} not found")
        
        protocol = self.protocols[protocol_id]
        
        power_prompt = f"""
        Perform a statistical power analysis for this experiment:
        
        - Type: {protocol.experiment_type.value}
        - Sample Size: {protocol.sample_size}
        - Variables: {json.dumps(protocol.variables, indent=2)}
        - Success Criteria: {protocol.success_criteria}
        
        Provide:
        1. Statistical power estimate (0-1)
        2. Effect size assumptions
        3. Alpha level recommendation
        4. Minimum sample size recommendations
        5. Power curve analysis
        6. Recommendations for increasing power
        """
        
        try:
            power_analysis = await self.ai_client.generate_text(
                power_prompt,
                max_tokens=800,
                temperature=0.3
            )
            
            return {
                "protocol_id": protocol_id,
                "power_analysis": power_analysis,
                "timestamp": datetime.now().isoformat(),
                "sample_size_current": protocol.sample_size
            }
            
        except Exception as e:
            raise ResearchError(f"Power analysis failed: {e}")
    
    def get_protocol(self, protocol_id: str) -> Optional[ExperimentProtocol]:
        """Get protocol by ID."""
        return self.protocols.get(protocol_id)
    
    def list_protocols(self, status: str = None) -> List[ExperimentProtocol]:
        """List protocols, optionally filtered by status."""
        protocols = list(self.protocols.values())
        
        if status:
            protocols = [p for p in protocols if p.status == status]
        
        return sorted(protocols, key=lambda p: p.created_at, reverse=True)
    
    async def validate_protocol(self, protocol_id: str) -> Dict[str, Any]:
        """Validate protocol for scientific rigor and feasibility."""
        
        if protocol_id not in self.protocols:
            raise ResearchError(f"Protocol {protocol_id} not found")
        
        protocol = self.protocols[protocol_id]
        
        validation_prompt = f"""
        Validate this experimental protocol for scientific rigor and feasibility:
        
        Protocol Summary:
        - Type: {protocol.experiment_type.value}
        - Hypothesis: {protocol.hypothesis}
        - Sample Size: {protocol.sample_size}
        - Duration: {protocol.duration_days} days
        - Variables: {json.dumps(protocol.variables, indent=2)}
        - Methodology: {json.dumps(protocol.methodology, indent=2)}
        
        Check for:
        1. Scientific validity
        2. Statistical adequacy
        3. Feasibility concerns
        4. Ethical compliance
        5. Missing elements
        6. Potential improvements
        
        Provide a validation score (0-10) and detailed feedback.
        """
        
        try:
            validation_result = await self.ai_client.generate_text(
                validation_prompt,
                max_tokens=1000,
                temperature=0.4
            )
            
            return {
                "protocol_id": protocol_id,
                "validation": validation_result,
                "timestamp": datetime.now().isoformat(),
                "validator": "ExperimentDesigner"
            }
            
        except Exception as e:
            raise ResearchError(f"Protocol validation failed: {e}")
    
    def export_protocols_summary(self) -> str:
        """Export summary of all protocols."""
        
        summary = "# Experiment Protocols Summary\n\n"
        summary += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"Total Protocols: {len(self.protocols)}\n\n"
        
        # Status overview
        status_counts = {}
        for protocol in self.protocols.values():
            status_counts[protocol.status] = status_counts.get(protocol.status, 0) + 1
        
        summary += "## Status Overview\n"
        for status, count in status_counts.items():
            summary += f"- {status.title()}: {count}\n"
        summary += "\n"
        
        # Protocol details
        summary += "## Protocol Details\n\n"
        for protocol in sorted(self.protocols.values(), key=lambda p: p.created_at, reverse=True):
            summary += f"### {protocol.title}\n"
            summary += f"- **ID**: {protocol.id}\n"
            summary += f"- **Type**: {protocol.experiment_type.value.title()}\n"
            summary += f"- **Status**: {protocol.status}\n"
            summary += f"- **Sample Size**: {protocol.sample_size}\n"
            summary += f"- **Duration**: {protocol.duration_days} days\n"
            summary += f"- **Created**: {protocol.created_at.strftime('%Y-%m-%d')}\n\n"
        
        return summary