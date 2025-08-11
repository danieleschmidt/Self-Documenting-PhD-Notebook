"""
Hypothesis Generation and Testing Engine - Core research functionality.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import dataclass

from ..ai.client_factory import AIClientFactory
from ..core.note import Note, NoteType
from ..utils.exceptions import ResearchError


@dataclass
class Hypothesis:
    """Research hypothesis with testability metrics."""
    
    id: str
    statement: str
    confidence: float
    testability_score: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    variables: Dict[str, str]
    methodology_suggestions: List[str]
    estimated_duration: int  # days
    resources_required: List[str]
    created_at: datetime
    status: str = "draft"  # draft, active, tested, validated, refuted


class HypothesisEngine:
    """AI-powered hypothesis generation and testing engine."""
    
    def __init__(self, notebook=None, ai_provider="auto"):
        self.notebook = notebook
        self.ai_client = AIClientFactory.get_client(ai_provider)
        self.hypotheses: Dict[str, Hypothesis] = {}
    
    async def generate_hypotheses(
        self, 
        research_context: str,
        observations: List[str],
        max_hypotheses: int = 5
    ) -> List[Hypothesis]:
        """Generate research hypotheses from observations and context."""
        
        prompt = f"""
        Based on this research context and observations, generate {max_hypotheses} 
        testable research hypotheses. For each hypothesis, provide:
        
        Research Context: {research_context}
        
        Observations:
        {chr(10).join([f"- {obs}" for obs in observations])}
        
        For each hypothesis, provide:
        1. Clear, testable statement
        2. Independent and dependent variables
        3. Methodology suggestions
        4. Estimated duration and resources
        5. Confidence level (0-1)
        
        Format as JSON array.
        """
        
        try:
            response = await self.ai_client.generate_text(
                prompt, 
                max_tokens=1500,
                temperature=0.7
            )
            
            # Parse AI response into hypotheses
            hypotheses = []
            hypothesis_data = self._parse_hypothesis_response(response)
            
            for i, data in enumerate(hypothesis_data):
                hypothesis = Hypothesis(
                    id=f"hyp_{datetime.now().strftime('%Y%m%d')}_{i+1:03d}",
                    statement=data.get('statement', ''),
                    confidence=float(data.get('confidence', 0.5)),
                    testability_score=await self._assess_testability(data.get('statement', '')),
                    supporting_evidence=[],
                    contradicting_evidence=[],
                    variables=data.get('variables', {}),
                    methodology_suggestions=data.get('methodology', []),
                    estimated_duration=int(data.get('duration_days', 30)),
                    resources_required=data.get('resources', []),
                    created_at=datetime.now()
                )
                
                self.hypotheses[hypothesis.id] = hypothesis
                hypotheses.append(hypothesis)
            
            return hypotheses
            
        except Exception as e:
            raise ResearchError(f"Hypothesis generation failed: {e}")
    
    async def _assess_testability(self, hypothesis_statement: str) -> float:
        """Assess how testable a hypothesis is (0-1 score)."""
        
        assessment_prompt = f"""
        Rate the testability of this hypothesis on a scale of 0-1:
        
        Hypothesis: {hypothesis_statement}
        
        Consider:
        - Are variables clearly defined and measurable?
        - Is the relationship specific enough to test?
        - Can it be falsified?
        - Are methods available to test it?
        
        Return only a number between 0 and 1.
        """
        
        try:
            response = await self.ai_client.generate_text(
                assessment_prompt, 
                max_tokens=10,
                temperature=0.1
            )
            
            score = float(response.strip())
            return max(0.0, min(1.0, score))  # Clamp to [0,1]
            
        except (ValueError, Exception):
            return 0.5  # Default moderate testability
    
    def _parse_hypothesis_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse AI response into structured hypothesis data."""
        
        try:
            # Try to parse as JSON first
            if response.strip().startswith('['):
                return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Fallback: Parse structured text
        hypotheses = []
        lines = response.strip().split('\n')
        current_hypothesis = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('Hypothesis') and current_hypothesis:
                hypotheses.append(current_hypothesis)
                current_hypothesis = {}
            
            if 'Statement:' in line:
                current_hypothesis['statement'] = line.split('Statement:')[1].strip()
            elif 'Confidence:' in line:
                try:
                    current_hypothesis['confidence'] = float(line.split('Confidence:')[1].strip())
                except ValueError:
                    current_hypothesis['confidence'] = 0.5
            elif 'Variables:' in line:
                current_hypothesis['variables'] = {'main': line.split('Variables:')[1].strip()}
        
        if current_hypothesis:
            hypotheses.append(current_hypothesis)
        
        return hypotheses[:5]  # Limit to 5 hypotheses
    
    async def validate_hypothesis(
        self, 
        hypothesis_id: str, 
        experimental_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate hypothesis against experimental results."""
        
        if hypothesis_id not in self.hypotheses:
            raise ResearchError(f"Hypothesis {hypothesis_id} not found")
        
        hypothesis = self.hypotheses[hypothesis_id]
        
        validation_prompt = f"""
        Evaluate this hypothesis against experimental data:
        
        Hypothesis: {hypothesis.statement}
        Variables: {hypothesis.variables}
        
        Experimental Data: {json.dumps(experimental_data, indent=2)}
        
        Provide:
        1. Validation result (supported/refuted/inconclusive)
        2. Statistical significance (if applicable)
        3. Effect size
        4. Limitations and next steps
        5. Confidence in conclusion (0-1)
        """
        
        try:
            validation_result = await self.ai_client.generate_text(
                validation_prompt,
                max_tokens=500,
                temperature=0.3
            )
            
            # Update hypothesis status based on result
            if "supported" in validation_result.lower():
                hypothesis.status = "validated"
            elif "refuted" in validation_result.lower():
                hypothesis.status = "refuted"
            else:
                hypothesis.status = "tested"
            
            # Create validation note if notebook available
            if self.notebook:
                self._create_validation_note(hypothesis, validation_result, experimental_data)
            
            return {
                "hypothesis_id": hypothesis_id,
                "result": validation_result,
                "status": hypothesis.status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise ResearchError(f"Hypothesis validation failed: {e}")
    
    def _create_validation_note(
        self, 
        hypothesis: Hypothesis, 
        validation_result: str,
        experimental_data: Dict[str, Any]
    ):
        """Create a note documenting hypothesis validation."""
        
        note_content = f"""
# Hypothesis Validation: {hypothesis.id}

## Original Hypothesis
{hypothesis.statement}

## Experimental Data
```json
{json.dumps(experimental_data, indent=2)}
```

## Validation Result
{validation_result}

## Hypothesis Details
- **Confidence**: {hypothesis.confidence:.2f}
- **Testability Score**: {hypothesis.testability_score:.2f}
- **Status**: {hypothesis.status}
- **Duration**: {hypothesis.estimated_duration} days
- **Created**: {hypothesis.created_at.strftime('%Y-%m-%d')}

## Variables Tested
{json.dumps(hypothesis.variables, indent=2)}

## Methodology Used
{chr(10).join([f"- {method}" for method in hypothesis.methodology_suggestions])}
        """
        
        validation_note = self.notebook.create_note(
            title=f"Hypothesis Validation - {hypothesis.id}",
            content=note_content,
            note_type=NoteType.EXPERIMENT,
            tags=["#hypothesis", "#validation", f"#{hypothesis.status}"]
        )
        
        return validation_note
    
    def get_hypothesis_by_id(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Get hypothesis by ID."""
        return self.hypotheses.get(hypothesis_id)
    
    def list_hypotheses(self, status: str = None) -> List[Hypothesis]:
        """List hypotheses, optionally filtered by status."""
        hypotheses = list(self.hypotheses.values())
        
        if status:
            hypotheses = [h for h in hypotheses if h.status == status]
        
        return sorted(hypotheses, key=lambda h: h.created_at, reverse=True)
    
    async def suggest_next_experiments(self, hypothesis_id: str) -> List[Dict[str, Any]]:
        """Suggest follow-up experiments for a hypothesis."""
        
        if hypothesis_id not in self.hypotheses:
            raise ResearchError(f"Hypothesis {hypothesis_id} not found")
        
        hypothesis = self.hypotheses[hypothesis_id]
        
        suggestion_prompt = f"""
        Based on this hypothesis and its current status, suggest 3-5 follow-up experiments:
        
        Hypothesis: {hypothesis.statement}
        Status: {hypothesis.status}
        Variables: {hypothesis.variables}
        Previous methodology: {hypothesis.methodology_suggestions}
        
        For each experiment, provide:
        1. Experiment title
        2. Objective
        3. Methodology overview
        4. Expected outcome
        5. Priority (1-5)
        """
        
        try:
            suggestions = await self.ai_client.generate_text(
                suggestion_prompt,
                max_tokens=800,
                temperature=0.6
            )
            
            return self._parse_experiment_suggestions(suggestions)
            
        except Exception as e:
            raise ResearchError(f"Experiment suggestion failed: {e}")
    
    def _parse_experiment_suggestions(self, suggestions_text: str) -> List[Dict[str, Any]]:
        """Parse experiment suggestions from AI response."""
        
        experiments = []
        lines = suggestions_text.strip().split('\n')
        current_exp = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith(('Experiment', '1.', '2.', '3.', '4.', '5.')):
                if current_exp:
                    experiments.append(current_exp)
                    current_exp = {}
                current_exp['title'] = line
            elif 'Objective:' in line:
                current_exp['objective'] = line.split('Objective:')[1].strip()
            elif 'Methodology:' in line:
                current_exp['methodology'] = line.split('Methodology:')[1].strip()
            elif 'Priority:' in line:
                try:
                    priority_str = line.split('Priority:')[1].strip()
                    current_exp['priority'] = int(priority_str[0])
                except (ValueError, IndexError):
                    current_exp['priority'] = 3
        
        if current_exp:
            experiments.append(current_exp)
        
        return experiments[:5]  # Limit to 5 suggestions
    
    def export_hypotheses_summary(self) -> str:
        """Export summary of all hypotheses for reporting."""
        
        summary = "# Hypothesis Summary Report\n\n"
        summary += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        status_counts = {}
        for hypothesis in self.hypotheses.values():
            status_counts[hypothesis.status] = status_counts.get(hypothesis.status, 0) + 1
        
        summary += "## Status Overview\n"
        for status, count in status_counts.items():
            summary += f"- {status.title()}: {count}\n"
        
        summary += "\n## Hypothesis Details\n\n"
        
        for hypothesis in sorted(self.hypotheses.values(), key=lambda h: h.created_at, reverse=True):
            summary += f"### {hypothesis.id}\n"
            summary += f"**Statement**: {hypothesis.statement}\n\n"
            summary += f"- **Status**: {hypothesis.status}\n"
            summary += f"- **Confidence**: {hypothesis.confidence:.2f}\n"
            summary += f"- **Testability**: {hypothesis.testability_score:.2f}\n"
            summary += f"- **Created**: {hypothesis.created_at.strftime('%Y-%m-%d')}\n\n"
        
        return summary