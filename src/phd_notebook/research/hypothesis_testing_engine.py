"""
Advanced Hypothesis Testing Engine for Research
Implements hypothesis-driven research with automated testing and validation.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from pathlib import Path

from ..core.note import Note, NoteType
from ..utils.logging import setup_logger


class HypothesisStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    TESTING = "testing"
    VALIDATED = "validated"
    REJECTED = "rejected"
    REFINED = "refined"


class EvidenceType(Enum):
    EXPERIMENTAL = "experimental"
    OBSERVATIONAL = "observational"
    LITERATURE = "literature"
    COMPUTATIONAL = "computational"
    THEORETICAL = "theoretical"


@dataclass
class Evidence:
    """Single piece of evidence for or against a hypothesis."""
    id: str
    type: EvidenceType
    description: str
    source: str
    confidence: float  # 0.0 to 1.0
    supports_hypothesis: bool
    timestamp: datetime
    data: Optional[Dict] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['type'] = self.type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class Hypothesis:
    """Research hypothesis with testing capabilities."""
    id: str
    title: str
    statement: str
    field: str
    subfield: str
    author: str
    created_at: datetime
    status: HypothesisStatus
    
    # Hypothesis components
    independent_variables: List[str]
    dependent_variables: List[str]
    predicted_relationship: str
    testable_predictions: List[str]
    
    # Evidence and validation
    evidence: List[Evidence]
    confidence_score: float = 0.0
    statistical_power: Optional[float] = None
    effect_size: Optional[float] = None
    
    # Research context
    related_hypotheses: List[str] = None
    background_literature: List[str] = None
    methodology: Optional[str] = None
    
    # Tracking
    last_updated: Optional[datetime] = None
    validation_criteria: List[str] = None
    rejection_criteria: List[str] = None
    
    def __post_init__(self):
        if self.related_hypotheses is None:
            self.related_hypotheses = []
        if self.background_literature is None:
            self.background_literature = []
        if self.validation_criteria is None:
            self.validation_criteria = []
        if self.rejection_criteria is None:
            self.rejection_criteria = []
            
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['status'] = self.status.value
        result['created_at'] = self.created_at.isoformat()
        result['last_updated'] = self.last_updated.isoformat() if self.last_updated else None
        result['evidence'] = [e.to_dict() for e in self.evidence]
        return result


class HypothesisTestingEngine:
    """
    Advanced engine for hypothesis-driven research with automated testing,
    validation, and research optimization.
    """
    
    def __init__(self, notebook_path: Path):
        self.logger = setup_logger("research.hypothesis_engine")
        self.notebook_path = notebook_path
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.evidence_store: Dict[str, Evidence] = {}
        
        # Create research directories
        self.hypotheses_dir = notebook_path / "research" / "hypotheses"
        self.evidence_dir = notebook_path / "research" / "evidence"
        self.analyses_dir = notebook_path / "research" / "analyses"
        
        for dir_path in [self.hypotheses_dir, self.evidence_dir, self.analyses_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self._load_existing_data()
        
    def _load_existing_data(self):
        """Load existing hypotheses and evidence from storage."""
        try:
            # Load hypotheses
            for hyp_file in self.hypotheses_dir.glob("*.json"):
                with open(hyp_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    hypothesis = self._dict_to_hypothesis(data)
                    self.hypotheses[hypothesis.id] = hypothesis
                    
            # Load evidence
            for ev_file in self.evidence_dir.glob("*.json"):
                with open(ev_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    evidence = self._dict_to_evidence(data)
                    self.evidence_store[evidence.id] = evidence
                    
            self.logger.info(f"Loaded {len(self.hypotheses)} hypotheses and {len(self.evidence_store)} evidence items")
            
        except Exception as e:
            self.logger.error(f"Error loading existing data: {e}")
    
    def _dict_to_hypothesis(self, data: Dict) -> Hypothesis:
        """Convert dictionary to Hypothesis object."""
        # Convert string dates back to datetime
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data['last_updated']:
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        data['status'] = HypothesisStatus(data['status'])
        
        # Convert evidence
        evidence_list = []
        for ev_data in data.get('evidence', []):
            evidence_list.append(self._dict_to_evidence(ev_data))
        data['evidence'] = evidence_list
        
        return Hypothesis(**data)
    
    def _dict_to_evidence(self, data: Dict) -> Evidence:
        """Convert dictionary to Evidence object."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['type'] = EvidenceType(data['type'])
        return Evidence(**data)
    
    def create_hypothesis(
        self,
        title: str,
        statement: str,
        field: str,
        author: str,
        independent_vars: List[str],
        dependent_vars: List[str],
        predicted_relationship: str,
        subfield: str = "",
        testable_predictions: List[str] = None,
        **kwargs
    ) -> Hypothesis:
        """
        Create a new research hypothesis with automatic validation.
        """
        # Generate unique ID
        hypothesis_id = f"hyp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate hypothesis components
        if not all([title, statement, field, author]):
            raise ValueError("Title, statement, field, and author are required")
            
        if not independent_vars or not dependent_vars:
            raise ValueError("At least one independent and dependent variable required")
        
        # Create hypothesis
        hypothesis = Hypothesis(
            id=hypothesis_id,
            title=title.strip(),
            statement=statement.strip(),
            field=field.strip(),
            subfield=subfield.strip(),
            author=author.strip(),
            created_at=datetime.now(),
            status=HypothesisStatus.DRAFT,
            independent_variables=independent_vars,
            dependent_variables=dependent_vars,
            predicted_relationship=predicted_relationship,
            testable_predictions=testable_predictions or [],
            evidence=[],
            **kwargs
        )
        
        # Auto-generate validation criteria if not provided
        if not hypothesis.validation_criteria:
            hypothesis.validation_criteria = self._generate_validation_criteria(hypothesis)
        
        # Store hypothesis
        self.hypotheses[hypothesis_id] = hypothesis
        self._save_hypothesis(hypothesis)
        
        # Create hypothesis note
        self._create_hypothesis_note(hypothesis)
        
        self.logger.info(f"Created hypothesis: {hypothesis_id} - {title}")
        return hypothesis
    
    def _generate_validation_criteria(self, hypothesis: Hypothesis) -> List[str]:
        """Auto-generate validation criteria based on hypothesis structure."""
        criteria = []
        
        # Statistical criteria
        criteria.append("Statistical significance (p < 0.05)")
        criteria.append("Effect size > 0.2 (small effect)")
        criteria.append("Minimum sample size met (power >= 0.8)")
        
        # Replication criteria
        criteria.append("Results replicated in independent study")
        criteria.append("Consistent direction of effect across studies")
        
        # Domain-specific criteria
        if "machine learning" in hypothesis.field.lower():
            criteria.extend([
                "Cross-validation accuracy > baseline + 5%",
                "Results hold across different datasets"
            ])
        elif "psychology" in hypothesis.field.lower():
            criteria.extend([
                "Effect observed in diverse populations",
                "Controlled for confounding variables"
            ])
        elif "physics" in hypothesis.field.lower():
            criteria.extend([
                "Theoretical predictions match experimental data",
                "Results within measurement uncertainty"
            ])
            
        return criteria
    
    def add_evidence(
        self,
        hypothesis_id: str,
        evidence_type: EvidenceType,
        description: str,
        source: str,
        supports_hypothesis: bool,
        confidence: float,
        data: Optional[Dict] = None,
        notes: Optional[str] = None
    ) -> Evidence:
        """Add evidence for or against a hypothesis."""
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")
            
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        # Create evidence
        evidence_id = f"ev_{hypothesis_id}_{len(self.evidence_store) + 1:04d}"
        evidence = Evidence(
            id=evidence_id,
            type=evidence_type,
            description=description,
            source=source,
            confidence=confidence,
            supports_hypothesis=supports_hypothesis,
            timestamp=datetime.now(),
            data=data,
            notes=notes
        )
        
        # Add to stores
        self.evidence_store[evidence_id] = evidence
        self.hypotheses[hypothesis_id].evidence.append(evidence)
        self.hypotheses[hypothesis_id].last_updated = datetime.now()
        
        # Update confidence score
        self._update_hypothesis_confidence(hypothesis_id)
        
        # Save updates
        self._save_evidence(evidence)
        self._save_hypothesis(self.hypotheses[hypothesis_id])
        
        # Update hypothesis note
        self._update_hypothesis_note(self.hypotheses[hypothesis_id])
        
        self.logger.info(f"Added evidence {evidence_id} to hypothesis {hypothesis_id}")
        return evidence
    
    def _update_hypothesis_confidence(self, hypothesis_id: str):
        """Update hypothesis confidence based on evidence."""
        hypothesis = self.hypotheses[hypothesis_id]
        
        if not hypothesis.evidence:
            hypothesis.confidence_score = 0.0
            return
            
        # Calculate weighted confidence
        supporting_evidence = []
        contradicting_evidence = []
        
        for evidence in hypothesis.evidence:
            weight = evidence.confidence
            if evidence.supports_hypothesis:
                supporting_evidence.append(weight)
            else:
                contradicting_evidence.append(weight)
        
        # Simple Bayesian-style update
        total_support = sum(supporting_evidence) if supporting_evidence else 0
        total_contradiction = sum(contradicting_evidence) if contradicting_evidence else 0
        
        if total_support + total_contradiction == 0:
            confidence = 0.0
        else:
            # Scale to [0, 1] with bias toward requiring strong evidence
            confidence = total_support / (total_support + total_contradiction + 1.0)
            
            # Apply evidence quality bonus
            avg_evidence_quality = statistics.mean([e.confidence for e in hypothesis.evidence])
            confidence *= avg_evidence_quality
        
        hypothesis.confidence_score = round(confidence, 3)
        
        # Auto-update status based on confidence
        if hypothesis.confidence_score >= 0.8 and len(hypothesis.evidence) >= 3:
            hypothesis.status = HypothesisStatus.VALIDATED
        elif hypothesis.confidence_score <= 0.2 and len(hypothesis.evidence) >= 3:
            hypothesis.status = HypothesisStatus.REJECTED
        elif hypothesis.evidence:
            hypothesis.status = HypothesisStatus.TESTING
    
    def analyze_hypothesis(self, hypothesis_id: str) -> Dict[str, Any]:
        """Perform comprehensive analysis of a hypothesis."""
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")
            
        hypothesis = self.hypotheses[hypothesis_id]
        
        # Basic statistics
        total_evidence = len(hypothesis.evidence)
        supporting_count = sum(1 for e in hypothesis.evidence if e.supports_hypothesis)
        contradicting_count = total_evidence - supporting_count
        
        # Evidence quality analysis
        if hypothesis.evidence:
            evidence_qualities = [e.confidence for e in hypothesis.evidence]
            avg_evidence_quality = statistics.mean(evidence_qualities)
            evidence_std = statistics.stdev(evidence_qualities) if len(evidence_qualities) > 1 else 0
        else:
            avg_evidence_quality = 0
            evidence_std = 0
        
        # Time analysis
        if hypothesis.evidence:
            evidence_dates = [e.timestamp for e in hypothesis.evidence]
            research_duration = (max(evidence_dates) - min(evidence_dates)).days
            evidence_velocity = len(hypothesis.evidence) / max(research_duration, 1)
        else:
            research_duration = 0
            evidence_velocity = 0
        
        # Evidence type distribution
        evidence_types = {}
        for evidence in hypothesis.evidence:
            type_name = evidence.type.value
            evidence_types[type_name] = evidence_types.get(type_name, 0) + 1
        
        # Research recommendations
        recommendations = self._generate_research_recommendations(hypothesis)
        
        analysis = {
            "hypothesis_id": hypothesis_id,
            "current_status": hypothesis.status.value,
            "confidence_score": hypothesis.confidence_score,
            "evidence_summary": {
                "total_evidence": total_evidence,
                "supporting": supporting_count,
                "contradicting": contradicting_count,
                "support_ratio": supporting_count / max(total_evidence, 1),
                "avg_quality": round(avg_evidence_quality, 3),
                "quality_std": round(evidence_std, 3)
            },
            "research_progress": {
                "duration_days": research_duration,
                "evidence_velocity": round(evidence_velocity, 2),
                "evidence_types": evidence_types
            },
            "validation_status": self._check_validation_criteria(hypothesis),
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Save analysis
        self._save_analysis(hypothesis_id, analysis)
        
        return analysis
    
    def _generate_research_recommendations(self, hypothesis: Hypothesis) -> List[Dict[str, str]]:
        """Generate AI-powered research recommendations."""
        recommendations = []
        
        # Evidence diversity recommendations
        evidence_types = set(e.type for e in hypothesis.evidence)
        all_types = set(EvidenceType)
        missing_types = all_types - evidence_types
        
        if missing_types:
            for missing_type in missing_types:
                recommendations.append({
                    "type": "evidence_diversification",
                    "priority": "medium",
                    "action": f"Collect {missing_type.value} evidence",
                    "rationale": "Diversifying evidence types strengthens hypothesis validation"
                })
        
        # Sample size recommendations
        if len(hypothesis.evidence) < 5:
            recommendations.append({
                "type": "sample_size",
                "priority": "high",
                "action": "Collect more evidence",
                "rationale": f"Need minimum 5 pieces of evidence (current: {len(hypothesis.evidence)})"
            })
        
        # Quality improvement recommendations
        if hypothesis.evidence:
            low_quality_evidence = [e for e in hypothesis.evidence if e.confidence < 0.5]
            if low_quality_evidence:
                recommendations.append({
                    "type": "quality_improvement",
                    "priority": "medium",
                    "action": "Strengthen evidence quality",
                    "rationale": f"{len(low_quality_evidence)} evidence items have low confidence"
                })
        
        # Statistical power recommendations
        if not hypothesis.statistical_power or hypothesis.statistical_power < 0.8:
            recommendations.append({
                "type": "statistical_power",
                "priority": "high",
                "action": "Increase sample size or effect size",
                "rationale": "Statistical power should be ≥ 0.8 for reliable results"
            })
        
        # Replication recommendations
        experimental_evidence = [e for e in hypothesis.evidence if e.type == EvidenceType.EXPERIMENTAL]
        if len(experimental_evidence) == 1:
            recommendations.append({
                "type": "replication",
                "priority": "high",
                "action": "Replicate experimental findings",
                "rationale": "Single experimental result needs replication for validation"
            })
        
        return recommendations
    
    def _check_validation_criteria(self, hypothesis: Hypothesis) -> Dict[str, bool]:
        """Check which validation criteria are met."""
        status = {}
        
        for criterion in hypothesis.validation_criteria:
            # Simple rule-based checking (in production, this would be more sophisticated)
            if "statistical significance" in criterion.lower():
                # Check if we have statistical data
                statistical_evidence = any(
                    e.data and "p_value" in e.data 
                    for e in hypothesis.evidence 
                    if e.supports_hypothesis
                )
                status[criterion] = statistical_evidence
                
            elif "effect size" in criterion.lower():
                # Check for effect size data
                effect_evidence = any(
                    e.data and "effect_size" in e.data 
                    for e in hypothesis.evidence
                )
                status[criterion] = effect_evidence
                
            elif "replicated" in criterion.lower():
                # Check for multiple experimental evidence
                experimental_count = sum(
                    1 for e in hypothesis.evidence 
                    if e.type == EvidenceType.EXPERIMENTAL and e.supports_hypothesis
                )
                status[criterion] = experimental_count >= 2
                
            else:
                # Default: check if we have any supporting evidence
                status[criterion] = any(e.supports_hypothesis for e in hypothesis.evidence)
        
        return status
    
    def _create_hypothesis_note(self, hypothesis: Hypothesis):
        """Create an Obsidian note for the hypothesis."""
        note_path = self.hypotheses_dir / f"{hypothesis.id}_{hypothesis.title.replace(' ', '_')}.md"
        
        content = f"""# {hypothesis.title}

## Hypothesis Statement
{hypothesis.statement}

## Research Context
- **Field**: {hypothesis.field}
- **Subfield**: {hypothesis.subfield}
- **Author**: {hypothesis.author}
- **Status**: {hypothesis.status.value}
- **Confidence**: {hypothesis.confidence_score:.1%}

## Variables
- **Independent Variables**: {', '.join(hypothesis.independent_variables)}
- **Dependent Variables**: {', '.join(hypothesis.dependent_variables)}
- **Predicted Relationship**: {hypothesis.predicted_relationship}

## Testable Predictions
{"".join(f"- {pred}\\n" for pred in hypothesis.testable_predictions)}

## Validation Criteria
{"".join(f"- {criteria}\\n" for criteria in hypothesis.validation_criteria)}

## Evidence Summary
- Total Evidence: {len(hypothesis.evidence)}
- Supporting: {sum(1 for e in hypothesis.evidence if e.supports_hypothesis)}
- Contradicting: {sum(1 for e in hypothesis.evidence if not e.supports_hypothesis)}

## Research Notes
[Add your research notes, observations, and insights here]

## Related Work
- Related Hypotheses: {', '.join(hypothesis.related_hypotheses) if hypothesis.related_hypotheses else 'None'}
- Background Literature: {', '.join(hypothesis.background_literature) if hypothesis.background_literature else 'None'}

---
*Created: {hypothesis.created_at.strftime('%Y-%m-%d %H:%M:%S')}*  
*Last Updated: {(hypothesis.last_updated or hypothesis.created_at).strftime('%Y-%m-%d %H:%M:%S')}*  
*Hypothesis ID: {hypothesis.id}*
"""
        
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _update_hypothesis_note(self, hypothesis: Hypothesis):
        """Update the Obsidian note for the hypothesis."""
        # For simplicity, recreate the note (in production, would do smarter updates)
        self._create_hypothesis_note(hypothesis)
    
    def _save_hypothesis(self, hypothesis: Hypothesis):
        """Save hypothesis to JSON storage."""
        file_path = self.hypotheses_dir / f"{hypothesis.id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(hypothesis.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _save_evidence(self, evidence: Evidence):
        """Save evidence to JSON storage."""
        file_path = self.evidence_dir / f"{evidence.id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(evidence.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _save_analysis(self, hypothesis_id: str, analysis: Dict):
        """Save analysis results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = self.analyses_dir / f"{hypothesis_id}_analysis_{timestamp}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    def get_research_dashboard(self) -> Dict[str, Any]:
        """Generate a comprehensive research dashboard."""
        active_hypotheses = [h for h in self.hypotheses.values() if h.status != HypothesisStatus.REJECTED]
        
        # Confidence distribution
        if active_hypotheses:
            confidences = [h.confidence_score for h in active_hypotheses]
            avg_confidence = statistics.mean(confidences)
            confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0
        else:
            avg_confidence = confidence_std = 0
        
        # Status distribution
        status_counts = {}
        for status in HypothesisStatus:
            status_counts[status.value] = sum(1 for h in self.hypotheses.values() if h.status == status)
        
        # Research velocity
        if self.hypotheses:
            creation_dates = [h.created_at for h in self.hypotheses.values()]
            research_span = (max(creation_dates) - min(creation_dates)).days
            hypothesis_velocity = len(self.hypotheses) / max(research_span, 1)
        else:
            research_span = 0
            hypothesis_velocity = 0
        
        # Field distribution
        fields = {}
        for hypothesis in self.hypotheses.values():
            field = hypothesis.field
            fields[field] = fields.get(field, 0) + 1
        
        return {
            "summary": {
                "total_hypotheses": len(self.hypotheses),
                "active_hypotheses": len(active_hypotheses),
                "validated_hypotheses": status_counts.get("validated", 0),
                "total_evidence": len(self.evidence_store),
                "avg_confidence": round(avg_confidence, 3),
                "confidence_std": round(confidence_std, 3)
            },
            "status_distribution": status_counts,
            "research_velocity": {
                "hypotheses_per_day": round(hypothesis_velocity, 3),
                "research_span_days": research_span
            },
            "field_distribution": fields,
            "recent_activity": self._get_recent_activity(),
            "dashboard_generated": datetime.now().isoformat()
        }
    
    def _get_recent_activity(self, days: int = 7) -> List[Dict]:
        """Get recent research activity."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_activities = []
        
        # Recent hypotheses
        for hypothesis in self.hypotheses.values():
            if hypothesis.created_at >= cutoff_date:
                recent_activities.append({
                    "type": "hypothesis_created",
                    "timestamp": hypothesis.created_at.isoformat(),
                    "description": f"Created hypothesis: {hypothesis.title}",
                    "hypothesis_id": hypothesis.id
                })
        
        # Recent evidence
        for evidence in self.evidence_store.values():
            if evidence.timestamp >= cutoff_date:
                recent_activities.append({
                    "type": "evidence_added",
                    "timestamp": evidence.timestamp.isoformat(),
                    "description": f"Added evidence: {evidence.description[:50]}...",
                    "evidence_id": evidence.id,
                    "supports": evidence.supports_hypothesis
                })
        
        return sorted(recent_activities, key=lambda x: x["timestamp"], reverse=True)


# Integration with main notebook system
class ResearchDrivenNotebook:
    """
    Extension of ResearchNotebook with hypothesis-driven research capabilities.
    """
    
    def __init__(self, notebook_instance):
        self.notebook = notebook_instance
        self.hypothesis_engine = HypothesisTestingEngine(notebook_instance.vault_path)
        self.logger = setup_logger("research.driven_notebook")
    
    def create_research_hypothesis(self, **kwargs) -> Hypothesis:
        """Create a new research hypothesis."""
        return self.hypothesis_engine.create_hypothesis(**kwargs)
    
    def add_experimental_evidence(self, hypothesis_id: str, experiment_results: Dict, **kwargs) -> Evidence:
        """Add experimental evidence with automatic statistical analysis."""
        # Enhanced evidence with statistical processing
        processed_data = self._process_experimental_data(experiment_results)
        
        return self.hypothesis_engine.add_evidence(
            hypothesis_id=hypothesis_id,
            evidence_type=EvidenceType.EXPERIMENTAL,
            data=processed_data,
            **kwargs
        )
    
    def _process_experimental_data(self, results: Dict) -> Dict:
        """Process experimental data for statistical insights."""
        processed = results.copy()
        
        # Add statistical measures if raw data provided
        if 'measurements' in results and isinstance(results['measurements'], list):
            measurements = results['measurements']
            processed.update({
                'sample_size': len(measurements),
                'mean': statistics.mean(measurements),
                'std_dev': statistics.stdev(measurements) if len(measurements) > 1 else 0,
                'confidence_interval_95': self._calculate_confidence_interval(measurements)
            })
        
        return processed
    
    def _calculate_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for data."""
        if len(data) < 2:
            return (0.0, 0.0)
            
        mean = statistics.mean(data)
        std_err = statistics.stdev(data) / (len(data) ** 0.5)
        # Simplified: using 1.96 for 95% CI (should use t-distribution for small samples)
        margin = 1.96 * std_err if confidence == 0.95 else 2.58 * std_err
        
        return (mean - margin, mean + margin)
    
    def get_research_insights(self) -> Dict[str, Any]:
        """Get AI-powered insights about research progress."""
        dashboard = self.hypothesis_engine.get_research_dashboard()
        
        # Generate insights
        insights = {
            "research_health": self._assess_research_health(dashboard),
            "recommendations": self._generate_research_recommendations(dashboard),
            "focus_areas": self._identify_focus_areas(dashboard),
            "collaboration_opportunities": self._suggest_collaborations(dashboard)
        }
        
        return insights
    
    def _assess_research_health(self, dashboard: Dict) -> Dict[str, str]:
        """Assess overall research health."""
        summary = dashboard['summary']
        
        health_score = 0
        health_factors = []
        
        # Activity level
        if summary['total_hypotheses'] >= 5:
            health_score += 25
            health_factors.append("Good hypothesis generation")
        elif summary['total_hypotheses'] >= 2:
            health_score += 15
        
        # Validation rate
        validation_rate = summary['validated_hypotheses'] / max(summary['total_hypotheses'], 1)
        if validation_rate >= 0.3:
            health_score += 25
            health_factors.append("High validation rate")
        elif validation_rate >= 0.1:
            health_score += 15
        
        # Evidence quality
        if summary['avg_confidence'] >= 0.7:
            health_score += 25
            health_factors.append("High-quality evidence")
        elif summary['avg_confidence'] >= 0.5:
            health_score += 15
        
        # Research pace
        velocity = dashboard['research_velocity']['hypotheses_per_day']
        if velocity >= 0.1:
            health_score += 25
            health_factors.append("Good research pace")
        elif velocity >= 0.05:
            health_score += 15
        
        if health_score >= 80:
            health_status = "Excellent"
        elif health_score >= 60:
            health_status = "Good"
        elif health_score >= 40:
            health_status = "Fair"
        else:
            health_status = "Needs Improvement"
        
        return {
            "overall_score": health_score,
            "status": health_status,
            "positive_factors": health_factors,
            "improvement_areas": self._identify_improvement_areas(dashboard)
        }
    
    def _identify_improvement_areas(self, dashboard: Dict) -> List[str]:
        """Identify areas for research improvement."""
        areas = []
        summary = dashboard['summary']
        
        if summary['total_hypotheses'] < 3:
            areas.append("Increase hypothesis generation")
        
        if summary['avg_confidence'] < 0.5:
            areas.append("Improve evidence quality")
        
        validation_rate = summary['validated_hypotheses'] / max(summary['total_hypotheses'], 1)
        if validation_rate < 0.1:
            areas.append("Focus on hypothesis validation")
        
        if dashboard['research_velocity']['hypotheses_per_day'] < 0.05:
            areas.append("Increase research pace")
        
        return areas
    
    def _generate_research_recommendations(self, dashboard: Dict) -> List[str]:
        """Generate high-level research recommendations."""
        recommendations = []
        
        # Based on field distribution
        fields = dashboard['field_distribution']
        if len(fields) == 1:
            recommendations.append("Consider exploring interdisciplinary research opportunities")
        
        # Based on status distribution
        status_dist = dashboard['status_distribution']
        draft_count = status_dist.get('draft', 0)
        if draft_count > 2:
            recommendations.append("Activate and test draft hypotheses")
        
        testing_count = status_dist.get('testing', 0)
        if testing_count > 3:
            recommendations.append("Focus on completing ongoing hypothesis tests")
        
        return recommendations
    
    def _identify_focus_areas(self, dashboard: Dict) -> List[str]:
        """Identify current research focus areas."""
        # Simple analysis based on field distribution and recent activity
        fields = list(dashboard['field_distribution'].keys())
        recent_activity = dashboard['recent_activity']
        
        focus_areas = fields.copy()
        
        # Add areas from recent activity
        for activity in recent_activity[:5]:  # Last 5 activities
            if 'hypothesis' in activity['description'].lower():
                focus_areas.append("Hypothesis Development")
            elif 'evidence' in activity['description'].lower():
                focus_areas.append("Evidence Collection")
        
        return list(set(focus_areas))
    
    def _suggest_collaborations(self, dashboard: Dict) -> List[str]:
        """Suggest collaboration opportunities based on research patterns."""
        suggestions = []
        
        # Based on field diversity
        fields = dashboard['field_distribution']
        if len(fields) > 1:
            suggestions.append("Cross-field collaboration opportunities available")
        
        # Based on validation challenges
        summary = dashboard['summary']
        validation_rate = summary['validated_hypotheses'] / max(summary['total_hypotheses'], 1)
        if validation_rate < 0.2:
            suggestions.append("Consider collaboration for hypothesis validation")
        
        # Based on evidence diversity needs
        suggestions.append("Seek collaborators for evidence diversification")
        
        return suggestions