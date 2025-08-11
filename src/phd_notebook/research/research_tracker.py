"""
Research Progress Tracking and Analytics Engine.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import dataclass, field
from enum import Enum
import statistics

from ..ai.client_factory import AIClientFactory
from ..core.note import Note, NoteType
from ..utils.exceptions import ResearchError


class MilestoneType(Enum):
    """Types of research milestones."""
    
    LITERATURE_REVIEW = "literature_review"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    EXPERIMENT_DESIGN = "experiment_design"
    DATA_COLLECTION = "data_collection"
    ANALYSIS = "analysis"
    WRITING = "writing"
    PRESENTATION = "presentation"
    PUBLICATION = "publication"
    DEFENSE = "defense"


@dataclass
class ResearchMilestone:
    """Research milestone with tracking."""
    
    id: str
    title: str
    milestone_type: MilestoneType
    description: str
    target_date: datetime
    completion_date: Optional[datetime] = None
    status: str = "pending"  # pending, in_progress, completed, overdue
    progress_percentage: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ResearchMetrics:
    """Research progress metrics."""
    
    total_notes: int
    notes_by_type: Dict[str, int]
    experiment_count: int
    papers_read: int
    hypotheses_generated: int
    writing_velocity: float  # words per day
    collaboration_score: float
    productivity_trend: str  # improving, stable, declining
    knowledge_graph_size: int
    research_gaps: List[str]
    completion_percentage: float


class ResearchTracker:
    """Comprehensive research progress tracking and analytics."""
    
    def __init__(self, notebook=None, ai_provider="auto"):
        self.notebook = notebook
        self.ai_client = AIClientFactory.get_client(ai_provider)
        self.milestones: Dict[str, ResearchMilestone] = {}
        self.metrics_history: List[Tuple[datetime, ResearchMetrics]] = []
        
        # PhD timeline templates
        self.phd_timeline_templates = {
            "computer_science": {
                "year_1": ["coursework", "literature_review", "advisor_selection"],
                "year_2": ["qualifying_exam", "research_proposal", "preliminary_research"],
                "year_3": ["main_research", "conference_presentations", "journal_submissions"],
                "year_4": ["thesis_writing", "job_search", "dissertation_defense"],
                "year_5": ["final_revisions", "graduation", "career_transition"]
            },
            "biology": {
                "year_1": ["coursework", "lab_rotations", "literature_review"],
                "year_2": ["qualifying_exam", "thesis_committee", "research_proposal"],
                "year_3": ["experimental_design", "data_collection", "preliminary_analysis"],
                "year_4": ["continued_research", "conference_presentations", "manuscript_prep"],
                "year_5": ["thesis_writing", "dissertation_defense", "postdoc_search"]
            }
        }
    
    async def initialize_phd_timeline(
        self,
        field: str = "computer_science",
        start_date: datetime = None,
        expected_duration: int = 5
    ) -> List[ResearchMilestone]:
        """Initialize a PhD timeline with field-specific milestones."""
        
        start_date = start_date or datetime.now()
        template = self.phd_timeline_templates.get(field.lower(), self.phd_timeline_templates["computer_science"])
        
        milestones = []
        current_date = start_date
        
        for year, year_milestones in template.items():
            year_num = int(year.split('_')[1])
            
            for i, milestone_title in enumerate(year_milestones):
                milestone_date = current_date + timedelta(days=(year_num - 1) * 365 + (i * 122))  # ~4 months apart
                
                milestone_type = self._infer_milestone_type(milestone_title)
                
                milestone = ResearchMilestone(
                    id=f"phd_{year}_{i+1:02d}",
                    title=milestone_title.replace('_', ' ').title(),
                    milestone_type=milestone_type,
                    description=f"{milestone_title.replace('_', ' ').title()} for {field} PhD",
                    target_date=milestone_date,
                )
                
                self.milestones[milestone.id] = milestone
                milestones.append(milestone)
        
        # Create overview note
        if self.notebook:
            self._create_timeline_note(milestones, field, start_date, expected_duration)
        
        return milestones
    
    def _infer_milestone_type(self, milestone_title: str) -> MilestoneType:
        """Infer milestone type from title."""
        
        title_lower = milestone_title.lower()
        
        if any(word in title_lower for word in ['literature', 'review', 'reading']):
            return MilestoneType.LITERATURE_REVIEW
        elif any(word in title_lower for word in ['experiment', 'data', 'collection']):
            return MilestoneType.DATA_COLLECTION
        elif any(word in title_lower for word in ['analysis', 'results']):
            return MilestoneType.ANALYSIS
        elif any(word in title_lower for word in ['writing', 'thesis', 'dissertation']):
            return MilestoneType.WRITING
        elif any(word in title_lower for word in ['defense', 'presentation']):
            return MilestoneType.DEFENSE
        elif any(word in title_lower for word in ['publication', 'manuscript', 'journal']):
            return MilestoneType.PUBLICATION
        else:
            return MilestoneType.HYPOTHESIS_GENERATION
    
    def _create_timeline_note(
        self, 
        milestones: List[ResearchMilestone], 
        field: str, 
        start_date: datetime,
        duration: int
    ):
        """Create a timeline overview note."""
        
        timeline_content = f"""
# PhD Timeline - {field.title()}

## Overview
- **Start Date**: {start_date.strftime('%Y-%m-%d')}
- **Expected Duration**: {duration} years
- **Total Milestones**: {len(milestones)}

## Timeline

{self._generate_timeline_visualization(milestones)}

## Milestones by Year

{self._generate_milestones_by_year(milestones)}

## Critical Path
{self._identify_critical_path(milestones)}

---
*Generated by ResearchTracker on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
        """
        
        timeline_note = self.notebook.create_note(
            title=f"PhD Timeline - {field.title()}",
            content=timeline_content,
            note_type=NoteType.PROJECT,
            tags=["#phd_timeline", "#milestones", f"#{field}"]
        )
        
        return timeline_note
    
    def _generate_timeline_visualization(self, milestones: List[ResearchMilestone]) -> str:
        """Generate ASCII timeline visualization."""
        
        vis = "```\n"
        vis += "Year 1: |" + "==" * 12 + "|\n"
        vis += "Year 2: |" + "==" * 12 + "|\n"
        vis += "Year 3: |" + "==" * 12 + "|\n"
        vis += "Year 4: |" + "==" * 12 + "|\n"
        vis += "Year 5: |" + "==" * 12 + "|\n"
        vis += "```"
        
        return vis
    
    def _generate_milestones_by_year(self, milestones: List[ResearchMilestone]) -> str:
        """Generate milestones grouped by year."""
        
        milestones_by_year = {}
        for milestone in sorted(milestones, key=lambda m: m.target_date):
            year = f"Year {milestone.target_date.year - milestones[0].target_date.year + 1}"
            if year not in milestones_by_year:
                milestones_by_year[year] = []
            milestones_by_year[year].append(milestone)
        
        content = ""
        for year, year_milestones in milestones_by_year.items():
            content += f"\n### {year}\n"
            for milestone in year_milestones:
                status_icon = "✅" if milestone.status == "completed" else "⏳"
                content += f"- {status_icon} **{milestone.title}** ({milestone.target_date.strftime('%b %Y')})\n"
        
        return content
    
    def _identify_critical_path(self, milestones: List[ResearchMilestone]) -> str:
        """Identify critical path milestones."""
        
        critical = [
            "Literature Review",
            "Research Proposal", 
            "Main Research",
            "Thesis Writing",
            "Dissertation Defense"
        ]
        
        critical_milestones = [m for m in milestones if any(c in m.title for c in critical)]
        
        content = ""
        for milestone in sorted(critical_milestones, key=lambda m: m.target_date):
            content += f"- **{milestone.title}** ({milestone.target_date.strftime('%b %Y')})\n"
        
        return content
    
    async def update_milestone_progress(
        self,
        milestone_id: str,
        progress_percentage: float,
        notes: str = ""
    ) -> ResearchMilestone:
        """Update milestone progress."""
        
        if milestone_id not in self.milestones:
            raise ResearchError(f"Milestone {milestone_id} not found")
        
        milestone = self.milestones[milestone_id]
        milestone.progress_percentage = max(0.0, min(100.0, progress_percentage))
        
        if notes:
            milestone.notes.append(f"{datetime.now().strftime('%Y-%m-%d')}: {notes}")
        
        # Update status based on progress
        if progress_percentage >= 100.0:
            milestone.status = "completed"
            milestone.completion_date = datetime.now()
        elif progress_percentage > 0:
            milestone.status = "in_progress"
        
        # Check if overdue
        if datetime.now() > milestone.target_date and milestone.status != "completed":
            milestone.status = "overdue"
        
        return milestone
    
    async def calculate_research_metrics(self) -> ResearchMetrics:
        """Calculate comprehensive research metrics."""
        
        if not self.notebook:
            raise ResearchError("Notebook required for metrics calculation")
        
        notes = self.notebook.list_notes()
        
        # Basic counts
        total_notes = len(notes)
        notes_by_type = {}
        for note in notes:
            note_type = note.note_type.value if hasattr(note.note_type, 'value') else str(note.note_type)
            notes_by_type[note_type] = notes_by_type.get(note_type, 0) + 1
        
        experiment_count = notes_by_type.get('experiment', 0)
        papers_read = notes_by_type.get('literature', 0)
        
        # Calculate writing velocity (words per day over last 30 days)
        writing_velocity = await self._calculate_writing_velocity(notes)
        
        # Calculate collaboration score based on notes with multiple contributors
        collaboration_score = await self._calculate_collaboration_score(notes)
        
        # Analyze productivity trend
        productivity_trend = await self._analyze_productivity_trend()
        
        # Count hypotheses
        hypotheses_generated = len([n for n in notes if '#hypothesis' in n.frontmatter.tags])
        
        # Calculate completion percentage based on milestones
        completion_percentage = self._calculate_completion_percentage()
        
        # Identify research gaps
        research_gaps = await self._identify_research_gaps(notes)
        
        metrics = ResearchMetrics(
            total_notes=total_notes,
            notes_by_type=notes_by_type,
            experiment_count=experiment_count,
            papers_read=papers_read,
            hypotheses_generated=hypotheses_generated,
            writing_velocity=writing_velocity,
            collaboration_score=collaboration_score,
            productivity_trend=productivity_trend,
            knowledge_graph_size=len(notes),  # Simplified
            research_gaps=research_gaps,
            completion_percentage=completion_percentage
        )
        
        # Store metrics in history
        self.metrics_history.append((datetime.now(), metrics))
        
        return metrics
    
    async def _calculate_writing_velocity(self, notes: List[Note]) -> float:
        """Calculate average words per day over recent period."""
        
        recent_date = datetime.now() - timedelta(days=30)
        recent_notes = [n for n in notes if hasattr(n.frontmatter, 'created') and 
                      n.frontmatter.created > recent_date]
        
        if not recent_notes:
            return 0.0
        
        total_words = sum(len(note.content.split()) for note in recent_notes)
        return total_words / 30.0  # Average per day
    
    async def _calculate_collaboration_score(self, notes: List[Note]) -> float:
        """Calculate collaboration score (0-10)."""
        
        if not notes:
            return 0.0
        
        collaboration_indicators = 0
        for note in notes:
            # Look for collaboration indicators
            if any(indicator in note.content.lower() for indicator in ['meeting', 'discussion', 'feedback', 'review']):
                collaboration_indicators += 1
            if any(tag in note.frontmatter.tags for tag in ['#meeting', '#collaboration', '#discussion']):
                collaboration_indicators += 1
        
        return min(10.0, (collaboration_indicators / len(notes)) * 100)
    
    async def _analyze_productivity_trend(self) -> str:
        """Analyze productivity trend over time."""
        
        if len(self.metrics_history) < 3:
            return "stable"
        
        recent_metrics = [m for _, m in self.metrics_history[-3:]]
        note_counts = [m.total_notes for m in recent_metrics]
        
        if len(note_counts) >= 2:
            if note_counts[-1] > note_counts[-2]:
                return "improving"
            elif note_counts[-1] < note_counts[-2]:
                return "declining"
        
        return "stable"
    
    def _calculate_completion_percentage(self) -> float:
        """Calculate overall PhD completion percentage."""
        
        if not self.milestones:
            return 0.0
        
        total_progress = sum(milestone.progress_percentage for milestone in self.milestones.values())
        return total_progress / len(self.milestones)
    
    async def _identify_research_gaps(self, notes: List[Note]) -> List[str]:
        """Identify potential research gaps."""
        
        # Analyze note content for gaps
        all_content = " ".join([note.content for note in notes[:10]])  # Sample for efficiency
        
        gap_analysis_prompt = f"""
        Analyze this research content and identify 3-5 potential research gaps:
        
        Content: {all_content[:2000]}...
        
        Look for:
        1. Mentioned limitations
        2. Areas marked as "future work"
        3. Contradictory findings
        4. Unexplored connections
        5. Missing methodologies
        
        Return a list of specific research gaps.
        """
        
        try:
            gaps_response = await self.ai_client.generate_text(
                gap_analysis_prompt,
                max_tokens=300,
                temperature=0.6
            )
            
            gaps = []
            for line in gaps_response.split('\n'):
                if line.strip() and (line.startswith('-') or line.startswith('*') or line[0].isdigit()):
                    gap = line.strip().lstrip('-*0123456789. ')
                    if gap:
                        gaps.append(gap)
            
            return gaps[:5]
            
        except Exception as e:
            return [f"Gap analysis failed: {e}"]
    
    async def generate_progress_report(self, period_days: int = 30) -> str:
        """Generate a comprehensive progress report."""
        
        metrics = await self.calculate_research_metrics()
        recent_milestones = [m for m in self.milestones.values() 
                           if (datetime.now() - m.created_at).days <= period_days]
        
        report = f"""
# Research Progress Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Reporting Period: Last {period_days} days

## Key Metrics
- **Total Notes**: {metrics.total_notes}
- **Experiments Conducted**: {metrics.experiment_count}
- **Papers Read**: {metrics.papers_read}
- **Hypotheses Generated**: {metrics.hypotheses_generated}
- **Writing Velocity**: {metrics.writing_velocity:.1f} words/day
- **Collaboration Score**: {metrics.collaboration_score:.1f}/10
- **Overall Completion**: {metrics.completion_percentage:.1f}%

## Notes by Type
{chr(10).join([f"- {note_type.title()}: {count}" for note_type, count in metrics.notes_by_type.items()])}

## Productivity Trend
📈 **{metrics.productivity_trend.title()}**

## Recent Milestones
{self._format_recent_milestones(recent_milestones)}

## Research Gaps Identified
{chr(10).join([f"- {gap}" for gap in metrics.research_gaps])}

## Recommendations
{await self._generate_recommendations(metrics)}

---
*Generated by ResearchTracker*
        """
        
        # Create progress report note
        if self.notebook:
            report_note = self.notebook.create_note(
                title=f"Progress Report - {datetime.now().strftime('%Y-%m-%d')}",
                content=report,
                note_type=NoteType.PROJECT,
                tags=["#progress_report", "#analytics", "#phd_tracking"]
            )
        
        return report
    
    def _format_recent_milestones(self, milestones: List[ResearchMilestone]) -> str:
        """Format recent milestones for report."""
        
        if not milestones:
            return "No recent milestones"
        
        formatted = ""
        for milestone in sorted(milestones, key=lambda m: m.target_date):
            status_icon = {
                "completed": "✅",
                "in_progress": "🔄", 
                "pending": "⏳",
                "overdue": "⚠️"
            }.get(milestone.status, "❓")
            
            formatted += f"- {status_icon} **{milestone.title}** ({milestone.progress_percentage:.0f}%)\n"
        
        return formatted
    
    async def _generate_recommendations(self, metrics: ResearchMetrics) -> str:
        """Generate personalized recommendations."""
        
        recommendations = []
        
        if metrics.writing_velocity < 100:
            recommendations.append("📝 Consider increasing daily writing goals - current velocity is below target")
        
        if metrics.collaboration_score < 5:
            recommendations.append("🤝 Increase collaboration activities - schedule more meetings or discussions")
        
        if metrics.experiment_count < metrics.papers_read / 3:
            recommendations.append("🧪 Balance reading with experimentation - consider more hands-on research")
        
        if len(metrics.research_gaps) > 3:
            recommendations.append("🔍 Focus research direction - multiple gaps identified suggest broad scope")
        
        if not recommendations:
            recommendations.append("👍 Research progress looks balanced - maintain current momentum")
        
        return chr(10).join(recommendations)
    
    async def predict_completion_timeline(self, confidence_level: float = 0.8) -> Dict[str, Any]:
        """Predict PhD completion timeline based on current progress."""
        
        metrics = await self.calculate_research_metrics()
        
        # Calculate completion rate based on milestones
        completed_milestones = [m for m in self.milestones.values() if m.status == "completed"]
        total_milestones = len(self.milestones)
        
        if not total_milestones:
            return {"error": "No milestones defined for prediction"}
        
        completion_rate = len(completed_milestones) / total_milestones
        remaining_progress = 1.0 - completion_rate
        
        # Estimate based on current velocity
        if completion_rate > 0:
            time_per_milestone = sum((m.completion_date - m.created_at).days 
                                   for m in completed_milestones if m.completion_date) / len(completed_milestones)
        else:
            time_per_milestone = 90  # Default 3 months per milestone
        
        remaining_milestones = total_milestones - len(completed_milestones)
        estimated_days = remaining_milestones * time_per_milestone
        
        # Apply confidence interval
        confidence_multiplier = 1.0 + (1.0 - confidence_level)
        estimated_days_with_confidence = estimated_days * confidence_multiplier
        
        completion_date = datetime.now() + timedelta(days=estimated_days_with_confidence)
        
        return {
            "estimated_completion_date": completion_date.strftime('%Y-%m-%d'),
            "estimated_days_remaining": int(estimated_days_with_confidence),
            "confidence_level": confidence_level,
            "completion_percentage": completion_rate * 100,
            "remaining_milestones": remaining_milestones,
            "current_velocity": f"{time_per_milestone:.1f} days per milestone",
            "factors": {
                "writing_velocity": metrics.writing_velocity,
                "productivity_trend": metrics.productivity_trend,
                "collaboration_score": metrics.collaboration_score
            }
        }
    
    def get_milestone(self, milestone_id: str) -> Optional[ResearchMilestone]:
        """Get milestone by ID."""
        return self.milestones.get(milestone_id)
    
    def list_milestones(self, status: str = None) -> List[ResearchMilestone]:
        """List milestones, optionally filtered by status."""
        milestones = list(self.milestones.values())
        
        if status:
            milestones = [m for m in milestones if m.status == status]
        
        return sorted(milestones, key=lambda m: m.target_date)
    
    def export_analytics_dashboard(self) -> Dict[str, Any]:
        """Export data for analytics dashboard."""
        
        return {
            "milestones": [
                {
                    "id": m.id,
                    "title": m.title,
                    "type": m.milestone_type.value,
                    "progress": m.progress_percentage,
                    "status": m.status,
                    "target_date": m.target_date.isoformat(),
                    "completion_date": m.completion_date.isoformat() if m.completion_date else None
                }
                for m in self.milestones.values()
            ],
            "metrics_history": [
                {
                    "date": timestamp.isoformat(),
                    "total_notes": metrics.total_notes,
                    "experiments": metrics.experiment_count,
                    "writing_velocity": metrics.writing_velocity,
                    "collaboration_score": metrics.collaboration_score
                }
                for timestamp, metrics in self.metrics_history
            ],
            "generated_at": datetime.now().isoformat()
        }