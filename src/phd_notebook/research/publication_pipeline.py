"""
Automated Publication Pipeline - From research to submission.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import json
import re
from dataclasses import dataclass, field
from enum import Enum

from ..ai.client_factory import AIClientFactory
from ..core.note import Note, NoteType
from ..utils.exceptions import ResearchError


class PublicationStage(Enum):
    """Publication pipeline stages."""
    
    IDEATION = "ideation"
    DRAFT = "draft"
    REVISION = "revision"
    REVIEW = "review"
    SUBMISSION = "submission"
    UNDER_REVIEW = "under_review"
    ACCEPTED = "accepted"
    PUBLISHED = "published"
    REJECTED = "rejected"


class VenueType(Enum):
    """Types of publication venues."""
    
    JOURNAL = "journal"
    CONFERENCE = "conference"
    WORKSHOP = "workshop"
    PREPRINT = "preprint"
    BOOK_CHAPTER = "book_chapter"
    THESIS = "thesis"


@dataclass
class PublicationVenue:
    """Publication venue information."""
    
    name: str
    venue_type: VenueType
    impact_factor: Optional[float] = None
    acceptance_rate: Optional[float] = None
    submission_deadline: Optional[datetime] = None
    review_timeline: Optional[int] = None  # days
    requirements: Dict[str, Any] = field(default_factory=dict)
    formatting_guidelines: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Publication:
    """Publication tracking and management."""
    
    id: str
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str]
    venue: PublicationVenue
    stage: PublicationStage
    content: str = ""
    figures: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    submission_date: Optional[datetime] = None
    acceptance_date: Optional[datetime] = None
    revision_history: List[Dict[str, Any]] = field(default_factory=list)
    reviewer_feedback: List[str] = field(default_factory=list)
    collaboration_notes: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class PublicationPipeline:
    """Automated publication pipeline with AI assistance."""
    
    def __init__(self, notebook=None, ai_provider="auto"):
        self.notebook = notebook
        self.ai_client = AIClientFactory.get_client(ai_provider)
        self.publications: Dict[str, Publication] = {}
        self.venues: Dict[str, PublicationVenue] = {}
        
        # Initialize common venues
        self._initialize_common_venues()
    
    def _initialize_common_venues(self):
        """Initialize database of common publication venues."""
        
        common_venues = [
            # Computer Science Conferences
            {
                "name": "NeurIPS",
                "type": VenueType.CONFERENCE,
                "acceptance_rate": 0.20,
                "submission_deadline": None,  # Variable each year
                "review_timeline": 90,
                "requirements": {
                    "page_limit": 9,
                    "anonymization": True,
                    "format": "LaTeX",
                    "supplementary_allowed": True
                }
            },
            {
                "name": "ICML", 
                "type": VenueType.CONFERENCE,
                "acceptance_rate": 0.22,
                "review_timeline": 90,
                "requirements": {
                    "page_limit": 8,
                    "anonymization": True,
                    "format": "LaTeX"
                }
            },
            # Journals
            {
                "name": "Nature",
                "type": VenueType.JOURNAL,
                "impact_factor": 49.96,
                "acceptance_rate": 0.08,
                "review_timeline": 120,
                "requirements": {
                    "word_limit": 4000,
                    "figure_limit": 6,
                    "significance": "high"
                }
            },
            {
                "name": "Science",
                "type": VenueType.JOURNAL,
                "impact_factor": 47.73,
                "acceptance_rate": 0.07,
                "review_timeline": 100
            },
            # Preprint servers
            {
                "name": "arXiv",
                "type": VenueType.PREPRINT,
                "acceptance_rate": 1.0,  # No rejection
                "review_timeline": 1,
                "requirements": {
                    "format": "LaTeX or PDF",
                    "subject_classification": True
                }
            }
        ]
        
        for venue_data in common_venues:
            venue = PublicationVenue(
                name=venue_data["name"],
                venue_type=venue_data["type"],
                impact_factor=venue_data.get("impact_factor"),
                acceptance_rate=venue_data.get("acceptance_rate"),
                review_timeline=venue_data.get("review_timeline"),
                requirements=venue_data.get("requirements", {})
            )
            self.venues[venue.name] = venue
    
    async def create_publication_from_research(
        self,
        research_notes: List[Note],
        target_venue: str,
        title: str = "",
        authors: List[str] = None
    ) -> Publication:
        """Create publication from research notes using SPARC methodology."""
        
        if target_venue not in self.venues:
            raise ResearchError(f"Venue {target_venue} not found")
        
        venue = self.venues[target_venue]
        
        # Generate title if not provided
        if not title:
            title = await self._generate_title(research_notes, venue)
        
        # Generate abstract using SPARC
        abstract = await self._generate_sparc_abstract(research_notes, venue)
        
        # Extract keywords
        keywords = await self._extract_keywords(research_notes, venue)
        
        # Generate main content using SPARC framework
        content = await self._generate_sparc_content(research_notes, venue)
        
        # Create publication
        publication = Publication(
            id=f"pub_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=title,
            authors=authors or ["Author Name"],
            abstract=abstract,
            keywords=keywords,
            venue=venue,
            stage=PublicationStage.DRAFT,
            content=content
        )
        
        # Store publication
        self.publications[publication.id] = publication
        
        # Create publication note
        if self.notebook:
            self._create_publication_note(publication)
        
        return publication
    
    async def _generate_title(self, research_notes: List[Note], venue: PublicationVenue) -> str:
        """Generate compelling publication title."""
        
        research_content = " ".join([note.content for note in research_notes[:5]])
        
        title_prompt = f"""
        Generate a compelling academic title for a {venue.venue_type.value} submission to {venue.name}.
        
        Research content summary:
        {research_content[:1500]}
        
        Requirements:
        - Specific and informative
        - Captures main contribution
        - Appropriate for {venue.name}
        - 8-15 words
        - Academic style
        
        Generate 3 title options and select the best one.
        """
        
        try:
            title_response = await self.ai_client.generate_text(
                title_prompt,
                max_tokens=200,
                temperature=0.7
            )
            
            # Extract first/best title
            lines = title_response.strip().split('\n')
            for line in lines:
                if line.strip() and not line.startswith(('1.', '2.', '3.', 'Best:', 'Selected:')):
                    title = line.strip().strip('"')
                    if len(title.split()) >= 5:  # Reasonable length
                        return title
            
            # Fallback
            return "Research Contribution Analysis"
            
        except Exception as e:
            return f"Research Analysis: {research_notes[0].title if research_notes else 'Study'}"
    
    async def _generate_sparc_abstract(self, research_notes: List[Note], venue: PublicationVenue) -> str:
        """Generate abstract using SPARC (Situation-Problem-Action-Result-Conclusion) framework."""
        
        research_content = " ".join([note.content for note in research_notes])
        
        sparc_prompt = f"""
        Write an academic abstract using the SPARC framework for {venue.name}:
        
        Research content:
        {research_content[:2000]}
        
        SPARC Structure:
        - **Situation**: Context and background (2-3 sentences)
        - **Problem**: Specific problem/research question (1-2 sentences) 
        - **Action**: Methodology/approach taken (2-3 sentences)
        - **Result**: Key findings/outcomes (2-3 sentences)
        - **Conclusion**: Implications and significance (1-2 sentences)
        
        Requirements:
        - Word limit: 250 words
        - Academic tone
        - Specific and quantitative where possible
        - No citations in abstract
        
        Write as a single paragraph combining all SPARC elements smoothly.
        """
        
        try:
            abstract = await self.ai_client.generate_text(
                sparc_prompt,
                max_tokens=400,
                temperature=0.6
            )
            
            # Clean up abstract
            abstract = abstract.strip()
            # Remove any section headers that might have been included
            abstract = re.sub(r'\*\*[^*]+\*\*:?\s*', '', abstract)
            
            return abstract
            
        except Exception as e:
            raise ResearchError(f"Abstract generation failed: {e}")
    
    async def _extract_keywords(self, research_notes: List[Note], venue: PublicationVenue) -> List[str]:
        """Extract relevant keywords from research."""
        
        research_content = " ".join([note.content for note in research_notes[:3]])
        
        keyword_prompt = f"""
        Extract 5-8 relevant academic keywords for a {venue.name} submission.
        
        Research content:
        {research_content[:1000]}
        
        Guidelines:
        - Technical terms and concepts
        - Field-specific terminology
        - Methodological approaches
        - Domain areas
        - Avoid generic terms
        
        Return as comma-separated list.
        """
        
        try:
            keywords_response = await self.ai_client.generate_text(
                keyword_prompt,
                max_tokens=100,
                temperature=0.5
            )
            
            keywords = [k.strip() for k in keywords_response.replace('\n', ',').split(',')]
            keywords = [k for k in keywords if k and len(k) > 2]
            
            return keywords[:8]  # Limit to 8 keywords
            
        except Exception as e:
            return ["research", "analysis", "methodology"]
    
    async def _generate_sparc_content(self, research_notes: List[Note], venue: PublicationVenue) -> str:
        """Generate full paper content using SPARC framework."""
        
        content_sections = []
        
        # Introduction (Situation + Problem)
        introduction = await self._generate_introduction(research_notes, venue)
        content_sections.append(f"# Introduction\n\n{introduction}")
        
        # Methodology (Action)
        methodology = await self._generate_methodology(research_notes, venue)
        content_sections.append(f"# Methodology\n\n{methodology}")
        
        # Results (Result)
        results = await self._generate_results(research_notes, venue)
        content_sections.append(f"# Results\n\n{results}")
        
        # Discussion & Conclusion (Conclusion)
        discussion = await self._generate_discussion(research_notes, venue)
        content_sections.append(f"# Discussion and Conclusion\n\n{discussion}")
        
        return "\n\n".join(content_sections)
    
    async def _generate_introduction(self, research_notes: List[Note], venue: PublicationVenue) -> str:
        """Generate introduction section (Situation + Problem)."""
        
        intro_prompt = f"""
        Write an introduction section for a {venue.name} paper covering:
        
        Research context:
        {" ".join([note.content for note in research_notes[:2]])[:1500]}
        
        Structure:
        1. **Situation**: Background and context (2-3 paragraphs)
        2. **Problem**: Specific research gap/question (1-2 paragraphs)  
        3. **Contributions**: Preview of key contributions (1 paragraph)
        
        Requirements:
        - Academic tone
        - Proper motivation
        - Clear problem statement
        - 800-1200 words
        """
        
        try:
            introduction = await self.ai_client.generate_text(
                intro_prompt,
                max_tokens=800,
                temperature=0.6
            )
            return introduction.strip()
        except Exception as e:
            return "Introduction section generation failed."
    
    async def _generate_methodology(self, research_notes: List[Note], venue: PublicationVenue) -> str:
        """Generate methodology section (Action)."""
        
        method_content = " ".join([note.content for note in research_notes if note.note_type == NoteType.EXPERIMENT])
        
        method_prompt = f"""
        Write a methodology section describing the research approach:
        
        Experimental content:
        {method_content[:1500]}
        
        Include:
        - Experimental design
        - Data collection procedures
        - Analysis methods
        - Tools and technologies
        - Validation approaches
        
        Requirements:
        - Reproducible descriptions
        - Technical accuracy
        - Appropriate detail level
        - 600-1000 words
        """
        
        try:
            methodology = await self.ai_client.generate_text(
                method_prompt,
                max_tokens=600,
                temperature=0.5
            )
            return methodology.strip()
        except Exception as e:
            return "Methodology section generation failed."
    
    async def _generate_results(self, research_notes: List[Note], venue: PublicationVenue) -> str:
        """Generate results section."""
        
        results_content = " ".join([note.content for note in research_notes 
                                   if any(keyword in note.content.lower() 
                                         for keyword in ['result', 'finding', 'data', 'analysis'])])
        
        results_prompt = f"""
        Write a results section presenting key findings:
        
        Research results:
        {results_content[:1500]}
        
        Structure:
        - Present findings objectively
        - Include quantitative results where available
        - Reference figures/tables (placeholder)
        - Statistical significance where applicable
        - 600-1000 words
        
        Style: Objective, factual reporting without interpretation.
        """
        
        try:
            results = await self.ai_client.generate_text(
                results_prompt,
                max_tokens=600,
                temperature=0.4
            )
            return results.strip()
        except Exception as e:
            return "Results section generation failed."
    
    async def _generate_discussion(self, research_notes: List[Note], venue: PublicationVenue) -> str:
        """Generate discussion and conclusion section."""
        
        all_content = " ".join([note.content for note in research_notes])
        
        discussion_prompt = f"""
        Write discussion and conclusion sections:
        
        Complete research context:
        {all_content[:2000]}
        
        Discussion should cover:
        - Interpretation of results
        - Comparison with related work
        - Limitations
        - Implications
        
        Conclusion should cover:
        - Summary of contributions
        - Future work directions
        - Broader impact
        
        Requirements:
        - Critical analysis
        - Balanced perspective
        - Clear implications
        - 800-1200 words
        """
        
        try:
            discussion = await self.ai_client.generate_text(
                discussion_prompt,
                max_tokens=800,
                temperature=0.6
            )
            return discussion.strip()
        except Exception as e:
            return "Discussion section generation failed."
    
    def _create_publication_note(self, publication: Publication):
        """Create a note for publication tracking."""
        
        note_content = f"""
# Publication: {publication.title}

## Details
- **Publication ID**: {publication.id}
- **Venue**: {publication.venue.name} ({publication.venue.venue_type.value})
- **Stage**: {publication.stage.value}
- **Authors**: {', '.join(publication.authors)}
- **Created**: {publication.created_at.strftime('%Y-%m-%d')}

## Abstract
{publication.abstract}

## Keywords
{', '.join(publication.keywords)}

## Progress Tracking
- [ ] Draft completion
- [ ] Internal review
- [ ] Revision based on feedback
- [ ] Final proofreading
- [ ] Submission preparation
- [ ] Venue submission
- [ ] Review process
- [ ] Final publication

## Venue Requirements
{json.dumps(publication.venue.requirements, indent=2)}

## Content
{publication.content[:500]}...

---
*Managed by PublicationPipeline*
        """
        
        pub_note = self.notebook.create_note(
            title=f"Publication - {publication.title}",
            content=note_content,
            note_type=NoteType.PROJECT,
            tags=["#publication", f"#{publication.venue.name.lower()}", f"#{publication.stage.value}"]
        )
        
        return pub_note
    
    async def review_and_improve(self, publication_id: str, review_type: str = "comprehensive") -> Dict[str, Any]:
        """AI-powered review and improvement suggestions."""
        
        if publication_id not in self.publications:
            raise ResearchError(f"Publication {publication_id} not found")
        
        publication = self.publications[publication_id]
        
        review_prompt = f"""
        Perform a {review_type} review of this academic paper for {publication.venue.name}:
        
        **Title**: {publication.title}
        **Abstract**: {publication.abstract}
        **Content**: {publication.content[:2000]}...
        
        Evaluate:
        1. **Clarity**: Is the writing clear and well-structured?
        2. **Contribution**: Are the contributions significant and novel?
        3. **Methodology**: Is the methodology sound and appropriate?
        4. **Evidence**: Are claims well-supported by evidence?
        5. **Presentation**: Is the paper well-organized and professionally written?
        6. **Venue Fit**: Does it match {publication.venue.name} standards?
        
        Provide:
        - Overall assessment (1-10 scale)
        - Specific improvement suggestions
        - Priority areas for revision
        - Venue-specific recommendations
        """
        
        try:
            review_result = await self.ai_client.generate_text(
                review_prompt,
                max_tokens=1000,
                temperature=0.4
            )
            
            # Store review in publication history
            review_entry = {
                "date": datetime.now().isoformat(),
                "type": review_type,
                "reviewer": "AI Assistant",
                "feedback": review_result
            }
            publication.revision_history.append(review_entry)
            
            return {
                "publication_id": publication_id,
                "review_type": review_type,
                "review": review_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise ResearchError(f"Review generation failed: {e}")
    
    async def suggest_target_venues(self, publication_id: str, max_suggestions: int = 5) -> List[Dict[str, Any]]:
        """Suggest appropriate venues for publication."""
        
        if publication_id not in self.publications:
            raise ResearchError(f"Publication {publication_id} not found")
        
        publication = self.publications[publication_id]
        
        venue_prompt = f"""
        Suggest {max_suggestions} appropriate publication venues for this paper:
        
        **Title**: {publication.title}
        **Abstract**: {publication.abstract}
        **Keywords**: {', '.join(publication.keywords)}
        **Content Preview**: {publication.content[:1000]}
        
        For each venue, consider:
        - Research area alignment
        - Quality/prestige level
        - Acceptance rate
        - Timeline requirements
        - Paper novelty/significance
        
        Suggest venues from different tiers (top-tier, mid-tier, specialized).
        Include both journals and conferences where appropriate.
        """
        
        try:
            venue_suggestions = await self.ai_client.generate_text(
                venue_prompt,
                max_tokens=600,
                temperature=0.6
            )
            
            # Parse suggestions (simplified)
            suggestions = []
            lines = venue_suggestions.split('\n')
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ['venue', 'journal', 'conference']):
                    # Extract venue name and rationale
                    suggestion = {
                        "venue": line.strip(),
                        "rationale": "AI suggested based on content alignment",
                        "confidence": 0.8
                    }
                    suggestions.append(suggestion)
            
            return suggestions[:max_suggestions]
            
        except Exception as e:
            return [{"error": f"Venue suggestion failed: {e}"}]
    
    async def prepare_submission(self, publication_id: str, target_venue: str = None) -> Dict[str, Any]:
        """Prepare publication for submission to venue."""
        
        if publication_id not in self.publications:
            raise ResearchError(f"Publication {publication_id} not found")
        
        publication = self.publications[publication_id]
        
        if target_venue and target_venue in self.venues:
            publication.venue = self.venues[target_venue]
        
        # Check venue requirements
        requirements = publication.venue.requirements
        submission_ready = True
        issues = []
        
        # Check word limit
        word_count = len(publication.content.split()) + len(publication.abstract.split())
        if "word_limit" in requirements and word_count > requirements["word_limit"]:
            submission_ready = False
            issues.append(f"Word count ({word_count}) exceeds limit ({requirements['word_limit']})")
        
        # Check anonymization
        if requirements.get("anonymization") and any(author.lower() in publication.content.lower() 
                                                    for author in publication.authors):
            submission_ready = False
            issues.append("Anonymization required - author names found in content")
        
        # Update publication stage
        if submission_ready:
            publication.stage = PublicationStage.SUBMISSION
        
        return {
            "publication_id": publication_id,
            "venue": publication.venue.name,
            "submission_ready": submission_ready,
            "issues": issues,
            "word_count": word_count,
            "requirements": requirements,
            "checklist": self._generate_submission_checklist(publication)
        }
    
    def _generate_submission_checklist(self, publication: Publication) -> List[str]:
        """Generate submission checklist for venue."""
        
        checklist = [
            "Review abstract for accuracy and completeness",
            "Check all figures and tables are referenced",
            "Verify references are properly formatted",
            "Proofread for grammar and style",
            "Confirm authorship and affiliations",
            "Check compliance with venue requirements"
        ]
        
        # Add venue-specific items
        if publication.venue.requirements.get("anonymization"):
            checklist.append("Remove all identifying information")
        
        if "supplementary_allowed" in publication.venue.requirements:
            checklist.append("Prepare supplementary materials if needed")
        
        return checklist
    
    def get_publication(self, publication_id: str) -> Optional[Publication]:
        """Get publication by ID."""
        return self.publications.get(publication_id)
    
    def list_publications(self, stage: PublicationStage = None) -> List[Publication]:
        """List publications, optionally filtered by stage."""
        publications = list(self.publications.values())
        
        if stage:
            publications = [p for p in publications if p.stage == stage]
        
        return sorted(publications, key=lambda p: p.created_at, reverse=True)
    
    def add_venue(self, venue: PublicationVenue):
        """Add custom publication venue."""
        self.venues[venue.name] = venue
    
    def get_venue(self, name: str) -> Optional[PublicationVenue]:
        """Get venue by name."""
        return self.venues.get(name)
    
    def list_venues(self, venue_type: VenueType = None) -> List[PublicationVenue]:
        """List venues, optionally filtered by type."""
        venues = list(self.venues.values())
        
        if venue_type:
            venues = [v for v in venues if v.venue_type == venue_type]
        
        return sorted(venues, key=lambda v: v.name)
    
    async def track_submission_status(self, publication_id: str, status_update: str):
        """Update publication submission status."""
        
        if publication_id not in self.publications:
            raise ResearchError(f"Publication {publication_id} not found")
        
        publication = self.publications[publication_id]
        
        # Update status based on keywords
        if "accepted" in status_update.lower():
            publication.stage = PublicationStage.ACCEPTED
            publication.acceptance_date = datetime.now()
        elif "rejected" in status_update.lower():
            publication.stage = PublicationStage.REJECTED
        elif "revision" in status_update.lower():
            publication.stage = PublicationStage.REVISION
        elif "under review" in status_update.lower():
            publication.stage = PublicationStage.UNDER_REVIEW
        
        # Add to collaboration notes
        publication.collaboration_notes.append(
            f"{datetime.now().strftime('%Y-%m-%d')}: {status_update}"
        )
        
        # Create update note if notebook available
        if self.notebook:
            update_note = self.notebook.create_note(
                title=f"Publication Update - {publication.title}",
                content=f"Status Update: {status_update}\n\nCurrent Stage: {publication.stage.value}",
                note_type=NoteType.PROJECT,
                tags=["#publication_update", f"#{publication.stage.value}"]
            )
    
    def export_publication_summary(self) -> str:
        """Export summary of all publications."""
        
        summary = "# Publication Pipeline Summary\n\n"
        summary += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"Total Publications: {len(self.publications)}\n\n"
        
        # Stage overview
        stage_counts = {}
        for pub in self.publications.values():
            stage_counts[pub.stage.value] = stage_counts.get(pub.stage.value, 0) + 1
        
        summary += "## Publication Stages\n"
        for stage, count in stage_counts.items():
            summary += f"- {stage.title()}: {count}\n"
        summary += "\n"
        
        # Publication details
        summary += "## Publications\n\n"
        for pub in sorted(self.publications.values(), key=lambda p: p.created_at, reverse=True):
            summary += f"### {pub.title}\n"
            summary += f"- **Venue**: {pub.venue.name}\n"
            summary += f"- **Stage**: {pub.stage.value}\n"
            summary += f"- **Authors**: {', '.join(pub.authors)}\n"
            summary += f"- **Created**: {pub.created_at.strftime('%Y-%m-%d')}\n\n"
        
        return summary