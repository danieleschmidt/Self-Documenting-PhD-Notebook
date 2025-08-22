"""
Intelligent Research Paper Generation Engine
Automates academic paper creation from research notes using advanced AI and SPARC methodology.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, Counter

from ..core.note import Note, NoteType
from ..agents.sparc_agent import SPARCAgent
from ..utils.logging import setup_logger


class PaperSection(Enum):
    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    LITERATURE_REVIEW = "literature_review"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    APPENDIX = "appendix"


class PaperStatus(Enum):
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    REVISING = "revising"
    READY_FOR_SUBMISSION = "ready_for_submission"
    SUBMITTED = "submitted"
    PUBLISHED = "published"
    REJECTED = "rejected"


class VenueType(Enum):
    JOURNAL = "journal"
    CONFERENCE = "conference"
    WORKSHOP = "workshop"
    PREPRINT = "preprint"
    THESIS_CHAPTER = "thesis_chapter"


@dataclass
class PaperTemplate:
    """Template configuration for different paper types."""
    name: str
    venue_type: VenueType
    sections: List[PaperSection]
    word_limits: Dict[PaperSection, int]
    style_requirements: Dict[str, Any]
    citation_style: str = "apa"
    
    # Template-specific requirements
    required_elements: List[str] = None
    optional_elements: List[str] = None
    structural_requirements: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.required_elements is None:
            self.required_elements = []
        if self.optional_elements is None:
            self.optional_elements = []
        if self.structural_requirements is None:
            self.structural_requirements = {}


@dataclass
class ResearchContent:
    """Structured research content for paper generation."""
    content_id: str
    title: str
    content_type: str  # "experiment", "literature_review", "analysis", etc.
    raw_content: str
    processed_content: str
    key_insights: List[str]
    
    # Metadata
    source_files: List[str]
    date_created: datetime
    confidence_score: float
    quality_score: float
    
    # Research context
    research_questions: List[str]
    hypotheses: List[str]
    methods_used: List[str]
    results_summary: str
    
    # Paper integration
    suggested_sections: List[PaperSection]
    citation_count: int = 0
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['date_created'] = self.date_created.isoformat()
        return result


@dataclass
class GeneratedPaper:
    """A complete generated research paper."""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    
    # Paper sections
    sections: Dict[PaperSection, str]
    
    # Generation metadata
    template_used: str
    source_content_ids: List[str]
    generation_date: datetime
    status: PaperStatus
    
    # Quality metrics
    coherence_score: float
    novelty_score: float
    completeness_score: float
    citation_quality_score: float
    
    # Target venue
    target_venue: Optional[str] = None
    venue_type: Optional[VenueType] = None
    
    # References and citations
    references: List[Dict[str, str]] = None
    citation_network: Dict[str, List[str]] = None
    
    # Review and revision tracking
    reviews: List[Dict[str, Any]] = None
    revisions: List[Dict[str, Any]] = None
    
    # Export formats
    latex_version: Optional[str] = None
    word_version: Optional[str] = None
    html_version: Optional[str] = None
    
    def __post_init__(self):
        if self.references is None:
            self.references = []
        if self.citation_network is None:
            self.citation_network = {}
        if self.reviews is None:
            self.reviews = []
        if self.revisions is None:
            self.revisions = []
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['generation_date'] = self.generation_date.isoformat()
        result['status'] = self.status.value
        result['venue_type'] = self.venue_type.value if self.venue_type else None
        
        # Convert PaperSection enums to strings
        sections_dict = {}
        for section, content in self.sections.items():
            sections_dict[section.value] = content
        result['sections'] = sections_dict
        
        return result


class IntelligentPaperGenerator:
    """
    Advanced system for generating research papers from notebook content
    using AI agents, SPARC methodology, and intelligent content synthesis.
    """
    
    def __init__(self, notebook_path: Path, ai_client=None):
        self.logger = setup_logger("research.paper_generator")
        self.notebook_path = notebook_path
        self.ai_client = ai_client
        
        # Initialize SPARC agent
        self.sparc_agent = SPARCAgent(ai_client=ai_client)
        
        # Data stores
        self.templates: Dict[str, PaperTemplate] = {}
        self.research_content: Dict[str, ResearchContent] = {}
        self.generated_papers: Dict[str, GeneratedPaper] = {}
        
        # Create directories
        self.papers_dir = notebook_path / "papers"
        self.templates_dir = self.papers_dir / "templates"
        self.drafts_dir = self.papers_dir / "drafts"
        self.published_dir = self.papers_dir / "published"
        self.content_dir = self.papers_dir / "research_content"
        
        for dir_path in [self.papers_dir, self.templates_dir, self.drafts_dir, 
                        self.published_dir, self.content_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize default templates
        self._initialize_default_templates()
        self._load_existing_data()
    
    def _initialize_default_templates(self):
        """Initialize default paper templates for common venues."""
        
        # Conference paper template (e.g., NeurIPS, ICML)
        conference_template = PaperTemplate(
            name="conference_ai",
            venue_type=VenueType.CONFERENCE,
            sections=[
                PaperSection.TITLE, PaperSection.ABSTRACT, PaperSection.INTRODUCTION,
                PaperSection.METHODOLOGY, PaperSection.RESULTS, PaperSection.DISCUSSION,
                PaperSection.CONCLUSION, PaperSection.REFERENCES
            ],
            word_limits={
                PaperSection.ABSTRACT: 250,
                PaperSection.INTRODUCTION: 1500,
                PaperSection.METHODOLOGY: 2000,
                PaperSection.RESULTS: 2500,
                PaperSection.DISCUSSION: 1500,
                PaperSection.CONCLUSION: 500
            },
            style_requirements={
                "page_limit": 8,
                "font_size": 11,
                "line_spacing": "single",
                "figures_included_in_limit": True,
                "anonymization_required": True
            },
            required_elements=["reproducibility_statement", "ethical_considerations"],
            citation_style="acl"
        )
        
        # Journal paper template
        journal_template = PaperTemplate(
            name="journal_standard",
            venue_type=VenueType.JOURNAL,
            sections=[
                PaperSection.TITLE, PaperSection.ABSTRACT, PaperSection.INTRODUCTION,
                PaperSection.LITERATURE_REVIEW, PaperSection.METHODOLOGY,
                PaperSection.RESULTS, PaperSection.DISCUSSION, PaperSection.CONCLUSION,
                PaperSection.REFERENCES, PaperSection.APPENDIX
            ],
            word_limits={
                PaperSection.ABSTRACT: 300,
                PaperSection.INTRODUCTION: 2000,
                PaperSection.LITERATURE_REVIEW: 3000,
                PaperSection.METHODOLOGY: 3000,
                PaperSection.RESULTS: 4000,
                PaperSection.DISCUSSION: 2500,
                PaperSection.CONCLUSION: 800
            },
            style_requirements={
                "word_limit": 12000,
                "font_size": 12,
                "line_spacing": "double",
                "figures_separate": True,
                "anonymization_required": False
            },
            citation_style="apa"
        )
        
        # Workshop paper template
        workshop_template = PaperTemplate(
            name="workshop_short",
            venue_type=VenueType.WORKSHOP,
            sections=[
                PaperSection.TITLE, PaperSection.ABSTRACT, PaperSection.INTRODUCTION,
                PaperSection.METHODOLOGY, PaperSection.RESULTS, PaperSection.CONCLUSION,
                PaperSection.REFERENCES
            ],
            word_limits={
                PaperSection.ABSTRACT: 150,
                PaperSection.INTRODUCTION: 800,
                PaperSection.METHODOLOGY: 1000,
                PaperSection.RESULTS: 1200,
                PaperSection.CONCLUSION: 400
            },
            style_requirements={
                "page_limit": 4,
                "font_size": 10,
                "preliminary_results_acceptable": True
            },
            citation_style="ieee"
        )
        
        # Store templates
        self.templates = {
            "conference_ai": conference_template,
            "journal_standard": journal_template,
            "workshop_short": workshop_template
        }
    
    def _load_existing_data(self):
        """Load existing papers and content from storage."""
        try:
            # Load research content
            for content_file in self.content_dir.glob("*.json"):
                with open(content_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content = self._dict_to_research_content(data)
                    self.research_content[content.content_id] = content
            
            # Load generated papers
            for paper_file in self.drafts_dir.glob("*.json"):
                with open(paper_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    paper = self._dict_to_generated_paper(data)
                    self.generated_papers[paper.paper_id] = paper
            
            self.logger.info(f"Loaded {len(self.research_content)} content items, "
                           f"{len(self.generated_papers)} papers")
                           
        except Exception as e:
            self.logger.error(f"Error loading paper generator data: {e}")
    
    def _dict_to_research_content(self, data: Dict) -> ResearchContent:
        """Convert dictionary to ResearchContent object."""
        data['date_created'] = datetime.fromisoformat(data['date_created'])
        return ResearchContent(**data)
    
    def _dict_to_generated_paper(self, data: Dict) -> GeneratedPaper:
        """Convert dictionary to GeneratedPaper object."""
        data['generation_date'] = datetime.fromisoformat(data['generation_date'])
        data['status'] = PaperStatus(data['status'])
        if data['venue_type']:
            data['venue_type'] = VenueType(data['venue_type'])
        
        # Convert section strings back to enums
        sections_dict = {}
        for section_str, content in data['sections'].items():
            sections_dict[PaperSection(section_str)] = content
        data['sections'] = sections_dict
        
        return GeneratedPaper(**data)
    
    async def analyze_research_content(
        self,
        content_sources: List[str],
        research_focus: str,
        content_types: List[str] = None
    ) -> List[ResearchContent]:
        """
        Analyze and structure research content from various sources for paper generation.
        """
        processed_content = []
        
        if content_types is None:
            content_types = ["experiment", "literature_review", "analysis", "observation"]
        
        for source in content_sources:
            try:
                # Read source content (assuming these are file paths)
                source_path = Path(source)
                if not source_path.exists():
                    self.logger.warning(f"Source file not found: {source}")
                    continue
                
                with open(source_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                
                # Use AI to analyze and structure the content
                analysis_result = await self._analyze_content_with_ai(
                    raw_content, research_focus, source_path.name
                )
                
                if analysis_result:
                    content_id = f"content_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(processed_content)}"
                    
                    research_content = ResearchContent(
                        content_id=content_id,
                        title=analysis_result.get('title', source_path.stem),
                        content_type=analysis_result.get('content_type', 'analysis'),
                        raw_content=raw_content,
                        processed_content=analysis_result.get('processed_content', raw_content),
                        key_insights=analysis_result.get('key_insights', []),
                        source_files=[source],
                        date_created=datetime.now(),
                        confidence_score=analysis_result.get('confidence_score', 0.7),
                        quality_score=analysis_result.get('quality_score', 0.6),
                        research_questions=analysis_result.get('research_questions', []),
                        hypotheses=analysis_result.get('hypotheses', []),
                        methods_used=analysis_result.get('methods_used', []),
                        results_summary=analysis_result.get('results_summary', ''),
                        suggested_sections=analysis_result.get('suggested_sections', [])
                    )
                    
                    # Store content
                    self.research_content[content_id] = research_content
                    self._save_research_content(research_content)
                    processed_content.append(research_content)
                    
            except Exception as e:
                self.logger.error(f"Error processing content source {source}: {e}")
        
        return processed_content
    
    async def _analyze_content_with_ai(
        self,
        content: str,
        research_focus: str,
        filename: str
    ) -> Optional[Dict[str, Any]]:
        """Use AI to analyze and extract structured information from research content."""
        
        if not self.ai_client:
            # Fallback to basic analysis
            return self._basic_content_analysis(content, filename)
        
        analysis_prompt = f"""
        Analyze the following research content in the context of {research_focus}:
        
        Content from file: {filename}
        ---
        {content[:3000]}  # Truncate for API limits
        ---
        
        Please provide a structured analysis including:
        1. Content type (experiment, literature_review, analysis, observation, etc.)
        2. Key insights and findings (list)
        3. Research questions addressed (list)
        4. Hypotheses mentioned or implied (list)
        5. Methods used (list)
        6. Results summary (paragraph)
        7. Suggested paper sections where this content would fit
        8. Quality assessment (0-1 score)
        9. Confidence in analysis (0-1 score)
        10. Processed/cleaned version of the content
        
        Return as JSON format.
        """
        
        try:
            result = await self.ai_client.generate_structured_response(
                analysis_prompt,
                response_format="json"
            )
            return result
            
        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            return self._basic_content_analysis(content, filename)
    
    def _basic_content_analysis(self, content: str, filename: str) -> Dict[str, Any]:
        """Basic fallback content analysis without AI."""
        
        # Simple keyword-based analysis
        experiment_keywords = ["experiment", "method", "procedure", "measurement", "data", "result"]
        literature_keywords = ["paper", "study", "research", "author", "citation", "review"]
        analysis_keywords = ["analysis", "finding", "conclusion", "insight", "pattern"]
        
        content_lower = content.lower()
        
        # Determine content type
        exp_score = sum(1 for kw in experiment_keywords if kw in content_lower)
        lit_score = sum(1 for kw in literature_keywords if kw in content_lower)
        ana_score = sum(1 for kw in analysis_keywords if kw in content_lower)
        
        if exp_score >= max(lit_score, ana_score):
            content_type = "experiment"
        elif lit_score >= ana_score:
            content_type = "literature_review"
        else:
            content_type = "analysis"
        
        # Extract basic insights (simplified)
        sentences = content.split('.')
        key_insights = [s.strip() for s in sentences if len(s.strip()) > 50 and 
                       any(kw in s.lower() for kw in ["finding", "result", "conclusion", "insight"])][:5]
        
        return {
            "content_type": content_type,
            "key_insights": key_insights,
            "research_questions": [],
            "hypotheses": [],
            "methods_used": [],
            "results_summary": content[:200] + "...",
            "suggested_sections": [PaperSection.METHODOLOGY, PaperSection.RESULTS],
            "quality_score": 0.6,
            "confidence_score": 0.5,
            "processed_content": content
        }
    
    async def generate_paper(
        self,
        title: str,
        authors: List[str],
        content_ids: List[str],
        template_name: str = "journal_standard",
        research_focus: str = "",
        target_venue: str = "",
        additional_requirements: Dict[str, Any] = None
    ) -> GeneratedPaper:
        """
        Generate a complete research paper from selected content using SPARC methodology.
        """
        
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        
        # Validate content exists
        selected_content = []
        for content_id in content_ids:
            if content_id in self.research_content:
                selected_content.append(self.research_content[content_id])
            else:
                self.logger.warning(f"Content ID {content_id} not found")
        
        if not selected_content:
            raise ValueError("No valid content found for paper generation")
        
        # Generate unique paper ID
        paper_id = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Generating paper {paper_id} using template {template_name}")
        
        # Generate paper sections using SPARC methodology
        sections = {}
        
        # Generate each required section
        for section in template.sections:
            if section == PaperSection.REFERENCES:
                sections[section] = self._generate_references(selected_content)
            else:
                section_content = await self._generate_section(
                    section, selected_content, template, research_focus, additional_requirements
                )
                sections[section] = section_content
        
        # Generate abstract last (after all content is generated)
        if PaperSection.ABSTRACT in sections:
            sections[PaperSection.ABSTRACT] = await self._generate_abstract(
                title, sections, template, research_focus
            )
        
        # Calculate quality scores
        quality_scores = self._assess_paper_quality(sections, selected_content, template)
        
        # Create generated paper object
        paper = GeneratedPaper(
            paper_id=paper_id,
            title=title,
            authors=authors,
            abstract=sections.get(PaperSection.ABSTRACT, ""),
            sections=sections,
            template_used=template_name,
            source_content_ids=content_ids,
            generation_date=datetime.now(),
            status=PaperStatus.DRAFT,
            target_venue=target_venue,
            venue_type=template.venue_type,
            coherence_score=quality_scores["coherence"],
            novelty_score=quality_scores["novelty"],
            completeness_score=quality_scores["completeness"],
            citation_quality_score=quality_scores["citation_quality"]
        )
        
        # Store paper
        self.generated_papers[paper_id] = paper
        self._save_generated_paper(paper)
        
        # Create paper files
        await self._create_paper_files(paper, template)
        
        self.logger.info(f"Successfully generated paper: {paper_id}")
        return paper
    
    async def _generate_section(
        self,
        section: PaperSection,
        content_list: List[ResearchContent],
        template: PaperTemplate,
        research_focus: str,
        requirements: Dict[str, Any] = None
    ) -> str:
        """Generate a specific section of the paper."""
        
        # Filter content relevant to this section
        relevant_content = [
            content for content in content_list
            if section in content.suggested_sections or self._is_content_relevant_to_section(content, section)
        ]
        
        if section == PaperSection.TITLE:
            return await self._generate_title(content_list, research_focus)
        
        elif section == PaperSection.INTRODUCTION:
            return await self._generate_introduction(relevant_content, research_focus, template)
        
        elif section == PaperSection.LITERATURE_REVIEW:
            return await self._generate_literature_review(relevant_content, research_focus)
        
        elif section == PaperSection.METHODOLOGY:
            return await self._generate_methodology(relevant_content, template)
        
        elif section == PaperSection.RESULTS:
            return await self._generate_results(relevant_content, template)
        
        elif section == PaperSection.DISCUSSION:
            return await self._generate_discussion(relevant_content, research_focus)
        
        elif section == PaperSection.CONCLUSION:
            return await self._generate_conclusion(content_list, research_focus)
        
        else:
            # Generic section generation
            return await self._generate_generic_section(section, relevant_content, template)
    
    def _is_content_relevant_to_section(self, content: ResearchContent, section: PaperSection) -> bool:
        """Determine if content is relevant to a specific section."""
        
        content_type = content.content_type.lower()
        
        if section == PaperSection.INTRODUCTION:
            return "background" in content_type or "motivation" in content_type
        elif section == PaperSection.LITERATURE_REVIEW:
            return "literature" in content_type or "review" in content_type
        elif section == PaperSection.METHODOLOGY:
            return "experiment" in content_type or "method" in content_type
        elif section == PaperSection.RESULTS:
            return "result" in content_type or "data" in content_type or "experiment" in content_type
        elif section == PaperSection.DISCUSSION:
            return "analysis" in content_type or "discussion" in content_type
        else:
            return True  # Default to including content
    
    async def _generate_title(self, content_list: List[ResearchContent], research_focus: str) -> str:
        """Generate an engaging and accurate title for the paper."""
        
        if not self.ai_client:
            # Fallback title generation
            key_terms = self._extract_key_terms(content_list)
            return f"{research_focus}: A Study on {', '.join(key_terms[:3])}"
        
        # Collect key insights and methods
        key_insights = []
        methods = []
        for content in content_list:
            key_insights.extend(content.key_insights)
            methods.extend(content.methods_used)
        
        title_prompt = f"""
        Generate an academic paper title based on the following research:
        
        Research Focus: {research_focus}
        
        Key Insights:
        {chr(10).join(f"- {insight}" for insight in key_insights[:10])}
        
        Methods Used:
        {chr(10).join(f"- {method}" for method in methods[:5])}
        
        Requirements:
        - Clear and concise (under 15 words)
        - Academically appropriate
        - Captures the main contribution
        - Avoids overly broad claims
        
        Generate 3 title options and select the best one.
        """
        
        try:
            result = await self.ai_client.generate_text(title_prompt)
            # Extract the best title (simplified)
            lines = result.strip().split('\n')
            for line in lines:
                if line.strip() and not line.startswith(('1.', '2.', '3.', '-', 'Best:')):
                    return line.strip().strip('"\'')
            return lines[-1].strip().strip('"\'') if lines else f"{research_focus} Research"
            
        except Exception as e:
            self.logger.error(f"Title generation failed: {e}")
            return f"{research_focus}: Research Findings and Analysis"
    
    async def _generate_introduction(
        self,
        content_list: List[ResearchContent],
        research_focus: str,
        template: PaperTemplate
    ) -> str:
        """Generate the introduction section using SPARC methodology."""
        
        word_limit = template.word_limits.get(PaperSection.INTRODUCTION, 1500)
        
        if not self.ai_client:
            return self._generate_basic_introduction(content_list, research_focus, word_limit)
        
        # SPARC: Situation - Problem - Action - Result - Conclusion structure
        
        # Extract situation context
        background_content = [c for c in content_list if "background" in c.content_type.lower()]
        research_questions = []
        for content in content_list:
            research_questions.extend(content.research_questions)
        
        introduction_prompt = f"""
        Write an academic introduction section ({word_limit} words max) using SPARC methodology:
        
        **Research Focus**: {research_focus}
        
        **Background Content**:
        {self._format_content_for_prompt(background_content)}
        
        **Research Questions**:
        {chr(10).join(f"- {q}" for q in research_questions[:5])}
        
        **Structure using SPARC**:
        1. **Situation**: Current state of the field and relevant background
        2. **Problem**: Gap/challenge this research addresses
        3. **Action**: Brief overview of your approach
        4. **Result**: Preview of main findings/contributions
        5. **Conclusion**: Paper organization and structure
        
        Requirements:
        - Academic writing style
        - Proper motivation and context
        - Clear problem statement
        - Brief methodology preview
        - Paper roadmap at the end
        """
        
        try:
            return await self.ai_client.generate_text(introduction_prompt, max_tokens=int(word_limit * 1.3))
        except Exception as e:
            self.logger.error(f"Introduction generation failed: {e}")
            return self._generate_basic_introduction(content_list, research_focus, word_limit)
    
    def _generate_basic_introduction(self, content_list: List[ResearchContent], research_focus: str, word_limit: int) -> str:
        """Basic introduction generation without AI."""
        
        intro_parts = []
        
        # Situation
        intro_parts.append(f"The field of {research_focus} has seen significant developments in recent years.")
        
        # Problem
        if content_list and content_list[0].research_questions:
            problem = f"However, important questions remain, including: {content_list[0].research_questions[0]}"
        else:
            problem = f"Despite these advances, several challenges persist in {research_focus}."
        intro_parts.append(problem)
        
        # Action & Results
        intro_parts.append("This paper addresses these challenges through comprehensive analysis and experimentation.")
        intro_parts.append("Our findings contribute to the understanding of this important research area.")
        
        # Conclusion
        intro_parts.append("The paper is organized as follows: Section 2 reviews related work, "
                          "Section 3 describes our methodology, Section 4 presents results, "
                          "and Section 5 discusses implications and future work.")
        
        return "\n\n".join(intro_parts)
    
    async def _generate_methodology(self, content_list: List[ResearchContent], template: PaperTemplate) -> str:
        """Generate the methodology section."""
        
        word_limit = template.word_limits.get(PaperSection.METHODOLOGY, 2000)
        
        # Extract experimental and methodological content
        method_content = [c for c in content_list if 
                         "experiment" in c.content_type.lower() or "method" in c.content_type.lower()]
        
        if not method_content:
            return "Methodology details were not available in the provided content."
        
        if not self.ai_client:
            return self._generate_basic_methodology(method_content, word_limit)
        
        # Collect methods and procedures
        all_methods = []
        procedures = []
        for content in method_content:
            all_methods.extend(content.methods_used)
            if content.processed_content:
                procedures.append(content.processed_content[:500])
        
        methodology_prompt = f"""
        Write a comprehensive methodology section ({word_limit} words max) based on:
        
        **Methods Used**:
        {chr(10).join(f"- {method}" for method in all_methods[:10])}
        
        **Detailed Procedures**:
        {chr(10).join(procedures)}
        
        **Structure**:
        1. Overview of research approach
        2. Data collection methods
        3. Analysis procedures
        4. Experimental setup (if applicable)
        5. Validation approach
        6. Ethical considerations
        
        Requirements:
        - Sufficient detail for reproducibility
        - Clear step-by-step procedures
        - Justification for methodological choices
        - Academic writing style
        """
        
        try:
            return await self.ai_client.generate_text(methodology_prompt, max_tokens=int(word_limit * 1.3))
        except Exception as e:
            self.logger.error(f"Methodology generation failed: {e}")
            return self._generate_basic_methodology(method_content, word_limit)
    
    def _generate_basic_methodology(self, method_content: List[ResearchContent], word_limit: int) -> str:
        """Basic methodology generation without AI."""
        
        method_parts = []
        
        method_parts.append("## Research Approach\n")
        method_parts.append("This study employs a systematic approach to address the research questions.")
        
        if method_content:
            methods_used = set()
            for content in method_content:
                methods_used.update(content.methods_used)
            
            if methods_used:
                method_parts.append("\n## Methods\n")
                method_parts.append("The following methods were employed:")
                for method in list(methods_used)[:5]:
                    method_parts.append(f"- {method}")
        
        method_parts.append("\n## Data Collection\n")
        method_parts.append("Data was collected systematically according to established protocols.")
        
        method_parts.append("\n## Analysis\n")
        method_parts.append("The collected data was analyzed using appropriate statistical and analytical methods.")
        
        return "\n".join(method_parts)
    
    async def _generate_results(self, content_list: List[ResearchContent], template: PaperTemplate) -> str:
        """Generate the results section."""
        
        word_limit = template.word_limits.get(PaperSection.RESULTS, 2500)
        
        # Filter for results-related content
        results_content = [c for c in content_list if 
                          "result" in c.content_type.lower() or 
                          "experiment" in c.content_type.lower() or
                          "data" in c.content_type.lower()]
        
        if not results_content:
            return "Results will be presented based on the conducted experiments and analysis."
        
        if not self.ai_client:
            return self._generate_basic_results(results_content, word_limit)
        
        # Collect results summaries and key findings
        results_summaries = []
        key_findings = []
        
        for content in results_content:
            if content.results_summary:
                results_summaries.append(content.results_summary)
            key_findings.extend(content.key_insights)
        
        results_prompt = f"""
        Write a comprehensive results section ({word_limit} words max) based on:
        
        **Results Summaries**:
        {chr(10).join(results_summaries)}
        
        **Key Findings**:
        {chr(10).join(f"- {finding}" for finding in key_findings[:15])}
        
        **Structure**:
        1. Overview of results
        2. Primary findings
        3. Secondary findings
        4. Statistical analysis
        5. Performance metrics
        6. Comparative analysis (if applicable)
        
        Requirements:
        - Objective presentation of findings
        - Clear organization by research question/hypothesis
        - Include quantitative results where available
        - Reference figures and tables appropriately
        - Avoid interpretation (save for discussion)
        """
        
        try:
            return await self.ai_client.generate_text(results_prompt, max_tokens=int(word_limit * 1.3))
        except Exception as e:
            self.logger.error(f"Results generation failed: {e}")
            return self._generate_basic_results(results_content, word_limit)
    
    def _generate_basic_results(self, results_content: List[ResearchContent], word_limit: int) -> str:
        """Basic results generation without AI."""
        
        results_parts = []
        
        results_parts.append("## Overview of Results\n")
        results_parts.append("This section presents the findings from our analysis and experiments.")
        
        # Extract key findings
        all_findings = []
        for content in results_content:
            all_findings.extend(content.key_insights)
        
        if all_findings:
            results_parts.append("\n## Key Findings\n")
            results_parts.append("The analysis revealed several important findings:")
            for i, finding in enumerate(all_findings[:8], 1):
                results_parts.append(f"{i}. {finding}")
        
        # Results summaries
        for i, content in enumerate(results_content[:3], 1):
            if content.results_summary:
                results_parts.append(f"\n## Results {i}\n")
                results_parts.append(content.results_summary)
        
        results_parts.append("\n## Summary\n")
        results_parts.append("These results provide important insights for the research questions addressed in this study.")
        
        return "\n".join(results_parts)
    
    async def _generate_discussion(self, content_list: List[ResearchContent], research_focus: str) -> str:
        """Generate the discussion section."""
        
        # Discussion should interpret results and place them in broader context
        if not self.ai_client:
            return self._generate_basic_discussion(content_list, research_focus)
        
        # Collect insights and implications
        all_insights = []
        research_questions = []
        
        for content in content_list:
            all_insights.extend(content.key_insights)
            research_questions.extend(content.research_questions)
        
        discussion_prompt = f"""
        Write a discussion section that interprets and contextualizes the results:
        
        **Research Focus**: {research_focus}
        
        **Key Insights to Discuss**:
        {chr(10).join(f"- {insight}" for insight in all_insights[:10])}
        
        **Research Questions Addressed**:
        {chr(10).join(f"- {q}" for q in research_questions[:5])}
        
        **Structure**:
        1. Interpretation of main findings
        2. Comparison with existing work
        3. Theoretical implications
        4. Practical implications
        5. Limitations of the study
        6. Future work directions
        
        Requirements:
        - Connect results to research questions
        - Discuss significance and implications
        - Acknowledge limitations honestly
        - Suggest future research directions
        - Academic writing style
        """
        
        try:
            return await self.ai_client.generate_text(discussion_prompt, max_tokens=2000)
        except Exception as e:
            self.logger.error(f"Discussion generation failed: {e}")
            return self._generate_basic_discussion(content_list, research_focus)
    
    def _generate_basic_discussion(self, content_list: List[ResearchContent], research_focus: str) -> str:
        """Basic discussion generation without AI."""
        
        discussion_parts = []
        
        discussion_parts.append("## Interpretation of Results\n")
        discussion_parts.append(f"The findings of this study contribute to our understanding of {research_focus}.")
        
        # Extract key insights for discussion
        key_insights = []
        for content in content_list:
            key_insights.extend(content.key_insights)
        
        if key_insights:
            discussion_parts.append("\n## Key Implications\n")
            discussion_parts.append("Several important implications emerge from our findings:")
            for insight in key_insights[:5]:
                discussion_parts.append(f"- {insight}")
        
        discussion_parts.append("\n## Limitations\n")
        discussion_parts.append("As with any research study, there are limitations to consider. "
                              "Future work should address these constraints and expand the scope of investigation.")
        
        discussion_parts.append("\n## Future Work\n")
        discussion_parts.append("This research opens several avenues for future investigation in this important area.")
        
        return "\n".join(discussion_parts)
    
    async def _generate_conclusion(self, content_list: List[ResearchContent], research_focus: str) -> str:
        """Generate the conclusion section."""
        
        if not self.ai_client:
            return self._generate_basic_conclusion(content_list, research_focus)
        
        # Collect main contributions and findings
        key_insights = []
        research_questions = []
        
        for content in content_list:
            key_insights.extend(content.key_insights)
            research_questions.extend(content.research_questions)
        
        conclusion_prompt = f"""
        Write a strong conclusion section that summarizes the research:
        
        **Research Focus**: {research_focus}
        
        **Main Findings**:
        {chr(10).join(f"- {insight}" for insight in key_insights[:8])}
        
        **Research Questions Addressed**:
        {chr(10).join(f"- {q}" for q in research_questions[:3])}
        
        **Structure**:
        1. Restate research objectives briefly
        2. Summarize key contributions
        3. Highlight main findings
        4. Discuss broader impact
        5. Final thoughts and call to action
        
        Requirements:
        - Concise summary of contributions
        - Clear statement of impact
        - No new information introduced
        - Strong closing statement
        - 300-500 words
        """
        
        try:
            return await self.ai_client.generate_text(conclusion_prompt, max_tokens=600)
        except Exception as e:
            self.logger.error(f"Conclusion generation failed: {e}")
            return self._generate_basic_conclusion(content_list, research_focus)
    
    def _generate_basic_conclusion(self, content_list: List[ResearchContent], research_focus: str) -> str:
        """Basic conclusion generation without AI."""
        
        conclusion_parts = []
        
        conclusion_parts.append(f"This research has investigated important aspects of {research_focus} "
                               "through systematic analysis and experimentation.")
        
        # Summarize key contributions
        key_insights = []
        for content in content_list:
            key_insights.extend(content.key_insights[:2])  # Limit for conclusion
        
        if key_insights:
            conclusion_parts.append("\nThe main contributions of this work include:")
            for insight in key_insights[:3]:
                conclusion_parts.append(f"- {insight}")
        
        conclusion_parts.append(f"\nThese findings advance our understanding of {research_focus} "
                               "and provide a foundation for future research in this area.")
        
        conclusion_parts.append("We hope this work will inspire further investigation and contribute "
                               "to continued progress in the field.")
        
        return "\n\n".join(conclusion_parts)
    
    async def _generate_abstract(
        self,
        title: str,
        sections: Dict[PaperSection, str],
        template: PaperTemplate,
        research_focus: str
    ) -> str:
        """Generate the abstract after all other sections are completed."""
        
        word_limit = template.word_limits.get(PaperSection.ABSTRACT, 250)
        
        if not self.ai_client:
            return self._generate_basic_abstract(sections, research_focus, word_limit)
        
        # Extract key information from generated sections
        intro_extract = sections.get(PaperSection.INTRODUCTION, "")[:500]
        method_extract = sections.get(PaperSection.METHODOLOGY, "")[:300]
        results_extract = sections.get(PaperSection.RESULTS, "")[:400]
        conclusion_extract = sections.get(PaperSection.CONCLUSION, "")[:300]
        
        abstract_prompt = f"""
        Write a comprehensive abstract ({word_limit} words max) for the paper titled: "{title}"
        
        **Research Focus**: {research_focus}
        
        **Introduction Summary**:
        {intro_extract}
        
        **Methodology Summary**:
        {method_extract}
        
        **Results Summary**:
        {results_extract}
        
        **Conclusion Summary**:
        {conclusion_extract}
        
        **Abstract Structure**:
        1. Background/motivation (1-2 sentences)
        2. Problem/objective (1 sentence)
        3. Methods/approach (1-2 sentences)
        4. Key results (2-3 sentences)
        5. Conclusions/implications (1-2 sentences)
        
        Requirements:
        - Self-contained and complete
        - No citations or references
        - Clear and concise
        - Highlights main contributions
        - Follows academic abstract conventions
        """
        
        try:
            return await self.ai_client.generate_text(abstract_prompt, max_tokens=int(word_limit * 1.2))
        except Exception as e:
            self.logger.error(f"Abstract generation failed: {e}")
            return self._generate_basic_abstract(sections, research_focus, word_limit)
    
    def _generate_basic_abstract(self, sections: Dict[PaperSection, str], research_focus: str, word_limit: int) -> str:
        """Basic abstract generation without AI."""
        
        abstract_parts = []
        
        # Background
        abstract_parts.append(f"This paper addresses important questions in {research_focus}.")
        
        # Objective
        abstract_parts.append("The objective is to advance understanding through systematic investigation.")
        
        # Methods
        abstract_parts.append("We employ comprehensive analysis and experimental methods.")
        
        # Results
        abstract_parts.append("The results demonstrate significant findings that contribute to the field.")
        
        # Conclusions
        abstract_parts.append("These findings have important implications for future research and practice.")
        
        return " ".join(abstract_parts)
    
    async def _generate_generic_section(
        self,
        section: PaperSection,
        content_list: List[ResearchContent],
        template: PaperTemplate
    ) -> str:
        """Generate a generic section when specific generators are not available."""
        
        if not content_list:
            return f"[{section.value.replace('_', ' ').title()} section content not available]"
        
        # Basic content compilation
        section_content = []
        section_content.append(f"## {section.value.replace('_', ' ').title()}")
        
        for content in content_list[:3]:  # Limit to top 3 content items
            if content.key_insights:
                section_content.append("\n**Key Points:**")
                for insight in content.key_insights[:3]:
                    section_content.append(f"- {insight}")
            
            if content.processed_content and len(content.processed_content) > 100:
                section_content.append(f"\n{content.processed_content[:300]}...")
        
        return "\n".join(section_content)
    
    def _generate_references(self, content_list: List[ResearchContent]) -> str:
        """Generate references section from content citations."""
        
        references = []
        
        # Extract citations from content (simplified)
        for content in content_list:
            if "citation" in content.raw_content.lower():
                # This would need more sophisticated citation extraction
                references.append(f"[Citation from {content.title}]")
        
        if not references:
            return "## References\n\n[References will be added based on citations in the text]"
        
        references_text = "## References\n\n"
        for i, ref in enumerate(references[:20], 1):
            references_text += f"{i}. {ref}\n"
        
        return references_text
    
    def _assess_paper_quality(
        self,
        sections: Dict[PaperSection, str],
        content_list: List[ResearchContent],
        template: PaperTemplate
    ) -> Dict[str, float]:
        """Assess the quality of the generated paper."""
        
        # Coherence score (based on section completeness and length)
        total_words = sum(len(content.split()) for content in sections.values())
        expected_words = sum(template.word_limits.values())
        completeness_ratio = min(total_words / max(expected_words, 1), 1.0)
        
        coherence_score = completeness_ratio * 0.7 + (len(sections) / len(template.sections)) * 0.3
        
        # Novelty score (based on content diversity and insights)
        unique_insights = set()
        for content in content_list:
            unique_insights.update(content.key_insights)
        
        novelty_score = min(len(unique_insights) / 10.0, 1.0)  # Normalize to 0-1
        
        # Completeness score
        required_sections = len([s for s in template.sections if s != PaperSection.REFERENCES])
        present_sections = len([s for s in sections if s != PaperSection.REFERENCES and sections[s].strip()])
        completeness_score = present_sections / max(required_sections, 1)
        
        # Citation quality (simplified)
        citation_indicators = sum(1 for content in sections.values() if "citation" in content.lower())
        citation_quality_score = min(citation_indicators / 5.0, 1.0)
        
        return {
            "coherence": round(coherence_score, 3),
            "novelty": round(novelty_score, 3),
            "completeness": round(completeness_score, 3),
            "citation_quality": round(citation_quality_score, 3)
        }
    
    def _extract_key_terms(self, content_list: List[ResearchContent]) -> List[str]:
        """Extract key terms from research content."""
        
        term_counts = Counter()
        
        for content in content_list:
            # Simple keyword extraction (would use NLP in production)
            words = re.findall(r'\b[A-Za-z]{4,}\b', content.processed_content.lower())
            # Filter out common words
            filtered_words = [w for w in words if w not in {'that', 'with', 'this', 'from', 'they', 'have', 'were', 'been', 'said'}]
            term_counts.update(filtered_words)
        
        return [term for term, count in term_counts.most_common(10)]
    
    def _format_content_for_prompt(self, content_list: List[ResearchContent]) -> str:
        """Format research content for AI prompts."""
        
        formatted_content = []
        for content in content_list[:3]:  # Limit to avoid prompt size issues
            formatted_content.append(f"**{content.title}**:")
            formatted_content.append(content.processed_content[:300] + "...")
            
            if content.key_insights:
                formatted_content.append("Key insights:")
                for insight in content.key_insights[:3]:
                    formatted_content.append(f"- {insight}")
            formatted_content.append("")
        
        return "\n".join(formatted_content)
    
    async def _create_paper_files(self, paper: GeneratedPaper, template: PaperTemplate):
        """Create various formats of the generated paper."""
        
        try:
            # Create Markdown version
            await self._create_markdown_version(paper)
            
            # Create LaTeX version if template supports it
            if template.style_requirements.get("latex_supported", True):
                await self._create_latex_version(paper, template)
            
            # Create summary note
            self._create_paper_summary_note(paper)
            
        except Exception as e:
            self.logger.error(f"Error creating paper files: {e}")
    
    async def _create_markdown_version(self, paper: GeneratedPaper):
        """Create Markdown version of the paper."""
        
        md_path = self.drafts_dir / f"{paper.paper_id}_draft.md"
        
        content_parts = []
        
        # Title and metadata
        content_parts.append(f"# {paper.title}")
        content_parts.append(f"\n**Authors**: {', '.join(paper.authors)}")
        content_parts.append(f"**Generated**: {paper.generation_date.strftime('%Y-%m-%d %H:%M')}")
        content_parts.append(f"**Status**: {paper.status.value}")
        content_parts.append(f"**Template**: {paper.template_used}")
        
        if paper.target_venue:
            content_parts.append(f"**Target Venue**: {paper.target_venue}")
        
        content_parts.append("\n---\n")
        
        # Abstract
        if PaperSection.ABSTRACT in paper.sections:
            content_parts.append("## Abstract")
            content_parts.append(paper.sections[PaperSection.ABSTRACT])
            content_parts.append("")
        
        # Other sections
        section_order = [
            PaperSection.INTRODUCTION,
            PaperSection.LITERATURE_REVIEW,
            PaperSection.METHODOLOGY,
            PaperSection.RESULTS,
            PaperSection.DISCUSSION,
            PaperSection.CONCLUSION,
            PaperSection.REFERENCES
        ]
        
        for section in section_order:
            if section in paper.sections and paper.sections[section].strip():
                section_title = section.value.replace('_', ' ').title()
                content_parts.append(f"## {section_title}")
                content_parts.append(paper.sections[section])
                content_parts.append("")
        
        # Quality metrics
        content_parts.append("---")
        content_parts.append("## Generation Metrics")
        content_parts.append(f"- **Coherence Score**: {paper.coherence_score:.1%}")
        content_parts.append(f"- **Novelty Score**: {paper.novelty_score:.1%}")
        content_parts.append(f"- **Completeness Score**: {paper.completeness_score:.1%}")
        content_parts.append(f"- **Citation Quality**: {paper.citation_quality_score:.1%}")
        
        # Write file
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_parts))
        
        self.logger.info(f"Created Markdown version: {md_path}")
    
    async def _create_latex_version(self, paper: GeneratedPaper, template: PaperTemplate):
        """Create LaTeX version of the paper."""
        
        tex_path = self.drafts_dir / f"{paper.paper_id}_draft.tex"
        
        # Basic LaTeX template
        latex_content = []
        
        # Document class and packages
        latex_content.append("\\documentclass[11pt,letterpaper]{article}")
        latex_content.append("\\usepackage[utf8]{inputenc}")
        latex_content.append("\\usepackage{amsmath,amsfonts,amssymb}")
        latex_content.append("\\usepackage{graphicx}")
        latex_content.append("\\usepackage{cite}")
        latex_content.append("\\usepackage{url}")
        latex_content.append("")
        
        # Title and authors
        latex_content.append(f"\\title{{{paper.title}}}")
        latex_content.append(f"\\author{{{' \\and '.join(paper.authors)}}}")
        latex_content.append("\\date{\\today}")
        latex_content.append("")
        
        # Begin document
        latex_content.append("\\begin{document}")
        latex_content.append("\\maketitle")
        latex_content.append("")
        
        # Abstract
        if PaperSection.ABSTRACT in paper.sections:
            latex_content.append("\\begin{abstract}")
            latex_content.append(paper.sections[PaperSection.ABSTRACT])
            latex_content.append("\\end{abstract}")
            latex_content.append("")
        
        # Sections
        section_order = [
            PaperSection.INTRODUCTION,
            PaperSection.LITERATURE_REVIEW,
            PaperSection.METHODOLOGY,
            PaperSection.RESULTS,
            PaperSection.DISCUSSION,
            PaperSection.CONCLUSION
        ]
        
        for section in section_order:
            if section in paper.sections and paper.sections[section].strip():
                section_title = section.value.replace('_', ' ').title()
                latex_content.append(f"\\section{{{section_title}}}")
                # Clean content for LaTeX
                cleaned_content = paper.sections[section].replace('#', '\\#').replace('_', '\\_')
                latex_content.append(cleaned_content)
                latex_content.append("")
        
        # References
        if PaperSection.REFERENCES in paper.sections:
            latex_content.append("\\section{References}")
            latex_content.append(paper.sections[PaperSection.REFERENCES])
        
        latex_content.append("\\end{document}")
        
        # Write file
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex_content))
        
        # Store reference to LaTeX version
        paper.latex_version = str(tex_path)
        
        self.logger.info(f"Created LaTeX version: {tex_path}")
    
    def _create_paper_summary_note(self, paper: GeneratedPaper):
        """Create an Obsidian note summarizing the generated paper."""
        
        note_path = self.drafts_dir / f"{paper.paper_id}_summary.md"
        
        content_parts = []
        
        content_parts.append(f"# Paper Summary: {paper.title}")
        content_parts.append("")
        content_parts.append("## Paper Details")
        content_parts.append(f"- **Paper ID**: {paper.paper_id}")
        content_parts.append(f"- **Authors**: {', '.join(paper.authors)}")
        content_parts.append(f"- **Status**: {paper.status.value}")
        content_parts.append(f"- **Generated**: {paper.generation_date.strftime('%Y-%m-%d %H:%M')}")
        content_parts.append(f"- **Template**: {paper.template_used}")
        
        if paper.target_venue:
            content_parts.append(f"- **Target Venue**: {paper.target_venue}")
        
        content_parts.append("")
        content_parts.append("## Abstract")
        content_parts.append(paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract)
        
        content_parts.append("")
        content_parts.append("## Quality Metrics")
        content_parts.append(f"- **Coherence**: {paper.coherence_score:.1%}")
        content_parts.append(f"- **Novelty**: {paper.novelty_score:.1%}")
        content_parts.append(f"- **Completeness**: {paper.completeness_score:.1%}")
        content_parts.append(f"- **Citation Quality**: {paper.citation_quality_score:.1%}")
        
        content_parts.append("")
        content_parts.append("## Source Content")
        for content_id in paper.source_content_ids:
            if content_id in self.research_content:
                content = self.research_content[content_id]
                content_parts.append(f"- [[{content.title}]] ({content.content_type})")
        
        content_parts.append("")
        content_parts.append("## Files Generated")
        content_parts.append(f"- [[{paper.paper_id}_draft.md|Markdown Draft]]")
        
        if paper.latex_version:
            content_parts.append(f"- [[{paper.paper_id}_draft.tex|LaTeX Version]]")
        
        content_parts.append("")
        content_parts.append("## Next Steps")
        content_parts.append("- [ ] Review and revise content")
        content_parts.append("- [ ] Add proper citations")
        content_parts.append("- [ ] Create figures and tables")
        content_parts.append("- [ ] Get feedback from collaborators")
        content_parts.append("- [ ] Submit to target venue")
        
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_parts))
        
        self.logger.info(f"Created paper summary note: {note_path}")
    
    def _save_research_content(self, content: ResearchContent):
        """Save research content to JSON storage."""
        file_path = self.content_dir / f"{content.content_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(content.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _save_generated_paper(self, paper: GeneratedPaper):
        """Save generated paper to JSON storage."""
        file_path = self.drafts_dir / f"{paper.paper_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(paper.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics about paper generation activities."""
        
        stats = {
            "total_content_items": len(self.research_content),
            "total_papers_generated": len(self.generated_papers),
            "papers_by_status": {},
            "papers_by_template": {},
            "content_by_type": {},
            "quality_statistics": {
                "avg_coherence": 0,
                "avg_novelty": 0,
                "avg_completeness": 0,
                "avg_citation_quality": 0
            },
            "recent_activity": []
        }
        
        # Papers by status
        for paper in self.generated_papers.values():
            status = paper.status.value
            stats["papers_by_status"][status] = stats["papers_by_status"].get(status, 0) + 1
        
        # Papers by template
        for paper in self.generated_papers.values():
            template = paper.template_used
            stats["papers_by_template"][template] = stats["papers_by_template"].get(template, 0) + 1
        
        # Content by type
        for content in self.research_content.values():
            content_type = content.content_type
            stats["content_by_type"][content_type] = stats["content_by_type"].get(content_type, 0) + 1
        
        # Quality statistics
        if self.generated_papers:
            papers = list(self.generated_papers.values())
            stats["quality_statistics"]["avg_coherence"] = sum(p.coherence_score for p in papers) / len(papers)
            stats["quality_statistics"]["avg_novelty"] = sum(p.novelty_score for p in papers) / len(papers)
            stats["quality_statistics"]["avg_completeness"] = sum(p.completeness_score for p in papers) / len(papers)
            stats["quality_statistics"]["avg_citation_quality"] = sum(p.citation_quality_score for p in papers) / len(papers)
        
        # Recent activity (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        
        recent_papers = [p for p in self.generated_papers.values() if p.generation_date >= cutoff_date]
        recent_content = [c for c in self.research_content.values() if c.date_created >= cutoff_date]
        
        for paper in recent_papers:
            stats["recent_activity"].append({
                "type": "paper_generated",
                "date": paper.generation_date.isoformat(),
                "title": paper.title,
                "paper_id": paper.paper_id
            })
        
        for content in recent_content:
            stats["recent_activity"].append({
                "type": "content_processed",
                "date": content.date_created.isoformat(),
                "title": content.title,
                "content_id": content.content_id
            })
        
        # Sort recent activity by date
        stats["recent_activity"].sort(key=lambda x: x["date"], reverse=True)
        
        return stats