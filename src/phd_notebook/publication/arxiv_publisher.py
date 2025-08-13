"""
ArXiv publication automation system.
Handles automated submission to arXiv with proper formatting and metadata.
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ArxivSubmission:
    """ArXiv submission data structure."""
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]  # cs.LG, stat.ML, etc.
    content_path: Path
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    comments: Optional[str] = None
    submission_id: Optional[str] = None
    status: str = "draft"


class ArxivPublisher:
    """
    Automated arXiv publication system.
    
    Handles formatting, validation, and submission to arXiv
    with academic best practices and automated quality checks.
    """
    
    def __init__(self, default_categories: List[str] = None):
        self.default_categories = default_categories or ["cs.LG"]
        self.valid_categories = self._load_valid_categories()
        self.submission_history: List[ArxivSubmission] = []
        
    def _load_valid_categories(self) -> Dict[str, List[str]]:
        """Load valid arXiv categories."""
        return {
            "cs": [  # Computer Science
                "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", 
                "cs.CV", "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", 
                "cs.ET", "cs.FL", "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", 
                "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS", "cs.NA", 
                "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", 
                "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY"
            ],
            "stat": [  # Statistics
                "stat.AP", "stat.CO", "stat.ME", "stat.ML", "stat.OT", "stat.TH"
            ],
            "math": [  # Mathematics
                "math.AG", "math.AT", "math.AP", "math.CA", "math.CO", "math.CT", 
                "math.CV", "math.DG", "math.DS", "math.FA", "math.GM", "math.GN", 
                "math.GR", "math.GT", "math.HO", "math.IT", "math.KT", "math.LO", 
                "math.MG", "math.MP", "math.NA", "math.NT", "math.OA", "math.OC", 
                "math.PR", "math.QA", "math.RA", "math.RT", "math.SG", "math.SP", 
                "math.ST"
            ],
            "physics": [  # Physics
                "physics.acc-ph", "physics.ao-ph", "physics.app-ph", "physics.atm-clus", 
                "physics.atom-ph", "physics.bio-ph", "physics.chem-ph", "physics.class-ph", 
                "physics.comp-ph", "physics.data-an", "physics.ed-ph", "physics.flu-dyn", 
                "physics.gen-ph", "physics.geo-ph", "physics.hist-ph", "physics.ins-det", 
                "physics.med-ph", "physics.optics", "physics.plasm-ph", "physics.pop-ph", 
                "physics.soc-ph", "physics.space-ph"
            ]
        }
    
    def validate_submission(self, submission: ArxivSubmission) -> Tuple[bool, List[str]]:
        """
        Validate arXiv submission for completeness and compliance.
        
        Returns:
            Tuple of (is_valid, validation_errors)
        """
        errors = []
        
        # Title validation
        if not submission.title or len(submission.title.strip()) < 10:
            errors.append("Title must be at least 10 characters long")
        
        if len(submission.title) > 200:
            errors.append("Title must be less than 200 characters")
        
        # Authors validation
        if not submission.authors:
            errors.append("At least one author is required")
        
        for author in submission.authors:
            if not re.match(r'^[A-Za-z\s\.\-]+$', author):
                errors.append(f"Invalid author name format: {author}")
        
        # Abstract validation
        if not submission.abstract or len(submission.abstract.strip()) < 100:
            errors.append("Abstract must be at least 100 characters long")
        
        if len(submission.abstract) > 1920:  # arXiv limit
            errors.append("Abstract must be less than 1920 characters")
        
        # Category validation
        if not submission.categories:
            errors.append("At least one category is required")
        
        for category in submission.categories:
            if not self._is_valid_category(category):
                errors.append(f"Invalid arXiv category: {category}")
        
        # Content validation
        if not submission.content_path or not submission.content_path.exists():
            errors.append("Content file does not exist")
        elif submission.content_path.suffix not in ['.tex', '.pdf']:
            errors.append("Content must be LaTeX (.tex) or PDF (.pdf)")
        
        # Format validation
        format_errors = self._validate_academic_format(submission)
        errors.extend(format_errors)
        
        return len(errors) == 0, errors
    
    def _is_valid_category(self, category: str) -> bool:
        """Check if category is valid arXiv category."""
        for domain, cats in self.valid_categories.items():
            if category in cats:
                return True
        return False
    
    def _validate_academic_format(self, submission: ArxivSubmission) -> List[str]:
        """Validate academic formatting requirements."""
        errors = []
        
        # Check title format
        title = submission.title
        
        # Title should not be all caps
        if title.isupper():
            errors.append("Title should not be in all capital letters")
        
        # Title should not end with period
        if title.endswith('.'):
            errors.append("Title should not end with a period")
        
        # Abstract format validation
        abstract = submission.abstract
        
        # Abstract should not contain citations
        if re.search(r'\[[^\]]+\]|\([^\)]*\d{4}[^\)]*\)', abstract):
            errors.append("Abstract should not contain citations")
        
        # Abstract should not contain figures/tables references
        if re.search(r'(figure|table|fig\.|tab\.)\s*\d+', abstract, re.IGNORECASE):
            errors.append("Abstract should not reference figures or tables")
        
        return errors
    
    def prepare_submission(self, paper_data: Dict[str, Any], 
                          categories: List[str] = None) -> ArxivSubmission:
        """
        Prepare paper data for arXiv submission.
        
        Args:
            paper_data: Paper content and metadata
            categories: arXiv categories (defaults to class default)
            
        Returns:
            ArxivSubmission object ready for validation and submission
        """
        categories = categories or self.default_categories
        
        # Extract authors from paper data
        authors = paper_data.get('authors', [])
        if isinstance(authors, str):
            # Split comma-separated author string
            authors = [author.strip() for author in authors.split(',')]
        
        # Clean and format title
        title = paper_data.get('title', '').strip()
        title = self._format_title(title)
        
        # Clean and format abstract
        abstract = paper_data.get('abstract', '').strip()
        if not abstract and 'content' in paper_data:
            # Try to extract abstract from content
            abstract = self._extract_abstract_from_content(paper_data['content'])
        
        # Determine content path
        content_path = paper_data.get('content_path')
        if content_path:
            content_path = Path(content_path)
        else:
            # Create temporary content file
            content_path = self._create_content_file(paper_data)
        
        # Generate comments
        comments = self._generate_comments(paper_data)
        
        submission = ArxivSubmission(
            title=title,
            authors=authors,
            abstract=abstract,
            categories=categories,
            content_path=content_path,
            doi=paper_data.get('doi'),
            journal_ref=paper_data.get('journal_ref'),
            comments=comments
        )
        
        return submission
    
    def _format_title(self, title: str) -> str:
        """Format title according to academic conventions."""
        if not title:
            return title
        
        # Remove extra whitespace
        title = ' '.join(title.split())
        
        # Capitalize first letter
        if title:
            title = title[0].upper() + title[1:]
        
        # Remove trailing period if present
        if title.endswith('.'):
            title = title[:-1]
        
        return title
    
    def _extract_abstract_from_content(self, content: str) -> str:
        """Extract abstract from paper content."""
        # Look for abstract section
        abstract_pattern = r'\\begin\{abstract\}(.*?)\\end\{abstract\}'
        match = re.search(abstract_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            abstract = match.group(1).strip()
            # Clean LaTeX commands
            abstract = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', abstract)
            abstract = re.sub(r'\\[a-zA-Z]+', '', abstract)
            abstract = ' '.join(abstract.split())
            return abstract
        
        return ""
    
    def _create_content_file(self, paper_data: Dict[str, Any]) -> Path:
        """Create content file from paper data."""
        # This would typically create a LaTeX file
        # For now, return a placeholder path
        return Path("/tmp/paper_content.tex")
    
    def _generate_comments(self, paper_data: Dict[str, Any]) -> str:
        """Generate comments field for arXiv."""
        comments = []
        
        # Add page count if available
        if 'page_count' in paper_data:
            comments.append(f"{paper_data['page_count']} pages")
        
        # Add figure count
        if 'figure_count' in paper_data:
            comments.append(f"{paper_data['figure_count']} figures")
        
        # Add conference/journal info if available
        if 'conference' in paper_data:
            comments.append(f"Submitted to {paper_data['conference']}")
        elif 'journal' in paper_data:
            comments.append(f"Submitted to {paper_data['journal']}")
        
        # Add funding acknowledgment if present
        if 'funding' in paper_data:
            comments.append("Funding information included")
        
        return ', '.join(comments) if comments else None
    
    def simulate_submission(self, submission: ArxivSubmission) -> Dict[str, Any]:
        """
        Simulate arXiv submission (for testing/demo purposes).
        
        In a real implementation, this would interact with arXiv API.
        """
        # Validate submission first
        is_valid, errors = self.validate_submission(submission)
        
        if not is_valid:
            return {
                "success": False,
                "errors": errors,
                "submission_id": None
            }
        
        # Generate mock arXiv ID
        timestamp = datetime.now().strftime("%y%m")
        submission_id = f"{timestamp}.{len(self.submission_history):04d}"
        
        # Update submission
        submission.submission_id = submission_id
        submission.status = "submitted"
        
        # Add to history
        self.submission_history.append(submission)
        
        # Generate mock response
        return {
            "success": True,
            "submission_id": submission_id,
            "arxiv_url": f"https://arxiv.org/abs/{submission_id}",
            "pdf_url": f"https://arxiv.org/pdf/{submission_id}.pdf",
            "submission_date": datetime.now().isoformat(),
            "status": "submitted",
            "estimated_publication": "Within 24-48 hours"
        }
    
    def check_submission_status(self, submission_id: str) -> Dict[str, Any]:
        """Check status of submitted paper."""
        # Find submission in history
        submission = None
        for sub in self.submission_history:
            if sub.submission_id == submission_id:
                submission = sub
                break
        
        if not submission:
            return {"error": "Submission not found"}
        
        # Mock status check
        return {
            "submission_id": submission_id,
            "title": submission.title,
            "status": submission.status,
            "arxiv_url": f"https://arxiv.org/abs/{submission_id}",
            "submitted_date": datetime.now().isoformat(),
            "version": "v1"
        }
    
    def update_submission(self, submission_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing arXiv submission."""
        # Find submission
        submission = None
        for sub in self.submission_history:
            if sub.submission_id == submission_id:
                submission = sub
                break
        
        if not submission:
            return {"success": False, "error": "Submission not found"}
        
        # Apply updates
        if 'title' in updates:
            submission.title = updates['title']
        if 'abstract' in updates:
            submission.abstract = updates['abstract']
        if 'categories' in updates:
            submission.categories = updates['categories']
        if 'comments' in updates:
            submission.comments = updates['comments']
        
        # Validate updated submission
        is_valid, errors = self.validate_submission(submission)
        
        if not is_valid:
            return {"success": False, "errors": errors}
        
        submission.status = "updated"
        
        return {
            "success": True,
            "submission_id": submission_id,
            "status": "updated",
            "new_version": "v2"
        }
    
    def get_submission_statistics(self) -> Dict[str, Any]:
        """Get statistics about submissions."""
        if not self.submission_history:
            return {"total_submissions": 0}
        
        total = len(self.submission_history)
        by_status = {}
        by_category = {}
        
        for submission in self.submission_history:
            # Count by status
            status = submission.status
            by_status[status] = by_status.get(status, 0) + 1
            
            # Count by category
            for category in submission.categories:
                by_category[category] = by_category.get(category, 0) + 1
        
        return {
            "total_submissions": total,
            "by_status": by_status,
            "by_category": by_category,
            "most_common_category": max(by_category.items(), key=lambda x: x[1])[0] if by_category else None,
            "success_rate": by_status.get("submitted", 0) / total if total > 0 else 0
        }
    
    def generate_arxiv_metadata(self, submission: ArxivSubmission) -> str:
        """Generate arXiv metadata in the required format."""
        metadata_lines = []
        
        # Title
        metadata_lines.append(f"Title: {submission.title}")
        
        # Authors
        authors_str = ", ".join(submission.authors)
        metadata_lines.append(f"Authors: {authors_str}")
        
        # Categories
        categories_str = ", ".join(submission.categories)
        metadata_lines.append(f"Categories: {categories_str}")
        
        # Abstract
        # Wrap abstract at 80 characters
        abstract_lines = []
        words = submission.abstract.split()
        current_line = "Abstract: "
        
        for word in words:
            if len(current_line + word) > 77:  # Leave room for line break
                abstract_lines.append(current_line)
                current_line = "  " + word  # Indent continuation lines
            else:
                if current_line.endswith(': '):
                    current_line += word
                else:
                    current_line += ' ' + word
        
        if current_line.strip():
            abstract_lines.append(current_line)
        
        metadata_lines.extend(abstract_lines)
        
        # Optional fields
        if submission.doi:
            metadata_lines.append(f"DOI: {submission.doi}")
        
        if submission.journal_ref:
            metadata_lines.append(f"Journal-ref: {submission.journal_ref}")
        
        if submission.comments:
            metadata_lines.append(f"Comments: {submission.comments}")
        
        return "\\n".join(metadata_lines)
    
    def export_submission_summary(self) -> Dict[str, Any]:
        """Export summary of all submissions for reporting."""
        summary = {
            "export_timestamp": datetime.now().isoformat(),
            "total_submissions": len(self.submission_history),
            "submissions": []
        }
        
        for submission in self.submission_history:
            summary["submissions"].append({
                "submission_id": submission.submission_id,
                "title": submission.title,
                "authors": submission.authors,
                "categories": submission.categories,
                "status": submission.status,
                "arxiv_url": f"https://arxiv.org/abs/{submission.submission_id}" if submission.submission_id else None
            })
        
        return summary