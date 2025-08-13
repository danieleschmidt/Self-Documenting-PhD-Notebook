"""
Citation management and bibliography generation system.
Supports multiple citation styles and automated reference formatting.
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class CitationStyle(Enum):
    """Supported citation styles."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    HARVARD = "harvard"
    VANCOUVER = "vancouver"
    NATURE = "nature"
    SCIENCE = "science"


@dataclass
class Citation:
    """Citation data structure."""
    id: str
    title: str
    authors: List[str]
    year: int
    publication_type: str  # journal, conference, book, thesis, etc.
    venue: str  # journal name, conference name, etc.
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    isbn: Optional[str] = None
    publisher: Optional[str] = None
    location: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = None
    notes: Optional[str] = None


class CitationManager:
    """
    Advanced citation management system for academic writing.
    
    Handles citation parsing, formatting, and bibliography generation
    with support for multiple academic citation styles.
    """
    
    def __init__(self, default_style: CitationStyle = CitationStyle.APA):
        self.default_style = default_style
        self.citations: Dict[str, Citation] = {}
        self.citation_keys: Dict[str, str] = {}  # Maps citation keys to IDs
        self.style_formatters = self._load_style_formatters()
        
    def _load_style_formatters(self) -> Dict[CitationStyle, Dict[str, str]]:
        """Load citation formatting templates for different styles."""
        return {
            CitationStyle.APA: {
                "journal": "{authors} ({year}). {title}. {venue}, {volume}({issue}), {pages}. {doi}",
                "conference": "{authors} ({year}). {title}. In {venue} (pp. {pages}). {publisher}.",
                "book": "{authors} ({year}). {title}. {publisher}.",
                "thesis": "{authors} ({year}). {title} ({publication_type}). {publisher}.",
                "in_text": "({authors}, {year})",
                "in_text_page": "({authors}, {year}, p. {page})"
            },
            CitationStyle.IEEE: {
                "journal": "[{number}] {authors}, \"{title},\" {venue}, vol. {volume}, no. {issue}, pp. {pages}, {year}. {doi}",
                "conference": "[{number}] {authors}, \"{title},\" in {venue}, {year}, pp. {pages}.",
                "book": "[{number}] {authors}, {title}. {publisher}, {year}.",
                "thesis": "[{number}] {authors}, \"{title},\" {publication_type}, {publisher}, {year}.",
                "in_text": "[{number}]",
                "in_text_page": "[{number}, p. {page}]"
            },
            CitationStyle.NATURE: {
                "journal": "{authors}. {title}. {venue} {volume}, {pages} ({year}). {doi}",
                "conference": "{authors}. {title}. in {venue} {pages} ({publisher}, {year}).",
                "book": "{authors}. {title} ({publisher}, {year}).",
                "thesis": "{authors}. {title}. {publication_type}, {publisher} ({year}).",
                "in_text": "{number}",
                "in_text_page": "{number}"
            }
        }
    
    def add_citation(self, citation: Citation, key: Optional[str] = None) -> str:
        """
        Add citation to the manager.
        
        Args:
            citation: Citation object to add
            key: Optional citation key (e.g., "smith2023")
            
        Returns:
            Citation ID
        """
        # Generate key if not provided
        if not key:
            key = self._generate_citation_key(citation)
        
        # Store citation
        self.citations[citation.id] = citation
        self.citation_keys[key] = citation.id
        
        return citation.id
    
    def _generate_citation_key(self, citation: Citation) -> str:
        """Generate citation key from citation data."""
        # Get first author last name
        first_author = citation.authors[0] if citation.authors else "unknown"
        last_name = first_author.split()[-1].lower()
        
        # Remove non-alphanumeric characters
        last_name = re.sub(r'[^a-z0-9]', '', last_name)
        
        # Create base key
        base_key = f"{last_name}{citation.year}"
        
        # Handle duplicates by adding suffix
        key = base_key
        counter = 1
        while key in self.citation_keys:
            key = f"{base_key}_{counter}"
            counter += 1
        
        return key
    
    def parse_bibtex_entry(self, bibtex_text: str) -> Citation:
        """Parse a BibTeX entry into a Citation object."""
        # Simple BibTeX parser
        entry_match = re.match(r'@(\w+)\{([^,]+),\s*(.*)\}', bibtex_text, re.DOTALL)
        
        if not entry_match:
            raise ValueError("Invalid BibTeX format")
        
        entry_type = entry_match.group(1).lower()
        entry_key = entry_match.group(2).strip()
        fields_text = entry_match.group(3)
        
        # Parse fields
        fields = {}
        field_pattern = r'(\w+)\s*=\s*\{([^}]*)\}|(\w+)\s*=\s*"([^"]*)"'
        
        for match in re.finditer(field_pattern, fields_text):
            if match.group(1):
                field_name = match.group(1).lower()
                field_value = match.group(2)
            else:
                field_name = match.group(3).lower()
                field_value = match.group(4)
            
            fields[field_name] = field_value.strip()
        
        # Extract citation data
        authors = self._parse_bibtex_authors(fields.get('author', ''))
        title = fields.get('title', '')
        year = int(fields.get('year', '0'))
        
        # Determine publication type and venue
        publication_type = entry_type
        venue = ''
        
        if entry_type in ['article', 'inproceedings']:
            venue = fields.get('journal') or fields.get('booktitle', '')
        elif entry_type == 'book':
            venue = fields.get('publisher', '')
        
        citation = Citation(
            id=entry_key,
            title=title,
            authors=authors,
            year=year,
            publication_type=publication_type,
            venue=venue,
            volume=fields.get('volume'),
            issue=fields.get('number'),
            pages=fields.get('pages'),
            doi=fields.get('doi'),
            url=fields.get('url'),
            publisher=fields.get('publisher'),
            location=fields.get('location') or fields.get('address')
        )
        
        return citation
    
    def _parse_bibtex_authors(self, author_string: str) -> List[str]:
        """Parse BibTeX author string into list of authors."""
        if not author_string:
            return []
        
        # Split by 'and'
        authors = re.split(r'\\s+and\\s+', author_string)
        
        # Clean up author names
        cleaned_authors = []
        for author in authors:
            author = author.strip()
            if author:
                # Handle "Last, First" format
                if ',' in author:
                    parts = author.split(',', 1)
                    last = parts[0].strip()
                    first = parts[1].strip()
                    cleaned_authors.append(f"{first} {last}")
                else:
                    cleaned_authors.append(author)
        
        return cleaned_authors
    
    def format_citation(self, citation_id: str, style: CitationStyle = None, 
                       citation_number: int = None) -> str:
        """
        Format citation according to specified style.
        
        Args:
            citation_id: ID of citation to format
            style: Citation style to use (defaults to default_style)
            citation_number: Number for numbered citation styles
            
        Returns:
            Formatted citation string
        """
        style = style or self.default_style
        
        if citation_id not in self.citations:
            return f"[Citation {citation_id} not found]"
        
        citation = self.citations[citation_id]
        
        # Get formatting template
        templates = self.style_formatters.get(style, self.style_formatters[CitationStyle.APA])
        template = templates.get(citation.publication_type, templates.get("journal", ""))
        
        # Format authors according to style
        formatted_authors = self._format_authors(citation.authors, style)
        
        # Prepare replacement values
        replacements = {
            "authors": formatted_authors,
            "title": citation.title,
            "year": str(citation.year),
            "venue": citation.venue,
            "volume": citation.volume or "",
            "issue": citation.issue or "",
            "pages": citation.pages or "",
            "doi": f"doi:{citation.doi}" if citation.doi else "",
            "url": citation.url or "",
            "publisher": citation.publisher or "",
            "location": citation.location or "",
            "publication_type": citation.publication_type,
            "number": str(citation_number) if citation_number else ""
        }
        
        # Replace placeholders in template
        formatted = template
        for key, value in replacements.items():
            formatted = formatted.replace(f"{{{key}}}", value)
        
        # Clean up formatting
        formatted = re.sub(r',\s*,', ',', formatted)  # Remove double commas
        formatted = re.sub(r'\s+', ' ', formatted)  # Normalize whitespace
        formatted = formatted.strip()
        
        return formatted
    
    def _format_authors(self, authors: List[str], style: CitationStyle) -> str:
        """Format author list according to citation style."""
        if not authors:
            return ""
        
        if style == CitationStyle.APA:
            if len(authors) == 1:
                return authors[0]
            elif len(authors) == 2:
                return f"{authors[0]} & {authors[1]}"
            elif len(authors) <= 6:
                return ", ".join(authors[:-1]) + f", & {authors[-1]}"
            else:
                return f"{authors[0]} et al."
                
        elif style == CitationStyle.IEEE:
            if len(authors) <= 3:
                return ", ".join(authors)
            else:
                return f"{authors[0]} et al."
                
        elif style == CitationStyle.NATURE:
            if len(authors) == 1:
                return authors[0]
            elif len(authors) <= 5:
                return ", ".join(authors)
            else:
                return f"{authors[0]} et al."
        
        # Default formatting
        return ", ".join(authors)
    
    def generate_in_text_citation(self, citation_key: str, style: CitationStyle = None, 
                                 page: str = None) -> str:
        """Generate in-text citation."""
        style = style or self.default_style
        
        if citation_key not in self.citation_keys:
            return f"[{citation_key}?]"
        
        citation_id = self.citation_keys[citation_key]
        citation = self.citations[citation_id]
        
        templates = self.style_formatters.get(style, self.style_formatters[CitationStyle.APA])
        
        if page:
            template = templates.get("in_text_page", templates.get("in_text", ""))
        else:
            template = templates.get("in_text", "")
        
        # Format authors for in-text citation
        if style == CitationStyle.APA:
            if len(citation.authors) == 1:
                authors = citation.authors[0].split()[-1]  # Last name only
            elif len(citation.authors) == 2:
                auth1 = citation.authors[0].split()[-1]
                auth2 = citation.authors[1].split()[-1]
                authors = f"{auth1} & {auth2}"
            else:
                authors = f"{citation.authors[0].split()[-1]} et al."
        else:
            authors = citation.authors[0].split()[-1] if citation.authors else ""
        
        replacements = {
            "authors": authors,
            "year": str(citation.year),
            "page": page or "",
            "number": "1"  # Would be actual citation number in numbered styles
        }
        
        formatted = template
        for key, value in replacements.items():
            formatted = formatted.replace(f"{{{key}}}", value)
        
        return formatted
    
    def generate_bibliography(self, style: CitationStyle = None, 
                            sort_by: str = "author") -> List[str]:
        """
        Generate complete bibliography.
        
        Args:
            style: Citation style to use
            sort_by: Sort method ("author", "year", "title", "added")
            
        Returns:
            List of formatted citations
        """
        style = style or self.default_style
        
        # Get all citations
        citation_list = list(self.citations.values())
        
        # Sort citations
        if sort_by == "author":
            citation_list.sort(key=lambda c: c.authors[0] if c.authors else "")
        elif sort_by == "year":
            citation_list.sort(key=lambda c: c.year)
        elif sort_by == "title":
            citation_list.sort(key=lambda c: c.title)
        # "added" keeps original order
        
        # Format citations
        bibliography = []
        for i, citation in enumerate(citation_list, 1):
            formatted = self.format_citation(citation.id, style, i)
            bibliography.append(formatted)
        
        return bibliography
    
    def search_citations(self, query: str, fields: List[str] = None) -> List[Citation]:
        """
        Search citations by query.
        
        Args:
            query: Search query
            fields: Fields to search in (default: all text fields)
            
        Returns:
            List of matching citations
        """
        fields = fields or ["title", "authors", "venue", "abstract", "keywords"]
        query = query.lower()
        
        matching_citations = []
        
        for citation in self.citations.values():
            match = False
            
            for field in fields:
                if field == "title" and query in citation.title.lower():
                    match = True
                    break
                elif field == "authors" and any(query in author.lower() for author in citation.authors):
                    match = True
                    break
                elif field == "venue" and query in citation.venue.lower():
                    match = True
                    break
                elif field == "abstract" and citation.abstract and query in citation.abstract.lower():
                    match = True
                    break
                elif field == "keywords" and citation.keywords and any(query in kw.lower() for kw in citation.keywords):
                    match = True
                    break
            
            if match:
                matching_citations.append(citation)
        
        return matching_citations
    
    def import_citations_from_file(self, file_path: str, format_type: str = "bibtex") -> int:
        """
        Import citations from file.
        
        Args:
            file_path: Path to citation file
            format_type: File format ("bibtex", "ris", "json")
            
        Returns:
            Number of citations imported
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if format_type == "bibtex":
                return self._import_bibtex(content)
            elif format_type == "json":
                return self._import_json(content)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            raise Exception(f"Failed to import citations: {e}")
    
    def _import_bibtex(self, content: str) -> int:
        """Import citations from BibTeX content."""
        # Split content into individual entries
        entries = re.findall(r'@\w+\{[^@]*\}', content, re.DOTALL)
        
        imported_count = 0
        for entry in entries:
            try:
                citation = self.parse_bibtex_entry(entry)
                self.add_citation(citation)
                imported_count += 1
            except Exception as e:
                print(f"Error parsing BibTeX entry: {e}")
                continue
        
        return imported_count
    
    def _import_json(self, content: str) -> int:
        """Import citations from JSON content."""
        try:
            data = json.loads(content)
            imported_count = 0
            
            for item in data:
                try:
                    citation = Citation(**item)
                    self.add_citation(citation)
                    imported_count += 1
                except Exception as e:
                    print(f"Error importing JSON citation: {e}")
                    continue
            
            return imported_count
            
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON format: {e}")
    
    def export_citations(self, format_type: str = "bibtex", 
                        citation_ids: List[str] = None) -> str:
        """
        Export citations to specified format.
        
        Args:
            format_type: Export format ("bibtex", "json", "ris")
            citation_ids: Specific citations to export (None for all)
            
        Returns:
            Exported citations as string
        """
        if citation_ids:
            citations_to_export = [self.citations[id] for id in citation_ids if id in self.citations]
        else:
            citations_to_export = list(self.citations.values())
        
        if format_type == "bibtex":
            return self._export_bibtex(citations_to_export)
        elif format_type == "json":
            return self._export_json(citations_to_export)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_bibtex(self, citations: List[Citation]) -> str:
        """Export citations to BibTeX format."""
        bibtex_entries = []
        
        for citation in citations:
            entry_type = citation.publication_type or "article"
            entry_key = citation.id
            
            fields = []
            fields.append(f"  title = {{{citation.title}}}")
            
            if citation.authors:
                authors_str = " and ".join(citation.authors)
                fields.append(f"  author = {{{authors_str}}}")
            
            fields.append(f"  year = {{{citation.year}}}")
            
            if citation.venue:
                if citation.publication_type in ["inproceedings", "incollection"]:
                    fields.append(f"  booktitle = {{{citation.venue}}}")
                else:
                    fields.append(f"  journal = {{{citation.venue}}}")
            
            if citation.volume:
                fields.append(f"  volume = {{{citation.volume}}}")
            if citation.issue:
                fields.append(f"  number = {{{citation.issue}}}")
            if citation.pages:
                fields.append(f"  pages = {{{citation.pages}}}")
            if citation.doi:
                fields.append(f"  doi = {{{citation.doi}}}")
            if citation.url:
                fields.append(f"  url = {{{citation.url}}}")
            if citation.publisher:
                fields.append(f"  publisher = {{{citation.publisher}}}")
            
            entry = f"@{entry_type}{{{entry_key},\\n" + ",\\n".join(fields) + "\\n}"
            bibtex_entries.append(entry)
        
        return "\\n\\n".join(bibtex_entries)
    
    def _export_json(self, citations: List[Citation]) -> str:
        """Export citations to JSON format."""
        citation_dicts = []
        
        for citation in citations:
            citation_dict = {
                "id": citation.id,
                "title": citation.title,
                "authors": citation.authors,
                "year": citation.year,
                "publication_type": citation.publication_type,
                "venue": citation.venue
            }
            
            # Add optional fields
            optional_fields = ["volume", "issue", "pages", "doi", "url", "publisher", "location"]
            for field in optional_fields:
                value = getattr(citation, field)
                if value:
                    citation_dict[field] = value
            
            citation_dicts.append(citation_dict)
        
        return json.dumps(citation_dicts, indent=2)
    
    def get_citation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the citation collection."""
        if not self.citations:
            return {"total_citations": 0}
        
        total = len(self.citations)
        
        # Count by publication type
        by_type = {}
        by_year = {}
        by_venue = {}
        
        for citation in self.citations.values():
            # By type
            pub_type = citation.publication_type or "unknown"
            by_type[pub_type] = by_type.get(pub_type, 0) + 1
            
            # By year
            year = citation.year
            by_year[year] = by_year.get(year, 0) + 1
            
            # By venue
            venue = citation.venue or "unknown"
            by_venue[venue] = by_venue.get(venue, 0) + 1
        
        return {
            "total_citations": total,
            "by_publication_type": by_type,
            "by_year": dict(sorted(by_year.items())),
            "by_venue": dict(sorted(by_venue.items(), key=lambda x: x[1], reverse=True)[:10]),
            "year_range": (min(by_year.keys()), max(by_year.keys())) if by_year else (0, 0),
            "most_common_type": max(by_type.items(), key=lambda x: x[1])[0] if by_type else None
        }