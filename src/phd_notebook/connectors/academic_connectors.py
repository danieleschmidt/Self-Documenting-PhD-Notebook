"""
Academic data connectors for research paper repositories and databases.
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Iterator, Any
from urllib.parse import quote

from .base import DataConnector


class ArXivConnector(DataConnector):
    """
    Connector for arXiv preprint repository.
    
    Fetches papers from arXiv API based on search queries.
    """
    
    def __init__(self, **kwargs):
        super().__init__("arxiv", **kwargs)
        self.base_url = "http://export.arxiv.org/api/query"
        self.max_results = kwargs.get('max_results', 10)
    
    def connect(self) -> bool:
        """Test connection to arXiv API."""
        try:
            import requests
            response = requests.get(f"{self.base_url}?search_query=cat:cs.AI&max_results=1", 
                                  timeout=10)
            self.is_connected = response.status_code == 200
            return self.is_connected
        except Exception as e:
            print(f"Failed to connect to arXiv: {e}")
            return False
    
    def fetch_data(self, search_query: str = "machine learning", **kwargs) -> Iterator[Dict[str, Any]]:
        """Fetch papers from arXiv."""
        if not self.is_connected:
            return
        
        try:
            import requests
            from xml.etree import ElementTree as ET
            
            # Construct query URL
            encoded_query = quote(search_query)
            url = f"{self.base_url}?search_query=all:{encoded_query}&max_results={self.max_results}"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Extract entries
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper_data = self._parse_arxiv_entry(entry)
                if paper_data:
                    yield paper_data
                    
        except Exception as e:
            print(f"Error fetching from arXiv: {e}")
    
    def _parse_arxiv_entry(self, entry) -> Dict[str, Any]:
        """Parse an arXiv entry XML element."""
        try:
            # Extract basic information
            title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else "Unknown Title"
            
            summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
            summary = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else ""
            
            # Extract authors
            authors = []
            for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                name_elem = author.find('{http://www.w3.org/2005/Atom}name')
                if name_elem is not None:
                    authors.append(name_elem.text)
            
            # Extract arXiv ID and URL
            id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
            arxiv_url = id_elem.text if id_elem is not None else ""
            arxiv_id = arxiv_url.split('/')[-1] if arxiv_url else ""
            
            # Extract publication date
            published_elem = entry.find('{http://www.w3.org/2005/Atom}published')
            published_date = published_elem.text if published_elem is not None else ""
            
            # Extract categories
            categories = []
            for category in entry.findall('{http://www.w3.org/2005/Atom}category'):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            return {
                'title': title,
                'authors': ', '.join(authors),
                'abstract': summary,
                'arxiv_id': arxiv_id,
                'arxiv_url': arxiv_url,
                'published_date': published_date,
                'categories': categories,
                'source': 'arXiv'
            }
            
        except Exception as e:
            print(f"Error parsing arXiv entry: {e}")
            return {}
    
    def _process_data_item(self, data_item: Dict[str, Any], notebook) -> None:
        """Process arXiv paper into literature note."""
        from ..agents import LiteratureAgent
        
        # Try to get the literature agent
        lit_agent = notebook.get_agent("LiteratureAgent")
        
        if lit_agent:
            # Use agent to create structured note
            try:
                year = data_item.get('published_date', '')[:4] if data_item.get('published_date') else 'Unknown'
                
                note = lit_agent.create_literature_note(
                    title=data_item.get('title', 'Unknown Title'),
                    authors=data_item.get('authors', 'Unknown Authors'),
                    year=year,
                    paper_content=data_item.get('abstract', ''),
                    doi=data_item.get('arxiv_id', ''),
                    journal='arXiv'
                )
                
                # Add arXiv-specific metadata
                note.frontmatter.metadata.update({
                    'arxiv_id': data_item.get('arxiv_id'),
                    'arxiv_url': data_item.get('arxiv_url'),
                    'categories': data_item.get('categories', []),
                    'import_date': datetime.now().isoformat()
                })
                
                if note.file_path:
                    note.save()
                
            except Exception as e:
                print(f"Error creating literature note: {e}")
                # Fallback to basic note creation
                super()._process_data_item(data_item, notebook)
        else:
            # Fallback to basic note creation
            super()._process_data_item(data_item, notebook)
    
    def disconnect(self) -> None:
        """Disconnect from arXiv (no persistent connection)."""
        self.is_connected = False


class CrossRefConnector(DataConnector):
    """
    Connector for CrossRef API to fetch DOI metadata.
    """
    
    def __init__(self, email: str = "", **kwargs):
        super().__init__("crossref", **kwargs)
        self.base_url = "https://api.crossref.org"
        self.email = email  # For polite API access
        self.session = None
    
    def connect(self) -> bool:
        """Test connection to CrossRef API."""
        try:
            import requests
            self.session = requests.Session()
            
            # Set polite headers
            if self.email:
                self.session.headers.update({
                    'User-Agent': f'PhD-Notebook-Bot/1.0 (mailto:{self.email})'
                })
            
            # Test query
            response = self.session.get(f"{self.base_url}/works?rows=1", timeout=10)
            self.is_connected = response.status_code == 200
            return self.is_connected
            
        except Exception as e:
            print(f"Failed to connect to CrossRef: {e}")
            return False
    
    def fetch_data(self, query: str = "", dois: List[str] = None, **kwargs) -> Iterator[Dict[str, Any]]:
        """Fetch metadata from CrossRef."""
        if not self.session:
            return
        
        try:
            if dois:
                # Fetch specific DOIs
                for doi in dois:
                    paper_data = self._fetch_doi(doi)
                    if paper_data:
                        yield paper_data
            else:
                # Search query
                papers = self._search_papers(query, **kwargs)
                for paper in papers:
                    yield paper
                    
        except Exception as e:
            print(f"Error fetching from CrossRef: {e}")
    
    def _fetch_doi(self, doi: str) -> Dict[str, Any]:
        """Fetch metadata for a specific DOI."""
        try:
            response = self.session.get(f"{self.base_url}/works/{doi}")
            response.raise_for_status()
            
            data = response.json()
            return self._parse_crossref_work(data['message'])
            
        except Exception as e:
            print(f"Error fetching DOI {doi}: {e}")
            return {}
    
    def _search_papers(self, query: str, rows: int = 10) -> List[Dict[str, Any]]:
        """Search papers by query."""
        try:
            params = {
                'query': query,
                'rows': rows,
                'sort': 'score',
                'order': 'desc'
            }
            
            response = self.session.get(f"{self.base_url}/works", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            papers = []
            for item in data['message']['items']:
                paper_data = self._parse_crossref_work(item)
                if paper_data:
                    papers.append(paper_data)
            
            return papers
            
        except Exception as e:
            print(f"Error searching CrossRef: {e}")
            return []
    
    def _parse_crossref_work(self, work: Dict[str, Any]) -> Dict[str, Any]:
        """Parse CrossRef work item."""
        try:
            # Extract title
            titles = work.get('title', [])
            title = titles[0] if titles else "Unknown Title"
            
            # Extract authors
            authors = []
            for author in work.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                name = f"{given} {family}".strip()
                if name:
                    authors.append(name)
            
            # Extract publication info
            published = work.get('published-print', work.get('published-online', {}))
            year = ''
            if 'date-parts' in published and published['date-parts']:
                year = str(published['date-parts'][0][0])
            
            # Extract journal
            container_titles = work.get('container-title', [])
            journal = container_titles[0] if container_titles else ''
            
            # Extract abstract (if available)
            abstract = work.get('abstract', '')
            
            return {
                'title': title,
                'authors': ', '.join(authors),
                'year': year,
                'journal': journal,
                'doi': work.get('DOI', ''),
                'abstract': abstract,
                'url': work.get('URL', ''),
                'type': work.get('type', ''),
                'source': 'CrossRef'
            }
            
        except Exception as e:
            print(f"Error parsing CrossRef work: {e}")
            return {}
    
    def disconnect(self) -> None:
        """Disconnect from CrossRef."""
        if self.session:
            self.session.close()
        self.is_connected = False


class SemanticScholarConnector(DataConnector):
    """
    Connector for Semantic Scholar API.
    """
    
    def __init__(self, api_key: str = "", **kwargs):
        super().__init__("semantic_scholar", **kwargs)
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.api_key = api_key
        self.session = None
    
    def connect(self) -> bool:
        """Test connection to Semantic Scholar API."""
        try:
            import requests
            self.session = requests.Session()
            
            if self.api_key:
                self.session.headers.update({
                    'x-api-key': self.api_key
                })
            
            # Test query
            response = self.session.get(f"{self.base_url}/paper/search?query=machine learning&limit=1", 
                                      timeout=10)
            self.is_connected = response.status_code == 200
            return self.is_connected
            
        except Exception as e:
            print(f"Failed to connect to Semantic Scholar: {e}")
            return False
    
    def fetch_data(self, query: str = "machine learning", limit: int = 10, **kwargs) -> Iterator[Dict[str, Any]]:
        """Fetch papers from Semantic Scholar."""
        if not self.session:
            return
        
        try:
            params = {
                'query': query,
                'limit': limit,
                'fields': 'title,authors,year,abstract,venue,citationCount,referenceCount,url'
            }
            
            response = self.session.get(f"{self.base_url}/paper/search", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            for paper in data.get('data', []):
                paper_data = self._parse_semantic_scholar_paper(paper)
                if paper_data:
                    yield paper_data
                    
        except Exception as e:
            print(f"Error fetching from Semantic Scholar: {e}")
    
    def _parse_semantic_scholar_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Semantic Scholar paper data."""
        try:
            # Extract authors
            authors = []
            for author in paper.get('authors', []):
                name = author.get('name', '')
                if name:
                    authors.append(name)
            
            return {
                'title': paper.get('title', 'Unknown Title'),
                'authors': ', '.join(authors),
                'year': str(paper.get('year', '')),
                'abstract': paper.get('abstract', ''),
                'venue': paper.get('venue', ''),
                'citation_count': paper.get('citationCount', 0),
                'reference_count': paper.get('referenceCount', 0),
                'url': paper.get('url', ''),
                'paper_id': paper.get('paperId', ''),
                'source': 'Semantic Scholar'
            }
            
        except Exception as e:
            print(f"Error parsing Semantic Scholar paper: {e}")
            return {}
    
    def disconnect(self) -> None:
        """Disconnect from Semantic Scholar."""
        if self.session:
            self.session.close()
        self.is_connected = False