"""
Cloud storage and service connectors for Google Drive, Dropbox, OneDrive etc.
"""

import asyncio
import json
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncIterator

from .base import DataConnector


class GoogleDriveConnector(DataConnector):
    """Connector for Google Drive."""
    
    def __init__(self, credentials_path: str = None, service_account_key: str = None):
        super().__init__()
        self.credentials_path = credentials_path
        self.service_account_key = service_account_key
        self.drive_service = None
        
    async def connect(self) -> bool:
        """Connect to Google Drive API."""
        if not self.credentials_path and not self.service_account_key:
            self.logger.warning("Google Drive credentials not provided - using mock mode")
            return True
            
        try:
            # In a real implementation, this would use the Google Drive API
            self.logger.info("Connected to Google Drive")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Google Drive: {e}")
            return False
    
    async def fetch_data(self, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Fetch files and documents from Google Drive."""
        folder_id = kwargs.get('folder_id')
        file_types = kwargs.get('file_types', ['application/pdf', 'text/plain', 'application/vnd.google-apps.document'])
        modified_since = kwargs.get('modified_since')
        
        # Mock implementation
        mock_files = [
            {
                'id': 'doc123',
                'name': 'Research Proposal Draft.docx',
                'mimeType': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'modifiedTime': datetime.now().isoformat(),
                'size': '45632',
                'webViewLink': 'https://drive.google.com/file/d/doc123/view',
                'parents': ['folder789']
            },
            {
                'id': 'pdf456',
                'name': 'Literature Review - Neural Networks.pdf',
                'mimeType': 'application/pdf',
                'modifiedTime': datetime.now().isoformat(),
                'size': '2048576',
                'webViewLink': 'https://drive.google.com/file/d/pdf456/view',
                'parents': ['folder789']
            },
            {
                'id': 'sheet789',
                'name': 'Experiment Results',
                'mimeType': 'application/vnd.google-apps.spreadsheet',
                'modifiedTime': datetime.now().isoformat(),
                'webViewLink': 'https://docs.google.com/spreadsheets/d/sheet789',
                'parents': ['folder789']
            }
        ]
        
        for file_data in mock_files:
            if not file_types or file_data['mimeType'] in file_types:
                yield {
                    'type': 'google_drive_file',
                    'file_id': file_data['id'],
                    'name': file_data['name'],
                    'mime_type': file_data['mimeType'],
                    'size': file_data.get('size'),
                    'modified_time': file_data['modifiedTime'],
                    'web_link': file_data['webViewLink'],
                    'parents': file_data.get('parents', []),
                    'content': await self._get_file_content(file_data['id'], file_data['mimeType'])
                }
    
    async def _get_file_content(self, file_id: str, mime_type: str) -> Optional[str]:
        """Get content of a file."""
        # Mock implementation - in reality would download and extract content
        if 'document' in mime_type or 'text' in mime_type:
            return f"Content of file {file_id}. This document contains research-related information."
        elif 'pdf' in mime_type:
            return f"PDF content from {file_id}. Abstract: This paper presents novel findings in research."
        elif 'spreadsheet' in mime_type:
            return f"Spreadsheet data from {file_id}. Contains experimental results and analysis."
        
        return None
    
    def extract_research_content(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract research content from Google Drive files."""
        name = data.get('name', '')
        content = data.get('content', '')
        
        # Check if file is research-related
        research_indicators = [
            'research', 'experiment', 'data', 'analysis', 'paper', 'draft',
            'literature', 'methodology', 'results', 'findings', 'study'
        ]
        
        name_lower = name.lower()
        content_lower = content.lower() if content else ''
        
        is_research = any(indicator in name_lower or indicator in content_lower 
                         for indicator in research_indicators)
        
        if is_research:
            return {
                'content_type': 'drive_document',
                'title': name,
                'content': content,
                'file_type': self._get_file_category(data.get('mime_type', '')),
                'source': 'Google Drive',
                'url': data.get('web_link'),
                'modified_time': data.get('modified_time'),
                'suggested_tags': self._suggest_drive_tags(name, content),
                'file_size': data.get('size')
            }
        
        return None
    
    def _get_file_category(self, mime_type: str) -> str:
        """Categorize file type."""
        if 'document' in mime_type or 'text' in mime_type:
            return 'document'
        elif 'pdf' in mime_type:
            return 'pdf'
        elif 'spreadsheet' in mime_type:
            return 'spreadsheet'
        elif 'presentation' in mime_type:
            return 'presentation'
        else:
            return 'other'
    
    def _suggest_drive_tags(self, name: str, content: str) -> List[str]:
        """Suggest tags based on file name and content."""
        tags = []
        text = f"{name} {content}".lower()
        
        tag_mapping = {
            'draft': ['draft', 'document', 'text/plain', 'word'],
            'literature': ['literature', 'review', 'paper', 'pdf'],
            'experiment': ['experiment', 'data', 'results', 'analysis'],
            'presentation': ['presentation', 'slides', 'talk'],
            'proposal': ['proposal', 'grant', 'funding'],
            'methodology': ['methodology', 'protocol', 'procedure'],
        }
        
        for tag, keywords in tag_mapping.items():
            if any(keyword in text for keyword in keywords):
                tags.append(tag)
        
        return tags


class DropboxConnector(DataConnector):
    """Connector for Dropbox."""
    
    def __init__(self, access_token: str = None):
        super().__init__()
        self.access_token = access_token
        
    async def connect(self) -> bool:
        """Connect to Dropbox API."""
        if not self.access_token:
            self.logger.warning("Dropbox access token not provided - using mock mode")
            return True
            
        try:
            self.logger.info("Connected to Dropbox")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Dropbox: {e}")
            return False
    
    async def fetch_data(self, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Fetch files from Dropbox."""
        folder_path = kwargs.get('folder_path', '')
        recursive = kwargs.get('recursive', True)
        file_extensions = kwargs.get('file_extensions', ['.pdf', '.docx', '.txt', '.md'])
        
        # Mock implementation
        mock_files = [
            {
                'name': 'Research Notes.md',
                'path_lower': '/research/research notes.md',
                'client_modified': datetime.now().isoformat(),
                'size': 12345,
                'content_hash': 'abc123'
            },
            {
                'name': 'Experiment Data.csv',
                'path_lower': '/research/data/experiment data.csv',
                'client_modified': datetime.now().isoformat(),
                'size': 67890,
                'content_hash': 'def456'
            }
        ]
        
        for file_data in mock_files:
            file_ext = Path(file_data['name']).suffix.lower()
            if not file_extensions or file_ext in file_extensions:
                yield {
                    'type': 'dropbox_file',
                    'name': file_data['name'],
                    'path': file_data['path_lower'],
                    'size': file_data['size'],
                    'modified_time': file_data['client_modified'],
                    'content_hash': file_data['content_hash'],
                    'content': await self._get_dropbox_content(file_data['path_lower'])
                }
    
    async def _get_dropbox_content(self, path: str) -> Optional[str]:
        """Get content from Dropbox file."""
        # Mock implementation
        if path.endswith('.md'):
            return "# Research Notes\n\nThis is a markdown file containing research notes."
        elif path.endswith('.csv'):
            return "experiment_id,condition,result\n1,control,0.85\n2,treatment,0.92"
        
        return None
    
    def extract_research_content(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract research content from Dropbox files."""
        name = data.get('name', '')
        content = data.get('content', '')
        path = data.get('path', '')
        
        # Check if in research-related folder or has research-related name
        research_paths = ['research', 'experiments', 'papers', 'literature', 'data']
        research_names = ['research', 'experiment', 'data', 'analysis', 'notes']
        
        path_lower = path.lower()
        name_lower = name.lower()
        
        is_research = (any(rp in path_lower for rp in research_paths) or
                      any(rn in name_lower for rn in research_names))
        
        if is_research:
            return {
                'content_type': 'dropbox_file',
                'title': name,
                'content': content,
                'file_path': path,
                'source': 'Dropbox',
                'modified_time': data.get('modified_time'),
                'file_size': data.get('size'),
                'suggested_tags': self._suggest_dropbox_tags(name, path, content)
            }
        
        return None
    
    def _suggest_dropbox_tags(self, name: str, path: str, content: str) -> List[str]:
        """Suggest tags for Dropbox files."""
        tags = ['dropbox']
        text = f"{name} {path} {content}".lower()
        
        if 'data' in text or '.csv' in name or '.xlsx' in name:
            tags.append('data')
        if 'notes' in text or '.md' in name or '.txt' in name:
            tags.append('notes')
        if 'paper' in text or 'draft' in text or '.docx' in name:
            tags.append('document')
        if 'experiment' in text:
            tags.append('experiment')
        
        return tags


class OneDriveConnector(DataConnector):
    """Connector for Microsoft OneDrive."""
    
    def __init__(self, client_id: str = None, client_secret: str = None):
        super().__init__()
        self.client_id = client_id
        self.client_secret = client_secret
        
    async def connect(self) -> bool:
        """Connect to OneDrive API."""
        if not self.client_id or not self.client_secret:
            self.logger.warning("OneDrive credentials not provided - using mock mode")
            return True
            
        try:
            self.logger.info("Connected to OneDrive")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to OneDrive: {e}")
            return False
    
    async def fetch_data(self, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Fetch files from OneDrive."""
        folder_id = kwargs.get('folder_id')
        file_types = kwargs.get('file_types', ['.pdf', '.docx', '.xlsx', '.pptx'])
        
        # Mock implementation
        mock_files = [
            {
                'id': 'file123',
                'name': 'Research Presentation.pptx',
                'lastModifiedDateTime': datetime.now().isoformat(),
                'size': 5432100,
                'webUrl': 'https://onedrive.live.com/file123'
            }
        ]
        
        for file_data in mock_files:
            yield {
                'type': 'onedrive_file',
                'file_id': file_data['id'],
                'name': file_data['name'],
                'size': file_data['size'],
                'modified_time': file_data['lastModifiedDateTime'],
                'web_url': file_data['webUrl'],
                'content': await self._get_onedrive_content(file_data['id'])
            }
    
    async def _get_onedrive_content(self, file_id: str) -> Optional[str]:
        """Get content from OneDrive file."""
        # Mock implementation
        return f"Content from OneDrive file {file_id}"
    
    def extract_research_content(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract research content from OneDrive files."""
        name = data.get('name', '')
        
        research_indicators = ['research', 'experiment', 'data', 'analysis', 'presentation', 'paper']
        
        if any(indicator in name.lower() for indicator in research_indicators):
            return {
                'content_type': 'onedrive_file',
                'title': name,
                'content': data.get('content'),
                'source': 'OneDrive',
                'web_url': data.get('web_url'),
                'modified_time': data.get('modified_time'),
                'file_size': data.get('size'),
                'suggested_tags': ['onedrive', 'cloud-storage']
            }
        
        return None


class GitHubConnector(DataConnector):
    """Connector for GitHub repositories."""
    
    def __init__(self, token: str = None):
        super().__init__()
        self.token = token
        
    async def connect(self) -> bool:
        """Connect to GitHub API."""
        if not self.token:
            self.logger.warning("GitHub token not provided - using mock mode")
            return True
            
        try:
            self.logger.info("Connected to GitHub API")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to GitHub: {e}")
            return False
    
    async def fetch_data(self, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Fetch repositories and content from GitHub."""
        username = kwargs.get('username')
        repo_name = kwargs.get('repo_name')
        include_issues = kwargs.get('include_issues', True)
        include_commits = kwargs.get('include_commits', True)
        
        # Mock implementation
        if include_commits:
            mock_commits = [
                {
                    'sha': 'abc123',
                    'commit': {
                        'message': 'Add experiment analysis code',
                        'author': {'name': 'Researcher', 'date': datetime.now().isoformat()}
                    },
                    'html_url': 'https://github.com/user/repo/commit/abc123'
                }
            ]
            
            for commit in mock_commits:
                yield {
                    'type': 'github_commit',
                    'repository': f"{username}/{repo_name}",
                    'sha': commit['sha'],
                    'message': commit['commit']['message'],
                    'author': commit['commit']['author']['name'],
                    'date': commit['commit']['author']['date'],
                    'url': commit['html_url']
                }
        
        if include_issues:
            mock_issues = [
                {
                    'number': 42,
                    'title': 'Bug in data processing pipeline',
                    'body': 'The preprocessing function fails with large datasets',
                    'state': 'open',
                    'created_at': datetime.now().isoformat(),
                    'html_url': 'https://github.com/user/repo/issues/42'
                }
            ]
            
            for issue in mock_issues:
                yield {
                    'type': 'github_issue',
                    'repository': f"{username}/{repo_name}",
                    'number': issue['number'],
                    'title': issue['title'],
                    'body': issue['body'],
                    'state': issue['state'],
                    'created_at': issue['created_at'],
                    'url': issue['html_url']
                }
    
    def extract_research_content(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract research content from GitHub data."""
        data_type = data.get('type')
        
        if data_type == 'github_commit':
            message = data.get('message', '')
            research_keywords = ['experiment', 'analysis', 'data', 'model', 'research', 'paper']
            
            if any(keyword in message.lower() for keyword in research_keywords):
                return {
                    'content_type': 'code_commit',
                    'title': f"Commit: {message}",
                    'content': f"Code commit by {data.get('author')}: {message}",
                    'source': f"GitHub ({data.get('repository')})",
                    'url': data.get('url'),
                    'timestamp': data.get('date'),
                    'suggested_tags': ['github', 'code', 'development']
                }
        
        elif data_type == 'github_issue':
            title = data.get('title', '')
            body = data.get('body', '')
            
            return {
                'content_type': 'development_issue',
                'title': title,
                'content': body,
                'source': f"GitHub Issues ({data.get('repository')})",
                'url': data.get('url'),
                'timestamp': data.get('created_at'),
                'status': data.get('state'),
                'suggested_tags': ['github', 'issue', 'development']
            }
        
        return None