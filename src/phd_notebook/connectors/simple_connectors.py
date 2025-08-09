"""
Simple connectors for basic functionality without abstract method issues.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Iterator
import logging

from .base import DataConnector


class SimpleSlackConnector(DataConnector):
    """Simple Slack connector for testing."""
    
    def __init__(self, workspace: str):
        super().__init__("slack", {})
        self.workspace = workspace
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """Connect to Slack workspace."""
        self.logger.info(f"Connected to Slack workspace: {self.workspace}")
        self.is_connected = True
        return True
    
    def disconnect(self) -> None:
        """Disconnect from Slack."""
        self.logger.info(f"Disconnected from Slack workspace: {self.workspace}")
        self.is_connected = False
    
    def fetch_data(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """Fetch mock data from Slack."""
        mock_data = [
            {
                'type': 'slack_message',
                'workspace': self.workspace,
                'channel': 'general',
                'user': 'researcher1',
                'text': 'Just published our paper on arXiv!',
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        for item in mock_data:
            yield item


class SimpleGoogleDriveConnector(DataConnector):
    """Simple Google Drive connector for testing."""
    
    def __init__(self):
        super().__init__("google_drive", {})
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """Connect to Google Drive."""
        self.logger.info("Connected to Google Drive")
        self.is_connected = True
        return True
    
    def disconnect(self) -> None:
        """Disconnect from Google Drive."""
        self.logger.info("Disconnected from Google Drive")
        self.is_connected = False
    
    def fetch_data(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """Fetch mock data from Google Drive."""
        mock_data = [
            {
                'type': 'drive_file',
                'name': 'Research Paper Draft.docx',
                'modified_time': datetime.now().isoformat(),
                'content': 'This is a research paper about machine learning...'
            }
        ]
        
        for item in mock_data:
            yield item