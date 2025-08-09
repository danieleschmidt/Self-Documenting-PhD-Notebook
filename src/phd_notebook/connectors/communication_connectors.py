"""
Communication platform connectors for Slack, Email, Discord etc.
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncIterator
from urllib.parse import urlparse

from .base import DataConnector


class SlackConnector(DataConnector):
    """Connector for Slack workspaces."""
    
    def __init__(self, workspace: str, token: str = None):
        super().__init__()
        self.workspace = workspace
        self.token = token
        self.channels: Dict[str, str] = {}
        
    async def connect(self) -> bool:
        """Connect to Slack workspace."""
        if not self.token:
            self.logger.warning("No Slack token provided - using mock mode")
            return True
            
        try:
            # In a real implementation, this would use the Slack SDK
            # For now, we'll simulate a connection
            self.logger.info(f"Connected to Slack workspace: {self.workspace}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Slack: {e}")
            return False
    
    async def fetch_data(self, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Fetch messages from Slack channels."""
        channels = kwargs.get('channels', ['general'])
        since = kwargs.get('since', datetime.now().replace(hour=0, minute=0, second=0))
        
        for channel in channels:
            async for message in self._fetch_channel_messages(channel, since):
                yield {
                    'type': 'slack_message',
                    'workspace': self.workspace,
                    'channel': channel,
                    'timestamp': message.get('timestamp'),
                    'user': message.get('user'),
                    'text': message.get('text'),
                    'reactions': message.get('reactions', []),
                    'thread_ts': message.get('thread_ts'),
                    'raw': message
                }
    
    async def _fetch_channel_messages(self, channel: str, since: datetime) -> AsyncIterator[Dict[str, Any]]:
        """Fetch messages from a specific channel."""
        # Mock implementation - in reality would use Slack API
        mock_messages = [
            {
                'timestamp': datetime.now().isoformat(),
                'user': 'researcher1',
                'text': 'Hey everyone, just published our paper on arXiv!',
                'reactions': [{'name': 'tada', 'count': 3}]
            },
            {
                'timestamp': datetime.now().isoformat(),
                'user': 'advisor',
                'text': 'Great work on the experiment results. Can you share the methodology?',
                'reactions': []
            }
        ]
        
        for message in mock_messages:
            yield message
    
    def extract_research_content(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract research-relevant content from Slack messages."""
        text = data.get('text', '')
        
        # Check for research indicators
        research_keywords = [
            'paper', 'research', 'experiment', 'results', 'data', 'analysis',
            'hypothesis', 'methodology', 'findings', 'publication', 'arxiv',
            'conference', 'journal', 'citation', 'dataset'
        ]
        
        if any(keyword in text.lower() for keyword in research_keywords):
            return {
                'content_type': 'research_discussion',
                'text': text,
                'participants': [data.get('user')],
                'timestamp': data.get('timestamp'),
                'source': f"Slack #{data.get('channel')}",
                'urls': self._extract_urls(text),
                'mentions': self._extract_mentions(text),
                'suggested_tags': self._suggest_tags(text)
            }
        
        return None
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, text)
    
    def _extract_mentions(self, text: str) -> List[str]:
        """Extract @mentions from text."""
        return re.findall(r'@(\w+)', text)
    
    def _suggest_tags(self, text: str) -> List[str]:
        """Suggest relevant tags based on content."""
        text_lower = text.lower()
        tags = []
        
        tag_keywords = {
            'machine-learning': ['ml', 'machine learning', 'neural', 'model'],
            'data-analysis': ['data', 'analysis', 'statistics', 'visualization'],
            'experiment': ['experiment', 'test', 'trial', 'study'],
            'collaboration': ['meeting', 'discuss', 'collaborate', 'team'],
            'publication': ['paper', 'publish', 'arxiv', 'journal', 'conference']
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(tag)
        
        return tags


class EmailConnector(DataConnector):
    """Connector for email services."""
    
    def __init__(self, email_address: str, imap_server: str = None, password: str = None):
        super().__init__()
        self.email_address = email_address
        self.imap_server = imap_server
        self.password = password
        
    async def connect(self) -> bool:
        """Connect to email server."""
        if not self.imap_server or not self.password:
            self.logger.warning("Email credentials not provided - using mock mode")
            return True
            
        try:
            # In a real implementation, this would connect to IMAP server
            self.logger.info(f"Connected to email: {self.email_address}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to email: {e}")
            return False
    
    async def fetch_data(self, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Fetch emails."""
        folder = kwargs.get('folder', 'INBOX')
        since = kwargs.get('since', datetime.now().replace(hour=0, minute=0, second=0))
        research_only = kwargs.get('research_only', True)
        
        # Mock implementation
        mock_emails = [
            {
                'subject': 'Experiment Results - Review Needed',
                'from': 'collaborator@university.edu',
                'to': self.email_address,
                'date': datetime.now().isoformat(),
                'body': 'Hi, I\'ve attached the experiment results. Can you review the statistical analysis?',
                'attachments': ['results.csv', 'analysis.py']
            },
            {
                'subject': 'Paper Accepted - Congratulations!',
                'from': 'editor@journal.com',
                'to': self.email_address,
                'date': datetime.now().isoformat(),
                'body': 'Congratulations! Your paper has been accepted for publication.',
                'attachments': []
            }
        ]
        
        for email in mock_emails:
            if not research_only or self._is_research_related(email):
                yield {
                    'type': 'email',
                    'subject': email['subject'],
                    'from': email['from'],
                    'to': email['to'],
                    'date': email['date'],
                    'body': email['body'],
                    'attachments': email['attachments'],
                    'folder': folder
                }
    
    def _is_research_related(self, email: Dict[str, Any]) -> bool:
        """Check if email is research-related."""
        research_indicators = [
            'paper', 'research', 'experiment', 'data', 'analysis', 'results',
            'publication', 'conference', 'journal', 'review', 'collaboration',
            'dataset', 'methodology', 'findings'
        ]
        
        content = f"{email['subject']} {email['body']}".lower()
        return any(indicator in content for indicator in research_indicators)
    
    def extract_research_content(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract research content from emails."""
        if not self._is_research_related(data):
            return None
        
        return {
            'content_type': 'email_communication',
            'subject': data['subject'],
            'sender': data['from'],
            'content': data['body'],
            'timestamp': data['date'],
            'attachments': data.get('attachments', []),
            'suggested_tags': self._suggest_email_tags(data),
            'action_items': self._extract_action_items(data['body']),
            'deadlines': self._extract_deadlines(data['body'])
        }
    
    def _suggest_email_tags(self, email: Dict[str, Any]) -> List[str]:
        """Suggest tags for email content."""
        tags = []
        content = f"{email['subject']} {email['body']}".lower()
        
        if 'accepted' in content or 'congratulations' in content:
            tags.append('good-news')
        if 'deadline' in content or 'urgent' in content:
            tags.append('urgent')
        if 'review' in content or 'feedback' in content:
            tags.append('review-needed')
        if 'collaboration' in content or 'meeting' in content:
            tags.append('collaboration')
        if 'attached' in content or email.get('attachments'):
            tags.append('has-attachments')
        
        return tags
    
    def _extract_action_items(self, text: str) -> List[str]:
        """Extract action items from email text."""
        action_patterns = [
            r'please\s+(\w+(?:\s+\w+)*)',
            r'can you\s+(\w+(?:\s+\w+)*)',
            r'need to\s+(\w+(?:\s+\w+)*)',
            r'should\s+(\w+(?:\s+\w+)*)'
        ]
        
        actions = []
        for pattern in action_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                action = match.group(1)
                if len(action) > 3:  # Filter out very short matches
                    actions.append(action)
        
        return actions[:5]  # Return top 5 action items
    
    def _extract_deadlines(self, text: str) -> List[Dict[str, str]]:
        """Extract deadlines from email text."""
        deadline_patterns = [
            r'deadline[:\s]+(\w+\s+\d+)',
            r'due[:\s]+(\w+\s+\d+)',
            r'by\s+(\w+\s+\d+)',
            r'before\s+(\w+\s+\d+)'
        ]
        
        deadlines = []
        for pattern in deadline_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                deadline_text = match.group(1)
                deadlines.append({
                    'raw_text': deadline_text,
                    'context': match.group(0)
                })
        
        return deadlines


class DiscordConnector(DataConnector):
    """Connector for Discord servers."""
    
    def __init__(self, server_id: str, token: str = None):
        super().__init__()
        self.server_id = server_id
        self.token = token
        
    async def connect(self) -> bool:
        """Connect to Discord server."""
        if not self.token:
            self.logger.warning("No Discord token provided - using mock mode")
            return True
            
        try:
            self.logger.info(f"Connected to Discord server: {self.server_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Discord: {e}")
            return False
    
    async def fetch_data(self, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Fetch messages from Discord channels."""
        channels = kwargs.get('channels', ['general'])
        since = kwargs.get('since', datetime.now().replace(hour=0, minute=0, second=0))
        
        # Mock implementation
        for channel in channels:
            mock_messages = [
                {
                    'id': '123456789',
                    'author': 'ResearchBot',
                    'content': 'New paper alert: "Advances in Neural Networks" just dropped!',
                    'timestamp': datetime.now().isoformat(),
                    'channel': channel,
                    'reactions': [],
                    'attachments': []
                }
            ]
            
            for message in mock_messages:
                yield {
                    'type': 'discord_message',
                    'server_id': self.server_id,
                    'channel': message['channel'],
                    'message_id': message['id'],
                    'author': message['author'],
                    'content': message['content'],
                    'timestamp': message['timestamp'],
                    'reactions': message['reactions'],
                    'attachments': message['attachments']
                }
    
    def extract_research_content(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract research content from Discord messages."""
        content = data.get('content', '')
        
        # Simple research content detection
        research_keywords = ['paper', 'research', 'study', 'experiment', 'arxiv', 'publication']
        
        if any(keyword in content.lower() for keyword in research_keywords):
            return {
                'content_type': 'discord_discussion',
                'text': content,
                'author': data.get('author'),
                'timestamp': data.get('timestamp'),
                'source': f"Discord #{data.get('channel')}",
                'suggested_tags': ['discord', 'discussion']
            }
        
        return None


class TeamsConnector(DataConnector):
    """Connector for Microsoft Teams."""
    
    def __init__(self, tenant_id: str, client_id: str = None, client_secret: str = None):
        super().__init__()
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        
    async def connect(self) -> bool:
        """Connect to Microsoft Teams."""
        if not self.client_id or not self.client_secret:
            self.logger.warning("Teams credentials not provided - using mock mode")
            return True
            
        try:
            self.logger.info(f"Connected to Microsoft Teams tenant: {self.tenant_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Teams: {e}")
            return False
    
    async def fetch_data(self, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Fetch data from Teams."""
        team_id = kwargs.get('team_id')
        channel_id = kwargs.get('channel_id')
        
        # Mock implementation
        mock_messages = [
            {
                'id': 'msg-123',
                'from': {'user': {'displayName': 'Dr. Smith'}},
                'body': {'content': 'Can everyone review the experiment protocol by Friday?'},
                'createdDateTime': datetime.now().isoformat(),
                'webUrl': 'https://teams.microsoft.com/...'
            }
        ]
        
        for message in mock_messages:
            yield {
                'type': 'teams_message',
                'tenant_id': self.tenant_id,
                'message_id': message['id'],
                'author': message['from']['user']['displayName'],
                'content': message['body']['content'],
                'timestamp': message['createdDateTime'],
                'web_url': message['webUrl']
            }
    
    def extract_research_content(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract research content from Teams messages."""
        content = data.get('content', '')
        
        # Research content detection
        if any(keyword in content.lower() for keyword in ['research', 'experiment', 'data', 'analysis', 'protocol']):
            return {
                'content_type': 'teams_discussion',
                'text': content,
                'author': data.get('author'),
                'timestamp': data.get('timestamp'),
                'source': 'Microsoft Teams',
                'web_url': data.get('web_url'),
                'suggested_tags': ['teams', 'collaboration']
            }
        
        return None
    def disconnect(self) -> None:
        """Disconnect from Slack."""
        pass

EOF < /dev/null
