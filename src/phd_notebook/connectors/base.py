"""
Base data connector for external integrations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Iterator
from datetime import datetime


class DataConnector(ABC):
    """
    Base class for data connectors that integrate external data sources.
    
    Connectors handle:
    - Authentication and connection management
    - Data fetching and synchronization
    - Format conversion and normalization
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.is_connected = False
        self.last_sync: Optional[datetime] = None
        
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the data source."""
        pass
    
    @abstractmethod
    def fetch_data(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """Fetch data from the source."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the data source."""
        pass
    
    def test_connection(self) -> bool:
        """Test if connection is working."""
        try:
            return self.connect()
        except Exception:
            return False
    
    def sync(self, notebook=None, **kwargs) -> int:
        """Sync data with the notebook."""
        if not self.is_connected:
            if not self.connect():
                return 0
        
        count = 0
        try:
            for data_item in self.fetch_data(**kwargs):
                if notebook:
                    self._process_data_item(data_item, notebook)
                count += 1
                
            self.last_sync = datetime.now()
            
        except Exception as e:
            print(f"Sync error for {self.name}: {e}")
        
        return count
    
    def _process_data_item(self, data_item: Dict[str, Any], notebook) -> None:
        """Process a single data item into the notebook."""
        # Override in subclasses for specific processing
        title = data_item.get('title', f"Data from {self.name}")
        content = data_item.get('content', str(data_item))
        
        notebook.create_note(
            title=title,
            content=content,
            tags=[f"#{self.name}", "#auto-imported"]
        )
    
    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        return f"{self.__class__.__name__}(name='{self.name}', status='{status}')"


# Example implementations for common data sources
class FileSystemConnector(DataConnector):
    """Simple file system connector for local data."""
    
    def __init__(self, path: str, file_pattern: str = "*.txt", **kwargs):
        super().__init__("filesystem", **kwargs)
        self.path = path
        self.file_pattern = file_pattern
    
    def connect(self) -> bool:
        from pathlib import Path
        self.is_connected = Path(self.path).exists()
        return self.is_connected
    
    def fetch_data(self, **kwargs) -> Iterator[Dict[str, Any]]:
        from pathlib import Path
        
        path = Path(self.path)
        for file_path in path.glob(self.file_pattern):
            try:
                content = file_path.read_text(encoding='utf-8')
                yield {
                    'title': file_path.stem,
                    'content': content,
                    'source_path': str(file_path),
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                }
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    def disconnect(self) -> None:
        self.is_connected = False


class WebConnector(DataConnector):
    """Simple web connector for HTTP endpoints."""
    
    def __init__(self, base_url: str, **kwargs):
        super().__init__("web", **kwargs)
        self.base_url = base_url
        self.session = None
    
    def connect(self) -> bool:
        try:
            import requests
            self.session = requests.Session()
            
            # Test connection
            response = self.session.get(self.base_url, timeout=10)
            self.is_connected = response.status_code == 200
            return self.is_connected
            
        except Exception:
            return False
    
    def fetch_data(self, endpoint: str = "", **kwargs) -> Iterator[Dict[str, Any]]:
        if not self.session:
            return
        
        try:
            url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            response = self.session.get(url, **kwargs)
            response.raise_for_status()
            
            # Assume JSON response
            data = response.json()
            
            if isinstance(data, list):
                for item in data:
                    yield item
            else:
                yield data
                
        except Exception as e:
            print(f"Error fetching from {endpoint}: {e}")
    
    def disconnect(self) -> None:
        if self.session:
            self.session.close()
        self.is_connected = False