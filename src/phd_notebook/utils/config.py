"""
Configuration management for PhD notebook.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from ..utils.exceptions import ConfigError


@dataclass
class AIConfig:
    """AI provider configuration."""
    provider: str = "auto"
    api_key: Optional[str] = None
    model: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7


@dataclass
class NotebookConfig:
    """Notebook configuration."""
    auto_save: bool = True
    backup_enabled: bool = True
    auto_tagging: bool = True
    smart_linking: bool = True
    template_validation: bool = True


@dataclass
class SecurityConfig:
    """Security configuration."""
    encrypt_sensitive: bool = True
    pii_detection: bool = True
    access_logging: bool = True
    sanitize_exports: bool = True


@dataclass
class PHDConfig:
    """Main configuration class."""
    ai: AIConfig
    notebook: NotebookConfig
    security: SecurityConfig
    version: str = "1.0.0"
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> 'PHDConfig':
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            return cls(
                ai=AIConfig(**data.get('ai', {})),
                notebook=NotebookConfig(**data.get('notebook', {})),
                security=SecurityConfig(**data.get('security', {})),
                version=data.get('version', '1.0.0')
            )
        except Exception as e:
            raise ConfigError(f"Failed to load config from {config_path}: {e}")
    
    @classmethod
    def load_from_env(cls) -> 'PHDConfig':
        """Load configuration from environment variables."""
        return cls(
            ai=AIConfig(
                provider=os.getenv('PHD_AI_PROVIDER', 'auto'),
                api_key=os.getenv('PHD_AI_API_KEY'),
                model=os.getenv('PHD_AI_MODEL'),
                max_tokens=int(os.getenv('PHD_AI_MAX_TOKENS', 1000)),
                temperature=float(os.getenv('PHD_AI_TEMPERATURE', 0.7))
            ),
            notebook=NotebookConfig(
                auto_save=os.getenv('PHD_AUTO_SAVE', 'true').lower() == 'true',
                backup_enabled=os.getenv('PHD_BACKUP', 'true').lower() == 'true',
                auto_tagging=os.getenv('PHD_AUTO_TAG', 'true').lower() == 'true',
                smart_linking=os.getenv('PHD_SMART_LINK', 'true').lower() == 'true',
                template_validation=os.getenv('PHD_TEMPLATE_VALIDATION', 'true').lower() == 'true'
            ),
            security=SecurityConfig(
                encrypt_sensitive=os.getenv('PHD_ENCRYPT', 'true').lower() == 'true',
                pii_detection=os.getenv('PHD_PII_DETECT', 'true').lower() == 'true',
                access_logging=os.getenv('PHD_ACCESS_LOG', 'true').lower() == 'true',
                sanitize_exports=os.getenv('PHD_SANITIZE', 'true').lower() == 'true'
            ),
            version=os.getenv('PHD_VERSION', '1.0.0')
        )
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to file."""
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(asdict(self), f, indent=2)
                
        except Exception as e:
            raise ConfigError(f"Failed to save config to {config_path}: {e}")
    
    def get_config_dir(self) -> Path:
        """Get configuration directory."""
        home = Path.home()
        config_dir = home / '.config' / 'phd-notebook'
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self._config: Optional[PHDConfig] = None
    
    @property
    def config(self) -> PHDConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def load_config(self) -> PHDConfig:
        """Load configuration from various sources."""
        # Priority order: file -> env -> defaults
        
        if self.config_path and self.config_path.exists():
            return PHDConfig.load_from_file(self.config_path)
        
        # Check default config location
        default_config = Path.home() / '.config' / 'phd-notebook' / 'config.json'
        if default_config.exists():
            return PHDConfig.load_from_file(default_config)
        
        # Load from environment
        return PHDConfig.load_from_env()
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return validation results."""
        results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        config = self.config
        
        # AI configuration validation
        if config.ai.provider not in ['auto', 'openai', 'anthropic', 'mock']:
            results['errors'].append(f"Invalid AI provider: {config.ai.provider}")
            results['valid'] = False
        
        if config.ai.provider in ['openai', 'anthropic'] and not config.ai.api_key:
            results['warnings'].append(f"No API key provided for {config.ai.provider}")
        
        # Notebook configuration validation
        if config.notebook.auto_save and not config.notebook.backup_enabled:
            results['warnings'].append("Auto-save enabled without backup - data loss risk")
        
        # Security configuration validation
        if not config.security.encrypt_sensitive:
            results['warnings'].append("Sensitive data encryption disabled - security risk")
        
        return results
    
    def create_default_config(self, path: Optional[Path] = None) -> Path:
        """Create default configuration file."""
        if path is None:
            path = Path.home() / '.config' / 'phd-notebook' / 'config.json'
        
        config = PHDConfig(
            ai=AIConfig(),
            notebook=NotebookConfig(),
            security=SecurityConfig()
        )
        
        config.save_to_file(path)
        return path
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        config = self.config
        
        # Update AI config
        if 'ai' in updates:
            ai_updates = updates['ai']
            for key, value in ai_updates.items():
                if hasattr(config.ai, key):
                    setattr(config.ai, key, value)
        
        # Update notebook config
        if 'notebook' in updates:
            notebook_updates = updates['notebook']
            for key, value in notebook_updates.items():
                if hasattr(config.notebook, key):
                    setattr(config.notebook, key, value)
        
        # Update security config
        if 'security' in updates:
            security_updates = updates['security']
            for key, value in security_updates.items():
                if hasattr(config.security, key):
                    setattr(config.security, key, value)
        
        self._config = config
    
    def save_config(self, path: Optional[Path] = None) -> None:
        """Save current configuration to file."""
        if path is None:
            path = self.config_path or (Path.home() / '.config' / 'phd-notebook' / 'config.json')
        
        self.config.save_to_file(path)


# Global config manager instance
_config_manager = ConfigManager()

def get_config() -> PHDConfig:
    """Get global configuration."""
    return _config_manager.config

def set_config_path(path: Path) -> None:
    """Set configuration file path."""
    global _config_manager
    _config_manager = ConfigManager(path)

def validate_config() -> Dict[str, Any]:
    """Validate current configuration."""
    return _config_manager.validate_config()