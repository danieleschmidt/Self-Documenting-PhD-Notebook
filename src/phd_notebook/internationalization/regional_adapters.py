"""
Regional adapters for cross-platform and multi-region deployment.
"""

import os
import platform
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class DeploymentRegion(Enum):
    """Supported deployment regions."""
    US_EAST = "us-east-1"      # United States East Coast
    US_WEST = "us-west-2"      # United States West Coast  
    EU_WEST = "eu-west-1"      # Europe (Ireland)
    EU_CENTRAL = "eu-central-1" # Europe (Germany)
    ASIA_PACIFIC = "ap-southeast-1"  # Asia Pacific (Singapore)
    ASIA_NORTHEAST = "ap-northeast-1"  # Asia Pacific (Tokyo)
    CANADA = "ca-central-1"    # Canada (Central)
    AUSTRALIA = "ap-southeast-2"  # Australia (Sydney)
    BRAZIL = "sa-east-1"       # South America (São Paulo)
    ON_PREMISES = "on-premises"  # On-premises deployment


class PlatformType(Enum):
    """Supported platform types."""
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "darwin"
    UNIX = "unix"
    UNKNOWN = "unknown"


@dataclass
class RegionalConfiguration:
    """Configuration for specific regions."""
    region: DeploymentRegion
    timezone: str
    data_residency_requirements: List[str]
    encryption_requirements: Dict[str, str]
    audit_logging_required: bool = True
    backup_regions: List[DeploymentRegion] = field(default_factory=list)
    compliance_frameworks: List[str] = field(default_factory=list)
    local_storage_path: Optional[str] = None
    cloud_storage_config: Dict[str, str] = field(default_factory=dict)


class RegionalAdapter:
    """Adapts PhD notebook functionality for different regions and platforms."""
    
    def __init__(self, target_region: Optional[DeploymentRegion] = None):
        self.current_platform = self._detect_platform()
        self.target_region = target_region or self._detect_region()
        self._regional_configs = self._init_regional_configs()
        
    def _detect_platform(self) -> PlatformType:
        """Detect the current operating system platform."""
        system = platform.system().lower()
        if system == "linux":
            return PlatformType.LINUX
        elif system == "windows":
            return PlatformType.WINDOWS
        elif system == "darwin":
            return PlatformType.MACOS
        elif system in ["unix", "freebsd", "openbsd"]:
            return PlatformType.UNIX
        else:
            return PlatformType.UNKNOWN
    
    def _detect_region(self) -> DeploymentRegion:
        """Auto-detect deployment region based on environment."""
        # Check environment variables for cloud deployment
        aws_region = os.environ.get("AWS_REGION", "")
        if aws_region:
            region_map = {
                "us-east-1": DeploymentRegion.US_EAST,
                "us-west-2": DeploymentRegion.US_WEST,
                "eu-west-1": DeploymentRegion.EU_WEST,
                "eu-central-1": DeploymentRegion.EU_CENTRAL,
                "ap-southeast-1": DeploymentRegion.ASIA_PACIFIC,
                "ap-northeast-1": DeploymentRegion.ASIA_NORTHEAST,
                "ca-central-1": DeploymentRegion.CANADA,
                "ap-southeast-2": DeploymentRegion.AUSTRALIA,
                "sa-east-1": DeploymentRegion.BRAZIL
            }
            return region_map.get(aws_region, DeploymentRegion.ON_PREMISES)
        
        # Default to on-premises for local development
        return DeploymentRegion.ON_PREMISES
    
    def _init_regional_configs(self) -> Dict[DeploymentRegion, RegionalConfiguration]:
        """Initialize regional configurations."""
        return {
            DeploymentRegion.US_EAST: RegionalConfiguration(
                region=DeploymentRegion.US_EAST,
                timezone="America/New_York",
                data_residency_requirements=["US"],
                encryption_requirements={
                    "data_at_rest": "AES-256",
                    "data_in_transit": "TLS 1.3",
                    "key_management": "AWS KMS"
                },
                backup_regions=[DeploymentRegion.US_WEST],
                compliance_frameworks=["SOC2", "FedRAMP", "CCPA"],
                cloud_storage_config={
                    "provider": "AWS",
                    "bucket_region": "us-east-1",
                    "storage_class": "STANDARD"
                }
            ),
            DeploymentRegion.EU_WEST: RegionalConfiguration(
                region=DeploymentRegion.EU_WEST,
                timezone="Europe/Dublin",
                data_residency_requirements=["EU", "EEA"],
                encryption_requirements={
                    "data_at_rest": "AES-256-GCM",
                    "data_in_transit": "TLS 1.3",
                    "key_management": "EU-only keys"
                },
                backup_regions=[DeploymentRegion.EU_CENTRAL],
                compliance_frameworks=["GDPR", "ISO 27001", "SOC2"],
                cloud_storage_config={
                    "provider": "AWS",
                    "bucket_region": "eu-west-1",
                    "storage_class": "STANDARD_IA"
                }
            ),
            DeploymentRegion.ASIA_PACIFIC: RegionalConfiguration(
                region=DeploymentRegion.ASIA_PACIFIC,
                timezone="Asia/Singapore",
                data_residency_requirements=["APAC"],
                encryption_requirements={
                    "data_at_rest": "AES-256",
                    "data_in_transit": "TLS 1.3",
                    "key_management": "Regional KMS"
                },
                backup_regions=[DeploymentRegion.ASIA_NORTHEAST],
                compliance_frameworks=["PDPA", "SOC2", "ISO 27001"],
                cloud_storage_config={
                    "provider": "AWS",
                    "bucket_region": "ap-southeast-1",
                    "storage_class": "STANDARD"
                }
            ),
            DeploymentRegion.ON_PREMISES: RegionalConfiguration(
                region=DeploymentRegion.ON_PREMISES,
                timezone="UTC",
                data_residency_requirements=["Local"],
                encryption_requirements={
                    "data_at_rest": "AES-256",
                    "data_in_transit": "TLS 1.2+",
                    "key_management": "Local"
                },
                backup_regions=[],
                compliance_frameworks=["Local policies"],
                local_storage_path="~/.phd_notebook/data",
                cloud_storage_config={}
            )
        }
    
    def get_regional_config(self) -> RegionalConfiguration:
        """Get configuration for the target region."""
        return self._regional_configs.get(self.target_region, 
                                        self._regional_configs[DeploymentRegion.ON_PREMISES])
    
    def get_storage_path(self) -> Path:
        """Get the appropriate storage path for the current platform and region."""
        config = self.get_regional_config()
        
        if config.local_storage_path:
            base_path = Path(config.local_storage_path).expanduser()
        else:
            # Platform-specific default paths
            if self.current_platform == PlatformType.WINDOWS:
                base_path = Path(os.environ.get("APPDATA", "~")) / "PhDNotebook"
            elif self.current_platform == PlatformType.MACOS:
                base_path = Path("~/Library/Application Support/PhDNotebook").expanduser()
            else:  # Linux/Unix
                base_path = Path("~/.phd_notebook").expanduser()
        
        # Create directory if it doesn't exist
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path
    
    def get_temp_path(self) -> Path:
        """Get platform-appropriate temporary directory."""
        if self.current_platform == PlatformType.WINDOWS:
            temp_base = Path(os.environ.get("TEMP", "C:\\temp"))
        else:
            temp_base = Path("/tmp")
        
        temp_path = temp_base / "phd_notebook"
        temp_path.mkdir(parents=True, exist_ok=True)
        return temp_path
    
    def get_encryption_config(self) -> Dict[str, str]:
        """Get encryption configuration for the region."""
        config = self.get_regional_config()
        return config.encryption_requirements
    
    def validate_data_residency(self, data_location: str) -> bool:
        """Validate that data location meets residency requirements."""
        config = self.get_regional_config()
        requirements = config.data_residency_requirements
        
        # If no specific requirements, allow any location
        if not requirements:
            return True
        
        # Check if location meets any of the requirements
        for requirement in requirements:
            if requirement.lower() in data_location.lower():
                return True
        
        logger.warning(f"Data location {data_location} does not meet residency requirements: {requirements}")
        return False
    
    def get_compliance_frameworks(self) -> List[str]:
        """Get applicable compliance frameworks for the region."""
        config = self.get_regional_config()
        return config.compliance_frameworks
    
    def adapt_file_paths(self, paths: Dict[str, str]) -> Dict[str, str]:
        """Adapt file paths for the current platform."""
        adapted_paths = {}
        
        for key, path_str in paths.items():
            path = Path(path_str)
            
            # Handle platform-specific path separators
            if self.current_platform == PlatformType.WINDOWS:
                # Ensure Windows-style paths
                adapted_path = str(path).replace("/", "\\")
            else:
                # Ensure Unix-style paths
                adapted_path = str(path).replace("\\", "/")
            
            adapted_paths[key] = adapted_path
        
        return adapted_paths
    
    def get_platform_capabilities(self) -> Dict[str, bool]:
        """Get capabilities available on the current platform."""
        capabilities = {
            "file_watching": True,
            "background_processing": True,
            "system_notifications": False,
            "native_encryption": True,
            "process_monitoring": True,
            "network_access": True
        }
        
        # Platform-specific capabilities
        if self.current_platform == PlatformType.LINUX:
            capabilities.update({
                "inotify_support": True,
                "systemd_integration": True,
                "docker_support": True
            })
        elif self.current_platform == PlatformType.WINDOWS:
            capabilities.update({
                "windows_service": True,
                "registry_access": True,
                "wmi_support": True
            })
        elif self.current_platform == PlatformType.MACOS:
            capabilities.update({
                "launchd_integration": True,
                "keychain_access": True,
                "fsevents_support": True
            })
        
        return capabilities
    
    def optimize_for_region(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize configuration for the target region."""
        regional_config = self.get_regional_config()
        optimized = config.copy()
        
        # Apply regional optimizations
        optimized.update({
            "timezone": regional_config.timezone,
            "encryption": regional_config.encryption_requirements,
            "backup_regions": [r.value for r in regional_config.backup_regions],
            "compliance_mode": regional_config.compliance_frameworks[0] if regional_config.compliance_frameworks else "standard"
        })
        
        # Platform-specific optimizations
        platform_optimizations = {
            PlatformType.LINUX: {
                "max_open_files": 65536,
                "use_sendfile": True,
                "enable_compression": True
            },
            PlatformType.WINDOWS: {
                "use_overlapped_io": True,
                "buffer_size": 8192,
                "enable_compression": False  # Can be CPU intensive on Windows
            },
            PlatformType.MACOS: {
                "use_kqueue": True,
                "enable_fsevents": True,
                "buffer_size": 4096
            }
        }
        
        platform_opts = platform_optimizations.get(self.current_platform, {})
        optimized.update(platform_opts)
        
        return optimized
    
    def generate_deployment_info(self) -> Dict[str, Any]:
        """Generate comprehensive deployment information."""
        config = self.get_regional_config()
        
        return {
            "deployment_info": {
                "platform": self.current_platform.value,
                "region": self.target_region.value,
                "timezone": config.timezone,
                "storage_path": str(self.get_storage_path()),
                "temp_path": str(self.get_temp_path()),
                "capabilities": self.get_platform_capabilities(),
                "compliance_frameworks": config.compliance_frameworks,
                "data_residency": config.data_residency_requirements,
                "encryption": config.encryption_requirements,
                "backup_regions": [r.value for r in config.backup_regions],
                "generated_at": datetime.now().isoformat()
            }
        }