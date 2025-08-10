"""
Backup and recovery system for the PhD notebook.
"""

import os
import json
import shutil
import tarfile
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass, asdict

from .exceptions import BackupError
# from .security import SecureStorage  # Will implement later


@dataclass
class BackupMetadata:
    """Metadata for backup operations."""
    backup_id: str
    created_at: datetime
    vault_path: str
    backup_path: str
    file_count: int
    total_size: int
    checksum: str
    backup_type: str  # 'full', 'incremental'
    compression: bool = True
    encrypted: bool = False


class BackupManager:
    """Manages backup and restoration operations."""
    
    def __init__(self, vault_path: Path, backup_root: Path = None):
        self.vault_path = Path(vault_path)
        self.backup_root = backup_root or (Path.home() / '.phd-notebook' / 'backups')
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.backup_root / 'backup_metadata.json'
        # self.secure_storage = SecureStorage()  # Will implement later
        
    def create_backup(
        self, 
        backup_type: str = 'full',
        compress: bool = True,
        encrypt: bool = False,
        description: str = ""
    ) -> BackupMetadata:
        """Create a backup of the vault."""
        
        if not self.vault_path.exists():
            raise BackupError(f"Vault path does not exist: {self.vault_path}")
        
        # Generate backup ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_id = f"{backup_type}_{timestamp}"
        
        # Create backup directory
        backup_dir = self.backup_root / backup_id
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create archive
            archive_path = backup_dir / f"{backup_id}.tar"
            if compress:
                archive_path = archive_path.with_suffix('.tar.gz')
            
            file_count, total_size = self._create_archive(
                self.vault_path, 
                archive_path, 
                compress
            )
            
            # Calculate checksum
            checksum = self._calculate_checksum(archive_path)
            
            # Encrypt if requested (simplified for now)
            if encrypt:
                # TODO: Implement encryption later
                print("Warning: Encryption not yet implemented")
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                created_at=datetime.now(),
                vault_path=str(self.vault_path),
                backup_path=str(archive_path),
                file_count=file_count,
                total_size=total_size,
                checksum=checksum,
                backup_type=backup_type,
                compression=compress,
                encrypted=encrypt
            )
            
            # Save backup info
            info_file = backup_dir / 'backup_info.json'
            with open(info_file, 'w') as f:
                # Convert datetime to string for JSON serialization
                metadata_dict = asdict(metadata)
                metadata_dict['created_at'] = metadata.created_at.isoformat()
                json.dump(metadata_dict, f, indent=2)
            
            # Update metadata index
            self._update_metadata_index(metadata)
            
            return metadata
            
        except Exception as e:
            # Clean up on failure
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            raise BackupError(f"Backup failed: {e}")
    
    def _create_archive(self, source_path: Path, archive_path: Path, compress: bool) -> tuple[int, int]:
        """Create tar archive of source path."""
        mode = 'w:gz' if compress else 'w'
        file_count = 0
        total_size = 0
        
        with tarfile.open(archive_path, mode) as tar:
            for item in source_path.rglob('*'):
                if item.is_file():
                    # Skip hidden files and temp files
                    if item.name.startswith('.') or item.name.endswith('.tmp'):
                        continue
                    
                    # Add to archive with relative path
                    arcname = item.relative_to(source_path)
                    tar.add(item, arcname=arcname)
                    
                    file_count += 1
                    total_size += item.stat().st_size
        
        return file_count, total_size
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _update_metadata_index(self, metadata: BackupMetadata) -> None:
        """Update the backup metadata index."""
        metadata_list = []
        
        # Load existing metadata
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata_list = json.load(f)
            except Exception:
                pass  # Start fresh if file is corrupted
        
        # Add new metadata
        metadata_dict = asdict(metadata)
        metadata_dict['created_at'] = metadata.created_at.isoformat()
        metadata_list.append(metadata_dict)
        
        # Save updated metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata_list, f, indent=2)
    
    def list_backups(self) -> List[BackupMetadata]:
        """List all available backups."""
        if not self.metadata_file.exists():
            return []
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata_list = json.load(f)
            
            backups = []
            for item in metadata_list:
                # Convert datetime string back to datetime object
                item['created_at'] = datetime.fromisoformat(item['created_at'])
                backups.append(BackupMetadata(**item))
            
            # Sort by creation time (newest first)
            return sorted(backups, key=lambda x: x.created_at, reverse=True)
            
        except Exception as e:
            raise BackupError(f"Failed to list backups: {e}")
    
    def restore_backup(
        self, 
        backup_id: str, 
        restore_path: Path = None,
        verify_checksum: bool = True
    ) -> None:
        """Restore a backup."""
        # Find backup metadata
        backups = self.list_backups()
        backup_metadata = None
        
        for backup in backups:
            if backup.backup_id == backup_id:
                backup_metadata = backup
                break
        
        if not backup_metadata:
            raise BackupError(f"Backup not found: {backup_id}")
        
        backup_path = Path(backup_metadata.backup_path)
        if not backup_path.exists():
            raise BackupError(f"Backup file not found: {backup_path}")
        
        # Verify checksum
        if verify_checksum:
            current_checksum = self._calculate_checksum(backup_path)
            if current_checksum != backup_metadata.checksum:
                raise BackupError(f"Backup corrupted: checksum mismatch")
        
        restore_target = restore_path or self.vault_path
        
        try:
            # Decrypt if necessary (simplified for now)
            if backup_metadata.encrypted:
                print("Warning: Decryption not yet implemented")
                raise BackupError("Encrypted backups not yet supported")
            
            # Extract archive
            mode = 'r:gz' if backup_metadata.compression else 'r'
            
            with tarfile.open(backup_path, mode) as tar:
                tar.extractall(path=restore_target)
            
            # Clean up temporary decrypted file
            if backup_metadata.encrypted and backup_path.suffix == '.dec':
                backup_path.unlink()
            
        except Exception as e:
            raise BackupError(f"Restore failed: {e}")
    
    def delete_backup(self, backup_id: str) -> None:
        """Delete a backup."""
        # Find and remove from metadata
        backups = self.list_backups()
        backup_to_delete = None
        remaining_backups = []
        
        for backup in backups:
            if backup.backup_id == backup_id:
                backup_to_delete = backup
            else:
                remaining_backups.append(backup)
        
        if not backup_to_delete:
            raise BackupError(f"Backup not found: {backup_id}")
        
        # Delete backup files
        backup_dir = Path(backup_to_delete.backup_path).parent
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        
        # Update metadata
        metadata_list = []
        for backup in remaining_backups:
            metadata_dict = asdict(backup)
            metadata_dict['created_at'] = backup.created_at.isoformat()
            metadata_list.append(metadata_dict)
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata_list, f, indent=2)
    
    def cleanup_old_backups(self, keep_count: int = 10, keep_days: int = 30) -> List[str]:
        """Clean up old backups based on retention policy."""
        backups = self.list_backups()
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        # Sort by creation time (oldest first)
        backups_by_age = sorted(backups, key=lambda x: x.created_at)
        
        to_delete = []
        
        # Keep recent backups within the time window
        recent_backups = [b for b in backups if b.created_at >= cutoff_date]
        old_backups = [b for b in backups if b.created_at < cutoff_date]
        
        # If we have too many recent backups, keep only the newest ones
        if len(recent_backups) > keep_count:
            to_delete.extend([b.backup_id for b in recent_backups[:-keep_count]])
        
        # Delete all old backups beyond the time window
        to_delete.extend([b.backup_id for b in old_backups])
        
        # Delete identified backups
        deleted = []
        for backup_id in to_delete:
            try:
                self.delete_backup(backup_id)
                deleted.append(backup_id)
            except Exception as e:
                print(f"Failed to delete backup {backup_id}: {e}")
        
        return deleted
    
    def verify_backup_integrity(self, backup_id: str = None) -> Dict[str, Any]:
        """Verify backup integrity."""
        if backup_id:
            backups = [b for b in self.list_backups() if b.backup_id == backup_id]
            if not backups:
                raise BackupError(f"Backup not found: {backup_id}")
        else:
            backups = self.list_backups()
        
        results = {
            'total_backups': len(backups),
            'verified': 0,
            'failed': 0,
            'errors': []
        }
        
        for backup in backups:
            backup_path = Path(backup.backup_path)
            
            try:
                if not backup_path.exists():
                    results['errors'].append(f"Backup file missing: {backup.backup_id}")
                    results['failed'] += 1
                    continue
                
                # Verify checksum
                current_checksum = self._calculate_checksum(backup_path)
                if current_checksum != backup.checksum:
                    results['errors'].append(f"Checksum mismatch: {backup.backup_id}")
                    results['failed'] += 1
                    continue
                
                results['verified'] += 1
                
            except Exception as e:
                results['errors'].append(f"Verification error for {backup.backup_id}: {e}")
                results['failed'] += 1
        
        return results
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """Get backup statistics."""
        backups = self.list_backups()
        
        if not backups:
            return {
                'total_backups': 0,
                'total_size': 0,
                'oldest_backup': None,
                'newest_backup': None,
                'backup_types': {}
            }
        
        total_size = sum(backup.total_size for backup in backups)
        backup_types = {}
        
        for backup in backups:
            backup_types[backup.backup_type] = backup_types.get(backup.backup_type, 0) + 1
        
        return {
            'total_backups': len(backups),
            'total_size': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'oldest_backup': min(backups, key=lambda x: x.created_at).created_at.isoformat(),
            'newest_backup': max(backups, key=lambda x: x.created_at).created_at.isoformat(),
            'backup_types': backup_types
        }


class AutoBackupScheduler:
    """Automatic backup scheduling."""
    
    def __init__(self, backup_manager: BackupManager):
        self.backup_manager = backup_manager
        self.schedule_config = {
            'enabled': False,
            'frequency': 'daily',  # daily, weekly, custom
            'time': '02:00',
            'backup_type': 'incremental',
            'retention_days': 30,
            'retention_count': 10
        }
    
    def configure_schedule(self, config: Dict[str, Any]) -> None:
        """Configure backup schedule."""
        self.schedule_config.update(config)
    
    def should_run_backup(self) -> bool:
        """Check if backup should run now."""
        if not self.schedule_config['enabled']:
            return False
        
        # Get last backup time
        backups = self.backup_manager.list_backups()
        if not backups:
            return True  # No backups exist
        
        last_backup = max(backups, key=lambda x: x.created_at)
        
        # Check based on frequency
        now = datetime.now()
        time_since_last = now - last_backup.created_at
        
        if self.schedule_config['frequency'] == 'daily':
            return time_since_last >= timedelta(days=1)
        elif self.schedule_config['frequency'] == 'weekly':
            return time_since_last >= timedelta(days=7)
        
        return False
    
    def run_scheduled_backup(self) -> Optional[BackupMetadata]:
        """Run scheduled backup if needed."""
        if not self.should_run_backup():
            return None
        
        try:
            # Create backup
            backup = self.backup_manager.create_backup(
                backup_type=self.schedule_config['backup_type']
            )
            
            # Clean up old backups
            self.backup_manager.cleanup_old_backups(
                keep_count=self.schedule_config['retention_count'],
                keep_days=self.schedule_config['retention_days']
            )
            
            return backup
            
        except Exception as e:
            raise BackupError(f"Scheduled backup failed: {e}")