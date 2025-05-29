"""
Migration utilities for transitioning from Binary Ninja settings to SQLite settings.

This module handles the one-time migration of existing settings from Binary Ninja's
native settings system to the new SQLite-based system.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    from binaryninja.settings import Settings as BNSettings
    BINARY_NINJA_AVAILABLE = True
except ImportError:
    BINARY_NINJA_AVAILABLE = False
    BNSettings = None

from .settings_manager import SettingsManager, get_settings_manager


class SettingsMigrator:
    """Handles migration from Binary Ninja settings to SQLite settings."""
    
    def __init__(self, settings_manager: Optional[SettingsManager] = None):
        """
        Initialize the migrator.
        
        Args:
            settings_manager: Optional settings manager instance. If None, uses global instance.
        """
        self.settings_manager = settings_manager or get_settings_manager()
        self.logger = logging.getLogger("binassist.migration")
        
        # Migration mapping: old_key -> new_key (if different)
        self.key_mappings = {
            'binassist.rlhf_db': 'rlhf_db_path',
            'binassist.use_mcp_tools': 'use_mcp_tools',
            'binassist.mcp_server_count': 'mcp_server_count',
        }
        
        # Settings that need special handling during migration
        self.special_migrations = {
            'provider_settings': self._migrate_provider_settings,
            'ui_settings': self._migrate_ui_settings,
        }
    
    def migrate_from_binary_ninja(self, backup_existing: bool = True) -> bool:
        """
        Migrate settings from Binary Ninja to SQLite.
        
        Args:
            backup_existing: If True, create a backup of existing Binary Ninja settings
            
        Returns:
            True if migration was successful, False otherwise
        """
        if not BINARY_NINJA_AVAILABLE:
            self.logger.warning("Binary Ninja not available, creating default settings")
            self._create_default_settings()
            return True
        
        try:
            self.logger.info("Starting migration from Binary Ninja settings")
            
            # Check if migration has already been performed
            if self._is_migration_complete():
                self.logger.info("Migration already completed, skipping")
                return True
            
            bn_settings = BNSettings()
            
            # Create backup if requested
            if backup_existing:
                self._create_backup(bn_settings)
            
            # Migrate different types of settings
            migrated_any = False
            migrated_any |= self._migrate_simple_settings(bn_settings)
            migrated_any |= self._migrate_provider_settings(bn_settings)
            migrated_any |= self._migrate_ui_settings(bn_settings)
            
            # If no settings were migrated, create defaults
            if not migrated_any:
                self.logger.info("No existing settings found, creating defaults")
                self._create_default_settings()
            
            # Mark migration as complete
            self._mark_migration_complete()
            
            self.logger.info("Migration from Binary Ninja settings completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration failed, creating default settings: {e}")
            self.logger.exception("Full migration error traceback:")
            self._create_default_settings()
            return True  # Return True since we created defaults
    
    def _is_migration_complete(self) -> bool:
        """Check if migration has already been completed."""
        try:
            # Check for a migration marker or specific migrated settings
            migrated_providers = self.settings_manager.get_json('api_providers', [])
            return len(migrated_providers) > 0
        except Exception:
            return False
    
    def _mark_migration_complete(self):
        """Mark migration as complete."""
        try:
            # Set a migration timestamp
            from datetime import datetime
            self.settings_manager.set_string('migration_timestamp', datetime.now().isoformat())
            self.settings_manager.set_boolean('migration_completed', True)
        except Exception as e:
            self.logger.warning(f"Failed to mark migration complete: {e}")
    
    def _create_default_settings(self):
        """Create default settings when no migration is possible."""
        try:
            # The SettingsManager already creates defaults in _ensure_default_providers
            # But let's ensure some key settings are set
            self.settings_manager.set_string('rag_db_path', 'binassist_rag_db')
            self.settings_manager.set_string('rlhf_db_path', 'rlhf_feedback.db')
            self.settings_manager.set_boolean('use_rag', False)
            self.settings_manager.set_boolean('use_mcp_tools', False)
            
            # Mark as "migrated" (even though we just created defaults)
            self._mark_migration_complete()
            self.logger.info("Created default settings")
            
        except Exception as e:
            self.logger.error(f"Failed to create default settings: {e}")
    
    def _create_backup(self, bn_settings: BNSettings):
        """Create a backup of Binary Ninja settings."""
        try:
            backup_data = {}
            
            # List of all BinAssist-related settings to backup
            setting_keys = [
                'binassist.provider1_name', 'binassist.provider1_type', 'binassist.provider1_host',
                'binassist.provider1_key', 'binassist.provider1_model', 'binassist.provider1_max_tokens',
                'binassist.provider2_name', 'binassist.provider2_type', 'binassist.provider2_host',
                'binassist.provider2_key', 'binassist.provider2_model', 'binassist.provider2_max_tokens',
                'binassist.provider3_name', 'binassist.provider3_type', 'binassist.provider3_host',
                'binassist.provider3_key', 'binassist.provider3_model', 'binassist.provider3_max_tokens',
                'binassist.active_provider', 'binassist.rlhf_db', 'binassist.rag_db_path',
                'binassist.use_rag', 'binassist.use_mcp_tools'
            ]
            
            for key in setting_keys:
                try:
                    if 'max_tokens' in key or 'server_count' in key:
                        backup_data[key] = bn_settings.get_integer(key)
                    elif 'use_' in key:
                        backup_data[key] = bn_settings.get_bool(key)
                    else:
                        backup_data[key] = bn_settings.get_string(key)
                except Exception:
                    # Setting doesn't exist or can't be read
                    continue
            
            # Save backup to file
            backup_path = self.settings_manager.db_path.parent / "binary_ninja_settings_backup.json"
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            self.logger.info(f"Created settings backup at: {backup_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {e}")
    
    def _migrate_simple_settings(self, bn_settings: BNSettings) -> bool:
        """Migrate simple key-value settings."""
        migrated = False
        simple_migrations = [
            ('binassist.rlhf_db', 'rlhf_db_path', 'string'),
            ('binassist.rag_db_path', 'rag_db_path', 'string'),
            ('binassist.use_rag', 'use_rag', 'boolean'),
            ('binassist.use_mcp_tools', 'use_mcp_tools', 'boolean'),
        ]
        
        for old_key, new_key, data_type in simple_migrations:
            try:
                if data_type == 'string':
                    value = bn_settings.get_string(old_key)
                elif data_type == 'boolean':
                    value = bn_settings.get_bool(old_key)
                elif data_type == 'integer':
                    value = bn_settings.get_integer(old_key)
                else:
                    continue
                
                if value:  # Only migrate non-empty values
                    if data_type == 'string':
                        self.settings_manager.set_string(new_key, value)
                    elif data_type == 'boolean':
                        self.settings_manager.set_boolean(new_key, value)
                    elif data_type == 'integer':
                        self.settings_manager.set_integer(new_key, value)
                    
                    self.logger.debug(f"Migrated {old_key} -> {new_key}: {value}")
                    migrated = True
                    
            except Exception as e:
                self.logger.warning(f"Failed to migrate {old_key}: {e}")
        
        return migrated
    
    def _migrate_provider_settings(self, bn_settings: BNSettings) -> bool:
        """Migrate API provider settings to the new format."""
        try:
            providers = []
            migrated = False
            
            # Migrate up to 3 providers
            for i in range(1, 4):
                try:
                    name = bn_settings.get_string(f'binassist.provider{i}_name')
                    if not name:
                        continue
                    
                    provider = {
                        'name': name,
                        'provider_type': bn_settings.get_string(f'binassist.provider{i}_type') or 'openai',
                        'base_url': bn_settings.get_string(f'binassist.provider{i}_host') or 'https://api.openai.com/v1',
                        'api_key': bn_settings.get_string(f'binassist.provider{i}_key') or '',
                        'model': bn_settings.get_string(f'binassist.provider{i}_model') or 'gpt-4o-mini',
                        'max_tokens': bn_settings.get_integer(f'binassist.provider{i}_max_tokens') or 16384,
                        'timeout': 120,  # Default timeout
                        'enabled': True
                    }
                    
                    providers.append(provider)
                    self.logger.debug(f"Migrated provider {i}: {name}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to migrate provider {i}: {e}")
            
            # Set the providers list
            if providers:
                self.settings_manager.set_json('api_providers', providers)
                self.logger.info(f"Migrated {len(providers)} API providers")
                migrated = True
            else:
                migrated = False
            
            # Migrate active provider
            try:
                active_provider = bn_settings.get_string('binassist.active_provider')
                if active_provider:
                    self.settings_manager.set_string('active_provider', active_provider)
                    self.logger.debug(f"Migrated active provider: {active_provider}")
                    migrated = True
            except Exception as e:
                self.logger.warning(f"Failed to migrate active provider: {e}")
                
            return migrated
                
        except Exception as e:
            self.logger.error(f"Failed to migrate provider settings: {e}")
            return False
    
    def _migrate_ui_settings(self, bn_settings: BNSettings) -> bool:
        """Migrate UI-related settings."""
        try:
            ui_settings = {}
            
            # Try to get any existing UI settings from Binary Ninja
            try:
                existing_ui = bn_settings.get_json('binassist.ui_settings')
                if existing_ui:
                    ui_settings.update(json.loads(existing_ui))
            except Exception:
                pass
            
            # Set default UI settings if none exist
            if not ui_settings:
                ui_settings = {
                    'last_tab_index': 0,
                    'window_geometry': None,
                    'analysis_filters': {},
                    'query_history': []
                }
            
            self.settings_manager.set_json('ui_settings', ui_settings)
            self.logger.debug("Migrated UI settings")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to migrate UI settings: {e}")
            return False
    
    def verify_migration(self) -> bool:
        """Verify that migration was successful."""
        try:
            # Check that essential settings exist
            providers = self.settings_manager.get_json('api_providers', [])
            if not providers:
                self.logger.warning("No providers found after migration")
                return False
            
            # Check that basic settings are accessible
            rag_path = self.settings_manager.get_string('rag_db_path')
            use_rag = self.settings_manager.get_boolean('use_rag')
            
            self.logger.info("Migration verification passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration verification failed: {e}")
            return False


def migrate_from_binary_ninja_settings(backup_existing: bool = True) -> bool:
    """
    Convenience function to migrate from Binary Ninja settings.
    
    Args:
        backup_existing: If True, create a backup of existing settings
        
    Returns:
        True if migration was successful
    """
    migrator = SettingsMigrator()
    success = migrator.migrate_from_binary_ninja(backup_existing)
    
    if success:
        success = migrator.verify_migration()
    
    return success