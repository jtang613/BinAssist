"""
Compatibility wrapper for Binary Ninja settings interface.

This module provides a drop-in replacement for Binary Ninja's Settings class
that uses the SQLite-based settings backend. This allows for minimal code changes
during migration.
"""

import logging
from typing import Any, Optional

from .settings_manager import get_settings_manager, SettingsManager


class BinAssistSettings:
    """
    Drop-in replacement for Binary Ninja Settings that uses SQLite backend.
    
    This class provides the same interface as Binary Ninja's Settings class
    but stores data in SQLite instead of Binary Ninja's native settings system.
    """
    
    def __init__(self, settings_manager: Optional[SettingsManager] = None):
        """
        Initialize the compatibility settings wrapper.
        
        Args:
            settings_manager: Optional settings manager instance
        """
        self.settings_manager = settings_manager or get_settings_manager()
        self.logger = logging.getLogger("binassist.settings.compat")
    
    def get_string(self, key: str, default: str = "") -> str:
        """
        Get a string setting value.
        
        Args:
            key: Setting key (will strip 'binassist.' prefix if present)
            default: Default value if setting doesn't exist
            
        Returns:
            String value of the setting
        """
        normalized_key = self._normalize_key(key)
        return self.settings_manager.get_string(normalized_key, default)
    
    def get_integer(self, key: str, default: int = 0) -> int:
        """
        Get an integer setting value.
        
        Args:
            key: Setting key (will strip 'binassist.' prefix if present)
            default: Default value if setting doesn't exist
            
        Returns:
            Integer value of the setting
        """
        normalized_key = self._normalize_key(key)
        return self.settings_manager.get_integer(normalized_key, default)
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        Get a boolean setting value.
        
        Args:
            key: Setting key (will strip 'binassist.' prefix if present)
            default: Default value if setting doesn't exist
            
        Returns:
            Boolean value of the setting
        """
        normalized_key = self._normalize_key(key)
        return self.settings_manager.get_boolean(normalized_key, default)
    
    def get_json(self, key: str, default: str = "{}") -> str:
        """
        Get a JSON setting value as a string.
        
        Args:
            key: Setting key (will strip 'binassist.' prefix if present)
            default: Default JSON string if setting doesn't exist
            
        Returns:
            JSON string representation of the setting
        """
        normalized_key = self._normalize_key(key)
        import json
        
        # Get the actual object and convert to JSON string for compatibility
        value = self.settings_manager.get_json(normalized_key, {})
        try:
            return json.dumps(value)
        except Exception:
            return default
    
    def set_string(self, key: str, value: str):
        """
        Set a string setting value.
        
        Args:
            key: Setting key (will strip 'binassist.' prefix if present)
            value: String value to set
        """
        normalized_key = self._normalize_key(key)
        self.settings_manager.set_string(normalized_key, value)
    
    def set_integer(self, key: str, value: int):
        """
        Set an integer setting value.
        
        Args:
            key: Setting key (will strip 'binassist.' prefix if present)
            value: Integer value to set
        """
        normalized_key = self._normalize_key(key)
        self.settings_manager.set_integer(normalized_key, value)
    
    def set_bool(self, key: str, value: bool):
        """
        Set a boolean setting value.
        
        Args:
            key: Setting key (will strip 'binassist.' prefix if present)
            value: Boolean value to set
        """
        normalized_key = self._normalize_key(key)
        self.settings_manager.set_boolean(normalized_key, value)
    
    def set_json(self, key: str, value: str):
        """
        Set a JSON setting value from a string.
        
        Args:
            key: Setting key (will strip 'binassist.' prefix if present)
            value: JSON string to set
        """
        normalized_key = self._normalize_key(key)
        import json
        
        try:
            # Parse the JSON string and store as object
            parsed_value = json.loads(value)
            self.settings_manager.set_json(normalized_key, parsed_value)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON for key {key}: {e}")
            raise
    
    def _normalize_key(self, key: str) -> str:
        """
        Normalize setting key by removing 'binassist.' prefix and mapping to new schema.
        
        Args:
            key: Original setting key
            
        Returns:
            Normalized key for the new settings system
        """
        # Remove binassist prefix if present
        if key.startswith('binassist.'):
            normalized = key[10:]  # Remove 'binassist.' prefix
        else:
            normalized = key
        
        # Handle special key mappings for legacy compatibility
        key_mappings = {
            # Provider mappings - convert old provider format to new
            'provider1_name': 'api_providers',  # Will need special handling
            'provider1_type': 'api_providers',
            'provider1_host': 'api_providers',
            'provider1_key': 'api_providers',
            'provider1_model': 'api_providers',
            'provider1_max_tokens': 'api_providers',
            'provider2_name': 'api_providers',
            'provider2_type': 'api_providers',
            'provider2_host': 'api_providers',
            'provider2_key': 'api_providers',
            'provider2_model': 'api_providers',
            'provider2_max_tokens': 'api_providers',
            'provider3_name': 'api_providers',
            'provider3_type': 'api_providers',
            'provider3_host': 'api_providers',
            'provider3_key': 'api_providers',
            'provider3_model': 'api_providers',
            'provider3_max_tokens': 'api_providers',
            
            # Direct mappings
            'rlhf_db': 'rlhf_db_path',
            'active_provider': 'active_provider',
            'rag_db_path': 'rag_db_path',
            'use_rag': 'use_rag',
            'use_mcp_tools': 'use_mcp_tools',
            'mcp_server_count': 'mcp_server_count',
            'ui_settings': 'ui_settings',
        }
        
        return key_mappings.get(normalized, normalized)
    
    def _handle_legacy_provider_access(self, key: str) -> Any:
        """
        Handle legacy provider setting access.
        
        This method converts old provider1_*, provider2_*, provider3_* access
        to the new unified api_providers structure.
        """
        # Extract provider number and field
        if '_' not in key:
            return None
        
        parts = key.split('_')
        if len(parts) < 2:
            return None
        
        provider_prefix = parts[0]  # provider1, provider2, provider3
        field = '_'.join(parts[1:])  # name, type, host, etc.
        
        # Extract provider index
        try:
            provider_num = int(provider_prefix.replace('provider', ''))
        except ValueError:
            return None
        
        # Get providers list
        providers = self.settings_manager.get_json('api_providers', [])
        
        # Check if provider exists at index
        provider_index = provider_num - 1  # Convert to 0-based index
        if provider_index >= len(providers):
            return None
        
        provider = providers[provider_index]
        
        # Map field names
        field_mappings = {
            'name': 'name',
            'type': 'provider_type',
            'host': 'base_url',
            'key': 'api_key',
            'model': 'model',
            'max_tokens': 'max_tokens'
        }
        
        mapped_field = field_mappings.get(field)
        if mapped_field and mapped_field in provider:
            return provider[mapped_field]
        
        return None


# Global compatibility instance
_compat_settings: Optional[BinAssistSettings] = None


def get_compat_settings() -> BinAssistSettings:
    """Get the global compatibility settings instance."""
    global _compat_settings
    if _compat_settings is None:
        _compat_settings = BinAssistSettings()
    return _compat_settings


def reset_compat_settings():
    """Reset the global compatibility settings instance."""
    global _compat_settings
    _compat_settings = None