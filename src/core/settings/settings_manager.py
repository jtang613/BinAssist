"""
SQLite-based settings manager for BinAssist.

This module provides a flexible settings management system using SQLite as the backend,
replacing Binary Ninja's restrictive native settings system.
"""

import sqlite3
import json
import os
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from ..debug_logger import setup_debug_logging


@dataclass
class SettingDefinition:
    """Definition of a setting including metadata and validation."""
    key: str
    data_type: str  # 'string', 'integer', 'boolean', 'json'
    default_value: Any
    description: str = ""
    category: str = "general"
    validation_rules: Optional[Dict[str, Any]] = None
    is_sensitive: bool = False
    ui_widget_type: str = "default"  # 'default', 'enum', 'password', 'file', 'directory'
    enum_values: Optional[List[str]] = None


class SettingsManager:
    """
    SQLite-based settings manager with validation, encryption support, and migration capabilities.
    """
    
    # Database schema version for migration tracking
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the settings manager.
        
        Args:
            db_path: Optional path to SQLite database. If None, uses default location.
        """
        self.logger = logging.getLogger("binassist.settings")
        
        # Determine database path
        if db_path is None:
            # Use user's home directory for settings
            home_dir = Path.home()
            self.db_path = home_dir / ".binassist" / "settings.db"
        else:
            self.db_path = Path(db_path)
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Register default settings
        self._register_default_settings()
        
        # Ensure we have default providers
        self._ensure_default_providers()
        
        self.logger.info(f"SettingsManager initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create settings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS settings (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        data_type TEXT NOT NULL,
                        default_value TEXT,
                        description TEXT,
                        category TEXT DEFAULT 'general',
                        validation_rules TEXT,
                        is_sensitive BOOLEAN DEFAULT 0,
                        ui_widget_type TEXT DEFAULT 'default',
                        enum_values TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create schema version table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS schema_info (
                        version INTEGER PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Insert or update schema version
                cursor.execute("""
                    INSERT OR REPLACE INTO schema_info (version) VALUES (?)
                """, (self.SCHEMA_VERSION,))
                
                # Create indexes for performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_settings_category ON settings(category)
                """)
                
                conn.commit()
                self.logger.debug("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _register_default_settings(self):
        """Register all default settings definitions."""
        default_settings = [
            # API Provider Settings
            SettingDefinition(
                key="api_providers",
                data_type="json",
                default_value=[],
                description="List of configured API providers",
                category="api"
            ),
            SettingDefinition(
                key="active_provider",
                data_type="string",
                default_value="",
                description="Currently active API provider name",
                category="api",
                ui_widget_type="enum"
            ),
            
            # RAG Settings
            SettingDefinition(
                key="rag_db_path",
                data_type="string",
                default_value="binassist_rag_db",
                description="Path to RAG vector database",
                category="rag",
                ui_widget_type="directory"
            ),
            SettingDefinition(
                key="use_rag",
                data_type="boolean",
                default_value=False,
                description="Enable Retrieval Augmented Generation",
                category="rag"
            ),
            
            # RLHF Settings
            SettingDefinition(
                key="rlhf_db_path",
                data_type="string",
                default_value="rlhf_feedback.db",
                description="Path to RLHF feedback database",
                category="rlhf",
                ui_widget_type="file"
            ),
            
            # MCP Settings
            SettingDefinition(
                key="use_mcp_tools",
                data_type="boolean",
                default_value=False,
                description="Enable Model Context Protocol tools",
                category="mcp"
            ),
            SettingDefinition(
                key="mcp_servers",
                data_type="json",
                default_value=[],
                description="List of configured MCP servers",
                category="mcp"
            ),
            
            # UI Settings
            SettingDefinition(
                key="ui_settings",
                data_type="json",
                default_value={},
                description="UI-specific settings and preferences",
                category="ui"
            ),
            SettingDefinition(
                key="default_il_level",
                data_type="string",
                default_value="HLIL",
                description="Default intermediate language level",
                category="ui",
                ui_widget_type="enum",
                enum_values=["ASM", "LLIL", "MLIL", "HLIL", "Pseudo-C"]
            ),
            SettingDefinition(
                key="context_lines",
                data_type="integer",
                default_value=10,
                description="Number of context lines to include",
                category="ui",
                validation_rules={"min": 0, "max": 100}
            ),
            SettingDefinition(
                key="analysis_mode",
                data_type="string",
                default_value="Balanced",
                description="Analysis mode for code analysis",
                category="ui",
                ui_widget_type="enum",
                enum_values=["Conservative", "Balanced", "Aggressive"]
            ),
            SettingDefinition(
                key="response_verbosity",
                data_type="string",
                default_value="Detailed",
                description="Response verbosity level",
                category="ui",
                ui_widget_type="enum",
                enum_values=["Concise", "Detailed", "Comprehensive"]
            ),
            
            # Analysis Options
            SettingDefinition(
                key="include_comments",
                data_type="boolean",
                default_value=True,
                description="Include existing comments in analysis",
                category="analysis"
            ),
            SettingDefinition(
                key="include_imports",
                data_type="boolean",
                default_value=True,
                description="Include import/library information",
                category="analysis"
            ),
            SettingDefinition(
                key="auto_apply_suggestions",
                data_type="boolean",
                default_value=False,
                description="Auto-apply high-confidence suggestions",
                category="analysis"
            ),
            SettingDefinition(
                key="syntax_highlighting",
                data_type="boolean",
                default_value=True,
                description="Enable syntax highlighting in responses",
                category="analysis"
            ),
            SettingDefinition(
                key="show_addresses",
                data_type="boolean",
                default_value=True,
                description="Show addresses in code snippets",
                category="analysis"
            ),
            
            # System Settings
            SettingDefinition(
                key="system_prompt",
                data_type="string",
                default_value=self._get_default_system_prompt(),
                description="System prompt for LLM interactions",
                category="system"
            ),
            SettingDefinition(
                key="log_level",
                data_type="string",
                default_value="INFO",
                description="Logging level",
                category="system",
                ui_widget_type="enum",
                enum_values=["DEBUG", "INFO", "WARNING", "ERROR"]
            ),
        ]
        
        # Register each setting
        for setting in default_settings:
            self._register_setting_definition(setting)
    
    def _ensure_default_providers(self):
        """Ensure that default providers exist."""
        try:
            providers = self.get_json('api_providers', [])
            
            # If no providers exist, create defaults
            if not providers:
                default_providers = [
                    {
                        'name': 'GPT-4o-Mini',
                        'provider_type': 'openai',
                        'base_url': 'https://api.openai.com/v1',
                        'api_key': '',
                        'model': 'gpt-4o-mini',
                        'max_tokens': 16384,
                        'timeout': 120,
                        'enabled': True
                    },
                    {
                        'name': 'Claude-3.5-Sonnet',
                        'provider_type': 'anthropic',
                        'base_url': 'https://api.anthropic.com',
                        'api_key': '',
                        'model': 'claude-3-5-sonnet-20241022',
                        'max_tokens': 8192,
                        'timeout': 120,
                        'enabled': True
                    },
                    {
                        'name': 'Ollama-Local',
                        'provider_type': 'ollama',
                        'base_url': 'http://localhost:11434/v1',
                        'api_key': '',
                        'model': 'llama3.1:8b',
                        'max_tokens': 4096,
                        'timeout': 120,
                        'enabled': True
                    }
                ]
                
                self.set_json('api_providers', default_providers)
                self.set_string('active_provider', 'GPT-4o-Mini')
                self.logger.info("Created default API providers")
                
        except Exception as e:
            self.logger.error(f"Failed to ensure default providers: {e}")
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt."""
        return '''You are a professional software reverse engineer specializing in cybersecurity. You are intimately 
familiar with x86_64, ARM, PPC and MIPS architectures. You are an expert C and C++ developer.
You are an expert Python and Rust developer. You are familiar with common frameworks and libraries 
such as WinSock, OpenSSL, MFC, etc. You are an expert in TCP/IP network programming and packet analysis.
You always respond to queries in a structured format using Markdown styling for headings and lists. 
You format code blocks using back-tick code-fencing.'''.strip()
    
    def _register_setting_definition(self, setting: SettingDefinition):
        """Register a setting definition in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if setting already exists
                cursor.execute("SELECT key FROM settings WHERE key = ?", (setting.key,))
                exists = cursor.fetchone() is not None
                
                if not exists:
                    # Insert new setting with definition
                    cursor.execute("""
                        INSERT INTO settings (
                            key, value, data_type, default_value, description, 
                            category, validation_rules, is_sensitive, ui_widget_type, enum_values
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        setting.key,
                        self._serialize_value(setting.default_value, setting.data_type),
                        setting.data_type,
                        self._serialize_value(setting.default_value, setting.data_type),
                        setting.description,
                        setting.category,
                        json.dumps(setting.validation_rules) if setting.validation_rules else None,
                        setting.is_sensitive,
                        setting.ui_widget_type,
                        json.dumps(setting.enum_values) if setting.enum_values else None
                    ))
                    self.logger.debug(f"Registered setting: {setting.key}")
                else:
                    # Update metadata for existing setting
                    cursor.execute("""
                        UPDATE settings SET 
                            description = ?, category = ?, validation_rules = ?,
                            is_sensitive = ?, ui_widget_type = ?, enum_values = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE key = ?
                    """, (
                        setting.description,
                        setting.category,
                        json.dumps(setting.validation_rules) if setting.validation_rules else None,
                        setting.is_sensitive,
                        setting.ui_widget_type,
                        json.dumps(setting.enum_values) if setting.enum_values else None,
                        setting.key
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to register setting {setting.key}: {e}")
            raise
    
    def get_string(self, key: str, default: Optional[str] = None) -> str:
        """Get a string setting value."""
        value = self._get_setting_value(key, default)
        return str(value) if value is not None else ""
    
    def get_integer(self, key: str, default: Optional[int] = None) -> int:
        """Get an integer setting value."""
        value = self._get_setting_value(key, default)
        if value is None:
            return 0
        try:
            return int(value)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid integer value for {key}: {value}")
            return default or 0
    
    def get_boolean(self, key: str, default: Optional[bool] = None) -> bool:
        """Get a boolean setting value."""
        value = self._get_setting_value(key, default)
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)
    
    def get_json(self, key: str, default: Optional[Dict] = None) -> Dict:
        """Get a JSON setting value."""
        value = self._get_setting_value(key, default)
        if value is None:
            return default or {}
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(str(value))
        except (json.JSONDecodeError, TypeError):
            self.logger.warning(f"Invalid JSON value for {key}: {value}")
            return default or {}
    
    def set_string(self, key: str, value: str):
        """Set a string setting value."""
        self._set_setting_value(key, value, "string")
    
    def set_integer(self, key: str, value: int):
        """Set an integer setting value."""
        self._set_setting_value(key, value, "integer")
    
    def set_boolean(self, key: str, value: bool):
        """Set a boolean setting value."""
        self._set_setting_value(key, value, "boolean")
    
    def set_json(self, key: str, value: Union[Dict, List]):
        """Set a JSON setting value."""
        self._set_setting_value(key, value, "json")
    
    def _get_setting_value(self, key: str, default: Any = None) -> Any:
        """Get a setting value from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT value, data_type, default_value FROM settings WHERE key = ?
                """, (key,))
                result = cursor.fetchone()
                
                if result is None:
                    self.logger.warning(f"Setting '{key}' not found, returning default")
                    return default
                
                value, data_type, default_value = result
                
                # Use stored value or fall back to default
                actual_value = value if value is not None else default_value
                
                # Deserialize based on data type
                return self._deserialize_value(actual_value, data_type)
                
        except Exception as e:
            self.logger.error(f"Failed to get setting {key}: {e}")
            return default
    
    def _set_setting_value(self, key: str, value: Any, data_type: str):
        """Set a setting value in the database."""
        try:
            # Validate the value
            if not self._validate_setting_value(key, value):
                raise ValueError(f"Invalid value for setting {key}: {value}")
            
            serialized_value = self._serialize_value(value, data_type)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update or insert the setting
                cursor.execute("""
                    INSERT OR REPLACE INTO settings 
                    (key, value, data_type, updated_at) 
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (key, serialized_value, data_type))
                
                conn.commit()
                self.logger.debug(f"Set setting {key} = {value}")
                
        except Exception as e:
            self.logger.error(f"Failed to set setting {key}: {e}")
            raise
    
    def _serialize_value(self, value: Any, data_type: str) -> str:
        """Serialize a value for storage."""
        if value is None:
            return None
        
        if data_type == "json":
            return json.dumps(value)
        elif data_type == "boolean":
            return "1" if value else "0"
        else:
            return str(value)
    
    def _deserialize_value(self, value: str, data_type: str) -> Any:
        """Deserialize a value from storage."""
        if value is None:
            return None
        
        try:
            if data_type == "integer":
                return int(value)
            elif data_type == "boolean":
                return value == "1" or value.lower() == "true"
            elif data_type == "json":
                return json.loads(value)
            else:  # string
                return value
        except (ValueError, json.JSONDecodeError) as e:
            self.logger.warning(f"Failed to deserialize {data_type} value '{value}': {e}")
            return None
    
    def _validate_setting_value(self, key: str, value: Any) -> bool:
        """Validate a setting value against its definition."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT validation_rules, enum_values, data_type FROM settings WHERE key = ?
                """, (key,))
                result = cursor.fetchone()
                
                if result is None:
                    return True  # Unknown setting, allow it
                
                validation_rules, enum_values, data_type = result
                
                # Check enum values
                if enum_values:
                    enum_list = json.loads(enum_values)
                    if value not in enum_list:
                        self.logger.warning(f"Value '{value}' not in enum {enum_list} for {key}")
                        return False
                
                # Check validation rules
                if validation_rules:
                    rules = json.loads(validation_rules)
                    
                    if data_type == "integer":
                        if "min" in rules and value < rules["min"]:
                            return False
                        if "max" in rules and value > rules["max"]:
                            return False
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to validate setting {key}: {e}")
            return True  # Allow on validation error
    
    def get_all_settings(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Get all settings, optionally filtered by category."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if category:
                    cursor.execute("""
                        SELECT key, value, data_type, default_value 
                        FROM settings WHERE category = ?
                    """, (category,))
                else:
                    cursor.execute("""
                        SELECT key, value, data_type, default_value FROM settings
                    """)
                
                settings = {}
                for row in cursor.fetchall():
                    key, value, data_type, default_value = row
                    actual_value = value if value is not None else default_value
                    settings[key] = self._deserialize_value(actual_value, data_type)
                
                return settings
                
        except Exception as e:
            self.logger.error(f"Failed to get all settings: {e}")
            return {}
    
    def get_setting_definition(self, key: str) -> Optional[Dict[str, Any]]:
        """Get the definition/metadata for a setting."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM settings WHERE key = ?
                """, (key,))
                result = cursor.fetchone()
                
                if result is None:
                    return None
                
                # Convert to dictionary
                columns = [desc[0] for desc in cursor.description]
                setting_dict = dict(zip(columns, result))
                
                # Deserialize JSON fields
                if setting_dict.get('validation_rules'):
                    setting_dict['validation_rules'] = json.loads(setting_dict['validation_rules'])
                if setting_dict.get('enum_values'):
                    setting_dict['enum_values'] = json.loads(setting_dict['enum_values'])
                
                return setting_dict
                
        except Exception as e:
            self.logger.error(f"Failed to get setting definition for {key}: {e}")
            return None
    
    def reset_to_default(self, key: str):
        """Reset a setting to its default value."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE settings SET value = default_value, updated_at = CURRENT_TIMESTAMP 
                    WHERE key = ?
                """, (key,))
                conn.commit()
                self.logger.info(f"Reset setting {key} to default")
                
        except Exception as e:
            self.logger.error(f"Failed to reset setting {key}: {e}")
            raise
    
    def reset_all_to_default(self):
        """Reset all settings to their default values."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE settings SET value = default_value, updated_at = CURRENT_TIMESTAMP
                """)
                conn.commit()
                self.logger.info("Reset all settings to default")
                
        except Exception as e:
            self.logger.error(f"Failed to reset all settings: {e}")
            raise
    
    def export_settings(self, file_path: str, include_sensitive: bool = False):
        """Export settings to a JSON file."""
        try:
            settings = {}
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if include_sensitive:
                    cursor.execute("SELECT key, value, data_type FROM settings")
                else:
                    cursor.execute("SELECT key, value, data_type FROM settings WHERE is_sensitive = 0")
                
                for row in cursor.fetchall():
                    key, value, data_type = row
                    if value is not None:
                        settings[key] = self._deserialize_value(value, data_type)
            
            with open(file_path, 'w') as f:
                json.dump(settings, f, indent=2)
            
            self.logger.info(f"Exported settings to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export settings: {e}")
            raise
    
    def import_settings(self, file_path: str, overwrite: bool = False):
        """Import settings from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                settings_data = json.load(f)
            
            for key, value in settings_data.items():
                # Get setting definition to determine data type
                definition = self.get_setting_definition(key)
                if definition:
                    data_type = definition['data_type']
                    
                    # Check if we should overwrite
                    if not overwrite:
                        current_value = self._get_setting_value(key)
                        if current_value is not None:
                            continue
                    
                    self._set_setting_value(key, value, data_type)
                else:
                    self.logger.warning(f"Unknown setting key during import: {key}")
            
            self.logger.info(f"Imported settings from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to import settings: {e}")
            raise
    
    def close(self):
        """Close the settings manager and clean up resources."""
        # SQLite connections are automatically closed, nothing to do here
        self.logger.debug("SettingsManager closed")


# Global instance
_settings_manager: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    """Get the global settings manager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager


def reset_settings_manager():
    """Reset the global settings manager instance (useful for testing)."""
    global _settings_manager
    if _settings_manager:
        _settings_manager.close()
    _settings_manager = None