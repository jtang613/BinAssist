import json
from binaryninja import Settings, SettingsScope
from binaryninjaui import UIAction, UIActionHandler
from PySide6.QtWidgets import QInputDialog, QWidget

class BinAssistSettings(Settings):
    """
    Manages the configuration settings for the BinAssist plugin, including API providers, RAG settings,
    and other preferences that need to be stored and retrieved across sessions.
    """

    def __init__(self) -> None:
        """
        Initializes the settings instance and registers all necessary settings for the BinAssist plugin.
        """
        super().__init__(instance_id='default')
        self._register_settings()
        self._register_ui_actions()

    def _register_settings(self) -> None:
        """
        Registers all settings groups and individual settings for the BinAssist plugin with the Binary Ninja 
        settings system.
        """
        self.register_group('binassist', 'BinAssist')

        # Simple provider settings - no complex objects
        settings_definitions = {
            'binassist.provider1_name': {
                'title': 'Provider 1 Name',
                'description': 'Name of the first API provider',
                'type': 'string',
                'default': 'GPT-4o-Mini'
            },
            'binassist.provider1_type': {
                'title': 'Provider 1 Type',
                'description': 'Type of the first API provider',
                'type': 'string',
                'enum': ['openai', 'anthropic', 'ollama', 'lm_studio', 'text_generation_webui', 'custom'],
                'default': 'openai'
            },
            'binassist.provider1_host': {
                'title': 'Provider 1 Host',
                'description': 'Base URL for the first API provider',
                'type': 'string',
                'default': 'https://api.openai.com/v1'
            },
            'binassist.provider1_key': {
                'title': 'Provider 1 API Key',
                'description': 'API key for the first provider',
                'type': 'string',
                'default': '',
                'hidden': True,
                "ignore": ["SettingsProjectScope", "SettingsResourceScope"]
            },
            'binassist.provider1_model': {
                'title': 'Provider 1 Model',
                'description': 'Model name for the first provider',
                'type': 'string',
                'default': 'gpt-4o-mini'
            },
            'binassist.provider1_max_tokens': {
                'title': 'Provider 1 Max Tokens',
                'description': 'Maximum tokens for the first provider',
                'type': 'number',
                'default': 16384,
                'minValue': 1,
                'maxValue': 128*1024
            },
            'binassist.provider2_name': {
                'title': 'Provider 2 Name',
                'description': 'Name of the second API provider',
                'type': 'string',
                'default': 'Claude-3.5-Sonnet'
            },
            'binassist.provider2_type': {
                'title': 'Provider 2 Type',
                'description': 'Type of the second API provider',
                'type': 'string',
                'enum': ['openai', 'anthropic', 'ollama', 'lm_studio', 'text_generation_webui', 'custom'],
                'default': 'anthropic'
            },
            'binassist.provider2_host': {
                'title': 'Provider 2 Host',
                'description': 'Base URL for the second API provider',
                'type': 'string',
                'default': 'https://api.anthropic.com'
            },
            'binassist.provider2_key': {
                'title': 'Provider 2 API Key',
                'description': 'API key for the second provider',
                'type': 'string',
                'default': '',
                'hidden': True,
                "ignore": ["SettingsProjectScope", "SettingsResourceScope"]
            },
            'binassist.provider2_model': {
                'title': 'Provider 2 Model',
                'description': 'Model name for the second provider',
                'type': 'string',
                'default': 'claude-3-5-sonnet-20241022'
            },
            'binassist.provider2_max_tokens': {
                'title': 'Provider 2 Max Tokens',
                'description': 'Maximum tokens for the second provider',
                'type': 'number',
                'default': 8192,
                'minValue': 1,
                'maxValue': 128*1024
            },
            'binassist.provider3_name': {
                'title': 'Provider 3 Name',
                'description': 'Name of the third API provider',
                'type': 'string',
                'default': 'o4-mini'
            },
            'binassist.provider3_type': {
                'title': 'Provider 3 Type',
                'description': 'Type of the third API provider',
                'type': 'string',
                'enum': ['openai', 'anthropic', 'ollama', 'lm_studio', 'text_generation_webui', 'custom'],
                'default': 'openai'
            },
            'binassist.provider3_host': {
                'title': 'Provider 3 Host',
                'description': 'Base URL for the third API provider',
                'type': 'string',
                'default': 'https://api.openai.com/v1'
            },
            'binassist.provider3_key': {
                'title': 'Provider 3 API Key',
                'description': 'API key for the third provider',
                'type': 'string',
                'default': '',
                'hidden': True,
                "ignore": ["SettingsProjectScope", "SettingsResourceScope"]
            },
            'binassist.provider3_model': {
                'title': 'Provider 3 Model',
                'description': 'Model name for the third provider',
                'type': 'string',
                'default': 'o4-mini'
            },
            'binassist.provider3_max_tokens': {
                'title': 'Provider 3 Max Tokens',
                'description': 'Maximum tokens for the third provider',
                'type': 'number',
                'default': 65536,
                'minValue': 1,
                'maxValue': 128*1024
            },
            'binassist.active_provider': {
                'title': 'Active API Provider',
                'description': 'The currently selected API provider',
                'type': 'string',
                'enum': ['GPT-4o-Mini', 'Claude-3.5-Sonnet', 'o4-mini'],
                'default': 'GPT-4o-Mini'
            },
            'binassist.rlhf_db': {
                'title': 'RLHF Database Path',
                'description': 'The path to store the RLHF database.',
                'type': 'string',
                'default': 'rlhf_feedback.db',
                'uiSelectionAction': 'file'
            },
            'binassist.rag_db_path': {
                'title': 'RAG Database Path',
                'description': 'Path to store the RAG vector database.',
                'type': 'string',
                'default': 'binassist_rag_db',
                'uiSelectionAction': 'directory'
            },
            'binassist.use_rag': {
                'title': 'Use RAG',
                'description': 'Enable Retrieval Augmented Generation for queries.',
                'type': 'boolean',
                'default': False
            }
        }

        for key, properties in settings_definitions.items():
            self.register_setting(key, json.dumps(properties))

    def _register_ui_actions(self):
        """
        Registers custom UI actions for the BinAssist plugin.
        """
        # No need for custom UI actions since we're using simple enums
        pass

    def get_provider_config(self, provider_name: str):
        """
        Get configuration for a specific provider.
        """
        # Map provider names to their setting prefixes
        provider_map = {
            'GPT-4o-Mini': 'provider1',
            'Claude-3.5-Sonnet': 'provider2', 
            'o4-mini': 'provider3'
        }
        
        prefix = provider_map.get(provider_name, 'provider1')
        
        return {
            'api___name': self.get_string(f'binassist.{prefix}_name'),
            'provider_type': self.get_string(f'binassist.{prefix}_type'),
            'api__host': self.get_string(f'binassist.{prefix}_host'),
            'api_key': self.get_string(f'binassist.{prefix}_key'),
            'api__model': self.get_string(f'binassist.{prefix}_model'),
            'api__max_tokens': self.get_integer(f'binassist.{prefix}_max_tokens')
        }