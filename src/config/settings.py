"""
Enhanced settings management for BinAssist.
"""

import json
from binaryninja import Settings, SettingsScope
from binaryninjaui import UIAction, UIActionHandler
from PySide6.QtWidgets import QInputDialog, QWidget

from .config_manager import ConfigManager


class BinAssistSettings(Settings):
    """
    Enhanced settings management for the BinAssist plugin.
    
    This class extends Binary Ninja's settings system with our
    configuration manager for better organization and type safety.
    """

    def __init__(self) -> None:
        """
        Initializes the settings instance and registers all necessary settings.
        """
        super().__init__(instance_id='default')
        self._register_settings()
        self._register_ui_actions()
        # Initialize config manager after settings are registered
        try:
            self.config_manager = ConfigManager()
        except Exception as e:
            print(f"Warning: Could not initialize config manager: {e}")
            self.config_manager = None

    def _register_settings(self) -> None:
        """
        Register all settings groups and individual settings.
        """
        self.register_group('binassist', 'BinAssist')

        settings_definitions = {
            'binassist.api_providers': {
                'title': 'API Providers Configuration',
                'description': 'JSON configuration for API providers (managed internally)',
                'type': 'string',
                'default': json.dumps([
                    {
                        'name': 'GPT-4o-Mini',
                        'provider_type': 'openai',
                        'base_url': 'https://api.openai.com/v1',
                        'api_key': '',
                        'model': 'gpt-4o-mini',
                        'max_tokens': 16384,
                        'timeout': 120
                    },
                    {
                        'name': 'Claude-3.5-Sonnet',
                        'provider_type': 'anthropic',
                        'base_url': 'https://api.anthropic.com',
                        'api_key': '',
                        'model': 'claude-3-5-sonnet-20241022',
                        'max_tokens': 8192,
                        'timeout': 120
                    },
                    {
                        'name': 'o4-mini',
                        'provider_type': 'openai',
                        'base_url': 'https://api.openai.com/v1',
                        'api_key': '',
                        'model': 'o4-mini',
                        'max_tokens': 65536,
                        'timeout': 300
                    }
                ], indent=2),
                'multiline': True,
                'readOnly': False
            },
            'binassist.active_provider': {
                'title': 'Active API Provider',
                'description': 'The currently selected API provider',
                'type': 'string',
                'default': 'GPT-4o-Mini',
                'readOnly': True,
                'uiSelectionAction': 'binassist_update_active_provider'
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
            },
            'binassist.max_tool_calls': {
                'title': 'Maximum Tool Calls',
                'description': 'Maximum number of tool calls per query sequence (1-50).',
                'type': 'number',
                'default': 10,
                'minValue': 1,
                'maxValue': 50
            },
            'binassist.ui_settings': {
                'title': 'UI Settings',
                'description': 'UI-specific settings (internal)',
                'type': 'string',
                'default': '{}',
                'hidden': True
            }
        }

        for key, properties in settings_definitions.items():
            self.register_setting(key, json.dumps(properties))

    def _register_ui_actions(self):
        """
        Register custom UI actions for the BinAssist plugin.
        """
        UIAction.registerAction("binassist_update_active_provider")
        UIActionHandler.globalActions().bindAction(
            "binassist_update_active_provider", 
            UIAction(self._update_active_provider_enum)
        )

    def _update_active_provider_enum(self, context):
        """
        Display a selection dialog for choosing the active API provider.
        """
        try:
            # Get providers from the settings JSON
            providers_json = self.get_string('binassist.api_providers')
            providers = json.loads(providers_json)
            provider_names = [provider['name'] for provider in providers]
            
            if not provider_names:
                return  # No providers configured
            
            # Create a parent widget
            parent = QWidget()
            
            # Get current active provider
            current_active = self.get_string('binassist.active_provider')
            current_index = 0
            if current_active in provider_names:
                current_index = provider_names.index(current_active)
            
            # Show the selection dialog
            selected_provider, ok = QInputDialog.getItem(
                parent,
                "Select API Provider",
                "Choose an API provider:",
                provider_names,
                current_index,
                False  # Non-editable
            )
            
            # If the user made a selection and clicked OK, update the active provider
            if ok and selected_provider:
                self.set_string('binassist.active_provider', selected_provider)
                
        except Exception as e:
            print(f"Error updating active provider: {e}")
    
    def get_config_manager(self) -> ConfigManager:
        """
        Get the configuration manager.
        
        Returns:
            ConfigManager instance
        """
        return self.config_manager