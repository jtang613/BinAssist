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

        settings_definitions = {
            'binassist.api_providers': {
                'title': 'API Providers',
                'description': 'List of API providers for BinAssist',
                'type': 'array',
                'elementType': 'object',
                'default': [
                    {
                        'api___name': 'GPT-4o-Mini',
                        'api__host': 'https://api.openai.com/v1',
                        'api_key': '',
                        'api__model': 'gpt-4o-mini',
                        'api__max_tokens': 16384
                    }
                ],
                'properties': {
                    'api___name': {'type': 'string', 'title': 'Provider Name'},
                    'api__host': {'type': 'string', 'title': 'Remote API Host'},
                    'api_key': {'type': 'string', 'title': 'API Key', 'hidden': True, "ignore" : ["SettingsProjectScope", "SettingsResourceScope"]},
                    'api__model': {'type': 'string', 'title': 'LLM Model'},
                    'api__max_tokens': {'type': 'number', 'title': 'Max Completion Tokens', 'minValue': 1, 'maxValue': 128*1024}
                }
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
            }
        }

        for key, properties in settings_definitions.items():
            self.register_setting(key, json.dumps(properties))

    def _register_ui_actions(self):
        """
        Registers custom UI actions for the BinAssist plugin.
        """
        UIAction.registerAction("binassist_update_active_provider")
        UIActionHandler.globalActions().bindAction("binassist_update_active_provider", UIAction(self._update_active_provider_enum))

    def _update_active_provider_enum(self, context):
        """
        Displays a PySide selection dialog populated with the list of API providers.
        Updates the active_provider field with the selected API provider name.
        """
        # Get the current list of API providers
        providers = json.loads(self.get_json('binassist.api_providers'))
        provider_names = [provider['api___name'] for provider in providers]

        # Create a parent widget (can be None if you don't have a specific parent)
        parent = QWidget()

        # Show the selection dialog
        selected_provider, ok = QInputDialog.getItem(
            parent,
            "Select API Provider",
            "Choose an API provider:",
            provider_names,
            0,  # Current index (0 for the first item)
            False  # Non-editable
        )

        # If the user made a selection and clicked OK, update the active provider
        if ok and selected_provider:
            self.set_string('binassist.active_provider', selected_provider)
