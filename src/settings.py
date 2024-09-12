import json
from binaryninja.settings import Settings
from .exceptions import RegisterSettingsGroupException, RegisterSettingsKeyException

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

    def _register_settings(self) -> None:
        """
        Registers all settings groups and individual settings for the BinAssist plugin with the Binary Ninja 
        settings system. It ensures that all required settings are available in the settings UI and can be 
        modified by the user.
        
        Raises:
            RegisterSettingsGroupException: If the settings group fails to be registered.
            RegisterSettingsKeyException: If an individual setting fails to be registered.
        """
        self.register_group('binassist', 'BinAssist')

        settings_definitions = {
            # API Provider fields have odd underscores so they sort sanely in the Settings view.
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
#                'enum': ['GPT-4o-Mini'],  # This will be dynamically updated
#                'uiSelectionAction': 'binassist_refresh_providers'
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
