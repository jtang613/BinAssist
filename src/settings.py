import json
from binaryninja.settings import Settings
from .exceptions import RegisterSettingsGroupException, RegisterSettingsKeyException

class BinAssistSettings(Settings):
    """
    Manages the configuration settings for the BinAssist plugin, including API keys, model settings, 
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
            'binassist.remote_host': {
                'title': 'Remote API Host',
                'description': 'The API host endpoint used to make requests.',
                'type': 'string',
                'default': 'https://api.openai.com/v1'
            },
            'binassist.api_key': {
                'title': 'API Key',
                'description': 'The API key used to make requests.',
                'type': 'string',
                'default': None,
                'ignore': ["SettingsProjectScope", "SettingsResourceScope"],
                'hidden': True
            },
            'binassist.model': {
                'title': 'LLM Model',
                'description': 'The LLM model used to generate the response.',
                'type': 'string',
                'default': 'gpt-4o-mini'
            },
            'binassist.rlhf_db': {
                'title': 'RLHF Database Path',
                'description': 'The path to store the RLHF database.',
                'type': 'string',
                'default': 'rlhf_feedback.db',
                'uiSelectionAction': 'file'
            },
            'binassist.max_tokens': {
                'title': 'Max Completion Tokens',
                'description': 'The maximum number of tokens used for completion.',
                'type': 'number',
                'default': 8192,
                'minValue': 1,
                'maxValue': 128*1024
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
            if 'minValue' in properties and 'maxValue' in properties:
                properties['message'] = f"Min: {properties['minValue']}, Max: {properties['maxValue']}"
            self.register_setting(key, json.dumps(properties))
            