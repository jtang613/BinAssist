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
        if not self.register_group('binassist', 'BinAssist'):
            raise RegisterSettingsGroupException('Failed to register BinAssist settings group.')

        settings_definitions = [
            ('binassist.remote_host', 'Remote API Host', 'The API host endpoint used to make requests.', 'string', None),
            ('binassist.api_key', 'API Key', 'The API key used to make requests.', 'string', None),
            ('binassist.model', 'LLM Model', 'The LLM model used to generate the response.', 'string', 'gpt-4-turbo'),
            ('binassist.rlhf_db', 'RLHF Database Path', 'The to store the RLHF database.', 'string', 'rlhf_feedback.db'),
            ('binassist.max_tokens', 'Max Completion Tokens', 'The maximum number of tokens used for completion.', 'number', 8192, 1, 32768),
        ]

        for setting in settings_definitions:
            if len(setting) == 5:
                key, title, description, setting_type, default = setting
                min_value, max_value = None, None
            elif len(setting) == 7:
                key, title, description, setting_type, default, min_value, max_value = setting

            properties = {
                'title': title,
                'type': setting_type,
                'description': description
            }
            if default is not None:
                properties['default'] = default
            if min_value is not None and max_value is not None:
                properties['minValue'] = min_value
                properties['maxValue'] = max_value
                properties['message'] = f"Min: {min_value}, Max: {max_value}"
            if not self.register_setting(key, json.dumps(properties)):
                raise RegisterSettingsKeyException(f'Failed to register {title.lower()} setting.')
