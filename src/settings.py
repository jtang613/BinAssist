# Import new settings system
from .core.settings import get_settings_manager, migrate_from_binary_ninja_settings

class BinAssistSettings:
    """
    Manages the configuration settings for the BinAssist plugin, including API providers, RAG settings,
    and other preferences that need to be stored and retrieved across sessions.
    """

    def __init__(self) -> None:
        """
        Initializes the settings instance and performs migration if needed.
        """
        # Get the settings manager (migration happens automatically in SettingsManager)
        self.settings_manager = get_settings_manager()
        
        # Perform migration if needed (but don't fail if it doesn't work)
        try:
            migrate_from_binary_ninja_settings(backup_existing=True)
        except Exception as e:
            print(f"Settings migration skipped: {e}")
        
        print("BinAssist settings initialized with SQLite backend")

    def _register_settings(self) -> None:
        """
        Legacy method - settings are now managed by SQLite backend.
        """
        # Settings are automatically registered by the SettingsManager
        pass

    def _register_ui_actions(self):
        """
        Legacy method - UI actions no longer needed with new settings system.
        """
        pass

    def get_provider_config(self, provider_name: str):
        """
        Get configuration for a specific provider using new settings system.
        """
        providers = self.settings_manager.get_json('api_providers', [])
        
        for provider in providers:
            if provider.get('name') == provider_name:
                return {
                    'api___name': provider.get('name', ''),
                    'provider_type': provider.get('provider_type', 'openai'),
                    'api__host': provider.get('base_url', 'https://api.openai.com/v1'),
                    'api_key': provider.get('api_key', ''),
                    'api__model': provider.get('model', 'gpt-4o-mini'),
                    'api__max_tokens': provider.get('max_tokens', 16384)
                }
        
        # Return default if not found
        return {
            'api___name': 'Default',
            'provider_type': 'openai',
            'api__host': 'https://api.openai.com/v1',
            'api_key': '',
            'api__model': 'gpt-4o-mini',
            'api__max_tokens': 16384
        }