"""
Configuration manager for BinAssist.
"""

from typing import Dict, Any, List, Optional
import json
import os
from dataclasses import dataclass, asdict
from binaryninja.settings import Settings

from ..core.api_provider.config import APIProviderConfig, ProviderType


@dataclass 
class BinAssistConfig:
    """
    Main configuration for BinAssist.
    
    Attributes:
        api_providers: List of API provider configurations
        active_provider: Name of the currently active provider
        rlhf_db_path: Path to RLHF database
        rag_db_path: Path to RAG database
        use_rag: Whether to use RAG by default
        default_system_prompt: Default system prompt for queries
        ui_settings: UI-specific settings
    """
    api_providers: List[APIProviderConfig]
    active_provider: str
    rlhf_db_path: str = "rlhf_feedback.db"
    rag_db_path: str = "binassist_rag_db"
    use_rag: bool = False
    default_system_prompt: str = ""
    ui_settings: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.ui_settings is None:
            self.ui_settings = {}
        
        if not self.default_system_prompt:
            self.default_system_prompt = '''
You are a professional software reverse engineer specializing in cybersecurity. You are intimately 
familiar with x86_64, ARM, PPC and MIPS architectures. You are an expert C and C++ developer.
You are an expert Python and Rust developer. You are familiar with common frameworks and libraries 
such as WinSock, OpenSSL, MFC, etc. You are an expert in TCP/IP network programming and packet analysis.
You always respond to queries in a structured format using Markdown styling for headings and lists. 
You format code blocks using back-tick code-fencing.
'''.strip()


class ConfigManager:
    """
    Manages configuration for BinAssist using Binary Ninja's settings system.
    
    This class provides a higher-level interface over Binary Ninja's settings
    and handles JSON serialization/deserialization of complex configuration.
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.settings = Settings()
        self._ensure_default_config()
    
    def get_config(self) -> BinAssistConfig:
        """
        Get the current configuration.
        
        Returns:
            BinAssistConfig object with current settings
        """
        try:
            # Get API providers
            providers_json = self.settings.get_string('binassist.api_providers')
            providers_data = json.loads(providers_json)
            
            api_providers = []
            for provider_data in providers_data:
                # Convert to APIProviderConfig
                config = APIProviderConfig.from_dict(provider_data)
                api_providers.append(config)
            
            # Get other settings
            active_provider = self.settings.get_string('binassist.active_provider')
            rlhf_db_path = self.settings.get_string('binassist.rlhf_db')
            rag_db_path = self.settings.get_string('binassist.rag_db_path')
            use_rag = self.settings.get_bool('binassist.use_rag')
            
            # Get UI settings if they exist
            ui_settings = {}
            try:
                ui_json = self.settings.get_json('binassist.ui_settings')
                ui_settings = json.loads(ui_json)
            except:
                pass
            
            return BinAssistConfig(
                api_providers=api_providers,
                active_provider=active_provider,
                rlhf_db_path=rlhf_db_path,
                rag_db_path=rag_db_path,
                use_rag=use_rag,
                ui_settings=ui_settings
            )
            
        except Exception as e:
            # Return default config on error
            return self._get_default_config()
    
    def save_config(self, config: BinAssistConfig) -> None:
        """
        Save the configuration.
        
        Args:
            config: Configuration to save
        """
        try:
            # Convert API providers to JSON
            providers_data = [provider.to_dict() for provider in config.api_providers]
            providers_json = json.dumps(providers_data)
            
            # Save settings
            self.settings.set_json('binassist.api_providers', providers_json)
            self.settings.set_string('binassist.active_provider', config.active_provider)
            self.settings.set_string('binassist.rlhf_db', config.rlhf_db_path)
            self.settings.set_string('binassist.rag_db_path', config.rag_db_path)
            self.settings.set_bool('binassist.use_rag', config.use_rag)
            
            # Save UI settings
            if config.ui_settings:
                ui_json = json.dumps(config.ui_settings)
                self.settings.set_json('binassist.ui_settings', ui_json)
                
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration: {e}")
    
    def get_active_provider_config(self) -> Optional[APIProviderConfig]:
        """
        Get the configuration for the currently active provider.
        
        Returns:
            APIProviderConfig for the active provider, or None if not found
        """
        config = self.get_config()
        
        for provider in config.api_providers:
            if provider.name == config.active_provider:
                return provider
        
        return None
    
    def add_provider(self, provider_config: APIProviderConfig) -> None:
        """
        Add a new API provider.
        
        Args:
            provider_config: Provider configuration to add
        """
        config = self.get_config()
        
        # Check if provider with same name already exists
        for i, existing in enumerate(config.api_providers):
            if existing.name == provider_config.name:
                config.api_providers[i] = provider_config
                break
        else:
            config.api_providers.append(provider_config)
        
        # If this is the first provider, make it active
        if not config.active_provider:
            config.active_provider = provider_config.name
        
        self.save_config(config)
    
    def remove_provider(self, provider_name: str) -> None:
        """
        Remove an API provider.
        
        Args:
            provider_name: Name of the provider to remove
        """
        config = self.get_config()
        
        # Remove the provider
        config.api_providers = [
            p for p in config.api_providers 
            if p.name != provider_name
        ]
        
        # If this was the active provider, reset to first available
        if config.active_provider == provider_name:
            if config.api_providers:
                config.active_provider = config.api_providers[0].name
            else:
                config.active_provider = ""
        
        self.save_config(config)
    
    def set_active_provider(self, provider_name: str) -> None:
        """
        Set the active API provider.
        
        Args:
            provider_name: Name of the provider to activate
        """
        config = self.get_config()
        
        # Verify the provider exists
        provider_exists = any(p.name == provider_name for p in config.api_providers)
        if not provider_exists:
            raise ValueError(f"Provider '{provider_name}' not found")
        
        config.active_provider = provider_name
        self.save_config(config)
    
    def _ensure_default_config(self) -> None:
        """Ensure default configuration exists."""
        try:
            # Check if configuration exists by trying to get the settings
            providers_json = self.settings.get_string('binassist.api_providers')
            if not providers_json or providers_json.strip() == '':
                raise ValueError("Empty providers configuration")
            # Try to parse it
            json.loads(providers_json)
        except:
            # Settings don't exist or are invalid, use the defaults that were registered
            print("Using default API providers configuration from settings registration")
    
    def _get_default_config(self) -> BinAssistConfig:
        """Get default configuration."""
        openai_provider = APIProviderConfig(
            name="GPT-4o-Mini",
            provider_type=ProviderType.OPENAI,
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            max_tokens=16384,
            api_key="",
            timeout=120
        )
        
        anthropic_provider = APIProviderConfig(
            name="Claude-3.5-Sonnet",
            provider_type=ProviderType.ANTHROPIC,
            base_url="https://api.anthropic.com",
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            api_key="",
            timeout=120
        )
        
        ollama_provider = APIProviderConfig(
            name="Ollama-Local",
            provider_type=ProviderType.OLLAMA,
            base_url="http://localhost:11434/v1",
            model="llama3.1:8b",
            max_tokens=4096,
            api_key="",  # Not needed for Ollama
            timeout=120
        )
        
        return BinAssistConfig(
            api_providers=[openai_provider, anthropic_provider, ollama_provider],
            active_provider="GPT-4o-Mini"
        )