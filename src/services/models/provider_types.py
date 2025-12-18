#!/usr/bin/env python3

"""
Provider Types Enum - Central definition of supported LLM provider types
"""

from enum import Enum


class ProviderType(Enum):
    """
    Enumeration of supported LLM provider types.
    Used throughout the plugin for type safety and consistency.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OPENWEBUI = "openwebui"
    LMSTUDIO = "lmstudio"
    LITELLM = "litellm"
    
    @classmethod
    def get_display_names(cls) -> dict[str, str]:
        """Get human-readable display names for each provider type"""
        return {
            cls.OPENAI: "OpenAI",
            cls.ANTHROPIC: "Anthropic (Claude)",
            cls.OLLAMA: "Ollama (Local)",
            cls.OPENWEBUI: "Open WebUI",
            cls.LMSTUDIO: "LM Studio",
            cls.LITELLM: "LiteLLM (Proxy)"
        }
    
    @classmethod
    def get_default_urls(cls) -> dict[str, str]:
        """Get default API URLs for each provider type"""
        return {
            cls.OPENAI: "https://api.openai.com/v1",
            cls.ANTHROPIC: "https://api.anthropic.com",
            cls.OLLAMA: "http://localhost:11434",
            cls.OPENWEBUI: "http://localhost:3000",
            cls.LMSTUDIO: "http://localhost:1234/v1",
            cls.LITELLM: "http://localhost:4000"
        }
    
    @classmethod
    def get_default_models(cls) -> dict[str, list[str]]:
        """Get default/popular models for each provider type"""
        return {
            cls.OPENAI: [
                "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"
            ],
            cls.ANTHROPIC: [
                "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229", "claude-3-sonnet-20240229"
            ],
            cls.OLLAMA: [
                "llama3.1:8b", "llama3.1:70b", "codellama:7b", "mistral:7b",
                "phi3:mini", "qwen2:7b"
            ],
            cls.OPENWEBUI: [
                # OpenWebUI uses models from connected providers
                "gpt-4o", "claude-3-5-sonnet", "llama3.1:8b"
            ],
            cls.LMSTUDIO: [
                # LM Studio uses locally downloaded models
                "microsoft/DialoGPT-medium", "microsoft/DialoGPT-large"
            ],
            cls.LITELLM: [
                # LiteLLM proxies to various providers - models are dynamic
                # Examples: bedrock/anthropic.claude-*, bedrock/amazon.nova-*, etc.
            ]
        }
    
    @classmethod
    def supports_embeddings(cls, provider_type: 'ProviderType') -> bool:
        """Check if provider type supports embedding generation"""
        return provider_type in {
            cls.OPENAI,     # text-embedding-3-small, text-embedding-3-large
            cls.OLLAMA,     # Various embedding models available
            cls.OPENWEBUI,  # Depends on connected providers
            cls.LMSTUDIO,   # Depends on loaded models
            cls.LITELLM     # Can proxy embeddings to various providers
        }
    
    @classmethod
    def supports_tool_calls(cls, provider_type: 'ProviderType') -> bool:
        """Check if provider type supports function/tool calling"""
        return provider_type in {
            cls.OPENAI,     # Full tool calling support
            cls.ANTHROPIC,  # Tool calling support
            cls.OLLAMA,     # Limited tool calling support
            cls.OPENWEBUI,  # Depends on connected providers
            cls.LMSTUDIO,   # Limited support
            cls.LITELLM     # Proxies tool calls to various providers
        }
    
    @classmethod
    def supports_streaming(cls, provider_type: 'ProviderType') -> bool:
        """Check if provider type supports streaming responses"""
        return provider_type in {
            cls.OPENAI,     # Full streaming support
            cls.ANTHROPIC,  # Full streaming support
            cls.OLLAMA,     # Full streaming support
            cls.OPENWEBUI,  # Full streaming support
            cls.LMSTUDIO,   # Full streaming support
            cls.LITELLM     # Full streaming support via proxy
        }
    
    @classmethod
    def requires_api_key(cls, provider_type: 'ProviderType') -> bool:
        """Check if provider type requires an API key"""
        return provider_type in {
            cls.OPENAI,     # Requires API key
            cls.ANTHROPIC   # Requires API key
            # Local providers (Ollama, LMStudio) typically don't require keys
            # OpenWebUI may or may not depending on configuration
        }
    
    @property
    def display_name(self) -> str:
        """Get human-readable display name for this provider type"""
        return self.get_display_names()[self]
    
    @property  
    def default_url(self) -> str:
        """Get default API URL for this provider type"""
        return self.get_default_urls()[self]
    
    @property
    def default_models(self) -> list[str]:
        """Get default models for this provider type"""
        return self.get_default_models()[self]
    
    def __str__(self) -> str:
        """String representation uses the enum value"""
        return self.value