"""
Ollama API provider implementation.
"""

from typing import List

from ..base_provider import BaseOpenAICompatibleProvider
from ..factory import APIProviderFactory
from ..config import APIProviderConfig, ProviderType


class OllamaProvider(BaseOpenAICompatibleProvider):
    """
    Ollama API provider.
    
    Supports chat completions using Ollama's OpenAI-compatible API.
    Note: Function calling support depends on the model being used.
    """
    
    def __init__(self, config: APIProviderConfig):
        """Initialize Ollama provider."""
        super().__init__(config)
        
        # Ollama typically runs locally and doesn't require API keys
        # Set default base URL if not provided
        if not config.base_url:
            config.base_url = "http://localhost:11434/v1"


class OllamaProviderFactory(APIProviderFactory):
    """Factory for creating Ollama providers."""
    
    def create_provider(self, config: APIProviderConfig) -> OllamaProvider:
        """Create an Ollama provider instance."""
        return OllamaProvider(config)
    
    def supports(self, provider_type: ProviderType) -> bool:
        """Check if this factory supports the provider type."""
        return provider_type == ProviderType.OLLAMA
    
    def get_supported_types(self) -> List[ProviderType]:
        """Get supported provider types."""
        return [ProviderType.OLLAMA]