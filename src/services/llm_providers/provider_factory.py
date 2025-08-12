#!/usr/bin/env python3

"""
Provider Factory - Factory pattern for creating LLM providers
Handles dynamic provider instantiation based on configuration
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from .base_provider import BaseLLMProvider, LLMProviderError
from ..models.provider_types import ProviderType


class ProviderFactory(ABC):
    """Abstract factory for creating LLM providers"""
    
    @abstractmethod
    def create_provider(self, config: Dict[str, Any]) -> BaseLLMProvider:
        """Create a provider instance from configuration"""
        pass
    
    @abstractmethod
    def supports_provider_type(self, provider_type: ProviderType) -> bool:
        """Check if this factory supports the given provider type"""
        pass


class LLMProviderFactory:
    """
    Main factory for creating LLM providers.
    
    Uses registry pattern to support multiple provider types.
    """
    
    def __init__(self):
        """Initialize factory with empty registry"""
        self._factories: Dict[ProviderType, ProviderFactory] = {}
        self._register_default_factories()
    
    def register_factory(self, provider_type: ProviderType, factory: ProviderFactory):
        """
        Register a provider factory for a specific type
        
        Args:
            provider_type: Type of provider this factory creates
            factory: Factory instance to register
        """
        self._factories[provider_type] = factory
    
    def create_provider(self, config: Dict[str, Any]) -> BaseLLMProvider:
        """
        Create provider instance from configuration
        
        Args:
            config: Provider configuration dictionary containing:
                - provider_type: Type of provider (ProviderType enum value)
                - name, model, url, api_key, etc.
        
        Returns:
            Initialized provider instance
            
        Raises:
            LLMProviderError: If provider type not supported or creation fails
        """
        # Extract and validate provider type
        provider_type_str = config.get('provider_type', 'openai')
        
        try:
            if isinstance(provider_type_str, str):
                provider_type = ProviderType(provider_type_str)
            else:
                provider_type = provider_type_str
        except ValueError:
            raise LLMProviderError(f"Unsupported provider type: {provider_type_str}")
        
        # Find appropriate factory
        if provider_type not in self._factories:
            raise LLMProviderError(f"No factory registered for provider type: {provider_type}")
        
        factory = self._factories[provider_type]
        
        try:
            return factory.create_provider(config)
        except Exception as e:
            raise LLMProviderError(f"Failed to create provider: {e}") from e
    
    def get_supported_types(self) -> list[ProviderType]:
        """Get list of supported provider types"""
        return list(self._factories.keys())
    
    def is_supported(self, provider_type: ProviderType) -> bool:
        """Check if provider type is supported"""
        return provider_type in self._factories
    
    def _register_default_factories(self):
        """Register default provider factories"""
        # Import and register available provider factories
        
        # Anthropic provider
        try:
            from .anthropic_provider import AnthropicProviderFactory
            self.register_factory(
                ProviderType.ANTHROPIC,
                AnthropicProviderFactory()
            )
        except ImportError:
            pass  # Anthropic provider not available
        
        # OpenAI provider
        try:
            from .openai_provider import OpenAIProviderFactory
            self.register_factory(
                ProviderType.OPENAI,
                OpenAIProviderFactory()
            )
        except ImportError:
            pass  # OpenAI provider not available
        
        # Ollama provider
        try:
            from .ollama_provider import OllamaProviderFactory
            self.register_factory(
                ProviderType.OLLAMA,
                OllamaProviderFactory()
            )
        except ImportError:
            pass  # Ollama provider not available


class AnthropicProviderFactory(ProviderFactory):
    """Factory for creating Anthropic providers"""
    
    def create_provider(self, config: Dict[str, Any]) -> BaseLLMProvider:
        """Create Anthropic provider instance"""
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(config)
    
    def supports_provider_type(self, provider_type: ProviderType) -> bool:
        """Check if this factory supports Anthropic providers"""
        return provider_type == ProviderType.ANTHROPIC


# Future provider factories will be added here as they are implemented

# class OpenAIProviderFactory(ProviderFactory):
#     """Factory for creating OpenAI providers"""
#     
#     def create_provider(self, config: Dict[str, Any]) -> BaseLLMProvider:
#         from .openai_provider import OpenAIProvider
#         return OpenAIProvider(config)
#     
#     def supports_provider_type(self, provider_type: ProviderType) -> bool:
#         return provider_type == ProviderType.OPENAI


# class OllamaProviderFactory(ProviderFactory):
#     """Factory for creating Ollama providers"""
#     
#     def create_provider(self, config: Dict[str, Any]) -> BaseLLMProvider:
#         from .ollama_provider import OllamaProvider
#         return OllamaProvider(config)
#     
#     def supports_provider_type(self, provider_type: ProviderType) -> bool:
#         return provider_type == ProviderType.OLLAMA


# Global factory instance
_provider_factory = None

def get_provider_factory() -> LLMProviderFactory:
    """Get the global provider factory instance"""
    global _provider_factory
    if _provider_factory is None:
        _provider_factory = LLMProviderFactory()
    return _provider_factory