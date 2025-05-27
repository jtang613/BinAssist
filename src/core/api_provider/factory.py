"""
Factory pattern for creating API providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, List
import threading

from .config import APIProviderConfig, ProviderType
from .base_provider import APIProvider


class APIProviderFactory(ABC):
    """Abstract factory for creating API providers."""
    
    @abstractmethod
    def create_provider(self, config: APIProviderConfig) -> APIProvider:
        """
        Create a provider instance.
        
        Args:
            config: Provider configuration
            
        Returns:
            APIProvider instance
        """
        pass
    
    @abstractmethod
    def supports(self, provider_type: ProviderType) -> bool:
        """
        Check if this factory supports the given provider type.
        
        Args:
            provider_type: The provider type
            
        Returns:
            True if supported
        """
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[ProviderType]:
        """
        Get list of supported provider types.
        
        Returns:
            List of supported provider types
        """
        pass


class ProviderRegistry:
    """
    Thread-safe registry for API provider factories.
    
    This class manages the registration and creation of API providers
    using the factory pattern.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the registry."""
        if not hasattr(self, '_initialized'):
            self._factories: Dict[ProviderType, APIProviderFactory] = {}
            self._initialized = True
            self._register_default_factories()
    
    def register_factory(self, factory: APIProviderFactory) -> None:
        """
        Register a provider factory.
        
        Args:
            factory: The factory to register
        """
        with self._lock:
            for provider_type in factory.get_supported_types():
                self._factories[provider_type] = factory
    
    def create_provider(self, config: APIProviderConfig) -> APIProvider:
        """
        Create a provider using the appropriate factory.
        
        Args:
            config: Provider configuration
            
        Returns:
            APIProvider instance
            
        Raises:
            ValueError: If no factory supports the provider type
        """
        factory = self._factories.get(config.provider_type)
        if factory is None:
            raise ValueError(f"No factory registered for provider type: {config.provider_type}")
        
        return factory.create_provider(config)
    
    def get_supported_types(self) -> List[ProviderType]:
        """
        Get list of all supported provider types.
        
        Returns:
            List of supported provider types
        """
        with self._lock:
            return list(self._factories.keys())
    
    def is_supported(self, provider_type: ProviderType) -> bool:
        """
        Check if a provider type is supported.
        
        Args:
            provider_type: The provider type to check
            
        Returns:
            True if supported
        """
        return provider_type in self._factories
    
    def _register_default_factories(self) -> None:
        """Register default provider factories."""
        # Import here to avoid circular imports
        try:
            from .providers.openai_provider import OpenAIProviderFactory
            self.register_factory(OpenAIProviderFactory())
        except ImportError:
            pass
        
        try:
            from .providers.anthropic_provider import AnthropicProviderFactory
            self.register_factory(AnthropicProviderFactory())
        except ImportError:
            pass
        
        try:
            from .providers.ollama_provider import OllamaProviderFactory
            self.register_factory(OllamaProviderFactory())
        except ImportError:
            pass


# Global registry instance
provider_registry = ProviderRegistry()