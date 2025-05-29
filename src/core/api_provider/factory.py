"""
Factory pattern for creating API providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, List
import threading

from binaryninja import log
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
            log.log_info(f"[BinAssist] Initializing ProviderRegistry")
            self._factories: Dict[ProviderType, APIProviderFactory] = {}
            self._initialized = True
            log.log_debug(f"[BinAssist] Registry initialized, registering default factories")
            self._register_default_factories()
            log.log_info(f"[BinAssist] Registry initialization complete. Factories: {list(self._factories.keys())}")
    
    def register_factory(self, factory: APIProviderFactory) -> None:
        """
        Register a provider factory.
        
        Args:
            factory: The factory to register
        """
        log.log_debug(f"[BinAssist] Registering factory: {type(factory).__name__}")
        with self._lock:
            for provider_type in factory.get_supported_types():
                log.log_debug(f"[BinAssist] Registering {type(factory).__name__} for provider type: {provider_type}")
                self._factories[provider_type] = factory
        log.log_debug(f"[BinAssist] Factory registration complete. Total factories: {len(self._factories)}")
    
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
        log.log_debug(f"[BinAssist] create_provider called for type: {config.provider_type}")
        log.log_debug(f"[BinAssist] Available factories: {list(self._factories.keys())}")
        log.log_debug(f"[BinAssist] Looking for factory for: {config.provider_type}")
        
        factory = self._factories.get(config.provider_type)
        if factory is None:
            log.log_error(f"[BinAssist] No factory found for provider type: {config.provider_type}")
            log.log_error(f"[BinAssist] Available factories: {self._factories}")
            raise ValueError(f"No factory registered for provider type: {config.provider_type}")
        
        log.log_debug(f"[BinAssist] Found factory: {type(factory).__name__}")
        provider = factory.create_provider(config)
        log.log_debug(f"[BinAssist] Created provider: {type(provider).__name__}")
        return provider
    
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
        log.log_debug(f"[BinAssist] _register_default_factories starting")
        
        # Import here to avoid circular imports
        try:
            log.log_debug(f"[BinAssist] Attempting to import OpenAIProviderFactory")
            from .providers.openai_provider import OpenAIProviderFactory
            log.log_debug(f"[BinAssist] OpenAIProviderFactory imported successfully")
            factory = OpenAIProviderFactory()
            log.log_debug(f"[BinAssist] OpenAIProviderFactory instantiated: {factory}")
            self.register_factory(factory)
            log.log_debug(f"[BinAssist] OpenAIProviderFactory registered")
        except ImportError as e:
            log.log_error(f"[BinAssist] Failed to import OpenAIProviderFactory: {e}")
        except Exception as e:
            log.log_error(f"[BinAssist] Error with OpenAIProviderFactory: {e}")
        
        try:
            log.log_debug(f"[BinAssist] Attempting to import AnthropicProviderFactory")
            from .providers.anthropic_provider import AnthropicProviderFactory
            log.log_debug(f"[BinAssist] AnthropicProviderFactory imported successfully")
            factory = AnthropicProviderFactory()
            log.log_debug(f"[BinAssist] AnthropicProviderFactory instantiated: {factory}")
            self.register_factory(factory)
            log.log_debug(f"[BinAssist] AnthropicProviderFactory registered")
        except ImportError as e:
            log.log_error(f"[BinAssist] Failed to import AnthropicProviderFactory: {e}")
        except Exception as e:
            log.log_error(f"[BinAssist] Error with AnthropicProviderFactory: {e}")
        
        try:
            log.log_debug(f"[BinAssist] Attempting to import OllamaProviderFactory")
            from .providers.ollama_provider import OllamaProviderFactory
            log.log_debug(f"[BinAssist] OllamaProviderFactory imported successfully")
            factory = OllamaProviderFactory()
            log.log_debug(f"[BinAssist] OllamaProviderFactory instantiated: {factory}")
            self.register_factory(factory)
            log.log_debug(f"[BinAssist] OllamaProviderFactory registered")
        except ImportError as e:
            log.log_error(f"[BinAssist] Failed to import OllamaProviderFactory: {e}")
        except Exception as e:
            log.log_error(f"[BinAssist] Error with OllamaProviderFactory: {e}")
        
        log.log_debug(f"[BinAssist] _register_default_factories complete")


# Global registry instance
provider_registry = ProviderRegistry()