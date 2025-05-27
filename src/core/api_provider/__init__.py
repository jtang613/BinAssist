"""
API Provider abstraction layer for BinAssist.

This module provides abstract base classes and interfaces for different types of
API providers, following capability-based design patterns.
"""

from .base_provider import APIProvider
from .capabilities import ChatProvider, FunctionCallingProvider, EmbeddingProvider, ModelListProvider
from .factory import APIProviderFactory, ProviderRegistry
from .config import APIProviderConfig, ProviderType
from .retry_handler import RetryHandler
from .exceptions import APIProviderError, AuthenticationError, RateLimitError, NetworkError

__all__ = [
    'APIProvider',
    'ChatProvider', 
    'FunctionCallingProvider',
    'EmbeddingProvider',
    'ModelListProvider',
    'APIProviderFactory',
    'ProviderRegistry',
    'APIProviderConfig',
    'ProviderType',
    'RetryHandler',
    'APIProviderError',
    'AuthenticationError', 
    'RateLimitError',
    'NetworkError'
]