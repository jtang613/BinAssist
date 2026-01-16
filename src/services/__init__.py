#!/usr/bin/env python3

"""
Services Module - Core business logic services

This module provides the main services for the BinAssist plugin:
- SettingsService: Persistent configuration management
- LLMService: Language model operations and provider management
- ServiceRegistry: Dependency injection and service lifecycle management
"""

from .settings_service import SettingsService, settings_service
from .llm_service import LLMService
from .service_registry import get_service_registry, reset_service_registry, ServiceRegistry

# Re-export commonly used models
from .models.provider_types import ProviderType
from .models.llm_models import (
    ChatMessage, MessageRole, ChatRequest, ChatResponse,
    EmbeddingRequest, EmbeddingResponse, ProviderCapabilities,
    ToolCall, Usage
)

# Re-export provider interfaces for extensibility
from .llm_providers.base_provider import (
    BaseLLMProvider, LLMProviderError, APIProviderError,
    AuthenticationError, RateLimitError, NetworkError
)
from .llm_providers.provider_factory import get_provider_factory


__all__ = [
    # Legacy exports (for backward compatibility)
    'SettingsService',
    'settings_service',
    
    # New architecture
    'LLMService', 
    'ServiceRegistry',
    'get_service_registry',
    'reset_service_registry',
    
    # Models and enums
    'ProviderType',
    'ChatMessage',
    'MessageRole', 
    'ChatRequest',
    'ChatResponse',
    'EmbeddingRequest',
    'EmbeddingResponse',
    'ProviderCapabilities',
    'ToolCall',
    'Usage',
    
    # Provider interfaces
    'BaseLLMProvider',
    'LLMProviderError',
    'APIProviderError',
    'AuthenticationError',
    'RateLimitError',
    'NetworkError',
    'get_provider_factory',
]