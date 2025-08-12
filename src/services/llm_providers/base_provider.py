#!/usr/bin/env python3

"""
Base LLM Provider - Abstract interface for all LLM providers
Defines the contract that all providers must implement
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any, Optional, Callable
from ..models.llm_models import (
    ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse, 
    ProviderCapabilities, ToolCall, ToolResult
)
from ..models.provider_types import ProviderType


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    Defines the interface that all providers must implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize provider with configuration from settings service.
        
        Args:
            config: Provider configuration dictionary containing:
                - name: Provider name
                - model: Model identifier
                - url: API endpoint URL
                - api_key: API key (if required)
                - max_tokens: Maximum tokens per request
                - disable_tls: Whether to disable TLS verification
                - provider_type: Type of provider (from ProviderType enum)
        """
        self.config = config
        self.name = config.get('name', 'Unknown')
        self.model = config.get('model', '')
        self.url = config.get('url', '')
        self.api_key = config.get('api_key', '')
        self.max_tokens = config.get('max_tokens', 4096)
        self.disable_tls = config.get('disable_tls', False)
        self.provider_type = config.get('provider_type', 'openai')
    
    @abstractmethod
    async def chat_completion(self, request: ChatRequest, 
                            native_message_callback: Optional[Callable[[Dict[str, Any], ProviderType], None]] = None) -> ChatResponse:
        """
        Generate a chat completion (non-streaming).
        
        Args:
            request: Chat completion request
            native_message_callback: Optional callback for provider-native message updates
                                   Signature: callback(native_message: Dict[str, Any], provider_type: ProviderType)
            
        Returns:
            Chat completion response
            
        Raises:
            APIProviderError: On API errors
            AuthenticationError: On authentication failures
            RateLimitError: On rate limit exceeded
            NetworkError: On network issues
        """
        pass
    
    @abstractmethod
    async def chat_completion_stream(self, request: ChatRequest, 
                                   native_message_callback: Optional[Callable[[Dict[str, Any], ProviderType], None]] = None) -> AsyncGenerator[ChatResponse, None]:
        """
        Generate a streaming chat completion.
        
        Args:
            request: Chat completion request (with stream=True)
            native_message_callback: Optional callback for provider-native message updates
                                   Signature: callback(native_message: Dict[str, Any], provider_type: ProviderType)
            
        Yields:
            Partial chat completion responses
            
        Raises:
            APIProviderError: On API errors
            AuthenticationError: On authentication failures
            RateLimitError: On rate limit exceeded
            NetworkError: On network issues
        """
        pass
    
    @abstractmethod
    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate text embeddings.
        
        Args:
            request: Embedding generation request
            
        Returns:
            Embedding response with vectors
            
        Raises:
            APIProviderError: On API errors
            AuthenticationError: On authentication failures
            RateLimitError: On rate limit exceeded
            NetworkError: On network issues
            NotImplementedError: If provider doesn't support embeddings
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test connectivity to the provider.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """
        Get provider capabilities and supported features.
        
        Returns:
            Provider capabilities object
        """
        pass
    
    @abstractmethod
    def get_provider_type(self) -> ProviderType:
        """
        Get the provider type for this provider.
        
        Returns:
            ProviderType enum value
        """
        pass
    
    # ===================================================================
    # Tool Support Methods
    # ===================================================================
    
    def supports_tools(self) -> bool:
        """
        Check if this provider supports tool calling.
        
        Returns:
            True if provider supports tools, False otherwise
        """
        capabilities = self.get_capabilities()
        return capabilities.supports_tools
    
    def prepare_tool_enabled_request(self, request: ChatRequest, tools: List[Dict[str, Any]]) -> ChatRequest:
        """
        Prepare a chat request with tool definitions enabled.
        
        Args:
            request: Original chat request
            tools: List of tool definitions in OpenAI format
            
        Returns:
            Modified chat request with tools enabled
            
        Note:
            Default implementation adds tools to request. Providers can override
            for provider-specific tool formatting.
        """
        if not self.supports_tools():
            return request
        
        # Create a copy of the request with tools added
        tool_request = ChatRequest(
            messages=request.messages,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
            tools=tools,
            tool_choice="auto",  # Let LLM decide when to use tools
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            user=request.user
        )
        
        return tool_request
    
    def parse_tool_calls(self, response: ChatResponse) -> List[ToolCall]:
        """
        Parse tool calls from a chat response.
        
        Args:
            response: Chat response that may contain tool calls
            
        Returns:
            List of parsed tool calls
            
        Note:
            Default implementation returns response.tool_calls if present.
            Providers can override for provider-specific parsing.
        """
        if response.tool_calls:
            return response.tool_calls
        return []
    
    def get_stop_reason(self, response: ChatResponse) -> str:
        """
        Get the stop reason from a chat response.
        
        Args:
            response: Chat response
            
        Returns:
            Stop reason string ("stop", "tool_calls", "length", etc.)
        """
        return response.finish_reason
    
    def format_tool_results_for_continuation(self, tool_calls: List[ToolCall], tool_results: List[str]) -> List[Dict[str, Any]]:
        """
        Format tool execution results for LLM conversation continuation.
        
        Args:
            tool_calls: Original tool calls from LLM
            tool_results: Results from tool execution
            
        Returns:
            List of message dictionaries for LLM continuation
            
        Note:
            Default implementation creates tool messages in OpenAI format.
            Providers can override for provider-specific formatting.
        """
        messages = []
        
        for tool_call, result in zip(tool_calls, tool_results):
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.id,
                "name": tool_call.name
            })
        
        return messages
    
    def has_tool_capability_changed(self, tools: List[Dict[str, Any]]) -> bool:
        """
        Check if tool capability requirements have changed.
        
        Args:
            tools: List of tool definitions
            
        Returns:
            True if provider needs to update tool capabilities
        """
        # Default implementation: no capability changes needed
        return False
    
    def validate_config(self) -> bool:
        """
        Validate provider configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.name:
            raise ValueError("Provider name is required")
        if not self.model:
            raise ValueError("Model is required")
        if not self.url:
            raise ValueError("URL is required")
        
        return True
    
    def __str__(self) -> str:
        """String representation of provider"""
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.model}')"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"{self.__class__.__name__}("
                f"name='{self.name}', "
                f"model='{self.model}', "
                f"url='{self.url}', "
                f"provider_type='{self.provider_type}')")


class LLMProviderError(Exception):
    """Base exception for LLM provider errors"""
    pass


class APIProviderError(LLMProviderError):
    """General API provider error"""
    pass


class AuthenticationError(LLMProviderError):
    """Authentication/authorization error"""
    pass


class RateLimitError(LLMProviderError):
    """Rate limit exceeded error"""
    pass


class NetworkError(LLMProviderError):
    """Network connectivity error"""
    pass