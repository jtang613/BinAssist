"""
Abstract base class for API providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Union
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import json

from .capabilities import ChatProvider, FunctionCallingProvider, EmbeddingProvider, ModelListProvider
from .config import APIProviderConfig
from .exceptions import APIProviderError, AuthenticationError, RateLimitError, NetworkError
from .retry_handler import RetryHandler
from ..models.chat_message import ChatMessage
from ..models.tool_call import ToolCall
from ..models.api_response import APIResponse


class APIProvider(ABC):
    """
    Abstract base class for all API providers.
    
    This class provides common functionality and defines the interface
    that all providers must implement. Providers can implement specific
    capability interfaces as needed.
    """
    
    def __init__(self, config: APIProviderConfig):
        """
        Initialize the provider with configuration.
        
        Args:
            config: Provider configuration
        """
        self.config = config
        self._client: Optional[httpx.Client] = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._stop_event = threading.Event()
        self._retry_handler = RetryHandler(max_retries=3, base_delay=1.0)
        
        # Validate configuration
        if not config.validate():
            raise ValueError("Invalid provider configuration")
    
    @property
    def client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            # Create timeout configuration
            timeout = httpx.Timeout(
                connect=self.config.timeout,
                read=self.config.timeout,
                write=self.config.timeout,
                pool=self.config.timeout
            )
            
            self._client = httpx.Client(
                base_url=self.config.base_url,
                headers=self.config.get_headers(),
                timeout=timeout,
                verify=False,  # For local development
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
        return self._client
    
    def close(self):
        """Close the provider and cleanup resources."""
        self._stop_event.set()
        if self._client:
            self._client.close()
            self._client = None
        self._executor.shutdown(wait=True)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @abstractmethod
    def get_capabilities(self) -> List[type]:
        """
        Get the list of capability interfaces this provider supports.
        
        Returns:
            List of capability interface classes
        """
        pass
    
    def supports_capability(self, capability: type) -> bool:
        """
        Check if this provider supports a specific capability.
        
        Args:
            capability: The capability interface class
            
        Returns:
            True if the capability is supported
        """
        return capability in self.get_capabilities()
    
    def test_connectivity(self) -> str:
        """
        Test provider connectivity with a simple request.
        
        Returns:
            Response text from the test request
            
        Raises:
            APIProviderError: If the test fails
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            NetworkError: If network issues occur
        """
        # Default implementation for chat providers
        if self.supports_capability(ChatProvider):
            from ..models.chat_message import MessageRole
            test_messages = [ChatMessage(role=MessageRole.USER, content="This is a test. Please respond only with the word 'OK'.")]
            return self.create_chat_completion(test_messages)
        else:
            raise APIProviderError("Provider does not support chat capabilities for testing")
    
    def _handle_error(self, response: httpx.Response, operation: str) -> None:
        """
        Handle HTTP errors and convert to appropriate exceptions.
        
        Args:
            response: HTTP response object
            operation: Description of the operation that failed
        """
        if response.status_code == 401:
            raise AuthenticationError(f"Authentication failed for {operation}")
        elif response.status_code == 429:
            raise RateLimitError(f"Rate limit exceeded for {operation}")
        elif response.status_code >= 500:
            raise NetworkError(f"Server error during {operation}: {response.status_code}")
        elif response.status_code >= 400:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except:
                error_msg = f"HTTP {response.status_code}"
            raise APIProviderError(f"API error during {operation}: {error_msg}")
    
    def _prepare_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """
        Prepare messages for API call.
        
        Args:
            messages: List of chat messages
            
        Returns:
            List of message dictionaries
        """
        return [msg.to_dict() for msg in messages]
    
    def stop_streaming(self):
        """Stop any ongoing streaming operations."""
        self._stop_event.set()
    
    def is_stopped(self) -> bool:
        """Check if streaming has been stopped."""
        return self._stop_event.is_set()
    
    def reset_stop_event(self):
        """Reset the stop event for new operations."""
        self._stop_event.clear()


class BaseOpenAICompatibleProvider(APIProvider, ChatProvider, FunctionCallingProvider):
    """
    Base class for OpenAI-compatible providers.
    
    This provides common functionality for providers that implement
    the OpenAI API format (OpenAI, Ollama, LM Studio, etc.).
    """
    
    def get_capabilities(self) -> List[type]:
        """Get supported capabilities."""
        return [ChatProvider, FunctionCallingProvider]
    
    def create_chat_completion(self, messages: List[ChatMessage], **kwargs) -> str:
        """Create a chat completion."""
        # Reset stop event for this request
        self.reset_stop_event()
        
        def _make_request():
            payload = {
                "model": self.config.model,
                "messages": self._prepare_messages(messages),
                "max_tokens": self.config.max_tokens,
                "stream": False,
                **kwargs
            }
            
            response = self.client.post("/chat/completions", json=payload)
            self._handle_error(response, "chat completion")
            
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return ""
            
            message = choices[0].get("message", {})
            return message.get("content", "")
        
        try:
            return self._retry_handler.retry(_make_request)
        except httpx.TimeoutException:
            raise NetworkError("Chat completion request timed out")
        except httpx.RequestError as e:
            raise NetworkError(f"Network error during chat completion: {e}")
        except Exception as e:
            raise APIProviderError(f"Unexpected error during chat completion: {e}")
    
    def stream_chat_completion(self, messages: List[ChatMessage], 
                             response_handler: Callable[[str], None], **kwargs) -> None:
        """Stream a chat completion."""
        try:
            payload = {
                "model": self.config.model,
                "messages": self._prepare_messages(messages),
                "max_tokens": self.config.max_tokens,
                "stream": True,
                **kwargs
            }
            
            # Reset stop event for this request
            self.reset_stop_event()
            
            with self.client.stream("POST", "/chat/completions", json=payload) as response:
                self._handle_error(response, "streaming chat completion")
                
                accumulated_content = ""
                try:
                    for line in response.iter_lines():
                        if self.is_stopped():
                            break
                            
                        line = line.strip()
                        if not line:
                            continue
                            
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            
                            if not data_str:
                                continue
                                
                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        accumulated_content += content
                                        if not self.is_stopped():
                                            response_handler(accumulated_content)
                            except json.JSONDecodeError:
                                # Skip malformed JSON
                                continue
                            except Exception as e:
                                # Log but continue processing
                                print(f"Error processing chunk: {e}")
                                continue
                                
                except Exception as e:
                    if not self.is_stopped():
                        raise NetworkError(f"Error reading stream: {e}")
                            
        except httpx.TimeoutException:
            raise NetworkError("Request timed out")
        except httpx.RequestError as e:
            raise NetworkError(f"Network error during streaming: {e}")
        except Exception as e:
            raise APIProviderError(f"Unexpected error during streaming: {e}")
    
    def create_function_call(self, messages: List[ChatMessage], 
                           tools: List[Dict[str, Any]], **kwargs) -> List[ToolCall]:
        """Create a function call completion."""
        # Reset stop event for this request
        self.reset_stop_event()
        
        def _make_request():
            payload = {
                "model": self.config.model,
                "messages": self._prepare_messages(messages),
                "tools": tools,
                "max_tokens": self.config.max_tokens,
                "stream": False,
                **kwargs
            }
            
            response = self.client.post("/chat/completions", json=payload)
            self._handle_error(response, "function call")
            
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return []
            
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            
            return [ToolCall.from_dict(tc) for tc in tool_calls]
        
        try:
            return self._retry_handler.retry(_make_request)
        except httpx.TimeoutException:
            raise NetworkError("Function call request timed out")
        except httpx.RequestError as e:
            raise NetworkError(f"Network error during function call: {e}")
        except Exception as e:
            raise APIProviderError(f"Unexpected error during function call: {e}")
    
    def stream_function_call(self, messages: List[ChatMessage], 
                           tools: List[Dict[str, Any]],
                           response_handler: Callable[[List[ToolCall]], None], 
                           **kwargs) -> None:
        """Stream a function call completion."""
        # For now, use non-streaming for function calls since streaming
        # tool calls is more complex to implement
        tool_calls = self.create_function_call(messages, tools, **kwargs)
        response_handler(tool_calls)