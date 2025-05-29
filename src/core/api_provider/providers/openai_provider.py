"""
OpenAI API provider implementation using the official OpenAI Python library.

This provider acts as an adapter between the BinAssist provider API and
the official OpenAI Python client library.
"""

from typing import List, Dict, Any, Callable, Optional
import json
import logging
import threading

try:
    from openai import OpenAI
    import openai
except ImportError:
    # Will be caught by factory registration and provider will be skipped
    raise ImportError("openai package not available")

from ..base_provider import APIProvider
from ..capabilities import ChatProvider, FunctionCallingProvider
from ..exceptions import NetworkError, APIProviderError, AuthenticationError, RateLimitError
from ..factory import APIProviderFactory
from ..config import APIProviderConfig, ProviderType
from ..retry_handler import RetryHandler
from ...models.chat_message import ChatMessage, MessageRole
from ...models.tool_call import ToolCall


class OpenAIProvider(APIProvider, ChatProvider, FunctionCallingProvider):
    """
    OpenAI API provider using the official OpenAI Python library.
    
    This provider acts as an adapter between the BinAssist provider API and
    the official OpenAI client library. Supports chat completions, streaming,
    and function calling with proper handling for o* models.
    """
    
    def __init__(self, config: APIProviderConfig):
        """Initialize OpenAI provider."""
        super().__init__(config)
        self.logger = logging.getLogger(f"binassist.openai_provider.{config.name}")
        
        self.logger.info(f"Initializing OpenAI provider: {config.name} with model {config.model}")
        
        # Validate OpenAI-specific configuration
        if not config.api_key:
            self.logger.error("API key is required for OpenAI provider")
            raise ValueError("API key is required for OpenAI provider")
        
        # Initialize base provider functionality
        self._stop_event = threading.Event()
        self._retry_handler = RetryHandler(max_retries=3, base_delay=1.0)
        
        # Initialize OpenAI client
        try:
            # o* models need longer timeouts due to reasoning time
            timeout = config.timeout
            if self._is_reasoning_model():
                # Increase timeout for o* models - they can take 30+ seconds
                timeout = max(timeout, 180)  # At least 3 minutes for o* models
                self.logger.info(f"Using extended timeout of {timeout}s for o* model: {config.model}")
            
            self._openai_client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=timeout,
                max_retries=0  # We handle retries ourselves
            )
            self.logger.debug(f"OpenAI client initialized with base_url: {config.base_url}, timeout: {timeout}s")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise APIProviderError(f"Failed to initialize OpenAI client: {e}")
            
        self.logger.debug(f"Provider initialized successfully for model: {config.model}")
    
    def get_capabilities(self) -> List[type]:
        """Get supported capabilities."""
        return [ChatProvider, FunctionCallingProvider]
    
    def close(self):
        """Close the provider and cleanup resources."""
        self._stop_event.set()
        # OpenAI client doesn't need explicit cleanup
    
    def reset_stop_event(self):
        """Reset the stop event for a new request."""
        self._stop_event.clear()
    
    def is_stopped(self) -> bool:
        """Check if the provider has been stopped."""
        return self._stop_event.is_set()
    
    def stop(self):
        """Stop the current operation."""
        self._stop_event.set()
    
    def _is_reasoning_model(self) -> bool:
        """Check if the current model is an o* reasoning model."""
        model_name = self.config.model.lower()
        is_reasoning = (model_name.startswith("o1") or 
                       model_name.startswith("o3") or 
                       model_name.startswith("o4"))
        self.logger.debug(f"Model {model_name} detected as reasoning model: {is_reasoning}")
        return is_reasoning
    
    def _prepare_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert BinAssist ChatMessage objects to OpenAI message format."""
        self.logger.debug(f"Converting {len(messages)} messages to OpenAI format")
        
        openai_messages = []
        for msg in messages:
            openai_msg = {
                "role": msg.role.value,  # Convert enum to string value
                "content": msg.content
            }
            
            # Add tool calls if present
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                openai_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function_name,
                            "arguments": tc.function_arguments
                        }
                    }
                    for tc in msg.tool_calls
                ]
            
            # Add tool call id if this is a tool response
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id
            
            openai_messages.append(openai_msg)
        
        self.logger.debug(f"Converted messages successfully")
        return openai_messages
    
    def create_chat_completion(self, messages: List[ChatMessage], **kwargs) -> str:
        """Create a chat completion with OpenAI-specific handling."""
        self.logger.info(f"Starting chat completion for {self.config.model} with {len(messages)} messages")
        
        try:
            # Reset stop event for this request
            self.reset_stop_event()
            self.logger.debug("Stop event reset")
            
            def _make_request():
                self.logger.debug("Making chat completion request")
                
                # Convert BinAssist messages to OpenAI format
                openai_messages = self._prepare_messages(messages)
                
                # Prepare kwargs with model-specific handling
                completion_kwargs = {
                    "model": self.config.model,
                    "messages": openai_messages,
                    **kwargs
                }
                
                # Handle different token field names based on model type
                if self._is_reasoning_model():
                    completion_kwargs["max_completion_tokens"] = self.config.max_tokens
                    self.logger.debug(f"Using max_completion_tokens={self.config.max_tokens} for o* model")
                else:
                    completion_kwargs["max_tokens"] = self.config.max_tokens
                    self.logger.debug(f"Using max_tokens={self.config.max_tokens} for regular model")
                
                self.logger.debug(f"Sending request to OpenAI chat completions API")
                response = self._openai_client.chat.completions.create(**completion_kwargs)
                self.logger.debug(f"Received response from OpenAI")
                
                if not response.choices:
                    self.logger.warning("No choices in response")
                    return ""
                
                content = response.choices[0].message.content or ""
                self.logger.debug(f"Extracted content length: {len(content)}")
                return content
            
            result = self._retry_handler.retry(_make_request)
            self.logger.info(f"Chat completion successful, content length: {len(result) if result else 0}")
            return result
            
        except openai.AuthenticationError as e:
            self.logger.error(f"Authentication error: {e}")
            raise AuthenticationError(f"OpenAI authentication failed: {e}")
        except openai.RateLimitError as e:
            self.logger.error(f"Rate limit error: {e}")
            raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise APIProviderError(f"OpenAI API error: {e}")
        except Exception as e:
            self.logger.error(f"Chat completion failed: {type(e).__name__}: {e}")
            if isinstance(e, (NetworkError, APIProviderError, AuthenticationError, RateLimitError)):
                raise
            raise APIProviderError(f"Unexpected error during chat completion: {e}")
    
    def stream_chat_completion(self, messages: List[ChatMessage], 
                             response_handler: Callable[[str], None], **kwargs) -> None:
        """Stream a chat completion with OpenAI-specific handling."""
        self.logger.info(f"Starting streaming chat completion for {self.config.model} with {len(messages)} messages")
        
        try:
            # Reset stop event for this request
            self.reset_stop_event()
            self.logger.debug("Stop event reset for streaming")
            
            # Convert BinAssist messages to OpenAI format
            openai_messages = self._prepare_messages(messages)
            
            # Prepare kwargs with model-specific handling
            completion_kwargs = {
                "model": self.config.model,
                "messages": openai_messages,
                "stream": True,
                **kwargs
            }
            
            # Handle different token field names based on model type
            if self._is_reasoning_model():
                completion_kwargs["max_completion_tokens"] = self.config.max_tokens
                self.logger.debug(f"Using max_completion_tokens={self.config.max_tokens} for o* model")
            else:
                completion_kwargs["max_tokens"] = self.config.max_tokens
                self.logger.debug(f"Using max_tokens={self.config.max_tokens} for regular model")
            
            self.logger.debug("Starting OpenAI streaming request")
            accumulated_content = ""
            chunk_count = 0
            
            try:
                stream = self._openai_client.chat.completions.create(**completion_kwargs)
                self.logger.debug("Streaming response received from OpenAI")
                
                for chunk in stream:
                    if self.is_stopped():
                        self.logger.debug("Streaming stopped by user")
                        break
                    
                    chunk_count += 1
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        accumulated_content += content
                        
                        if chunk_count <= 5:  # Only log first few chunks to avoid spam
                            self.logger.debug(f"Processing chunk {chunk_count}, accumulated length: {len(accumulated_content)}")
                        
                        if not self.is_stopped():
                            response_handler(accumulated_content)
                        else:
                            self.logger.debug("Skipping response handler - stopped")
                            break
                
                self.logger.info(f"Streaming completed successfully with {chunk_count} chunks, total content length: {len(accumulated_content)}")
                
            except Exception as e:
                self.logger.error(f"Error reading stream: {e}")
                if not self.is_stopped():
                    raise NetworkError(f"Error reading stream: {e}")
            
        except openai.AuthenticationError as e:
            self.logger.error(f"Authentication error: {e}")
            raise AuthenticationError(f"OpenAI authentication failed: {e}")
        except openai.RateLimitError as e:
            self.logger.error(f"Rate limit error: {e}")
            raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise APIProviderError(f"OpenAI API error: {e}")
        except Exception as e:
            self.logger.error(f"Streaming chat completion failed: {type(e).__name__}: {e}")
            if isinstance(e, (NetworkError, APIProviderError, AuthenticationError, RateLimitError)):
                raise
            raise APIProviderError(f"Unexpected error during streaming: {e}")
    
    def create_function_call(self, messages: List[ChatMessage], 
                           tools: List[Dict[str, Any]], **kwargs) -> List[ToolCall]:
        """Create a function call completion with OpenAI-specific handling."""
        self.logger.info(f"Starting function call for {self.config.model} with {len(messages)} messages and {len(tools)} tools")
        
        try:
            # Reset stop event for this request
            self.reset_stop_event()
            
            def _make_request():
                self.logger.debug("Making function call request")
                
                # Convert BinAssist messages to OpenAI format
                openai_messages = self._prepare_messages(messages)
                
                # Prepare kwargs with model-specific handling
                completion_kwargs = {
                    "model": self.config.model,
                    "messages": openai_messages,
                    "tools": tools,
                    **kwargs
                }
                
                # Handle different token field names based on model type
                if self._is_reasoning_model():
                    completion_kwargs["max_completion_tokens"] = self.config.max_tokens
                    self.logger.debug(f"Using max_completion_tokens={self.config.max_tokens} for o* model")
                else:
                    completion_kwargs["max_tokens"] = self.config.max_tokens
                    self.logger.debug(f"Using max_tokens={self.config.max_tokens} for regular model")
                
                self.logger.info(f"Sending function call request to OpenAI for o* model (may take up to 3 minutes)")
                self.logger.debug(f"Function call request params: {completion_kwargs}")
                
                import time
                start_time = time.time()
                response = self._openai_client.chat.completions.create(**completion_kwargs)
                end_time = time.time()
                
                self.logger.info(f"Received function call response from OpenAI (took {end_time - start_time:.1f}s)")
                
                if not response.choices:
                    self.logger.warning("No choices in function call response")
                    return []
                
                message = response.choices[0].message
                if not message.tool_calls:
                    self.logger.debug("No tool calls in response")
                    return []
                
                # Convert OpenAI tool calls to BinAssist ToolCall objects
                tool_calls = []
                for tc in message.tool_calls:
                    # Parse arguments from JSON string to dict
                    try:
                        arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse tool call arguments: {tc.function.arguments}")
                        arguments = {}
                    
                    tool_call = ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments
                    )
                    tool_calls.append(tool_call)
                
                self.logger.debug(f"Converted {len(tool_calls)} tool calls")
                return tool_calls
            
            result = self._retry_handler.retry(_make_request)
            self.logger.info(f"Function call successful, returned {len(result)} tool calls")
            return result
            
        except openai.AuthenticationError as e:
            self.logger.error(f"Authentication error: {e}")
            raise AuthenticationError(f"OpenAI authentication failed: {e}")
        except openai.RateLimitError as e:
            self.logger.error(f"Rate limit error: {e}")
            raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise APIProviderError(f"OpenAI API error: {e}")
        except Exception as e:
            self.logger.error(f"Function call failed: {type(e).__name__}: {e}")
            if isinstance(e, (NetworkError, APIProviderError, AuthenticationError, RateLimitError)):
                raise
            raise APIProviderError(f"Unexpected error during function call: {e}")
    
    def stream_function_call(self, messages: List[ChatMessage], 
                           tools: List[Dict[str, Any]],
                           response_handler: Callable[[List[ToolCall]], None], 
                           completion_handler: Callable[[], None] = None,
                           **kwargs) -> None:
        """Stream a function call completion with OpenAI-specific handling."""
        # Use non-streaming for function calls since streaming tool calls is complex
        tool_calls = self.create_function_call(messages, tools, **kwargs)
        response_handler(tool_calls)
        
        # Call completion handler if provided  
        from binaryninja import log
        if completion_handler:
            log.log_info("[BinAssist] OpenAI provider calling completion handler")
            completion_handler()
            log.log_info("[BinAssist] OpenAI provider completion handler finished")
        else:
            log.log_warn("[BinAssist] OpenAI provider: No completion handler provided")
    
    def test_connectivity(self) -> str:
        """Test OpenAI provider connectivity with detailed logging."""
        self.logger.info(f"Testing connectivity for OpenAI provider: {self.config.name}")
        
        try:
            # Create test message
            test_messages = [ChatMessage(role=MessageRole.USER, content="This is a test. Please respond only with the word 'OK'.")]
            
            # Log the test attempt
            self.logger.info(f"Sending test request to model: {self.config.model}")
            
            # Use the standard chat completion method
            response_text = self.create_chat_completion(test_messages)
            
            self.logger.info(f"Test successful, received response: '{response_text}'")
            return response_text
            
        except Exception as e:
            self.logger.error(f"Connectivity test failed: {e}")
            raise


class OpenAIProviderFactory(APIProviderFactory):
    """Factory for creating OpenAI providers."""
    
    def create_provider(self, config: APIProviderConfig) -> OpenAIProvider:
        """Create an OpenAI provider instance."""
        return OpenAIProvider(config)
    
    def supports(self, provider_type: ProviderType) -> bool:
        """Check if this factory supports the provider type."""
        return provider_type == ProviderType.OPENAI
    
    def get_supported_types(self) -> List[ProviderType]:
        """Get supported provider types."""
        return [ProviderType.OPENAI]