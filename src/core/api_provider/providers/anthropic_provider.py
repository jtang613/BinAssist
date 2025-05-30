"""
Anthropic API provider implementation.
"""

from typing import List, Dict, Any, Callable
import json
import threading

from binaryninja import log

try:
    import anthropic
except ImportError:
    # Will be caught by factory registration and provider will be skipped
    raise ImportError("anthropic package not available")

from ..base_provider import APIProvider
from ..exceptions import NetworkError, APIProviderError, AuthenticationError, RateLimitError
from ..capabilities import ChatProvider, FunctionCallingProvider
from ..factory import APIProviderFactory
from ..config import APIProviderConfig, ProviderType
from ..retry_handler import RetryHandler
from ...models.chat_message import ChatMessage, MessageRole
from ...models.tool_call import ToolCall


class AnthropicProvider(APIProvider, ChatProvider, FunctionCallingProvider):
    """
    Anthropic API provider.
    
    Supports chat completions and function calling using the Anthropic Claude API.
    """
    
    def __init__(self, config: APIProviderConfig):
        """Initialize Anthropic provider."""
        super().__init__(config)
        
        # Initialize base provider functionality
        self._stop_event = threading.Event()
        self._retry_handler = RetryHandler(max_retries=3, base_delay=1.0)
        
        # Validate Anthropic-specific configuration
        if not config.api_key:
            raise ValueError("API key is required for Anthropic provider")
        
        # Initialize Anthropic client
        try:
            # Handle base_url - Anthropic client expects None for default
            base_url = None if config.base_url == 'https://api.anthropic.com' else config.base_url
            
            self._anthropic_client = anthropic.Anthropic(
                api_key=config.api_key,
                base_url=base_url,
                timeout=config.timeout,
                max_retries=0  # We handle retries ourselves
            )
            log.log_debug(f"[BinAssist] Anthropic client initialized with base_url: {base_url}")
        except Exception as e:
            log.log_error(f"[BinAssist] Failed to initialize Anthropic client: {e}")
            raise APIProviderError(f"Failed to initialize Anthropic client: {e}")
            
        log.log_debug(f"[BinAssist] Provider initialized successfully for model: {config.model}")
    
    def get_capabilities(self) -> List[type]:
        """Get supported capabilities."""
        return [ChatProvider, FunctionCallingProvider]
    
    def close(self):
        """Close the provider and cleanup resources."""
        self._stop_event.set()
        # Anthropic client doesn't need explicit cleanup
    
    def reset_stop_event(self):
        """Reset the stop event for a new request."""
        self._stop_event.clear()
    
    def is_stopped(self) -> bool:
        """Check if the provider has been stopped."""
        return self._stop_event.is_set()
    
    def stop(self):
        """Stop the current operation."""
        self._stop_event.set()
    
    def create_chat_completion(self, messages: List[ChatMessage], **kwargs) -> str:
        """Create a chat completion."""
        try:
            log.log_info(f"[BinAssist] Anthropic API call starting")
            log.log_info(f"[BinAssist] Provider config - Name: {self.config.name}")
            log.log_info(f"[BinAssist] Provider config - Base URL: {self.config.base_url}")
            log.log_info(f"[BinAssist] Provider config - Model: {self.config.model}")
            log.log_info(f"[BinAssist] Provider config - API Key: {'***' + self.config.api_key[-4:] if self.config.api_key and len(self.config.api_key) > 4 else 'None'}")
            
            # Convert BinAssist messages to Anthropic format
            anthropic_messages = self._prepare_messages(messages)
            system_message = self._extract_system_message(messages)
            
            payload = {
                "model": self.config.model,
                "messages": anthropic_messages,
                "max_tokens": self.config.max_tokens,
                "stream": False,
                **kwargs
            }
            
            if system_message:
                payload["system"] = system_message
            
            log.log_info(f"[BinAssist] Anthropic request payload: {json.dumps(payload, indent=2)}")
            log.log_info(f"[BinAssist] Making request using Anthropic SDK")
            
            # Make the request using Anthropic SDK
            response = self._anthropic_client.messages.create(**payload)
            
            log.log_info(f"[BinAssist] Response received: {response}")
            
            # Extract text content from response
            text_content = ""
            for content_block in response.content:
                if content_block.type == "text":
                    text_content += content_block.text
            
            log.log_info(f"[BinAssist] Extracted text content: '{text_content}'")
            return text_content
            
        except anthropic.AuthenticationError as e:
            log.log_error(f"[BinAssist] Authentication error: {e}")
            raise AuthenticationError(f"Anthropic authentication failed: {e}")
        except anthropic.RateLimitError as e:
            log.log_error(f"[BinAssist] Rate limit error: {e}")
            raise RateLimitError(f"Anthropic rate limit exceeded: {e}")
        except anthropic.APIError as e:
            log.log_error(f"[BinAssist] Anthropic API error: {e}")
            raise APIProviderError(f"Anthropic API error: {e}")
        except Exception as e:
            log.log_error(f"[BinAssist] Chat completion failed: {type(e).__name__}: {e}")
            if isinstance(e, (NetworkError, APIProviderError, AuthenticationError, RateLimitError)):
                raise
            raise APIProviderError(f"Unexpected error during chat completion: {e}")
    
    def stream_chat_completion(self, messages: List[ChatMessage], 
                             response_handler: Callable[[str], None], **kwargs) -> None:
        """Stream a chat completion."""
        try:
            # Convert BinAssist messages to Anthropic format
            anthropic_messages = self._prepare_messages(messages)
            system_message = self._extract_system_message(messages)
            
            payload = {
                "model": self.config.model,
                "messages": anthropic_messages,
                "max_tokens": self.config.max_tokens,
                **kwargs
            }
            
            if system_message:
                payload["system"] = system_message
            
            # Reset stop event for this request
            self._stop_event.clear()
            
            log.log_info(f"[BinAssist] Anthropic streaming request payload: {json.dumps(payload, indent=2)}")
            log.log_info(f"[BinAssist] Making streaming request using Anthropic SDK")
            
            # Use Anthropic SDK for streaming
            with self._anthropic_client.messages.stream(**payload) as stream:
                accumulated_content = ""
                try:
                    for text in stream.text_stream:
                        if self.is_stopped():
                            break
                        
                        if text:
                            accumulated_content += text
                            if not self.is_stopped():
                                response_handler(accumulated_content)
                                
                except Exception as e:
                    if not self.is_stopped():
                        log.log_error(f"[BinAssist] Error reading stream: {e}")
                        raise NetworkError(f"Error reading stream: {e}")
                        
            log.log_info(f"[BinAssist] Anthropic streaming completed successfully")
            
        except anthropic.AuthenticationError as e:
            log.log_error(f"[BinAssist] Authentication error during streaming: {e}")
            raise AuthenticationError(f"Anthropic authentication failed: {e}")
        except anthropic.RateLimitError as e:
            log.log_error(f"[BinAssist] Rate limit error during streaming: {e}")
            raise RateLimitError(f"Anthropic rate limit exceeded: {e}")
        except anthropic.APIError as e:
            log.log_error(f"[BinAssist] Anthropic API error during streaming: {e}")
            raise APIProviderError(f"Anthropic API error: {e}")
        except Exception as e:
            log.log_error(f"[BinAssist] Streaming failed: {type(e).__name__}: {e}")
            if isinstance(e, (NetworkError, APIProviderError, AuthenticationError, RateLimitError)):
                raise
            raise APIProviderError(f"Unexpected error during streaming: {e}")
    
    def create_function_call(self, messages: List[ChatMessage], 
                           tools: List[Dict[str, Any]], **kwargs) -> List[ToolCall]:
        """Create a function call completion."""
        try:
            # Convert messages and tools to Anthropic format
            anthropic_messages = self._prepare_messages(messages)
            system_message = self._extract_system_message(messages)
            anthropic_tools = self._convert_tools_to_anthropic(tools)
            
            payload = {
                "model": self.config.model,
                "messages": anthropic_messages,
                "tools": anthropic_tools,
                "max_tokens": self.config.max_tokens,
                **kwargs
            }
            
            if system_message:
                payload["system"] = system_message
            
            log.log_info(f"[BinAssist] Anthropic function call payload: {json.dumps(payload, indent=2)}")
            log.log_info(f"[BinAssist] Making function call using Anthropic SDK")
            
            # Debug: Log the message sequence for tool call matching
            log.log_info("[BinAssist] === ANTHROPIC MESSAGE SEQUENCE DEBUG ===")
            for i, msg in enumerate(anthropic_messages):
                log.log_info(f"[BinAssist] Message[{i}]: role={msg['role']}")
                if isinstance(msg['content'], list):
                    for j, content_block in enumerate(msg['content']):
                        if content_block.get('type') == 'tool_use':
                            log.log_info(f"[BinAssist]   tool_use[{j}]: id={content_block['id']}, name={content_block['name']}")
                        elif content_block.get('type') == 'tool_result':
                            log.log_info(f"[BinAssist]   tool_result[{j}]: tool_use_id={content_block['tool_use_id']}")
            log.log_info("[BinAssist] === END MESSAGE SEQUENCE DEBUG ===")
            
            # Use Anthropic SDK for function calls
            response = self._anthropic_client.messages.create(**payload)
            
            log.log_info(f"[BinAssist] Function call response received: {response}")
            
            # Extract tool calls from response content
            tool_calls = []
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_call = ToolCall(
                        id=content_block.id,
                        name=content_block.name,
                        arguments=content_block.input,
                        call_id=content_block.id  # Set call_id to match id for Anthropic compatibility
                    )
                    tool_calls.append(tool_call)
                    log.log_info(f"[BinAssist] Created ToolCall: id={content_block.id}, call_id={content_block.id}, name={content_block.name}, args={content_block.input}")
            
            log.log_info(f"[BinAssist] Extracted {len(tool_calls)} tool calls total")
            log.log_debug(f"[BinAssist] All tool calls: {[f'{tc.name}({tc.arguments})' for tc in tool_calls]}")
            return tool_calls
            
        except anthropic.AuthenticationError as e:
            log.log_error(f"[BinAssist] Authentication error during function call: {e}")
            raise AuthenticationError(f"Anthropic authentication failed: {e}")
        except anthropic.RateLimitError as e:
            log.log_error(f"[BinAssist] Rate limit error during function call: {e}")
            raise RateLimitError(f"Anthropic rate limit exceeded: {e}")
        except anthropic.APIError as e:
            log.log_error(f"[BinAssist] Anthropic API error during function call: {e}")
            raise APIProviderError(f"Anthropic API error: {e}")
        except Exception as e:
            log.log_error(f"[BinAssist] Function call failed: {type(e).__name__}: {e}")
            if isinstance(e, (NetworkError, APIProviderError, AuthenticationError, RateLimitError)):
                raise
            raise APIProviderError(f"Unexpected error during function call: {e}")
    
    def create_function_call_with_content(self, messages: List[ChatMessage], 
                                        tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Create a function call completion that returns both tool calls and text content."""
        log.log_info(f"[BinAssist] Starting Anthropic function call with content for {self.config.model} with {len(messages)} messages and {len(tools)} tools")
        
        try:
            # Convert messages and tools to Anthropic format
            anthropic_messages = self._prepare_messages(messages)
            system_message = self._extract_system_message(messages)
            anthropic_tools = self._convert_tools_to_anthropic(tools)
            
            payload = {
                "model": self.config.model,
                "messages": anthropic_messages,
                "tools": anthropic_tools,
                "max_tokens": self.config.max_tokens,
                **kwargs
            }
            
            if system_message:
                payload["system"] = system_message
            
            log.log_info(f"[BinAssist] Anthropic function call with content payload: {json.dumps(payload, indent=2)}")
            
            # Debug: Log the message sequence for tool call matching
            log.log_info("[BinAssist] === ANTHROPIC MESSAGE SEQUENCE DEBUG ===")
            for i, msg in enumerate(anthropic_messages):
                log.log_info(f"[BinAssist] Message[{i}]: role={msg['role']}")
                if isinstance(msg['content'], list):
                    for j, content_block in enumerate(msg['content']):
                        if content_block.get('type') == 'tool_use':
                            log.log_info(f"[BinAssist]   tool_use[{j}]: id={content_block['id']}, name={content_block['name']}")
                        elif content_block.get('type') == 'tool_result':
                            log.log_info(f"[BinAssist]   tool_result[{j}]: tool_use_id={content_block['tool_use_id']}")
            log.log_info("[BinAssist] === END MESSAGE SEQUENCE DEBUG ===")
            
            # Use Anthropic SDK for function calls
            response = self._anthropic_client.messages.create(**payload)
            
            log.log_info(f"[BinAssist] Function call with content response received: {response}")
            
            # Log raw response structure for debugging
            log.log_info("[BinAssist] === RAW ANTHROPIC RESPONSE DEBUG ===")
            log.log_info(f"[BinAssist] Response type: {type(response)}")
            log.log_info(f"[BinAssist] Response content blocks: {len(response.content)}")
            for i, content_block in enumerate(response.content):
                log.log_info(f"[BinAssist] Content[{i}]: type={content_block.type}")
                if content_block.type == "text":
                    log.log_info(f"[BinAssist] Content[{i}]: text={repr(content_block.text[:200])}")
                elif content_block.type == "tool_use":
                    log.log_info(f"[BinAssist] Content[{i}]: tool_use id={content_block.id}, name={content_block.name}")
            log.log_info("[BinAssist] === END RAW ANTHROPIC RESPONSE DEBUG ===")
            
            result = {"tool_calls": [], "content": ""}
            
            # Extract both tool calls and text content from response
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_call = ToolCall(
                        id=content_block.id,
                        name=content_block.name,
                        arguments=content_block.input,
                        call_id=content_block.id  # Set call_id to match id for Anthropic compatibility
                    )
                    result["tool_calls"].append(tool_call)
                    log.log_info(f"[BinAssist] 🔧 Tool call extracted: id={content_block.id}, name={content_block.name}")
                elif content_block.type == "text":
                    result["content"] += content_block.text
                    log.log_info(f"[BinAssist] 📝 Text content extracted: {len(content_block.text)} characters")
            
            log.log_info(f"[BinAssist] 🎯 Anthropic response analysis: {len(result['tool_calls'])} tool calls, {len(result['content'])} characters of text")
            return result
            
        except anthropic.AuthenticationError as e:
            log.log_error(f"[BinAssist] Authentication error during function call with content: {e}")
            raise AuthenticationError(f"Anthropic authentication failed: {e}")
        except anthropic.RateLimitError as e:
            log.log_error(f"[BinAssist] Rate limit error during function call with content: {e}")
            raise RateLimitError(f"Anthropic rate limit exceeded: {e}")
        except anthropic.APIError as e:
            log.log_error(f"[BinAssist] Anthropic API error during function call with content: {e}")
            raise APIProviderError(f"Anthropic API error: {e}")
        except Exception as e:
            log.log_error(f"[BinAssist] Function call with content failed: {type(e).__name__}: {e}")
            if isinstance(e, (NetworkError, APIProviderError, AuthenticationError, RateLimitError)):
                raise
            raise APIProviderError(f"Unexpected error during function call with content: {e}")
    
    def stream_function_call(self, messages: List[ChatMessage], 
                           tools: List[Dict[str, Any]],
                           response_handler: Callable[[List[ToolCall]], None], 
                           completion_handler: Callable[[], None] = None,
                           text_handler: Callable[[str], None] = None,
                           **kwargs) -> None:
        """Stream a function call completion."""
        log.log_info(f"[BinAssist] AnthropicProvider: Starting stream_function_call with text_handler={'available' if text_handler else 'None'}")
        
        # Use function call API to get response (may contain tools OR text)
        response_data = self.create_function_call_with_content(messages, tools, **kwargs)
        tool_calls = response_data.get('tool_calls', [])
        text_content = response_data.get('content', '')
        
        log.log_info(f"[BinAssist] AnthropicProvider: Response contains {len(tool_calls)} tool calls and {len(text_content)} characters of text")
        
        try:
            # Handle tool calls if present
            if tool_calls:
                log.log_info(f"[BinAssist] AnthropicProvider: Handling {len(tool_calls)} tool calls")
                log.log_info(f"[BinAssist] AnthropicProvider: Text content with tool calls (explanatory): {repr(text_content[:100]) if text_content else 'None'}")
                
                # Display explanatory text alongside tool calls if present
                if text_content and text_handler:
                    log.log_info(f"[BinAssist] AnthropicProvider: Displaying explanatory text ({len(text_content)} chars) alongside tool calls")
                    formatted_text = f"💭 {text_content.strip()}\n"
                    text_handler(formatted_text)
                
                response_handler(tool_calls)
                log.log_info(f"[BinAssist] AnthropicProvider: Tool calls handled successfully")
            
            # Handle text content when there are NO tool calls (final response)
            elif text_content and text_handler:
                log.log_info(f"[BinAssist] AnthropicProvider: Handling final text response ({len(text_content)} characters)")
                text_handler(text_content)
                log.log_info(f"[BinAssist] AnthropicProvider: Final text response handled successfully")
            elif text_content:
                log.log_warn(f"[BinAssist] AnthropicProvider: Text content available ({len(text_content)} chars) but no text_handler provided")
            
            # Call completion handler if provided
            if completion_handler:
                log.log_debug(f"[BinAssist] AnthropicProvider: Calling completion handler")
                completion_handler()
                
        except Exception as e:
            log.log_error(f"[BinAssist] AnthropicProvider: Error in stream_function_call: {e}")
            log.log_error(f"[BinAssist] AnthropicProvider: Exception type: {type(e)}")
            raise
    
    def _prepare_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert messages to Anthropic format."""
        anthropic_messages = []
        
        log.log_info(f"[BinAssist] Anthropic _prepare_messages: Processing {len(messages)} messages")
        
        for i, message in enumerate(messages):
            log.log_info(f"[BinAssist] Anthropic message[{i}]: role={message.role}, content_len={len(message.content) if message.content else 0}")
            
            # Skip system messages - they're handled separately
            if message.role == MessageRole.SYSTEM:
                continue
            
            anthropic_msg = {
                "role": "user" if message.role == MessageRole.USER else "assistant",
                "content": message.content
            }
            
            # Handle tool calls for assistant messages
            if message.tool_calls:
                log.log_info(f"[BinAssist] Anthropic message[{i}]: Found {len(message.tool_calls)} tool calls")
                content = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                
                for j, tool_call in enumerate(message.tool_calls):
                    tool_use_block = {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": tool_call.arguments
                    }
                    content.append(tool_use_block)
                    log.log_info(f"[BinAssist] Anthropic tool_use[{j}]: id={tool_call.id}, name={tool_call.name}")
                
                anthropic_msg["content"] = content
            
            # Handle tool responses
            if message.role == MessageRole.TOOL:
                log.log_info(f"[BinAssist] Anthropic message[{i}]: Tool result with tool_call_id={message.tool_call_id}")
                anthropic_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": message.content
                        }
                    ]
                }
            
            anthropic_messages.append(anthropic_msg)
            log.log_info(f"[BinAssist] Anthropic message[{i}]: Converted to {anthropic_msg['role']}")
        
        log.log_info(f"[BinAssist] Anthropic _prepare_messages: Generated {len(anthropic_messages)} anthropic messages")
        return anthropic_messages
    
    def _extract_system_message(self, messages: List[ChatMessage]) -> str:
        """Extract system message content."""
        for message in messages:
            if message.role == MessageRole.SYSTEM:
                return message.content
        return ""
    
    def _convert_tools_to_anthropic(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic tool format."""
        anthropic_tools = []
        
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                anthropic_tool = {
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "input_schema": function.get("parameters", {})
                }
                anthropic_tools.append(anthropic_tool)
        
        return anthropic_tools
    
    def test_connectivity(self) -> str:
        """Test Anthropic provider connectivity with detailed logging."""
        log.log_info(f"[BinAssist] Testing connectivity for Anthropic provider: {self.config.name}")
        
        try:
            # Create test message
            test_messages = [ChatMessage(role=MessageRole.USER, content="This is a test. Please respond only with the word 'OK'.")]
            
            # Log the test attempt
            log.log_info(f"[BinAssist] Sending test request to model: {self.config.model}")
            
            # Use the standard chat completion method
            response_text = self.create_chat_completion(test_messages)
            
            log.log_info(f"[BinAssist] Test successful, received response: '{response_text}'")
            return response_text
            
        except Exception as e:
            log.log_error(f"[BinAssist] Connectivity test failed: {e}")
            raise


class AnthropicProviderFactory(APIProviderFactory):
    """Factory for creating Anthropic providers."""
    
    def create_provider(self, config: APIProviderConfig) -> AnthropicProvider:
        """Create an Anthropic provider instance."""
        return AnthropicProvider(config)
    
    def supports(self, provider_type: ProviderType) -> bool:
        """Check if this factory supports the provider type."""
        return provider_type == ProviderType.ANTHROPIC
    
    def get_supported_types(self) -> List[ProviderType]:
        """Get supported provider types."""
        return [ProviderType.ANTHROPIC]