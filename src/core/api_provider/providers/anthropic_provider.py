"""
Anthropic API provider implementation.
"""

from typing import List, Dict, Any, Callable
import json

from ..base_provider import APIProvider
from ..exceptions import NetworkError, APIProviderError
from ..capabilities import ChatProvider, FunctionCallingProvider
from ..factory import APIProviderFactory
from ..config import APIProviderConfig, ProviderType
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
        
        # Validate Anthropic-specific configuration
        if not config.api_key:
            raise ValueError("API key is required for Anthropic provider")
    
    def get_capabilities(self) -> List[type]:
        """Get supported capabilities."""
        return [ChatProvider, FunctionCallingProvider]
    
    def create_chat_completion(self, messages: List[ChatMessage], **kwargs) -> str:
        """Create a chat completion."""
        try:
            # Convert messages to Anthropic format
            anthropic_messages = self._convert_messages_to_anthropic(messages)
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
            
            response = self.client.post("/messages", json=payload)
            self._handle_error(response, "chat completion")
            
            data = response.json()
            content = data.get("content", [])
            
            # Extract text content
            text_content = ""
            for item in content:
                if item.get("type") == "text":
                    text_content += item.get("text", "")
            
            return text_content
            
        except Exception as e:
            if isinstance(e, (NetworkError, APIProviderError)):
                raise
            raise APIProviderError(f"Unexpected error during chat completion: {e}")
    
    def stream_chat_completion(self, messages: List[ChatMessage], 
                             response_handler: Callable[[str], None], **kwargs) -> None:
        """Stream a chat completion."""
        try:
            # Convert messages to Anthropic format
            anthropic_messages = self._convert_messages_to_anthropic(messages)
            system_message = self._extract_system_message(messages)
            
            payload = {
                "model": self.config.model,
                "messages": anthropic_messages,
                "max_tokens": self.config.max_tokens,
                "stream": True,
                **kwargs
            }
            
            if system_message:
                payload["system"] = system_message
            
            # Reset stop event for this request
            self._stop_event.clear()
            
            with self.client.stream("POST", "/messages", json=payload) as response:
                self._handle_error(response, "streaming chat completion")
                
                accumulated_content = ""
                try:
                    for line in response.iter_lines():
                        if self.is_stopped():
                            break
                            
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Anthropic uses SSE format
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            
                            if not data_str:
                                continue
                                
                            try:
                                data = json.loads(data_str)
                                event_type = data.get("type")
                                
                                if event_type == "content_block_delta":
                                    delta = data.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        text = delta.get("text", "")
                                        if text:
                                            accumulated_content += text
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
                            
        except Exception as e:
            if isinstance(e, (NetworkError, APIProviderError)):
                raise
            raise APIProviderError(f"Unexpected error during streaming: {e}")
    
    def create_function_call(self, messages: List[ChatMessage], 
                           tools: List[Dict[str, Any]], **kwargs) -> List[ToolCall]:
        """Create a function call completion."""
        try:
            # Convert messages and tools to Anthropic format
            anthropic_messages = self._convert_messages_to_anthropic(messages)
            system_message = self._extract_system_message(messages)
            anthropic_tools = self._convert_tools_to_anthropic(tools)
            
            payload = {
                "model": self.config.model,
                "messages": anthropic_messages,
                "tools": anthropic_tools,
                "max_tokens": self.config.max_tokens,
                "stream": False,
                **kwargs
            }
            
            if system_message:
                payload["system"] = system_message
            
            response = self.client.post("/messages", json=payload)
            self._handle_error(response, "function call")
            
            data = response.json()
            content = data.get("content", [])
            
            # Extract tool calls from content
            tool_calls = []
            for item in content:
                if item.get("type") == "tool_use":
                    tool_call = ToolCall(
                        id=item.get("id", ""),
                        name=item.get("name", ""),
                        arguments=item.get("input", {})
                    )
                    tool_calls.append(tool_call)
            
            return tool_calls
            
        except Exception as e:
            if isinstance(e, (NetworkError, APIProviderError)):
                raise
            raise APIProviderError(f"Unexpected error during function call: {e}")
    
    def stream_function_call(self, messages: List[ChatMessage], 
                           tools: List[Dict[str, Any]],
                           response_handler: Callable[[List[ToolCall]], None], 
                           **kwargs) -> None:
        """Stream a function call completion."""
        # For now, use non-streaming for function calls since streaming
        # tool calls is more complex to implement for Anthropic
        tool_calls = self.create_function_call(messages, tools, **kwargs)
        response_handler(tool_calls)
    
    def _convert_messages_to_anthropic(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert messages to Anthropic format."""
        anthropic_messages = []
        
        for message in messages:
            # Skip system messages - they're handled separately
            if message.role == MessageRole.SYSTEM:
                continue
            
            anthropic_msg = {
                "role": "user" if message.role == MessageRole.USER else "assistant",
                "content": message.content
            }
            
            # Handle tool calls for assistant messages
            if message.tool_calls:
                content = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                
                for tool_call in message.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": tool_call.arguments
                    })
                
                anthropic_msg["content"] = content
            
            # Handle tool responses
            if message.role == MessageRole.TOOL:
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