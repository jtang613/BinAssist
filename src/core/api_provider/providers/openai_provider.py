"""
OpenAI API provider implementation.
"""

from typing import List, Dict, Any, Callable
import json
import logging

from ..base_provider import BaseOpenAICompatibleProvider
from ..exceptions import NetworkError, APIProviderError
from ..factory import APIProviderFactory
from ..config import APIProviderConfig, ProviderType
from ...models.chat_message import ChatMessage
from ...models.tool_call import ToolCall


class OpenAIProvider(BaseOpenAICompatibleProvider):
    """
    OpenAI API provider.
    
    Supports chat completions and function calling using the OpenAI API.
    Handles special cases for o* models (o1-mini, o4-mini, etc.).
    """
    
    def __init__(self, config: APIProviderConfig):
        """Initialize OpenAI provider."""
        super().__init__(config)
        self.logger = logging.getLogger(f"binassist.openai_provider.{config.name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info(f"Initializing OpenAI provider: {config.name} with model {config.model}")
        
        # Validate OpenAI-specific configuration
        if not config.api_key:
            self.logger.error("API key is required for OpenAI provider")
            raise ValueError("API key is required for OpenAI provider")
            
        self.logger.debug(f"Provider initialized successfully for model: {config.model}")
    
    def _is_reasoning_model(self) -> bool:
        """Check if the current model is an o* reasoning model."""
        model_name = self.config.model.lower()
        is_reasoning = (model_name.startswith("o1-") or 
                       model_name.startswith("o3-") or 
                       model_name.startswith("o4-") or
                       model_name.startswith("o1") or
                       model_name.startswith("o3") or
                       model_name.startswith("o4"))
        self.logger.debug(f"Model {model_name} detected as reasoning model: {is_reasoning}")
        return is_reasoning
    
    def _prepare_payload(self, messages: List[ChatMessage], stream: bool = False, 
                        tools: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Prepare the API payload with model-specific handling."""
        self.logger.debug(f"Preparing payload for {len(messages)} messages, stream={stream}, tools={len(tools) if tools else 0}")
        
        try:
            payload = {
                "model": self.config.model,
                "messages": self._prepare_messages(messages),
                "stream": stream,
                **kwargs
            }
            
            # Handle different token field names based on model type
            if self._is_reasoning_model():
                payload["max_completion_tokens"] = self.config.max_tokens
                self.logger.debug(f"Using max_completion_tokens={self.config.max_tokens} for o* model")
            else:
                payload["max_tokens"] = self.config.max_tokens
                self.logger.debug(f"Using max_tokens={self.config.max_tokens} for regular model")
                
            # Add tools if provided
            if tools:
                payload["tools"] = tools
                self.logger.debug(f"Added {len(tools)} tools to payload")
            
            self.logger.debug(f"Payload prepared successfully with keys: {list(payload.keys())}")
            return payload
            
        except Exception as e:
            self.logger.error(f"Error preparing payload: {e}")
            raise
    
    def create_chat_completion(self, messages: List[ChatMessage], **kwargs) -> str:
        """Create a chat completion with OpenAI-specific handling."""
        self.logger.info(f"Starting chat completion for {self.config.model} with {len(messages)} messages")
        
        try:
            # Reset stop event for this request
            self.reset_stop_event()
            self.logger.debug("Stop event reset")
            
            def _make_request():
                self.logger.debug("Making chat completion request")
                payload = self._prepare_payload(messages, stream=False, **kwargs)
                
                self.logger.debug(f"Sending request to {self.client.base_url}/chat/completions")
                response = self.client.post("/chat/completions", json=payload)
                self.logger.debug(f"Received response with status: {response.status_code}")
                
                self._handle_error(response, "chat completion")
                
                data = response.json()
                self.logger.debug(f"Response data keys: {list(data.keys())}")
                
                choices = data.get("choices", [])
                if not choices:
                    self.logger.warning("No choices in response")
                    return ""
                
                message = choices[0].get("message", {})
                content = message.get("content", "")
                self.logger.debug(f"Extracted content length: {len(content) if content else 0}")
                return content
            
            result = self._retry_handler.retry(_make_request)
            self.logger.info(f"Chat completion successful, content length: {len(result) if result else 0}")
            return result
            
        except Exception as e:
            self.logger.error(f"Chat completion failed: {type(e).__name__}: {e}")
            if isinstance(e, (NetworkError, APIProviderError)):
                raise
            raise APIProviderError(f"Unexpected error during chat completion: {e}")
    
    def stream_chat_completion(self, messages: List[ChatMessage], 
                             response_handler: Callable[[str], None], **kwargs) -> None:
        """Stream a chat completion with OpenAI-specific handling."""
        self.logger.info(f"Starting streaming chat completion for {self.config.model} with {len(messages)} messages")
        
        try:
            self.logger.debug("Preparing payload for streaming")
            payload = self._prepare_payload(messages, stream=True, **kwargs)
            
            # Reset stop event for this request
            self.reset_stop_event()
            self.logger.debug("Stop event reset for streaming")
            
            self.logger.debug("Opening streaming connection")
            with self.client.stream("POST", "/chat/completions", json=payload) as response:
                self.logger.debug(f"Streaming response received with status: {response.status_code}")
                self._handle_error(response, "streaming chat completion")
                
                accumulated_content = ""
                chunk_count = 0
                
                try:
                    self.logger.debug("Starting to process streaming response")
                    
                    for line in response.iter_lines():
                        if self.is_stopped():
                            self.logger.debug("Streaming stopped by user")
                            break
                            
                        line = line.strip()
                        if not line:
                            continue
                            
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                self.logger.debug("Streaming completed with [DONE] marker")
                                break
                            
                            if not data_str:
                                continue
                                
                            try:
                                chunk_count += 1
                                if chunk_count <= 5:  # Only log first few chunks to avoid spam
                                    self.logger.debug(f"Processing chunk {chunk_count}: {data_str[:100]}...")
                                    
                                import json
                                data = json.loads(data_str)
                                choices = data.get("choices", [])
                                
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    
                                    if content:
                                        accumulated_content += content
                                        if chunk_count <= 5:
                                            self.logger.debug(f"Accumulated content length: {len(accumulated_content)}")
                                            
                                        if not self.is_stopped():
                                            self.logger.debug("Calling response handler")
                                            response_handler(accumulated_content)
                                            self.logger.debug("Response handler completed")
                                        else:
                                            self.logger.debug("Skipping response handler - stopped")
                                            
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"Skipping malformed JSON chunk: {e}")
                                continue
                            except Exception as e:
                                self.logger.error(f"Error processing chunk {chunk_count}: {e}")
                                continue
                                
                    self.logger.info(f"Streaming completed successfully with {chunk_count} chunks, total content length: {len(accumulated_content)}")
                    
                except Exception as e:
                    self.logger.error(f"Error reading stream: {e}")
                    if not self.is_stopped():
                        raise NetworkError(f"Error reading stream: {e}")
                            
        except Exception as e:
            self.logger.error(f"Streaming chat completion failed: {type(e).__name__}: {e}")
            if isinstance(e, (NetworkError, APIProviderError)):
                raise
            raise APIProviderError(f"Unexpected error during streaming: {e}")
    
    def create_function_call(self, messages: List[ChatMessage], 
                           tools: List[Dict[str, Any]], **kwargs) -> List[ToolCall]:
        """Create a function call completion with OpenAI-specific handling."""
        # Reset stop event for this request
        self.reset_stop_event()
        
        def _make_request():
            payload = self._prepare_payload(messages, stream=False, tools=tools, **kwargs)
            
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
        except Exception as e:
            if isinstance(e, (NetworkError, APIProviderError)):
                raise
            raise APIProviderError(f"Unexpected error during function call: {e}")
    
    def stream_function_call(self, messages: List[ChatMessage], 
                           tools: List[Dict[str, Any]],
                           response_handler: Callable[[List[ToolCall]], None], 
                           **kwargs) -> None:
        """Stream a function call completion with OpenAI-specific handling."""
        # Use non-streaming for function calls since streaming tool calls is complex
        tool_calls = self.create_function_call(messages, tools, **kwargs)
        response_handler(tool_calls)


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