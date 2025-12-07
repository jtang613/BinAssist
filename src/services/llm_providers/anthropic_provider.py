#!/usr/bin/env python3

"""
Anthropic Provider - Implementation for Anthropic Claude API
Adapted from reference implementation for new architecture
"""

import asyncio
import json
import re
import math
from typing import List, Dict, Any, AsyncGenerator, Optional, Callable
from collections import Counter

try:
    import anthropic
except ImportError:
    raise ImportError("anthropic package not available. Install with: pip install anthropic")

from .base_provider import (
    BaseLLMProvider, APIProviderError, AuthenticationError, 
    RateLimitError, NetworkError
)
from ..models.llm_models import (
    ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse,
    ChatMessage, MessageRole, ToolCall, ToolResult, Usage, ProviderCapabilities
)
from ..models.provider_types import ProviderType
from ..models.reasoning_models import ReasoningConfig

# Binary Ninja logging
try:
    import binaryninja
    log = binaryninja.log.Logger(0, "BinAssist")
except ImportError:
    # Fallback for testing outside Binary Ninja
    class MockLog:
        @staticmethod
        def log_info(msg): print(f"[BinAssist] INFO: {msg}")
        @staticmethod
        def log_error(msg): print(f"[BinAssist] ERROR: {msg}")
        @staticmethod
        def log_warn(msg): print(f"[BinAssist] WARN: {msg}")
        @staticmethod
        def log_debug(msg): print(f"[BinAssist] DEBUG: {msg}")
    log = MockLog()


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic API provider for Claude models.
    
    Supports chat completions, streaming, and function calling.
    Uses TF-IDF for embeddings as Anthropic doesn't provide native embeddings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Anthropic provider"""
        super().__init__(config)
        
        # Validate configuration
        self.validate_config()
        if not self.api_key:
            raise ValueError("API key is required for Anthropic provider")
        
        # Initialize Anthropic client
        try:
            # Handle base_url - Anthropic client expects None for default
            base_url = None if self.url == 'https://api.anthropic.com' else self.url

            # Handle TLS verification settings and create client
            if self.disable_tls:
                import httpx
                import ssl
                log.log_warn(f"TLS verification disabled for Anthropic provider '{self.name}'")

                # Create SSL context that doesn't verify certificates
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

                # Create httpx client with disabled verification
                http_client = httpx.Client(verify=False, timeout=30.0)

                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    base_url=base_url,
                    timeout=30.0,
                    max_retries=0,  # We handle retries ourselves
                    http_client=http_client
                )
            else:
                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    base_url=base_url,
                    timeout=30.0,
                    max_retries=0  # We handle retries ourselves
                )
            
        except Exception as e:
            raise APIProviderError(f"Failed to initialize Anthropic client: {e}")
    
    async def chat_completion(self, request: ChatRequest,
                            native_message_callback: Optional[Callable[[Dict[str, Any], ProviderType], None]] = None) -> ChatResponse:
        """Generate non-streaming chat completion with rate limit retry"""
        return await self._with_rate_limit_retry(self._chat_completion_impl, request, native_message_callback)

    async def _chat_completion_impl(self, request: ChatRequest,
                            native_message_callback: Optional[Callable[[Dict[str, Any], ProviderType], None]] = None) -> ChatResponse:
        """Internal implementation of chat completion"""
        try:
            # Convert to Anthropic format
            anthropic_messages = self._prepare_messages(request.messages)
            system_message = self._extract_system_message(request.messages)
            
            # Prepare payload
            payload = {
                "model": request.model or self.model,
                "messages": anthropic_messages,
                "max_tokens": min(request.max_tokens, self.max_tokens),
                "stream": False
            }

            # Check if extended thinking will be enabled
            reasoning_effort = self.config.get('reasoning_effort', 'none')
            thinking_enabled = reasoning_effort and reasoning_effort != 'none'

            # Anthropic doesn't allow both temperature and top_p
            # When thinking is enabled, temperature must be 1
            if thinking_enabled:
                payload["temperature"] = 1
            elif request.temperature is not None:
                payload["temperature"] = request.temperature
            elif request.top_p is not None:
                payload["top_p"] = request.top_p

            if system_message:
                payload["system"] = system_message

            if request.tools:
                payload["tools"] = self._convert_tools_to_anthropic(request.tools)

            if request.stop:
                payload["stop_sequences"] = request.stop if isinstance(request.stop, list) else [request.stop]

            # Add extended thinking if configured
            if thinking_enabled:
                reasoning_config = ReasoningConfig.from_string(reasoning_effort)
                reasoning_config.max_tokens = min(request.max_tokens, self.max_tokens)

                budget = reasoning_config.get_anthropic_budget()
                if budget and budget > 0:
                    payload["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget
                    }
                    log.log_debug(f"Anthropic thinking enabled with budget: {budget} tokens (temperature=1)")

            # Make API call
            response = await asyncio.to_thread(self._client.messages.create, **payload)
            
            # Extract content and tool calls
            content = ""
            tool_calls = []
            
            for content_block in response.content:
                if content_block.type == "text":
                    content += content_block.text
                elif content_block.type == "tool_use":
                    tool_call = ToolCall(
                        id=content_block.id,
                        name=content_block.name,
                        arguments=content_block.input
                    )
                    tool_calls.append(tool_call)
            
            # Call native message callback with actual API response
            if native_message_callback:
                # Convert the actual Anthropic response to dict for storage
                native_message = {
                    "role": "assistant",
                    "content": [block.model_dump() for block in response.content],
                    "model": response.model,
                    "id": response.id,
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    }
                }
                native_message_callback(native_message, self.get_provider_type())
            
            # Create usage info
            usage = Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens
            )
            
            return ChatResponse(
                content=content,
                model=response.model,
                usage=usage,
                tool_calls=tool_calls if tool_calls else None,
                finish_reason=self._map_stop_reason(response.stop_reason),
                response_id=response.id
            )
            
        except anthropic.AuthenticationError as e:
            raise AuthenticationError(f"Anthropic authentication failed: {e}")
        except anthropic.RateLimitError as e:
            raise RateLimitError(f"Anthropic rate limit exceeded: {e}")
        except anthropic.BadRequestError as e:
            raise APIProviderError(f"Anthropic bad request (400): {e}")
        except anthropic.PermissionDeniedError as e:
            raise AuthenticationError(f"Anthropic permission denied (403): {e}")
        except anthropic.NotFoundError as e:
            raise APIProviderError(f"Anthropic resource not found (404): {e}")
        except anthropic.ConflictError as e:
            raise APIProviderError(f"Anthropic conflict error (409): {e}")
        except anthropic.UnprocessableEntityError as e:
            raise APIProviderError(f"Anthropic unprocessable entity (422): {e}")
        except anthropic.InternalServerError as e:
            raise APIProviderError(f"Anthropic internal server error (500): {e}")
        except anthropic.APIConnectionError as e:
            raise NetworkError(f"Anthropic connection error: {e}")
        except anthropic.APITimeoutError as e:
            raise NetworkError(f"Anthropic timeout error: {e}")
        except anthropic.APIError as e:
            # Catch-all for other API errors
            raise APIProviderError(f"Anthropic API error: {e}")
        except Exception as e:
            raise APIProviderError(f"Unexpected error during chat completion: {e}")
    
    async def chat_completion_stream(self, request: ChatRequest,
                                   native_message_callback: Optional[Callable[[Dict[str, Any], ProviderType], None]] = None) -> AsyncGenerator[ChatResponse, None]:
        """Generate streaming chat completion with rate limit retry"""
        async for response in self._with_rate_limit_retry_stream(self._chat_completion_stream_impl, request, native_message_callback):
            yield response

    async def _chat_completion_stream_impl(self, request: ChatRequest,
                                   native_message_callback: Optional[Callable[[Dict[str, Any], ProviderType], None]] = None) -> AsyncGenerator[ChatResponse, None]:
        """Internal implementation of streaming chat completion"""
        try:
            # Convert to Anthropic format
            anthropic_messages = self._prepare_messages(request.messages)
            system_message = self._extract_system_message(request.messages)
            
            # Prepare payload (note: stream=True is not passed to messages.stream())
            payload = {
                "model": request.model or self.model,
                "messages": anthropic_messages,
                "max_tokens": min(request.max_tokens, self.max_tokens),
            }

            # Check if extended thinking will be enabled
            reasoning_effort = self.config.get('reasoning_effort', 'none')
            thinking_enabled = reasoning_effort and reasoning_effort != 'none'

            # Anthropic doesn't allow both temperature and top_p
            # When thinking is enabled, temperature must be 1
            if thinking_enabled:
                payload["temperature"] = 1
            elif request.temperature is not None:
                payload["temperature"] = request.temperature
            elif request.top_p is not None:
                payload["top_p"] = request.top_p

            if system_message:
                payload["system"] = system_message

            if request.tools:
                payload["tools"] = self._convert_tools_to_anthropic(request.tools)

            if request.stop:
                payload["stop_sequences"] = request.stop if isinstance(request.stop, list) else [request.stop]

            # Add extended thinking if configured (streaming)
            if thinking_enabled:
                reasoning_config = ReasoningConfig.from_string(reasoning_effort)
                reasoning_config.max_tokens = min(request.max_tokens, self.max_tokens)

                budget = reasoning_config.get_anthropic_budget()
                if budget and budget > 0:
                    payload["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget
                    }
                    log.log_debug(f"Anthropic streaming thinking enabled with budget: {budget} tokens (temperature=1)")

            # Stream response using context manager
            accumulated_content = ""
            tool_calls = []
            
            def _create_stream():
                return self._client.messages.stream(**payload)
            
            stream = await asyncio.to_thread(_create_stream)
            
            try:
                # Track tool calls being built during streaming
                building_tool_calls: Dict[str, Dict[str, Any]] = {}  # tool_id -> partial tool data
                
                # Use context manager for proper stream handling
                with stream as message_stream:
                    for event in message_stream:
                        # Handle text delta events
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, 'text'):
                                text_delta = event.delta.text
                                if text_delta:
                                    accumulated_content += text_delta
                                    
                                    # Create streaming response
                                    yield ChatResponse(
                                        content=text_delta,  # Send just the delta, not accumulated
                                        model=request.model or self.model,
                                        usage=Usage(0, 0, 0),  # Usage not available in streaming
                                        is_streaming=True,
                                        finish_reason="incomplete"
                                    )
                            
                            # Handle tool input delta events
                            elif hasattr(event.delta, 'partial_json'):
                                # Tool input is being built incrementally
                                # The tool_id should be in the event's index or content_block reference
                                tool_id = None
                                if hasattr(event, 'index'):
                                    # Find tool_id by matching content block index
                                    for tid, tool_data in building_tool_calls.items():
                                        if tool_data.get('index') == event.index:
                                            tool_id = tid
                                            break
                                
                                if tool_id and tool_id in building_tool_calls:
                                    # Accumulate partial JSON for tool input
                                    if 'input_json' not in building_tool_calls[tool_id]:
                                        building_tool_calls[tool_id]['input_json'] = ""
                                    building_tool_calls[tool_id]['input_json'] += event.delta.partial_json
                                    # Accumulating tool input parameters
                        
                        # Handle tool use start events
                        elif event.type == "content_block_start" and hasattr(event.content_block, 'type'):
                            if event.content_block.type == "tool_use":
                                # Start building this tool call
                                initial_input = event.content_block.input or {}
                                building_tool_calls[event.content_block.id] = {
                                    'id': event.content_block.id,
                                    'name': event.content_block.name,
                                    'input': initial_input,  # May be empty initially
                                    'input_json': "",
                                    'index': getattr(event, 'index', None)  # Store content block index for delta matching
                                }
                        
                        # Handle tool use completion events
                        elif event.type == "content_block_stop":
                            # Check if this was a tool use block being completed
                            # The stop event might not have content_block.id, so we need to match by index
                            tool_id = None
                            if hasattr(event, 'index'):
                                # Find tool by matching content block index
                                for tid, tool_data in building_tool_calls.items():
                                    if tool_data.get('index') == event.index:
                                        tool_id = tid
                                        break
                            
                            if tool_id and tool_id in building_tool_calls:
                                    # Complete the tool call
                                    tool_data = building_tool_calls[tool_id]
                                    
                                    # Use accumulated input or parse from JSON if needed
                                    final_input = tool_data['input']
                                    if tool_data.get('input_json'):
                                        try:
                                            import json
                                            final_input = json.loads(tool_data['input_json'])
                                        except json.JSONDecodeError:
                                            log.log_warn(f"Failed to parse tool input JSON: {tool_data['input_json']}")
                                    
                                    tool_call = ToolCall(
                                        id=tool_data['id'],
                                        name=tool_data['name'],
                                        arguments=final_input
                                    )
                                    tool_calls.append(tool_call)
                                    
                                    # Remove from building dict
                                    del building_tool_calls[tool_id]
                
                # Handle any remaining uncompleted tool calls as fallback
                for tool_id, tool_data in building_tool_calls.items():
                    final_input = tool_data['input']
                    if tool_data.get('input_json'):
                        try:
                            import json
                            final_input = json.loads(tool_data['input_json'])
                        except json.JSONDecodeError:
                            log.log_warn(f"Failed to parse tool input JSON: {tool_data['input_json']}")
                    
                    tool_call = ToolCall(
                        id=tool_data['id'],
                        name=tool_data['name'],
                        arguments=final_input
                    )
                    tool_calls.append(tool_call)
                
                # Call native message callback with complete streaming response
                if native_message_callback and (accumulated_content or tool_calls):
                    # Reconstruct the native Anthropic format from streaming data
                    content_blocks = []
                    if accumulated_content:
                        content_blocks.append({"type": "text", "text": accumulated_content})
                    
                    for tool_call in tool_calls:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "input": tool_call.arguments
                        })
                    
                    native_message = {
                        "role": "assistant",
                        "content": content_blocks,
                        "model": request.model or self.model,
                        # Note: streaming responses don't have usage/id info
                        "streaming": True
                    }
                    native_message_callback(native_message, self.get_provider_type())
                
                # Final response with complete content or tool calls
                if accumulated_content or tool_calls:
                    yield ChatResponse(
                        content="",  # Empty for final signal
                        model=request.model or self.model,
                        usage=Usage(0, 0, 0),  # Usage not available in streaming
                        tool_calls=tool_calls if tool_calls else None,
                        is_streaming=False,
                        finish_reason="tool_calls" if tool_calls else "stop"
                    )
                
            except Exception as e:
                # If context manager fails, try the simpler approach
                try:
                    for event in stream:
                        if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                            text_delta = event.delta.text
                            if text_delta:
                                yield ChatResponse(
                                    content=text_delta,
                                    model=request.model or self.model,
                                    usage=Usage(0, 0, 0),
                                    is_streaming=True,
                                    finish_reason="incomplete"
                                )
                except:
                    raise e
                
        except anthropic.AuthenticationError as e:
            raise AuthenticationError(f"Anthropic authentication failed: {e}")
        except anthropic.RateLimitError as e:
            raise RateLimitError(f"Anthropic rate limit exceeded: {e}")
        except anthropic.BadRequestError as e:
            raise APIProviderError(f"Anthropic bad request (400): {e}")
        except anthropic.PermissionDeniedError as e:
            raise AuthenticationError(f"Anthropic permission denied (403): {e}")
        except anthropic.NotFoundError as e:
            raise APIProviderError(f"Anthropic resource not found (404): {e}")
        except anthropic.ConflictError as e:
            raise APIProviderError(f"Anthropic conflict error (409): {e}")
        except anthropic.UnprocessableEntityError as e:
            raise APIProviderError(f"Anthropic unprocessable entity (422): {e}")
        except anthropic.InternalServerError as e:
            raise APIProviderError(f"Anthropic internal server error (500): {e}")
        except anthropic.APIConnectionError as e:
            raise NetworkError(f"Anthropic connection error: {e}")
        except anthropic.APITimeoutError as e:
            raise NetworkError(f"Anthropic timeout error: {e}")
        except anthropic.APIError as e:
            # Catch-all for other API errors
            raise APIProviderError(f"Anthropic API error: {e}")
        except Exception as e:
            raise APIProviderError(f"Unexpected error during streaming: {e}")
    
    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings using TF-IDF (Anthropic doesn't provide native embeddings)
        """
        try:
            # Generate TF-IDF embeddings as fallback
            embeddings = await asyncio.to_thread(self._generate_tfidf_embeddings, request.texts)
            
            # Create usage info (estimated)
            total_tokens = sum(len(text.split()) for text in request.texts)
            usage = Usage(
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens
            )
            
            return EmbeddingResponse(
                embeddings=embeddings,
                model=request.model,
                usage=usage,
                dimensions=len(embeddings[0]) if embeddings else 0
            )
            
        except Exception as e:
            raise APIProviderError(f"Error generating TF-IDF embeddings: {e}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Synchronous wrapper for embeddings generation (for RAG service compatibility)
        """
        try:
            return self._generate_tfidf_embeddings(texts)
        except Exception as e:
            raise APIProviderError(f"Error generating TF-IDF embeddings: {e}")
    
    async def test_connection(self) -> bool:
        """Test connectivity to Anthropic API"""
        try:
            # Create simple test request
            test_request = ChatRequest(
                messages=[
                    ChatMessage(role=MessageRole.USER, content="Test message. Please respond with 'OK'.")
                ],
                model=self.model,
                max_tokens=10
            )
            
            response = await self.chat_completion(test_request)
            return bool(response.content)
            
        except Exception:
            return False
    
    def get_capabilities(self) -> ProviderCapabilities:
        """Get Anthropic provider capabilities"""
        return ProviderCapabilities(
            supports_chat=True,
            supports_streaming=True,
            supports_tools=True,
            supports_embeddings=True,  # Via TF-IDF fallback
            supports_vision=False,  # Not implemented yet
            max_tokens=self.max_tokens,
            models=[
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022", 
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ]
        )
    
    def get_provider_type(self) -> ProviderType:
        """Get the provider type for this provider"""
        return ProviderType.ANTHROPIC
    
    def format_tool_results_for_continuation(self, tool_calls: List[ToolCall], tool_results: List[str]) -> List[Dict[str, Any]]:
        """
        Format tool execution results for Anthropic conversation continuation.
        
        Anthropic uses a different format than OpenAI - tool results go in user messages
        with content blocks containing tool_result type.
        """
        messages = []
        
        if tool_calls and tool_results:
            # Create user message with tool results as content blocks
            content_blocks = []
            
            for tool_call, result in zip(tool_calls, tool_results):
                content_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result
                })
            
            messages.append({
                "role": "user",
                "content": content_blocks
            })
        
        return messages
    
    def _prepare_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert messages to Anthropic format"""
        anthropic_messages = []
        
        for message in messages:
            # Skip system messages - handled separately
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
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content
                    }]
                }
            
            anthropic_messages.append(anthropic_msg)
        
        return anthropic_messages
    
    def _extract_system_message(self, messages: List[ChatMessage]) -> Optional[str]:
        """Extract system message content"""
        for message in messages:
            if message.role == MessageRole.SYSTEM:
                return message.content
        return None
    
    def _convert_tools_to_anthropic(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic tool format"""
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
    
    def _map_stop_reason(self, stop_reason: str) -> str:
        """Map Anthropic stop reasons to OpenAI format"""
        mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls"
        }
        return mapping.get(stop_reason, stop_reason)
    
    def _generate_tfidf_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate TF-IDF embeddings as fallback for Anthropic"""
        # Use a persistent vocabulary to ensure consistent dimensions
        if not hasattr(self, '_global_vocabulary'):
            self._global_vocabulary = {}
            self._global_tokenized_texts = []
        
        # Tokenize texts
        tokenized_texts = [self._tokenize(text) for text in texts]
        
        # If this is the first call or we have many texts (bulk ingestion), rebuild vocabulary
        if len(texts) > 10 or not self._global_vocabulary:
            # This is likely bulk ingestion - build vocabulary from these texts
            vocabulary = self._build_vocabulary(tokenized_texts)
            self._global_vocabulary = vocabulary
            self._global_tokenized_texts = tokenized_texts
        else:
            # This is likely a query - use existing vocabulary
            vocabulary = self._global_vocabulary
        
        # Calculate TF-IDF vectors using consistent vocabulary
        embeddings = []
        for tokens in tokenized_texts:
            tfidf_vector = self._calculate_tfidf(tokens, self._global_tokenized_texts, vocabulary)
            embeddings.append(tfidf_vector)
        
        return embeddings
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for TF-IDF"""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _build_vocabulary(self, tokenized_texts: List[List[str]]) -> Dict[str, int]:
        """Build vocabulary from all texts"""
        word_counts = Counter()
        for tokens in tokenized_texts:
            word_counts.update(set(tokens))  # Only count once per document
        
        # Take top N most common words to limit dimension
        vocab_size = min(1000, len(word_counts))
        vocabulary = {}
        for i, (word, _) in enumerate(word_counts.most_common(vocab_size)):
            vocabulary[word] = i
        
        return vocabulary
    
    def _calculate_tfidf(self, tokens: List[str], all_tokenized_texts: List[List[str]], 
                        vocabulary: Dict[str, int]) -> List[float]:
        """Calculate TF-IDF vector for a document"""
        # Initialize vector
        vector = [0.0] * len(vocabulary)
        
        # Calculate term frequencies
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        for word, tf_count in token_counts.items():
            if word in vocabulary:
                # Term frequency
                tf = tf_count / total_tokens if total_tokens > 0 else 0
                
                # Document frequency
                df = sum(1 for doc_tokens in all_tokenized_texts if word in doc_tokens)
                
                # Inverse document frequency
                idf = math.log(len(all_tokenized_texts) / df) if df > 0 else 0
                
                # TF-IDF score
                tfidf = tf * idf
                vector[vocabulary[word]] = tfidf
        
        # Normalize vector (L2 normalization)
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        
        return vector
    
# Factory for Anthropic provider
from .provider_factory import ProviderFactory
from ..models.provider_types import ProviderType

class AnthropicProviderFactory(ProviderFactory):
    """Factory for creating Anthropic provider instances"""
    
    def create_provider(self, config: Dict[str, Any]) -> AnthropicProvider:
        """Create Anthropic provider instance"""
        return AnthropicProvider(config)
    
    def supports_provider_type(self, provider_type: ProviderType) -> bool:
        """Check if this factory supports the provider type"""
        return provider_type == ProviderType.ANTHROPIC