#!/usr/bin/env python3

"""
LiteLLM Provider - Implementation for LiteLLM proxy API
Handles AWS Bedrock and other providers via LiteLLM proxy
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator

try:
    import openai
except ImportError:
    raise ImportError("openai package not available. Install with: pip install openai")

from .openai_provider import OpenAIProvider
from .base_provider import APIProviderError, AuthenticationError, RateLimitError, NetworkError
from ..models.llm_models import (
    ChatMessage, ChatRequest, ChatResponse, ToolCall, Usage
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


class LiteLLMProvider(OpenAIProvider):
    """
    LiteLLM provider for proxy access to AWS Bedrock and other models.

    Handles LiteLLM-specific quirks:
    1. Bedrock requires tools=[] even for no-tool calls
    2. Different thinking block message format for Bedrock Anthropic models
    3. Model-family-specific parameter handling

    Extends OpenAIProvider since LiteLLM uses OpenAI-compatible API.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize LiteLLM provider"""
        # Initialize attributes BEFORE parent init (parent may access them)
        # We need model from config to detect family/bedrock
        self.model = config.get('model', '')
        self.model_family = self._detect_model_family()
        self.is_bedrock = self._is_bedrock_model()

        # Don't require API key for LiteLLM (depends on proxy config)
        # Override the OpenAI validation temporarily
        original_api_key = config.get('api_key', '')
        if not original_api_key:
            config['api_key'] = 'dummy-key-for-litellm'

        # Call parent init (will also set self.model from config)
        super().__init__(config)

        # Restore original api_key
        if not original_api_key:
            self.api_key = ''

        # Override timeout and retry settings for LiteLLM/Bedrock
        # Bedrock can be significantly slower than direct API calls
        # LiteLLM adds overhead, and Bedrock throttles aggressively
        try:
            import httpx
            # Recreate client with longer timeout (5 minutes for LiteLLM/Bedrock)
            timeout = 300.0
            log.log_info(f"LiteLLM: Using extended timeout of {timeout}s for reliability")

            if self.disable_tls:
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                http_client = httpx.Client(verify=False, timeout=timeout)
                self._client.close()  # Close old client
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key or 'dummy',
                    base_url=self.url if self.url else None,
                    timeout=timeout,
                    max_retries=0,
                    http_client=http_client
                )
            else:
                self._client.close()  # Close old client
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key or 'dummy',
                    base_url=self.url if self.url else None,
                    timeout=timeout,
                    max_retries=0
                )
        except Exception as e:
            log.log_warn(f"Failed to update LiteLLM client timeout: {e}")

        # Use longer retry delays for LiteLLM/Bedrock (5s base instead of 2s)
        self.rate_limit_min_delay = 5.0
        self.rate_limit_max_delay = 15.0

        log.log_info(f"LiteLLM provider initialized for model: {self.model}")
        log.log_info(f"  Model family: {self.model_family}")
        log.log_info(f"  Is Bedrock: {self.is_bedrock}")
        log.log_info(f"  Retry delays: {self.rate_limit_min_delay}s - {self.rate_limit_max_delay}s")

    def _detect_model_family(self) -> str:
        """
        Detect underlying model family from model name.

        Examples:
        - bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0 -> anthropic
        - bedrock/amazon.nova-pro-v1:0 -> amazon
        - bedrock/meta.llama3-70b-instruct-v1:0 -> meta
        - claude-3-5-sonnet -> anthropic (non-Bedrock)
        - gpt-4o -> openai (non-Bedrock)
        """
        model_lower = self.model.lower()

        # Bedrock models: bedrock/<provider>.<model-name>
        if model_lower.startswith('bedrock/'):
            if 'anthropic' in model_lower or 'claude' in model_lower:
                return 'anthropic'
            elif 'amazon' in model_lower or 'nova' in model_lower:
                return 'amazon'
            elif 'meta' in model_lower or 'llama' in model_lower:
                return 'meta'
            elif 'cohere' in model_lower:
                return 'cohere'
            elif 'ai21' in model_lower:
                return 'ai21'

        # Non-Bedrock LiteLLM models
        if 'claude' in model_lower or 'anthropic' in model_lower:
            return 'anthropic'
        elif 'gpt' in model_lower or 'openai' in model_lower:
            return 'openai'
        elif 'gemini' in model_lower or 'google' in model_lower:
            return 'google'
        elif 'llama' in model_lower or 'meta' in model_lower:
            return 'meta'

        return 'unknown'

    def _is_bedrock_model(self) -> bool:
        """Check if this is a Bedrock model"""
        return self.model.startswith('bedrock/')

    def _prepare_completion_kwargs(self, request: ChatRequest) -> Dict[str, Any]:
        """
        Prepare completion kwargs with LiteLLM/Bedrock quirks handled.

        Quirks:
        1. Bedrock requires tools=[] even when not using tools
        2. Bedrock Anthropic has different reasoning parameter handling
        """
        # Start with base kwargs from OpenAI provider logic
        openai_messages = self._prepare_messages(request.messages)

        completion_kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "stream": request.stream
        }

        # Handle token limits
        if self._is_reasoning_model():
            completion_kwargs["max_completion_tokens"] = min(
                request.max_tokens or self.max_tokens, self.max_tokens
            )
        else:
            completion_kwargs["max_tokens"] = min(
                request.max_tokens or self.max_tokens, self.max_tokens
            )

            # Handle temperature
            reasoning_effort_str = self.config.get('reasoning_effort', 'none')
            thinking_enabled = reasoning_effort_str and reasoning_effort_str != 'none'

            if thinking_enabled:
                completion_kwargs["temperature"] = 1
                log.log_debug("LiteLLM: Thinking/reasoning enabled, forcing temperature=1")
            elif request.temperature is not None:
                completion_kwargs["temperature"] = request.temperature

        # Handle tools parameter
        # Get tools from request (may be None, [], or a list of tools)
        request_tools = getattr(request, 'tools', None)

        if request_tools:
            # Tools provided by orchestrator (either for actual use or for Bedrock compatibility)
            completion_kwargs["tools"] = request_tools
            log.log_debug(f"LiteLLM: Using {len(request_tools)} tools from request")
        else:
            # LITELLM/BEDROCK QUIRK #1: Empty tools array for compatibility
            # The orchestrator handles detecting when tools are needed for Bedrock compatibility
            # If no tools provided, use empty array
            completion_kwargs["tools"] = []
            log.log_debug("LiteLLM: Added empty tools array for compatibility")

        # Handle reasoning effort
        reasoning_effort_str = self.config.get('reasoning_effort', 'none')
        if reasoning_effort_str and reasoning_effort_str != 'none':
            # LITELLM/BEDROCK QUIRK #2: Don't use reasoning_effort parameter
            # LiteLLM/Bedrock interpret this as requiring tool calling, causing errors
            # Skip reasoning_effort for all LiteLLM providers to avoid compatibility issues
            log.log_debug(f"LiteLLM: Skipping reasoning_effort param (not supported via LiteLLM proxy)")
            # Note: Reasoning/thinking may still work via model's native capabilities

        return completion_kwargs

    def _prepare_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """
        Convert BinAssist ChatMessage objects to LiteLLM format.

        Handles Bedrock Anthropic thinking block format:
        - When thinking enabled, assistant messages must start with thinking blocks
        """
        # Start with OpenAI format conversion
        openai_messages = super()._prepare_messages(messages)

        # BEDROCK ANTHROPIC QUIRK: Thinking blocks must come first in assistant messages
        if self.is_bedrock and self.model_family == 'anthropic':
            reasoning_enabled = self.config.get('reasoning_effort', 'none') != 'none'

            if reasoning_enabled:
                # For Bedrock Anthropic with thinking enabled, we need to ensure
                # assistant messages with thinking blocks have them first.
                # However, OpenAI format messages are text-based, not multi-part content.
                #
                # Since LiteLLM translates OpenAI format to Bedrock format,
                # and the error occurs at Bedrock level, we may need to pass
                # native_content or rely on LiteLLM's translation.
                #
                # For now, log a warning if this combination is used.
                log.log_warn(
                    "LiteLLM: Bedrock Anthropic with thinking enabled may have "
                    "format compatibility issues. Consider disabling reasoning_effort "
                    "or using native Anthropic provider."
                )

        return openai_messages

    def get_provider_type(self) -> ProviderType:
        """Return LITELLM provider type"""
        return ProviderType.LITELLM

    def _is_reasoning_model(self) -> bool:
        """
        Check if the current model is a reasoning model.

        Extends parent to handle Bedrock model naming.
        """
        # Check if Bedrock model wraps a reasoning model
        if self.is_bedrock:
            model_lower = self.model.lower()
            # Bedrock doesn't currently support o* models, but future-proof
            if any(x in model_lower for x in ['o1', 'o2', 'o3', 'o4', 'gpt-5']):
                return True

        # Fall back to parent logic
        return super()._is_reasoning_model()

    async def _chat_completion_impl(self, request: ChatRequest,
                                    native_message_callback: Optional[Callable[[Dict[str, Any], ProviderType], None]] = None) -> ChatResponse:
        """Internal implementation of chat completion with LiteLLM/Bedrock quirks"""
        try:
            # Use our quirk-aware completion kwargs builder
            completion_kwargs = self._prepare_completion_kwargs(request)
            completion_kwargs["stream"] = False

            # Make API call
            response = self._client.chat.completions.create(**completion_kwargs)

            if not response.choices:
                raise APIProviderError("No choices in LiteLLM response")

            message = response.choices[0].message
            content = message.content or ""

            # Extract tool calls if present
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        log.log_warn(f"Failed to parse tool call arguments: {tc.function.arguments}")
                        arguments = {}

                    tool_call = ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments
                    )
                    tool_calls.append(tool_call)

            # Call native message callback with actual response
            if native_message_callback:
                choice = response.choices[0]
                native_message = {
                    "role": "assistant",
                    "content": choice.message.content,
                    "model": response.model,
                    "id": response.id,
                    "object": response.object,
                    "created": response.created,
                    "finish_reason": choice.finish_reason
                }

                if choice.message.tool_calls:
                    native_message["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in choice.message.tool_calls
                    ]

                if response.usage:
                    native_message["usage"] = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }

                native_message_callback(native_message, self.get_provider_type())

            # Calculate usage
            usage = None
            if response.usage:
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )

            return ChatResponse(
                content=content,
                model=self.model,
                usage=usage,
                tool_calls=tool_calls,
                finish_reason=response.choices[0].finish_reason or "stop"
            )

        except openai.AuthenticationError as e:
            log.log_error(f"Authentication error: {e}")
            raise AuthenticationError(f"LiteLLM authentication failed: {e}")
        except openai.RateLimitError as e:
            log.log_error(f"Rate limit error: {e}")
            raise RateLimitError(f"LiteLLM rate limit exceeded: {e}")
        except openai.APITimeoutError as e:
            # Timeouts are common with Bedrock due to slow inference and throttling
            log.log_warn(f"LiteLLM request timeout (may be Bedrock throttling): {e}")
            raise NetworkError(f"LiteLLM request timed out: {e}")
        except openai.APIConnectionError as e:
            log.log_error(f"Connection error: {e}")
            # Check if this might be throttling
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['throttl', 'too many requests', '429', 'rate limit']):
                log.log_warn("Detected potential throttling - will retry with backoff")
                raise RateLimitError(f"LiteLLM throttled: {e}")
            raise NetworkError(f"LiteLLM connection failed: {e}")
        except openai.BadRequestError as e:
            # Enhanced error handling for Bedrock-specific errors
            error_message = str(e)
            if "doesn't support tool use in streaming mode" in error_message:
                log.log_error(
                    f"Bedrock streaming+tools error: {e}. "
                    "This should be handled automatically for Meta models. Please report this issue."
                )
            elif "doesn't support tool calling without" in error_message:
                log.log_error(
                    f"Bedrock tool calling error: {e}. "
                    "This should be handled automatically. Please report this issue."
                )
            elif "Expected `thinking` or `redacted_thinking`" in error_message:
                log.log_error(
                    f"Bedrock thinking format error: {e}. "
                    "Try disabling reasoning_effort in settings."
                )
            elif any(keyword in error_message.lower() for keyword in ['throttl', 'too many requests']):
                log.log_warn("Detected throttling in error message - will retry with backoff")
                raise RateLimitError(f"LiteLLM throttled: {e}")
            raise APIProviderError(f"LiteLLM API error: {e}")
        except openai.APIError as e:
            log.log_error(f"LiteLLM API error: {e}")
            raise APIProviderError(f"LiteLLM API error: {e}")
        except Exception as e:
            log.log_error(f"Chat completion failed: {e}")
            raise APIProviderError(f"Unexpected error during chat completion: {e}")

    async def _chat_completion_stream_impl(self, request: ChatRequest,
                                          native_message_callback: Optional[Callable[[Dict[str, Any], ProviderType], None]] = None) -> AsyncGenerator[ChatResponse, None]:
        """Internal implementation of streaming chat completion with LiteLLM/Bedrock quirks"""
        try:
            # LITELLM/BEDROCK QUIRK #3: Meta models don't support tool use in streaming mode
            # Fall back to non-streaming if Meta model + tools
            # Note: Applies to all Meta models since they're typically routed through Bedrock
            request_tools = getattr(request, 'tools', None)
            needs_nonstreaming_fallback = (
                self.model_family == 'meta' and
                request_tools and len(request_tools) > 0
            )

            if needs_nonstreaming_fallback:
                log.log_info(
                    f"LiteLLM: Using non-streaming mode for Meta model '{self.model}' with tools "
                    f"(streaming+tools not supported on Bedrock)"
                )
                # Call non-streaming API
                non_streaming_response = await self.chat_completion(request)

                # Simulate streaming by yielding the response in chunks
                # First yield content if present
                if non_streaming_response.content:
                    # Break content into smaller chunks for smoother UI updates
                    chunk_size = 50  # characters per chunk
                    content = non_streaming_response.content
                    for i in range(0, len(content), chunk_size):
                        chunk_text = content[i:i+chunk_size]
                        yield ChatResponse(
                            content=chunk_text,
                            model=self.model,
                            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                            tool_calls=[],
                            finish_reason="",
                            is_streaming=True
                        )
                        # Small delay to simulate streaming (optional)
                        await asyncio.sleep(0.01)

                # Finally yield the complete response with tool calls
                yield ChatResponse(
                    content="",  # Already sent as chunks
                    model=non_streaming_response.model,
                    usage=non_streaming_response.usage,
                    tool_calls=non_streaming_response.tool_calls,
                    finish_reason=non_streaming_response.finish_reason,
                    is_streaming=False  # Mark as final
                )
                return

            # Use our quirk-aware completion kwargs builder
            completion_kwargs = self._prepare_completion_kwargs(request)
            completion_kwargs["stream"] = True

            # Make streaming API call
            stream = self._client.chat.completions.create(**completion_kwargs)

            accumulated_content = ""
            building_tool_calls = {}  # tool_index -> partial tool data

            # Batching variables to reduce UI update frequency
            batch_content = ""
            batch_count = 0
            BATCH_SIZE = 10  # Emit every 10 chunks

            try:
                finished = False
                for chunk in stream:
                    if not chunk.choices or finished:
                        continue

                    choice = chunk.choices[0]
                    delta = choice.delta

                    # Handle content delta with batching
                    if delta.content is not None and delta.content:
                        accumulated_content += delta.content
                        batch_content += delta.content
                        batch_count += 1

                        # Emit batched content every BATCH_SIZE chunks
                        if batch_count >= BATCH_SIZE:
                            yield ChatResponse(
                                content=batch_content,
                                model=self.model,
                                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                                tool_calls=[],
                                finish_reason="",
                                is_streaming=True
                            )
                            # Reset batch
                            batch_content = ""
                            batch_count = 0

                    # Handle tool call deltas - accumulate across chunks
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            tool_index = tc_delta.index if hasattr(tc_delta, 'index') else 0

                            # Initialize tool call if not seen before
                            if tool_index not in building_tool_calls:
                                building_tool_calls[tool_index] = {
                                    'id': tc_delta.id or f"tool_call_{tool_index}",
                                    'name': '',
                                    'arguments': ''
                                }

                            # Update tool call data from delta
                            if tc_delta.id:
                                building_tool_calls[tool_index]['id'] = tc_delta.id

                            if tc_delta.function:
                                if tc_delta.function.name:
                                    building_tool_calls[tool_index]['name'] = tc_delta.function.name

                                if tc_delta.function.arguments:
                                    building_tool_calls[tool_index]['arguments'] += tc_delta.function.arguments

                    # Handle finish reason
                    if choice.finish_reason and not finished:
                        # Emit any remaining batched content
                        if batch_content:
                            yield ChatResponse(
                                content=batch_content,
                                model=self.model,
                                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                                tool_calls=[],
                                finish_reason="",
                                is_streaming=True
                            )

                        # Convert accumulated tool calls to ToolCall objects
                        final_tool_calls = []
                        for tool_index, tool_data in building_tool_calls.items():
                            if tool_data['name']:  # Only include tools with names
                                try:
                                    arguments = json.loads(tool_data['arguments']) if tool_data['arguments'] else {}
                                except json.JSONDecodeError:
                                    log.log_warn(f"Failed to parse tool call arguments: {tool_data['arguments']}")
                                    arguments = {}

                                tool_call = ToolCall(
                                    id=tool_data['id'],
                                    name=tool_data['name'],
                                    arguments=arguments
                                )
                                final_tool_calls.append(tool_call)

                        # Emit final response with tool calls
                        final_response = ChatResponse(
                            content="",  # Empty content since all sent as deltas
                            model=self.model,
                            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                            tool_calls=final_tool_calls,
                            finish_reason=choice.finish_reason,
                            is_streaming=False  # Mark as final response
                        )
                        yield final_response
                        finished = True

            except StopIteration:
                pass

        except openai.AuthenticationError as e:
            log.log_error(f"Authentication error: {e}")
            raise AuthenticationError(f"LiteLLM authentication failed: {e}")
        except openai.RateLimitError as e:
            log.log_error(f"Rate limit error: {e}")
            raise RateLimitError(f"LiteLLM rate limit exceeded: {e}")
        except openai.APITimeoutError as e:
            # Timeouts are common with Bedrock due to slow inference and throttling
            log.log_warn(f"LiteLLM streaming timeout (may be Bedrock throttling): {e}")
            raise NetworkError(f"LiteLLM streaming timed out: {e}")
        except openai.APIConnectionError as e:
            log.log_error(f"Connection error: {e}")
            # Check if this might be throttling
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['throttl', 'too many requests', '429', 'rate limit']):
                log.log_warn("Detected potential throttling - will retry with backoff")
                raise RateLimitError(f"LiteLLM throttled: {e}")
            raise NetworkError(f"LiteLLM connection failed: {e}")
        except openai.BadRequestError as e:
            # Enhanced error handling for Bedrock-specific errors
            error_message = str(e)
            if "doesn't support tool use in streaming mode" in error_message:
                log.log_error(
                    f"Bedrock streaming+tools error: {e}. "
                    "This should be handled automatically for Meta models. Please report this issue."
                )
            elif "doesn't support tool calling without" in error_message:
                log.log_error(
                    f"Bedrock tool calling error: {e}. "
                    "This should be handled automatically. Please report this issue."
                )
            elif "Expected `thinking` or `redacted_thinking`" in error_message:
                log.log_error(
                    f"Bedrock thinking format error: {e}. "
                    "Try disabling reasoning_effort in settings."
                )
            elif any(keyword in error_message.lower() for keyword in ['throttl', 'too many requests']):
                log.log_warn("Detected throttling in error message - will retry with backoff")
                raise RateLimitError(f"LiteLLM throttled: {e}")
            raise APIProviderError(f"LiteLLM API error: {e}")
        except openai.APIError as e:
            log.log_error(f"LiteLLM API error: {e}")
            raise APIProviderError(f"LiteLLM API error: {e}")
        except Exception as e:
            log.log_error(f"Streaming completion failed: {e}")
            raise APIProviderError(f"Unexpected error during streaming completion: {e}")
