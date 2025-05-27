"""
Query service for handling LLM queries and RAG integration.
"""

from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass
import threading

from .base_service import BaseService, ServiceError
from ..api_provider.factory import provider_registry
from ..api_provider.config import APIProviderConfig
from ..api_provider.capabilities import ChatProvider, FunctionCallingProvider
from ..models.chat_message import ChatMessage, MessageRole
from ..models.tool_call import ToolCall
from ..models.api_response import APIResponse


@dataclass
class QueryRequest:
    """
    Represents a query request.
    
    Attributes:
        messages: List of chat messages
        use_rag: Whether to use RAG augmentation
        tools: Optional list of tools for function calling
        stream: Whether to stream the response
        provider_config: API provider configuration
        metadata: Additional request metadata
    """
    messages: List[ChatMessage]
    provider_config: APIProviderConfig
    use_rag: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    stream: bool = True
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QueryResponse:
    """
    Represents a query response.
    
    Attributes:
        content: Response text content
        tool_calls: List of tool calls if any
        usage: Token usage information
        metadata: Additional response metadata
    """
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryService(BaseService):
    """
    Service for handling LLM queries.
    
    This service manages the interaction with LLM providers,
    RAG augmentation, and response processing.
    """
    
    def __init__(self, rag_service=None):
        """
        Initialize the query service.
        
        Args:
            rag_service: Optional RAG service for document retrieval
        """
        super().__init__("query_service")
        self.rag_service = rag_service
        self._active_providers = {}
        self._lock = threading.Lock()
        
        # Configure logging
        import logging
        self.logger.setLevel(logging.DEBUG)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        self.logger.info("Query service initialized")
    
    def execute_query(self, request: QueryRequest, 
                     response_handler: Callable[[QueryResponse], None]) -> None:
        """
        Execute a query request.
        
        Args:
            request: The query request
            response_handler: Callback for handling responses
        """
        self.logger.info(f"Executing query with provider: {request.provider_config.name}, model: {request.provider_config.model}")
        self.logger.debug(f"Query details - messages: {len(request.messages)}, use_rag: {request.use_rag}, tools: {len(request.tools) if request.tools else 0}, stream: {request.stream}")
        
        try:
            # Reset stop event
            self.reset()
            self.logger.debug("Query service reset completed")
            
            # Get or create provider
            self.logger.debug("Getting provider instance")
            provider = self._get_provider(request.provider_config)
            self.logger.debug(f"Provider obtained: {type(provider).__name__}")
            
            # Augment with RAG if requested
            if request.use_rag and self.rag_service:
                self.logger.debug("Augmenting query with RAG")
                request = self._augment_with_rag(request)
                self.logger.debug("RAG augmentation completed")
            
            # Execute the query
            if request.tools and provider.supports_capability(FunctionCallingProvider):
                self.logger.debug("Executing function call query")
                self._execute_function_call_query(provider, request, response_handler)
            elif provider.supports_capability(ChatProvider):
                self.logger.debug("Executing chat query")
                self._execute_chat_query(provider, request, response_handler)
            else:
                self.logger.error("Provider does not support required capabilities")
                raise ServiceError("Provider does not support required capabilities")
                
            self.logger.info("Query execution completed successfully")
                
        except Exception as e:
            self.logger.error(f"Query execution failed: {type(e).__name__}: {e}")
            self.handle_error(e, "query execution")
            
            # Send error response
            error_response = QueryResponse(
                content=f"Error: {str(e)}",
                metadata={"error": True, "error_type": type(e).__name__}
            )
            
            try:
                self.logger.debug("Sending error response to handler")
                response_handler(error_response)
                self.logger.debug("Error response sent successfully")
            except Exception as handler_error:
                self.logger.error(f"Error in response handler: {handler_error}")
    
    def _get_provider(self, config: APIProviderConfig):
        """Get or create a provider instance."""
        provider_key = f"{config.provider_type.value}_{config.name}"
        
        with self._lock:
            if provider_key not in self._active_providers:
                self._active_providers[provider_key] = provider_registry.create_provider(config)
            
            return self._active_providers[provider_key]
    
    def _augment_with_rag(self, request: QueryRequest) -> QueryRequest:
        """
        Augment the request with RAG context.
        
        Args:
            request: Original request
            
        Returns:
            Augmented request
        """
        if not self.rag_service or not request.messages:
            return request
        
        # Get the last user message for RAG context
        last_user_message = None
        for msg in reversed(request.messages):
            if msg.role == MessageRole.USER:
                last_user_message = msg
                break
        
        if not last_user_message:
            return request
        
        try:
            # Get RAG context
            rag_results = self.rag_service.query(last_user_message.content)
            
            if rag_results:
                # Create context string
                context_parts = []
                for result in rag_results:
                    context_parts.append(f"Source: {result.get('metadata', {}).get('source', 'Unknown')}")
                    context_parts.append(result.get('text', ''))
                    context_parts.append("")  # Empty line for separation
                
                context = "\n".join(context_parts)
                
                # Create new messages with RAG context
                new_messages = request.messages.copy()
                
                # Add context before the last user message
                context_msg = ChatMessage.user(f"Context:\n{context}\n\nQuery: {last_user_message.content}")
                new_messages[-1] = context_msg
                
                # Create new request
                request = QueryRequest(
                    messages=new_messages,
                    provider_config=request.provider_config,
                    use_rag=request.use_rag,
                    tools=request.tools,
                    stream=request.stream,
                    metadata=request.metadata
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to augment with RAG: {e}")
        
        return request
    
    def _execute_chat_query(self, provider: ChatProvider, request: QueryRequest,
                          response_handler: Callable[[QueryResponse], None]) -> None:
        """Execute a chat query."""
        self.logger.debug(f"Executing chat query - streaming: {request.stream}")
        
        try:
            if request.stream:
                self.logger.debug("Starting streaming chat query")
                
                def stream_handler(content: str):
                    if not self.is_stopped():
                        self.logger.debug(f"Received streaming content, length: {len(content) if content else 0}")
                        try:
                            response = QueryResponse(content=content)
                            self.logger.debug("Created QueryResponse, calling response_handler")
                            response_handler(response)
                            self.logger.debug("Response handler completed for streaming content")
                        except Exception as e:
                            self.logger.error(f"Error in stream response handler: {e}")
                    else:
                        self.logger.debug("Skipping stream handler - service stopped")
                
                self.logger.debug("Calling provider.stream_chat_completion")
                provider.stream_chat_completion(request.messages, stream_handler)
                self.logger.debug("stream_chat_completion call completed")
                
            else:
                self.logger.debug("Starting non-streaming chat query")
                content = provider.create_chat_completion(request.messages)
                self.logger.debug(f"Received non-streaming content, length: {len(content) if content else 0}")
                
                response = QueryResponse(content=content)
                self.logger.debug("Created QueryResponse, calling response_handler")
                response_handler(response)
                self.logger.debug("Response handler completed for non-streaming content")
                
        except Exception as e:
            self.logger.error(f"Error in chat query execution: {type(e).__name__}: {e}")
            raise
    
    def _execute_function_call_query(self, provider: FunctionCallingProvider, 
                                   request: QueryRequest,
                                   response_handler: Callable[[QueryResponse], None]) -> None:
        """Execute a function call query."""
        if request.stream:
            def stream_handler(tool_calls: List[ToolCall]):
                if not self.is_stopped():
                    response = QueryResponse(tool_calls=tool_calls)
                    response_handler(response)
            
            provider.stream_function_call(request.messages, request.tools, stream_handler)
        else:
            tool_calls = provider.create_function_call(request.messages, request.tools)
            response = QueryResponse(tool_calls=tool_calls)
            response_handler(response)
    
    def stop_all_queries(self) -> None:
        """Stop all active queries."""
        self.stop()
        
        with self._lock:
            for provider in self._active_providers.values():
                try:
                    provider.stop_streaming()
                except Exception as e:
                    self.logger.warning(f"Error stopping provider: {e}")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.stop_all_queries()
        
        with self._lock:
            for provider in self._active_providers.values():
                try:
                    provider.close()
                except Exception as e:
                    self.logger.warning(f"Error closing provider: {e}")
            
            self._active_providers.clear()