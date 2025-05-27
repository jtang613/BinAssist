"""
Capability interfaces for API providers.

These interfaces define the capabilities that different providers can implement,
following the interface segregation principle.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Optional
from ..models.chat_message import ChatMessage
from ..models.tool_call import ToolCall


class ChatProvider(ABC):
    """Interface for providers that support basic chat completion."""
    
    @abstractmethod
    def create_chat_completion(self, messages: List[ChatMessage], **kwargs) -> str:
        """
        Create a chat completion.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    def stream_chat_completion(self, messages: List[ChatMessage], 
                             response_handler: Callable[[str], None], **kwargs) -> None:
        """
        Stream a chat completion.
        
        Args:
            messages: List of chat messages
            response_handler: Callback function to handle streaming response chunks
            **kwargs: Additional provider-specific parameters
        """
        pass


class FunctionCallingProvider(ABC):
    """Interface for providers that support function/tool calling."""
    
    @abstractmethod
    def create_function_call(self, messages: List[ChatMessage], 
                           tools: List[Dict[str, Any]], **kwargs) -> List[ToolCall]:
        """
        Create a function call completion.
        
        Args:
            messages: List of chat messages
            tools: List of available tools/functions
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of tool calls to execute
        """
        pass
    
    @abstractmethod
    def stream_function_call(self, messages: List[ChatMessage], 
                           tools: List[Dict[str, Any]],
                           response_handler: Callable[[List[ToolCall]], None], 
                           **kwargs) -> None:
        """
        Stream a function call completion.
        
        Args:
            messages: List of chat messages
            tools: List of available tools/functions
            response_handler: Callback function to handle tool calls
            **kwargs: Additional provider-specific parameters
        """
        pass


class EmbeddingProvider(ABC):
    """Interface for providers that support text embeddings."""
    
    @abstractmethod
    def get_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.
        
        Returns:
            Embedding dimension
        """
        pass


class ModelListProvider(ABC):
    """Interface for providers that support listing available models."""
    
    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models.
        
        Returns:
            List of model information dictionaries
        """
        pass
    
    @abstractmethod
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        pass