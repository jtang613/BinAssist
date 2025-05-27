"""
Chat message data model.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class MessageRole(Enum):
    """Enumeration of message roles in a conversation."""
    SYSTEM = "system"
    USER = "user" 
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatMessage:
    """
    Represents a message in a chat conversation.
    
    Attributes:
        role: The role of the message sender
        content: The text content of the message
        tool_calls: Optional tool calls if this is an assistant message
        tool_call_id: Optional tool call ID if this is a tool response
        metadata: Additional metadata for the message
    """
    role: MessageRole
    content: str
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format for API calls."""
        result = {
            "role": self.role.value,
            "content": self.content
        }
        
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
            
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create a ChatMessage from dictionary data."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            metadata=data.get("metadata")
        )
    
    @classmethod
    def system(cls, content: str) -> 'ChatMessage':
        """Create a system message."""
        return cls(MessageRole.SYSTEM, content)
    
    @classmethod
    def user(cls, content: str) -> 'ChatMessage':
        """Create a user message."""
        return cls(MessageRole.USER, content)
    
    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[list] = None) -> 'ChatMessage':
        """Create an assistant message."""
        return cls(MessageRole.ASSISTANT, content, tool_calls=tool_calls)
    
    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> 'ChatMessage':
        """Create a tool response message."""
        return cls(MessageRole.TOOL, content, tool_call_id=tool_call_id)