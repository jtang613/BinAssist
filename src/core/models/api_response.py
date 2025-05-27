"""
API response data model.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from .tool_call import ToolCall


@dataclass
class APIResponse:
    """
    Represents a response from an API provider.
    
    Attributes:
        content: The text content of the response
        tool_calls: List of tool calls if any
        model: Name of the model that generated the response
        usage: Token usage information
        finish_reason: Reason why the response finished
        metadata: Additional response metadata
    """
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def has_tool_calls(self) -> bool:
        """Check if this response contains tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0
    
    def has_content(self) -> bool:
        """Check if this response contains text content."""
        return self.content is not None and len(self.content.strip()) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {}
        
        if self.content is not None:
            result["content"] = self.content
            
        if self.tool_calls is not None:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
            
        if self.model is not None:
            result["model"] = self.model
            
        if self.usage is not None:
            result["usage"] = self.usage
            
        if self.finish_reason is not None:
            result["finish_reason"] = self.finish_reason
            
        if self.metadata is not None:
            result["metadata"] = self.metadata
            
        return result