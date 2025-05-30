"""
Tool call data models.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import json


@dataclass
class ToolCall:
    """
    Represents a tool/function call request.
    
    Attributes:
        id: Unique identifier for the tool call
        call_id: Call identifier used for linking responses (OpenAI format)
        name: Name of the tool/function to call
        arguments: Arguments to pass to the tool (as dict)
        metadata: Additional metadata
    """
    id: str
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments)
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCall':
        """Create ToolCall from dictionary data."""
        function_data = data.get("function", {})
        arguments_str = function_data.get("arguments", "{}")
        
        try:
            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        except json.JSONDecodeError:
            arguments = {}
            
        return cls(
            id=data.get("id", ""),
            name=function_data.get("name", ""),
            arguments=arguments,
            metadata=data.get("metadata")
        )


@dataclass
class ToolResult:
    """
    Represents the result of a tool execution.
    
    Attributes:
        tool_call_id: ID of the tool call this result is for
        result: The result data (can be any serializable type)
        success: Whether the tool execution was successful
        error: Error message if execution failed
        metadata: Additional metadata
    """
    tool_call_id: str
    result: Any
    success: bool = True
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "tool_call_id": self.tool_call_id,
            "result": self.result,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata
        }
    
    @classmethod
    def success_result(cls, tool_call_id: str, result: Any) -> 'ToolResult':
        """Create a successful tool result."""
        return cls(tool_call_id=tool_call_id, result=result, success=True)
    
    @classmethod
    def error_result(cls, tool_call_id: str, error: str) -> 'ToolResult':
        """Create an error tool result."""
        return cls(tool_call_id=tool_call_id, result=None, success=False, error=error)