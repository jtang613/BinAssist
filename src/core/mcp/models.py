"""
Unified MCP data models for BinAssist.

This module provides clean, reusable data structures for MCP integration,
eliminating duplication between client implementations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class MCPConnectionStatus(Enum):
    """Status of MCP server connection."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"


@dataclass
class MCPTool:
    """Represents an MCP tool with formatting capabilities."""
    name: str
    description: str
    schema: Dict[str, Any]
    server_name: str
    
    def to_llm_format(self, prefix: str = "") -> Dict[str, Any]:
        """
        Convert to OpenAI tool format for LLM integration.
        
        Args:
            prefix: Prefix to add to tool name (default: no prefix)
            
        Returns:
            Tool definition in OpenAI format
        """
        return {
            "type": "function",
            "function": {
                "name": f"{prefix}{self.name}",
                "description": self.description,
                "parameters": self.schema
            }
        }
    
    def to_test_format(self) -> Dict[str, Any]:
        """
        Convert to test result format for UI display.
        
        Returns:
            Tool info for test results
        """
        return {
            "name": self.name,
            "description": self.description,
            "server": self.server_name
        }


@dataclass
class MCPResource:
    """Represents an MCP resource."""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None
    server_name: Optional[str] = None
    
    def to_test_format(self) -> Dict[str, Any]:
        """
        Convert to test result format for UI display.
        
        Returns:
            Resource info for test results
        """
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description
        }


@dataclass
class MCPConnectionInfo:
    """Information about an MCP server connection."""
    server_config: 'MCPServerConfig'  # Import will be resolved at runtime
    status: MCPConnectionStatus
    tools: Dict[str, MCPTool]
    resources: Dict[str, MCPResource]
    error_message: Optional[str] = None
    
    @property
    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self.status == MCPConnectionStatus.CONNECTED
    
    @property
    def tools_count(self) -> int:
        """Get number of available tools."""
        return len(self.tools)
    
    @property
    def resources_count(self) -> int:
        """Get number of available resources."""
        return len(self.resources)


@dataclass
class MCPTestResult:
    """Result of testing an MCP server connection."""
    success: bool
    tools_count: int = 0
    resources_count: int = 0
    tools: List[Dict[str, Any]] = None
    resources: List[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        """Initialize lists if None."""
        if self.tools is None:
            self.tools = []
        if self.resources is None:
            self.resources = []
    
    @classmethod
    def success_result(cls, tools: List[MCPTool], resources: List[MCPResource]) -> 'MCPTestResult':
        """Create successful test result."""
        return cls(
            success=True,
            tools_count=len(tools),
            resources_count=len(resources),
            tools=[tool.to_test_format() for tool in tools],
            resources=[resource.to_test_format() for resource in resources]
        )
    
    @classmethod
    def failure_result(cls, error: str) -> 'MCPTestResult':
        """Create failed test result."""
        return cls(
            success=False,
            error=error
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API compatibility."""
        return {
            "success": self.success,
            "tools_count": self.tools_count,
            "resources_count": self.resources_count,
            "tools": self.tools,
            "resources": self.resources,
            "error": self.error
        }