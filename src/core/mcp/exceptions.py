"""
MCP-specific exceptions for error handling.
"""

class MCPError(Exception):
    """Base exception for all MCP-related errors."""
    pass

class MCPConnectionError(MCPError):
    """Raised when connection to MCP server fails."""
    pass

class MCPTimeoutError(MCPError):
    """Raised when MCP operation times out."""
    pass

class MCPToolError(MCPError):
    """Raised when MCP tool execution fails."""
    pass

class MCPResourceError(MCPError):
    """Raised when MCP resource access fails."""
    pass