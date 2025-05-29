"""
MCP Client Configuration.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str
    transport_type: str  # "stdio" or "sse"
    command: Optional[str] = None  # For stdio transport
    args: Optional[List[str]] = None  # For stdio transport
    url: Optional[str] = None  # For SSE transport
    env: Optional[Dict[str, str]] = None  # Environment variables
    timeout: int = 30  # Connection timeout in seconds
    enabled: bool = True

@dataclass 
class MCPConfig:
    """Overall MCP configuration."""
    servers: List[MCPServerConfig]
    global_timeout: int = 60
    max_concurrent_connections: int = 5
    retry_attempts: int = 3
    
    def get_enabled_servers(self) -> List[MCPServerConfig]:
        """Get list of enabled servers."""
        return [server for server in self.servers if server.enabled]
    
    def get_server_by_name(self, name: str) -> Optional[MCPServerConfig]:
        """Get server configuration by name."""
        for server in self.servers:
            if server.name == name:
                return server
        return None