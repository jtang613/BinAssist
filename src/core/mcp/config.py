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
    enabled: bool = True
    command: Optional[str] = None  # For stdio transport
    args: Optional[List[str]] = None  # For stdio transport
    url: Optional[str] = None  # For SSE transport
    env: Optional[Dict[str, str]] = None  # Environment variables
    timeout: int = 30  # Connection timeout in seconds
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'transport_type': self.transport_type,
            'enabled': self.enabled,
            'command': self.command,
            'args': self.args if self.args else [],
            'url': self.url,
            'env': self.env if self.env else {},
            'timeout': self.timeout
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MCPServerConfig':
        """Create from dictionary."""
        return cls(
            name=data.get('name', ''),
            transport_type=data.get('transport_type', 'stdio'),
            enabled=data.get('enabled', True),
            command=data.get('command'),
            args=data.get('args', []) if data.get('args') else None,
            url=data.get('url'),
            env=data.get('env', {}) if data.get('env') else None,
            timeout=data.get('timeout', 30)
        )

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