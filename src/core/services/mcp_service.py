"""
MCP Service for integrating MCP client functionality with BinAssist services.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from binaryninja.settings import Settings

from .base_service import BaseService
from ..mcp import MCPClient, MCPConfig, MCPServerConfig
from ..mcp.exceptions import MCPError, MCPConnectionError, MCPToolError, MCPResourceError

class MCPService(BaseService):
    """Service for managing MCP client connections and operations."""
    
    def __init__(self, settings: Settings):
        super().__init__("mcp")
        self.settings = settings
        self.client: Optional[MCPClient] = None
        self._initialized = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
    def initialize(self) -> bool:
        """Initialize the MCP service."""
        try:
            self.logger.info("Initializing MCP service")
            
            # Create or get event loop for async operations
            try:
                self.loop = asyncio.get_event_loop()
            except RuntimeError:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
            
            # Load MCP configuration from settings
            config = self._load_config()
            
            if not config.servers:
                self.logger.info("No MCP servers configured")
                self._initialized = True
                return True
            
            # Create MCP client
            self.client = MCPClient(config)
            
            # Connect to servers asynchronously
            self._run_async(self._connect_to_servers())
            
            self._initialized = True
            self.logger.info("MCP service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP service: {e}")
            return False
    
    def _load_config(self) -> MCPConfig:
        """Load MCP configuration from BinaryNinja settings."""
        servers = []
        
        # Check for configured MCP servers in settings
        # This is a placeholder - actual implementation would load from settings
        server_count = self.settings.get_integer('binassist.mcp_server_count', 0)
        
        for i in range(server_count):
            server_config = MCPServerConfig(
                name=self.settings.get_string(f'binassist.mcp_server_{i}_name', f'server_{i}'),
                transport_type=self.settings.get_string(f'binassist.mcp_server_{i}_transport', 'stdio'),
                command=self.settings.get_string(f'binassist.mcp_server_{i}_command', ''),
                args=self.settings.get_string_list(f'binassist.mcp_server_{i}_args', []),
                timeout=self.settings.get_integer(f'binassist.mcp_server_{i}_timeout', 30),
                enabled=self.settings.get_bool(f'binassist.mcp_server_{i}_enabled', True)
            )
            servers.append(server_config)
        
        return MCPConfig(
            servers=servers,
            global_timeout=self.settings.get_integer('binassist.mcp_global_timeout', 60),
            max_concurrent_connections=self.settings.get_integer('binassist.mcp_max_connections', 5),
            retry_attempts=self.settings.get_integer('binassist.mcp_retry_attempts', 3)
        )
    
    def _run_async(self, coro):
        """Run an async coroutine in the event loop."""
        if self.loop and not self.loop.is_running():
            return self.loop.run_until_complete(coro)
        else:
            # If loop is running, create a task
            return asyncio.create_task(coro)
    
    async def _connect_to_servers(self):
        """Connect to all configured MCP servers."""
        if not self.client:
            return
            
        try:
            results = await self.client.connect_all()
            
            connected_count = sum(1 for success in results.values() if success)
            total_count = len(results)
            
            self.logger.info(f"Connected to {connected_count}/{total_count} MCP servers")
            
            if connected_count > 0:
                tools = self.client.get_available_tools()
                resources = self.client.get_available_resources()
                self.logger.info(f"Available MCP tools: {len(tools)}, resources: {len(resources)}")
                
        except Exception as e:
            self.logger.error(f"Error connecting to MCP servers: {e}")
    
    def is_available(self) -> bool:
        """Check if MCP service is available and has connections."""
        return self._initialized and self.client is not None and len(self.client.connections) > 0
    
    def get_available_tools(self) -> Dict[str, Any]:
        """Get all available MCP tools."""
        if not self.is_available():
            return {}
            
        tools = self.client.get_available_tools()
        return {name: {
            'name': tool.name,
            'description': tool.description,
            'schema': tool.schema,
            'server': tool.server_name
        } for name, tool in tools.items()}
    
    def get_available_resources(self) -> Dict[str, Any]:
        """Get all available MCP resources."""
        if not self.is_available():
            return {}
            
        resources = self.client.get_available_resources()
        return {uri: {
            'uri': resource.uri,
            'name': resource.name,
            'description': resource.description,
            'mime_type': resource.mime_type,
            'server': resource.server_name
        } for uri, resource in resources.items()}
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool synchronously."""
        if not self.is_available():
            raise MCPError("MCP service not available")
            
        try:
            return self._run_async(self.client.call_tool(tool_name, arguments))
        except Exception as e:
            self.logger.error(f"Error calling MCP tool '{tool_name}': {e}")
            raise MCPToolError(f"Tool call failed: {e}")
    
    def get_resource(self, uri: str) -> Any:
        """Get an MCP resource synchronously."""
        if not self.is_available():
            raise MCPError("MCP service not available")
            
        try:
            return self._run_async(self.client.get_resource(uri))
        except Exception as e:
            self.logger.error(f"Error getting MCP resource '{uri}': {e}")
            raise MCPResourceError(f"Resource access failed: {e}")
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get MCP tools formatted for LLM tool calling."""
        if not self.is_available():
            return []
            
        tools = self.client.get_available_tools()
        llm_tools = []
        
        for name, tool in tools.items():
            llm_tool = {
                "type": "function",
                "function": {
                    "name": f"mcp_{name}",  # Prefix to distinguish from native tools
                    "description": f"MCP Tool: {tool.description}",
                    "parameters": tool.schema
                }
            }
            llm_tools.append(llm_tool)
            
        return llm_tools
    
    def handle_llm_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Handle an LLM tool call that targets an MCP tool."""
        # Remove mcp_ prefix if present
        if tool_name.startswith("mcp_"):
            tool_name = tool_name[4:]
            
        return self.call_tool(tool_name, arguments)
    
    def cleanup(self):
        """Clean up MCP service resources."""
        if self.client:
            self._run_async(self.client.disconnect_all())
            self.client = None
        
        if self.loop:
            try:
                self.loop.close()
            except Exception as e:
                self.logger.warning(f"Error closing event loop: {e}")
            finally:
                self.loop = None
                
        self._initialized = False
        self.logger.info("MCP service cleaned up")