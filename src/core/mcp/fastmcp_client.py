"""
FastMCP-based MCP Client implementation for BinAssist.

This module replaces our custom MCP client with the FastMCP library
for better compatibility and reliability.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json

try:
    from fastmcp import Client as FastMCPClient
    FASTMCP_AVAILABLE = True
except ImportError:
    FastMCPClient = None
    FASTMCP_AVAILABLE = False

from .config import MCPConfig, MCPServerConfig
from .exceptions import MCPError, MCPConnectionError, MCPTimeoutError, MCPToolError, MCPResourceError

@dataclass
class MCPTool:
    """Represents an MCP tool."""
    name: str
    description: str
    schema: Dict[str, Any]
    server_name: str

@dataclass
class MCPResource:
    """Represents an MCP resource."""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None
    server_name: Optional[str] = None

class FastMCPConnection:
    """Manages connection to a single MCP server using FastMCP."""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.logger = logging.getLogger(f"binassist.fastmcp.{config.name}")
        self.connected = False
        self.client: Optional[FastMCPClient] = None
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        
    async def connect(self) -> bool:
        """Connect to the MCP server using FastMCP."""
        try:
            if not FASTMCP_AVAILABLE:
                raise MCPError("FastMCP library not available. Install with: pip install fastmcp")
            
            self.logger.info(f"Connecting to MCP server: {self.config.name}")
            
            # Create appropriate FastMCP client based on transport
            if self.config.transport_type == "stdio":
                client_config = self._create_stdio_config()
            elif self.config.transport_type == "sse":
                client_config = self._create_sse_config()
            else:
                raise MCPError(f"Unsupported transport type: {self.config.transport_type}")
            
            # Create and connect to FastMCP client
            self.client = FastMCPClient(client_config)
            
            # Connect and discover capabilities
            self.connected = True
            await self._discover_capabilities()
            self.logger.info(f"Successfully connected to MCP server: {self.config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server {self.config.name}: {e}")
            raise MCPConnectionError(f"Connection failed: {e}")
    
    def _create_stdio_config(self) -> str:
        """Create FastMCP config for stdio transport."""
        self.logger.info(f"=== CONNECTING TO {self.config.name} VIA STDIO (FastMCP) ===")
        
        if not self.config.command:
            self.logger.error(f"No command specified for stdio transport to {self.config.name}")
            raise MCPError("Command not specified for stdio transport")
            
        # For stdio, FastMCP expects the command as a string or script path
        if self.config.args:
            # Build full command with args
            cmd_with_args = f"{self.config.command} {' '.join(self.config.args)}"
            self.logger.info(f"FastMCP stdio command: {cmd_with_args}")
            return cmd_with_args
        else:
            self.logger.info(f"FastMCP stdio command: {self.config.command}")
            return self.config.command
        
    def _create_sse_config(self) -> str:
        """Create FastMCP config for SSE transport."""
        self.logger.info(f"=== CONNECTING TO {self.config.name} VIA SSE (FastMCP) ===")
        
        if not self.config.url:
            self.logger.error(f"No URL specified for SSE transport to {self.config.name}")
            raise MCPError("URL not specified for SSE transport")
        
        self.logger.info(f"FastMCP SSE URL: {self.config.url}")
        return self.config.url
        
    async def _discover_capabilities(self):
        """Discover tools and resources using FastMCP."""
        try:
            self.logger.info(f"Discovering capabilities for {self.config.name}...")
            
            # Use async context manager for FastMCP operations
            async with self.client as client:
                # Discover tools
                await self._discover_tools(client)
                
                # Discover resources  
                await self._discover_resources(client)
            
        except Exception as e:
            self.logger.error(f"Capability discovery failed: {e}")
            raise MCPError(f"Capability discovery failed: {e}")
    
    async def _discover_tools(self, client):
        """Discover available tools using FastMCP."""
        try:
            self.logger.info(f"Discovering tools from {self.config.name}...")
            
            # Get tools list from FastMCP client
            tools_response = await client.list_tools()
            
            # FastMCP returns Tool objects with .name, .description, .inputSchema attributes
            if tools_response:
                self.logger.info(f"Server {self.config.name} returned {len(tools_response)} tools")
                
                for tool_obj in tools_response:
                    tool_name = tool_obj.name
                    tool_desc = tool_obj.description or "No description"
                    tool_schema = getattr(tool_obj, 'inputSchema', {})
                    
                    self.logger.debug(f"Discovered tool: {tool_name} - {tool_desc}")
                    
                    tool = MCPTool(
                        name=tool_name,
                        description=tool_desc,
                        schema=tool_schema,
                        server_name=self.config.name
                    )
                    self.tools[tool_name] = tool
                    
                self.logger.info(f"Successfully discovered {len(self.tools)} tools from {self.config.name}")
            else:
                self.logger.warning(f"No tools found from {self.config.name}")
                
        except Exception as e:
            self.logger.warning(f"Tool discovery failed for {self.config.name}: {e}")
    
    async def _discover_resources(self, client):
        """Discover available resources using FastMCP."""
        try:
            self.logger.info(f"Discovering resources from {self.config.name}...")
            
            # Try to get resources list from FastMCP client
            try:
                resources_response = await client.list_resources()
                
                if resources_response:
                    for resource_obj in resources_response:
                        resource = MCPResource(
                            uri=resource_obj.uri,
                            name=getattr(resource_obj, 'name', ''),
                            description=getattr(resource_obj, 'description', None),
                            mime_type=getattr(resource_obj, 'mimeType', None),
                            server_name=self.config.name
                        )
                        self.resources[resource.uri] = resource
                        
                    self.logger.info(f"Discovered {len(self.resources)} resources from {self.config.name}")
                else:
                    self.logger.info(f"No resources found from {self.config.name}")
            except AttributeError:
                # Some MCP servers might not support resources
                self.logger.info(f"Server {self.config.name} does not support resources")
                
        except Exception as e:
            self.logger.warning(f"Resource discovery failed for {self.config.name}: {e}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool using FastMCP."""
        if tool_name not in self.tools:
            raise MCPToolError(f"Tool '{tool_name}' not found")
            
        try:
            self.logger.info(f"Calling tool '{tool_name}' on {self.config.name}")
            self.logger.debug(f"Tool arguments: {arguments}")
            
            # Use FastMCP context manager to call the tool
            async with self.client as client:
                result = await client.call_tool(tool_name, arguments)
            
            self.logger.debug(f"Tool '{tool_name}' result: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Tool call failed for '{tool_name}': {e}")
            raise MCPToolError(f"Tool call failed: {e}")
    
    async def get_resource(self, uri: str) -> Any:
        """Get a resource using FastMCP."""
        if uri not in self.resources:
            raise MCPResourceError(f"Resource '{uri}' not found")
            
        try:
            self.logger.info(f"Getting resource '{uri}' from {self.config.name}")
            
            # Use FastMCP context manager to get the resource
            async with self.client as client:
                result = await client.read_resource(uri)
            
            self.logger.debug(f"Resource '{uri}' result: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Resource access failed for '{uri}': {e}")
            raise MCPResourceError(f"Resource access failed: {e}")
    
    async def disconnect(self):
        """Disconnect from the server using FastMCP."""
        self.connected = False
        self.client = None  # FastMCP handles cleanup via context manager

class BinAssistMCPClient:
    """BinAssist MCP client for managing multiple FastMCP server connections."""
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.logger = logging.getLogger("binassist.fastmcp.client")
        self.connections: Dict[str, FastMCPConnection] = {}
        self.all_tools: Dict[str, MCPTool] = {}
        self.all_resources: Dict[str, MCPResource] = {}
        
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all enabled servers using FastMCP."""
        results = {}
        
        for server_config in self.config.get_enabled_servers():
            try:
                self.logger.info(f"Connecting to server: {server_config.name}")
                
                connection = FastMCPConnection(server_config)
                success = await connection.connect()
                
                if success:
                    self.connections[server_config.name] = connection
                    
                    # Add tools to global collection
                    for tool_name, tool in connection.tools.items():
                        # Prefix with server name to avoid conflicts
                        global_tool_name = f"{server_config.name}_{tool_name}"
                        self.all_tools[global_tool_name] = tool
                    
                    # Add resources to global collection
                    for resource_uri, resource in connection.resources.items():
                        self.all_resources[resource_uri] = resource
                
                results[server_config.name] = success
                
            except Exception as e:
                self.logger.error(f"Failed to connect to {server_config.name}: {e}")
                results[server_config.name] = False
        
        self.logger.info(f"Connected to {sum(results.values())} out of {len(results)} servers")
        return results
    
    async def disconnect_all(self):
        """Disconnect from all servers."""
        for connection in self.connections.values():
            await connection.disconnect()
        self.connections.clear()
        self.all_tools.clear()
        self.all_resources.clear()
    
    def get_available_tools(self) -> Dict[str, MCPTool]:
        """Get all available tools from all connected servers."""
        return self.all_tools.copy()
    
    def get_available_resources(self) -> Dict[str, MCPResource]:
        """Get all available resources from all connected servers."""
        return self.all_resources.copy()
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool by name (searches all connected servers)."""
        # Find which server has this tool
        for connection in self.connections.values():
            if tool_name in connection.tools:
                return await connection.call_tool(tool_name, arguments)
        
        # Try with server prefix removed
        for server_name, connection in self.connections.items():
            if tool_name.startswith(f"{server_name}_"):
                actual_tool_name = tool_name[len(f"{server_name}_"):]
                if actual_tool_name in connection.tools:
                    return await connection.call_tool(actual_tool_name, arguments)
        
        raise MCPToolError(f"Tool '{tool_name}' not found in any connected server")
    
    async def get_resource(self, uri: str) -> Any:
        """Get a resource by URI (searches all connected servers)."""
        for connection in self.connections.values():
            if uri in connection.resources:
                return await connection.get_resource(uri)
        
        raise MCPResourceError(f"Resource '{uri}' not found in any connected server")