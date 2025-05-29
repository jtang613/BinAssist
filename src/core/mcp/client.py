"""
MCP Client implementation for connecting to and interacting with MCP servers.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json

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

class MCPConnection:
    """Manages connection to a single MCP server."""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.logger = logging.getLogger(f"binassist.mcp.{config.name}")
        self.connected = False
        self.process = None
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        
    async def connect(self) -> bool:
        """Connect to the MCP server."""
        try:
            self.logger.info(f"Connecting to MCP server: {self.config.name}")
            
            if self.config.transport_type == "stdio":
                await self._connect_stdio()
            elif self.config.transport_type == "sse":
                await self._connect_sse()
            else:
                raise MCPError(f"Unsupported transport type: {self.config.transport_type}")
                
            self.connected = True
            await self._initialize_server()
            self.logger.info(f"Successfully connected to MCP server: {self.config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server {self.config.name}: {e}")
            raise MCPConnectionError(f"Connection failed: {e}")
    
    async def _connect_stdio(self):
        """Connect using stdio transport."""
        self.logger.info(f"=== CONNECTING TO {self.config.name} VIA STDIO ===")
        
        if not self.config.command:
            self.logger.error(f"No command specified for stdio transport to {self.config.name}")
            raise MCPError("Command not specified for stdio transport")
            
        cmd = [self.config.command]
        if self.config.args:
            cmd.extend(self.config.args)
            
        self.logger.info(f"Executing command for {self.config.name}: {' '.join(cmd)}")
        
        env = self.config.env or {}
        if env:
            self.logger.debug(f"Environment variables for {self.config.name}: {env}")
        
        try:
            self.logger.debug(f"Creating subprocess for {self.config.name}...")
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            self.logger.info(f"✓ Successfully created subprocess for {self.config.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create subprocess for {self.config.name}: {e}")
            raise MCPError(f"Failed to start MCP server process: {e}")
        
    async def _connect_sse(self):
        """Connect using SSE transport."""
        self.logger.info(f"=== CONNECTING TO {self.config.name} VIA SSE ===")
        
        if not self.config.url:
            self.logger.error(f"No URL specified for SSE transport to {self.config.name}")
            raise MCPError("URL not specified for SSE transport")
        
        try:
            import httpx
            self.logger.info(f"Connecting to SSE endpoint: {self.config.url}")
            
            # Create HTTP client for SSE connection
            self.http_client = httpx.AsyncClient(timeout=30.0)
            
            self.logger.info(f"SSE HTTP client created for {self.config.name}")
            
        except ImportError:
            self.logger.error("httpx library required for SSE transport")
            raise MCPError("httpx library required for SSE transport")
        except Exception as e:
            self.logger.error(f"SSE connection setup failed to {self.config.url}: {e}")
            raise MCPError(f"SSE connection failed: {e}")
        
    async def _initialize_server(self):
        """Initialize server and discover capabilities."""
        try:
            # Send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": True,
                        "resources": True
                    },
                    "clientInfo": {
                        "name": "BinAssist",
                        "version": "1.0.0"
                    }
                }
            }
            
            response = await self._send_request(init_request)
            self.logger.debug(f"Initialize response: {response}")
            
            # Discover tools
            await self._discover_tools()
            
            # Discover resources  
            await self._discover_resources()
            
        except Exception as e:
            self.logger.error(f"Server initialization failed: {e}")
            raise MCPError(f"Initialization failed: {e}")
    
    async def _discover_tools(self):
        """Discover available tools from the server."""
        try:
            self.logger.info(f"=== DISCOVERING TOOLS FROM SERVER: {self.config.name} ===")
            
            request = {
                "jsonrpc": "2.0", 
                "id": 2,
                "method": "tools/list"
            }
            
            self.logger.info(f"Sending tools/list request to {self.config.name}")
            self.logger.debug(f"Request JSON: {json.dumps(request, indent=2)}")
            
            response = await self._send_request(request)
            
            self.logger.info(f"Received tools/list response from {self.config.name}")
            self.logger.debug(f"Response JSON: {json.dumps(response, indent=2)}")
            
            if "result" in response and "tools" in response["result"]:
                tools_list = response["result"]["tools"]
                self.logger.info(f"Server {self.config.name} returned {len(tools_list)} tools")
                
                for i, tool_info in enumerate(tools_list):
                    tool_name = tool_info.get("name", f"unnamed_tool_{i}")
                    tool_desc = tool_info.get("description", "No description")
                    
                    self.logger.info(f"Processing tool {i+1}: {tool_name}")
                    self.logger.debug(f"Tool description: {tool_desc}")
                    self.logger.debug(f"Tool schema: {json.dumps(tool_info.get('inputSchema', {}), indent=2)}")
                    
                    tool = MCPTool(
                        name=tool_name,
                        description=tool_desc,
                        schema=tool_info.get("inputSchema", {}),
                        server_name=self.config.name
                    )
                    self.tools[tool.name] = tool
                    
                self.logger.info(f"✓ Successfully discovered {len(self.tools)} tools from {self.config.name}")
                
                # Log summary of all discovered tools
                if self.tools:
                    self.logger.info("Tool summary:")
                    for tool_name, tool in self.tools.items():
                        self.logger.info(f"  - {tool_name}: {tool.description}")
                
            else:
                self.logger.warning(f"No tools found in response from {self.config.name}")
                self.logger.debug(f"Response structure: {response}")
                
        except Exception as e:
            self.logger.error(f"Tool discovery failed for {self.config.name}: {e}")
            self.logger.exception("Full traceback for tool discovery error:")
    
    async def _discover_resources(self):
        """Discover available resources from the server."""
        try:
            request = {
                "jsonrpc": "2.0",
                "id": 3, 
                "method": "resources/list"
            }
            
            response = await self._send_request(request)
            
            if "result" in response and "resources" in response["result"]:
                for resource_info in response["result"]["resources"]:
                    resource = MCPResource(
                        uri=resource_info["uri"],
                        name=resource_info.get("name", ""),
                        description=resource_info.get("description"),
                        mime_type=resource_info.get("mimeType"),
                        server_name=self.config.name
                    )
                    self.resources[resource.uri] = resource
                    
                self.logger.info(f"Discovered {len(self.resources)} resources from {self.config.name}")
                
        except Exception as e:
            self.logger.warning(f"Resource discovery failed: {e}")
    
    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request to the server."""
        self.logger.debug(f"=== SENDING REQUEST TO {self.config.name} ===")
        
        if not self.connected:
            self.logger.error(f"Cannot send request to {self.config.name}: not connected")
            raise MCPConnectionError("Not connected to server")
        
        if self.config.transport_type == "stdio":
            return await self._send_stdio_request(request)
        elif self.config.transport_type == "sse":
            return await self._send_sse_request(request)
        else:
            raise MCPError(f"Unsupported transport type: {self.config.transport_type}")
    
    async def _send_stdio_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request via stdio."""
        if not self.process:
            raise MCPConnectionError("No stdio process available")
            
        try:
            request_json = json.dumps(request) + "\n"
            self.logger.debug(f"Sending JSON-RPC request to {self.config.name}:")
            self.logger.debug(f"Request: {json.dumps(request, indent=2)}")
            
            self.process.stdin.write(request_json.encode())
            await self.process.stdin.drain()
            self.logger.debug(f"Request sent to {self.config.name}, waiting for response...")
            
            response_line = await asyncio.wait_for(
                self.process.stdout.readline(),
                timeout=self.config.timeout
            )
            
            if not response_line:
                self.logger.error(f"Empty response from server {self.config.name}")
                raise MCPError("Empty response from server")
                
            response_text = response_line.decode().strip()
            self.logger.debug(f"Raw response from {self.config.name}: {response_text}")
            
            response = json.loads(response_text)
            self.logger.debug(f"Parsed JSON response from {self.config.name}:")
            self.logger.debug(f"Response: {json.dumps(response, indent=2)}")
            
            return response
            
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout waiting for response from {self.config.name}")
            raise MCPTimeoutError(f"Request timeout after {self.config.timeout}s")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response from {self.config.name}: {e}")
            raise MCPError(f"Invalid JSON response: {e}")
        except Exception as e:
            self.logger.error(f"Error sending request to {self.config.name}: {e}")
            raise MCPError(f"Request failed: {e}")
    
    async def _send_sse_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request via SSE/HTTP."""
        if not hasattr(self, 'http_client') or not self.http_client:
            raise MCPConnectionError("No SSE client available")
            
        try:
            self.logger.debug(f"Sending HTTP POST request to {self.config.name}:")
            self.logger.debug(f"Request: {json.dumps(request, indent=2)}")
            
            # Try common MCP SSE endpoints
            endpoints_to_try = ['/message', '/rpc', '']  # Last one is just the base URL
            last_error = None
            
            for endpoint_path in endpoints_to_try:
                endpoint_url = self.config.url.rstrip('/') + endpoint_path
                self.logger.debug(f"Trying POST request to: {endpoint_url}")
                
                try:
                    response = await self.http_client.post(
                        endpoint_url,
                        json=request,
                        headers={"Content-Type": "application/json"},
                        timeout=self.config.timeout
                    )
                    
                    response.raise_for_status()
                    response_data = response.json()
                    
                    self.logger.info(f"✅ Successfully connected to {endpoint_url}")
                    self.logger.debug(f"Parsed JSON response from {self.config.name}:")
                    self.logger.debug(f"Response: {json.dumps(response_data, indent=2)}")
                    
                    return response_data
                    
                except Exception as e:
                    last_error = e
                    self.logger.debug(f"Failed to connect to {endpoint_url}: {e}")
                    continue
            
            # If we get here, all endpoints failed
            tried_urls = [self.config.url.rstrip('/') + path for path in endpoints_to_try]
            raise Exception(f"All endpoints failed. Tried: {tried_urls}. Last error: {last_error}")
            
        except Exception as e:
            self.logger.error(f"Error sending SSE request to {self.config.name}: {e}")
            raise MCPError(f"SSE request failed: {e}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the server."""
        if tool_name not in self.tools:
            raise MCPToolError(f"Tool '{tool_name}' not found")
            
        try:
            request = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call", 
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            response = await self._send_request(request)
            
            if "result" in response:
                return response["result"]
            else:
                raise MCPToolError(f"No result in tool response: {response}")
                
        except Exception as e:
            self.logger.error(f"Tool call failed: {e}")
            raise MCPToolError(f"Tool call failed: {e}")
    
    async def get_resource(self, uri: str) -> Any:
        """Get a resource from the server."""
        if uri not in self.resources:
            raise MCPResourceError(f"Resource '{uri}' not found")
            
        try:
            request = {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "resources/read",
                "params": {
                    "uri": uri
                }
            }
            
            response = await self._send_request(request)
            
            if "result" in response:
                return response["result"]
            else:
                raise MCPResourceError(f"No result in resource response: {response}")
                
        except Exception as e:
            self.logger.error(f"Resource access failed: {e}")
            raise MCPResourceError(f"Resource access failed: {e}")
    
    async def disconnect(self):
        """Disconnect from the server."""
        if self.connected:
            try:
                if self.config.transport_type == "stdio" and self.process:
                    self.process.terminate()
                    await self.process.wait()
                elif self.config.transport_type == "sse" and hasattr(self, 'http_client'):
                    await self.http_client.aclose()
                    self.http_client = None
            except Exception as e:
                self.logger.warning(f"Error during disconnect: {e}")
            finally:
                self.connected = False
                self.process = None

class MCPClient:
    """Main MCP client for managing multiple server connections."""
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.logger = logging.getLogger("binassist.mcp.client")
        self.connections: Dict[str, MCPConnection] = {}
        self.all_tools: Dict[str, MCPTool] = {}
        self.all_resources: Dict[str, MCPResource] = {}
        
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all enabled servers."""
        results = {}
        
        for server_config in self.config.get_enabled_servers():
            try:
                connection = MCPConnection(server_config)
                success = await connection.connect()
                
                if success:
                    self.connections[server_config.name] = connection
                    self._merge_tools(connection.tools)
                    self._merge_resources(connection.resources)
                    
                results[server_config.name] = success
                
            except Exception as e:
                self.logger.error(f"Failed to connect to {server_config.name}: {e}")
                results[server_config.name] = False
                
        self.logger.info(f"Connected to {len(self.connections)} MCP servers")
        return results
    
    def _merge_tools(self, tools: Dict[str, MCPTool]):
        """Merge tools from a connection into the global tool registry."""
        for name, tool in tools.items():
            # Handle name conflicts by prefixing with server name
            if name in self.all_tools and self.all_tools[name].server_name != tool.server_name:
                prefixed_name = f"{tool.server_name}.{name}"
                self.all_tools[prefixed_name] = tool
            else:
                self.all_tools[name] = tool
    
    def _merge_resources(self, resources: Dict[str, MCPResource]):
        """Merge resources from a connection into the global resource registry."""
        self.all_resources.update(resources)
    
    def get_available_tools(self) -> Dict[str, MCPTool]:
        """Get all available tools across all connected servers."""
        return self.all_tools.copy()
    
    def get_available_resources(self) -> Dict[str, MCPResource]:
        """Get all available resources across all connected servers.""" 
        return self.all_resources.copy()
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the appropriate server."""
        if tool_name not in self.all_tools:
            raise MCPToolError(f"Tool '{tool_name}' not available")
            
        tool = self.all_tools[tool_name]
        server_name = tool.server_name
        
        if server_name not in self.connections:
            raise MCPConnectionError(f"Not connected to server '{server_name}'")
            
        connection = self.connections[server_name]
        return await connection.call_tool(tool_name, arguments)
    
    async def get_resource(self, uri: str) -> Any:
        """Get a resource from the appropriate server."""
        if uri not in self.all_resources:
            raise MCPResourceError(f"Resource '{uri}' not available")
            
        resource = self.all_resources[uri]
        server_name = resource.server_name
        
        if server_name not in self.connections:
            raise MCPConnectionError(f"Not connected to server '{server_name}'")
            
        connection = self.connections[server_name]
        return await connection.get_resource(uri)
    
    async def disconnect_all(self):
        """Disconnect from all servers."""
        for connection in self.connections.values():
            await connection.disconnect()
        self.connections.clear()
        self.all_tools.clear()
        self.all_resources.clear()