"""
MCP Client implementation for connecting to and interacting with MCP servers.
Uses the official MCP Python SDK.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json
from binaryninja import log

from .config import MCPConfig, MCPServerConfig
from .exceptions import MCPError, MCPConnectionError, MCPTimeoutError, MCPToolError, MCPResourceError

try:
    from mcp import ClientSession, StdioServerParameters, types
    from mcp.client.stdio import stdio_client
    # Check if SSE client is available in the SDK
    try:
        from mcp.client.sse import sse_client
        SSE_CLIENT_AVAILABLE = True
    except ImportError:
        sse_client = None
        SSE_CLIENT_AVAILABLE = False
    MCP_SDK_AVAILABLE = True
except ImportError:
    MCP_SDK_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    sse_client = None
    SSE_CLIENT_AVAILABLE = False

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
    """Manages connection to a single MCP server using official MCP SDK."""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.connected = False
        self.session: Optional[ClientSession] = None
        self.read = None
        self.write = None
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        
    async def connect(self) -> bool:
        """Connect to the MCP server using official SDK."""
        if not MCP_SDK_AVAILABLE:
            raise MCPError("MCP SDK not available. Please install with: pip install mcp")
            
        try:
            log.log_info(f"[BinAssist] Connecting to MCP server: {self.config.name}")
            
            if self.config.transport_type == "stdio":
                await self._connect_stdio_sdk()
            elif self.config.transport_type == "sse":
                await self._connect_sse_sdk()
            else:
                raise MCPError(f"Unsupported transport type: {self.config.transport_type}")
                
            self.connected = True
            await self._discover_capabilities()
            log.log_info(f"[BinAssist] Successfully connected to MCP server: {self.config.name}")
            return True
            
        except Exception as e:
            log.log_error(f"[BinAssist] Failed to connect to MCP server {self.config.name}: {e}")
            raise MCPConnectionError(f"Connection failed: {e}")
    
    async def _connect_stdio_sdk(self):
        """Connect using stdio transport with official MCP SDK."""
        log.log_info(f"[BinAssist] === CONNECTING TO {self.config.name} VIA STDIO SDK ===")
        
        if not self.config.command:
            log.log_error(f"[BinAssist] No command specified for stdio transport to {self.config.name}")
            raise MCPError("Command not specified for stdio transport")
            
        try:
            # Create server parameters using official SDK
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args or [],
                env=self.config.env or None
            )
            
            log.log_info(f"[BinAssist] Creating MCP stdio client for {self.config.name}")
            log.log_debug(f"[BinAssist] Command: {self.config.command}")
            log.log_debug(f"[BinAssist] Args: {self.config.args}")
            
            # Create the stdio client connection
            self.read, self.write = await stdio_client(server_params).__aenter__()
            
            # Create client session
            self.session = ClientSession(self.read, self.write)
            await self.session.__aenter__()
            
            # Initialize the connection
            log.log_info(f"[BinAssist] Initializing MCP session for {self.config.name}")
            await self.session.initialize()
            
            log.log_info(f"[BinAssist] ✓ Successfully connected to MCP server {self.config.name} via SDK")
            
        except Exception as e:
            log.log_error(f"[BinAssist] Failed to connect to MCP server {self.config.name}: {e}")
            raise MCPError(f"Failed to start MCP server: {e}")
    
    async def _connect_sse_sdk(self):
        """Connect using SSE transport with official MCP SDK."""
        log.log_info(f"[BinAssist] === CONNECTING TO {self.config.name} VIA SSE SDK ===")
        
        if not self.config.url:
            log.log_error(f"[BinAssist] No URL specified for SSE transport to {self.config.name}")
            raise MCPError("URL not specified for SSE transport")
        
        # Check if SSE client is available in the SDK
        if not SSE_CLIENT_AVAILABLE:
            log.log_warn("[BinAssist] SSE client not available in MCP SDK, trying manual implementation")
            # Fall back to manual SSE implementation if SDK doesn't have it
            await self._connect_sse_manual()
            return
            
        try:
            log.log_info(f"[BinAssist] Creating MCP SSE client for {self.config.name}")
            log.log_debug(f"[BinAssist] URL: {self.config.url}")
            
            # Use the proper async with pattern like the official example
            log.log_info("[BinAssist] Connecting to SSE server using official SDK pattern")
            
            # Create SSE client connection
            self.sse_client_context = sse_client(self.config.url, timeout=30)
            self.read, self.write = await self.sse_client_context.__aenter__()
            log.log_info("[BinAssist] SSE client connection established")
            
            # Create and initialize session
            self.session = ClientSession(self.read, self.write)
            await self.session.__aenter__()
            log.log_info("[BinAssist] ClientSession context entered")
            
            # Initialize the connection with timeout
            log.log_info(f"[BinAssist] Initializing MCP SSE session for {self.config.name}")
            try:
                await asyncio.wait_for(self.session.initialize(), timeout=30.0)
                log.log_info(f"[BinAssist] MCP session initialization completed for {self.config.name}")
            except asyncio.TimeoutError:
                log.log_error(f"[BinAssist] MCP session initialization timed out after 30 seconds for {self.config.name}")
                raise MCPError(f"Session initialization timed out - server may not be responding to MCP handshake")
            except Exception as init_error:
                log.log_error(f"[BinAssist] MCP session initialization failed for {self.config.name}: {init_error}")
                raise MCPError(f"Session initialization failed: {init_error}")
            
            log.log_info(f"[BinAssist] ✓ Successfully connected to MCP SSE server {self.config.name} via SDK")
            
        except Exception as e:
            log.log_error(f"[BinAssist] Failed to connect to MCP SSE server {self.config.name}: {e}")
            raise MCPError(f"Failed to connect to SSE server: {e}")
        
    async def _connect_sse_manual(self):
        """Manual SSE connection implementation when SDK doesn't support it."""
        log.log_info(f"[BinAssist] Using manual SSE implementation for {self.config.name}")
        
        try:
            import httpx
            
            # Create HTTP client
            self.http_client = httpx.AsyncClient(timeout=30.0)
            
            # Test basic connectivity
            test_url = self.config.url.rstrip('/')
            log.log_info(f"[BinAssist] Testing SSE server connectivity: {test_url}")
            
            # Try to connect to the server and see what it responds with
            try:
                response = await self.http_client.get(test_url, timeout=10.0)
                log.log_debug(f"[BinAssist] Base response status: {response.status_code}")
                log.log_debug(f"[BinAssist] Base response headers: {dict(response.headers)}")
                log.log_debug(f"[BinAssist] Base response text: {response.text[:200]}...")
                
                # For now, let's simulate successful connection and manual tool discovery
                # This is a temporary workaround until we have proper SSE SDK support
                self.connected = True
                
                # We'll need to implement manual JSON-RPC over HTTP for SSE servers
                # that don't have SDK support yet
                log.log_info(f"[BinAssist] Manual SSE connection established to {self.config.name}")
                
            except Exception as test_error:
                log.log_error(f"[BinAssist] SSE server connectivity test failed: {test_error}")
                raise MCPError(f"Cannot connect to SSE server: {test_error}")
                
        except ImportError:
            raise MCPError("httpx library required for SSE transport")
        except Exception as e:
            raise MCPError(f"SSE connection failed: {e}")
        
    async def _connect_sse(self):
        """Connect using SSE transport."""
        log.log_info(f"[BinAssist] === CONNECTING TO {self.config.name} VIA SSE ===")
        
        if not self.config.url:
            log.log_error(f"[BinAssist] No URL specified for SSE transport to {self.config.name}")
            raise MCPError("URL not specified for SSE transport")
        
        try:
            import httpx
            log.log_info(f"[BinAssist] Connecting to SSE endpoint: {self.config.url}")
            
            # Create HTTP client for SSE connection
            self.http_client = httpx.AsyncClient(timeout=30.0)
            
            # Test basic connectivity to the MCP server
            try:
                test_url = self.config.url.rstrip('/')
                log.log_debug(f"[BinAssist] Testing basic connectivity to: {test_url}")
                
                # Try a simple GET request to see what endpoints are available
                test_response = await self.http_client.get(test_url, timeout=5.0)
                log.log_debug(f"[BinAssist] Base URL response status: {test_response.status_code}")
                log.log_debug(f"[BinAssist] Base URL response headers: {dict(test_response.headers)}")
                
                if test_response.status_code == 404:
                    log.log_warn(f"[BinAssist] Base URL returned 404, server might be on a different path")
                
            except Exception as test_error:
                log.log_warn(f"[BinAssist] Base URL test failed: {test_error}")
            
            log.log_info(f"[BinAssist] SSE HTTP client created for {self.config.name}")
            
        except ImportError:
            log.log_error(f"[BinAssist] httpx library required for SSE transport")
            raise MCPError("httpx library required for SSE transport")
        except Exception as e:
            log.log_error(f"[BinAssist] SSE connection setup failed to {self.config.url}: {e}")
            raise MCPError(f"SSE connection failed: {e}")
        
    async def _discover_capabilities(self):
        """Discover server capabilities using official SDK or manual methods."""
        try:
            log.log_info(f"[BinAssist] === DISCOVERING CAPABILITIES FROM SERVER: {self.config.name} ===")
            
            if self.session:
                # Use SDK for capability discovery
                await self._discover_tools_sdk()
                await self._discover_resources_sdk()
            elif hasattr(self, 'http_client'):
                # Use manual methods for SSE servers without SDK support
                await self._discover_tools_manual()
                await self._discover_resources_manual()
            else:
                raise MCPError("No session or http client available for capability discovery")
            
        except Exception as e:
            log.log_error(f"[BinAssist] Capability discovery failed: {e}")
            raise MCPError(f"Capability discovery failed: {e}")
    
    async def _discover_tools_sdk(self):
        """Discover available tools using official SDK."""
        try:
            log.log_info(f"[BinAssist] Discovering tools from {self.config.name}")
            
            # Use SDK to list tools
            tools_result = await self.session.list_tools()
            
            log.log_info(f"[BinAssist] Server {self.config.name} returned {len(tools_result.tools)} tools")
            
            for tool_info in tools_result.tools:
                tool_name = tool_info.name
                tool_desc = tool_info.description or "No description"
                
                log.log_info(f"[BinAssist] Found tool: {tool_name}")
                log.log_debug(f"[BinAssist] Tool description: {tool_desc}")
                log.log_debug(f"[BinAssist] Tool schema: {tool_info.inputSchema}")
                
                # Handle both Pydantic models and dicts for inputSchema
                if tool_info.inputSchema:
                    if hasattr(tool_info.inputSchema, 'model_dump'):
                        # It's a Pydantic model
                        schema = tool_info.inputSchema.model_dump()
                    else:
                        # It's already a dict
                        schema = tool_info.inputSchema
                else:
                    schema = {}
                
                tool = MCPTool(
                    name=tool_name,
                    description=tool_desc,
                    schema=schema,
                    server_name=self.config.name
                )
                self.tools[tool.name] = tool
                
            log.log_info(f"[BinAssist] ✓ Successfully discovered {len(self.tools)} tools from {self.config.name}")
            
            # Log summary of all discovered tools
            if self.tools:
                log.log_info(f"[BinAssist] Tool summary:")
                for tool_name, tool in self.tools.items():
                    log.log_info(f"[BinAssist]   - {tool_name}: {tool.description}")
                
        except Exception as e:
            log.log_error(f"[BinAssist] Tool discovery failed for {self.config.name}: {e}")
            log.log_error(f"[BinAssist] Full traceback for tool discovery error:")
    
    async def _discover_resources_sdk(self):
        """Discover available resources using official SDK."""
        try:
            log.log_info(f"[BinAssist] Discovering resources from {self.config.name}")
            
            # Use SDK to list resources
            resources_result = await self.session.list_resources()
            
            log.log_info(f"[BinAssist] Server {self.config.name} returned {len(resources_result.resources)} resources")
            
            for resource_info in resources_result.resources:
                resource = MCPResource(
                    uri=resource_info.uri,
                    name=resource_info.name or "",
                    description=resource_info.description,
                    mime_type=resource_info.mimeType,
                    server_name=self.config.name
                )
                self.resources[resource.uri] = resource
                
            log.log_info(f"[BinAssist] ✓ Successfully discovered {len(self.resources)} resources from {self.config.name}")
                
        except Exception as e:
            log.log_warn(f"[BinAssist] Resource discovery failed: {e}")
    
    async def _discover_tools_manual(self):
        """Discover tools manually for SSE servers without SDK support."""
        try:
            log.log_info(f"[BinAssist] Manual tool discovery for {self.config.name}")
            
            # Try to send a JSON-RPC request to list tools
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            }
            
            response = await self._send_manual_request(request)
            
            if response and "result" in response and "tools" in response["result"]:
                tools_list = response["result"]["tools"]
                log.log_info(f"[BinAssist] Manual discovery found {len(tools_list)} tools")
                
                for tool_info in tools_list:
                    tool_name = tool_info.get("name", "unnamed_tool")
                    tool_desc = tool_info.get("description", "No description")
                    
                    tool = MCPTool(
                        name=tool_name,
                        description=tool_desc,
                        schema=tool_info.get("inputSchema", {}),
                        server_name=self.config.name
                    )
                    self.tools[tool.name] = tool
                    
                log.log_info(f"[BinAssist] ✓ Manual discovery found {len(self.tools)} tools")
            else:
                log.log_warn(f"[BinAssist] No tools found in manual discovery response")
                
        except Exception as e:
            log.log_error(f"[BinAssist] Manual tool discovery failed: {e}")
    
    async def _discover_resources_manual(self):
        """Discover resources manually for SSE servers without SDK support."""
        try:
            log.log_info(f"[BinAssist] Manual resource discovery for {self.config.name}")
            
            # Try to send a JSON-RPC request to list resources
            request = {
                "jsonrpc": "2.0", 
                "id": 2,
                "method": "resources/list"
            }
            
            response = await self._send_manual_request(request)
            
            if response and "result" in response and "resources" in response["result"]:
                resources_list = response["result"]["resources"]
                log.log_info(f"[BinAssist] Manual discovery found {len(resources_list)} resources")
                
                for resource_info in resources_list:
                    resource = MCPResource(
                        uri=resource_info["uri"],
                        name=resource_info.get("name", ""),
                        description=resource_info.get("description"),
                        mime_type=resource_info.get("mimeType"),
                        server_name=self.config.name
                    )
                    self.resources[resource.uri] = resource
                    
                log.log_info(f"[BinAssist] ✓ Manual discovery found {len(self.resources)} resources")
            else:
                log.log_warn(f"[BinAssist] No resources found in manual discovery response")
                
        except Exception as e:
            log.log_warn(f"[BinAssist] Manual resource discovery failed: {e}")
    
    async def _send_manual_request(self, request: dict) -> dict:
        """Send a manual JSON-RPC request for SSE servers."""
        if not hasattr(self, 'http_client'):
            raise MCPError("No HTTP client available for manual requests")
            
        try:
            # Try common MCP endpoints
            endpoints = ['/sse', '/message', '/rpc', '']
            
            for endpoint in endpoints:
                url = self.config.url.rstrip('/') + endpoint
                log.log_debug(f"[BinAssist] Trying manual request to: {url}")
                
                try:
                    response = await self.http_client.post(
                        url,
                        json=request,
                        headers={"Content-Type": "application/json"},
                        timeout=self.config.timeout
                    )
                    
                    if response.status_code == 200:
                        log.log_debug(f"[BinAssist] Manual request successful to: {url}")
                        return response.json()
                    elif response.status_code == 404:
                        continue  # Try next endpoint
                    else:
                        log.log_debug(f"[BinAssist] Manual request failed with status {response.status_code}")
                        
                except Exception as endpoint_error:
                    log.log_debug(f"[BinAssist] Manual request to {url} failed: {endpoint_error}")
                    continue
            
            log.log_warn(f"[BinAssist] All manual request endpoints failed")
            return {}
            
        except Exception as e:
            log.log_error(f"[BinAssist] Manual request failed: {e}")
            return {}
    
    # Note: Old request methods removed since we now use the official MCP SDK
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the server using SDK."""
        if tool_name not in self.tools:
            raise MCPToolError(f"Tool '{tool_name}' not found")
            
        if not self.session:
            raise MCPToolError("No active session for tool calling")
            
        try:
            log.log_info(f"[BinAssist] Calling tool '{tool_name}' with arguments: {arguments}")
            
            # Use SDK to call tool
            result = await self.session.call_tool(tool_name, arguments)
            
            log.log_debug(f"[BinAssist] Tool call result: {result}")
            return result
                
        except Exception as e:
            log.log_error(f"[BinAssist] Tool call failed: {e}")
            raise MCPToolError(f"Tool call failed: {e}")
    
    async def get_resource(self, uri: str) -> Any:
        """Get a resource from the server using SDK."""
        if uri not in self.resources:
            raise MCPResourceError(f"Resource '{uri}' not found")
            
        if not self.session:
            raise MCPResourceError("No active session for resource access")
            
        try:
            log.log_info(f"[BinAssist] Reading resource: {uri}")
            
            # Use SDK to read resource
            content, mime_type = await self.session.read_resource(uri)
            
            return {"content": content, "mime_type": mime_type}
                
        except Exception as e:
            log.log_error(f"[BinAssist] Resource access failed: {e}")
            raise MCPResourceError(f"Resource access failed: {e}")
    
    async def disconnect(self):
        """Disconnect from the server using SDK."""
        if self.connected:
            try:
                if self.session:
                    await self.session.__aexit__(None, None, None)
                    self.session = None
                    
                # Clean up SSE client context if it exists
                if hasattr(self, 'sse_client_context') and self.sse_client_context:
                    await self.sse_client_context.__aexit__(None, None, None)
                    self.sse_client_context = None
                    
                if self.read and self.write:
                    # The stdio_client context manager should handle cleanup
                    # when we exit the session
                    self.read = None 
                    self.write = None
                    
            except Exception as e:
                log.log_warn(f"[BinAssist] Error during disconnect: {e}")
            finally:
                self.connected = False

class MCPClient:
    """Main MCP client for managing multiple server connections."""
    
    def __init__(self, config: MCPConfig):
        self.config = config
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
                log.log_error(f"[BinAssist] Failed to connect to {server_config.name}: {e}")
                results[server_config.name] = False
                
        log.log_info(f"[BinAssist] Connected to {len(self.connections)} MCP servers")
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