"""
Unified MCP Service for BinAssist.

This service provides a clean interface for MCP integration while preserving
all working test functionality. Designed for Custom Query integration.
"""

import asyncio
import concurrent.futures
from typing import Dict, List, Optional, Any
from binaryninja.settings import Settings
from binaryninja import log

from ..services.base_service import BaseService
from .config import MCPConfig, MCPServerConfig
from .models import MCPTool, MCPResource, MCPConnectionInfo, MCPConnectionStatus, MCPTestResult
from .client import MCPConnection  # PRESERVE: Use working test implementation
from .exceptions import MCPError, MCPConnectionError, MCPToolError, MCPResourceError


class MCPService(BaseService):
    """
    Unified MCP service providing clean interface for:
    - Custom Query integration
    - Plugin settings management  
    - Tool discovery and execution
    - PRESERVED: Complete test functionality
    """
    
    def __init__(self, settings: Settings):
        super().__init__("mcp")
        self.settings = settings
        self.connections: Dict[str, MCPConnectionInfo] = {}
        self._initialized = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
    # ===================================================================
    # PRESERVED: Complete Test Functionality (ZERO CHANGES)
    # ===================================================================
    
    def test_connection_sync(self, server_config: MCPServerConfig) -> Dict[str, Any]:
        """
        PRESERVED: Synchronous wrapper for testing server connection.
        Returns EXACT same format as current working implementation.
        
        This is the critical method that powers the UI test functionality.
        """
        log.log_info(f"[BinAssist] Starting test_connection_sync for {server_config.name}")
        try:
            # PRESERVED: Use same thread executor pattern that works
            try:
                # Try to get the current loop
                current_loop = asyncio.get_running_loop()
                log.log_info("[BinAssist] Found running event loop, will use thread executor")
                # If we're here, a loop is running, so we need to run in a new thread
                import concurrent.futures
                import threading
                
                def run_test():
                    # Create new event loop for this thread
                    log.log_info("[BinAssist] Creating new event loop in thread")
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self.test_server_connection(server_config))
                        log.log_info(f"[BinAssist] Thread test completed with result: {result}")
                        return result
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_test)
                    result = future.result(timeout=60)  # 60 second timeout for test
                    log.log_info(f"[BinAssist] Thread executor completed with result: {result}")
                    return result
                    
            except RuntimeError:
                # No loop running, we can run directly
                log.log_info("[BinAssist] No running event loop found, creating new one")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.test_server_connection(server_config))
                    log.log_info(f"[BinAssist] Direct loop test completed with result: {result}")
                    return result
                finally:
                    loop.close()
                    
        except Exception as e:
            log.log_error(f"[BinAssist] Exception in test_connection_sync: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_server_connection(self, server_config: MCPServerConfig) -> Dict[str, Any]:
        """
        PRESERVED: Test connection to a single MCP server and return tools list.
        Uses EXACT same implementation that works.
        """
        try:
            log.log_info(f"[BinAssist] Testing connection to MCP server: {server_config.name}")
            log.log_info(f"[BinAssist] Server config: transport={server_config.transport_type}, command={server_config.command}, url={server_config.url}")
            
            # PRESERVED: Create a temporary connection for testing using WORKING code
            log.log_info("[BinAssist] Creating temporary MCPConnection for testing")
            test_connection = MCPConnection(server_config)
            
            # PRESERVED: Try to connect using WORKING code
            log.log_info("[BinAssist] Attempting to connect to MCP server")
            success = await test_connection.connect()
            log.log_info(f"[BinAssist] Connection attempt result: {success}")
            
            if success:
                # PRESERVED: Get tools list using WORKING code
                log.log_info("[BinAssist] Successfully connected, getting tools and resources")
                tools = list(test_connection.tools.values())
                resources = list(test_connection.resources.values())
                log.log_info(f"[BinAssist] Found {len(tools)} tools and {len(resources)} resources")
                
                # PRESERVED: Disconnect the test connection
                log.log_info("[BinAssist] Disconnecting test connection")
                await test_connection.disconnect()
                
                # PRESERVED: Return EXACT same format as current working implementation
                result = {
                    "success": True,
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "server": tool.server_name
                        } for tool in tools
                    ],
                    "resources": [
                        {
                            "uri": resource.uri,
                            "name": resource.name,
                            "description": resource.description
                        } for resource in resources
                    ],
                    "tools_count": len(tools),
                    "resources_count": len(resources)
                }
                log.log_info(f"[BinAssist] Returning success result: {result}")
                return result
            else:
                log.log_warn("[BinAssist] Connection failed")
                return {
                    "success": False,
                    "error": "Failed to connect to server"
                }
                
        except Exception as e:
            log.log_error(f"[BinAssist] Error testing MCP server {server_config.name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # ===================================================================
    # NEW: Settings Integration (Clean Interface)
    # ===================================================================
    
    def load_from_settings(self) -> bool:
        """Load MCP configuration from plugin settings."""
        try:
            log.log_info("[BinAssist] Loading MCP configuration from settings")
            
            # Load servers from settings (same format as current code)
            servers_data = self.settings.get_json('mcp_servers', [])
            log.log_info(f"[BinAssist] Loaded {len(servers_data)} MCP servers from settings")
            
            # Convert to server configs
            server_configs = []
            for server_data in servers_data:
                try:
                    config = MCPServerConfig.from_dict(server_data)
                    server_configs.append(config)
                except Exception as e:
                    log.log_error(f"[BinAssist] Failed to load MCP server config: {e}")
                    continue
            
            # Create MCP config
            self.config = MCPConfig(servers=server_configs)
            self._initialized = True
            
            log.log_info(f"[BinAssist] Successfully loaded {len(server_configs)} MCP servers")
            return True
            
        except Exception as e:
            log.log_error(f"[BinAssist] Failed to load MCP settings: {e}")
            return False
    
    def save_to_settings(self) -> None:
        """Save current configuration to plugin settings."""
        try:
            if hasattr(self, 'config') and self.config:
                servers_data = [server.to_dict() for server in self.config.servers]
                self.settings.set_json('mcp_servers', servers_data)
                log.log_info(f"[BinAssist] Saved {len(servers_data)} MCP servers to settings")
            else:
                log.log_warn("[BinAssist] No MCP config to save")
        except Exception as e:
            log.log_error(f"[BinAssist] Failed to save MCP settings: {e}")
    
    # ===================================================================
    # NEW: Custom Query Interface (Clean API)
    # ===================================================================
    
    def get_available_servers(self) -> List[str]:
        """Get list of available server names."""
        if not hasattr(self, 'config') or not self.config:
            return []
        
        return [server.name for server in self.config.get_enabled_servers()]
    
    def get_server_status(self, server_name: str) -> MCPConnectionStatus:
        """Get connection status for specific server."""
        if server_name in self.connections:
            return self.connections[server_name].status
        return MCPConnectionStatus.DISCONNECTED
    
    def get_available_tools(self, server_filter: Optional[List[str]] = None) -> List[MCPTool]:
        """
        Get all available tools, optionally filtered by servers.
        
        Args:
            server_filter: Optional list of server names to include
            
        Returns:
            List of available MCP tools
        """
        tools = []
        
        for connection_info in self.connections.values():
            if not connection_info.is_connected:
                continue
                
            # Apply server filter if provided
            if server_filter and connection_info.server_config.name not in server_filter:
                continue
                
            tools.extend(connection_info.tools.values())
        
        return tools
    
    def get_tools_for_llm(self, server_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get tools formatted for LLM in OpenAI format.
        
        Args:
            server_filter: Optional list of server names to include
            
        Returns:
            List of tool definitions in OpenAI format
        """
        tools = self.get_available_tools(server_filter)
        return [tool.to_llm_format() for tool in tools]
    
    def execute_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Synchronous tool execution for Custom Query integration.
        
        Args:
            tool_name: Name of the tool (may have mcp_ prefix)
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        # Strip mcp_ prefix if present
        actual_tool_name = tool_name.replace("mcp_", "", 1) if tool_name.startswith("mcp_") else tool_name
        
        # Find tool and execute
        for connection_info in self.connections.values():
            if not connection_info.is_connected:
                continue
                
            if actual_tool_name in connection_info.tools:
                # TODO: Implement async tool execution with sync wrapper
                # For now, raise not implemented
                raise NotImplementedError("Tool execution will be implemented in next phase")
        
        raise MCPToolError(f"Tool '{actual_tool_name}' not found in any connected server")
    
    # ===================================================================
    # NEW: Connection Management (Future Implementation)
    # ===================================================================
    
    async def connect_all_servers(self) -> Dict[str, bool]:
        """Connect to all enabled servers."""
        # TODO: Implement full connection management
        # For now, focus on preserving test functionality
        log.log_info("[BinAssist] Full connection management not yet implemented")
        return {}
    
    async def disconnect_all_servers(self) -> None:
        """Disconnect from all servers."""
        # TODO: Implement disconnection
        log.log_info("[BinAssist] Disconnection not yet implemented")
        pass