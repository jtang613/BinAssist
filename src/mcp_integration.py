"""
MCP Integration for BinAssist Plugin.

This module provides integration between the BinAssist plugin and MCP servers,
handling tool discovery, execution, and logging.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .core.mcp.fastmcp_client import BinAssistMCPClient, FASTMCP_AVAILABLE
from .core.mcp.config import MCPConfig, MCPServerConfig
from .core.mcp.exceptions import MCPError


class McpIntegrationService:
    """
    Service for integrating MCP with the BinAssist plugin.
    
    This service handles:
    - MCP server connections based on plugin settings
    - Tool discovery and formatting for LLM
    - Tool execution and result handling
    - Session logging of MCP interactions
    """
    
    def __init__(self, settings):
        """Initialize the MCP integration service."""
        self.settings = settings
        self.logger = logging.getLogger("binassist.mcp_integration")
        self.mcp_client: Optional[MCPClient] = None
        self.session_log: List[Dict[str, Any]] = []
        self.available_tools: Dict[str, Any] = {}
        self.available_resources: Dict[str, Any] = {}
        self._initialized = False
        
    def initialize_from_plugin_settings(self, plugin_servers: List[Dict[str, Any]]) -> bool:
        """
        Initialize MCP client from plugin server configurations.
        
        Args:
            plugin_servers: List of server configs from plugin settings
            
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("=== INITIALIZING MCP INTEGRATION ===")
            self.logger.info(f"Plugin servers provided: {len(plugin_servers) if plugin_servers else 0}")
            
            if not plugin_servers:
                self.logger.info("No MCP servers configured")
                return True
                
            # Convert plugin server configs to MCP server configs
            server_configs = []
            enabled_count = 0
            
            for i, server in enumerate(plugin_servers):
                self.logger.debug(f"Processing server {i+1}: {server}")
                
                if server.get('enabled', False):
                    enabled_count += 1
                    args = []
                    if server.get('args'):
                        args = server['args'].split() if isinstance(server['args'], str) else server['args']
                    
                    self.logger.info(f"Adding enabled server: {server['name']} ({server['transport']})")
                    
                    mcp_server = MCPServerConfig(
                        name=server['name'],
                        transport_type=server['transport'],
                        command=server.get('command') if server['transport'] == 'stdio' else None,
                        args=args if server['transport'] == 'stdio' else None,
                        url=server.get('url') if server['transport'] == 'sse' else None,
                        timeout=30,
                        enabled=True
                    )
                    server_configs.append(mcp_server)
                else:
                    self.logger.debug(f"Skipping disabled server: {server.get('name', 'unnamed')}")
            
            self.logger.info(f"Found {enabled_count} enabled servers out of {len(plugin_servers)} total")
            
            if not server_configs:
                self.logger.info("No enabled MCP servers found")
                return True
                
            # Create MCP configuration
            self.logger.info("Creating MCP configuration...")
            config = MCPConfig(
                servers=server_configs,
                global_timeout=60,
                max_concurrent_connections=5,
                retry_attempts=2
            )
            
            # Check FastMCP availability
            if not FASTMCP_AVAILABLE:
                self.logger.error("FastMCP library not available. Install with: pip install fastmcp")
                return False
            
            # Create BinAssist MCP client (using FastMCP)
            self.logger.info("Creating BinAssist MCP client (FastMCP-based)...")
            self.mcp_client = BinAssistMCPClient(config)
            self.logger.info("BinAssist MCP client created successfully")
            
            # Connect to servers asynchronously
            self.logger.info("Starting async initialization...")
            self._run_async_init()
            
            self._initialized = True
            self.logger.info(f"MCP integration initialized with {len(server_configs)} servers")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP integration: {e}")
            self.logger.exception("Full traceback for MCP initialization error:")
            return False
    
    def _run_async_init(self):
        """Run async initialization using existing event loop."""
        try:
            self.logger.info("=== STARTING ASYNC MCP INITIALIZATION ===")
            
            # Try to get the existing event loop
            try:
                loop = asyncio.get_running_loop()
                self.logger.info("Using existing event loop for MCP initialization")
                # Schedule the async work in the existing loop
                import concurrent.futures
                import threading
                
                # Run async code in a separate thread to avoid blocking
                def run_in_thread():
                    # Create a new event loop in this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self._async_init_work())
                    finally:
                        new_loop.close()
                
                # Run in a separate thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    results = future.result(timeout=30)  # 30 second timeout
                
            except RuntimeError:
                # No event loop running, safe to create one
                self.logger.info("No existing event loop, creating new one for MCP initialization")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(self._async_init_work())
                finally:
                    loop.close()
            
            self.logger.info("=== ASYNC MCP INITIALIZATION COMPLETE ===")
                    
        except Exception as e:
            self.logger.error(f"Error during async MCP initialization: {e}")
            self.logger.exception("Full traceback for async MCP init error:")
    
    async def _async_init_work(self):
        """The actual async initialization work."""
        try:
            # Connect to servers
            self.logger.info("Connecting to MCP servers...")
            results = await self.mcp_client.connect_all()
            self.logger.info(f"Connection results: {results}")
            
            # Update available tools and resources
            self.logger.info("Discovering tools and resources...")
            self.available_tools = self.mcp_client.get_available_tools()
            self.available_resources = self.mcp_client.get_available_resources()
            
            self.logger.info(f"Connected to MCP servers. Available tools: {len(self.available_tools)}, resources: {len(self.available_resources)}")
            
            # Log detailed tool information
            if self.available_tools:
                tool_names = [tool.name for tool in self.available_tools.values()]
                self.logger.info(f"Available tool names: {tool_names}")
            
            # Log connection results
            for server_name, success in results.items():
                if success:
                    self.logger.info(f"✓ Successfully connected to MCP server: {server_name}")
                else:
                    self.logger.warning(f"✗ Failed to connect to MCP server: {server_name}")
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Error in async init work: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if MCP integration is available and has tools."""
        return (self._initialized and 
                self.mcp_client is not None and 
                len(self.mcp_client.connections) > 0)
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        Get MCP tools formatted for LLM tool calling.
        
        Returns:
            List of tool definitions in OpenAI tool format
        """
        self.logger.info("=== FORMATTING MCP TOOLS FOR LLM ===")
        
        if not self.is_available():
            self.logger.warning("MCP integration not available, returning empty tool list")
            return []
            
        try:
            llm_tools = []
            
            self.logger.info(f"Processing {len(self.available_tools)} available tools")
            
            for tool_name, tool in self.available_tools.items():
                self.logger.debug(f"Formatting tool: {tool_name} from server: {tool.server_name}")
                
                llm_tool = {
                    "type": "function",
                    "function": {
                        "name": f"mcp_{tool_name}",  # Prefix to distinguish from native tools
                        "description": f"[MCP:{tool.server_name}] {tool.description}",
                        "parameters": tool.schema
                    }
                }
                llm_tools.append(llm_tool)
                
            self.logger.info(f"Successfully formatted {len(llm_tools)} MCP tools for LLM")
            self.logger.debug(f"Tool names: {[tool['function']['name'] for tool in llm_tools]}")
            
            return llm_tools
            
        except Exception as e:
            self.logger.error(f"Error formatting MCP tools for LLM: {e}")
            self.logger.exception("Full traceback for MCP tool formatting error:")
            return []
    
    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an MCP tool call.
        
        Args:
            tool_name: Name of the tool (may have mcp_ prefix)
            arguments: Tool arguments
            
        Returns:
            Dict with execution result
        """
        self.logger.info(f"=== EXECUTING MCP TOOL CALL: {tool_name} ===")
        self.logger.info(f"Arguments: {arguments}")
        
        if not self.is_available():
            self.logger.warning("MCP integration not available for tool execution")
            return {
                "success": False,
                "error": "MCP integration not available",
                "result": None
            }
        
        try:
            # Remove mcp_ prefix if present
            actual_tool_name = tool_name
            if tool_name.startswith("mcp_"):
                actual_tool_name = tool_name[4:]
                self.logger.debug(f"Removed mcp_ prefix, actual tool name: {actual_tool_name}")
            
            # Check if tool exists
            if actual_tool_name not in self.available_tools:
                self.logger.error(f"Tool '{actual_tool_name}' not found in available MCP tools")
                self.logger.debug(f"Available tools: {list(self.available_tools.keys())}")
                return {
                    "success": False,
                    "error": f"Tool '{actual_tool_name}' not found in available MCP tools",
                    "result": None
                }
            
            tool = self.available_tools[actual_tool_name]
            self.logger.info(f"Found tool '{actual_tool_name}' on server '{tool.server_name}'")
            
            # Log tool execution
            execution_log = {
                "timestamp": datetime.now().isoformat(),
                "tool_name": actual_tool_name,
                "server_name": tool.server_name,
                "arguments": arguments,
                "status": "executing"
            }
            self.session_log.append(execution_log)
            
            self.logger.info(f"Executing MCP tool '{actual_tool_name}' on server '{tool.server_name}' with args: {arguments}")
            
            # Execute tool using proper async handling
            self.logger.debug("Executing tool with async handling")
            
            try:
                # Try to use existing event loop with thread pool
                try:
                    asyncio.get_running_loop()
                    # There's an existing loop, run in separate thread
                    import concurrent.futures
                    
                    def run_tool_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(
                                self.mcp_client.call_tool(actual_tool_name, arguments)
                            )
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_tool_in_thread)
                        result = future.result(timeout=30)
                        
                except RuntimeError:
                    # No event loop running, safe to create one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            self.mcp_client.call_tool(actual_tool_name, arguments)
                        )
                    finally:
                        loop.close()
                
                # Update log with success
                execution_log["status"] = "completed"
                execution_log["result"] = result
                execution_log["completed_at"] = datetime.now().isoformat()
                
                self.logger.info(f"MCP tool '{actual_tool_name}' executed successfully")
                
                return {
                    "success": True,
                    "error": None,
                    "result": result,
                    "tool_name": actual_tool_name,
                    "server_name": tool.server_name
                }
                
            finally:
                try:
                    loop.close()
                except:
                    pass
                
        except Exception as e:
            # Update log with error
            if 'execution_log' in locals():
                execution_log["status"] = "failed"
                execution_log["error"] = str(e)
                execution_log["completed_at"] = datetime.now().isoformat()
            
            self.logger.error(f"Error executing MCP tool '{tool_name}': {e}")
            
            return {
                "success": False,
                "error": str(e),
                "result": None
            }
    
    def get_session_log(self) -> List[Dict[str, Any]]:
        """Get the current session log of MCP interactions."""
        return self.session_log.copy()
    
    def clear_session_log(self):
        """Clear the session log."""
        self.session_log.clear()
        self.logger.debug("MCP session log cleared")
    
    def format_tool_execution_for_chat(self, execution_result: Dict[str, Any]) -> str:
        """
        Format tool execution result for display in chat.
        
        Args:
            execution_result: Result from execute_tool_call
            
        Returns:
            Formatted string for chat display
        """
        try:
            if execution_result["success"]:
                tool_name = execution_result.get("tool_name", "unknown")
                server_name = execution_result.get("server_name", "unknown")
                result = execution_result.get("result", "")
                
                return f"🔧 **MCP Tool Executed**: `{tool_name}` (Server: {server_name})\n**Result**: {result}"
            else:
                error = execution_result.get("error", "Unknown error")
                return f"❌ **MCP Tool Failed**: {error}"
                
        except Exception as e:
            self.logger.error(f"Error formatting tool execution for chat: {e}")
            return f"❌ **MCP Tool Error**: {str(e)}"
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of the current MCP integration status."""
        if not self._initialized:
            return {
                "initialized": False,
                "servers": 0,
                "tools": 0,
                "resources": 0
            }
        
        return {
            "initialized": self._initialized,
            "available": self.is_available(),
            "servers": len(self.mcp_client.connections) if self.mcp_client else 0,
            "tools": len(self.available_tools),
            "resources": len(self.available_resources),
            "session_executions": len(self.session_log)
        }
    
    def cleanup(self):
        """Clean up MCP resources."""
        try:
            if self.mcp_client:
                # Handle async cleanup with event loop conflict handling
                try:
                    loop = asyncio.get_running_loop()
                    # Run async cleanup in a separate thread to avoid blocking
                    import concurrent.futures
                    
                    def run_cleanup_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(self.mcp_client.disconnect_all())
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_cleanup_in_thread)
                        future.result(timeout=10)  # Shorter timeout for cleanup
                        
                except RuntimeError:
                    # No event loop running, safe to create one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.mcp_client.disconnect_all())
                    finally:
                        loop.close()
                
                self.mcp_client = None
                
            self.available_tools.clear()
            self.available_resources.clear()
            self._initialized = False
            
            self.logger.info("MCP integration cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during MCP cleanup: {e}")


def create_mcp_integration_for_plugin(plugin) -> Optional[McpIntegrationService]:
    """
    Create MCP integration service for the BinAssist plugin.
    
    Args:
        plugin: BinAssist plugin instance
        
    Returns:
        McpIntegrationService instance or None if creation fails
    """
    logger = logging.getLogger("binassist.mcp_integration.factory")
    
    try:
        logger.info("=== CREATING MCP INTEGRATION FOR PLUGIN ===")
        
        # Check if MCP tools are enabled
        try:
            use_mcp_tools = plugin.settings.get_bool('binassist.use_mcp_tools')
            logger.info(f"MCP tools enabled: {use_mcp_tools}")
        except Exception as e:
            logger.warning(f"Failed to read MCP tools setting: {e}")
            use_mcp_tools = False
            
        if not use_mcp_tools:
            logger.info("MCP tools disabled, returning None")
            return None
        
        # Get enabled MCP servers from plugin
        logger.info("Getting enabled MCP servers from plugin...")
        enabled_servers = plugin.getEnabledMcpServers()
        logger.info(f"Found {len(enabled_servers) if enabled_servers else 0} enabled servers")
        
        # Debug log all server details
        if enabled_servers:
            for i, server in enumerate(enabled_servers):
                logger.info(f"Server {i+1} details: {server}")
        else:
            logger.warning("⚠️  No enabled servers returned from plugin!")
        
        if enabled_servers:
            for i, server in enumerate(enabled_servers):
                logger.debug(f"Server {i+1}: {server.get('name', 'unnamed')} ({server.get('transport', 'unknown')})")
        
        if not enabled_servers:
            logger.info("No enabled servers found, returning None")
            return None
        
        # Create and initialize MCP integration service
        logger.info("Creating MCP integration service...")
        mcp_service = McpIntegrationService(plugin.settings)
        
        logger.info("Initializing MCP service with plugin settings...")
        success = mcp_service.initialize_from_plugin_settings(enabled_servers)
        logger.info(f"MCP service initialization success: {success}")
        
        if success and mcp_service.is_available():
            logger.info("✓ MCP integration service created and available")
            return mcp_service
        else:
            return None
            
    except Exception as e:
        logger = logging.getLogger("binassist.mcp_integration")
        logger.error(f"Failed to create MCP integration for plugin: {e}")
        return None