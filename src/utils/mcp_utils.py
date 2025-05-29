"""
Utilities for MCP configuration and testing.
"""

from typing import Dict, List, Any
from binaryninja.settings import Settings
import logging

from ..core.mcp import MCPConfig, MCPServerConfig
from ..core.services.mcp_service import MCPService

class MCPConfigHelper:
    """Helper class for MCP configuration management."""
    
    @staticmethod
    def create_sample_config() -> MCPConfig:
        """Create a sample MCP configuration for testing."""
        servers = [
            MCPServerConfig(
                name="filesystem",
                transport_type="stdio",
                command="mcp-server-filesystem",
                args=["--root", "/tmp"],
                timeout=30,
                enabled=False  # Disabled by default
            ),
            MCPServerConfig(
                name="git",
                transport_type="stdio", 
                command="mcp-server-git",
                args=[],
                timeout=30,
                enabled=False  # Disabled by default
            )
        ]
        
        return MCPConfig(
            servers=servers,
            global_timeout=60,
            max_concurrent_connections=3,
            retry_attempts=2
        )
    
    @staticmethod
    def save_config_to_settings(config: MCPConfig, settings: Settings):
        """Save MCP configuration to BinaryNinja settings."""
        # Save server count
        settings.set_integer('binassist.mcp_server_count', len(config.servers))
        
        # Save each server configuration
        for i, server in enumerate(config.servers):
            settings.set_string(f'binassist.mcp_server_{i}_name', server.name)
            settings.set_string(f'binassist.mcp_server_{i}_transport', server.transport_type)
            settings.set_string(f'binassist.mcp_server_{i}_command', server.command or '')
            settings.set_string_list(f'binassist.mcp_server_{i}_args', server.args or [])
            settings.set_integer(f'binassist.mcp_server_{i}_timeout', server.timeout)
            settings.set_bool(f'binassist.mcp_server_{i}_enabled', server.enabled)
        
        # Save global settings
        settings.set_integer('binassist.mcp_global_timeout', config.global_timeout)
        settings.set_integer('binassist.mcp_max_connections', config.max_concurrent_connections)
        settings.set_integer('binassist.mcp_retry_attempts', config.retry_attempts)
    
    @staticmethod
    def initialize_default_config(settings: Settings):
        """Initialize default MCP configuration if none exists."""
        if settings.get_integer('binassist.mcp_server_count', -1) == -1:
            config = MCPConfigHelper.create_sample_config()
            MCPConfigHelper.save_config_to_settings(config, settings)
            
            logger = logging.getLogger("binassist.mcp.config")
            logger.info("Initialized default MCP configuration")

class MCPTester:
    """Helper class for testing MCP functionality."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger("binassist.mcp.tester")
    
    def test_mcp_service(self) -> Dict[str, Any]:
        """Test MCP service initialization and connection."""
        results = {
            "service_init": False,
            "connections": {},
            "tools": {},
            "resources": {},
            "errors": []
        }
        
        try:
            # Initialize MCP service
            mcp_service = MCPService(self.settings)
            results["service_init"] = mcp_service.initialize()
            
            if results["service_init"] and mcp_service.is_available():
                # Get available tools and resources
                results["tools"] = mcp_service.get_available_tools()
                results["resources"] = mcp_service.get_available_resources()
                
                # Test a simple tool call if tools are available
                if results["tools"]:
                    tool_name = list(results["tools"].keys())[0]
                    try:
                        # This would need to be customized based on actual tool parameters
                        test_result = mcp_service.call_tool(tool_name, {})
                        results[f"test_tool_{tool_name}"] = test_result
                    except Exception as e:
                        results["errors"].append(f"Tool test failed: {e}")
                
                # Clean up
                mcp_service.cleanup()
            
        except Exception as e:
            results["errors"].append(f"Service test failed: {e}")
            self.logger.error(f"MCP service test failed: {e}")
        
        return results
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current MCP configuration."""
        results = {
            "valid": False,
            "server_count": 0,
            "enabled_servers": 0,
            "issues": []
        }
        
        try:
            server_count = self.settings.get_integer('binassist.mcp_server_count', 0)
            results["server_count"] = server_count
            
            if server_count == 0:
                results["issues"].append("No MCP servers configured")
                return results
            
            enabled_count = 0
            for i in range(server_count):
                enabled = self.settings.get_bool(f'binassist.mcp_server_{i}_enabled', False)
                if enabled:
                    enabled_count += 1
                    
                    # Validate server configuration
                    name = self.settings.get_string(f'binassist.mcp_server_{i}_name', '')
                    command = self.settings.get_string(f'binassist.mcp_server_{i}_command', '')
                    
                    if not name:
                        results["issues"].append(f"Server {i} has no name")
                    if not command:
                        results["issues"].append(f"Server {i} ({name}) has no command")
            
            results["enabled_servers"] = enabled_count
            
            if enabled_count == 0:
                results["issues"].append("No MCP servers enabled")
            
            results["valid"] = len(results["issues"]) == 0
            
        except Exception as e:
            results["issues"].append(f"Configuration validation failed: {e}")
            self.logger.error(f"MCP config validation failed: {e}")
        
        return results

def setup_mcp_integration(settings: Settings) -> bool:
    """Set up MCP integration with default configuration."""
    try:
        # Initialize default configuration
        MCPConfigHelper.initialize_default_config(settings)
        
        # Test the configuration
        tester = MCPTester(settings)
        validation = tester.validate_config()
        
        logger = logging.getLogger("binassist.mcp.setup")
        logger.info(f"MCP setup complete. Validation: {validation}")
        
        return validation["valid"] or validation["server_count"] > 0
        
    except Exception as e:
        logger = logging.getLogger("binassist.mcp.setup")
        logger.error(f"MCP setup failed: {e}")
        return False