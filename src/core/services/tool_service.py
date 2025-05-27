"""
Tool service for managing and executing Binary Ninja tools.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import json

from .base_service import BaseService, ServiceError
from ..models.tool_call import ToolCall, ToolResult


@dataclass
class ToolDefinition:
    """
    Represents a tool definition.
    
    Attributes:
        name: Tool name
        description: Tool description
        parameters: Parameter schema
        handler: Function to execute the tool
        metadata: Additional tool metadata
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable
    metadata: Optional[Dict[str, Any]] = None
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class ToolService(BaseService):
    """
    Service for managing and executing tools.
    
    This service handles tool registration, execution, and result processing
    for Binary Ninja automation.
    """
    
    def __init__(self):
        """Initialize the tool service."""
        super().__init__("tool_service")
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()
    
    def register_tool(self, tool: ToolDefinition) -> None:
        """
        Register a tool.
        
        Args:
            tool: Tool definition to register
        """
        self._tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get all tool definitions in OpenAI format.
        
        Returns:
            List of tool definitions
        """
        return [tool.to_openai_format() for tool in self._tools.values()]
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def execute_tool(self, tool_call: ToolCall, bv, address: int = None) -> ToolResult:
        """
        Execute a tool call.
        
        Args:
            tool_call: The tool call to execute
            bv: Binary view (for Binary Ninja operations)
            address: Optional address context
            
        Returns:
            Tool execution result
        """
        try:
            tool = self._tools.get(tool_call.name)
            if not tool:
                return ToolResult.error_result(
                    tool_call.id, 
                    f"Unknown tool: {tool_call.name}"
                )
            
            # Execute the tool
            result = tool.handler(bv, address, **tool_call.arguments)
            
            return ToolResult.success_result(tool_call.id, result)
            
        except Exception as e:
            self.handle_error(e, f"tool execution: {tool_call.name}")
            return ToolResult.error_result(
                tool_call.id,
                f"Tool execution failed: {str(e)}"
            )
    
    def execute_tools(self, tool_calls: List[ToolCall], bv, 
                     address: int = None) -> List[ToolResult]:
        """
        Execute multiple tool calls.
        
        Args:
            tool_calls: List of tool calls to execute
            bv: Binary view
            address: Optional address context
            
        Returns:
            List of tool execution results
        """
        results = []
        for tool_call in tool_calls:
            if self.is_stopped():
                break
            result = self.execute_tool(tool_call, bv, address)
            results.append(result)
        
        return results
    
    def _register_default_tools(self) -> None:
        """Register default Binary Ninja tools."""
        # Import tool handlers
        try:
            from ...tools.binary_ninja_tools import (
                rename_function_handler,
                rename_variable_handler,
                retype_variable_handler,
                auto_create_struct_handler
            )
            
            # Register rename_function tool
            self.register_tool(ToolDefinition(
                name="rename_function",
                description="Rename a function",
                parameters={
                    "type": "object",
                    "properties": {
                        "new_name": {
                            "type": "string",
                            "description": "The new name for the function"
                        }
                    },
                    "required": ["new_name"]
                },
                handler=rename_function_handler
            ))
            
            # Register rename_variable tool
            self.register_tool(ToolDefinition(
                name="rename_variable",
                description="Rename a variable within a function",
                parameters={
                    "type": "object",
                    "properties": {
                        "func_name": {
                            "type": "string",
                            "description": "The name of the function containing the variable"
                        },
                        "var_name": {
                            "type": "string",
                            "description": "The current name of the variable"
                        },
                        "new_name": {
                            "type": "string",
                            "description": "The new name for the variable"
                        }
                    },
                    "required": ["func_name", "var_name", "new_name"]
                },
                handler=rename_variable_handler
            ))
            
            # Register retype_variable tool
            self.register_tool(ToolDefinition(
                name="retype_variable",
                description="Set a variable data type within a function",
                parameters={
                    "type": "object",
                    "properties": {
                        "func_name": {
                            "type": "string",
                            "description": "The name of the function containing the variable"
                        },
                        "var_name": {
                            "type": "string",
                            "description": "The current name of the variable"
                        },
                        "new_type": {
                            "type": "string",
                            "description": "The new type for the variable"
                        }
                    },
                    "required": ["func_name", "var_name", "new_type"]
                },
                handler=retype_variable_handler
            ))
            
            # Register auto_create_struct tool
            self.register_tool(ToolDefinition(
                name="auto_create_struct",
                description="Automatically create a structure datatype from a variable",
                parameters={
                    "type": "object",
                    "properties": {
                        "func_name": {
                            "type": "string",
                            "description": "The name of the function containing the variable"
                        },
                        "var_name": {
                            "type": "string",
                            "description": "The current name of the variable"
                        }
                    },
                    "required": ["func_name", "var_name"]
                },
                handler=auto_create_struct_handler
            ))
            
        except ImportError as e:
            self.logger.warning(f"Failed to import tool handlers: {e}")
    
    def get_action_prompts(self) -> Dict[str, str]:
        """
        Get action prompts for tool calling.
        
        Returns:
            Dictionary mapping tool names to their prompts
        """
        # These could be moved to a configuration file
        return {
            "rename_function": (
                "Use the 'rename_function' tool:\n```\n{code}\n```\n"
                "Examine the code functionality, strings and log parameters.\n"
                "If you detect C++ Super::Derived::Method or Class::Method style class names, recommend that name first.\n"
                "CREATE A JSON TOOL_CALL LIST WITH SUGGESTIONS FOR THREE POSSIBLE FUNCTION NAMES "
                "THAT ALIGN AS CLOSELY AS POSSIBLE TO WHAT THE CODE ABOVE DOES.\n"
                "RESPOND ONLY WITH THE RENAME_FUNCTION PARAMETER (new_name). DO NOT INCLUDE ANY OTHER TEXT.\n"
                "ALL JSON MUST BE PROPERLY FORMATTED WITH NO EMBEDDED COMMENTS.\n"
            ),
            "rename_variable": (
                "Use the 'rename_variable' tool:\n```\n{code}\n```\n"
                "Examine the code functionality, strings and log parameters.\n"
                "SUGGEST VARIABLE NAMES THAT BETTER ALIGN WITH THE CODE FUNCTIONALITY.\n"
                "RESPOND ONLY WITH THE RENAME_VARIABLE PARAMETERS (func_name, var_name, new_name). DO NOT INCLUDE ANY OTHER TEXT.\n"
                "ALL JSON VALUES MUST BE TEXT STRINGS, INCLUDING NUMBERS AND ADDRESSES. ie: \"0x1234abcd\"\n"
                "ALL JSON MUST BE PROPERLY FORMATTED WITH NO EMBEDDED COMMENTS.\n"
            ),
            "retype_variable": (
                "Use the 'retype_variable' tool:\n```\n{code}\n```\n"
                "Examine the code functionality, strings and log parameters.\n"
                "SUGGEST VARIABLE TYPES THAT BETTER ALIGN WITH THE CODE FUNCTIONALITY.\n"
                "RESPOND ONLY WITH THE RETYPE_VARIABLE PARAMETERS (func_name, var_name, new_type). DO NOT INCLUDE ANY OTHER TEXT.\n"
                "ALL JSON VALUES MUST BE TEXT STRINGS, INCLUDING NUMBERS AND ADDRESSES. ie: \"0x1234abcd\"\n"
                "ALL JSON MUST BE PROPERLY FORMATTED WITH NO EMBEDDED COMMENTS.\n"
            ),
            "auto_create_struct": (
                "Use the 'auto_create_struct' tool:\n```\n{code}\n```\n"
                "Examine the code functionality, parameters and variables being used.\n"
                "IF YOU DETECT A VARIABLE THAT USES OFFSET ACCESS SUCH AS `*(arg1 + 0c)` OR VARIABLES LIKELY TO BE STRUCTURES OR CLASSES,\n"
                "RESPOND ONLY WITH THE AUTO_CREATE_STRUCT PARAMETERS (func_name, var_name). DO NOT INCLUDE ANY OTHER TEXT.\n"
                "ALL JSON VALUES MUST BE TEXT STRINGS, INCLUDING NUMBERS AND ADDRESSES. ie: \"0x1234abcd\"\n"
                "ALL JSON MUST BE PROPERLY FORMATTED WITH NO EMBEDDED COMMENTS.\n"
            )
        }