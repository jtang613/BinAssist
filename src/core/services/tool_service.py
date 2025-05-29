"""
Tool service for managing and executing Binary Ninja tools.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import json

from .base_service import BaseService, ServiceError
from ..models.tool_call import ToolCall, ToolResult

# Import Binary Ninja modules for tool implementations
import binaryninja as bn
from binaryninja import BinaryView, log
from binaryninja.types import StructureBuilder, Type, TypeClass
from binaryninja.highlevelil import HighLevelILOperation, HighLevelILInstruction
from PySide6 import QtWidgets


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
        """Register default Binary Ninja tools using working implementations."""
        # Register rename_function tool
        self.register_tool(ToolDefinition(
            name="rename_function",
            description="Rename a function",
            parameters={
                "type": "object",
                "properties": {
                    "new_name": {
                        "type": "string",
                        "description": "The new name for the function. (ie: recv_data)"
                    }
                },
                "required": ["new_name"]
            },
            handler=self._handle_rename_function
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
                        "description": "The name of the function containing the variable. (ie: sub_40001234)"
                    },
                    "var_name": {
                        "type": "string",
                        "description": "The current name of the variable. (ie: var_20)"
                    },
                    "new_name": {
                        "type": "string",
                        "description": "The new name for the variable. (ie: recv_buf)"
                    }
                },
                "required": ["func_name", "var_name", "new_name"]
            },
            handler=self._handle_rename_variable
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
                        "description": "The name of the function containing the variable. (ie: sub_40001234)"
                    },
                    "var_name": {
                        "type": "string",
                        "description": "The current name of the variable. (ie: rax_12)"
                    },
                    "new_type": {
                        "type": "string",
                        "description": "The new type for the variable. (ie: int32_t)"
                    }
                },
                "required": ["func_name", "var_name", "new_type"]
            },
            handler=self._handle_retype_variable
        ))
        
        # Register auto_create_struct tool
        self.register_tool(ToolDefinition(
            name="auto_create_struct",
            description="Automatically create a structure datatype from a variable given its offset uses in a given function.",
            parameters={
                "type": "object",
                "properties": {
                    "func_name": {
                        "type": "string",
                        "description": "The name of the function containing the variable. (ie: sub_40001234)"
                    },
                    "var_name": {
                        "type": "string",
                        "description": "The current name of the variable. (ie: rax_12)"
                    }
                },
                "required": ["func_name", "var_name"]
            },
            handler=self._handle_auto_create_struct
        ))
    
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
            ),
            "rename_variable": (
                "Use the 'rename_variable' tool:\n```\n{code}\n```\n"
                "Examine the code functionality, strings and log parameters.\n"
                "SUGGEST VARIABLE NAMES THAT BETTER ALIGN WITH THE CODE FUNCTIONALITY.\n"
            ),
            "retype_variable": (
                "Use the 'retype_variable' tool:\n```\n{code}\n```\n"
                "Examine the code functionality, strings and log parameters.\n"
                "SUGGEST VARIABLE TYPES THAT BETTER ALIGN WITH THE CODE FUNCTIONALITY.\n"
            ),
            "auto_create_struct": (
                "Use the 'auto_create_struct' tool:\n```\n{code}\n```\n"
                "Examine the code functionality, parameters and variables being used.\n"
                "IF YOU DETECT A VARIABLE THAT USES OFFSET ACCESS SUCH AS `*(arg1 + 0c)` OR VARIABLES LIKELY TO BE STRUCTURES OR CLASSES,\n"
            )
        }
    
    # Working handler implementations from toolcalling.py
    def _handle_rename_function(self, bv: BinaryView, address: int, new_name: str, **kwargs) -> Dict[str, Any]:
        """Rename a function at the given address."""
        try:
            functions = bv.get_functions_containing(address)
            if not functions:
                return {"success": False, "error": f"No function found at address {hex(address)}"}
            
            function = functions[0]
            old_name = function.name
            
            # Use undoable transaction for undo support
            with bv.undoable_transaction():
                function.name = new_name
            
            return {
                "success": True,
                "message": f"Renamed function from '{old_name}' to '{new_name}'",
                "old_name": old_name,
                "new_name": new_name
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to rename function: {e}"}
    
    def _handle_rename_variable(self, bv: BinaryView, address: int, func_name: str, var_name: str, new_name: str, **kwargs) -> Dict[str, Any]:
        """Rename a variable within a function."""
        try:
            functions = bv.get_functions_containing(address)
            if not functions:
                return {"success": False, "error": f"No function found at address {hex(address)}"}
            
            function = functions[0]
            
            # Find the variable
            target_var = None
            for var in function.vars:
                if var.name == var_name.replace('&', ''):
                    target_var = var
                    break
            
            if not target_var:
                return {"success": False, "error": f"Variable '{var_name}' not found in function"}
            
            old_name = target_var.name
            
            # Use undoable transaction for undo support
            with bv.undoable_transaction():
                target_var.name = new_name
                bv.reanalyze()
            
            return {
                "success": True,
                "message": f"Renamed variable from '{old_name}' to '{new_name}'",
                "old_name": old_name,
                "new_name": new_name
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to rename variable: {e}"}
    
    def _handle_retype_variable(self, bv: BinaryView, address: int, func_name: str, var_name: str, new_type: str, **kwargs) -> Dict[str, Any]:
        """Change the type of a variable within a function."""
        try:
            functions = bv.get_functions_containing(address)
            if not functions:
                return {"success": False, "error": f"No function found at address {hex(address)}"}
            
            function = functions[0]
            
            # Find the variable
            target_var = None
            for var in function.vars:
                if var.name == var_name:
                    target_var = var
                    break
            
            if not target_var:
                return {"success": False, "error": f"Variable '{var_name}' not found in function"}
            
            old_type = str(target_var.type)
            
            # Parse the new type
            try:
                parsed_type = bv.parse_type_string(new_type)[0]
            except Exception as e:
                return {"success": False, "error": f"Failed to parse type '{new_type}': {e}"}
            
            # Use undoable transaction for undo support
            with bv.undoable_transaction():
                target_var.type = parsed_type
                bv.reanalyze()
            
            return {
                "success": True,
                "message": f"Changed type of variable '{var_name}' from '{old_type}' to '{new_type}'",
                "old_type": old_type,
                "new_type": new_type
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to retype variable: {e}"}
    
    def _handle_auto_create_struct(self, bv: BinaryView, address: int, func_name: str, var_name: str, **kwargs) -> Dict[str, Any]:
        """Auto-create a structure from variable usage patterns (working implementation from toolcalling.py)."""
        try:
            functions = bv.get_functions_containing(address)
            if not functions:
                return {"success": False, "error": f"No function found at address {hex(address)}"}
            
            function = functions[0]
            
            # Find the variable
            target_var = None
            for var in function.vars:
                if var.name == var_name:
                    target_var = var
                    break
            
            if not target_var:
                return {"success": False, "error": f"Variable '{var_name}' not found in function"}
            
            # Use the working implementation from toolcalling.py
            offsets = {}
            
            def collect_offsets(expr, arg1_var, offsets, parent_expr=None):
                if expr.operation == HighLevelILOperation.HLIL_ADD:
                    left = expr.left
                    right = expr.right
                    if ((left.operation == HighLevelILOperation.HLIL_VAR and left.var == arg1_var and right.operation in (HighLevelILOperation.HLIL_CONST, HighLevelILOperation.HLIL_CONST_PTR)) or
                        (right.operation == HighLevelILOperation.HLIL_VAR and right.var == arg1_var and left.operation in (HighLevelILOperation.HLIL_CONST, HighLevelILOperation.HLIL_CONST_PTR))):
                        offset = right.constant if left.var == arg1_var else left.constant
                        if parent_expr is not None:
                            if parent_expr.operation == HighLevelILOperation.HLIL_DEREF:
                                inferred_type = parent_expr.expr_type
                                offsets[offset] = inferred_type
                            elif parent_expr.operation == HighLevelILOperation.HLIL_ASSIGN:
                                inferred_type = parent_expr.dest.expr_type
                                offsets[offset] = inferred_type
                            else:
                                inferred_type = expr.expr_type
                                offsets[offset] = inferred_type
                # Recurse on child expressions
                for operand in expr.operands:
                    if isinstance(operand, HighLevelILInstruction):
                        collect_offsets(operand, arg1_var, offsets, expr)
                    elif isinstance(operand, list):
                        for suboperand in operand:
                            if isinstance(suboperand, HighLevelILInstruction):
                                collect_offsets(suboperand, arg1_var, offsets, expr)
            
            # Traverse the HLIL instructions to collect offsets
            for block in function.hlil:
                for instruction in block:
                    collect_offsets(instruction, target_var, offsets)
            
            if not offsets:
                return {"success": False, "error": "No structure offsets detected for variable"}
            
            # Create the structure using undoable transaction
            with bv.undoable_transaction():
                struct_builder = StructureBuilder.create()
                for offset in sorted(offsets.keys()):
                    field_type = offsets[offset] or Type.int(4)  # Default to int32 if type is None
                    field_name = f'field_{offset:x}'
                    try:
                        struct_builder.insert(offset, field_type, field_name)
                    except ValueError:
                        pass  # Field already exists at this offset; skip
                
                struct_type = Type.structure_type(struct_builder)
                struct_name = f"{function.name}_struct_{var_name}"
                bv.define_user_type(struct_name, struct_type)
                named_struct_type = Type.named_type_from_type(struct_name, struct_type)
                
                if target_var.type.type_class == TypeClass.PointerTypeClass:
                    named_struct_ptr_type = Type.pointer(bv.arch, named_struct_type)
                    target_var.type = named_struct_ptr_type
                else:
                    target_var.type = named_struct_type
                
                bv.reanalyze()
            
            return {
                "success": True,
                "message": f"Created structure '{struct_name}' with {len(offsets)} fields",
                "struct_name": struct_name,
                "field_count": len(offsets)
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to auto-create struct: {e}"}
    
    # UI integration methods for Actions table population
    def handle_action_for_ui(self, bv: BinaryView, actions_table: QtWidgets.QTableWidget, 
                           offset_addr: int, tool_name: str, description: str, row: int) -> None:
        """Handle tool execution for UI Actions table (preserving toolcalling.py behavior)."""
        log.log_info(f"[BinAssist] ToolService.handle_action_for_ui called: tool_name='{tool_name}', description='{description}', row={row}, offset_addr=0x{offset_addr:x}")
        
        try:
            if tool_name == "rename_function":
                log.log_debug(f"[BinAssist] Processing rename_function action")
                new_name = description.strip()
                log.log_debug(f"[BinAssist] Extracted new_name: '{new_name}'")
                result = self._handle_rename_function(bv, offset_addr, new_name=new_name)
                log.log_debug(f"[BinAssist] rename_function result: {result}")
                if result["success"]:
                    actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Applied"))
                    log.log_info(f"[BinAssist] rename_function applied successfully")
                else:
                    actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"Failed: {result['error']}"))
                    log.log_warn(f"[BinAssist] rename_function failed: {result['error']}")
                    
            elif tool_name == "rename_variable":
                log.log_debug(f"[BinAssist] Processing rename_variable action")
                var_name, new_name = description.split(' -> ')
                func_name = bv.get_functions_containing(offset_addr)[0].name if bv.get_functions_containing(offset_addr) else ""
                log.log_debug(f"[BinAssist] Extracted var_name: '{var_name}', new_name: '{new_name}', func_name: '{func_name}'")
                result = self._handle_rename_variable(bv, offset_addr, func_name=func_name, var_name=var_name, new_name=new_name)
                log.log_debug(f"[BinAssist] rename_variable result: {result}")
                if result["success"]:
                    actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Applied"))
                    log.log_info(f"[BinAssist] rename_variable applied successfully")
                else:
                    actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"Failed: {result['error']}"))
                    log.log_warn(f"[BinAssist] rename_variable failed: {result['error']}")
                    
            elif tool_name == "retype_variable":
                log.log_debug(f"[BinAssist] Processing retype_variable action")
                var_name, new_type = description.split(' -> ')
                func_name = bv.get_functions_containing(offset_addr)[0].name if bv.get_functions_containing(offset_addr) else ""
                log.log_debug(f"[BinAssist] Extracted var_name: '{var_name}', new_type: '{new_type}', func_name: '{func_name}'")
                result = self._handle_retype_variable(bv, offset_addr, func_name=func_name, var_name=var_name, new_type=new_type)
                log.log_debug(f"[BinAssist] retype_variable result: {result}")
                if result["success"]:
                    actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Applied"))
                    log.log_info(f"[BinAssist] retype_variable applied successfully")
                else:
                    actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"Failed: {result['error']}"))
                    log.log_warn(f"[BinAssist] retype_variable failed: {result['error']}")
                    
            elif tool_name == "auto_create_struct":
                log.log_debug(f"[BinAssist] Processing auto_create_struct action")
                variable_name = description
                func_name = bv.get_functions_containing(offset_addr)[0].name if bv.get_functions_containing(offset_addr) else ""
                log.log_debug(f"[BinAssist] Extracted variable_name: '{variable_name}', func_name: '{func_name}'")
                result = self._handle_auto_create_struct(bv, offset_addr, func_name=func_name, var_name=variable_name)
                log.log_debug(f"[BinAssist] auto_create_struct result: {result}")
                if result["success"]:
                    actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Applied"))
                    log.log_info(f"[BinAssist] auto_create_struct applied successfully")
                else:
                    actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"Failed: {result['error']}"))
                    log.log_warn(f"[BinAssist] auto_create_struct failed: {result['error']}")
            else:
                log.log_error(f"[BinAssist] Unknown tool_name: '{tool_name}'")
                actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"Error: Unknown action '{tool_name}'"))
                    
        except Exception as e:
            log.log_error(f"[BinAssist] Exception in handle_action_for_ui: {e}")
            actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"Error: {str(e)}"))
            self.handle_error(e, f"UI tool execution: {tool_name}")