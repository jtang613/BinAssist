"""
Binary Ninja tool implementations.

These functions handle the actual execution of tools within Binary Ninja,
including function renaming, variable operations, and struct creation.
"""

import binaryninja as bn
from binaryninja.enums import SourceType
from binaryninja.types import StructureBuilder, Type, TypeClass
from binaryninja.highlevelil import HighLevelILOperation, HighLevelILInstruction
from typing import Dict, Any, Optional


class ToolExecutionError(Exception):
    """Exception raised during tool execution."""
    pass


def rename_function_handler(bv: bn.BinaryView, address: int, 
                          new_name: str, **kwargs) -> Dict[str, Any]:
    """
    Rename a function at the given address.
    
    Args:
        bv: Binary view
        address: Address within the function
        new_name: New name for the function
        
    Returns:
        Result dictionary with success status and message
    """
    try:
        functions = bv.get_functions_containing(address)
        if not functions:
            raise ToolExecutionError(f"No function found at address {hex(address)}")
        
        function = functions[0]
        old_name = function.name
        
        # Start transaction for undo support
        with bv.undoable_transaction():
            function.name = new_name
        
        return {
            "success": True,
            "message": f"Renamed function from '{old_name}' to '{new_name}'",
            "old_name": old_name,
            "new_name": new_name,
            "address": hex(address)
        }
        
    except Exception as e:
        raise ToolExecutionError(f"Failed to rename function: {e}")


def rename_variable_handler(bv: bn.BinaryView, address: int,
                          func_name: str, var_name: str, new_name: str, 
                          **kwargs) -> Dict[str, Any]:
    """
    Rename a variable within a function.
    
    Args:
        bv: Binary view
        address: Address within the function
        func_name: Name of the function containing the variable
        var_name: Current name of the variable
        new_name: New name for the variable
        
    Returns:
        Result dictionary with success status and message
    """
    try:
        functions = bv.get_functions_containing(address)
        if not functions:
            raise ToolExecutionError(f"No function found at address {hex(address)}")
        
        function = functions[0]
        
        # Find the variable
        target_var = None
        for var in function.vars:
            if var.name == var_name.replace('&', ''):
                target_var = var
                break
        
        if not target_var:
            raise ToolExecutionError(f"Variable '{var_name}' not found in function '{func_name}'")
        
        old_name = target_var.name
        
        # Start transaction for undo support
        with bv.undoable_transaction():
            target_var.name = new_name
            bv.reanalyze()
        
        return {
            "success": True,
            "message": f"Renamed variable from '{old_name}' to '{new_name}' in function '{func_name}'",
            "old_name": old_name,
            "new_name": new_name,
            "function_name": func_name,
            "address": hex(address)
        }
        
    except Exception as e:
        raise ToolExecutionError(f"Failed to rename variable: {e}")


def retype_variable_handler(bv: bn.BinaryView, address: int,
                          func_name: str, var_name: str, new_type: str,
                          **kwargs) -> Dict[str, Any]:
    """
    Change the type of a variable within a function.
    
    Args:
        bv: Binary view
        address: Address within the function
        func_name: Name of the function containing the variable
        var_name: Name of the variable
        new_type: New type for the variable
        
    Returns:
        Result dictionary with success status and message
    """
    try:
        functions = bv.get_functions_containing(address)
        if not functions:
            raise ToolExecutionError(f"No function found at address {hex(address)}")
        
        function = functions[0]
        
        # Find the variable
        target_var = None
        for var in function.vars:
            if var.name == var_name:
                target_var = var
                break
        
        if not target_var:
            raise ToolExecutionError(f"Variable '{var_name}' not found in function '{func_name}'")
        
        old_type = str(target_var.type)
        
        # Parse the new type
        try:
            parsed_type = bv.parse_type_string(new_type)[0]
        except Exception as e:
            raise ToolExecutionError(f"Failed to parse type '{new_type}': {e}")
        
        # Start transaction for undo support
        with bv.undoable_transaction():
            target_var.type = parsed_type
            bv.reanalyze()
        
        return {
            "success": True,
            "message": f"Changed type of variable '{var_name}' from '{old_type}' to '{new_type}' in function '{func_name}'",
            "variable_name": var_name,
            "old_type": old_type,
            "new_type": new_type,
            "function_name": func_name,
            "address": hex(address)
        }
        
    except Exception as e:
        raise ToolExecutionError(f"Failed to retype variable: {e}")


def auto_create_struct_handler(bv: bn.BinaryView, address: int,
                             func_name: str, var_name: str,
                             **kwargs) -> Dict[str, Any]:
    """
    Automatically create a structure datatype from a variable's usage patterns.
    
    Args:
        bv: Binary view
        address: Address within the function
        func_name: Name of the function containing the variable
        var_name: Name of the variable to analyze
        
    Returns:
        Result dictionary with success status and message
    """
    try:
        functions = bv.get_functions_containing(address)
        if not functions:
            raise ToolExecutionError(f"No function found at address {hex(address)}")
        
        function = functions[0]
        
        # Find the variable
        target_var = None
        for var in function.vars:
            if var.name == var_name:
                target_var = var
                break
        
        if not target_var:
            raise ToolExecutionError(f"Variable '{var_name}' not found in function '{func_name}'")
        
        # Analyze the variable's usage to infer structure
        offsets = _analyze_variable_offsets(function, target_var)
        
        if not offsets:
            raise ToolExecutionError(f"No structure offsets detected for variable '{var_name}'")
        
        # Create the structure
        struct_name = f"{func_name}_struct_{var_name}"
        struct_type = _create_structure_from_offsets(bv, offsets, struct_name)
        
        # Apply the structure type to the variable
        with bv.undoable_transaction():
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
            "message": f"Created structure '{struct_name}' with {len(offsets)} fields for variable '{var_name}'",
            "variable_name": var_name,
            "struct_name": struct_name,
            "field_count": len(offsets),
            "function_name": func_name,
            "address": hex(address)
        }
        
    except Exception as e:
        raise ToolExecutionError(f"Failed to auto-create struct: {e}")


def _analyze_variable_offsets(function: bn.Function, var: bn.Variable) -> Dict[int, Type]:
    """
    Analyze a variable's usage to detect structure offsets.
    
    Args:
        function: Function containing the variable
        var: Variable to analyze
        
    Returns:
        Dictionary mapping offsets to inferred types
    """
    offsets = {}
    
    def collect_offsets(expr, target_var, offsets, parent_expr=None):
        """Recursively collect offset usage patterns."""
        if expr.operation == HighLevelILOperation.HLIL_ADD:
            left = expr.left
            right = expr.right
            
            # Look for patterns like var + constant
            if ((left.operation == HighLevelILOperation.HLIL_VAR and 
                 left.var == target_var and 
                 right.operation in (HighLevelILOperation.HLIL_CONST, HighLevelILOperation.HLIL_CONST_PTR)) or
                (right.operation == HighLevelILOperation.HLIL_VAR and 
                 right.var == target_var and 
                 left.operation in (HighLevelILOperation.HLIL_CONST, HighLevelILOperation.HLIL_CONST_PTR))):
                
                offset = right.constant if left.var == target_var else left.constant
                
                # Infer type from usage context
                if parent_expr and parent_expr.operation == HighLevelILOperation.HLIL_DEREF:
                    inferred_type = parent_expr.expr_type or Type.int(4)
                    offsets[offset] = inferred_type
                elif parent_expr and parent_expr.operation == HighLevelILOperation.HLIL_ASSIGN:
                    inferred_type = parent_expr.dest.expr_type or Type.int(4)
                    offsets[offset] = inferred_type
                else:
                    offsets[offset] = Type.int(4)  # Default type
        
        # Recurse through child expressions
        for operand in expr.operands:
            if isinstance(operand, HighLevelILInstruction):
                collect_offsets(operand, target_var, offsets, expr)
            elif isinstance(operand, list):
                for suboperand in operand:
                    if isinstance(suboperand, HighLevelILInstruction):
                        collect_offsets(suboperand, target_var, offsets, expr)
    
    # Analyze all HLIL instructions in the function
    try:
        for block in function.hlil:
            for instruction in block:
                collect_offsets(instruction, var, offsets)
    except Exception:
        # If HLIL analysis fails, return empty offsets
        pass
    
    return offsets


def _create_structure_from_offsets(bv: bn.BinaryView, offsets: Dict[int, Type], 
                                 struct_name: str) -> Type:
    """
    Create a structure type from offset information.
    
    Args:
        bv: Binary view
        offsets: Dictionary mapping offsets to types
        struct_name: Name for the structure
        
    Returns:
        Structure type
    """
    struct_builder = StructureBuilder.create()
    
    for offset in sorted(offsets.keys()):
        field_type = offsets[offset]
        field_name = f'field_{offset:x}'
        
        try:
            struct_builder.insert(offset, field_type, field_name)
        except ValueError:
            # Field already exists at this offset; skip
            pass
    
    return Type.structure_type(struct_builder)