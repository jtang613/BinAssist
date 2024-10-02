from binaryninja import *
from PySide6 import QtWidgets

class ToolCalling:
    """
    A class containing handler methods for various Binary Ninja actions and their associated templates and prompts.
    """

    FN_TEMPLATES = [
        {
            "type": "function",
            "function": {
                "name": "rename_function",
                "description": "Rename a function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_name": {
                            "type": "string",
                            "description": "The new name for the function. (ie: recv_data)"
                        }
                    },
                    "required": ["new_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rename_variable",
                "description": "Rename a variable within a function",
                "parameters": {
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
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "retype_variable",
                "description": "Set a variable data type within a function",
                "parameters": {
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
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "auto_create_struct",
                "description": "Automatically create a sructure datatype from a variable given its offset uses in a given function.",
                "parameters": {
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
                }
            }
        },
    ]

    ACTION_PROMPTS = {
        "rename_function": "Use the 'rename_function' tool:\n```\n{code}\n```\n" +
                           "Examine the code functionality, strings and log parameters.\n" +
                           "If you detect C++ Super::Derived::Method or Class::Method style class names, recommend that name first.\n" +
                           "CREATE A JSON TOOL_CALL LIST WITH SUGGESTIONS FOR THREE POSSIBLE FUNCTION NAMES " +
                           "THAT ALIGN AS CLOSELY AS POSSIBLE TO WHAT THE CODE ABOVE DOES.\n" +
                           "RESPOND ONLY WITH THE RENAME_FUNCTION PARAMETER (new_name). DO NOT INCLUDE ANY OTHER TEXT.\n" +
                           "ALL JSON MUST BE PROPERLY FORMATTED WITH NO EMBEDDED COMMENTS.\n",

        "rename_variable": "Use the 'rename_variable' tool:\n```\n{code}\n```\n" +
                           "Examine the code functionality, strings and log parameters.\n" +
                           "SUGGEST VARIABLE NAMES THAT BETTER ALIGN WITH THE CODE FUNCTIONALITY.\n" +
                           "RESPOND ONLY WITH THE RENAME_VARIABLE PARAMETERS (func_name, var_name, new_name). DO NOT INCLUDE ANY OTHER TEXT.\n" +
                           "ALL JSON VALUES MUST BE TEXT STRINGS, INCLUDING NUMBERS AND ADDRESSES. ie: \"0x1234abcd\"\n" +
                           "ALL JSON MUST BE PROPERLY FORMATTED WITH NO EMBEDDED COMMENTS.\n",

        "retype_variable": "Use the 'retype_variable' tool:\n```\n{code}\n```\n" +
                           "Examine the code functionality, strings and log parameters.\n" +
                           "SUGGEST VARIABLE TYPES THAT BETTER ALIGN WITH THE CODE FUNCTIONALITY.\n" +
                           "RESPOND ONLY WITH THE RETYPE_VARIABLE PARAMETERS (func_name, var_name, new_type). DO NOT INCLUDE ANY OTHER TEXT.\n" +
                           "ALL JSON VALUES MUST BE TEXT STRINGS, INCLUDING NUMBERS AND ADDRESSES. ie: \"0x1234abcd\"\n" +
                           "ALL JSON MUST BE PROPERLY FORMATTED WITH NO EMBEDDED COMMENTS.\n",

        "auto_create_struct": "Use the 'auto_create_struct' tool:\n```\n{code}\n```\n" +
                           "Examine the code functionality, parameters and variables being used.\n" +
                           "IF YOU DETECT A VARIABLE THAT USES OFFSET ACCESS SUCH AS `*(arg1 + 0c)` OR VARIABLES LIKELY TO BE STRUCTURES OR CLASSES,\n" +
                           "RESPOND ONLY WITH THE AUT_CREATE_SRTUCT PARAMETERS (func_name, var_name). DO NOT INCLUDE ANY OTHER TEXT.\n" +
                           "ALL JSON VALUES MUST BE TEXT STRINGS, INCLUDING NUMBERS AND ADDRESSES. ie: \"0x1234abcd\"\n" +
                           "ALL JSON MUST BE PROPERLY FORMATTED WITH NO EMBEDDED COMMENTS.\n",
    }

    @staticmethod
    def handle_rename_function(bv, actions_table, offset_addr, description: str, row: int) -> None:
        new_name = description.strip()
        current_function = bv.get_functions_containing(offset_addr)[0].name = new_name
        if current_function:
            actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Applied"))
        else:
            actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Failed: Function not found"))

    @staticmethod
    def handle_rename_variable(bv, actions_table, offset_addr, description: str, row: int) -> None:
        var_name, new_name = description.split(' -> ')
        current_function = bv.get_functions_containing(offset_addr)[0]
        if current_function:
            for var in current_function.vars:
                if var.name == var_name.replace('&',''):
                    var.name = new_name
                    actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Applied"))
                    bv.reanalyze()
                    break
            else:
                actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Failed: Variable not found"))
        else:
            actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Failed: Function not found"))

    @staticmethod
    def handle_retype_variable(bv, actions_table, offset_addr, description: str, row: int) -> None:
        var_name, new_type = description.split(' -> ')
        current_function = bv.get_functions_containing(offset_addr)[0]
        if current_function:
            for var in current_function.vars:
                if var.name == var_name:
                    try:
                        t = bv.parse_type_string(new_type)[0]
                        var.type = t
                        actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Applied"))
                        bv.reanalyze()
                    except Exception as e:
                        print(f"Failed to parse type: {new_type}: {e}")
                    break
            else:
                actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Failed: Variable not found"))
        else:
            actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Failed: Function not found"))

    @staticmethod
    def handle_auto_create_struct(bv: BinaryView, actions_table, offset_addr, description: str, row: int) -> None:

        variable_name = description

        # Get the function at the given address
        function = bv.get_functions_containing(offset_addr)[0]
        if function is None:
            print(f"Function at {hex(offset_addr)} not found")
            return

        # Find the variable with the given name in the function
        var = None
        for tmp_var in function.vars:
            if tmp_var.name == variable_name:
                var = tmp_var
                break
        if var is None:
            print(f"Variable {variable_name} not found in function at {hex(function_address)}")
            return

        # Dictionary to store offsets and their inferred types
        offsets = {}

        # Recursive function to collect offsets and infer types
        def collect_offsets(expr, arg1_var, offsets, parent_expr=None):
            if expr.operation == HighLevelILOperation.HLIL_ADD:
                left = expr.left
                right = expr.right
                if ((left.operation == HighLevelILOperation.HLIL_VAR and left.var == arg1_var and right.operation in (HighLevelILOperation.HLIL_CONST, HighLevelILOperation.HLIL_CONST_PTR)) or
                    (right.operation == HighLevelILOperation.HLIL_VAR and right.var == arg1_var and left.operation in (HighLevelILOperation.HLIL_CONST, HighLevelILOperation.HLIL_CONST_PTR))):
                    # We have arg1_var + constant
                    offset = right.constant if left.var == arg1_var else left.constant
                    # Now, see how this ADD is used
                    if parent_expr is not None:
                        if parent_expr.operation == HighLevelILOperation.HLIL_DEREF:
                            # We have *(arg1 + offset)
                            inferred_type = parent_expr.expr_type
                            offsets[offset] = inferred_type
                        elif parent_expr.operation == HighLevelILOperation.HLIL_ASSIGN:
                            # Assignment to a variable
                            inferred_type = parent_expr.dest.expr_type
                            offsets[offset] = inferred_type
                        elif parent_expr.operation == HighLevelILOperation.HLIL_CALL:
                            # arg1 + offset is passed as a parameter
                            try:
                                param_index = parent_expr.params.index(expr)
                                callee = None
                                if parent_expr.dest.operation == HighLevelILOperation.HLIL_CONST_PTR:
                                    callee = parent_expr.dest.constant
                                elif parent_expr.dest.operation == HighLevelILOperation.HLIL_IMPORT:
                                    callee_name = parent_expr.dest.name
                                    symbol = bv.get_symbol_by_raw_name(callee_name)
                                    callee = symbol.address if symbol else None
                                elif parent_expr.dest.operation == HighLevelILOperation.HLIL_VAR:
                                    # Handle function pointers
                                    var = parent_expr.dest.var
                                    var_type = function.get_type_of_var(var)
                                    if var_type and var_type.type_class == TypeClass.PointerTypeClass:
                                        func_type = var_type.target
                                        if func_type and func_type.parameters:
                                            callee_func_params = func_type.parameters
                                            if param_index < len(callee_func_params):
                                                inferred_type = callee_func_params[param_index].type
                                                offsets[offset] = inferred_type
                                            return
                                if callee:
                                    callee_func = bv.get_function_at(callee)
                                    if callee_func and callee_func.parameter_vars:
                                        if param_index < len(callee_func.parameter_vars):
                                            inferred_type = callee_func.parameter_vars[param_index].type
                                            offsets[offset] = inferred_type
                            except ValueError:
                                pass  # expr not found in params
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
                collect_offsets(instruction, var, offsets)

        # Create the structure
        from binaryninja.types import StructureBuilder
        struct_builder = StructureBuilder.create()
        for offset in sorted(offsets.keys()):
            field_type = offsets[offset] or Type.int(4)  # Default to int32 if type is None
            field_name = f'field_{offset:x}'
            try:
                struct_builder.insert(offset, field_type, field_name)
            except ValueError:
                # Field already exists at this offset; skip or handle as needed
                pass
        struct_type = Type.structure_type(struct_builder)
        struct_name = f"{function.name}_struct_{variable_name}"
        bv.define_user_type(struct_name, struct_type)
        named_struct_type = Type.named_type_from_type(struct_name, struct_type)
        if var.type.type_class == TypeClass.PointerTypeClass:
            named_struct_ptr_type = Type.pointer(bv.arch, named_struct_type)
            var.type = named_struct_ptr_type
        else:
            var.type = named_struct_type
        actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Applied"))
        bv.reanalyze()


