"""
Code analysis service for Binary Ninja context extraction.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import binaryninja as bn
from binaryninja import FunctionGraphType, DisassemblySettings
from binaryninja.enums import DisassemblyOption
from binaryninja.lineardisassembly import LinearViewObject, LinearViewCursor

from .base_service import BaseService, ServiceError


@dataclass
class CodeContext:
    """
    Represents code context extracted from Binary Ninja.
    
    Attributes:
        function_text: Disassembled function text
        line_text: Single line disassembly
        address: Current address
        function_name: Name of the function
        il_type: Intermediate language type
        platform: Target platform name
        metadata: Additional context metadata
    """
    function_text: Optional[str] = None
    line_text: Optional[str] = None
    address: Optional[int] = None
    function_name: Optional[str] = None
    il_type: Optional[str] = None
    platform: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CodeAnalysisService(BaseService):
    """
    Service for extracting and analyzing code context from Binary Ninja.
    
    This service provides methods to extract various representations
    of code (ASM, LLIL, MLIL, HLIL, Pseudo-C) for LLM analysis.
    """
    
    def __init__(self):
        """Initialize the code analysis service."""
        super().__init__("code_analysis_service")
    
    def get_function_context(self, bv: bn.BinaryView, address: int, 
                           il_type: FunctionGraphType) -> CodeContext:
        """
        Get function context at a specific address and IL type.
        
        Args:
            bv: Binary view
            address: Address within the function
            il_type: Intermediate language type
            
        Returns:
            CodeContext object with function information
        """
        try:
            function = self._get_function_containing(bv, address)
            if not function:
                raise ServiceError(f"No function found at address {hex(address)}")
            
            # Get function text based on IL type
            function_text = self._get_function_text(bv, address, il_type)
            
            # Get line text
            line_text = self._get_line_text(bv, address, il_type)
            
            return CodeContext(
                function_text=function_text,
                line_text=line_text,
                address=address,
                function_name=function.name,
                il_type=il_type.name if hasattr(il_type, 'name') else str(il_type),
                platform=bv.platform.name if bv.platform else None,
                metadata={
                    "function_start": function.start,
                    "function_end": function.highest_address,
                    "architecture": bv.arch.name if bv.arch else None
                }
            )
            
        except Exception as e:
            self.handle_error(e, "function context extraction")
            raise ServiceError(f"Failed to get function context: {e}")
    
    def get_data_context(self, bv: bn.BinaryView, start_addr: int, 
                        end_addr: int) -> CodeContext:
        """
        Get data context for a specific address range.
        
        Args:
            bv: Binary view
            start_addr: Start address
            end_addr: End address
            
        Returns:
            CodeContext object with data information
        """
        try:
            data_text = self._data_to_text(bv, start_addr, end_addr)
            
            return CodeContext(
                function_text=data_text,
                address=start_addr,
                platform=bv.platform.name if bv.platform else None,
                metadata={
                    "start_address": start_addr,
                    "end_address": end_addr,
                    "data_type": "linear_data"
                }
            )
            
        except Exception as e:
            self.handle_error(e, "data context extraction")
            raise ServiceError(f"Failed to get data context: {e}")
    
    def _get_function_containing(self, bv: bn.BinaryView, address: int) -> Optional[bn.Function]:
        """Get the function containing the given address."""
        functions = bv.get_functions_containing(address)
        return functions[0] if functions else None
    
    def _get_function_text(self, bv: bn.BinaryView, address: int, 
                          il_type: FunctionGraphType) -> str:
        """Get function text based on IL type."""
        if il_type == FunctionGraphType.NormalFunctionGraph:
            return self._asm_to_text(bv, address)
        elif il_type == FunctionGraphType.LowLevelILFunctionGraph:
            return self._llil_to_text(bv, address)
        elif il_type == FunctionGraphType.MediumLevelILFunctionGraph:
            return self._mlil_to_text(bv, address)
        elif il_type == FunctionGraphType.HighLevelILFunctionGraph:
            return self._hlil_to_text(bv, address)
        elif il_type == FunctionGraphType.HighLevelLanguageRepresentationFunctionGraph:
            return self._pseudo_c_to_text(bv, address)
        else:
            raise ServiceError(f"Unsupported IL type: {il_type}")
    
    def _get_line_text(self, bv: bn.BinaryView, address: int, 
                      il_type: FunctionGraphType) -> str:
        """Get single line text at address."""
        if il_type == FunctionGraphType.NormalFunctionGraph:
            return f"0x{address:08x}  {bv.get_disassembly(address)}"
        else:
            # For IL types, we need to get the current IL instruction
            # This is more complex and would require integration with Binary Ninja's UI
            return f"0x{address:08x}  {bv.get_disassembly(address)}"
    
    def _asm_to_text(self, bv: bn.BinaryView, address: int) -> str:
        """Convert assembly instructions to text."""
        function = self._get_function_containing(bv, address)
        if not function:
            return ""
        
        asm_instructions = ""
        for bb in function.basic_blocks:
            for dt in bb.disassembly_text:
                s = str(dt)
                asm_instructions += f"\n0x{dt.address:08x}  {s}"
        
        tokens = function.get_type_tokens()[0].tokens
        function_header = ''.join(x.text for x in tokens)
        
        return f"{function_header}\n{asm_instructions}\n"
    
    def _llil_to_text(self, bv: bn.BinaryView, address: int) -> str:
        """Convert Low Level IL to text."""
        function = self._get_function_containing(bv, address)
        if not function:
            return ""
        
        tokens = function.get_type_tokens()[0].tokens
        function_header = ''.join(x.text for x in tokens)
        
        llil_instructions = '\n'.join(
            f'0x{instr.address:08x}  {instr}' 
            for instr in function.low_level_il.instructions
        )
        
        return f"{function_header}\n{{\n{llil_instructions}\n}}\n"
    
    def _mlil_to_text(self, bv: bn.BinaryView, address: int) -> str:
        """Convert Medium Level IL to text."""
        function = self._get_function_containing(bv, address)
        if not function:
            return ""
        
        tokens = function.get_type_tokens()[0].tokens
        function_header = ''.join(x.text for x in tokens)
        
        mlil_instructions = '\n'.join(
            f'0x{instr.address:08x}  {instr}' 
            for instr in function.medium_level_il.instructions
        )
        
        return f"{function_header}\n{{\n{mlil_instructions}\n}}\n"
    
    def _hlil_to_text(self, bv: bn.BinaryView, address: int) -> str:
        """Convert High Level IL to text."""
        function = self._get_function_containing(bv, address)
        if not function:
            return ""
        
        tokens = function.get_type_tokens()[0].tokens
        function_header = ''.join(x.text for x in tokens)
        
        hlil_instructions = '\n'.join(
            f'  {instr};' 
            for instr in function.high_level_il.instructions
        )
        
        return f"{function_header}\n{{\n{hlil_instructions}\n}}\n"
    
    def _pseudo_c_to_text(self, bv: bn.BinaryView, address: int) -> str:
        """Convert Pseudo-C to text."""
        function = self._get_function_containing(bv, address)
        if not function:
            return ""
        
        lines = []
        settings = DisassemblySettings()
        settings.set_option(DisassemblyOption.ShowAddress, False)
        
        obj = LinearViewObject.language_representation(bv, settings)
        cursor_end = LinearViewCursor(obj)
        cursor_end.seek_to_address(function.highest_address)
        
        body = bv.get_next_linear_disassembly_lines(cursor_end)
        cursor_end.seek_to_address(function.highest_address)
        header = bv.get_previous_linear_disassembly_lines(cursor_end)
        
        for line in header:
            lines.append(f'{str(line)}\n')
        
        for line in body:
            lines.append(f'{str(line)}\n')
        
        return ''.join(lines)
    
    def _data_to_text(self, bv: bn.BinaryView, start_addr: int, end_addr: int) -> str:
        """Convert data range to text."""
        lines = []
        settings = DisassemblySettings()
        settings.set_option(DisassemblyOption.ShowAddress, True)
        
        obj = LinearViewObject.language_representation(bv, settings)
        cursor = LinearViewCursor(obj)
        cursor.seek_to_address(start_addr)
        
        while cursor.current_object.start <= end_addr:
            lines.extend([str(line) for line in cursor.lines])
            cursor.next()
        
        return "\n".join(lines)