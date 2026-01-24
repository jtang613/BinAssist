#!/usr/bin/env python3

from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import hashlib
import os
import binaryninja as bn
from binaryninja import BinaryView, Function, BasicBlock, DisassemblySettings, DisassemblyOption, LinearViewObject, LinearViewCursor

# Setup BinAssist logger
try:
    import binaryninja
    log = binaryninja.log.Logger(0, "BinAssist")
except ImportError:
    # Fallback for testing outside Binary Ninja
    class MockLog:
        @staticmethod
        def log_info(msg): print(f"[BinAssist] INFO: {msg}")
        @staticmethod
        def log_error(msg): print(f"[BinAssist] ERROR: {msg}")
        @staticmethod
        def log_warn(msg): print(f"[BinAssist] WARN: {msg}")
    log = MockLog()


class ViewLevel(Enum):
    """Binary Ninja abstraction levels"""
    ASM = "assembly"
    LLIL = "llil"
    MLIL = "mlil" 
    HLIL = "hlil"
    PSEUDO_C = "pseudo_c"
    PSEUDO_RUST = "pseudo_rust"
    PSEUDO_PYTHON = "pseudo_python"


class BinaryContextService:
    """Service for extracting context-aware information from Binary Ninja"""

    def __init__(self, binary_view: Optional[BinaryView] = None, view_frame=None):
        """Initialize with optional binary view and view frame"""
        self._binary_view = binary_view
        self._current_offset = 0
        self._view_frame = view_frame
        self._cached_binary_hash = None  # Cache hash to avoid recalculating

    def set_binary_view(self, binary_view: BinaryView) -> None:
        """Set the current binary view"""
        # Only clear cached hash if the binary view actually changed
        if self._binary_view != binary_view:
            self._cached_binary_hash = None
        self._binary_view = binary_view
    
    def set_view_frame(self, view_frame) -> None:
        """Set the current view frame for UI context"""
        self._view_frame = view_frame

    def set_current_offset(self, offset: int) -> None:
        """Set the current offset/address"""
        self._current_offset = offset

    def set_binary_hash(self, binary_hash: str) -> None:
        """
        Set the cached binary hash.

        This should be called ONCE when the binary is first opened,
        after calculating the hash and performing any necessary migration.

        Args:
            binary_hash: The content-based binary hash to cache
        """
        self._cached_binary_hash = binary_hash
        log.log_debug(f"Cached binary hash: {binary_hash}")

    def get_binary_hash(self) -> Optional[str]:
        """
        Get the cached binary hash.

        Returns the hash that was previously set via set_binary_hash().
        If no hash has been cached, returns None.

        Controllers should call AnalysisDBService.get_binary_hash() once
        when the binary is opened, perform migration if needed, then cache
        the result here using set_binary_hash().

        Returns:
            The cached binary hash, or None if not yet set
        """
        return self._cached_binary_hash
    
    def get_current_context(self) -> Dict[str, Any]:
        """Get complete context snapshot for current state"""
        if not self._binary_view:
            return {"error": "No binary view available"}
        
        context = {
            "offset": self._current_offset,
            "offset_hex": f"0x{self._current_offset:x}",
            "current_view_level": self.get_current_view_level().value,
            "binary_info": self._get_binary_info(),
            "function_context": self._get_function_context(self._current_offset),
            "view_capabilities": self._get_view_capabilities(),
        }
        
        return context
    
    def _get_binary_info(self) -> Dict[str, Any]:
        """Extract basic binary metadata"""
        if not self._binary_view:
            return {}
        
        try:
            # Get the full file path
            full_path = getattr(self._binary_view.file, 'filename', 'Unknown')
            
            # Extract just the filename from the path
            filename_only = os.path.basename(full_path) if full_path != 'Unknown' else 'Unknown'
            
            return {
                "filename": filename_only,  # Just the filename (for backward compatibility)
                "filepath": full_path,      # Full path to the binary
                "architecture": str(self._binary_view.arch) if self._binary_view.arch else 'Unknown',
                "platform": str(self._binary_view.platform) if self._binary_view.platform else 'Unknown',
                "entry_point": self._binary_view.entry_point,
                "entry_point_hex": f"0x{self._binary_view.entry_point:x}" if self._binary_view.entry_point else None,
                "address_size": getattr(self._binary_view, 'address_size', 0),
                "endianness": "little" if getattr(self._binary_view, 'endianness', None) == bn.Endianness.LittleEndian else "big",
                "file_type": str(self._binary_view.view_type) if hasattr(self._binary_view, 'view_type') else 'Unknown',
                "total_functions": len(self._binary_view.functions) if self._binary_view.functions else 0,
                "segments": self._get_segments_info()
            }
        except Exception as e:
            return {"error": f"Failed to get binary info: {str(e)}"}
    
    def get_binary_metadata_for_rlhf(self) -> Dict[str, Any]:
        """
        Extract binary metadata specifically for RLHF feedback storage.
        Returns filename, size, and sha256 hash as required by RLHF schema.
        """
        if not self._binary_view:
            return {
                "filename": "Unknown",
                "size": 0,
                "sha256": "unknown"
            }
        
        try:
            # Get filename
            filename = "Unknown"
            if hasattr(self._binary_view, 'file') and self._binary_view.file:
                if hasattr(self._binary_view.file, 'filename') and self._binary_view.file.filename:
                    filename = os.path.basename(self._binary_view.file.filename)
            
            # Calculate file size and hash
            size = 0
            sha256_hash = "unknown"
            
            if hasattr(self._binary_view, 'file') and self._binary_view.file:
                # Try to get the raw data for hashing
                try:
                    # Get all data from the binary
                    raw_data = self._binary_view.read(0, len(self._binary_view))
                    if raw_data:
                        size = len(raw_data)
                        sha256_hash = hashlib.sha256(raw_data).hexdigest()
                except Exception:
                    # Fallback: try to get file size from filesystem if available
                    if hasattr(self._binary_view.file, 'filename') and self._binary_view.file.filename:
                        try:
                            if os.path.exists(self._binary_view.file.filename):
                                size = os.path.getsize(self._binary_view.file.filename)
                                with open(self._binary_view.file.filename, 'rb') as f:
                                    sha256_hash = hashlib.sha256(f.read()).hexdigest()
                        except Exception:
                            pass
            
            return {
                "filename": filename,
                "size": size,
                "sha256": sha256_hash
            }
            
        except Exception as e:
            log.log_error(f"Failed to get binary metadata for RLHF: {e}")
            return {
                "filename": "Unknown",
                "size": 0,
                "sha256": "unknown"
            }
    
    def _get_segments_info(self) -> List[Dict[str, Any]]:
        """Get segment information with error handling"""
        try:
            segments = []
            for seg in self._binary_view.segments:
                seg_info = {
                    "start": f"0x{seg.start:x}",
                    "end": f"0x{seg.end:x}",
                    "length": getattr(seg, 'length', seg.end - seg.start),
                }
                
                # Add permission flags with safe attribute access
                for attr in ['readable', 'writable', 'executable']:
                    seg_info[attr] = getattr(seg, attr, False)
                
                segments.append(seg_info)
            return segments
        except Exception:
            return []
    
    def _get_function_context(self, address: int) -> Optional[Dict[str, Any]]:
        """Get context for function containing the given address"""
        if not self._binary_view:
            return None
        
        function = self._binary_view.get_function_at(address)
        if not function:
            # Try to find function containing this address
            functions = self._binary_view.get_functions_containing(address)
            function = functions[0] if functions else None
        
        if not function:
            return None
        
        return {
            "name": function.name,
            "start": f"0x{function.start:x}",
            "end": f"0x{function.start + function.total_bytes:x}",
            "size": function.total_bytes,
            "basic_blocks": len(function.basic_blocks),
            "call_sites": len(function.call_sites),
            "callers": [f"0x{caller.start:x}" for caller in function.callers],
            "callees": [f"0x{callee.start:x}" for callee in function.callees],
            "symbol": function.symbol.full_name if function.symbol else None,
            "analysis_skipped": function.analysis_skipped,
            "can_return": function.can_return.value if hasattr(function.can_return, 'value') else str(function.can_return),
            "prototype": self._get_function_prototype(function),
        }
    
    def _get_function_prototype(self, function) -> str:
        """Get function prototype/signature"""
        try:
            # Try to get the function type from Binary Ninja
            if hasattr(function, 'function_type') and function.function_type:
                return str(function.function_type)
            
            # Fallback: construct basic prototype from name and parameters
            params = []
            if hasattr(function, 'parameter_vars') and function.parameter_vars:
                for i, param in enumerate(function.parameter_vars.vars):
                    param_type = str(param.type) if hasattr(param, 'type') and param.type else f"arg{i+1}_type"
                    param_name = param.name if hasattr(param, 'name') and param.name else f"arg{i+1}"
                    params.append(f"{param_type} {param_name}")
            
            # Get return type
            return_type = "void"
            if hasattr(function, 'return_type') and function.return_type:
                return_type = str(function.return_type)
            
            param_str = ", ".join(params) if params else "void"
            return f"{return_type} {function.name}({param_str})"
            
        except Exception:
            # Ultimate fallback
            return f"unknown {function.name}(...)"
    
    def _get_view_capabilities(self) -> Dict[str, bool]:
        """Determine what abstraction levels are available"""
        if not self._binary_view:
            return {}
        
        function = self._binary_view.get_function_at(self._current_offset)
        if not function:
            functions = self._binary_view.get_functions_containing(self._current_offset)
            function = functions[0] if functions else None
        
        capabilities = {
            "assembly": True,  # Always available
            "llil": function is not None,
            "mlil": function is not None,
            "hlil": function is not None,
            "pseudo_c": function is not None,
            "data_view": True,  # Can always view as data
        }
        
        return capabilities
    
    def get_current_view_level(self) -> ViewLevel:
        """Detect the current view level from Binary Ninja UI"""
        try:
            from binaryninjaui import UIContext
            context = UIContext.activeContext()
            if not context:
                return ViewLevel.ASM
            
            view_frame = context.getCurrentViewFrame()
            if not view_frame:
                return ViewLevel.ASM
            
            view_interface = view_frame.getCurrentViewInterface()
            if not view_interface:
                return ViewLevel.ASM
            
            # Get the IL view type
            if hasattr(view_interface, 'getILViewType'):
                il_view_type = view_interface.getILViewType()
                
                # Extract the view_type enum value
                if hasattr(il_view_type, 'view_type'):
                    view_type_enum = il_view_type.view_type
                    view_type_value = view_type_enum.value if hasattr(view_type_enum, 'value') else int(view_type_enum)
                    
                    try:
                        log.log_info(f"View type detected: {view_type_enum} (value: {view_type_value})")
                    except:
                        pass
                    
                    # Map FunctionGraphType enum values to our ViewLevel enum
                    view_mapping = {
                        0: ViewLevel.ASM,          # NormalFunctionGraph (disassembly)
                        1: ViewLevel.LLIL,         # LowLevelILFunctionGraph  
                        4: ViewLevel.MLIL,         # MediumLevelILFunctionGraph
                        8: ViewLevel.HLIL,         # HighLevelILFunctionGraph
                        10: ViewLevel.PSEUDO_C,    # HighLevelLanguageRepresentationFunctionGraph (Pseudo-C, etc.)
                    }
                    
                    detected_level = view_mapping.get(view_type_value, ViewLevel.ASM)
                    
                    # For HighLevelLanguageRepresentationFunctionGraph (value 10), check the name
                    if view_type_value == 10 and hasattr(il_view_type, 'name') and il_view_type.name:
                        name = il_view_type.name.lower()
                        if 'rust' in name:
                            detected_level = ViewLevel.PSEUDO_RUST
                        elif 'python' in name:
                            detected_level = ViewLevel.PSEUDO_PYTHON
                        # Default to PSEUDO_C for other pseudo languages
                    
                    try:
                        log.log_info(f"Detected view level: {detected_level.value}")
                    except:
                        pass
                    
                    return detected_level
                
        except Exception as e:
            try:
                log.log_error(f"View detection failed: {str(e)}")
            except:
                pass
        
        # Fallback to assembly
        return ViewLevel.ASM
    
    def get_code_at_level(self, address: int, view_level: ViewLevel, context_lines: int = 5) -> Dict[str, Any]:
        """Get code at specified abstraction level with context"""
        if not self._binary_view:
            return {"error": "No binary view available"}
        
        result = {
            "address": f"0x{address:x}",
            "view_level": view_level.value,
            "lines": [],
            "error": None
        }
        
        try:
            if view_level == ViewLevel.ASM:
                result["lines"] = self._get_assembly_code(address, context_lines)
            elif view_level in [ViewLevel.LLIL, ViewLevel.MLIL, ViewLevel.HLIL]:
                result["lines"] = self._get_il_code(address, view_level, context_lines)
            elif view_level in [ViewLevel.PSEUDO_C, ViewLevel.PSEUDO_RUST, ViewLevel.PSEUDO_PYTHON]:
                result["lines"] = self._get_pseudo_code(address, view_level, context_lines)
            else:
                result["error"] = f"Unsupported view level: {view_level.value}"
                
        except Exception as e:
            result["error"] = f"Failed to get code: {str(e)}"
        
        return result
    
    def _get_assembly_code(self, address: int, context_lines: int) -> List[Dict[str, Any]]:
        """Get assembly code with context using Binary Ninja's formatting"""
        try:
            function = self._binary_view.get_functions_containing(address)
            if not function:
                return [{"error": "No function found at address"}]
            
            function = function[0]
            
            # Get formatted assembly using Binary Ninja's API
            asm_text = self._asm_to_text(address)
            if not asm_text:
                return [{"error": "Failed to get assembly text"}]
            
            # Parse the formatted text into lines
            lines = []
            for line_num, line in enumerate(asm_text.split('\n')):
                line = line.strip()
                if not line:
                    continue
                    
                # Extract address if present
                addr_match = None
                if line.startswith('0x'):
                    parts = line.split('  ', 1)
                    if len(parts) == 2:
                        addr_str = parts[0]
                        content = parts[1]
                        try:
                            addr_int = int(addr_str, 16)
                            addr_match = addr_int
                        except ValueError:
                            pass
                    else:
                        content = line
                else:
                    content = line
                
                
                lines.append({
                    "address": f"0x{addr_match:x}" if addr_match else "",
                    "content": content,
                    "is_current": addr_match == address if addr_match else False,
                    "line_number": line_num + 1
                })
            
            return lines
            
        except Exception as e:
            return [{"error": f"Failed to get assembly: {str(e)}"}]
    
    def _asm_to_text(self, addr: int) -> str:
        """Convert assembly instructions at a specific address to text"""
        function = self._binary_view.get_functions_containing(addr)
        if not function:
            return None
        
        function = function[0]
        asm_instructions = ""
        
        for bb in function.basic_blocks:
            for dt in bb.disassembly_text:
                s = str(dt)
                asm_instructions += f"\n0x{dt.address:08x}  {s}"
        
        tokens = function.get_type_tokens()[0].tokens
        return f"{''.join(x.text for x in tokens)}\n{asm_instructions}\n"
    
    def _get_il_code(self, address: int, view_level: ViewLevel, context_lines: int) -> List[Dict[str, Any]]:
        """Get Intermediate Language code with context using Binary Ninja's formatting"""
        try:
            # Get formatted IL text using Binary Ninja's APIs
            if view_level == ViewLevel.LLIL:
                il_text = self._llil_to_text(address)
            elif view_level == ViewLevel.MLIL:
                il_text = self._mlil_to_text(address)
            elif view_level == ViewLevel.HLIL:
                il_text = self._hlil_to_text(address)
            else:
                return [{"error": f"Invalid IL level: {view_level.value}"}]
            
            if not il_text:
                return [{"error": f"Failed to get {view_level.value} text"}]
            
            # Parse the formatted text into lines
            lines = []
            for line_num, line in enumerate(il_text.split('\n')):
                line = line.strip()
                if not line:
                    continue
                
                # Extract address if present
                addr_match = None
                if line.startswith('0x'):
                    parts = line.split('  ', 1)
                    if len(parts) == 2:
                        addr_str = parts[0]
                        content = parts[1]
                        try:
                            addr_int = int(addr_str, 16)
                            addr_match = addr_int
                        except ValueError:
                            pass
                    else:
                        content = line
                else:
                    content = line
                
                
                lines.append({
                    "address": f"0x{addr_match:x}" if addr_match else "",
                    "content": content,
                    "is_current": addr_match == address if addr_match else False,
                    "line_number": line_num + 1
                })
            
            return lines
            
        except Exception as e:
            return [{"error": f"Failed to get {view_level.value}: {str(e)}"}]
    
    def _llil_to_text(self, addr: int) -> str:
        """Convert Low Level Intermediate Language (LLIL) instructions to text"""
        function = self._binary_view.get_functions_containing(addr)
        if not function:
            return None
        
        function = function[0]
        tokens = function.get_type_tokens()[0].tokens
        llil_instructions = '\n'.join(f'0x{instr.address:08x}  {instr}' for instr in function.low_level_il.instructions)
        return f"{''.join(x.text for x in tokens)}\n{llil_instructions}\n"
    
    def _mlil_to_text(self, addr: int) -> str:
        """Convert Medium Level Intermediate Language (MLIL) instructions to text"""
        function = self._binary_view.get_functions_containing(addr)
        if not function:
            return None
        
        function = function[0]
        tokens = function.get_type_tokens()[0].tokens
        mlil_instructions = '\n'.join(f'0x{instr.address:08x}  {instr}' for instr in function.medium_level_il.instructions)
        return f"{''.join(x.text for x in tokens)}\n{{\n{mlil_instructions}\n}}\n"
    
    def _hlil_to_text(self, addr: int) -> str:
        """Convert High Level Intermediate Language (HLIL) instructions to text"""
        function = self._binary_view.get_functions_containing(addr)
        if not function:
            return None
        
        function = function[0]
        tokens = function.get_type_tokens()[0].tokens
        hlil_instructions = '\n'.join(f'  {instr};' for instr in function.high_level_il.instructions)
        return f"{''.join(x.text for x in tokens)}\n{{\n{hlil_instructions}\n}}\n"
    
    def _get_pseudo_code(self, address: int, view_level: ViewLevel, context_lines: int) -> List[Dict[str, Any]]:
        """Get pseudo code representation using Binary Ninja's Linear View API"""
        try:
            # Get formatted pseudo-C text using Binary Ninja's API
            pseudo_text = self._pseudo_c_to_text(address)
            if not pseudo_text:
                return [{"error": "Failed to get pseudo-C text"}]
            
            # Parse the formatted text into lines
            lines = []
            for line_num, line in enumerate(pseudo_text.split('\n')):
                # Don't strip leading whitespace to preserve indentation
                original_line = line
                line_stripped = line.strip()
                
                if not line_stripped:
                    continue
                
                # For pseudo-C, we typically don't have addresses on every line
                # but we can try to detect them
                addr_match = None
                content = original_line
                
                # Check if line has an address prefix
                if line_stripped.startswith('0x'):
                    parts = line_stripped.split('  ', 1)
                    if len(parts) == 2:
                        try:
                            addr_int = int(parts[0], 16)
                            addr_match = addr_int
                            # Preserve original formatting but use the content after address
                            content = line.replace(parts[0], '').lstrip()
                        except ValueError:
                            pass
                
                
                lines.append({
                    "address": f"0x{addr_match:x}" if addr_match else "",
                    "content": content,
                    "is_current": addr_match == address if addr_match else False,
                    "line_number": line_num + 1
                })
            
            return lines
            
        except Exception as e:
            return [{"error": f"Failed to generate pseudo code: {str(e)}"}]
    
    def _pseudo_c_to_text(self, addr: int) -> str:
        """Convert Pseudo-C instructions at a specific address to text using Linear View"""
        function = self._binary_view.get_functions_containing(addr)
        if not function:
            return None
        
        function = function[0]
        
        lines = []
        settings = DisassemblySettings()
        settings.set_option(DisassemblyOption.ShowAddress, False)
        obj = LinearViewObject.language_representation(self._binary_view, settings)
        cursor_end = LinearViewCursor(obj)
        cursor_end.seek_to_address(function.highest_address)
        body = self._binary_view.get_next_linear_disassembly_lines(cursor_end)
        cursor_end.seek_to_address(function.highest_address)
        header = self._binary_view.get_previous_linear_disassembly_lines(cursor_end)


        for line in header:
            lines.append(f'{str(line)}\n')

        for line in body:
            lines.append(f'{str(line)}\n')

        c_instructions = ''.join(lines)
        return f"{c_instructions}"
    
    
    def get_hexdump(self, address: int, size: int = 256) -> Dict[str, Any]:
        """Get hexdump for address range"""
        if not self._binary_view:
            return {"error": "No binary view available"}
        
        result = {
            "address": f"0x{address:x}",
            "size": size,
            "lines": [],
            "error": None
        }
        
        try:
            data = self._binary_view.read(address, size)
            if not data:
                result["error"] = "Failed to read data at address"
                return result
            
            # Format as hexdump (16 bytes per line)
            for i in range(0, len(data), 16):
                line_addr = address + i
                line_data = data[i:i+16]
                
                # Hex representation
                hex_bytes = ' '.join(f'{b:02x}' for b in line_data)
                hex_bytes = hex_bytes.ljust(47)  # Pad to consistent width
                
                # ASCII representation
                ascii_repr = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in line_data)
                
                result["lines"].append({
                    "address": f"0x{line_addr:08x}",
                    "hex": hex_bytes,
                    "ascii": ascii_repr,
                    "bytes": line_data.hex()
                })
                
        except Exception as e:
            result["error"] = f"Failed to generate hexdump: {str(e)}"
        
        return result
    
    def get_line_with_context(self, address: int, view_level: ViewLevel, context_lines: int = 5) -> Dict[str, Any]:
        """Get line at address with N lines of context above and below.

        Returns:
            {
                "address": "0x...",
                "view_level": "pseudo_c",
                "current_line": {"content": "...", "address": "0x..."},
                "lines_before": [{"content": "...", "address": "0x..."}, ...],
                "lines_after": [{"content": "...", "address": "0x..."}, ...],
                "function_context": {...}
            }
        """
        if not self._binary_view:
            return {"error": "No binary view available"}

        try:
            target_addr = f"0x{address:x}"

            # For HLIL and pseudo-C views, use direct HLIL instruction matching
            # This is more reliable than trying to match addresses in formatted output
            if view_level in [ViewLevel.HLIL, ViewLevel.PSEUDO_C, ViewLevel.PSEUDO_RUST, ViewLevel.PSEUDO_PYTHON]:
                return self._get_hlil_line_with_context(address, view_level, context_lines)

            # For ASM view, use direct instruction access
            if view_level == ViewLevel.ASM:
                return self._get_asm_line_with_context(address, context_lines)

            # For other IL levels (LLIL, MLIL), use the formatted code approach
            code_data = self.get_code_at_level(address, view_level, context_lines=context_lines)

            if code_data.get("error"):
                return {"error": code_data.get("error")}

            lines = code_data.get("lines", [])
            if not lines:
                return {"error": "No lines found at address"}

            # Find the index of the current line (best match by address)
            current_index = -1

            # First pass: exact address match
            for i, line in enumerate(lines):
                if isinstance(line, dict):
                    line_addr = line.get("address", "")
                    if line_addr == target_addr:
                        current_index = i
                        break

            # Second pass: find line marked as current
            if current_index == -1:
                for i, line in enumerate(lines):
                    if isinstance(line, dict) and line.get("is_current", False):
                        current_index = i
                        break

            # Third pass: find closest address that's <= target
            if current_index == -1:
                best_addr = None
                for i, line in enumerate(lines):
                    if isinstance(line, dict):
                        line_addr_str = line.get("address", "")
                        if line_addr_str and line_addr_str.startswith("0x"):
                            try:
                                line_addr = int(line_addr_str, 16)
                                if line_addr <= address:
                                    if best_addr is None or line_addr > best_addr:
                                        best_addr = line_addr
                                        current_index = i
                            except ValueError:
                                continue

            # Fallback to first line if no match found
            if current_index == -1:
                current_index = 0

            # Extract current line
            current_line = lines[current_index] if current_index < len(lines) else {"content": "", "address": target_addr}

            # Extract context before
            start_index = max(0, current_index - context_lines)
            lines_before = lines[start_index:current_index]

            # Extract context after
            end_index = min(len(lines), current_index + context_lines + 1)
            lines_after = lines[current_index + 1:end_index]

            return {
                "address": target_addr,
                "view_level": view_level.value,
                "current_line": current_line,
                "lines_before": lines_before,
                "lines_after": lines_after,
                "function_context": self._get_function_context(address)
            }

        except Exception as e:
            return {"error": f"Failed to get line with context: {str(e)}"}

    def _get_hlil_line_with_context(self, address: int, view_level: ViewLevel, context_lines: int = 5) -> Dict[str, Any]:
        """Get HLIL/Pseudo-C line at address with context by directly querying HLIL instructions."""
        target_addr = f"0x{address:x}"

        functions = self._binary_view.get_functions_containing(address)
        if not functions:
            return {"error": "No function found at address"}

        function = functions[0]
        if not function.hlil:
            return {"error": "No HLIL available for function"}

        # Collect all HLIL instructions with their addresses
        hlil_instructions = []
        for instr in function.hlil.instructions:
            if view_level == ViewLevel.HLIL:
                content = f"  {str(instr)};"
            else:
                content = f"    {str(instr)};"

            hlil_instructions.append({
                "address": f"0x{instr.address:x}",
                "content": content,
                "instr_index": instr.instr_index
            })

        if not hlil_instructions:
            return {"error": "No HLIL instructions found"}

        # Find the instruction at or closest to the target address
        current_index = -1
        best_distance = float('inf')

        for i, instr in enumerate(hlil_instructions):
            instr_addr = int(instr["address"], 16)
            distance = abs(instr_addr - address)

            # Exact match
            if instr_addr == address:
                current_index = i
                break

            # Find closest instruction (prefer instructions before the target)
            if distance < best_distance:
                best_distance = distance
                current_index = i

        if current_index == -1:
            current_index = 0

        # Mark current line
        current_line = hlil_instructions[current_index].copy()
        current_line["is_current"] = True

        # Extract context before
        start_index = max(0, current_index - context_lines)
        lines_before = hlil_instructions[start_index:current_index]

        # Extract context after
        end_index = min(len(hlil_instructions), current_index + context_lines + 1)
        lines_after = hlil_instructions[current_index + 1:end_index]

        return {
            "address": target_addr,
            "view_level": view_level.value,
            "current_line": current_line,
            "lines_before": lines_before,
            "lines_after": lines_after,
            "function_context": self._get_function_context(address)
        }

    def _get_asm_line_with_context(self, address: int, context_lines: int = 5) -> Dict[str, Any]:
        """Get assembly line at address with context by directly querying instructions."""
        target_addr = f"0x{address:x}"

        functions = self._binary_view.get_functions_containing(address)
        if not functions:
            # Not in a function, just get the single instruction
            length = self._binary_view.get_instruction_length(address)
            if length and length > 0:
                instruction_text = self._binary_view.get_disassembly(address)
                bytes_data = self._binary_view.read(address, length)
                return {
                    "address": target_addr,
                    "view_level": "assembly",
                    "current_line": {
                        "address": target_addr,
                        "content": instruction_text or "???",
                        "is_current": True,
                        "bytes": bytes_data.hex() if bytes_data else ""
                    },
                    "lines_before": [],
                    "lines_after": [],
                    "function_context": None
                }
            return {"error": "No instruction found at address"}

        function = functions[0]

        # Collect all instructions in the function
        asm_instructions = []
        current_addr = function.start
        while current_addr < function.start + function.total_bytes:
            length = self._binary_view.get_instruction_length(current_addr)
            if not length or length == 0:
                break

            instruction_text = self._binary_view.get_disassembly(current_addr)
            bytes_data = self._binary_view.read(current_addr, length)

            asm_instructions.append({
                "address": f"0x{current_addr:x}",
                "content": instruction_text or "???",
                "bytes": bytes_data.hex() if bytes_data else ""
            })

            current_addr += length

        if not asm_instructions:
            return {"error": "No assembly instructions found"}

        # Find the instruction at the target address
        current_index = -1
        for i, instr in enumerate(asm_instructions):
            if instr["address"] == target_addr:
                current_index = i
                break

        # If exact match not found, find closest
        if current_index == -1:
            best_distance = float('inf')
            for i, instr in enumerate(asm_instructions):
                instr_addr = int(instr["address"], 16)
                distance = abs(instr_addr - address)
                if distance < best_distance:
                    best_distance = distance
                    current_index = i

        if current_index == -1:
            current_index = 0

        # Mark current line
        current_line = asm_instructions[current_index].copy()
        current_line["is_current"] = True

        # Extract context before
        start_index = max(0, current_index - context_lines)
        lines_before = asm_instructions[start_index:current_index]

        # Extract context after
        end_index = min(len(asm_instructions), current_index + context_lines + 1)
        lines_after = asm_instructions[current_index + 1:end_index]

        return {
            "address": target_addr,
            "view_level": "assembly",
            "current_line": current_line,
            "lines_before": lines_before,
            "lines_after": lines_after,
            "function_context": self._get_function_context(address)
        }

    def get_line_context(self, address: int, view_level: ViewLevel) -> Dict[str, Any]:
        """Get specific line context at cursor position"""
        if not self._binary_view:
            return {"error": "No binary view available"}
        
        try:
            # Get the instruction/line directly at the address
            current_line = None
            
            if view_level == ViewLevel.ASM:
                # For assembly, get the specific instruction at this address
                length = self._binary_view.get_instruction_length(address)
                if length and length > 0:
                    instruction_text = self._binary_view.get_disassembly(address)
                    if instruction_text:
                        bytes_data = self._binary_view.read(address, length)
                        current_line = {
                            "address": f"0x{address:x}",
                            "content": instruction_text,
                            "is_current": True,
                            "bytes": bytes_data.hex() if bytes_data else ""
                        }
            else:
                # For IL levels and pseudo-C, we need to be smarter about finding the right line
                if view_level in [ViewLevel.HLIL, ViewLevel.PSEUDO_C, ViewLevel.PSEUDO_RUST, ViewLevel.PSEUDO_PYTHON]:
                    # For HLIL and pseudo-C, get the HLIL instruction at this address directly
                    # Since these views often don't have reliable address matching in formatted output
                    function = self._binary_view.get_functions_containing(address)
                    if function:
                        function = function[0]
                        if function.hlil:
                            # Find the HLIL instruction at this address
                            for instr in function.hlil.instructions:
                                if instr.address == address:
                                    if view_level == ViewLevel.HLIL:
                                        content = f"  {str(instr)};"  # HLIL formatting
                                    else:
                                        content = f"    {str(instr)};"  # Pseudo-C formatting
                                    
                                    current_line = {
                                        "address": f"0x{address:x}",
                                        "content": content,
                                        "is_current": True,
                                    }
                                    break
                else:
                    # For IL levels, get code at level and find matching address
                    code_result = self.get_code_at_level(address, view_level, context_lines=2)
                    if not code_result.get("error") and code_result.get("lines"):
                        # Find the line at exact address or closest
                        for line in code_result["lines"]:
                            if line.get("is_current", False):
                                current_line = line
                                break
                        
                        # If no exact match, find closest address
                        if not current_line:
                            target_addr = f"0x{address:x}"
                            for line in code_result["lines"]:
                                if line.get("address") == target_addr:
                                    current_line = line
                                    break
                            
                            # Ultimate fallback
                            if not current_line and code_result["lines"]:
                                current_line = code_result["lines"][0]
            
            if not current_line:
                current_line = {
                    "address": f"0x{address:x}",
                    "content": "No instruction found at address",
                    "error": "Could not retrieve instruction"
                }
            
            return {
                "address": f"0x{address:x}",
                "view_level": view_level.value,
                "line": current_line,
                "context": self._get_function_context(address)
            }
            
        except Exception as e:
            return {
                "address": f"0x{address:x}",
                "view_level": view_level.value,
                "error": f"Failed to get line context: {str(e)}"
            }
    
    def get_triage_metadata(self) -> Dict[str, Any]:
        """Get comprehensive triage information about the binary"""
        if not self._binary_view:
            return {"error": "No binary view available"}
        
        metadata = {
            "basic_info": self._get_binary_info(),
            "imports": self._get_imports(),
            "exports": self._get_exports(),
            "strings": self._get_interesting_strings(),
            "entry_points": self._get_entry_points(),
            "security_features": self._get_security_features(),
        }
        
        return metadata
    
    def _get_imports(self) -> List[Dict[str, Any]]:
        """Get imported functions and libraries"""
        imports = []
        
        for symbol in self._binary_view.get_symbols():
            if symbol.type == bn.SymbolType.ImportedFunctionSymbol:
                imports.append({
                    "name": symbol.full_name,
                    "address": f"0x{symbol.address:x}",
                    "namespace": symbol.namespace.name if symbol.namespace else None,
                })
        
        return imports[:50]  # Limit for performance
    
    def _get_exports(self) -> List[Dict[str, Any]]:
        """Get exported functions"""
        exports = []
        
        for symbol in self._binary_view.get_symbols():
            if symbol.type == bn.SymbolType.FunctionSymbol:
                exports.append({
                    "name": symbol.full_name,
                    "address": f"0x{symbol.address:x}",
                    "namespace": symbol.namespace.name if symbol.namespace else None,
                })
        
        return exports[:50]  # Limit for performance
    
    def _get_interesting_strings(self) -> List[Dict[str, Any]]:
        """Get potentially interesting strings from the binary"""
        strings = []
        
        try:
            for string_ref in self._binary_view.strings:
                if len(string_ref.value) > 4:  # Filter short strings
                    strings.append({
                        "value": string_ref.value,
                        "address": f"0x{string_ref.start:x}",
                        "length": len(string_ref.value),
                        "type": str(string_ref.string_type)
                    })
        except:
            pass  # String analysis might not be available
        
        return strings[:100]  # Limit for performance
    
    def _get_entry_points(self) -> List[Dict[str, Any]]:
        """Get all entry points in the binary"""
        entries = []
        
        if self._binary_view.entry_point:
            entries.append({
                "name": "main_entry",
                "address": f"0x{self._binary_view.entry_point:x}",
                "type": "primary"
            })
        
        # Add other potential entry points
        for function in self._binary_view.functions:
            if function.name.lower() in ['main', '_main', 'start', '_start', 'wmain']:
                entries.append({
                    "name": function.name,
                    "address": f"0x{function.start:x}",
                    "type": "function_entry"
                })
        
        return entries
    
    def _get_security_features(self) -> Dict[str, Any]:
        """Analyze security features and mitigations"""
        features = {
            "nx_bit": False,
            "stack_canaries": False,
            "aslr": False,
            "pie": False,
            "relro": False,
            "stripped": True,  # Assume stripped unless symbols found
        }
        
        # Check for symbols (indicates not stripped)
        if len(self._binary_view.get_symbols()) > 10:
            features["stripped"] = False
        
        # Basic heuristics for security features
        # (This would need platform-specific implementation for accuracy)
        
        return features