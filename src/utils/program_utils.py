"""
Program Utilities for BinAssist

Provides utilities for program identification and hashing.
"""

import hashlib
from binaryninja import BinaryView, log
from typing import Optional


def get_program_hash(bv: BinaryView) -> Optional[str]:
    """
    Generate a consistent SHA256 hash for the program.
    
    This hash is used as a unique identifier for the program in the analysis database.
    The hash is based on the file content to ensure consistency across sessions.
    
    Args:
        bv: BinaryView instance
        
    Returns:
        SHA256 hash string, or None if generation fails
    """
    try:
        if not bv or not bv.file:
            log.log_error("[BinAssist] Cannot generate program hash: invalid BinaryView")
            return None
            
        # Method 1: Try to use the raw file data for most consistent hashing
        if bv.file.raw and hasattr(bv.file.raw, 'length') and bv.file.raw.length > 0:
            file_data = bv.file.raw.read(0, bv.file.raw.length)
            hash_obj = hashlib.sha256(file_data)
            program_hash = hash_obj.hexdigest()
            
            log.log_debug(f"[BinAssist] Generated program hash from raw data: {program_hash[:16]}...")
            return program_hash
            
        # Method 2: Fallback to using file path + basic metadata
        elif bv.file.filename:
            # Combine filename, file size, and entry point for uniqueness
            hash_input = bv.file.filename
            
            # Add file size if available
            if bv.file.raw and hasattr(bv.file.raw, 'length'):
                hash_input += str(bv.file.raw.length)
            
            # Add entry point if available
            if bv.entry_point:
                hash_input += str(bv.entry_point)
                
            # Add architecture info for additional uniqueness
            if bv.arch:
                hash_input += str(bv.arch.name)
                
            hash_obj = hashlib.sha256(hash_input.encode('utf-8'))
            program_hash = hash_obj.hexdigest()
            
            log.log_debug(f"[BinAssist] Generated program hash from metadata: {program_hash[:16]}...")
            log.log_debug(f"[BinAssist] Hash input: filename={bv.file.filename}, arch={bv.arch.name if bv.arch else 'unknown'}")
            return program_hash
            
        else:
            log.log_error("[BinAssist] Cannot generate program hash: no file data or filename available")
            return None
            
    except Exception as e:
        log.log_error(f"[BinAssist] Failed to generate program hash: {e}")
        return None


def get_function_start_address(bv, current_address: int) -> int:
    """
    Get the start address of the function containing the current address.
    
    Args:
        bv: Binary view
        current_address: Current cursor/line address
        
    Returns:
        Function start address, or current_address if no function found
    """
    if not bv:
        return current_address
        
    functions = bv.get_functions_containing(current_address)
    if functions:
        return functions[0].start
    return current_address


def get_function_address_string(address: int) -> str:
    """
    Convert a function address to a consistent string format.
    
    Args:
        address: Function address as integer
        
    Returns:
        Hexadecimal address string (e.g., "0x401000")
    """
    return f"0x{address:x}"


def validate_program_hash(program_hash: str) -> bool:
    """
    Validate that a program hash is in the expected format.
    
    Args:
        program_hash: Hash string to validate
        
    Returns:
        True if valid SHA256 hash format, False otherwise
    """
    if not program_hash or not isinstance(program_hash, str):
        return False
        
    # SHA256 hashes are 64 hexadecimal characters
    if len(program_hash) != 64:
        return False
        
    # Check if all characters are valid hexadecimal
    try:
        int(program_hash, 16)
        return True
    except ValueError:
        return False


def get_program_info(bv: BinaryView) -> dict:
    """
    Get basic program information for debugging and logging.
    
    Args:
        bv: BinaryView instance
        
    Returns:
        Dictionary containing program information
    """
    try:
        info = {
            'filename': bv.file.filename if bv.file else 'unknown',
            'arch': bv.arch.name if bv.arch else 'unknown',
            'platform': str(bv.platform) if bv.platform else 'unknown',
            'entry_point': f"0x{bv.entry_point:x}" if bv.entry_point else 'unknown',
            'length': bv.file.raw.length if bv.file and bv.file.raw and hasattr(bv.file.raw, 'length') else 0,
            'hash': get_program_hash(bv)
        }
        return info
    except Exception as e:
        log.log_error(f"[BinAssist] Failed to get program info: {e}")
        return {
            'filename': 'error',
            'arch': 'error', 
            'platform': 'error',
            'entry_point': 'error',
            'length': 0,
            'hash': None
        }