#!/usr/bin/env python3

"""
Actions Service for BinAssist Actions Tab

Service for managing built-in Binary Ninja analysis actions.
Follows the same patterns as other BinAssist services.
"""

import time
from typing import List, Dict, Any, Optional
from .models.action_models import ActionType, ActionProposal, ActionResult
from .binary_context_service import BinaryContextService

# Binary Ninja imports
try:
    import binaryninja as bn
    from binaryninja import BinaryView, Function
    log = bn.log.Logger(0, "BinAssist")
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


class ActionsService:
    """Service for managing built-in Binary Ninja analysis actions"""
    
    def __init__(self, context_service: BinaryContextService):
        self.context_service = context_service
        self._available_actions = self._create_available_actions()
        log.log_info("ActionsService initialized")
    
    def _create_available_actions(self) -> List[Dict[str, Any]]:
        """Create the list of available actions"""
        return [
            {
                'name': 'Rename Function',
                'description': 'Suggest a better name for the current function',
                'action_type': ActionType.RENAME_FUNCTION
            },
            {
                'name': 'Rename Variable', 
                'description': 'Suggest meaningful names for variables',
                'action_type': ActionType.RENAME_VARIABLE
            },
            {
                'name': 'Retype Variable',
                'description': 'Suggest better types for variables',
                'action_type': ActionType.RETYPE_VARIABLE
            },
            {
                'name': 'Auto Create Struct',
                'description': 'Create structures from offset patterns',
                'action_type': ActionType.AUTO_CREATE_STRUCT
            }
        ]
    
    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Get all available action definitions"""
        return self._available_actions.copy()
    
    def get_function_context(self) -> Optional[Dict[str, Any]]:
        """Get current function context for LLM analysis"""
        try:
            context = self.context_service.get_current_context()
            if context.get("error"):
                log.log_error(f"Cannot get function context: {context['error']}")
                return None
            
            return context
            
        except Exception as e:
            log.log_error(f"Error getting function context: {e}")
            return None
    
    def apply_action(self, proposal: ActionProposal) -> ActionResult:
        """Apply an action proposal to the binary"""
        try:
            # Get current context and function
            context = self.context_service.get_current_context()
            if context.get("error"):
                return ActionResult(
                    success=False,
                    message="Cannot apply action: no valid context",
                    action_type=proposal.action_type,
                    target=proposal.target,
                    error=context["error"]
                )
            
            binary_view = self.context_service._binary_view
            function_address = context["offset"]
            
            # Apply the specific action
            if proposal.action_type == ActionType.RENAME_FUNCTION:
                return self._apply_rename_function(binary_view, function_address, proposal)
            elif proposal.action_type == ActionType.RENAME_VARIABLE:
                return self._apply_rename_variable(binary_view, function_address, proposal)
            elif proposal.action_type == ActionType.RETYPE_VARIABLE:
                return self._apply_retype_variable(binary_view, function_address, proposal)
            elif proposal.action_type == ActionType.AUTO_CREATE_STRUCT:
                return self._apply_auto_create_struct(binary_view, function_address, proposal)
            else:
                return ActionResult(
                    success=False,
                    message=f"Unknown action type: {proposal.action_type}",
                    action_type=proposal.action_type,
                    target=proposal.target,
                    error="Unknown action type"
                )
                
        except Exception as e:
            log.log_error(f"Error applying action: {e}")
            return ActionResult(
                success=False,
                message=f"Failed to apply action: {str(e)}",
                action_type=proposal.action_type,
                target=proposal.target,
                error=str(e)
            )
    
    # Helper methods for LLM-driven actions
    def validate_function_name(self, current_name: str, suggested_name: str) -> bool:
        """Validate that a function rename is reasonable"""
        try:
            # Basic validation rules
            if not suggested_name or len(suggested_name) < 3:
                return False
            
            # Check for valid C identifier
            if not suggested_name.replace('_', '').replace('-', '').isalnum():
                return False
            
            # Don't rename to same name
            if current_name == suggested_name:
                return False
            
            return True
            
        except Exception:
            return False
    
    def validate_variable_name(self, current_name: str, suggested_name: str) -> bool:
        """Validate that a variable rename is reasonable"""
        try:
            # Basic validation rules
            if not suggested_name or len(suggested_name) < 2:
                return False
            
            # Check for valid C identifier
            if not suggested_name.replace('_', '').isalnum():
                return False
            
            # Don't rename to same name
            if current_name == suggested_name:
                return False
            
            return True
            
        except Exception:
            return False
    
    def validate_variable_type(self, current_type: str, suggested_type: str) -> bool:
        """Validate that a variable retype is reasonable"""
        try:
            # Basic validation
            if not suggested_type or len(suggested_type) < 3:
                return False
            
            # Don't retype to same type
            if current_type == suggested_type:
                return False
            
            # Could add more sophisticated type validation here
            return True
            
        except Exception:
            return False
    
    # Action application methods
    def _apply_rename_function(self, binary_view: BinaryView, function_address: int, proposal: ActionProposal) -> ActionResult:
        """Apply function rename"""
        try:
            functions = binary_view.get_functions_containing(function_address)
            if not functions:
                return ActionResult(
                    success=False,
                    message=f"No function found at address {hex(function_address)}",
                    action_type=proposal.action_type,
                    target=proposal.target,
                    error="Function not found"
                )
            
            function = functions[0]
            old_name = function.name
            new_name = proposal.proposed_value
            
            # Apply the change directly without transactions to avoid analysis conflicts
            function.name = new_name
            
            # Force completion of any pending analysis
            binary_view.update_analysis_and_wait()
            
            return ActionResult(
                success=True,
                message=f"Renamed function from '{old_name}' to '{new_name}'",
                action_type=proposal.action_type,
                target=proposal.target,
                old_value=old_name,
                new_value=new_name
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to rename function: {str(e)}",
                action_type=proposal.action_type,
                target=proposal.target,
                error=str(e)
            )
    
    def _apply_rename_variable(self, binary_view: BinaryView, function_address: int, proposal: ActionProposal) -> ActionResult:
        """Apply variable rename"""
        try:
            functions = binary_view.get_functions_containing(function_address)
            if not functions:
                return ActionResult(
                    success=False,
                    message=f"No function found at address {hex(function_address)}",
                    action_type=proposal.action_type,
                    target=proposal.target,
                    error="Function not found"
                )
            
            function = functions[0]
            var_name = proposal.target
            new_name = proposal.proposed_value
            
            # Find the variable
            target_var = None
            for var in function.vars:
                if var.name == var_name:
                    target_var = var
                    break
            
            if not target_var:
                return ActionResult(
                    success=False,
                    message=f"Variable '{var_name}' not found in function",
                    action_type=proposal.action_type,
                    target=proposal.target,
                    error="Variable not found"
                )
            
            old_name = target_var.name
            
            # Apply the change directly without transactions to avoid analysis conflicts
            target_var.name = new_name
            
            # Force completion of any pending analysis
            binary_view.update_analysis_and_wait()
            
            return ActionResult(
                success=True,
                message=f"Renamed variable from '{old_name}' to '{new_name}'",
                action_type=proposal.action_type,
                target=proposal.target,
                old_value=old_name,
                new_value=new_name
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to rename variable: {str(e)}",
                action_type=proposal.action_type,
                target=proposal.target,
                error=str(e)
            )
    
    def _apply_retype_variable(self, binary_view: BinaryView, function_address: int, proposal: ActionProposal) -> ActionResult:
        """Apply variable retype"""
        try:
            functions = binary_view.get_functions_containing(function_address)
            if not functions:
                return ActionResult(
                    success=False,
                    message=f"No function found at address {hex(function_address)}",
                    action_type=proposal.action_type,
                    target=proposal.target,
                    error="Function not found"
                )
            
            function = functions[0]
            var_name = proposal.target
            new_type_str = proposal.proposed_value
            
            # Find the variable
            target_var = None
            for var in function.vars:
                if var.name == var_name:
                    target_var = var
                    break
            
            if not target_var:
                return ActionResult(
                    success=False,
                    message=f"Variable '{var_name}' not found in function",
                    action_type=proposal.action_type,
                    target=proposal.target,
                    error="Variable not found"
                )
            
            old_type = str(target_var.type)
            
            # Parse the new type
            try:
                parsed_type = binary_view.parse_type_string(new_type_str)[0]
            except Exception as e:
                return ActionResult(
                    success=False,
                    message=f"Failed to parse type '{new_type_str}': {str(e)}",
                    action_type=proposal.action_type,
                    target=proposal.target,
                    error=f"Type parsing failed: {str(e)}"
                )
            
            # Apply the change directly without transactions to avoid analysis conflicts
            target_var.type = parsed_type
            
            # Force completion of any pending analysis
            binary_view.update_analysis_and_wait()
            
            return ActionResult(
                success=True,
                message=f"Changed type of variable '{var_name}' from '{old_type}' to '{new_type_str}'",
                action_type=proposal.action_type,
                target=proposal.target,
                old_value=old_type,
                new_value=new_type_str
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to retype variable: {str(e)}",
                action_type=proposal.action_type,
                target=proposal.target,
                error=str(e)
            )
    
    def _apply_auto_create_struct(self, binary_view: BinaryView, function_address: int, proposal: ActionProposal) -> ActionResult:
        """Apply automatic struct creation"""
        # For now, return not implemented
        return ActionResult(
            success=False,
            message="Auto create struct not yet implemented",
            action_type=proposal.action_type,
            target=proposal.target,
            error="Not implemented"
        )
    
    # Binary Ninja integration helpers
    def get_current_function(self):
        """Get the current function object for analysis"""
        try:
            context = self.context_service.get_current_context()
            if context.get("error"):
                return None
            
            binary_view = self.context_service._binary_view
            function_address = context["offset"]
            functions = binary_view.get_functions_containing(function_address)
            
            return functions[0] if functions else None
            
        except Exception as e:
            log.log_error(f"Error getting current function: {e}")
            return None