"""
Enhanced Query Handler with MCP and RAG Integration.

This module provides an enhanced query handler that can:
- Integrate MCP tools with LLM queries
- Handle tool execution workflows
- Manage conversation loops with tool calls
- Log tool executions to chat
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from binaryninja import log
from .llm_api import LlmApi
from .mcp_integration import McpIntegrationService, create_mcp_integration_for_plugin


class EnhancedQueryHandler:
    """
    Enhanced query handler that supports MCP tools, RAG, and conversation loops.
    """
    
    def __init__(self, plugin):
        """Initialize the enhanced query handler."""
        self.plugin = plugin
        self.settings = plugin.settings
        self.llm_api = plugin.LlmApi
        self.logger = logging.getLogger("binassist.enhanced_query")
        
        # MCP integration
        self.mcp_integration: Optional[McpIntegrationService] = None
        
        # Conversation state
        self.conversation_active = False
        self.tool_execution_logs: List[str] = []
        
    def initialize_mcp_if_needed(self) -> bool:
        """Initialize MCP integration if enabled and needed."""
        try:
            self.logger.info("Checking MCP tool configuration...")
            
            # Check if MCP tools are enabled
            try:
                use_mcp_tools = self.settings.get_boolean('use_mcp_tools')
                self.logger.info(f"MCP tools enabled: {use_mcp_tools}")
            except Exception as e:
                self.logger.warning(f"Failed to read MCP tools setting: {e}")
                use_mcp_tools = False
                
            if not use_mcp_tools:
                self.logger.info("MCP tools disabled, skipping initialization")
                return False
            
            # Create MCP integration if not already created
            if self.mcp_integration is None:
                self.logger.info("Creating MCP integration service...")
                self.logger.info("Calling create_mcp_integration_for_plugin()...")
                self.mcp_integration = create_mcp_integration_for_plugin(self.plugin)
                if self.mcp_integration:
                    self.logger.info("✅ MCP integration service created successfully")
                    self.logger.info(f"MCP integration type: {type(self.mcp_integration)}")
                else:
                    self.logger.error("❌ Failed to create MCP integration service - returned None")
                    self.logger.error("This means either MCP tools are disabled or no servers are configured")
            else:
                self.logger.debug("Using existing MCP integration service")
                
            # Check if MCP is available
            is_available = self.mcp_integration is not None and self.mcp_integration.is_available()
            self.logger.info(f"MCP integration available: {is_available}")
            
            if is_available:
                # Get available tools for logging
                try:
                    tools_dict = self.mcp_integration.get_available_tools()
                    tool_names = [tool.name for tool in tools_dict.values()]
                    self.logger.info(f"Available MCP tools: {tool_names}")
                except Exception as e:
                    self.logger.warning(f"Failed to get available tools: {e}")
                    
            return is_available
            
        except Exception as e:
            self.logger.error(f"Error initializing MCP: {e}")
            self.logger.exception("Full traceback for MCP initialization error:")
            return False
    
    def execute_enhanced_query(self, query: str, response_callback: Callable[[Dict[str, Any]], None]):
        """
        Execute an enhanced query with MCP and RAG support.
        
        Args:
            query: The user query
            response_callback: Callback for streaming responses
        """
        try:
            log.log_info("=== ENHANCED QUERY EXECUTION START ===")
            log.log_info(f"Query: {query}")
            self.logger.info("=== ENHANCED QUERY EXECUTION START ===")
            self.logger.info(f"Query: {query}")
            
            self.conversation_active = True
            self.tool_execution_logs.clear()
            
            # Initialize MCP if needed
            log.log_info("Initializing MCP integration...")
            self.logger.info("Initializing MCP integration...")
            mcp_available = self.initialize_mcp_if_needed()
            log.log_info(f"MCP available for this query: {mcp_available}")
            self.logger.info(f"MCP available for this query: {mcp_available}")
            
            # Start the conversation loop
            log.log_info("Starting conversation loop...")
            self.logger.info("Starting conversation loop...")
            self._start_conversation_loop(query, response_callback, mcp_available)
            log.log_info("✅ Enhanced query execution completed")
            self.logger.info("✅ Enhanced query execution completed")
            
        except Exception as e:
            self.logger.error(f"Error executing enhanced query: {e}")
            self.logger.exception("Full traceback for enhanced query execution error:")
            response_callback({
                "type": "error",
                "content": f"Error executing query: {str(e)}"
            })
    
    def _start_conversation_loop(self, initial_query: str, response_callback: Callable, mcp_available: bool):
        """Start the conversation loop with tool support."""
        try:
            log.log_info("=== STARTING CONVERSATION LOOP ===")
            self.logger.info("=== STARTING CONVERSATION LOOP ===")
            
            # Prepare the query with context
            log.log_info("Preparing query with context...")
            full_query = self._prepare_query_with_context(initial_query)
            log.log_info(f"Full query length: {len(full_query)}")
            
            # Get available tools for LLM
            tools = []
            if mcp_available:
                log.log_info("Getting MCP tools for LLM...")
                mcp_tools = self.mcp_integration.get_tools_for_llm()
                
                # TEMPORARY: Limit tools to avoid overwhelming the LLM
                MAX_TOOLS = 5
                if len(mcp_tools) > MAX_TOOLS:
                    mcp_tools = mcp_tools[:MAX_TOOLS]
                    log.log_info(f"⚠️ Limited to {MAX_TOOLS} tools (was {len(self.mcp_integration.get_tools_for_llm())})")
                
                tools.extend(mcp_tools)
                log.log_info(f"Added {len(mcp_tools)} MCP tools to query")
                self.logger.info(f"Added {len(mcp_tools)} MCP tools to query")
            
            # If we have tools, use function calling
            if tools:
                log.log_info(f"Executing query with {len(tools)} tools")
                self._execute_query_with_tools(full_query, tools, response_callback)
            else:
                log.log_info("Executing regular query without tools")
                # Regular query without tools
                self._execute_regular_query(full_query, response_callback)
                
        except Exception as e:
            self.logger.error(f"Error in conversation loop: {e}")
            response_callback({
                "type": "error", 
                "content": f"Error in conversation: {str(e)}"
            })
    
    def _prepare_query_with_context(self, query: str) -> str:
        """Prepare query with session context."""
        try:
            # Add session log for context
            context_parts = []
            
            # Add previous conversation
            for entry in self.plugin.session_log[-5:]:  # Last 5 entries
                context_parts.append(f"User: {entry.get('user', '')}")
                context_parts.append(f"Assistant: {entry.get('assistant', '')}")
            
            # Add tool execution logs if any
            if self.tool_execution_logs:
                context_parts.append("\nTool Executions in this session:")
                context_parts.extend(self.tool_execution_logs[-3:])  # Last 3 tool executions
            
            # Combine with current query
            if context_parts:
                full_query = "\n".join(context_parts) + f"\n\nUser: {query}"
            else:
                full_query = query
                
            return full_query
            
        except Exception as e:
            self.logger.error(f"Error preparing query context: {e}")
            return query
    
    def _execute_query_with_tools(self, query: str, tools: List[Dict], response_callback: Callable):
        """Execute query with tool calling support."""
        try:
            log.log_info(f"=== EXECUTING QUERY WITH {len(tools)} TOOLS ===")
            self.logger.info(f"=== EXECUTING QUERY WITH {len(tools)} TOOLS ===")
            
            # Log all tool names for debugging
            tool_names = [tool.get('function', {}).get('name', 'unknown') for tool in tools]
            log.log_info(f"Tool names being sent to LLM: {tool_names}")
            self.logger.info(f"Tool names being sent to LLM: {tool_names}")
            
            # Create a custom response handler that can handle tool calls
            def tool_aware_response_handler(response_data):
                try:
                    self.logger.info(f"=== RECEIVED RESPONSE FROM LLM ===")
                    self.logger.debug(f"Response data type: {type(response_data)}")
                    self.logger.debug(f"Response data: {response_data}")
                    
                    if isinstance(response_data, dict) and "response" in response_data:
                        response = response_data["response"]
                        
                        # Check if we got a complete OpenAI response object
                        if hasattr(response, 'choices') and response.choices:
                            choice = response.choices[0]
                            finish_reason = choice.finish_reason
                            self.logger.info(f"LLM response finish_reason: {finish_reason}")
                            
                            if finish_reason == 'tool_calls' and hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                                self.logger.info(f"LLM responded with {len(choice.message.tool_calls)} tool calls")
                                self._handle_tool_calls(choice.message.tool_calls, query, response_callback)
                            else:
                                # Regular text response or no tool calls
                                content = choice.message.content if choice.message.content else ""
                                self.logger.info(f"LLM responded with text (finish_reason: {finish_reason})")
                                self._handle_text_response(content, response_callback)
                        # Check if response contains tool calls (legacy format)
                        elif isinstance(response, list):  # Tool calls
                            self.logger.info(f"LLM responded with {len(response)} tool calls")
                            self._handle_tool_calls(response, query, response_callback)
                        else:  # Regular text response
                            self.logger.info("LLM responded with text (no tool calls)")
                            self._handle_text_response(response, response_callback)
                    else:
                        # Fallback for other response formats
                        self.logger.info("Using fallback response handling")
                        response_callback({
                            "type": "text",
                            "content": str(response_data)
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error in tool aware response handler: {e}")
                    self.logger.exception("Full traceback:")
                    response_callback({
                        "type": "error",
                        "content": f"Error processing response: {str(e)}"
                    })
            
            # Use direct LLM API call with tools
            log.log_info("Calling _execute_direct_llm_call...")
            self._execute_direct_llm_call(query, tools, tool_aware_response_handler)
            log.log_info("✅ _execute_direct_llm_call completed")
            
        except Exception as e:
            self.logger.error(f"Error executing query with tools: {e}")
            self.logger.exception("Full traceback:")
            self._execute_regular_query(query, response_callback)
    
    def _execute_direct_llm_call(self, query: str, tools: List[Dict], response_callback: Callable):
        """Execute a direct LLM API call with tools."""
        try:
            log.log_info("=== MAKING DIRECT LLM CALL WITH TOOLS ===")
            self.logger.info("=== MAKING DIRECT LLM CALL WITH TOOLS ===")
            
            # Get LLM configuration
            log.log_info("Getting LLM configuration...")
            api_provider = self.llm_api.get_active_provider()
            client = self.llm_api._create_client()
            
            model = api_provider.get('api__model', 'gpt-4o-mini')
            max_tokens = api_provider.get('api__max_tokens', 4096)
            
            log.log_info(f"Using model: {model}")
            log.log_info(f"Max tokens: {max_tokens}")
            log.log_info(f"Number of tools: {len(tools)}")
            self.logger.info(f"Using model: {model}")
            self.logger.info(f"Max tokens: {max_tokens}")
            self.logger.info(f"Number of tools: {len(tools)}")
            
            # Create Qt signal for response handling
            from PySide6 import QtCore
            
            class ResponseSignal(QtCore.QObject):
                response_ready = QtCore.Signal(dict)
            
            signal_obj = ResponseSignal()
            signal_obj.response_ready.connect(response_callback)
            
            # Use the LLM API's _start_thread method directly with tools
            log.log_info("Starting LLM thread with tools...")
            self.logger.info("Starting LLM thread with tools...")
            thread = self.llm_api._start_thread(
                client=client,
                model=model,
                max_tokens=max_tokens,
                query=query,
                system=self.llm_api.SYSTEM_PROMPT,
                signal=signal_obj.response_ready,
                tools=tools
            )
            
            log.log_info("LLM thread started successfully")
            self.logger.info("LLM thread started successfully")
            
        except Exception as e:
            self.logger.error(f"Error in direct LLM call: {e}")
            self.logger.exception("Full traceback:")
            response_callback({
                "type": "error",
                "content": f"Error calling LLM: {str(e)}"
            })
    
    def _execute_with_function_calling(self, query: str, tools: List[Dict], response_callback: Callable):
        """Execute using function calling approach."""
        try:
            # Create a mock address and binary view for the analyze_function call
            # This is a workaround since we're adapting the tool calling system
            
            # Get current binary view if available
            bv = self.plugin.bv
            addr = self.plugin.offset_addr
            
            if bv is None:
                # Fallback to regular query
                self._execute_regular_query(query, response_callback)
                return
            
            # Use the analyze_function method which supports tool calling
            def addr_to_text_func(bv, addr):
                return query  # Return our query as the "code" to analyze
            
            # Execute with tool calling
            self.llm_api.analyze_function(
                "custom_query",  # Custom action type
                bv, 
                addr,
                "query",  # bin_type
                "text",   # il_type
                addr_to_text_func,
                response_callback
            )
            
        except Exception as e:
            self.logger.error(f"Error in function calling execution: {e}")
            self._execute_regular_query(query, response_callback)
    
    def _execute_regular_query(self, query: str, response_callback: Callable):
        """Execute regular query without tools."""
        try:
            self.llm_api.query(query, response_callback)
        except Exception as e:
            self.logger.error(f"Error executing regular query: {e}")
            response_callback({
                "type": "error",
                "content": f"Error executing query: {str(e)}"
            })
    
    def _handle_tool_calls(self, tool_calls: List, original_query: str, response_callback: Callable):
        """Handle tool calls from LLM response."""
        try:
            self.logger.info(f"Handling {len(tool_calls)} tool calls")
            
            tool_results = []
            
            for tool_call in tool_calls:
                try:
                    # Extract tool call details
                    if hasattr(tool_call, 'function'):
                        tool_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)
                    else:
                        # Handle dict format
                        tool_name = tool_call.get('name', '')
                        arguments = tool_call.get('arguments', {})
                    
                    self.logger.info(f"Executing tool: {tool_name} with args: {arguments}")
                    
                    # Check if this is an MCP tool
                    if tool_name.startswith("mcp_") and self.mcp_integration:
                        result = self._execute_mcp_tool(tool_name, arguments)
                        tool_results.append(result)
                        
                        # Log tool execution for chat display
                        log_entry = self.mcp_integration.format_tool_execution_for_chat(result)
                        self.tool_execution_logs.append(log_entry)
                        
                        # Send tool execution log to UI
                        response_callback({
                            "type": "tool_log",
                            "content": log_entry
                        })
                    else:
                        # Handle native Binary Ninja tools if needed
                        result = {"success": False, "error": "Native tool execution not implemented in enhanced handler"}
                        tool_results.append(result)
                        
                except Exception as e:
                    self.logger.error(f"Error executing individual tool call: {e}")
                    tool_results.append({"success": False, "error": str(e)})
            
            # Continue conversation with tool results
            self._continue_conversation_with_tool_results(original_query, tool_results, response_callback)
            
        except Exception as e:
            self.logger.error(f"Error handling tool calls: {e}")
            response_callback({
                "type": "error",
                "content": f"Error handling tool calls: {str(e)}"
            })
    
    def _execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool call."""
        try:
            if not self.mcp_integration:
                return {"success": False, "error": "MCP integration not available"}
            
            return self.mcp_integration.execute_tool_call(tool_name, arguments)
            
        except Exception as e:
            self.logger.error(f"Error executing MCP tool {tool_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def _continue_conversation_with_tool_results(self, original_query: str, tool_results: List[Dict], response_callback: Callable):
        """Continue the conversation with tool results."""
        try:
            # Format tool results for LLM
            results_text = self._format_tool_results_for_llm(tool_results)
            
            # Create follow-up query
            follow_up_query = f"{original_query}\n\nTool execution results:\n{results_text}\n\nPlease provide a response based on these results."
            
            # Execute follow-up query (without tools to avoid infinite loops)
            self._execute_regular_query(follow_up_query, response_callback)
            
        except Exception as e:
            self.logger.error(f"Error continuing conversation with tool results: {e}")
            response_callback({
                "type": "error",
                "content": f"Error processing tool results: {str(e)}"
            })
    
    def _format_tool_results_for_llm(self, tool_results: List[Dict]) -> str:
        """Format tool results for LLM consumption."""
        try:
            formatted_results = []
            
            for i, result in enumerate(tool_results):
                if result.get("success"):
                    formatted_results.append(f"Tool {i+1}: SUCCESS - {result.get('result', '')}")
                else:
                    formatted_results.append(f"Tool {i+1}: FAILED - {result.get('error', 'Unknown error')}")
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            self.logger.error(f"Error formatting tool results: {e}")
            return "Error formatting tool results"
    
    def _handle_text_response(self, response: str, response_callback: Callable):
        """Handle regular text response."""
        try:
            response_callback({
                "type": "text",
                "content": response
            })
        except Exception as e:
            self.logger.error(f"Error handling text response: {e}")
    
    def stop_conversation(self):
        """Stop the active conversation."""
        try:
            self.conversation_active = False
            self.llm_api.stop_threads()
            self.logger.info("Conversation stopped")
        except Exception as e:
            self.logger.error(f"Error stopping conversation: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.mcp_integration:
                self.mcp_integration.cleanup()
                self.mcp_integration = None
            
            self.conversation_active = False
            self.tool_execution_logs.clear()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the enhanced query handler."""
        status = {
            "conversation_active": self.conversation_active,
            "mcp_available": self.mcp_integration is not None and self.mcp_integration.is_available(),
            "tool_execution_count": len(self.tool_execution_logs)
        }
        
        if self.mcp_integration:
            status["mcp_status"] = self.mcp_integration.get_status_summary()
        
        return status