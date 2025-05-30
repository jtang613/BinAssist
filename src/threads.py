from binaryninja.settings import Settings
from binaryninja import log
from PySide6 import QtCore

import re
import json
import random
import string

class StreamingThread(QtCore.QThread):
    """
    A thread for managing streaming API calls to an OpenAI model, specifically designed to handle long-running 
    requests that stream data back as they process. This class handles the setup and execution of these requests 
    and signals the main application thread upon updates or completion.
    
    Supports tool execution workflow with MCP integration.
    """

    update_response = QtCore.Signal(dict)
    streaming_finished = QtCore.Signal()

    def __init__(self, provider, query: str, system: str, tools=None, completion_callback=None, mcp_service=None) -> None:
        """
        Initializes the thread with the necessary parameters for making a streaming API call.

        Parameters:
            provider: The API provider instance to use for making calls.
            query (str): The user's query to be processed.
            system (str): System-level instructions or context for the API call.
            tools (list): A list of tools that the LLM can call during the response.
            mcp_service: Shared MCP service instance for tool execution.
        """
        super().__init__()
        
        # Setup logging
        log.log_info(f"[BinAssist] Initializing StreamingThread for provider: {provider.config.name}")
        
        try:
            self.settings = Settings()
            log.log_debug("[BinAssist] Settings initialized")
            
            self.provider = provider
            self.query = query
            self.system = system
            self.tools = tools or None
            self.completion_callback = completion_callback
            self.running = True
            self.mcp_service = mcp_service  # Shared MCP service instance
            
            # Tool execution state
            self.messages = []  # Conversation history for tool execution
            self.tool_execution_count = 0
            self.max_tool_executions = 10  # Safety limit
            
            log.log_debug(f"[BinAssist] Thread initialized - model: {provider.config.model}, max_tokens: {provider.config.max_tokens}")
            log.log_debug(f"[BinAssist] Query length: {len(query)}, system length: {len(system)}, has tools: {tools is not None}")
            log.log_info("[BinAssist] StreamingThread initialization completed")
            
        except Exception as e:
            log.log_error(f"[BinAssist] Error during StreamingThread initialization: {type(e).__name__}: {e}")
            raise

    def run(self) -> None:
        """
        Executes the streaming API call in a separate thread, processing responses as they come in and 
        signaling the main thread upon updates or when an error occurs.
        """
        log.log_info(f"[BinAssist] Starting thread execution for model: {self.provider.config.model}")
        
        try:
            log.log_debug("[BinAssist] Preparing API call parameters")
            
            log.log_debug(f"[BinAssist] Using provider: {type(self.provider).__name__}")
            log.log_debug(f"[BinAssist] Model: {self.provider.config.model}, max_tokens: {self.provider.config.max_tokens}")
            log.log_debug(f"[BinAssist] Has tools: {self.tools is not None}")
            
            log.log_info("[BinAssist] Making API call using provider - this is the critical point")
            log.log_debug(f"[BinAssist] Completion callback available: {self.completion_callback is not None}")
            
            # Use provider's streaming method
            from .core.models.chat_message import ChatMessage, MessageRole
            
            # Create messages for the provider
            messages = []
            if self.system:
                messages.append(ChatMessage(role=MessageRole.SYSTEM, content=self.system))
            messages.append(ChatMessage(role=MessageRole.USER, content=self.query))
            
            # ===================================================================
            # DEBUG LOGGING: Show raw LLM request details
            # ===================================================================
            log.log_info("=" * 60)
            log.log_info("[BinAssist] 🚀 RAW LLM REQUEST DEBUG INFO")
            log.log_info("=" * 60)
            
            log.log_info(f"[BinAssist] 📝 Provider: {type(self.provider).__name__}")
            log.log_info(f"[BinAssist] 🤖 Model: {self.provider.config.model}")
            log.log_info(f"[BinAssist] 💬 Message count: {len(messages)}")
            
            for i, msg in enumerate(messages):
                log.log_info(f"[BinAssist] 📨 Message {i+1}: {msg.role.value}")
                log.log_debug(f"[BinAssist] 📄 Content preview: {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
            
            if self.tools:
                log.log_info(f"[BinAssist] 🔧 Tools included: {len(self.tools)}")
                for i, tool in enumerate(self.tools):
                    tool_name = tool.get('function', {}).get('name', 'Unknown')
                    tool_desc = tool.get('function', {}).get('description', 'No description')
                    log.log_info(f"[BinAssist] 🔨 Tool {i+1}: {tool_name}")
                    log.log_debug(f"[BinAssist] 📋 Description: {tool_desc}")
                
                # Show full tools JSON for deep debugging
                import json
                try:
                    tools_json = json.dumps(self.tools, indent=2)
                    log.log_debug(f"[BinAssist] 🔧 FULL TOOLS JSON:\n{tools_json}")
                except Exception as e:
                    log.log_warn(f"[BinAssist] Failed to serialize tools JSON: {e}")
            else:
                log.log_info("[BinAssist] 🚫 No tools included in this request")
            
            log.log_info("=" * 60)
            
            # Use provider's streaming chat completion
            response_accumulator = ""
            
            def handle_streaming_response(accumulated_text):
                nonlocal response_accumulator
                response_accumulator = accumulated_text
                
                # Log response length periodically for debugging
                if len(accumulated_text) % 500 == 0 or '\n' in accumulated_text[-10:]:
                    log.log_debug(f"[BinAssist] 📥 Streaming response length: {len(accumulated_text)}")
                
                if self.running:
                    self.update_response.emit({"response": accumulated_text})
            
            def handle_function_call_response(tool_calls):
                nonlocal response_accumulator
                
                # ===================================================================
                # DEBUG LOGGING: Show raw LLM response with tool calls
                # ===================================================================
                log.log_info("=" * 60)
                log.log_info("[BinAssist] 🎯 RAW LLM RESPONSE DEBUG INFO")
                log.log_info("=" * 60)
                
                log.log_info(f"[BinAssist] 🔧 Tool calls received: {len(tool_calls)}")
                
                # Log each tool call in detail
                for i, tool_call in enumerate(tool_calls):
                    log.log_info(f"[BinAssist] 🔨 Tool Call {i+1}:")
                    log.log_info(f"[BinAssist]   📝 ID: {getattr(tool_call, 'id', 'No ID')}")
                    log.log_info(f"[BinAssist]   🏷️  Name: {getattr(tool_call, 'name', 'No Name')}")
                    log.log_info(f"[BinAssist]   📋 Arguments: {getattr(tool_call, 'arguments', 'No Arguments')}")
                
                # Show raw tool calls object for deep debugging
                import json
                try:
                    # Try to serialize the tool calls for inspection
                    tool_calls_data = []
                    for tc in tool_calls:
                        if hasattr(tc, '__dict__'):
                            tool_calls_data.append(vars(tc))
                        else:
                            tool_calls_data.append(str(tc))
                    
                    tool_calls_json = json.dumps(tool_calls_data, indent=2, default=str)
                    log.log_debug(f"[BinAssist] 🔧 RAW TOOL CALLS JSON:\n{tool_calls_json}")
                except Exception as e:
                    log.log_warn(f"[BinAssist] Failed to serialize tool calls: {e}")
                    log.log_debug(f"[BinAssist] Tool calls raw: {tool_calls}")
                
                log.log_info("=" * 60)
                
                if not self.running:
                    log.log_error(f"[BinAssist] StreamingThread: Not running, skipping tool call processing")
                    return
                
                # Check tool execution limit
                if self.tool_execution_count >= self.max_tool_executions:
                    log.log_warn(f"[BinAssist] Maximum tool executions ({self.max_tool_executions}) reached, stopping")
                    self.update_response.emit({"response": f"⚠️ Maximum tool executions ({self.max_tool_executions}) reached. Stopping tool execution."})
                    handle_completion()
                    return
                
                self.tool_execution_count += 1
                log.log_info(f"[BinAssist] Tool execution round {self.tool_execution_count}/{self.max_tool_executions}")
                
                # Execute tools and continue conversation
                try:
                    tool_results = self._execute_tools_and_continue(tool_calls)
                    
                    # Update response accumulator
                    tool_summary = f"Executed {len(tool_calls)} tools in round {self.tool_execution_count}"
                    response_accumulator += f"\n\n{tool_summary}"
                    
                    # Continue conversation with tool results
                    log.log_info(f"[BinAssist] Continuing conversation with tool results")
                    
                except Exception as e:
                    log.log_error(f"[BinAssist] Error in tool execution workflow: {e}")
                    error_msg = f"❌ Tool execution failed: {str(e)}"
                    self.update_response.emit({"response": error_msg})
                    response_accumulator += f"\n\n{error_msg}"
                    handle_completion()
            
            def handle_completion():
                """Called when streaming/function calling is complete."""
                # ===================================================================
                # DEBUG LOGGING: Show completion details
                # ===================================================================
                log.log_info("=" * 60)
                log.log_info("[BinAssist] 🏁 LLM REQUEST COMPLETION")
                log.log_info("=" * 60)
                log.log_info(f"[BinAssist] 📊 Final response length: {len(response_accumulator)}")
                log.log_info(f"[BinAssist] 🔧 Tools were available: {self.tools is not None}")
                log.log_info(f"[BinAssist] 🎯 Response type: {'Tool calls' if 'tool calls' in response_accumulator.lower() else 'Text response'}")
                log.log_info("=" * 60)
                
                log.log_info("[BinAssist] Provider API call completed successfully - handle_completion called")
                if self.running and self.completion_callback:
                    log.log_debug("[BinAssist] Calling completion callback from handle_completion")
                    try:
                        self.completion_callback()
                        log.log_debug("[BinAssist] Completion callback executed successfully")
                    except Exception as e:
                        log.log_error(f"[BinAssist] Error in completion callback: {e}")
                elif not self.running:
                    log.log_debug("[BinAssist] Thread was stopped, not calling completion callback")
                else:
                    log.log_debug("[BinAssist] No completion callback provided")

            if self.tools:
                # Use function calling
                log.log_debug("[BinAssist] Calling stream_function_call with completion handler")
                self.provider.stream_function_call(messages, self.tools, handle_function_call_response, handle_completion)
                log.log_debug("[BinAssist] stream_function_call returned")
            else:
                # Use regular streaming  
                log.log_debug("[BinAssist] Calling stream_chat_completion")
                self.provider.stream_chat_completion(messages, handle_streaming_response)
                log.log_debug("[BinAssist] stream_chat_completion returned")
                # For regular streaming, call completion handler after streaming finishes
                log.log_debug("[BinAssist] Calling handle_completion for regular streaming")
                handle_completion()
            
            if not self.running:  # Check before processing response
                log.log_debug("[BinAssist] Thread was stopped, returning early")
                return

            log.log_debug("[BinAssist] Provider API call method returned - response handled via callback")
            log.log_debug(f"[BinAssist] Final response accumulated: {len(response_accumulator)} characters")
                
        except Exception as e:
            error_msg = f"CRITICAL ERROR in StreamingThread.run(): {type(e).__name__}: {e}"
            log.log_error(f"[BinAssist] {error_msg}")
            
            # Try to emit an error response if possible
            try:
                self.update_response.emit({"response": f"Error: {str(e)}"})
            except Exception as emit_error:
                log.log_error(f"[BinAssist] Failed to emit error response: {emit_error}")
            
            # Re-raise the exception so it can be caught by higher levels
            raise

    def stop(self):
        """
        Stops the thread by setting the running flag to False and calling the built-in quit and terminate methods.
        """
        self.running = False  # Signal to stop processing the API response
        self.quit()  # Graceful stop (it allows the thread to clean up)
        self.terminate()  # Forcefully kill the thread if it doesn't stop immediately
        self.wait()  # Ensure the thread has completely stopped


    def _generate_random_string(self, length=8):
        """
        Generate a random alphanumeric string of specified length.

        Args:
            length (int, optional): Length of the generated string. Defaults to 8.

        Returns:
            str: Random alphanumeric string.
        """
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

    def _execute_tools_and_continue(self, tool_calls):
        """
        Execute tool calls via MCP service and continue the conversation.
        
        Args:
            tool_calls: List of tool calls from the LLM
            
        Returns:
            List of tool execution results
        """
        log.log_info(f"[BinAssist] 🔧 Executing {len(tool_calls)} tool calls")
        
        tool_results = []
        
        # Use the shared MCP service instance passed to the thread
        if not self.mcp_service:
            log.log_error("[BinAssist] No MCP service provided to thread, cannot execute tools")
            raise Exception("No MCP service available for tool execution")
        
        mcp_service = self.mcp_service
        
        # Execute each tool call
        for i, tool_call in enumerate(tool_calls):
            try:
                tool_name = getattr(tool_call, 'name', 'Unknown')
                tool_args = getattr(tool_call, 'arguments', {})
                tool_id = getattr(tool_call, 'id', f'call_{i}')
                
                log.log_info(f"[BinAssist] 🔨 Executing tool {i+1}/{len(tool_calls)}: {tool_name}")
                
                # Log tool execution in UI
                tool_log_msg = f"🔧 Executing: {tool_name}({json.dumps(tool_args, indent=2)})"
                self.update_response.emit({"response": tool_log_msg})
                
                # Parse arguments if they're a string
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        log.log_warn(f"[BinAssist] Failed to parse tool arguments as JSON: {tool_args}")
                        tool_args = {}
                
                # Execute tool via MCP service
                result = mcp_service.execute_tool_sync(tool_name, tool_args)
                
                log.log_info(f"[BinAssist] ✅ Tool {tool_name} executed successfully")
                
                # Log result in UI
                result_preview = str(result)[:200] + ("..." if len(str(result)) > 200 else "")
                result_log_msg = f"✅ Result: {result_preview}"
                self.update_response.emit({"response": result_log_msg})
                
                tool_results.append({
                    'id': tool_id,
                    'name': tool_name,
                    'result': result
                })
                
            except Exception as e:
                log.log_error(f"[BinAssist] ❌ Tool execution failed for {tool_name}: {e}")
                
                error_msg = f"❌ Error in {tool_name}: {str(e)}"
                self.update_response.emit({"response": error_msg})
                
                tool_results.append({
                    'id': getattr(tool_call, 'id', f'call_{i}'),
                    'name': tool_name,
                    'result': f"Error: {str(e)}"
                })
        
        # Build tool result messages for LLM
        tool_messages = []
        for result in tool_results:
            tool_message = {
                "role": "tool",
                "tool_call_id": result['id'],
                "content": str(result['result'])
            }
            tool_messages.append(tool_message)
        
        # Continue conversation with tool results
        self._continue_conversation_with_tool_results(tool_messages)
        
        return tool_results
    
    def _continue_conversation_with_tool_results(self, tool_messages):
        """
        Continue the conversation by sending tool results back to the LLM.
        
        Args:
            tool_messages: List of tool result messages to send to LLM
        """
        try:
            log.log_info(f"[BinAssist] 🔄 Continuing conversation with {len(tool_messages)} tool results")
            
            # Build conversation history including tool results
            from .core.models.chat_message import ChatMessage, MessageRole
            
            # Add system message if not already present
            if not self.messages:
                if self.system:
                    self.messages.append(ChatMessage(role=MessageRole.SYSTEM, content=self.system))
                self.messages.append(ChatMessage(role=MessageRole.USER, content=self.query))
            
            # Add tool result messages
            for tool_msg in tool_messages:
                # Convert to our ChatMessage format
                tool_chat_msg = ChatMessage(
                    role=MessageRole.TOOL,
                    content=tool_msg['content'],
                    tool_call_id=tool_msg.get('tool_call_id')
                )
                self.messages.append(tool_chat_msg)
            
            # Continue with streaming function call
            log.log_info("[BinAssist] 🚀 Restarting LLM conversation with tool results")
            
            # Use the same handlers for recursive tool execution
            def continue_function_call_response(new_tool_calls):
                # Recursive call - handle more tool calls if needed
                self._handle_function_call_response_recursive(new_tool_calls)
            
            def continue_completion():
                self._handle_completion_recursive()
            
            # Call provider with updated message history
            self.provider.stream_function_call(
                self.messages, 
                self.tools, 
                continue_function_call_response, 
                continue_completion
            )
            
        except Exception as e:
            log.log_error(f"[BinAssist] Error continuing conversation: {e}")
            error_msg = f"❌ Failed to continue conversation: {str(e)}"
            self.update_response.emit({"response": error_msg})
            self._handle_completion_recursive()

    def _extract_json_objects(self, text):
        # Regular expression to match JSON objects
        json_pattern = r'\s*({\s*(?:"[^"]*"\s*:\s*(?:"[^"]*"|\{[^}]*\}|\[[^\]]*\]|null|true|false|\d+(?:\.\d+)?)\s*,?\s*)*})\s*'
        
        text = text.replace("'",'"')

        # Find all matches in the text
        json_matches = re.findall(json_pattern, text, re.DOTALL)
        
        # Parse each match into a Python object
        json_objects = []
        for match in json_matches:
            try:
                json_obj = json.loads(match)
                json_objects.append(json_obj)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON: {match}")
        
        return json_objects
    
    def _handle_function_call_response_recursive(self, tool_calls):
        """Handle function call response recursively for continued tool execution."""
        # Check tool execution limit
        if self.tool_execution_count >= self.max_tool_executions:
            log.log_warn(f"[BinAssist] Maximum tool executions ({self.max_tool_executions}) reached, stopping")
            self.update_response.emit({"response": f"⚠️ Maximum tool executions ({self.max_tool_executions}) reached. Stopping tool execution."})
            self._handle_completion_recursive()
            return
        
        self.tool_execution_count += 1
        log.log_info(f"[BinAssist] Recursive tool execution round {self.tool_execution_count}/{self.max_tool_executions}")
        
        # Execute tools and continue conversation
        try:
            self._execute_tools_and_continue(tool_calls)
        except Exception as e:
            log.log_error(f"[BinAssist] Error in recursive tool execution: {e}")
            error_msg = f"❌ Recursive tool execution failed: {str(e)}"
            self.update_response.emit({"response": error_msg})
            self._handle_completion_recursive()
    
    def _handle_completion_recursive(self):
        """Handle completion for recursive tool execution."""
        log.log_info("[BinAssist] 🏁 Recursive tool execution completed")
        
        # Call the original completion callback if available
        if self.running and self.completion_callback:
            try:
                self.completion_callback()
            except Exception as e:
                log.log_error(f"[BinAssist] Error in recursive completion callback: {e}")
        
        log.log_info("[BinAssist] Tool execution workflow completed")
