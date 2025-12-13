#!/usr/bin/env python3

from typing import Optional, Dict, Any, List, Callable
import asyncio
import binaryninja as bn
from PySide6.QtCore import QDateTime, QThread, QTimer, Signal, Qt
from ..services.binary_context_service import BinaryContextService, ViewLevel
from ..services.llm_providers.provider_factory import LLMProviderFactory
from ..services.settings_service import SettingsService
from ..services.analysis_db_service import AnalysisDBService
from ..services.rag_service import rag_service
from ..services.models.rag_models import SearchRequest, SearchType
from ..services.mcp_client_service import MCPClientService
from ..services.models.mcp_models import MCPToolExecutionRequest
from ..services.mcp_connection_manager import MCPConnectionManager
from ..services.mcp_tool_orchestrator import MCPToolOrchestrator
from ..services.models.llm_models import ToolCall, ToolResult
from ..services.rlhf_service import rlhf_service
from ..services.models.rlhf_models import RLHFFeedbackEntry
from .chat_edit_manager import ChatEditManager
from ..services.debounced_renderer import DebouncedRenderer
from .react_thread import ReActOrchestratorThread
from ..services.models.react_models import ReActConfig, ReActResult, ReActStatus

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


class LLMQueryThread(QThread):
    """Thread for handling async LLM queries with streaming and tool call detection"""
    response_chunk = Signal(str)
    response_complete = Signal()
    response_error = Signal(str)
    tool_calls_detected = Signal(list)  # List[ToolCall]
    stop_reason_received = Signal(str)  # "stop", "tool_calls", etc.
    
    def __init__(self, messages: List[Dict[str, Any]], provider_config: dict, llm_factory, mcp_tools: List[Dict[str, Any]] = None, native_message_callback: Callable = None):
        super().__init__()
        self.messages = messages
        self.provider_config = provider_config
        self.llm_factory = llm_factory
        self.mcp_tools = mcp_tools or []
        self.native_message_callback = native_message_callback
        self.cancelled = False
    
    def cancel(self):
        """Cancel the running query"""
        self.cancelled = True
    
    def run(self):
        """Execute LLM query in background thread"""
        try:
            # Run async query in new event loop
            asyncio.run(self._async_query())
        except Exception as e:
            if not self.cancelled:  # Don't emit error if cancelled
                self.response_error.emit(str(e))
    
    async def _async_query(self):
        """Execute async LLM query"""
        try:
            # Check for cancellation before starting
            if self.cancelled:
                return
            
            # Import required models
            from ..services.models.llm_models import ChatRequest, ChatMessage, MessageRole
            
            # Create provider instance
            provider = self.llm_factory.create_provider(self.provider_config)
            
            # Check for cancellation after provider creation
            if self.cancelled:
                return
            
            # Convert messages to ChatMessage format while preserving all roles and structure
            chat_messages = []
            for msg in self.messages:
                if msg["role"] == "system":
                    chat_messages.append(ChatMessage(role=MessageRole.SYSTEM, content=msg["content"]))
                elif msg["role"] == "tool":
                    # Tool result message - preserve tool_call_id
                    chat_messages.append(ChatMessage(
                        role=MessageRole.TOOL,
                        content=msg["content"],
                        tool_call_id=msg.get("tool_call_id"),
                        name=msg.get("name")
                    ))
                else:
                    # User or assistant message
                    role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
                    content = msg["content"]
                    
                    # Handle tool calls for assistant messages
                    tool_calls = None
                    if msg.get("tool_calls"):
                        from ..services.models.llm_models import ToolCall
                        tool_calls = []
                        for tc in msg["tool_calls"]:
                            if tc.get("type") == "function" and "function" in tc:
                                # Parse OpenAI format
                                func_data = tc["function"]
                                arguments = func_data.get("arguments", "{}")
                                if isinstance(arguments, str):
                                    try:
                                        arguments = json.loads(arguments)
                                    except:
                                        arguments = {}
                                
                                tool_call = ToolCall(
                                    id=tc.get("id", ""),
                                    name=func_data.get("name", ""),
                                    arguments=arguments
                                )
                                tool_calls.append(tool_call)
                    
                    chat_messages.append(ChatMessage(
                        role=role,
                        content=content,
                        tool_calls=tool_calls
                    ))
            
            # Create chat request
            request = ChatRequest(
                messages=chat_messages, 
                model=self.provider_config.get('model', ''),
                stream=True,
                max_tokens=self.provider_config.get('max_tokens', 4096)
            )
            
            # Add tools if available and provider supports tools
            if self.mcp_tools and provider.supports_tools():
                request = provider.prepare_tool_enabled_request(request, self.mcp_tools)
            
            # DEBUG: Log the actual request messages being sent to LLM
            import json
            debug_messages = []
            for msg in request.messages:
                debug_msg = {
                    "role": msg.role.value,
                    "content": msg.content[:200] + "..." if len(str(msg.content)) > 200 else msg.content
                }
                if msg.tool_calls:
                    debug_msg["tool_calls"] = [
                        {"id": tc.id, "name": tc.name, "arguments": str(tc.arguments)[:100] + "..." if len(str(tc.arguments)) > 100 else tc.arguments}
                        for tc in msg.tool_calls
                    ]
                if msg.tool_call_id:
                    debug_msg["tool_call_id"] = msg.tool_call_id
                debug_messages.append(debug_msg)
            
            log.log_info(f"🔍 LLM REQUEST DEBUG - Initial Query ({len(request.messages)} messages):")
            log.log_info(json.dumps(debug_messages, indent=2))
            
            # Execute streaming query with cancellation checks and tool call detection
            accumulated_tool_calls = []
            final_stop_reason = "stop"
            
            async for response in provider.chat_completion_stream(request, self.native_message_callback):
                if self.cancelled:
                    break
                
                # Handle content chunks
                if response.content:
                    self.response_chunk.emit(response.content)
                
                # Handle tool calls
                if response.tool_calls:
                    log.log_info(f"Received {len(response.tool_calls)} tool calls from LLM")
                    accumulated_tool_calls.extend(response.tool_calls)
                
                # Track stop reason
                if response.finish_reason:
                    final_stop_reason = response.finish_reason
            
            # Only emit signals if not cancelled
            if not self.cancelled:
                # Emit tool calls if detected
                if accumulated_tool_calls:
                    self.tool_calls_detected.emit(accumulated_tool_calls)
                
                # Emit stop reason
                self.stop_reason_received.emit(final_stop_reason)
                
                # Emit completion
                self.response_complete.emit()
            
        except Exception as e:
            if not self.cancelled:
                self.response_error.emit(str(e))


class LLMContinuationThread(QThread):
    """Thread for handling LLM continuation after tool execution with proper Qt signal handling"""
    continuation_chunk = Signal(str)        # Streaming response chunks
    continuation_complete = Signal(str)     # Final complete response
    continuation_error = Signal(str)        # Error signal
    additional_tools_detected = Signal(list)  # Additional tool calls detected
    
    def __init__(self, provider_factory, provider_config: dict, messages: List[Dict[str, Any]], 
                 mcp_tools: List[Dict[str, Any]] = None, native_message_callback: Callable = None, parent=None):
        super().__init__(parent)
        self.provider_factory = provider_factory
        self.provider_config = provider_config
        self.messages = messages
        self.mcp_tools = mcp_tools or []
        self.native_message_callback = native_message_callback
        self.continuation_buffer = ""
        self.accumulated_tool_calls = []
        
    def run(self):
        """Run LLM continuation in thread"""
        try:
            log.log_info("LLMContinuationThread starting")
            # Run async continuation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._continue_async())
            finally:
                loop.close()
                
        except Exception as e:
            log.log_error(f"LLM continuation thread error: {e}")
            import traceback
            log.log_error(f"Continuation thread traceback: {traceback.format_exc()}")
            self.continuation_error.emit(str(e))
    
    async def _continue_async(self):
        """Execute LLM continuation asynchronously"""
        try:
            # Import required models
            from ..services.models.llm_models import ChatRequest, ChatMessage, MessageRole
            
            # Create provider instance
            provider = self.provider_factory.create_provider(self.provider_config)
            
            # DEBUG: Log what messages we received for continuation
            import json
            log.log_info(f"🔍 CONTINUATION INPUT DEBUG - Received {len(self.messages)} messages:")
            for i, msg in enumerate(self.messages):
                debug_msg = dict(msg) if isinstance(msg, dict) else str(msg)
                if isinstance(debug_msg, dict) and 'content' in debug_msg and len(str(debug_msg['content'])) > 100:
                    debug_msg['content'] = str(debug_msg['content'])[:100] + "..."
                log.log_info(f"   Input Message {i}: {json.dumps(debug_msg, indent=2) if isinstance(debug_msg, dict) else debug_msg}")
            
            # Convert messages to ChatMessage format while preserving tool structure
            chat_messages = []
            for msg in self.messages:
                if msg["role"] == "system":
                    chat_messages.append(ChatMessage(role=MessageRole.SYSTEM, content=msg["content"]))
                elif msg["role"] == "tool":
                    # Tool result message - preserve tool_call_id
                    chat_messages.append(ChatMessage(
                        role=MessageRole.TOOL,
                        content=msg["content"],
                        tool_call_id=msg.get("tool_call_id"),
                        name=msg.get("name")
                    ))
                else:
                    # User or assistant message
                    role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
                    content = msg["content"]
                    
                    # Extract text content from structured messages
                    if isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get('type') == 'text':
                                text_parts.append(part.get('text', ''))
                        content = ' '.join(text_parts) if text_parts else str(content)
                    
                    # Handle tool calls for assistant messages
                    tool_calls = None
                    if msg.get("tool_calls"):
                        from ..services.models.llm_models import ToolCall
                        tool_calls = []
                        for tc in msg["tool_calls"]:
                            if tc.get("type") == "function" and "function" in tc:
                                # Parse OpenAI format
                                func_data = tc["function"]
                                arguments = func_data.get("arguments", "{}")
                                if isinstance(arguments, str):
                                    try:
                                        arguments = json.loads(arguments)
                                    except:
                                        arguments = {}
                                
                                tool_call = ToolCall(
                                    id=tc.get("id", ""),
                                    name=func_data.get("name", ""),
                                    arguments=arguments
                                )
                                tool_calls.append(tool_call)
                    
                    chat_messages.append(ChatMessage(
                        role=role,
                        content=content,
                        tool_calls=tool_calls
                    ))
            
            # Create continuation request
            request = ChatRequest(
                messages=chat_messages,
                model=self.provider_config.get('model', ''),
                stream=True,
                max_tokens=self.provider_config.get('max_tokens', 4096)
            )
            
            # Add tools if available and provider supports them
            if self.mcp_tools and provider.supports_tools():
                request = provider.prepare_tool_enabled_request(request, self.mcp_tools)
            
            # DEBUG: Log the actual continuation request messages being sent to LLM
            import json
            debug_messages = []
            for msg in request.messages:
                debug_msg = {
                    "role": msg.role.value,
                    "content": msg.content[:200] + "..." if len(str(msg.content)) > 200 else msg.content
                }
                if msg.tool_calls:
                    debug_msg["tool_calls"] = [
                        {"id": tc.id, "name": tc.name, "arguments": str(tc.arguments)[:100] + "..." if len(str(tc.arguments)) > 100 else tc.arguments}
                        for tc in msg.tool_calls
                    ]
                if msg.tool_call_id:
                    debug_msg["tool_call_id"] = msg.tool_call_id
                debug_messages.append(debug_msg)
            
            log.log_info(f"🔍 LLM REQUEST DEBUG - Continuation ({len(request.messages)} messages):")
            log.log_info(json.dumps(debug_messages, indent=2))
            
            # Stream the continuation response
            log.log_info("Starting LLM continuation stream")
            response_count = 0
            empty_response_count = 0
            final_stop_reason = None
            
            async for response in provider.chat_completion_stream(request, self.native_message_callback):
                response_count += 1
                
                # Handle content chunks
                if response.content:
                    self.continuation_buffer += response.content
                    # Emit streaming chunk signal for real-time UI updates
                    self.continuation_chunk.emit(response.content)
                else:
                    empty_response_count += 1
                    # Only log every 50 empty responses to avoid spam
                    if empty_response_count % 50 == 0:
                        log.log_debug(f"Response {response_count} has no content ({empty_response_count} empty so far)")
                
                # Handle additional tool calls
                if response.tool_calls:
                    log.log_info(f"Additional tool calls detected: {len(response.tool_calls)}")
                    self.accumulated_tool_calls.extend(response.tool_calls)
                
                # Track stop reason
                if response.finish_reason:
                    final_stop_reason = response.finish_reason
            
            log.log_info(f"LLM continuation stream completed after {response_count} responses ({empty_response_count} empty), stop_reason: {final_stop_reason}")
            
            # Decide next action based on stop reason, supporting both Anthropic and OpenAI formats
            tool_reasons = ["tool_use", "tool_calls"]  # Anthropic and OpenAI formats
            completion_reasons = ["end_turn", "stop"]  # Anthropic and OpenAI formats
            
            if final_stop_reason in tool_reasons and self.accumulated_tool_calls:
                # More tools needed
                log.log_info(f"Emitting additional_tools_detected with {len(self.accumulated_tool_calls)} tools")
                self.additional_tools_detected.emit(self.accumulated_tool_calls)
            elif final_stop_reason in completion_reasons:
                # LLM is completely done
                log.log_info(f"Emitting continuation_complete with {len(self.continuation_buffer)} chars")
                self.continuation_complete.emit(self.continuation_buffer)
            elif self.accumulated_tool_calls:
                # Fallback: if we have tool calls but unclear stop reason
                log.log_info(f"Fallback: Emitting additional_tools_detected with {len(self.accumulated_tool_calls)} tools")
                self.additional_tools_detected.emit(self.accumulated_tool_calls)
            else:
                # No more tool calls and no clear stop reason - assume done
                log.log_info(f"Fallback: Emitting continuation_complete with {len(self.continuation_buffer)} chars")
                self.continuation_complete.emit(self.continuation_buffer)
                
        except Exception as e:
            log.log_error(f"Error in LLM continuation: {e}")
            self.continuation_error.emit(str(e))


class ToolExecutorThread(QThread):
    """Thread for handling async tool execution with proper Qt signal handling"""
    tool_execution_update = Signal(str)  # UI update signal
    tool_execution_error = Signal(str)   # Error signal
    tool_results_ready = Signal(list)    # Tool results signal - emitted for ALL results (success or failure)
    
    def __init__(self, orchestrator, tool_calls: List[ToolCall], response_buffer: str, parent=None):
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.tool_calls = tool_calls
        self.response_buffer = response_buffer
        self.tool_results = []
        
    def run(self):
        """Run tool execution in thread"""
        try:
            log.log_info("ToolExecutorThread starting")
            # Run async tool execution
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(self._execute_tools_async())
                self.tool_results = results
                log.log_info(f"ToolExecutorThread completed, emitting results signal with {len(results)} results")
                # Emit results signal - this should trigger continuation
                self.tool_results_ready.emit(results)
            finally:
                loop.close()
                
        except Exception as e:
            log.log_error(f"Tool execution thread error: {e}")
            import traceback
            log.log_error(f"Tool execution traceback: {traceback.format_exc()}")
            self.tool_execution_error.emit(str(e))
    
    async def _execute_tools_async(self):
        """Execute tools asynchronously"""
        # Execute all tool calls
        results = await self.orchestrator.execute_tool_calls(self.tool_calls)
        
        # Update UI with tool execution results
        tool_summary = self.orchestrator.get_tool_execution_summary(self.tool_calls, results)
        current_response = self.response_buffer + "\n\n" + tool_summary
        
        # Emit UI update signal
        self.tool_execution_update.emit(current_response)
        
        # Log the results for debugging
        log.log_info(f"Tool execution completed with {len(results)} results")
        for i, result in enumerate(results):
            if result.error:
                log.log_info(f"Tool result {i}: ERROR - {result.error}")
            else:
                log.log_info(f"Tool result {i}: SUCCESS - {len(result.content)} chars")
        
        return results


class QueryController:
    """Controller for the Query tab functionality"""
    
    def __init__(self, view, binary_view: Optional[bn.BinaryView] = None, view_frame=None):
        self.view = view
        self.context_service = BinaryContextService(binary_view, view_frame)
        self.settings_service = SettingsService()
        self.llm_factory = LLMProviderFactory()
        self.analysis_db = AnalysisDBService()
        self.mcp_service = MCPClientService()
        
        # MCP integration components
        self.mcp_connection_manager = MCPConnectionManager()
        self.mcp_orchestrator = MCPToolOrchestrator(self.mcp_service, self.context_service)
        
        # Chat editing
        self.chat_edit_manager = ChatEditManager()
        
        # Chat management
        self.chats = {}  # chat_id -> chat_data
        self.current_chat_id = None
        
        # RLHF context tracking
        self._current_rlhf_context = {
            'model_name': None,
            'prompt': None,
            'system': None,
            'response': None
        }
        self.next_chat_id = 1
        
        # LLM streaming state
        self.llm_thread = None
        self._llm_response_buffer = ""
        self._query_active = False
        self._current_query_binary_hash = None  # Cache binary hash during query
        
        # Tool execution state
        self._tool_execution_active = False
        self._pending_tool_calls: List[ToolCall] = []
        self._tool_results: List[ToolResult] = []
        self._continuation_complete = False
        self._tool_call_attempts = {}  # Track tool call attempts for loop prevention

        # Debounced rendering for streaming responses
        self._debounced_renderer = DebouncedRenderer(
            update_callback=self._debounced_update_callback
        )

        # ReAct (agentic mode) state
        self._react_thread: Optional[ReActOrchestratorThread] = None
        self._react_active = False

        # Connect view signals
        self._connect_signals()

        # Initialize binary view and load chats if binary view is available
        if binary_view:
            # Properly initialize the binary view with hash calculation
            self.set_binary_view(binary_view)
            # Note: set_binary_view will call load_existing_chats() if hash changes
    
    def _connect_signals(self):
        """Connect view signals to controller methods"""
        self.view.submit_query_requested.connect(self.submit_query)
        self.view.stop_query_requested.connect(self.stop_query)
        self.view.new_chat_requested.connect(self.new_chat)
        self.view.delete_chats_requested.connect(self.delete_chats)
        self.view.chat_selected.connect(self.chat_selected)
        self.view.chat_name_changed.connect(self.on_chat_name_changed)
        self.view.edit_mode_changed.connect(self.on_edit_mode_changed)
        self.view.rag_enabled_changed.connect(self.on_rag_enabled_changed)
        self.view.mcp_enabled_changed.connect(self.on_mcp_enabled_changed)
        self.view.agentic_enabled_changed.connect(self.on_agentic_enabled_changed)
    
    def set_binary_view(self, binary_view: bn.BinaryView):
        """Update the binary view for context service"""
        self.context_service.set_binary_view(binary_view)

        # Calculate the new binary's hash
        new_binary_hash = self.analysis_db.get_binary_hash(binary_view)
        current_binary_hash = self.context_service.get_binary_hash()

        # Check if this is a different binary (hash changed)
        if new_binary_hash != current_binary_hash:
            # Perform automatic migration if needed (legacy hash -> new hash)
            # This gracefully transitions existing chats/analysis to the new hashing scheme
            self.analysis_db.migrate_legacy_hash_if_needed(binary_view, new_binary_hash)

            # Cache the new hash in context service
            self.context_service.set_binary_hash(new_binary_hash)

            # Load existing chats for the new binary
            self.load_existing_chats()
    
    def set_view_frame(self, view_frame):
        """Update the view frame for context service"""
        self.context_service.set_view_frame(view_frame)
    
    def set_current_offset(self, offset: int):
        """Update current offset in context service"""
        self.context_service.set_current_offset(offset)
    
    def load_existing_chats(self):
        """Load existing chats from database on startup"""
        # CRITICAL FIX: Don't reload chats during active query/tool execution/react analysis
        if self._query_active or self._tool_execution_active or self._react_active:
            log.log_debug("Skipping chat reload during active query/tool execution/react analysis")
            return
        
        try:
            binary_hash = self._get_current_binary_hash()
            if not binary_hash:
                log.log_debug("No binary hash available, skipping chat history load")
                return

            log.log_info(f"Loading chats for binary hash: {binary_hash[:16]}...")

            # Clear existing chats first
            self.chats.clear()
            self.current_chat_id = None

            # Clear the view's history table
            self.view.history_table.setRowCount(0)
            
            # Temporarily disable sorting and itemChanged signals during bulk loading for better performance
            was_sorting_enabled = self.view.history_table.isSortingEnabled()
            self.view.history_table.setSortingEnabled(False)
            
            # Temporarily disconnect itemChanged signal to prevent false triggers during bulk loading
            try:
                self.view.history_table.itemChanged.disconnect()
                signals_disconnected = True
            except:
                signals_disconnected = False
            
            # Get all chats for this binary
            existing_chats = self.analysis_db.get_all_chats(binary_hash)

            log.log_info(f"Found {len(existing_chats)} chats in database for this binary")
            for chat in existing_chats:
                log.log_info(f"  Chat ID: {chat['chat_id']}, Hash: {chat.get('binary_hash', 'N/A')[:16]}...")

            if not existing_chats:
                log.log_debug("No existing chats found in database")
                
                # Reconnect itemChanged signal if it was disconnected
                if signals_disconnected:
                    self.view.history_table.itemChanged.connect(self.view.on_history_item_changed)
                
                # Re-enable sorting even when no chats exist
                if was_sorting_enabled:
                    self.view.history_table.setSortingEnabled(True)
                # Set default content since no chats exist
                self.view.set_chat_content("No chat selected. Click 'New' to start a conversation.")
                return
            
            log.log_info(f"Loading {len(existing_chats)} existing chats from database")
            
            # Load chat metadata for names
            chat_metadata = {}
            try:
                all_metadata = self.analysis_db.get_all_chat_metadata(binary_hash)
                for metadata in all_metadata:
                    chat_metadata[metadata['chat_id']] = metadata['name']
            except Exception as e:
                log.log_warn(f"Failed to load chat metadata: {e}")
            
            # Load chats and add to view
            for chat_data in existing_chats:
                chat_id = int(chat_data['chat_id'])
                message_count = chat_data['message_count']
                first_message = chat_data['first_message']
                last_message = chat_data['last_message']
                
                # Get chat name from metadata, or use default
                chat_name = chat_metadata.get(str(chat_id), f"Chat {chat_id}")
                
                # Add chat entry to in-memory storage (messages will be loaded lazily)
                self.chats[chat_id] = {
                    "name": chat_name,
                    "messages": [],  # Will be loaded when chat is selected
                    "created": first_message,
                    "updated": last_message
                }
                
                # Convert UTC timestamp to local time for display consistency
                display_timestamp = self._convert_utc_to_local_display(last_message)
                
                # Add to view's history table directly (without triggering signals)
                self._add_chat_to_history_direct(chat_id, chat_name, display_timestamp)
                
                # Update next chat ID to avoid conflicts
                if chat_id >= self.next_chat_id:
                    self.next_chat_id = chat_id + 1
            
            # Reconnect itemChanged signal if it was disconnected
            if signals_disconnected:
                self.view.history_table.itemChanged.connect(self.view.on_history_item_changed)
            
            # Re-enable sorting and ensure proper order (newest first)
            if was_sorting_enabled:
                self.view.history_table.setSortingEnabled(True)
                self.view.history_table.sortByColumn(1, Qt.DescendingOrder)  # Sort by timestamp, descending
            
            log.log_info(f"Successfully loaded {len(existing_chats)} chats")
            
        except Exception as e:
            log.log_error(f"Failed to load existing chats: {e}")
            # Re-enable sorting and reconnect signals even on error
            try:
                if 'signals_disconnected' in locals() and signals_disconnected:
                    self.view.history_table.itemChanged.connect(self.view.on_history_item_changed)
                if 'was_sorting_enabled' in locals() and was_sorting_enabled:
                    self.view.history_table.setSortingEnabled(True)
            except:
                pass
    
    def _convert_utc_to_local_display(self, utc_timestamp: str) -> str:
        """Convert UTC timestamp from database to local display format"""
        try:
            from PySide6.QtCore import QDateTime, QTimeZone
            
            # Parse the UTC timestamp and explicitly set it as UTC time
            qdt_utc = QDateTime.fromString(utc_timestamp, "yyyy-MM-dd hh:mm:ss")
            qdt_utc.setTimeZone(QTimeZone.utc())  # This time is in UTC
            
            # Convert to local time zone  
            qdt_local = qdt_utc.toLocalTime()
            
            # Return in display format
            return qdt_local.toString("yyyy-MM-dd hh:mm:ss")
        except Exception as e:
            log.log_warn(f"Failed to convert timestamp {utc_timestamp}: {e}")
            # Fallback: return original timestamp
            return utc_timestamp
    
    def _add_chat_to_history_direct(self, chat_id: int, description: str, timestamp: str):
        """Add chat to history table directly without triggering signals (for bulk loading)"""
        from PySide6.QtWidgets import QTableWidgetItem
        from PySide6.QtCore import Qt
        
        row_count = self.view.history_table.rowCount()
        self.view.history_table.insertRow(row_count)
        
        desc_item = QTableWidgetItem(description)
        desc_item.setData(Qt.UserRole, chat_id)  # Store chat ID
        timestamp_item = QTableWidgetItem(timestamp)
        timestamp_item.setFlags(timestamp_item.flags() & ~Qt.ItemIsEditable)  # Make timestamp read-only
        
        self.view.history_table.setItem(row_count, 0, desc_item)
        self.view.history_table.setItem(row_count, 1, timestamp_item)
    
    def _expand_macros(self, query_text: str) -> str:
        """Expand macros in the query text using current Binary Ninja context"""
        try:
            expanded_text = query_text
            
            # #func - current function code at current view level (like Explain tab)
            if '#func' in expanded_text:
                try:
                    context = self.context_service.get_current_context()
                    if context and context.get("offset"):
                        current_offset = context["offset"]
                        current_level = self.context_service.get_current_view_level()
                        code_data = self.context_service.get_code_at_level(current_offset, current_level)
                        
                        if code_data and not code_data.get("error") and code_data.get('lines'):
                            func_code = '\n'.join(line['content'] for line in code_data['lines'])
                            func_expansion = f"\n\n```\n{func_code}\n```\n\n"
                            expanded_text = expanded_text.replace('#func', func_expansion)
                            log.log_info(f"Expanded #func to {len(func_code)} characters of code")
                        else:
                            expanded_text = expanded_text.replace('#func', 'current function')
                            log.log_warn("Could not get function code for #func macro")
                    else:
                        expanded_text = expanded_text.replace('#func', 'current function')
                        log.log_warn("No context available for #func macro")
                except Exception as e:
                    log.log_error(f"Error expanding #func macro: {e}")
                    expanded_text = expanded_text.replace('#func', 'current function')
            
            # #line - current line context from BinaryContextService
            if '#line' in expanded_text:
                try:
                    context = self.context_service.get_current_context()
                    if context and context.get("offset"):
                        current_offset = context["offset"]
                        current_level = self.context_service.get_current_view_level()
                        line_context = self.context_service.get_line_context(current_offset, current_level)
                        
                        if line_context:
                            # Format with newlines and code fencing
                            line_expansion = f"\n\n```\n{line_context}\n```\n\n"
                            expanded_text = expanded_text.replace('#line', line_expansion)
                            log.log_info(f"Expanded #line to line context")
                        else:
                            expanded_text = expanded_text.replace('#line', 'current line')
                            log.log_warn("Could not get line context for #line macro")
                    else:
                        expanded_text = expanded_text.replace('#line', 'current line')
                        log.log_warn("No context available for #line macro")
                except Exception as e:
                    log.log_error(f"Error expanding #line macro: {e}")
                    expanded_text = expanded_text.replace('#line', 'current line')
            
            # #addr - current address from BinaryView
            if '#addr' in expanded_text:
                try:
                    context = self.context_service.get_current_context()
                    current_addr = context.get('current_address') if context else None
                    if current_addr is not None:
                        addr_str = f"0x{current_addr:x}"
                        expanded_text = expanded_text.replace('#addr', addr_str)
                        log.log_info(f"Expanded #addr to: {addr_str}")
                    else:
                        expanded_text = expanded_text.replace('#addr', '0x0')
                        log.log_warn("Could not get current address for #addr macro")
                except Exception as e:
                    log.log_error(f"Error expanding #addr macro: {e}")
                    expanded_text = expanded_text.replace('#addr', '0x0')
            
            # #range(start, end) - hexdump of specified address range
            import re
            range_pattern = r'#range\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)'
            range_matches = re.findall(range_pattern, expanded_text)
            
            for start_str, end_str in range_matches:
                try:
                    # Convert string addresses to integers (support both hex and decimal)
                    start_str = start_str.strip()
                    end_str = end_str.strip()
                    
                    # Auto-detect base and convert to int
                    if start_str.startswith('0x') or start_str.startswith('0X'):
                        start_addr = int(start_str, 16)
                    else:
                        try:
                            start_addr = int(start_str, 16)  # Try hex first
                        except ValueError:
                            start_addr = int(start_str, 10)  # Fall back to decimal
                    
                    if end_str.startswith('0x') or end_str.startswith('0X'):
                        end_addr = int(end_str, 16)
                    else:
                        try:
                            end_addr = int(end_str, 16)  # Try hex first
                        except ValueError:
                            end_addr = int(end_str, 10)  # Fall back to decimal
                    
                    # Calculate size and get hexdump
                    size = end_addr - start_addr
                    if size > 0:
                        hexdump_data = self.context_service.get_hexdump(start_addr, size)
                        if hexdump_data and not hexdump_data.get('error') and hexdump_data.get('lines'):
                            hexdump_lines = []
                            for line in hexdump_data['lines']:
                                # Format as traditional hexdump: address: hex_bytes  ascii
                                formatted_line = f"{line['address']}: {line['hex']}  {line['ascii']}"
                                hexdump_lines.append(formatted_line)
                            hexdump_text = '\n'.join(hexdump_lines)
                            # Format with newlines and code fencing
                            range_expansion = f"\n\n```\n{hexdump_text}\n```\n\n"
                            # Replace the specific #range(start, end) call
                            range_call = f"#range({start_str}, {end_str})"
                            expanded_text = expanded_text.replace(range_call, range_expansion)
                            log.log_info(f"Expanded #range({start_str}, {end_str}) to hexdump of 0x{start_addr:x}-0x{end_addr:x} ({size} bytes)")
                        else:
                            range_str = f"0x{start_addr:x}-0x{end_addr:x}"
                            range_call = f"#range({start_str}, {end_str})"
                            expanded_text = expanded_text.replace(range_call, range_str)
                            log.log_warn(f"Could not get hexdump for #range({start_str}, {end_str}), using address range")
                    else:
                        range_call = f"#range({start_str}, {end_str})"
                        expanded_text = expanded_text.replace(range_call, f"0x{start_addr:x}")
                        log.log_warn(f"Invalid range for #range({start_str}, {end_str}): end <= start")
                        
                except Exception as e:
                    log.log_error(f"Error expanding #range({start_str}, {end_str}) macro: {e}")
                    range_call = f"#range({start_str}, {end_str})"
                    expanded_text = expanded_text.replace(range_call, f"invalid range ({start_str}, {end_str})")
            
            if expanded_text != query_text:
                log.log_info("Macro expansion completed successfully")
            
            return expanded_text
            
        except Exception as e:
            log.log_error(f"Error during macro expansion: {e}")
            return query_text

    def submit_query(self, query_text: str):
        """Handle query submission with LLM integration"""
        log.log_info(f"Query submitted: {query_text[:50]}...")

        # Expand macros in the query text
        query_text = self._expand_macros(query_text)

        # Route to agentic mode if enabled
        if self.view.is_agentic_enabled():
            self._submit_agentic_query(query_text)
            return

        # Check if query is already active
        if self._query_active:
            log.log_warn("Query already active, ignoring request")
            return
        
        try:
            # Get active LLM provider first  
            active_provider = self.settings_service.get_active_llm_provider()
            if not active_provider:
                # Fall back to static response
                self._handle_no_llm_provider(query_text)
                return
                
            # Ensure we have an active chat
            if self.current_chat_id is None:
                self.new_chat()
            
            # Set query active state and cache binary hash
            self._query_active = True
            self._current_query_binary_hash = self._get_current_binary_hash()
            self.view.set_query_running(True)

            # Reset auto-scroll to follow new response by default
            self.view.enable_auto_scroll()
            
            # Add user query to chat immediately (both legacy and native formats)
            self._add_message_to_chat(self.current_chat_id, "user", query_text)
            
            # Save user query in native format for the active provider
            provider_type = active_provider.get('provider_type', 'anthropic')
            self._save_user_message_native(query_text, provider_type)
            
            # Add placeholder for assistant response (don't save to database)
            self._add_message_to_chat(self.current_chat_id, "assistant", "*Thinking...*", save_to_db=False)
            self._update_chat_display()
            
            
            # Get current context and settings
            context = self.context_service.get_current_context()
            rag_enabled = self.view.is_rag_enabled()
            mcp_enabled = self.view.is_mcp_enabled()
            
            # Prepare conversation messages for LLM in provider's native format
            provider_type = active_provider.get('provider_type', 'anthropic')
            messages = self._prepare_native_messages(query_text, context, rag_enabled, mcp_enabled, provider_type)
            
            # Track RLHF context for this query
            self._track_rlhf_context(active_provider['name'], query_text, messages)
            
            # Get MCP tools if enabled
            mcp_tools = []
            if mcp_enabled:
                if self.mcp_connection_manager.ensure_connections():
                    mcp_tools = self.mcp_connection_manager.get_available_tools_for_llm()
                    log.log_info(f"MCP enabled with {len(mcp_tools)} tools available")
                else:
                    log.log_warn("MCP enabled but connection failed")
            
            # Start LLM query thread
            self._start_llm_query(messages, active_provider, mcp_tools)
            
            log.log_info("Query sent to LLM")
            
        except Exception as e:
            error_msg = f"Exception in submit_query: {str(e)}"
            log.log_error(error_msg)
            
            # Add error message to chat
            if self.current_chat_id:
                self._update_last_assistant_message(f"**Error**: {str(e)}")
                self._update_chat_display()
            
            self._query_active = False
            self._current_query_binary_hash = None
            self.view.set_query_running(False)
    
    def stop_query(self):
        """Stop the current query"""
        log.log_info("Stop query requested")

        # Cancel debounced rendering
        self._debounced_renderer.cancel()

        # Cancel ReAct if active
        if self._react_active and self._react_thread:
            self._react_thread.cancel()
            self._cleanup_react_thread()
            self._react_active = False

        # Cancel standard query if active
        if self._query_active and hasattr(self, 'llm_thread') and self.llm_thread:
            self.llm_thread.cancel()
            self._cleanup_llm_thread()
        self._query_active = False
        self._current_query_binary_hash = None
        self.view.set_query_running(False)
    
    def new_chat(self):
        """Create a new chat session"""
        chat_id = self.next_chat_id
        self.next_chat_id += 1
        
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        chat_name = f"Chat {chat_id}"
        
        # Create chat data
        self.chats[chat_id] = {
            "name": chat_name,
            "messages": [],
            "created": timestamp,
            "updated": timestamp
        }
        
        # Save chat metadata to database
        binary_hash = self._get_current_binary_hash()
        if binary_hash:
            try:
                self.analysis_db.save_chat_metadata(binary_hash, str(chat_id), chat_name)
            except Exception as e:
                log.log_warn(f"Failed to save initial chat metadata: {e}")
        
        # Add to view
        self.view.add_chat_to_history(chat_id, chat_name, timestamp)
        
        # Set as current chat
        self.current_chat_id = chat_id
        
        # Set initial content
        initial_content = self._get_initial_chat_content()
        self.view.set_chat_content(initial_content)
        
        log.log_info(f"New chat created: {chat_name}")
    
    def delete_chats(self, selected_rows: List[int]):
        """Delete selected chats"""
        if not selected_rows:
            log.log_warn("No chats selected for deletion")
            return
        
        log.log_info(f"Deleting {len(selected_rows)} chats")
        
        # Get chat IDs from selected rows
        chat_ids_to_delete = []
        chat_names = []
        for row in selected_rows:
            try:
                item = self.view.history_table.item(row, 0)
                if item:
                    chat_id = item.data(Qt.UserRole)
                    chat_name = item.text()
                    if chat_id is not None:  # Check for None explicitly
                        chat_ids_to_delete.append(chat_id)
                        chat_names.append(chat_name)
            except Exception as e:
                log.log_error(f"Error getting chat ID from row {row}: {e}")
                continue
        
        if not chat_ids_to_delete:
            log.log_warn("No valid chat IDs found for deletion")
            return
        
        # Show confirmation dialog
        from PySide6.QtWidgets import QMessageBox
        chat_list = "\n".join([f"• {name}" for name in chat_names[:5]])  # Show first 5
        if len(chat_names) > 5:
            chat_list += f"\n• ... and {len(chat_names) - 5} more"
        
        reply = QMessageBox.question(
            self.view, 
            "Confirm Chat Deletion",
            f"Are you sure you want to delete {len(chat_ids_to_delete)} chat(s)?\n\n{chat_list}\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            log.log_info("Chat deletion cancelled by user")
            return
        
        # Remove from chats dict and database
        binary_hash = self._get_current_binary_hash()
        for chat_id in chat_ids_to_delete:
            if chat_id in self.chats:
                del self.chats[chat_id]
            
            # Delete from database
            if binary_hash:
                try:
                    self.analysis_db.delete_chat(binary_hash, str(chat_id))
                    log.log_info(f"Deleted chat {chat_id} from legacy database")
                except Exception as e:
                    log.log_error(f"Failed to delete chat {chat_id} from legacy database: {e}")
                
                # Also delete native chat messages
                try:
                    self.analysis_db.delete_native_chat(binary_hash, str(chat_id))
                    log.log_info(f"Deleted native chat {chat_id} from database")
                except Exception as e:
                    log.log_error(f"Failed to delete native chat {chat_id} from database: {e}")
                
                # Also delete chat metadata
                try:
                    self.analysis_db.delete_chat_metadata(binary_hash, str(chat_id))
                    log.log_info(f"Deleted chat metadata {chat_id} from database")
                except Exception as e:
                    log.log_error(f"Failed to delete chat metadata {chat_id} from database: {e}")
            
            # Clear current chat if it was deleted
            if chat_id == self.current_chat_id:
                self.current_chat_id = None
        
        # Remove from view
        self.view.remove_selected_chats()
        
        # Set default content if no chats remain
        if not self.chats:
            self.view.set_chat_content("No chat selected. Click 'New' to start a conversation.")
        
        # Show success message
        deleted_count = len(chat_ids_to_delete)
        log.log_info(f"Successfully deleted {deleted_count} chat(s)")
        
        # Optional: Show brief success notification (uncomment if desired)
        # from PySide6.QtWidgets import QMessageBox
        # QMessageBox.information(
        #     self.view, 
        #     "Chats Deleted", 
        #     f"Successfully deleted {deleted_count} chat(s).",
        #     QMessageBox.Ok
        # )
    
    def chat_selected(self, chat_id: int):
        """Handle chat selection"""
        
        if chat_id in self.chats:
            self.current_chat_id = chat_id
            
            # Load chat history from database if not already loaded
            if not self.chats[chat_id]["messages"]:
                try:
                    binary_hash = self._get_current_binary_hash()
                    if binary_hash:
                        db_messages = self.analysis_db.get_chat_history(binary_hash, str(chat_id))
                        # Convert database messages to in-memory format
                        for db_msg in db_messages:
                            message = {
                                "role": db_msg["role"],
                                "content": db_msg["content"],
                                "timestamp": db_msg["created_at"]
                            }
                            self.chats[chat_id]["messages"].append(message)
                        
                except Exception as e:
                    log.log_error(f"Failed to load chat history from database: {e}")
            
            self._update_chat_display()
        else:
            log.log_warn(f"Chat ID {chat_id} not found")
    
    def on_chat_name_changed(self, chat_id: int, new_name: str):
        """Handle chat name change from table editing"""
        log.log_info(f"Chat name changed for {chat_id}: '{new_name}'")
        
        if chat_id not in self.chats:
            log.log_warn(f"Chat ID {chat_id} not found for name change")
            return
        
        # Update in-memory chat data
        old_name = self.chats[chat_id]["name"]
        self.chats[chat_id]["name"] = new_name
        
        # Save to database
        binary_hash = self._get_current_binary_hash()
        if binary_hash:
            try:
                self.analysis_db.save_chat_metadata(binary_hash, str(chat_id), new_name)
                log.log_info(f"Saved chat name change to database: '{old_name}' -> '{new_name}'")
            except Exception as e:
                log.log_error(f"Failed to save chat name change to database: {e}")
                # Revert the in-memory change if database save failed
                self.chats[chat_id]["name"] = old_name
                return
        
        # Update the chat content heading if this is the current chat
        if chat_id == self.current_chat_id:
            self._update_chat_display()
            log.log_info(f"Updated chat display with new heading: '{new_name}'")
    
    def on_edit_mode_changed(self, is_edit_mode: bool):
        """Handle edit mode change using ChatEditManager"""
        log.log_info(f"Query edit mode changed to: {is_edit_mode}")
        
        if is_edit_mode:
            # Entering edit mode - generate editable content
            self._prepare_edit_mode()
        else:
            # Exiting edit mode - save changes
            if self.current_chat_id:
                try:
                    self._save_edited_chat_content_with_manager()
                    log.log_info("Successfully saved edited chat content using ChatEditManager")
                except Exception as e:
                    log.log_error(f"Failed to save edited chat content: {e}")
    
    def _prepare_edit_mode(self):
        """Prepare chat content for editing using ChatEditManager"""
        if not self.current_chat_id or self.current_chat_id not in self.chats:
            log.log_warn("No current chat to edit")
            return
        
        try:
            # Get current chat data
            chat_data = self.chats[self.current_chat_id]
            
            # Generate editable markdown content
            editable_content = self.chat_edit_manager.generate_editable_content(chat_data)
            
            # Set the content in the view for editing
            self.view.set_chat_content(editable_content)
            
            log.log_info(f"Prepared chat {self.current_chat_id} for editing with {len(self.chat_edit_manager.message_map)} tracked chunks")
            
        except Exception as e:
            log.log_error(f"Failed to prepare edit mode: {e}")
    
    def _save_edited_chat_content_with_manager(self):
        """Save edited chat content using ChatEditManager"""
        if not self.current_chat_id:
            log.log_warn("No current chat to save")
            return
        
        try:
            # Get the edited content from the view
            edited_content = self.view.get_chat_content()
            if not edited_content:
                log.log_warn("No content to save")
                return
            
            # Parse changes using ChatEditManager
            changes = self.chat_edit_manager.parse_edited_content(edited_content)
            
            if not changes:
                log.log_info("No changes detected in edited content")
                return
            
            log.log_info(f"Detected {len(changes)} changes in edited content")
            
            # Apply changes to database
            binary_hash = self._get_current_binary_hash()
            if not binary_hash:
                log.log_warn("No binary hash available for saving edited content")
                return
            
            # Apply changes based on their type
            messages_updated = self._apply_chat_changes(binary_hash, changes)
            
            # Reload the chat to reflect changes
            if messages_updated:
                self._reload_current_chat()
                log.log_info(f"Successfully applied {len(changes)} changes to chat {self.current_chat_id}")
            
        except Exception as e:
            log.log_error(f"Failed to save edited chat content with manager: {e}")
    
    def _apply_chat_changes(self, binary_hash: str, changes) -> bool:
        """Apply detected changes to the database"""
        try:
            from .chat_edit_manager import ChangeType
            
            messages_updated = False
            title_updated = False
            new_title = None
            
            for change in changes:
                if change.change_type == ChangeType.MODIFIED:
                    # Check if this is a title change
                    if getattr(change, 'role', None) == 'title':
                        title_updated = True
                        new_title = change.new_content
                        log.log_info(f"Title changed to: '{new_title}'")
                    # Update existing message
                    elif change.db_id:
                        # For now, we'll delete and recreate the entire chat
                        # In the future, we could implement more granular updates
                        messages_updated = True
                        log.log_info(f"Modified message with db_id {change.db_id}")
                
                elif change.change_type == ChangeType.DELETED:
                    # Mark message as deleted
                    if change.db_id:
                        messages_updated = True
                        log.log_info(f"Deleted message with db_id {change.db_id}")
                
                elif change.change_type == ChangeType.ADDED:
                    # Add new message
                    if change.new_content:
                        messages_updated = True
                        log.log_info(f"Added new {change.role} message")
            
            # For now, implement simple approach: rebuild entire chat
            if messages_updated:
                # Collect all current message data from changes
                final_messages = []
                
                # Add modified and unchanged messages
                for chunk_id, chat_msg in self.chat_edit_manager.message_map.items():
                    # Check if this message was modified
                    modified_change = None
                    for change in changes:
                        if change.change_type == ChangeType.MODIFIED and change.chunk_id == chunk_id:
                            modified_change = change
                            break
                    
                    # Check if this message was deleted
                    deleted = any(change.change_type == ChangeType.DELETED and change.chunk_id == chunk_id 
                                 for change in changes)
                    
                    if not deleted:
                        content = modified_change.new_content if modified_change else chat_msg.content
                        final_messages.append({
                            "role": chat_msg.role,
                            "content": content,
                            "timestamp": chat_msg.timestamp,
                            "db_id": chat_msg.db_id
                        })
                
                # Add new messages
                for change in changes:
                    if change.change_type == ChangeType.ADDED:
                        final_messages.append({
                            "role": change.role,
                            "content": change.new_content,
                            "timestamp": change.timestamp or "edited",
                            "db_id": None  # New message
                        })
                
                # Sort by order if available
                final_messages.sort(key=lambda m: m.get("db_id") or 999999)
                
                # Delete existing chat and recreate
                self.analysis_db.delete_chat(binary_hash, str(self.current_chat_id))
                
                # Save all messages to database
                for i, message in enumerate(final_messages):
                    self.analysis_db.save_chat_message(
                        binary_hash, str(self.current_chat_id),
                        message["role"], message["content"],
                        metadata={"message_order": i}
                    )
                
                # Update in-memory chat
                self.chats[self.current_chat_id]["messages"] = final_messages
                
                log.log_info(f"Rebuilt chat with {len(final_messages)} messages")
            
            # Handle title changes
            if title_updated and new_title:
                # Update in-memory chat title
                old_title = self.chats[self.current_chat_id]["name"]
                self.chats[self.current_chat_id]["name"] = new_title
                
                # Save updated title to database
                try:
                    self.analysis_db.save_chat_metadata(binary_hash, str(self.current_chat_id), new_title)
                    log.log_info(f"Saved updated chat title to database: '{new_title}'")
                except Exception as e:
                    log.log_error(f"Failed to save updated chat title to database: {e}")
                
                # Update the chat name in the history table by finding and updating the row
                try:
                    # Find the row in the history table that corresponds to this chat
                    for row in range(self.view.history_table.rowCount()):
                        item = self.view.history_table.item(row, 0)  # First column contains chat name
                        if item and item.data(Qt.UserRole) == self.current_chat_id:
                            # Update the displayed text
                            item.setText(new_title)
                            break
                except Exception as e:
                    log.log_warn(f"Could not update chat name in history table: {e}")
                
                log.log_info(f"Updated chat title from '{old_title}' to '{new_title}'")
            
            return messages_updated or title_updated
            
        except Exception as e:
            log.log_error(f"Failed to apply chat changes: {e}")
            return False
    
    def _reload_current_chat(self):
        """Reload the current chat from database"""
        if not self.current_chat_id:
            return
        
        try:
            binary_hash = self._get_current_binary_hash()
            if not binary_hash:
                return
            
            # Load fresh messages from database
            messages = self.analysis_db.get_chat_history(binary_hash, str(self.current_chat_id))
            
            # Convert to our format
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg["created_at"],
                    "db_id": msg["id"]
                })
            
            # Update in-memory chat
            self.chats[self.current_chat_id]["messages"] = formatted_messages
            
            # Display the updated chat
            self._update_chat_display()
            
            log.log_info(f"Reloaded chat {self.current_chat_id} with {len(formatted_messages)} messages")
            
        except Exception as e:
            log.log_error(f"Failed to reload current chat: {e}")
    
    def on_rag_enabled_changed(self, enabled: bool):
        """Handle RAG checkbox change"""
        log.log_info(f"RAG enabled changed to: {enabled}")
        # TODO: Update RAG settings for future queries
    
    def on_mcp_enabled_changed(self, enabled: bool):
        """Handle MCP checkbox change"""
        log.log_info(f"MCP enabled changed to: {enabled}")
        # TODO: Update MCP settings for future queries

    def on_agentic_enabled_changed(self, enabled: bool):
        """Handle Agentic mode checkbox change"""
        log.log_info(f"Agentic mode enabled changed to: {enabled}")
        # Agentic mode requires MCP to be enabled
        if enabled and not self.view.is_mcp_enabled():
            log.log_info("Enabling MCP for agentic mode")
            self.view.set_mcp_enabled(True)

    def _add_message_to_chat(self, chat_id: int, role: str, content: str, save_to_db: bool = True):
        """Add a message to the specified chat"""
        if chat_id not in self.chats:
            log.log_error(f"Chat ID {chat_id} not found in chats")
            return
        
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        
        self.chats[chat_id]["messages"].append(message)
        self.chats[chat_id]["updated"] = timestamp
        
        # Persist to database only if requested
        if save_to_db:
            try:
                binary_hash = self._current_query_binary_hash or self._get_current_binary_hash()
                if binary_hash:
                    message_order = len(self.chats[chat_id]["messages"]) - 1
                    self.analysis_db.save_chat_message(
                        binary_hash, str(chat_id), role, content, 
                        metadata={"message_order": message_order}
                    )
                else:
                    log.log_warn(f"No binary hash available for saving {role} message")
            except Exception as e:
                log.log_error(f"Failed to persist chat message: {e}")

    def _debounced_update_callback(self, accumulated_content: str):
        """
        Callback for debounced renderer - called periodically with accumulated content.

        This is called every 1 second (by default) instead of on every chunk,
        reducing UI updates by 10-40x and preventing stuttering.

        We do proper markdown rendering here, but only ~10-20 times per response
        instead of 200+ times. Even though markdown parsing is on the main thread,
        it only happens periodically (1s intervals), keeping the UI responsive.

        This matches the GhidrAssist implementation strategy.

        Args:
            accumulated_content: The accumulated response content from all deltas
        """
        if not self.current_chat_id or self.current_chat_id not in self.chats:
            return

        # Format chat as markdown and update the view
        # The view handles scroll tracking internally via _on_scroll_changed
        # which detects user-initiated scrolls and auto-enables/disables scroll following
        chat = self.chats[self.current_chat_id]
        markdown_content = self._format_chat_as_markdown(chat)
        self.view.set_chat_content(markdown_content)

    def _update_chat_display(self):
        """Update the chat display with current chat content"""
        if not self.current_chat_id or self.current_chat_id not in self.chats:
            return

        # Only load from native storage when not actively streaming
        # During streaming (standard or ReAct), use in-memory messages that are being updated
        if not self._query_active and not self._react_active:
            try:
                self._load_chat_from_native_storage(self.current_chat_id)
            except Exception as e:
                log.log_debug(f"Native storage load failed, using legacy: {e}")
        
        chat = self.chats[self.current_chat_id]
        markdown_content = self._format_chat_as_markdown(chat)
        self.view.set_chat_content(markdown_content)
    
    def _load_chat_from_native_storage(self, chat_id: int):
        """Load chat messages from native storage for display"""
        try:
            binary_hash = self._get_current_binary_hash()
            if not binary_hash:
                return
            
            # Get display-optimized messages from native storage
            display_messages = self.analysis_db.get_display_messages(binary_hash, str(chat_id))
            
            # Update in-memory chat with display messages
            if chat_id in self.chats:
                formatted_messages = []
                for msg in display_messages:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": msg["created_at"],
                        "message_type": msg.get("message_type", "standard"),
                        "provider_type": msg.get("provider_type", "unknown")
                    })
                
                self.chats[chat_id]["messages"] = formatted_messages
                log.log_debug(f"Loaded {len(formatted_messages)} messages from native storage")
        
        except Exception as e:
            log.log_warn(f"Failed to load chat from native storage: {e}")
            # Will fall back to legacy loading
    
    def _format_chat_as_markdown(self, chat: Dict[str, Any]) -> str:
        """Format chat messages as markdown with horizontal rules between conversation pairs"""
        if not chat["messages"]:
            return f"# {chat['name']}\n\n*No messages yet. Enter your query below.*"
        
        content = f"# {chat['name']}\n\n"
        
        messages = chat["messages"]
        conversation_pair_count = 0
        
        for i, message in enumerate(messages):
            role = message["role"]
            text = message["content"] 
            timestamp = message["timestamp"]
            
            if role == "user":
                # Add horizontal rule before each new conversation pair (except the first)
                if conversation_pair_count > 0:
                    content += "---\n\n"
                
                content += f"## User ({timestamp})\n{text}\n\n"
                conversation_pair_count += 1
                
            elif role == "assistant":
                content += f"## BinAssist ({timestamp})\n{text}\n\n"
                
            elif role == "tool":
                content += f"### 🔧 Tool ({timestamp})\n{text}\n\n"
                
            elif role == "tool_call":
                content += f"### 🔧 Tool Call ({timestamp})\n{text}\n\n"
                
            elif role == "tool_response":
                content += f"### 📊 Tool Response ({timestamp})\n{text}\n\n"

            elif role == "tool_status":
                # Tool status messages: just success/failure indicators
                content += f"*{text}*\n\n"

            elif role == "error":
                content += f"## Error ({timestamp})\n**{text}**\n\n"

        return content
    
    def _get_initial_chat_content(self) -> str:
        """Get initial content for new chat"""
        try:
            context = self.context_service.get_current_context()
            
            content = f"""# New Chat Started

## Current Context
**Offset**: {context.get('offset_hex', 'N/A')}  
**Filename**: {context.get('binary_info', {}).get('filename', 'Unknown')}  
**Path**: {context.get('binary_info', {}).get('filepath', 'Unknown')}  
**Architecture**: {context.get('binary_info', {}).get('architecture', 'Unknown')}  

"""
            
            func_ctx = context.get("function_context")
            if func_ctx:
                content += f"**Function**: {func_ctx['name']} ({func_ctx['start']} - {func_ctx['end']})\n\n"
            
            content += "*Enter your query below to start the conversation.*"
            
        except Exception as e:
            content = f"# New Chat Started\n\n*Error getting context: {str(e)}*\n\n*Enter your query below to start the conversation.*"
        
        return content
    
    def _format_query_with_context(self, query: str, context: Dict[str, Any], rag_enabled: bool, mcp_enabled: bool) -> str:
        """Format query with context information for LLM"""
        formatted = f"""Query: {query}

Context Information:
- Offset: {context.get('offset_hex', 'N/A')}
- Filename: {context.get('binary_info', {}).get('filename', 'Unknown')}
- Path: {context.get('binary_info', {}).get('filepath', 'Unknown')}
- Architecture: {context.get('binary_info', {}).get('architecture', 'Unknown')}
"""
        
        func_ctx = context.get("function_context")
        if func_ctx:
            formatted += f"- Function: {func_ctx['name']} ({func_ctx['start']} - {func_ctx['end']})\n"
        
        formatted += f"""
Settings:
- RAG Enabled: {rag_enabled}
- MCP Enabled: {mcp_enabled}
"""
        
        return formatted
    
    def _generate_mock_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate a mock response for testing (will be replaced with LLM)"""
        func_ctx = context.get("function_context")
        binary_info = context.get("binary_info", {})
        
        response = f"""I understand you're asking: "{query}"

Based on the current context:
- **Filename**: {binary_info.get('filename', 'Unknown')}
- **Path**: {binary_info.get('filepath', 'Unknown')}
- **Architecture**: {binary_info.get('architecture', 'Unknown')}
- **Current Offset**: {context.get('offset_hex', 'N/A')}
"""
        
        if func_ctx:
            response += f"- **Function**: {func_ctx['name']} (Size: {func_ctx['size']} bytes)\n"
        
        response += """
*This is a mock response. LLM integration will be implemented in a future update to provide intelligent analysis and answers to your reverse engineering questions.*

Some capabilities that will be available:
- Code analysis and explanation
- Vulnerability identification
- Algorithm recognition
- Decompilation assistance
- Binary comparison
- And much more!
"""
        
        return response
    
    # LLM Integration Methods
    
    def _handle_no_llm_provider(self, query_text: str):
        """Handle case where no LLM provider is configured"""
        response = f"""**No LLM provider configured**

To get AI-powered responses to your queries, please:
1. Go to Settings → LLM Providers
2. Add and configure an LLM provider (OpenAI, Anthropic, etc.)
3. Set it as the active provider

For now, here's some basic information about your query: "{query_text}"

Current context:
- Filename: {self.context_service.get_current_context().get('binary_info', {}).get('filename', 'Unknown')}
- Path: {self.context_service.get_current_context().get('binary_info', {}).get('filepath', 'Unknown')}
- Offset: {self.context_service.get_current_context().get('offset_hex', 'N/A')}
"""
        
        self._update_last_assistant_message(response, final_update=True)
        self._update_chat_display()
        self._query_active = False
        self._current_query_binary_hash = None
        self.view.set_query_running(False)
    
    def _prepare_llm_messages(self, query_text: str, context: Dict[str, Any], rag_enabled: bool, mcp_enabled: bool) -> List[Dict[str, Any]]:
        """Prepare conversation messages for LLM, including context and history (LEGACY - use _prepare_native_messages)"""
        messages = []
        
        # Add system message with context
        system_content = self._build_system_message(context, rag_enabled, mcp_enabled)
        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history (excluding the current query and placeholder response)
        if self.current_chat_id in self.chats:
            chat_messages = self.chats[self.current_chat_id]["messages"]
            # Exclude the last 2 messages (current user query and placeholder assistant response)
            for msg in chat_messages[:-2]:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"], 
                        "content": msg["content"]
                    })
        
        # Add current user query with RAG/MCP enhancements
        enhanced_query = self._enhance_query_with_context(query_text, context, rag_enabled, mcp_enabled)
        messages.append({"role": "user", "content": enhanced_query})
        
        return messages
    
    def _prepare_native_messages(self, query_text: str, context: Dict[str, Any], 
                               rag_enabled: bool, mcp_enabled: bool, provider_type: str) -> List[Dict[str, Any]]:
        """Prepare conversation messages in provider's native format using stored history"""
        from ..services.message_format_service import get_message_format_service
        from ..services.models.provider_types import ProviderType
        
        try:
            provider_enum = ProviderType(provider_type)
        except ValueError:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        format_service = get_message_format_service()
        messages = []
        
        # Add system message in provider's native format
        system_content = self._build_system_message(context, rag_enabled, mcp_enabled)
        system_message = format_service.create_system_message(system_content, provider_enum)
        messages.append(system_message)
        
        # Add conversation history from native storage
        # Note: This should already include the current user query if it was saved via _save_user_message_native()
        if self.current_chat_id:
            try:
                binary_hash = self._get_current_binary_hash()
                if binary_hash:
                    # Get native messages for this specific provider
                    native_messages = self.analysis_db.get_native_messages_for_provider(
                        binary_hash, str(self.current_chat_id), provider_type
                    )
                    messages.extend(native_messages)
                    
                    # Check if the last message is the current user query to avoid duplication
                    if native_messages:
                        last_message = native_messages[-1]
                        # Extract content from the native message to compare
                        last_content = format_service.extract_display_info(last_message, provider_enum).get('content', '')
                        enhanced_query = self._enhance_query_with_context(query_text, context, rag_enabled, mcp_enabled)
                        
                        # If the last message matches our current query, don't add it again
                        if enhanced_query in last_content or last_content in enhanced_query:
                            log.log_debug("Current user query already in conversation history, skipping duplicate")
                            return messages
            except Exception as e:
                log.log_warn(f"Failed to load native conversation history: {e}")
                # Fallback to legacy method if native history not available
                return self._prepare_llm_messages(query_text, context, rag_enabled, mcp_enabled)
        
        # Add current user query in provider's native format (only if not already in history)
        enhanced_query = self._enhance_query_with_context(query_text, context, rag_enabled, mcp_enabled)
        user_message = format_service.create_user_message(enhanced_query, provider_enum)
        messages.append(user_message)
        
        return messages
    
    def _save_user_message_native(self, query_text: str, provider_type: str):
        """Save user message in provider's native format"""
        try:
            binary_hash = self._get_current_binary_hash()
            if not binary_hash or not self.current_chat_id:
                log.log_warn("Cannot save user message: missing binary hash or chat ID")
                return
            
            from ..services.message_format_service import get_message_format_service
            from ..services.models.provider_types import ProviderType
            
            provider_enum = ProviderType(provider_type)
            format_service = get_message_format_service()
            
            # Create native user message
            context = self.context_service.get_current_context()
            enhanced_query = self._enhance_query_with_context(
                query_text, context, self.view.is_rag_enabled(), self.view.is_mcp_enabled()
            )
            native_message = format_service.create_user_message(enhanced_query, provider_enum)
            
            # Save to database
            self.analysis_db.save_native_message(
                binary_hash, str(self.current_chat_id), native_message, provider_type
            )
            
            log.log_debug(f"Saved user message in {provider_type} native format")
            
        except Exception as e:
            log.log_error(f"Failed to save user message in native format: {e}")
    
    def _save_assistant_response_native(self, response_content: str, tool_calls: List = None, 
                                      provider_type: str = None, finish_reason: str = None):
        """Save assistant response in provider's native format"""
        try:
            binary_hash = self._get_current_binary_hash()
            if not binary_hash or not self.current_chat_id:
                log.log_warn("Cannot save assistant response: missing binary hash or chat ID")
                return
            
            if not provider_type:
                # Get provider type from active provider
                active_provider = self.settings_service.get_active_llm_provider()
                if not active_provider:
                    log.log_warn("Cannot save assistant response: no active provider")
                    return
                provider_type = active_provider.get('provider_type', 'anthropic')
            
            from ..services.message_format_service import get_message_format_service
            from ..services.models.provider_types import ProviderType
            from ..services.models.llm_models import ChatMessage, MessageRole, ToolCall
            
            provider_enum = ProviderType(provider_type)
            format_service = get_message_format_service()
            
            # Convert tool calls to proper format if provided
            structured_tool_calls = []
            if tool_calls:
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        # Convert dict to ToolCall object
                        structured_tool_calls.append(ToolCall(
                            id=tc.get('id', ''),
                            name=tc.get('name', ''),
                            arguments=tc.get('arguments', {})
                        ))
                    else:
                        structured_tool_calls.append(tc)
            
            # Create ChatMessage
            assistant_message = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response_content,
                tool_calls=structured_tool_calls if structured_tool_calls else None
            )
            
            # Convert to native format
            native_message = format_service.to_native_format(assistant_message, provider_enum)
            
            # Save to database
            self.analysis_db.save_native_message(
                binary_hash, str(self.current_chat_id), native_message, provider_type
            )
            
            log.log_debug(f"Saved assistant response in {provider_type} native format")
            
        except Exception as e:
            log.log_error(f"Failed to save assistant response in native format: {e}")
    
    def _save_tool_response_native(self, tool_call_id: str, response_content: str, provider_type: str):
        """Save tool response in provider's native format"""
        try:
            binary_hash = self._get_current_binary_hash()
            if not binary_hash or not self.current_chat_id:
                log.log_warn("Cannot save tool response: missing binary hash or chat ID")
                return
            
            from ..services.message_format_service import get_message_format_service
            from ..services.models.provider_types import ProviderType
            from ..services.models.llm_models import ChatMessage, MessageRole
            
            provider_enum = ProviderType(provider_type)
            format_service = get_message_format_service()
            
            # Create tool response message
            tool_message = ChatMessage(
                role=MessageRole.TOOL,
                content=response_content,
                tool_call_id=tool_call_id
            )
            
            # Convert to native format
            native_message = format_service.to_native_format(tool_message, provider_enum)
            
            # Save to database
            self.analysis_db.save_native_message(
                binary_hash, str(self.current_chat_id), native_message, provider_type
            )
            
            log.log_debug(f"Saved tool response in {provider_type} native format")
            
        except Exception as e:
            log.log_error(f"Failed to save tool response in native format: {e}")
    
    def _create_native_message_callback(self) -> Callable[[Dict[str, Any], Any], None]:
        """Create callback function for provider native message updates"""
        def callback(native_message: Dict[str, Any], provider_type) -> None:
            try:
                # Only save assistant messages via callback - user and tool messages saved elsewhere
                if native_message.get("role") == "assistant":
                    binary_hash = self._current_query_binary_hash or self._get_current_binary_hash()
                    if binary_hash and self.current_chat_id:
                        provider_type_str = provider_type.value if hasattr(provider_type, 'value') else str(provider_type)
                        self.analysis_db.save_native_message(
                            binary_hash, str(self.current_chat_id), native_message, provider_type_str
                        )
                        log.log_debug(f"Saved assistant native message via callback for {provider_type_str}")
                    else:
                        log.log_warn("Cannot save native message: missing binary hash or chat ID")
            except Exception as e:
                log.log_error(f"Error in native message callback: {e}")
        
        return callback
    
    def _build_system_message(self, context: Dict[str, Any], rag_enabled: bool, mcp_enabled: bool) -> str:
        """Build system message with current context"""
        binary_info = context.get('binary_info', {})
        func_ctx = context.get('function_context')
        
        system_msg = f"""You are BinAssist, an AI assistant specialized in reverse engineering and binary analysis.

Current Analysis Context:
- Filename: {binary_info.get('filename', 'Unknown')}
- Path: {binary_info.get('filepath', 'Unknown')}
- Architecture: {binary_info.get('architecture', 'Unknown')}
- Platform: {binary_info.get('platform', 'Unknown')}
- Current Offset: {context.get('offset_hex', 'N/A')}"""
        
        if func_ctx:
            system_msg += f"\n- Current Function: {func_ctx['name']} ({func_ctx['start']} - {func_ctx['end']})"
            system_msg += f"\n- Function Size: {func_ctx['size']} bytes"
        
        system_msg += f"""

Capabilities:
- RAG (Documentation Search): {'Enabled' if rag_enabled else 'Disabled'}
- MCP (External Tools): {'Enabled' if mcp_enabled else 'Disabled'}

Instructions:
- Provide detailed, technical analysis relevant to reverse engineering
- Focus on the current binary context when applicable
- Use clear, structured responses with code examples when helpful
- If asked about specific code, relate it to the current offset/function when relevant"""

        if mcp_enabled:
            system_msg += """

Tool Usage Guidelines:
- When asked to analyze functions, binaries, or code, use multiple tool calls to gather comprehensive information
- Start with basic tools like get_binary_status, then use specific analysis tools based on what you discover
- Don't stop after one tool call - use the results to determine what additional information is needed
- For function analysis, typically use: get_binary_status → get_function_pseudo_c → get_references, etc.
- Always explain your analysis based on the tool results you gather
- If a tool result suggests more investigation is needed, make additional tool calls
- Provide a comprehensive answer only after you've gathered sufficient context through tool calls
"""
        
        return system_msg
    
    def _enhance_query_with_context(self, query_text: str, context: Dict[str, Any], rag_enabled: bool, mcp_enabled: bool) -> str:
        """Enhance user query with additional context from RAG and MCP"""
        enhanced_query = query_text
        
        # Add RAG context if enabled
        if rag_enabled:
            rag_context = self._get_rag_context(query_text)
            if rag_context:
                enhanced_query += f"\n\n{rag_context}"
        
        # Add MCP context if enabled (tools will be available to LLM)
        if mcp_enabled:
            mcp_context = "Use the available tool calls as needed."
            if mcp_context:
                enhanced_query += f"\n\n{mcp_context}"
        
        return enhanced_query
    
    def _get_rag_context(self, query_text: str) -> str:
        """Get RAG context for the query"""
        try:
            request = SearchRequest(
                query=query_text,
                search_type=SearchType.HYBRID,
                max_results=3,
                similarity_threshold=0.3,
                include_metadata=True
            )
            
            results = rag_service.search(request)
            
            if not results:
                return ""
            
            rag_context = "## Additional Reference Context\n"
            rag_context += "The following documentation may be helpful:\n\n"
            
            for i, result in enumerate(results, 1):
                score_percent = int(result.score * 100)
                rag_context += f"### Reference {i} (Relevance: {score_percent}%)\n"
                rag_context += f"**Source:** {result.filename}\n"
                rag_context += f"**Content:** {result.snippet}\n\n"
            
            return rag_context
            
        except Exception as e:
            log.log_error(f"Error getting RAG context: {e}")
            return ""
    
    def _get_mcp_context(self) -> str:
        """Get MCP context (available tools)"""
        try:
            # Use connection manager to get available tools
            if not self.mcp_connection_manager.is_available():
                return ""
            
            tools = self.mcp_connection_manager.get_available_tools_for_llm()
            
            if not tools:
                return ""
            
            mcp_context = "## Available MCP Tools\n"
            mcp_context += "The following external tools are available for your assistance:\n\n"
            
            for tool in tools[:5]:  # Show first 5 tools
                tool_name = tool.get("function", {}).get("name", "Unknown")
                tool_desc = tool.get("function", {}).get("description", "No description")
                mcp_context += f"- **{tool_name}**: {tool_desc}\n"
            
            if len(tools) > 5:
                mcp_context += f"\n... and {len(tools) - 5} more tools available.\n"
            
            return mcp_context
            
        except Exception as e:
            log.log_error(f"Error getting MCP context: {e}")
            return ""
    
    def _start_llm_query(self, messages: List[Dict[str, Any]], provider_config: dict, mcp_tools: List[Dict[str, Any]] = None):
        """Start the LLM query thread with optional MCP tools"""
        # Clear previous response buffer and tool state
        self._llm_response_buffer = ""
        self._pending_tool_calls = []
        self._tool_results = []
        self._tool_execution_active = False
        self._continuation_complete = False
        self._tool_call_attempts = {}  # Reset tool call attempt tracking

        # Start debounced rendering for streaming responses
        self._debounced_renderer.start()

        # Create and start the LLM query thread
        native_callback = self._create_native_message_callback()
        self.llm_thread = LLMQueryThread(messages, provider_config, self.llm_factory, mcp_tools, native_callback)
        self.llm_thread.response_chunk.connect(self._on_llm_response_chunk)
        self.llm_thread.response_complete.connect(self._on_llm_response_complete)
        self.llm_thread.response_error.connect(self._on_llm_response_error)
        self.llm_thread.tool_calls_detected.connect(self._on_tool_calls_detected)
        self.llm_thread.stop_reason_received.connect(self._on_stop_reason_received)
        self.llm_thread.start()
    
    def _on_llm_response_chunk(self, chunk: str):
        """Handle streaming response chunk from LLM"""
        self._llm_response_buffer += chunk

        # Feed chunk to debounced renderer instead of immediate update
        # This accumulates chunks and renders periodically (every 1 second)
        # instead of on every delta, reducing UI updates by 10-40x
        self._debounced_renderer.on_delta(chunk)

        # Update the last assistant message in memory (no UI update yet)
        self._update_last_assistant_message(self._llm_response_buffer)
    
    def _on_llm_response_complete(self):
        """Handle completion of LLM response"""
        log.log_info(f"LLM response complete called - tool_execution_active: {self._tool_execution_active}")
        log.log_info(f"LLM response buffer: {len(self._llm_response_buffer)} chars")
        log.log_info(f"Response content: '{self._llm_response_buffer[:200]}{'...' if len(self._llm_response_buffer) > 200 else ''}'")

        # Stop debounced rendering and do final render
        # This ensures the final complete response is displayed
        self._debounced_renderer.complete(self._llm_response_buffer)

        # Only complete if no tool execution is active
        if not self._tool_execution_active:
            # Check if we have content in the response buffer OR if the assistant message was already updated during streaming
            has_content = bool(self._llm_response_buffer)

            # Check if the assistant message already has content from streaming
            if not has_content and self.current_chat_id in self.chats:
                messages = self.chats[self.current_chat_id]["messages"]
                if messages and messages[-1]["role"] == "assistant":
                    existing_content = messages[-1]["content"]
                    if existing_content and existing_content != "*Thinking...*":
                        has_content = True
                        # Use the existing content as our response buffer for RLHF and final update
                        self._llm_response_buffer = existing_content

            # Final update of the assistant message (debouncer already did the display update)
            if has_content:
                log.log_info("Updating last assistant message with LLM response")
                self._update_last_assistant_message(self._llm_response_buffer, final_update=True)
                # Update RLHF context with final response
                self._update_rlhf_response(self._llm_response_buffer)
            else:
                log.log_debug("No LLM response content - this might be a tool-only response")

            # Clean up
            self._cleanup_llm_thread()
            self._query_active = False
            self._current_query_binary_hash = None
            self.view.set_query_running(False)
        else:
            log.log_info("Tool execution active - not completing LLM response yet")
    
    def _on_llm_response_error(self, error: str):
        """Handle LLM response error"""
        log.log_error(f"LLM query failed: {error}")

        # Cancel debounced rendering
        self._debounced_renderer.cancel()

        # Cancel any pending tool execution
        self._tool_execution_active = False

        error_response = f"**Error**: {error}\n\n*The LLM query failed. Please check your provider configuration and try again.*"
        self._update_last_assistant_message(error_response, final_update=True)
        self._update_chat_display()

        # Clean up
        self._cleanup_llm_thread()
        self._query_active = False
        self._current_query_binary_hash = None
        self.view.set_query_running(False)
    
    def _on_tool_calls_detected(self, tool_calls: List[ToolCall]):
        """Handle tool calls detected from LLM response or continuation"""
        # Check for loop detection (for all tool calls, not just additional ones)
        if not self._should_allow_tool_execution(tool_calls):
            # Stop execution and complete conversation
            self._add_message_to_chat(self.current_chat_id, "assistant", "**Note:** Tool execution stopped to prevent infinite loop.")
            self._cleanup_continuation_thread()
            self._tool_execution_active = False
            self._query_active = False
            self._current_query_binary_hash = None
            self.view.set_query_running(False)
            return
        
        # Determine if this is from continuation thread (additional tools) or initial LLM response
        is_additional_tools = hasattr(self, 'continuation_thread') and self.continuation_thread
        
        # For initial tool calls, check if execution is already active
        if self._tool_execution_active and not is_additional_tools:
            log.log_warn("Tool execution already active, ignoring new tool calls")
            return
        
        # If this is additional tool calls (continuation), clean up the continuation thread
        if is_additional_tools:
            self._cleanup_continuation_thread()
        
        # Store tool calls and mark execution as active
        self._pending_tool_calls = tool_calls
        self._tool_execution_active = True
        
        # Save the initial assistant response (before tool execution) - only for initial calls
        if self._llm_response_buffer and not hasattr(self, '_initial_response_saved'):
            self._update_last_assistant_message(self._llm_response_buffer, final_update=True)
            self._initial_response_saved = True
        
        # Execute tool calls using common helpers
        self._create_tool_call_messages(tool_calls)
        self._setup_tool_executor(tool_calls)
    
    def _create_tool_call_messages(self, tool_calls: List[ToolCall]):
        """Create and add tool call messages to chat for display"""
        for i, tool_call in enumerate(tool_calls):
            # Log tool call details (similar to original logging)
            log.log_info(f"Tool call {i}: {tool_call.name} with args: {tool_call.arguments}")
            
            tool_call_content = f"**Calling tool:** `{tool_call.name}`"
            if tool_call.arguments:
                import json
                try:
                    args_str = json.dumps(tool_call.arguments)
                    tool_call_content += f"\n\n**Arguments:**`{args_str}`"
                except:
                    tool_call_content += f"\n\n**Arguments:** {tool_call.arguments}"
            
            self._add_message_to_chat(self.current_chat_id, "tool_call", tool_call_content)
        
        self._update_chat_display()
    
    def _setup_tool_executor(self, tool_calls: List[ToolCall]):
        """Create, configure, and start the tool executor thread"""
        # Create tool executor thread (no parent since QueryController is not a QObject)
        self.tool_executor_thread = ToolExecutorThread(
            self.mcp_orchestrator,
            tool_calls,
            self._llm_response_buffer
        )
        
        # Connect signals
        self.tool_executor_thread.tool_execution_update.connect(self._on_tool_execution_update)
        self.tool_executor_thread.tool_execution_error.connect(self._handle_tool_execution_error_signal)
        # CRITICAL: tool_results_ready handles ALL results and continues conversation
        self.tool_executor_thread.tool_results_ready.connect(self._on_tool_results_ready)
        
        # Start thread
        self.tool_executor_thread.start()
    
    def _should_allow_tool_execution(self, tool_calls: List[ToolCall]) -> bool:
        """Check if tool execution should proceed (includes loop detection)"""
        # Initialize tool call tracking if needed
        if not hasattr(self, '_tool_call_attempts'):
            self._tool_call_attempts = {}
        
        # Check for potential infinite loops
        for tool_call in tool_calls:
            call_key = f"{tool_call.name}:{hash(str(tool_call.arguments))}"
            current_count = self._tool_call_attempts.get(call_key, 0) + 1
            self._tool_call_attempts[call_key] = current_count
            
            # Log attempt counts for visibility
            if current_count > 1:
                log.log_info(f"Tool call {tool_call.name} attempt #{current_count}")
            
            if current_count > 3:  # Max 3 attempts per unique tool call
                log.log_warn(f"Tool call loop detected for {tool_call.name}, stopping after {current_count-1} attempts")
                return False
        
        return True
    
    def _on_stop_reason_received(self, stop_reason: str):
        """Handle stop reason from LLM response"""
        log.log_info(f"Stop reason received: {stop_reason}, tool_execution_active: {self._tool_execution_active}")
        
        if stop_reason == "end_turn":
            # LLM is completely done - this is the final response
            log.log_info("LLM finished with end_turn - completing response")
            self._on_llm_response_complete()
        elif stop_reason == "tool_use":
            # LLM wants to use tools - tool execution will handle this
            log.log_info("LLM finished with tool_use - tools will be executed")
            # Tool execution is handled by the tool_calls_detected signal
        elif stop_reason != "tool_calls" and not self._tool_execution_active:
            # Other stop reasons without active tool execution - complete normally
            log.log_info(f"LLM finished with {stop_reason} - completing response")
            self._on_llm_response_complete()
    
    # Removed _execute_tools_async - tool execution now handled by ToolExecutorThread
    
    def _on_tool_execution_update(self, response_content: str):
        """Handle tool execution update - just refresh the display"""
        self._update_chat_display()
    
    # Removed _update_continuation_response_ui - handled by continuation thread signals
    
    def _handle_tool_execution_error(self, response_content: str):
        """Handle tool execution error (runs on main thread)"""
        try:
            # Add error as a new message
            self._add_message_to_chat(self.current_chat_id, "error", response_content)
            self._update_chat_display()
            
            # Complete the response on error
            self._tool_execution_active = False
            self._query_active = False
            self._current_query_binary_hash = None
            self.view.set_query_running(False)
        except Exception as e:
            log.log_error(f"Error handling tool execution error: {e}")
    
    # Removed _complete_tool_conversation - handled by continuation thread signals
    
    # Removed _complete_tool_conversation_with_final_ui_update - handled by continuation thread signals
    
    def _handle_tool_execution_error_signal(self, error_msg: str):
        """Handle tool execution error from signal"""
        self._handle_tool_execution_error(f"**Tool Execution Error:** {error_msg}")
    
    def _on_tool_results_ready(self, results: List[ToolResult]):
        """Handle tool results ready signal - ALWAYS continue conversation"""
        self._tool_results = results

        # Add tool response messages for each result
        # Only show success/failure status to user, but keep full content for LLM context
        for tool_call, result in zip(self._pending_tool_calls, results):
            # User-facing message: Just show success/failure
            if result.error:
                user_message = f"🔧 Tool `{tool_call.name}` **failed**: {result.error}"
            else:
                # Show success with content length instead of full content
                content_preview = f"{len(result.content)} characters" if result.content else "no output"
                user_message = f"✅ Tool `{tool_call.name}` **succeeded** ({content_preview})"

            # Add user-facing message for display only (not saved to DB or sent to LLM)
            self._add_message_to_chat(self.current_chat_id, "tool_status", user_message, save_to_db=False)

        self._update_chat_display()
        
        # ALWAYS continue LLM conversation with results (success or failure)
        try:
            
            # Prepare conversation messages for continuation
            messages = self._prepare_continuation_messages(self._pending_tool_calls, results)
            
            # Get active provider and MCP tools
            active_provider = self.settings_service.get_active_llm_provider()
            if not active_provider:
                self._handle_tool_execution_error("**Error:** No active LLM provider")
                return
            
            mcp_tools = []
            if self.view.is_mcp_enabled():
                mcp_tools = self.mcp_connection_manager.get_available_tools_for_llm()
            
            # Create and start continuation thread
            native_callback = self._create_native_message_callback()
            self.continuation_thread = LLMContinuationThread(
                self.llm_factory,
                active_provider,
                messages,
                mcp_tools,
                native_callback
            )

            # Connect signals
            self.continuation_thread.continuation_chunk.connect(self._on_continuation_chunk)
            self.continuation_thread.continuation_complete.connect(self._on_continuation_complete)
            self.continuation_thread.continuation_error.connect(self._on_continuation_error)
            self.continuation_thread.additional_tools_detected.connect(self._on_tool_calls_detected)

            # CRITICAL: Reset the LLM response buffer for this NEW continuation
            # Each continuation should start fresh, not accumulate previous responses
            self._llm_response_buffer = ""

            # Restart debounced renderer for this new continuation
            self._debounced_renderer.start()

            # Start thread
            self.continuation_thread.start()
            
        except Exception as e:
            log.log_error(f"Error starting continuation thread: {e}")
            self._handle_tool_execution_error(f"**Error:** {str(e)}")
    
    def _prepare_continuation_messages(self, tool_calls: List[ToolCall], results: List[ToolResult]) -> List[Dict[str, Any]]:
        """Prepare messages for LLM continuation after tool execution"""
        import json
        try:
            # Get active provider to format tool results properly
            active_provider = self.settings_service.get_active_llm_provider()
            if not active_provider:
                return []
            
            provider = self.llm_factory.create_provider(active_provider)
            
            # Format tool results for the specific provider
            tool_messages = provider.format_tool_results_for_continuation(tool_calls, [r.content for r in results])
            
            # Save tool result messages to native storage
            try:
                binary_hash = self._get_current_binary_hash()
                if binary_hash and self.current_chat_id and tool_messages:
                    provider_type = active_provider.get('provider_type', 'anthropic')
                    for tool_msg in tool_messages:
                        self.analysis_db.save_native_message(
                            binary_hash, str(self.current_chat_id), tool_msg, provider_type
                        )
                    log.log_debug(f"Saved {len(tool_messages)} tool result messages to native storage")
            except Exception as e:
                log.log_error(f"Failed to save tool result messages to native storage: {e}")
            
            # Prepare conversation history up to this point
            conversation_messages = []
            
            # Add system message
            context = self.context_service.get_current_context()
            system_content = self._build_system_message(context, self.view.is_rag_enabled(), self.view.is_mcp_enabled())
            conversation_messages.append({"role": "system", "content": system_content})
            
            # Add chat history from native storage
            try:
                binary_hash = self._get_current_binary_hash()
                if binary_hash and self.current_chat_id:
                    provider_type = active_provider.get('provider_type', 'anthropic')
                    native_messages = self.analysis_db.get_native_messages_for_provider(
                        binary_hash, str(self.current_chat_id), provider_type
                    )

                    # Add native messages directly (they're already in provider format)
                    conversation_messages.extend(native_messages)
            except Exception as e:
                log.log_warn(f"Failed to load native messages for continuation: {e}")
                # Fallback to in-memory messages if native storage fails
                if self.current_chat_id in self.chats:
                    chat_messages = self.chats[self.current_chat_id]["messages"]
                    for msg in chat_messages[:-1]:  # Exclude current placeholder response
                        if msg["role"] in ["user", "assistant"]:
                            conversation_messages.append({
                                "role": msg["role"], 
                                "content": msg["content"]
                            })
            
            # NOTE: Assistant message with tool calls and tool results are now loaded from native storage above
            # No need to manually add them again - that would cause duplicates
            
            return conversation_messages
            
        except Exception as e:
            log.log_error(f"Error preparing continuation messages: {e}")
            return []
    
    def _on_continuation_chunk(self, chunk: str):
        """Handle streaming continuation chunk"""
        # Add the chunk directly to the main LLM response buffer (just like initial streaming)
        self._llm_response_buffer += chunk

        # Feed chunk to debounced renderer (same as initial streaming)
        self._debounced_renderer.on_delta(chunk)

        # For continuation, we need to add a new assistant message or update existing one
        # This happens in memory only - debounced renderer handles UI updates
        self._update_or_create_continuation_message(self._llm_response_buffer)
    
    def _update_or_create_continuation_message(self, content: str):
        """Update existing assistant message or create new one for continuation"""
        if self.current_chat_id not in self.chats:
            return
        
        messages = self.chats[self.current_chat_id]["messages"]
        
        # Check if last message is assistant message we can update
        if messages and messages[-1]["role"] == "assistant":
            # Update existing assistant message
            messages[-1]["content"] = content
            messages[-1]["timestamp"] = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
            self.chats[self.current_chat_id]["updated"] = messages[-1]["timestamp"]
        else:
            # Create new assistant message for continuation
            timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
            new_message = {
                "role": "assistant",
                "content": content,
                "timestamp": timestamp
            }
            messages.append(new_message)
            self.chats[self.current_chat_id]["updated"] = timestamp
    
    def _on_continuation_complete(self, final_response: str):
        """Handle continuation completion - this means the LLM is truly done (no more tool calls)"""
        try:
            # Stop debounced rendering and do final render with full markdown
            self._debounced_renderer.complete(self._llm_response_buffer)

            # Update final assistant message and save to storage
            if self._llm_response_buffer:
                self._update_last_assistant_message(self._llm_response_buffer, final_update=True)
                self._update_rlhf_response(self._llm_response_buffer)

            # Final display with full markdown rendering (not plain text)
            self._update_chat_display()

            # Clean up and end the query
            self._cleanup_continuation_thread()
            self._tool_execution_active = False
            self._query_active = False
            self._current_query_binary_hash = None
            self.view.set_query_running(False)

        except Exception as e:
            log.log_error(f"Error handling continuation completion: {e}")
    
    def _on_continuation_error(self, error: str):
        """Handle continuation error"""
        log.log_error(f"Continuation error: {error}")
        self._handle_tool_execution_error(f"**Continuation Error:** {error}")
    
    def _cleanup_continuation_thread(self):
        """Safely cleanup the continuation thread"""
        if hasattr(self, 'continuation_thread') and self.continuation_thread:
            if self.continuation_thread.isRunning():
                self.continuation_thread.quit()
                self.continuation_thread.wait(5000)  # Wait up to 5 seconds
            self.continuation_thread.deleteLater()
            self.continuation_thread = None
    
    # Removed _continue_llm_with_tool_results - using LLMContinuationThread instead
    
    def _cleanup_llm_thread(self):
        """Safely cleanup the LLM thread"""
        if hasattr(self, 'llm_thread') and self.llm_thread:
            if self.llm_thread.isRunning():
                self.llm_thread.quit()
                self.llm_thread.wait(5000)  # Wait up to 5 seconds
            self.llm_thread.deleteLater()
            self.llm_thread = None
        
        # Cleanup tool executor thread if exists
        if hasattr(self, 'tool_executor_thread') and self.tool_executor_thread:
            if self.tool_executor_thread.isRunning():
                self.tool_executor_thread.quit()
                self.tool_executor_thread.wait(5000)  # Wait up to 5 seconds
            self.tool_executor_thread.deleteLater()
            self.tool_executor_thread = None
        
        # Cleanup continuation thread if exists
        self._cleanup_continuation_thread()
        
        # Clear response buffer
        self._llm_response_buffer = ""
    
    def _update_last_assistant_message(self, content: str, final_update: bool = False):
        """Update the last assistant message in the current chat"""
        if self.current_chat_id not in self.chats:
            log.log_debug("_update_last_assistant_message: No current chat")
            return
        
        messages = self.chats[self.current_chat_id]["messages"]
        if messages and messages[-1]["role"] == "assistant":
            old_length = len(messages[-1]["content"])
            messages[-1]["content"] = content
            messages[-1]["timestamp"] = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
            self.chats[self.current_chat_id]["updated"] = messages[-1]["timestamp"]
            log.log_debug(f"_update_last_assistant_message: Updated assistant message from {old_length} to {len(content)} chars")
            
            # Save to native format on final update
            if final_update:
                try:
                    # Get active provider type for native saving
                    active_provider = self.settings_service.get_active_llm_provider()
                    if active_provider:
                        provider_type = active_provider.get('provider_type', 'anthropic')
                        self._save_assistant_response_native(content, provider_type=provider_type)
                    
                    # Also save to legacy format for backward compatibility
                    binary_hash = self._current_query_binary_hash
                    if binary_hash:
                        message_order = len(messages) - 1
                        self.analysis_db.save_chat_message(
                            binary_hash, str(self.current_chat_id), "assistant", content,
                            metadata={"message_order": message_order}
                        )
                    else:
                        log.log_warn("No binary hash available for persisting assistant message")
                except Exception as e:
                    log.log_error(f"Failed to persist updated assistant message: {e}")
        else:
            log.log_debug(f"_update_last_assistant_message: Last message is not assistant (messages={len(messages)}, last_role={messages[-1]['role'] if messages else 'none'})")
    
    # Database helper methods
    
    def _get_current_binary_hash(self) -> Optional[str]:
        """
        Get current binary hash from cached value.

        The hash is calculated ONCE in set_binary_view() and cached
        in the context service. This avoids recalculating the hash
        on every UI update.

        Returns:
            Cached binary hash, or None if not available
        """
        return self.context_service.get_binary_hash()
    
    # RLHF methods
    
    def _track_rlhf_context(self, model_name: str, prompt: str, messages: List[Dict[str, Any]]):
        """Track current query context for RLHF feedback"""
        # Extract system message if present
        system_message = ""
        for msg in messages:
            if msg.get('role') == 'system':
                system_message = msg.get('content', '')
                break
        
        self._current_rlhf_context = {
            'model_name': model_name,
            'prompt': prompt,
            'system': system_message,
            'response': None  # Will be set when response is complete
        }
    
    def _update_rlhf_response(self, response: str):
        """Update the RLHF context with the completed response"""
        self._current_rlhf_context['response'] = response
    
    def handle_rlhf_feedback(self, is_upvote: bool):
        """Handle RLHF feedback submission"""
        try:
            # Debug the current context
            log.log_info(f"RLHF feedback requested: {'upvote' if is_upvote else 'downvote'}")
            #log.log_info(f"Current RLHF context: {self._current_rlhf_context}")
            
            # Check for incomplete context more specifically
            if not self._current_rlhf_context.get('model_name'):
                log.log_warn("RLHF feedback failed: No model name in context")
                return
            if not self._current_rlhf_context.get('prompt'):
                log.log_warn("RLHF feedback failed: No prompt in context") 
                return
            if not self._current_rlhf_context.get('response'):
                log.log_warn("RLHF feedback failed: No response in context - query may not be complete yet")
                return
            
            # Get binary metadata
            metadata = self.context_service.get_binary_metadata_for_rlhf()
            metadata_json = RLHFFeedbackEntry.create_metadata_json(
                metadata['filename'], metadata['size'], metadata['sha256']
            )
            
            # Create feedback entry
            feedback_entry = RLHFFeedbackEntry(
                model_name=self._current_rlhf_context['model_name'],
                prompt=self._current_rlhf_context['prompt'],
                system=self._current_rlhf_context['system'],
                response=self._current_rlhf_context['response'],
                feedback=is_upvote,
                timestamp="",  # Will be set by service
                metadata=metadata_json
            )
            
            # Store feedback
            success = rlhf_service.store_feedback(feedback_entry)
            if success:
                feedback_type = "upvote" if is_upvote else "downvote"
                log.log_info(f"RLHF {feedback_type} feedback stored successfully")
            else:
                log.log_error("Failed to store RLHF feedback")
                
        except Exception as e:
            log.log_error(f"Error handling RLHF feedback: {e}")

    # =========================================================================
    # ReAct (Agentic Mode) Methods
    # =========================================================================

    def _submit_agentic_query(self, query_text: str):
        """Handle agentic (ReAct) query submission"""
        log.log_info(f"Agentic query submitted: {query_text[:50]}...")

        # Check if query is already active
        if self._query_active or self._react_active:
            log.log_warn("Query already active, ignoring agentic request")
            return

        try:
            # Get active LLM provider
            active_provider = self.settings_service.get_active_llm_provider()
            if not active_provider:
                self._handle_no_llm_provider(query_text)
                return

            # Ensure we have an active chat
            if self.current_chat_id is None:
                self.new_chat()

            # Set active state
            self._react_active = True
            self._current_query_binary_hash = self._get_current_binary_hash()
            self.view.set_query_running(True)

            # Reset auto-scroll to follow new response by default
            self.view.enable_auto_scroll()

            # Add user query to chat (both legacy and native formats)
            self._add_message_to_chat(self.current_chat_id, "user", query_text)

            # Save user query in native format for the active provider
            provider_type = active_provider.get('provider_type', 'anthropic')
            self._save_user_message_native(query_text, provider_type)

            # Add placeholder for assistant response (will be filled by content chunks)
            self._add_message_to_chat(self.current_chat_id, "assistant",
                                      "*Initializing agentic investigation...*", save_to_db=False)

            # Update display to show user query and assistant placeholder
            self._update_chat_display()

            log.log_debug(f"Agentic query: Added user message and assistant placeholder to chat {self.current_chat_id}")

            # Get context and MCP tools
            context = self.context_service.get_current_context()
            initial_context = self._format_initial_context_for_react(context)

            # MCP tools are required for agentic mode
            mcp_tools = []
            if self.mcp_connection_manager.ensure_connections():
                mcp_tools = self.mcp_connection_manager.get_available_tools_for_llm()
                log.log_info(f"Agentic mode with {len(mcp_tools)} tools available")
            else:
                log.log_warn("MCP connection failed for agentic mode")

            # Create provider instance
            provider = self.llm_factory.create_provider(active_provider)

            # Create ReAct config
            config = ReActConfig(
                max_iterations=15,
                reflection_enabled=True
            )

            # Start ReAct thread
            self._start_react_analysis(query_text, initial_context, provider, mcp_tools, config)

        except Exception as e:
            error_msg = f"Exception in submit_agentic_query: {str(e)}"
            log.log_error(error_msg)
            self._handle_react_error(str(e))

    def _format_initial_context_for_react(self, context: Dict[str, Any]) -> str:
        """Format initial binary context for ReAct agent"""
        binary_info = context.get('binary_info', {})
        func_ctx = context.get('function_context')

        ctx_str = f"""Binary: {binary_info.get('filename', 'Unknown')}
Architecture: {binary_info.get('architecture', 'Unknown')}
Current Offset: {context.get('offset_hex', 'N/A')}"""

        if func_ctx:
            ctx_str += f"\nCurrent Function: {func_ctx['name']} ({func_ctx['start']} - {func_ctx['end']})"

        return ctx_str

    def _start_react_analysis(self, objective: str, initial_context: str,
                              provider, mcp_tools: List[Dict[str, Any]],
                              config: ReActConfig):
        """Start the ReAct analysis thread"""
        log.log_info("Starting ReAct analysis thread")

        # Start debounced rendering
        self._debounced_renderer.start()
        self._llm_response_buffer = ""

        # Create and configure thread
        self._react_thread = ReActOrchestratorThread(
            objective=objective,
            initial_context=initial_context,
            llm_provider=provider,
            mcp_orchestrator=self.mcp_orchestrator,
            mcp_tools=mcp_tools,
            config=config
        )

        # Connect signals
        self._react_thread.planning_complete.connect(self._on_react_planning_complete)
        self._react_thread.todos_updated.connect(self._on_react_todos_updated)
        self._react_thread.iteration_started.connect(self._on_react_iteration_started)
        self._react_thread.iteration_complete.connect(self._on_react_iteration_complete)
        self._react_thread.finding_discovered.connect(self._on_react_finding)
        self._react_thread.progress_update.connect(self._on_react_progress)
        self._react_thread.content_chunk.connect(self._on_react_content_chunk)
        self._react_thread.analysis_complete.connect(self._on_react_complete)
        self._react_thread.analysis_error.connect(self._on_react_error)

        # Start thread
        self._react_thread.start()
        log.log_info("ReAct analysis thread started")

    def _on_react_planning_complete(self, todos_formatted: str):
        """Handle planning phase completion"""
        log.log_debug(f"ReAct planning complete: {todos_formatted[:100]}...")

    def _on_react_todos_updated(self, todos_formatted: str):
        """Handle todo list updates"""
        log.log_debug(f"ReAct todos updated")

    def _on_react_iteration_started(self, iteration: int, current_todo: str):
        """Handle iteration start"""
        log.log_debug(f"ReAct iteration {iteration} started: {current_todo[:50]}...")

    def _on_react_iteration_complete(self, iteration: int, summary: str):
        """Handle iteration completion"""
        log.log_debug(f"ReAct iteration {iteration} complete")

    def _on_react_finding(self, finding: str):
        """Handle new finding discovered"""
        log.log_debug(f"ReAct finding: {finding[:50]}...")

    def _on_react_progress(self, message: str, iteration: int):
        """Handle progress update"""
        log.log_debug(f"ReAct progress: {message} (iteration {iteration})")

    def _on_react_content_chunk(self, chunk: str):
        """Handle streaming content chunk from ReAct"""
        if not chunk:
            return

        self._llm_response_buffer += chunk

        # Update the assistant message in chat (will be rendered by debounced callback)
        if self.current_chat_id and self.current_chat_id in self.chats:
            chat = self.chats[self.current_chat_id]
            if chat["messages"] and chat["messages"][-1]["role"] == "assistant":
                chat["messages"][-1]["content"] = self._llm_response_buffer
            else:
                log.log_warn("No assistant message found to update in chat")
        else:
            log.log_warn(f"Chat {self.current_chat_id} not found for content update")

        # Feed to debounced renderer for periodic UI updates
        self._debounced_renderer.on_delta(chunk)

    def _on_react_complete(self, result: ReActResult):
        """Handle ReAct analysis completion"""
        log.log_info(f"ReAct complete: {result.status.value}, {result.iteration_count} iterations, {result.tool_call_count} tool calls")

        # Stop debounced rendering and do final render
        self._debounced_renderer.complete()

        # Format final response with statistics
        status_emoji = "✅" if result.status == ReActStatus.SUCCESS else "⚠️"
        final_content = f"""{self._llm_response_buffer}

---

## {status_emoji} Investigation Complete

| Metric | Value |
|--------|-------|
| Status | {result.status.value} |
| Iterations | {result.iteration_count} |
| Tool Calls | {result.tool_call_count} |
| Duration | {result.duration_seconds:.1f}s |
"""

        # Update assistant message with final content
        self._update_last_assistant_message(final_content, final_update=True)
        self._update_chat_display()

        # Cleanup
        self._cleanup_react_thread()
        self._react_active = False
        self._current_query_binary_hash = None
        self.view.set_query_running(False)

    def _on_react_error(self, error: str):
        """Handle ReAct analysis error"""
        self._handle_react_error(error)

    def _handle_react_error(self, error: str):
        """Handle ReAct error"""
        log.log_error(f"ReAct error: {error}")

        # Cancel debounced rendering
        self._debounced_renderer.cancel()

        error_response = f"**Agentic Investigation Error:** {error}"
        self._update_last_assistant_message(error_response, final_update=True)
        self._update_chat_display()

        # Cleanup
        self._cleanup_react_thread()
        self._react_active = False
        self._current_query_binary_hash = None
        self.view.set_query_running(False)

    def _cleanup_react_thread(self):
        """Safely cleanup the ReAct thread"""
        if hasattr(self, '_react_thread') and self._react_thread:
            try:
                if self._react_thread.isRunning():
                    # Request cancellation and wait for thread to finish
                    self._react_thread.cancel()
                    self._react_thread.wait(3000)  # Wait up to 3 seconds
                # Just set to None, let Qt handle cleanup
                self._react_thread = None
            except Exception as e:
                log.log_error(f"Error during ReAct thread cleanup: {e}")
                self._react_thread = None
        log.log_debug("ReAct thread cleaned up")