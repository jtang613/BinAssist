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
    """

    update_response = QtCore.Signal(dict)
    streaming_finished = QtCore.Signal()

    def __init__(self, provider, query: str, system: str, tools=None, completion_callback=None) -> None:
        """
        Initializes the thread with the necessary parameters for making a streaming API call.

        Parameters:
            provider: The API provider instance to use for making calls.
            query (str): The user's query to be processed.
            system (str): System-level instructions or context for the API call.
            tools (list): A list of tools that the LLM can call during the response.
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
            
            # Use provider's streaming chat completion
            response_accumulator = ""
            
            def handle_streaming_response(accumulated_text):
                nonlocal response_accumulator
                response_accumulator = accumulated_text
                if self.running:
                    self.update_response.emit({"response": accumulated_text})
            
            def handle_function_call_response(tool_calls):
                nonlocal response_accumulator
                # For function calls, use our provider-agnostic ToolCall interface
                log.log_info(f"[BinAssist] StreamingThread: handle_function_call_response called with {len(tool_calls)} tool calls")
                log.log_debug(f"[BinAssist] StreamingThread: Tool calls received: {tool_calls}")
                
                if self.running:
                    try:
                        # Convert our ToolCall objects to OpenAI format for UI compatibility
                        # This is the proper place to do UI-specific formatting
                        from openai.types.chat import ChatCompletionMessageToolCall
                        import json
                        
                        openai_tool_calls = []
                        for tool_call in tool_calls:
                            openai_tool_call = ChatCompletionMessageToolCall(
                                id=tool_call.id,
                                type="function",
                                function={
                                    "name": tool_call.name,
                                    "arguments": json.dumps(tool_call.arguments)
                                }
                            )
                            openai_tool_calls.append(openai_tool_call)
                        
                        log.log_info(f"[BinAssist] StreamingThread: Converted {len(openai_tool_calls)} provider-agnostic tool calls to UI format")
                        log.log_info(f"[BinAssist] StreamingThread: Emitting tool calls to UI via update_response signal")
                        
                        self.update_response.emit({"response": openai_tool_calls})
                        log.log_info(f"[BinAssist] StreamingThread: Signal emitted successfully")
                        response_accumulator = f"Function calls generated: {len(tool_calls)} tool calls"
                        
                    except Exception as e:
                        log.log_error(f"[BinAssist] StreamingThread: Error converting/emitting tool calls: {e}")
                        # Fallback - emit the original tool calls
                        self.update_response.emit({"response": tool_calls})
                else:
                    log.log_error(f"[BinAssist] StreamingThread: Not running, skipping tool call emission")
            
            def handle_completion():
                """Called when streaming/function calling is complete."""
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
