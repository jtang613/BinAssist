from binaryninja.settings import Settings
from PySide6 import QtCore
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall

import re
import json
import random
import string
import logging

class StreamingThread(QtCore.QThread):
    """
    A thread for managing streaming API calls to an OpenAI model, specifically designed to handle long-running 
    requests that stream data back as they process. This class handles the setup and execution of these requests 
    and signals the main application thread upon updates or completion.
    """

    update_response = QtCore.Signal(dict)

    def __init__(self, client: OpenAI, model: str, max_tokens: int, query: str, system: str, tools=None) -> None:
        """
        Initializes the thread with the necessary parameters for making a streaming API call.

        Parameters:
            client (OpenAI): The OpenAI client used for making API calls.
            model (str): The model to use.
            max_tokens (int): The max number of context tokens.
            query (str): The user's query to be processed.
            system (str): System-level instructions or context for the API call.
            tools (list): A list of tools that the LLM can call during the response.
        """
        super().__init__()
        
        # Setup logging
        self.logger = logging.getLogger(f"binassist.streaming_thread.{model}")
        self.logger.info(f"Initializing StreamingThread for model: {model}")
        
        try:
            self.settings = Settings()
            self.logger.debug("Settings initialized")
            
            self.client = client
            self.model = model
            self.max_tokens = max_tokens
            self.query = query
            self.system = system
            self.tools = tools or None
            self.running = True
            
            self.logger.debug(f"Thread initialized - model: {model}, max_tokens: {max_tokens}")
            self.logger.debug(f"Query length: {len(query)}, system length: {len(system)}, has tools: {tools is not None}")
            self.logger.info("StreamingThread initialization completed")
            
        except Exception as e:
            self.logger.error(f"Error during StreamingThread initialization: {type(e).__name__}: {e}")
            self.logger.exception("Full traceback:")
            raise

    def run(self) -> None:
        """
        Executes the streaming API call in a separate thread, processing responses as they come in and 
        signaling the main thread upon updates or when an error occurs.
        """
        self.logger.info(f"Starting thread execution for model: {self.model}")
        
        try:
            # Check if we're dealing with o* models which need special handling
            is_reasoning_model = (self.model.lower().startswith("o1-") or 
                                 self.model.lower().startswith("o3-") or 
                                 self.model.lower().startswith("o4-"))
            self.logger.debug(f"Is reasoning model (o*): {is_reasoning_model}")
            
            self.logger.debug("Preparing API call parameters")
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": self.query}
                ],
                "stream": False if self.tools else True,
                "tools": self.tools,
            }
            
            # o* models use max_completion_tokens instead of max_tokens
            if is_reasoning_model:
                api_params["max_completion_tokens"] = self.max_tokens
                self.logger.debug(f"Using max_completion_tokens={self.max_tokens} for o* model")
            else:
                api_params["max_tokens"] = self.max_tokens
                self.logger.debug(f"Using max_tokens={self.max_tokens} for regular model")
                
            self.logger.debug(f"API parameters: stream={api_params['stream']}, has_tools={self.tools is not None}")
            
            self.logger.info("Making OpenAI API call - this is the critical point")
            response = self.client.chat.completions.create(**api_params)
            self.logger.info("OpenAI API call completed successfully")

            if not self.running:  # Check before processing response
                self.logger.debug("Thread was stopped, returning early")
                return

            self.logger.debug("Processing API response")
            if self.tools:
                self.logger.debug("Processing tool-based response")
                self.logger.debug(f"Response finish reason: {response.choices[0].finish_reason}")
                
                if response.choices[0].finish_reason == 'tool_calls':
                    self.logger.debug("Response contains tool calls")
                    self.update_response.emit({"response":response.choices[0].message.tool_calls})
                    self.logger.debug("Tool calls response emitted")
                    return
                else: # Not a 'tool_calls' response, try parsing tool calls from content.
                    self.logger.debug("Extracting tool calls from content")
                    content = response.choices[0].message.content
                    self.logger.debug(f"Response content length: {len(content) if content else 0}")
                    
                    tcjs = self._extract_json_objects(content)
                    tool_calls = []
                    for tcj in tcjs:
                        # Best effort normalize to expected dict format
                        if 'tool_calls' not in tcj: tcj = {'tool_calls': [tcj]}
                        for it in tcj['tool_calls']:
                            try:
                                tool_calls.append(
                                    ChatCompletionMessageToolCall(
                                        id=f"call_{self._generate_random_string()}", 
                                        function={"name":it["name"], "arguments":json.dumps(it["arguments"])}, 
                                        type="function")
                                )
                            except:
                                pass
                    self.logger.debug(f"Extracted {len(tool_calls)} tool calls")
                    self.update_response.emit({"response":tool_calls})
                    self.logger.debug("Extracted tool calls response emitted")
                    return
            else: # Not self.tools
                self.logger.debug("Processing streaming response")
                response_buffer = ""
                chunk_count = 0
                for chunk in response:
                    chunk_count += 1
                    if chunk_count % 10 == 0:  # Log every 10th chunk to avoid spam
                        self.logger.debug(f"Processing chunk {chunk_count}")
                        
                    if not self.running:  # Stop consuming stream if interrupted
                        self.logger.debug(f"Thread stopped while processing chunk {chunk_count}")
                        return
                    message_chunk = chunk.choices[0].delta.content or ""
                    response_buffer += message_chunk
                    self.update_response.emit({"response":response_buffer})
                
                self.logger.debug(f"Streaming completed. Total chunks processed: {chunk_count}")
                
        except Exception as e:
            error_msg = f"CRITICAL ERROR in StreamingThread.run(): {type(e).__name__}: {e}"
            self.logger.error(error_msg)
            self.logger.exception("Full traceback:")
            
            # Try to emit an error response if possible
            try:
                self.update_response.emit({"response": f"Error: {str(e)}"})
            except Exception as emit_error:
                self.logger.error(f"Failed to emit error response: {emit_error}")
            
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
