from binaryninja.settings import Settings
from PySide6 import QtCore
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall

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

    def __init__(self, client: OpenAI, query: str, system: str, tools=None) -> None:
        """
        Initializes the thread with the necessary parameters for making a streaming API call.

        Parameters:
            client (OpenAI): The OpenAI client used for making API calls.
            query (str): The user's query to be processed.
            system (str): System-level instructions or context for the API call.
            tools (list): A list of tools that the LLM can call during the response.
        """
        super().__init__()
        self.settings = Settings()
        self.client = client
        self.query = query
        self.system = system
        self.tools = tools or []

    def run(self) -> None:
        """
        Executes the streaming API call in a separate thread, processing responses as they come in and 
        signaling the main thread upon updates or when an error occurs.
        """
        response = self.client.chat.completions.create(
            model=self.settings.get_string('binassist.model'),
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.query}
            ],
            stream=False if self.tools else True,
            max_tokens=self.settings.get_integer('binassist.max_tokens'),
            tools=self.tools,
        )
        if self.tools:
            print(f"finish_reason: {response.choices[0].finish_reason}")
            print(f"{response.choices[0].message.content}")
            if response.choices[0].finish_reason == 'tool_calls':
                self.update_response.emit({"response":response.choices[0].message.tool_calls})
                return
            else: # Not a 'tool_calls' response, try parsing tool calls from content.
                tcjs = self._extract_json_objects(response.choices[0].message.content)
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
                self.update_response.emit({"response":tool_calls})
                return
        else: # Not self.tools
            response_buffer = ""
            for chunk in response:
                message_chunk = chunk.choices[0].delta.content or ""
                response_buffer += message_chunk
                self.update_response.emit({"response":response_buffer})


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
