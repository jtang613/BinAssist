from binaryninja.settings import Settings
from PySide6 import QtCore
from openai import OpenAI


class StreamingThread(QtCore.QThread):
    """
    A thread for managing streaming API calls to an OpenAI model, specifically designed to handle long-running 
    requests that stream data back as they process. This class handles the setup and execution of these requests 
    and signals the main application thread upon updates or completion.
    """

    update_response = QtCore.Signal(str)

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
        try:
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
                    self.update_response.emit(response.choices[0].message.tool_calls)
                    print(f"{response.choices[0].message.tool_calls}")
                    return
                else:
                    self.update_response.emit(response.choices[0].message.content)
                    return
            else:
                response_buffer = ""
                for chunk in response:
                    message_chunk = chunk.choices[0].delta.content or ""
                    response_buffer += message_chunk
                    self.update_response.emit(response_buffer)
        except Exception as e:
            self.update_response.emit(f"Failed to get response: {e}")
