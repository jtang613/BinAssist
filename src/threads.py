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

    def __init__(self, client: OpenAI, query: str, system: str, addr_to_text_func) -> None:
        """
        Initializes the thread with the necessary parameters for making a streaming API call.

        Parameters:
            client (OpenAI): The OpenAI client used for making API calls.
            query (str): The user's query to be processed.
            system (str): System-level instructions or context for the API call.
            addr_to_text_func (callable): A function that converts addresses to text, used in constructing queries.
        """
        super().__init__()
        self.settings = Settings()
        self.client = client
        self.query = query
        self.addr_to_text_func = addr_to_text_func
        self.system = system

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
                stream=True,
                max_tokens=self.settings.get_integer('binassist.max_tokens'),
            )
            response_buffer = ""
            for chunk in response:
                message_chunk = chunk.choices[0].delta.content or ""
                response_buffer += message_chunk
                self.update_response.emit(response_buffer)
        except Exception as e:
            self.update_response.emit(f"Failed to get response: {e}")
