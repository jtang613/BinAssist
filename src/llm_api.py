from binaryninja.settings import Settings
from openai import OpenAI
from .threads import StreamingThread
import sqlite3
import httpx
http_client = httpx.Client(verify=False)

SYSTEM_PROMPT = '''
You are a professional software reverse engineer specializing in cybersecurity. You are intimately 
familiar with x86_64, ARM, PPC and MIPS architectures. You are an expert C and C++ developer.
You are an expert Python and Rust developer. You are familiar with common frameworks and libraries 
such as WinSock, OpenSSL, MFC, etc. You are an expert in TCP/IP network programming and packet analysis.
You always respond to queries in a structured format using Markdown styling for headings and lists. 
You format code blocks using back-tick code-fencing.
'''

class LlmApi:
    """
    Handles interactions with an LLM (Large Language Model) for providing automated responses 
    based on queries related to binary analysis. This class manages API client configuration, database 
    interactions for feedback, and spawning threads for asynchronous API requests.
    """

    def __init__(self):
        """
        Initializes the LlmApi instance, setting up settings and preparing the database for feedback storage.
        """
        self.settings = Settings()
        self.threads = []  # Keep a list of active threads
        self.initialize_database()

    def _create_client(self) -> OpenAI:
        """
        Creates and configures an API client based on settings.

        Returns:
            OpenAI: Configured API client instance.
        """
        base_url = self.settings.get_string('binassist.remote_host')
        api_key = self.settings.get_string('binassist.api_key')
        return OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

    def initialize_database(self) -> None:
        """
        Initializes the SQLite database for storing feedback, ensuring the feedback table exists.
        """
        conn = sqlite3.connect(self.settings.get_string('binassist.rlhf_db'))
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                model_name TEXT NOT NULL,
                prompt_context TEXT NOT NULL,
                system_context TEXT NOT NULL,
                response TEXT NOT NULL,
                feedback INTEGER NOT NULL  -- 1 for thumbs up, 0 for thumbs down
            )
        ''')
        conn.commit()
        conn.close()

    def store_feedback(self, model_name, prompt_context, response, feedback) -> None:
        """
        Stores feedback about the LLM response in the database.

        Parameters:
            model_name (str): The name of the model used for generating responses.
            prompt_context (str): The input prompt context to the model.
            response (str): The model-generated response.
            feedback (int): User feedback (1 for positive, 0 for negative).
        """
        conn = sqlite3.connect(self.settings.get_string('binassist.rlhf_db'))
        c = conn.cursor()
        c.execute('''
            INSERT INTO feedback (model_name, prompt_context, system_context, response, feedback)
            VALUES (?, ?, ?, ?, ?)
        ''', (model_name, prompt_context, SYSTEM_PROMPT, response, feedback))
        conn.commit()
        conn.close()


    def explain(self, bv, addr, bin_type, il_type, addr_to_text_func, signal) -> str:
        """
        Generates a description of binary code at a specific address and il_type using the LLM.

        Parameters:
            bv (BinaryView): The binary view containing the address.
            addr (int): The address within the binary to describe.
            bin_type (str): The type of binary view.
            il_type (str): The intermediate language type used.
            addr_to_text_func (callable): Function converting addresses to text.
            signal (Signal): Qt signal to handle the response asynchronously.

        Returns:
            str: The query sent to the LLM.
        """
        client = self._create_client()
        query = f"Describe the functionality of the decompiled {bv.platform.name} {bin_type} code " +\
                f"below (represented as {il_type}). Provide a summary paragraph section followed by " +\
                f"an analysis section that lists the functionality of each line of code. The analysis " +\
                f"section should be a Markdown formatted list. Try to identify the function name from " +\
                f"the functionality present, or from string constants or log messages if they are " +\
                f"present. But only fallback to strings or log messages that are clearly function " +\
                f"names for this function.\n```\n" +\
                f"{addr_to_text_func(bv, addr)}\n```"
        self._start_thread(client, query, SYSTEM_PROMPT, lambda x: addr_to_text_func(bv, x), signal)
        return query

    def query(self, query, signal) -> str:
        """
        Sends a custom query to the LLM.

        Parameters:
            query (str): The query text.
            signal (Signal): Qt signal to handle the response asynchronously.

        Returns:
            str: The query sent to the LLM.
        """
        client = self._create_client()
        self._start_thread(client, query, SYSTEM_PROMPT, None, signal)
        return query

    def _start_thread(self, client, query, system, addr_to_text_func, signal) -> None:
        """
        Starts a new thread to handle streaming responses from the LLM.

        Parameters:
            client (OpenAI): The API client.
            query (str): The query text.
            system (str): System context for the query.
            addr_to_text_func (callable): Function to convert addresses to text.
            signal (Signal): Qt signal to update with the response.
        """
        thread = StreamingThread(client, query, system, addr_to_text_func)
        thread.update_response.connect(signal)
        self.threads.append(thread)  # Keep track of the thread
        thread.start()

    def HLILToText(self, bv, addr) -> str:
        """
        Converts High Level Intermediate Language (HLIL) instructions at a specific address to text.

        Parameters:
            bv (BinaryView): The binary view containing the address.
            addr (int): The address to convert.

        Returns:
            str: Text representation of HLIL.
        """
        function = bv.get_functions_containing(addr)[0]
        tokens = function.get_type_tokens()[0].tokens
        hlil_instructions = '\n'.join(f'  {instr};' for instr in function.high_level_il.instructions)
        return f"{''.join(x.text for x in tokens)}\n{{\n{hlil_instructions}\n}}\n"

    def MLILToText(self, bv, addr) -> str:
        """
        Converts Medium Level Intermediate Language (MLIL) instructions at a specific address to text.

        Parameters:
            bv (BinaryView): The binary view containing the address.
            addr (int): The address to convert.

        Returns:
            str: Text representation of MLIL.
        """
        function = bv.get_functions_containing(addr)[0]
        tokens = function.get_type_tokens()[0].tokens
        mlil_instructions = '\n'.join(f'0x{instr.address:08x}  {instr}' for instr in function.medium_level_il.instructions)
        return f"{''.join(x.text for x in tokens)}\n{{\n{mlil_instructions}\n}}\n"

    def LLILToText(self, bv, addr) -> str:
        """
        Converts Low Level Intermediate Language (LLIL) instructions at a specific address to text.

        Parameters:
            bv (BinaryView): The binary view containing the address.
            addr (int): The address to convert.

        Returns:
            str: Text representation of LLIL.
        """
        function = bv.get_functions_containing(addr)[0]
        tokens = function.get_type_tokens()[0].tokens
        llil_instructions = '\n'.join(f'0x{instr.address:08x}  {instr}' for instr in function.low_level_il.instructions)
        return f"{''.join(x.text for x in tokens)}\n{llil_instructions}\n"

    def AsmToText(self, bv, addr) -> str:
        """
        Converts assembly instructions at a specific address to text.

        Parameters:
            bv (BinaryView): The binary view containing the address.
            addr (int): The address to convert.

        Returns:
            str: Text representation of assembly instructions.
        """
        asm_instructions = ""
        function = bv.get_functions_containing(addr)[0]
        for bb in function.basic_blocks:
            for dt in bb.disassembly_text:
                s = str(dt)
                asm_instructions += f"0x{dt.address}  {s}"
        tokens = function.get_type_tokens()[0].tokens
        return f"{''.join(x.text for x in tokens)}\n{asm_instructions}\n"


    def stop_threads(self):
        """
        Stops all active threads used for handling LLM queries.
        """
        for thread in self.threads:
            thread.quit()
            thread.wait()
