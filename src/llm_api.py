from binaryninja import BackgroundTaskThread, BinaryView
from binaryninja.settings import Settings
from binaryninja.function import Function, DisassemblySettings
from binaryninja.enums import DisassemblyOption
from binaryninja.lineardisassembly import LinearViewObject, LinearViewCursor, LinearDisassemblyLine
from binaryninja.highlevelil import HighLevelILOperation, HighLevelILInstruction
from binaryninja.types import StructureType, Type, TypeClass
from PySide6 import QtWidgets
from openai import OpenAI
from .threads import StreamingThread
from .rag import RAG
from .toolcalling import ToolCalling
from typing import List
import sqlite3
import httpx
import json

http_client = httpx.Client(verify=False)

class LlmApi:
    """
    Handles interactions with an LLM (Large Language Model) for providing automated responses 
    based on queries related to binary analysis. This class manages API client configuration, database 
    interactions for feedback, and spawning threads for asynchronous API requests.
    """

    SYSTEM_PROMPT = '''
    You are a professional software reverse engineer specializing in cybersecurity. You are intimately 
    familiar with x86_64, ARM, PPC and MIPS architectures. You are an expert C and C++ developer.
    You are an expert Python and Rust developer. You are familiar with common frameworks and libraries 
    such as WinSock, OpenSSL, MFC, etc. You are an expert in TCP/IP network programming and packet analysis.
    You always respond to queries in a structured format using Markdown styling for headings and lists. 
    You format code blocks using back-tick code-fencing.\n
    '''

    FUNCTION_PROMPT = '''
    USE THE PROVIDED TOOLS WHEN NECESSARY. YOU ALWAYS RESPOND WITH TOOL CALLS WHEN POSSIBLE.\n
    '''

    FORMAT_PROMPT = '''
    The output MUST strictly adhere to the following JSON format, do not include any other text.
    The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make tool_calls an empty list '[]'.
    ```
    {
        "tool_calls": [
        {"name": "rename_function", "arguments": {"new_name": "new_name"}},
        ... (more tool calls as required)
        ]
    }
    ```
    REMEMBER, YOU MUST ALWAYS PRODUCE A JSON LIST OF TOOL_CALLS!\n
    '''

    def __init__(self):
        """
        Initializes the LlmApi instance, setting up settings and preparing the database for feedback storage.
        """
        self.settings = Settings()
        self.threads = []  # Keep a list of active threads
        self.thread = None
        self.initialize_database()
        self.rag = RAG(self.settings.get_string('binassist.rag_db_path'))
        self.api_provider = self.get_active_provider()

    def get_active_provider(self):
        """
        Returns the currently active API provider.
        """
        active_name = self.settings.get_string('binassist.active_provider')
        providers = json.loads(self.settings.get_json('binassist.api_providers'))
        return next((p for p in providers if p['api___name'] == active_name), None)

    def rag_init(self, markdown_files: List[str]) -> None:
        """
        Initialize the RAG context database
        """
        class RAGInitTask(BackgroundTaskThread):
            def __init__(self, rag, markdown_files):
                BackgroundTaskThread.__init__(self, "Initializing RAG Database", can_cancel=True)
                self.rag = rag
                self.markdown_files = markdown_files

            def run(self):
                self.rag.rag_init(self.markdown_files)

        RAGInitTask(self.rag, markdown_files).start()

    def use_rag(self) -> bool:
        """
        Returns:
            The current state of the 'use_rag' setting.
        """
        return self.settings.get_bool('binassist.use_rag')

    def _create_client(self) -> OpenAI:
        """
        Creates and configures an API client based on settings.

        Returns:
            OpenAI: Configured API client instance.
        """
        self.api_provider = self.get_active_provider()
        base_url = self.api_provider['api__host']
        api_key = self.api_provider['api_key']
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
        ''', (model_name, prompt_context, self.SYSTEM_PROMPT, response, feedback))
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
        model = self.api_provider['api__model']
        max_tokens = self.api_provider['api__max_tokens']
        query = f"Describe the functionality of the decompiled {bv.platform.name} {bin_type} code " +\
                f"below (represented as {il_type}). Provide a summary paragraph section followed by " +\
                f"an analysis section that lists the functionality of each line of code. The analysis " +\
                f"section should be a Markdown formatted list. Try to identify the function name from " +\
                f"the functionality present, or from string constants or log messages if they are " +\
                f"present. But only fallback to strings or log messages that are clearly function " +\
                f"names for this function.\n```\n" +\
                f"{addr_to_text_func(bv, addr)}\n```"
        self.thread = self._start_thread(client, model, max_tokens, query, self.SYSTEM_PROMPT, signal)
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
        model = self.api_provider['api__model']
        max_tokens = self.api_provider['api__max_tokens']
        if self.use_rag():
            context = self._get_rag_context(query)
            augmented_query = f"Context:\n{context}\n\nQuery: {query}"
            self.thread = self._start_thread(client, model, max_tokens, augmented_query, self.SYSTEM_PROMPT, signal)
            return augmented_query
        else:
            self.thread = self._start_thread(client, model, max_tokens, query, self.SYSTEM_PROMPT, signal)
            return query

    def analyze_function(self, action: str, bv, addr, bin_type, il_type, addr_to_text_func, signal) -> str:
        """
        Analyzes the function at a specific address and il_type using the LLM to produce a set of
        recommended actions for the specified action type.

        Parameters:
            action (str): The type of action to analyze (e.g., "rename_function", "rename_variable", "retype_variable")
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
        model = self.api_provider['api__model']
        max_tokens = self.api_provider['api__max_tokens']
        code = addr_to_text_func(bv, addr)
        prompt = ToolCalling.ACTION_PROMPTS.get(action, "").format(code=code)

        if not prompt:
            raise ValueError(f"Unknown action type: {action}")

        query = f"{prompt}\n{self.FUNCTION_PROMPT}{self.FORMAT_PROMPT}"
        self.thread = self._start_thread(client, model, max_tokens, query, f"{self.SYSTEM_PROMPT}{self.FUNCTION_PROMPT}{self.FORMAT_PROMPT}", signal, ToolCalling.FN_TEMPLATES)

        return query

    def isRunning(self):
        return self.thread.isRunning()

    def _get_rag_context(self, query: str) -> str:
        """
        Query the RAG database for query context.

        Parameters:
            query (str): The query text.

        Returns:
            str: The query context retrieved from the RAG DB.
        """
        results = self.rag.query(query)
        context = "\n\n".join([f"Source: {result['metadata']['source']}\n{result['text']}" for result in results])
        return context

    def delete_rag_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the RAG database.

        Parameters:
            document_ids (str): The list of documents.
        """
        self.rag.delete_documents(document_ids)

    def get_rag_document_list(self) -> List[str]:
        """
        Retrieve the list of documents from the RAG database.

        Returns:
            List[str]: The list of documents.
        """
        return self.rag.get_document_list()

    def _start_thread(self, client, model, max_tokens, query, system, signal, tools=None) -> None:
        """
        Starts a new thread to handle streaming responses from the LLM.

        Parameters:
            client (OpenAI): The API client.
            query (str): The query text.
            system (str): System context for the query.
            signal (Signal): Qt signal to update with the response.
            tools (dict): A dictionary of available toold for the LLM to consdider.
        """
        thread = StreamingThread(client, model, max_tokens, query, system, tools)
        thread.update_response.connect(signal)
        self.threads.append(thread)  # Keep track of the thread
        thread.start()
        return thread

    def DataToText(self, bv: BinaryView, start_addr: int, end_addr: int) -> str:
        """
        Converts Data range (ie: rdata, data) at a specific address range to text.

        Parameters:
            bv (BinaryView): The binary view containing the address.
            start_addr (int): The start address of the range.
            end_addr (int): The end address.

        Returns:
            str: Text representation of Data.
        """
        lines = []
        settings = DisassemblySettings()
        settings.set_option(DisassemblyOption.ShowAddress, True)
        obj = LinearViewObject.language_representation(bv, settings)
        cursor = LinearViewCursor(obj)
        cursor.seek_to_address(start_addr)
        while cursor.current_object.start <= end_addr:
            lines.extend([str(line) for line in cursor.lines])
            cursor.next()

        data = "\n".join(lines)

        return f"{data}"

    def PseudoCToText(self, bv: BinaryView, addr: int) -> str:
        """
        Converts Pseudo-C instructions at a specific address to text.

        Parameters:
            bv (BinaryView): The binary view containing the address.
            addr (int): The address to convert.

        Returns:
            str: Text representation of Pseudo-C.
        """
        function = bv.get_functions_containing(addr)
        if len(function) > 0:
            function = function[0]
        else:
            return None

        lines = []
        settings = DisassemblySettings()
        settings.set_option(DisassemblyOption.ShowAddress, False)
        obj = LinearViewObject.language_representation(bv, settings)
        cursor_end = LinearViewCursor(obj)
        cursor_end.seek_to_address(function.highest_address)
        body = bv.get_next_linear_disassembly_lines(cursor_end)
        cursor_end.seek_to_address(function.highest_address)
        header = bv.get_previous_linear_disassembly_lines(cursor_end)

        for line in header:
            lines.append(f'{str(line)}\n')

        for line in body:
            lines.append(f'{str(line)}\n')

        c_instructions = ''.join(lines)

        return f"{c_instructions}"

    def HLILToText(self, bv: BinaryView, addr: int) -> str:
        """
        Converts High Level Intermediate Language (HLIL) instructions at a specific address to text.

        Parameters:
            bv (BinaryView): The binary view containing the address.
            addr (int): The address to convert.

        Returns:
            str: Text representation of HLIL.
        """
        function = bv.get_functions_containing(addr)
        if len(function) > 0:
            function = function[0]
        else:
            return None

        tokens = function.get_type_tokens()[0].tokens
        hlil_instructions = '\n'.join(f'  {instr};' for instr in function.high_level_il.instructions)
        return f"{''.join(x.text for x in tokens)}\n{{\n{hlil_instructions}\n}}\n"

    def MLILToText(self, bv: BinaryView, addr: int) -> str:
        """
        Converts Medium Level Intermediate Language (MLIL) instructions at a specific address to text.

        Parameters:
            bv (BinaryView): The binary view containing the address.
            addr (int): The address to convert.

        Returns:
            str: Text representation of MLIL.
        """
        function = bv.get_functions_containing(addr)
        if len(function) > 0:
            function = function[0]
        else:
            return None

        tokens = function.get_type_tokens()[0].tokens
        mlil_instructions = '\n'.join(f'0x{instr.address:08x}  {instr}' for instr in function.medium_level_il.instructions)
        return f"{''.join(x.text for x in tokens)}\n{{\n{mlil_instructions}\n}}\n"

    def LLILToText(self, bv: BinaryView, addr: int) -> str:
        """
        Converts Low Level Intermediate Language (LLIL) instructions at a specific address to text.

        Parameters:
            bv (BinaryView): The binary view containing the address.
            addr (int): The address to convert.

        Returns:
            str: Text representation of LLIL.
        """
        function = bv.get_functions_containing(addr)
        if len(function) > 0:
            function = function[0]
        else:
            return None

        tokens = function.get_type_tokens()[0].tokens
        llil_instructions = '\n'.join(f'0x{instr.address:08x}  {instr}' for instr in function.low_level_il.instructions)
        return f"{''.join(x.text for x in tokens)}\n{llil_instructions}\n"

    def AsmToText(self, bv: BinaryView, addr: int) -> str:
        """
        Converts assembly instructions at a specific address to text.

        Parameters:
            bv (BinaryView): The binary view containing the address.
            addr (int): The address to convert.

        Returns:
            str: Text representation of assembly instructions.
        """
        asm_instructions = ""
        function = bv.get_functions_containing(addr)
        if len(function) > 0:
            function = function[0]
        else:
            return None

        for bb in function.basic_blocks:
            for dt in bb.disassembly_text:
                s = str(dt)
                asm_instructions += f"\n0x{dt.address:08x}  {s}"
        tokens = function.get_type_tokens()[0].tokens
        return f"{''.join(x.text for x in tokens)}\n{asm_instructions}\n"


    def stop_threads(self):
        """
        Stops all active threads used for handling LLM queries.
        """
        for thread in self.threads:
            thread.stop()
        self.threads.clear()  # Clear the list after stopping all threads
