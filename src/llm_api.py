from binaryninja import BackgroundTaskThread, BinaryView, log
from binaryninja.function import Function, DisassemblySettings
from binaryninja.enums import DisassemblyOption
from binaryninja.lineardisassembly import LinearViewObject, LinearViewCursor, LinearDisassemblyLine
from binaryninja.highlevelil import HighLevelILOperation, HighLevelILInstruction
from binaryninja.types import StructureType, Type, TypeClass
from PySide6 import QtWidgets
from openai import OpenAI
from .threads import StreamingThread, ProposalThread
from .rag import RAG
from .core.services.tool_service import ToolService
from .core.settings import get_settings_manager
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
    Use the available tools to perform the requested actions. When you identify functions, variables, or types that should be renamed or retyped, use the appropriate tools to make those changes.\n
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
        log.log_info("[BinAssist] Initializing LlmApi")
        
        try:
            self.settings = get_settings_manager()
            log.log_debug("[BinAssist] Settings initialized with SQLite backend")
            
            self.threads = []  # Keep a list of active threads
            self.thread = None
            log.log_debug("[BinAssist] Thread management initialized")
            
            self.initialize_database()
            log.log_debug("[BinAssist] Database initialized")
            
            self.rag = RAG(self.settings.get_string('rag_db_path'))
            log.log_debug("[BinAssist] RAG system initialized")
            
            self.tool_service = ToolService()
            log.log_debug("[BinAssist] Tool service initialized")
            
            self.api_provider = self.get_active_provider()
            if self.api_provider:
                log.log_debug(f"[BinAssist] Active provider: {self.api_provider.get('api___name', 'unknown')}")
            else:
                log.log_warn("[BinAssist] No active provider found, creating default provider")
                self._create_default_provider()
            
            log.log_info("[BinAssist] LlmApi initialization completed successfully")
        except Exception as e:
            log.log_error(f"[BinAssist] Error during LlmApi initialization: {type(e).__name__}: {e}")
            # Exception details already logged above
            raise

    def get_active_provider(self):
        """
        Returns the currently active API provider using simple settings.
        """
        try:
            active_name = self.settings.get_string('active_provider')
            providers = self.settings.get_json('api_providers', [])
            
            # Find the active provider
            for provider in providers:
                if provider.get('name') == active_name:
                    return {
                        'api___name': provider.get('name', ''),
                        'provider_type': provider.get('provider_type', 'openai'),
                        'api__host': provider.get('base_url', 'https://api.openai.com/v1'),
                        'api_key': provider.get('api_key', ''),
                        'api__model': provider.get('model', 'gpt-4o-mini'),
                        'api__max_tokens': provider.get('max_tokens', 16384)
                    }
            
            # If no active provider found, return first available or default
            if providers:
                provider = providers[0]
                return {
                    'api___name': provider.get('name', ''),
                    'provider_type': provider.get('provider_type', 'openai'),
                    'api__host': provider.get('base_url', 'https://api.openai.com/v1'),
                    'api_key': provider.get('api_key', ''),
                    'api__model': provider.get('model', 'gpt-4o-mini'),
                    'api__max_tokens': provider.get('max_tokens', 16384)
                }
            
        except Exception as e:
            log.log_error(f"[BinAssist] Error getting active provider: {e}")
            return None
    
    def _create_default_provider(self):
        """Create a default provider if none exists."""
        try:
            # Create default providers
            default_providers = [
                {
                    'name': 'GPT-4o-Mini',
                    'provider_type': 'openai',
                    'base_url': 'https://api.openai.com/v1',
                    'api_key': '',
                    'model': 'gpt-4o-mini',
                    'max_tokens': 16384,
                    'timeout': 120,
                    'enabled': True
                },
                {
                    'name': 'Claude-3.5-Sonnet',
                    'provider_type': 'anthropic',
                    'base_url': 'https://api.anthropic.com',
                    'api_key': '',
                    'model': 'claude-3-5-sonnet-20241022',
                    'max_tokens': 8192,
                    'timeout': 120,
                    'enabled': True
                },
                {
                    'name': 'Ollama-Local',
                    'provider_type': 'ollama',
                    'base_url': 'http://localhost:11434/v1',
                    'api_key': '',
                    'model': 'llama3.1:8b',
                    'max_tokens': 4096,
                    'timeout': 120,
                    'enabled': True
                }
            ]
            
            # Save default providers
            self.settings.set_json('api_providers', default_providers)
            self.settings.set_string('active_provider', 'GPT-4o-Mini')
            
            # Update api_provider
            self.api_provider = self.get_active_provider()
            log.log_info("[BinAssist] Created default providers")
            
        except Exception as e:
            log.log_error(f"[BinAssist] Failed to create default provider: {e}")
            # Return a minimal fallback
            self.api_provider = {
                'api___name': 'Fallback',
                'provider_type': 'openai',
                'api__host': 'https://api.openai.com/v1',
                'api_key': '',
                'api__model': 'gpt-4o-mini',
                'api__max_tokens': 16384
            }

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
        return self.settings.get_boolean('use_rag')

    def _create_client(self):
        """
        Creates and configures an API provider based on settings.

        Returns:
            APIProvider: Configured API provider instance.
        """
        try:
            from .core.api_provider.factory import provider_registry
            from .core.api_provider.config import APIProviderConfig, ProviderType
            
            self.api_provider = self.get_active_provider()
            
            # Convert to APIProviderConfig
            provider_type = ProviderType(self.api_provider['provider_type'])
            api_config = APIProviderConfig(
                name=self.api_provider['api___name'],
                provider_type=provider_type,
                api_key=self.api_provider['api_key'],
                base_url=self.api_provider['api__host'],
                model=self.api_provider['api__model'],
                max_tokens=self.api_provider.get('api__max_tokens', 1000),
                timeout=self.api_provider.get('timeout', 30)
            )
            
            # Create provider using factory
            provider = provider_registry.create_provider(api_config)
            log.log_debug(f"[BinAssist] Created provider: {type(provider).__name__}")
            return provider
            
        except Exception as e:
            log.log_error(f"[BinAssist] Failed to create provider: {e}")
            # Fallback to OpenAI client for compatibility
            base_url = self.api_provider['api__host']
            api_key = self.api_provider['api_key']
            return OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

    def initialize_database(self) -> None:
        """
        Initializes the SQLite database for storing feedback, ensuring the feedback table exists.
        """
        conn = sqlite3.connect(self.settings.get_string('rlhf_db_path'))
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
        conn = sqlite3.connect(self.settings.get_string('rlhf_db_path'))
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
        provider = self._create_client()
        query = f"Describe the functionality of the decompiled {bv.platform.name} {bin_type} code " +\
                f"below (represented as {il_type}). Provide a summary paragraph section followed by " +\
                f"an analysis section that lists the functionality of each line of code. The analysis " +\
                f"section should be a Markdown formatted list. Try to identify the function name from " +\
                f"the functionality present, or from string constants or log messages if they are " +\
                f"present. But only fallback to strings or log messages that are clearly function " +\
                f"names for this function.\n```\n" +\
                f"{addr_to_text_func(bv, addr)}\n```"
        
        # Create a completion callback for regular queries too
        def completion_callback():
            log.log_info("[BinAssist] regular query completion callback triggered")
            signal({"streaming_complete": True})
            log.log_info("[BinAssist] regular query completion signal emitted")
            
        self.thread = self._start_thread(provider, query, self.SYSTEM_PROMPT, signal, None, completion_callback, None, bv, addr)
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
        log.log_info(f"[BinAssist] Starting query with length: {len(query)}")
        log.log_debug(f"[BinAssist] Query preview: {query[:200]}...")
        
        try:
            log.log_debug("[BinAssist] Creating provider client")
            provider = self._create_client()
            log.log_debug("[BinAssist] Provider created successfully")
            
            if self.use_rag():
                log.log_debug("[BinAssist] Using RAG - getting context")
                context = self._get_rag_context(query)
                augmented_query = f"Context:\n{context}\n\nQuery: {query}"
                log.log_debug(f"[BinAssist] RAG context length: {len(context)}, augmented query length: {len(augmented_query)}")
                
                log.log_debug("[BinAssist] Starting thread with RAG-augmented query")
                
                # Create a completion callback for RAG queries too
                def rag_completion_callback():
                    log.log_info("[BinAssist] RAG query completion callback triggered")
                    signal({"streaming_complete": True})
                    log.log_info("[BinAssist] RAG query completion signal emitted")
                    
                self.thread = self._start_thread(provider, augmented_query, self.SYSTEM_PROMPT, signal, None, rag_completion_callback)
                log.log_debug("[BinAssist] Thread started successfully with RAG")
                return augmented_query
            else:
                log.log_debug("[BinAssist] Starting thread without RAG")
                
                # Create a completion callback for regular queries too
                def regular_completion_callback():
                    log.log_info("[BinAssist] regular query completion callback triggered")
                    signal({"streaming_complete": True})
                    log.log_info("[BinAssist] regular query completion signal emitted")
                    
                self.thread = self._start_thread(provider, query, self.SYSTEM_PROMPT, signal, None, regular_completion_callback)
                log.log_debug("[BinAssist] Thread started successfully without RAG")
                return query
                
        except Exception as e:
            log.log_error(f"[BinAssist] Error in query method: {type(e).__name__}: {e}")
            # Exception details already logged above
            raise

    def query_with_tools(self, query, signal, tools=None, mcp_service=None) -> str:
        """
        Sends a custom query to the LLM with optional tool support.

        Parameters:
            query (str): The query text.
            signal (Signal): Qt signal to handle the response asynchronously.
            tools (list): Optional list of tools in OpenAI format for the LLM to use.
            mcp_service: Shared MCP service instance for tool execution.

        Returns:
            str: The query sent to the LLM.
        """
        log.log_info(f"[BinAssist] Starting query with tools. Query length: {len(query)}, Tools: {len(tools) if tools else 0}")
        log.log_debug(f"[BinAssist] Query preview: {query[:200]}...")
        
        try:
            log.log_debug("[BinAssist] Creating provider client")
            provider = self._create_client()
            log.log_debug("[BinAssist] Provider created successfully")
            
            # Create completion callback
            def tool_completion_callback():
                log.log_info("[BinAssist] Tool-enabled query completion callback triggered")
                signal({"streaming_complete": True})
                log.log_info("[BinAssist] Tool-enabled query completion signal emitted")
            
            if self.use_rag():
                log.log_debug("[BinAssist] Starting thread with RAG and tools")
                
                # Use similar RAG logic but with tools
                augmented_query = self.rag.augment_query(query)
                log.log_debug(f"[BinAssist] RAG augmented query length: {len(augmented_query)}")
                
                self.thread = self._start_thread(provider, augmented_query, self.SYSTEM_PROMPT, signal, tools, tool_completion_callback, mcp_service)
                log.log_debug("[BinAssist] Thread started successfully with RAG and tools")
                return augmented_query
            else:
                log.log_debug("[BinAssist] Starting thread with tools (no RAG)")
                    
                self.thread = self._start_thread(provider, query, self.SYSTEM_PROMPT, signal, tools, tool_completion_callback, mcp_service)
                log.log_debug("[BinAssist] Thread started successfully with tools")
                return query
                
        except Exception as e:
            log.log_error(f"[BinAssist] Error in query_with_tools method: {type(e).__name__}: {e}")
            raise

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
        log.log_info(f"[BinAssist] analyze_function called with action: {action}")
        provider = self._create_client()
        code = addr_to_text_func(bv, addr)
        
        # Get prompts and tools from ToolService
        action_prompts = self.tool_service.get_action_prompts()
        prompt = action_prompts.get(action, "").format(code=code)

        if not prompt:
            raise ValueError(f"Unknown action type: {action}")

        query = f"{prompt}\n{self.FUNCTION_PROMPT}"
        tools = self.tool_service.get_tool_definitions()
        log.log_debug(f"[BinAssist] analyze_function got {len(tools) if tools else 0} tools from tool_service")
        # Create a completion callback that emits the completion signal
        def completion_callback():
            log.log_info("[BinAssist] analyze_function completion callback triggered")
            signal({"streaming_complete": True})
            log.log_info("[BinAssist] analyze_function completion signal emitted")
            
        # For analyze_function, we want tool call PROPOSALS, not execution
        # So we'll create a special thread that captures tool calls without executing them
        self.thread = self._start_thread_for_proposals(provider, query, f"{self.SYSTEM_PROMPT}{self.FUNCTION_PROMPT}", signal, tools, completion_callback, bv, addr)

        return query

    def isRunning(self):
        """Check if any threads are currently running."""
        if not self.threads:
            return False
        
        # Check if any thread in the list is still running
        running_threads = [thread for thread in self.threads if thread.isRunning()]
        
        # Clean up finished threads
        if len(running_threads) != len(self.threads):
            self.threads = running_threads
        
        return len(running_threads) > 0

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

    def _start_thread(self, provider, query, system, signal, tools=None, completion_callback=None, mcp_service=None, bv=None, addr=None) -> None:
        """
        Starts a new thread to handle streaming responses from the LLM.

        Parameters:
            provider (APIProvider): The API provider instance.
            query (str): The query text.
            system (str): System context for the query.
            signal (Signal): Qt signal to update with the response.
            tools (dict): A dictionary of available tools for the LLM to consider.
            mcp_service: Shared MCP service instance for tool execution.
        """
        try:
            log.log_info(f"[BinAssist] Starting new thread for provider: {type(provider).__name__}")
            log.log_debug(f"[BinAssist] Thread parameters - query_length: {len(query)}, system_length: {len(system)}, tools: {tools is not None}")
            log.log_debug(f"[BinAssist] Tools count: {len(tools) if tools else 0}")
            
            log.log_debug("[BinAssist] Creating StreamingThread instance")
            thread = StreamingThread(provider, query, system, tools, completion_callback, mcp_service, self.tool_service, self.settings, bv, addr)
            log.log_debug("[BinAssist] StreamingThread instance created")
            
            log.log_debug("[BinAssist] Connecting signal")
            thread.update_response.connect(signal)
            log.log_debug("[BinAssist] Signal connected")
            
            log.log_debug("[BinAssist] Adding thread to tracking list")
            self.threads.append(thread)  # Keep track of the thread
            log.log_debug(f"[BinAssist] Thread added to list. Total active threads: {len(self.threads)}")
            
            log.log_debug("[BinAssist] Starting thread execution")
            thread.start()
            log.log_info("[BinAssist] Thread started successfully")
            
            return thread
            
        except Exception as e:
            log.log_error(f"[BinAssist] Error in _start_thread: {type(e).__name__}: {e}")
            # Exception details already logged above
            raise
    
    def _start_thread_for_proposals(self, provider, query, system, signal, tools=None, completion_callback=None, bv=None, addr=None) -> None:
        """
        Starts a thread to get tool call proposals without executing them.
        This is used for the Actions tab to populate the actions table.
        """
        try:
            log.log_info(f"[BinAssist] Starting proposals thread for provider: {type(provider).__name__}")
            
            # Create a custom thread that captures tool calls without executing them
            thread = ProposalThread(provider, query, system, tools, completion_callback, signal, self.settings, bv, addr)
            thread.update_response.connect(signal)
            
            self.threads.append(thread)
            thread.start()
            
            log.log_info(f"[BinAssist] Proposals thread started successfully")
            
        except Exception as e:
            log.log_error(f"[BinAssist] Failed to start proposals thread: {e}")
            raise

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
        self.thread = None  # Reset single thread reference

    def continue_paused_execution(self):
        """
        Continue tool execution for any paused threads.
        """
        try:
            paused_threads = [thread for thread in self.threads if getattr(thread, 'conversation_paused', False)]
            
            if not paused_threads:
                log.log_warn("[BinAssist] No paused threads found to continue")
                return False
                
            if len(paused_threads) > 1:
                log.log_warn(f"[BinAssist] Multiple paused threads found ({len(paused_threads)}), continuing the first one")
                
            thread = paused_threads[0]
            log.log_info(f"[BinAssist] Continuing paused thread")
            thread.continue_tool_execution()
            return True
            
        except Exception as e:
            log.log_error(f"[BinAssist] Error continuing paused execution: {e}")
            return False
