from binaryninja import *
from binaryninja.settings import Settings
from binaryninjaui import SidebarWidget, SidebarWidgetType, UIActionHandler, SidebarWidgetLocation, \
    SidebarContextSensitivity, ViewFrame, ViewType, UIContext
from binaryninja import FunctionGraphType, PythonScriptingProvider, PythonScriptingInstance
from PySide6 import QtCore, QtGui, QtWidgets
import markdown
import logging

# Enable debug logging for BinAssist
try:
    from .core.debug_logger import setup_debug_logging
    debug_logger = setup_debug_logging(level=logging.DEBUG, log_to_file=True)
    debug_logger.info("BinAssist plugin loading with debug logging enabled")
except Exception as e:
    print(f"Failed to setup debug logging: {e}")

from .llm_api import LlmApi
from .llm_api import ToolCalling

class BinAssistWidget(SidebarWidget):
    """
    A custom widget for Binary Ninja that provides functionalities for querying and displaying
    responses from a language model, along with code analysis tools.
    """

    def __init__(self, name, frame, data) -> None:
        """
        Initializes the BinAssistWidget with the required components and settings.

        Parameters:
            name (str): The name of the widget.
            frame (ViewFrame): The frame context in which the widget is used.
            data: Additional data or configurations required for the widget initialization.
        """
        try:
            super().__init__(name)
            
            # Setup logging for this widget
            self.logger = logging.getLogger(f"binassist.widget.{name}")
            self.logger.info(f"Initializing BinAssistWidget: {name}")
            
            self.settings = Settings()
            self.logger.debug("Settings initialized")
            
            self.LlmApi = LlmApi()
            self.logger.debug("LlmApi initialized")
            
            self.offset_addr = 0
            self.actionHandler = UIActionHandler()
            self.actionHandler.setupActionHandler(self)
            self.logger.debug("Action handler setup completed")
            
            self.bv = None
            self.datatype = None
            self.il_type = None
            self.request = None
            self.response = None
            self.session_log = []

            self.offset = QtWidgets.QLabel(hex(0))

            self._init_ui()
            self.logger.info("BinAssistWidget initialization completed successfully")
            
        except Exception as e:
            error_msg = f"Critical error during BinAssistWidget initialization: {type(e).__name__}: {e}"
            print(error_msg)  # Ensure it gets to console even if logging fails
            try:
                self.logger.error(error_msg)
                self.logger.exception("Full traceback:")
            except:
                pass
            raise

    def _init_ui(self) -> None:
        """
        Sets up the initial UI components and layouts for the widget.
        """
        offset_layout = QtWidgets.QHBoxLayout()
        offset_layout.addWidget(QtWidgets.QLabel("Offset: "))
        offset_layout.addWidget(self.offset)
        offset_layout.setAlignment(QtCore.Qt.AlignCenter)

        explain_tab = self.createExplainTab()
        query_tab = self.createQueryTab()
        actions_tab = self.createActionsTab()
        settings_tab = self.createSettingsTab()
        rag_management_tab = self.createRAGManagementTab()

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(explain_tab, "Explain")
        self.tabs.addTab(query_tab, "Custom Query")
        self.tabs.addTab(actions_tab, "Actions")
        self.tabs.addTab(settings_tab, "Settings")
        self.tabs.addTab(rag_management_tab, "RAG Management")

        self.submit_button = None
        self.submit_label = None

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def createExplainTab(self) -> QtWidgets.QWidget:
        """
        Creates the tab and layout for the explanation functionalities.

        Returns:
            QWidget: A widget configured with explanation functionalities.
        """
        self.text_box = QtWidgets.QTextBrowser()
        self.text_box.setReadOnly(True)
        self.text_box.setOpenLinks(False)
        self.text_box.anchorClicked.connect(self.onAnchorClicked)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self._create_offset_layout())
        layout.addWidget(self.text_box)
        layout.addLayout(self._create_explain_buttons_layout())
        explain_widget = QtWidgets.QWidget()
        explain_widget.setLayout(layout)
        return explain_widget

    def createQueryTab(self) -> QtWidgets.QWidget:
        """
        Creates the tab and layout for custom queries.

        Returns:
            QWidget: A widget configured with query functionalities.
        """
        layout = QtWidgets.QVBoxLayout()
        
        self.query_edit = QtWidgets.QTextEdit()
        self.query_edit.setPlaceholderText("Enter your query here...")
        self.query_response_browser = QtWidgets.QTextBrowser()
        self.query_response_browser.setReadOnly(True)
        self.query_response_browser.setOpenLinks(False)
        self.query_response_browser.anchorClicked.connect(self.onAnchorClicked)

        # Create and add the 'Use RAG' checkbox
        self.use_rag_checkbox = QtWidgets.QCheckBox("Use RAG")
        self.use_rag_checkbox.setChecked(self.settings.get_bool('binassist.use_rag'))
        self.use_rag_checkbox.stateChanged.connect(self.onUseRAGChanged)
        layout.addWidget(self.use_rag_checkbox)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self.query_response_browser)
        splitter.addWidget(self.query_edit)
        splitter.setSizes([400, 100])
        layout.addWidget(splitter)
        layout.addLayout(self._create_query_buttons_layout())
        
        query_widget = QtWidgets.QWidget()
        query_widget.setLayout(layout)
        return query_widget

    def createActionsTab(self) -> QtWidgets.QWidget:
        layout = QtWidgets.QVBoxLayout()

        # Create the 4-column table view
        self.actions_table = QtWidgets.QTableWidget()
        self.actions_table.setColumnCount(4)
        self.actions_table.setHorizontalHeaderLabels(["Select", "Action", "Description", "Status"])
        layout.addWidget(self.actions_table)

        # Create the filter checkboxes
        filter_label = QtWidgets.QLabel("Filters:")
        layout.addWidget(filter_label)
        
        filter_scroll_area = QtWidgets.QScrollArea()
        filter_scroll_area.setWidgetResizable(True)
        filter_scroll_area.setFixedHeight(120)  # Set to 3 lines height (approximate)
        
        filter_widget = QtWidgets.QWidget()
        filter_layout = QtWidgets.QVBoxLayout(filter_widget)
        
        self.filter_checkboxes = {}
        for fn_dict in ToolCalling.FN_TEMPLATES:
            if fn_dict["type"] == "function":
                fn_name = f"{fn_dict['function']['name'].replace('_',' ')}: {fn_dict['function']['description']}"
                checkbox = QtWidgets.QCheckBox(fn_name)
                checkbox.setChecked(True)  # Set all checkboxes to checked by default
                self.filter_checkboxes[fn_name] = checkbox
                filter_layout.addWidget(checkbox)
        
        filter_scroll_area.setWidget(filter_widget)
        layout.addWidget(filter_scroll_area)
    
        # Create the buttons
        buttons_layout = QtWidgets.QHBoxLayout()
        analyze_button = QtWidgets.QPushButton("Analyze Function")
        analyze_clear_button = QtWidgets.QPushButton("Clear")
        apply_button = QtWidgets.QPushButton("Apply Actions")

        analyze_button.clicked.connect(self.onAnalyzeFunctionClicked)
        analyze_clear_button.clicked.connect(self.onAnalyzeClearClicked)
        apply_button.clicked.connect(self.onApplyActionsClicked)

        buttons_layout.addWidget(analyze_button)
        buttons_layout.addWidget(analyze_clear_button)
        buttons_layout.addWidget(apply_button)

        layout.addLayout(buttons_layout)

        actions_widget = QtWidgets.QWidget()
        actions_widget.setLayout(layout)
        return actions_widget

    def createSettingsTab(self) -> QtWidgets.QWidget:
        """
        Creates the comprehensive Settings tab with all BinAssist configuration options.

        Returns:
            QWidget: A widget configured with all settings.
        """
        # Create scrollable area for all settings
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        settings_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # API Providers Section
        providers_group = QtWidgets.QGroupBox("API Providers")
        providers_layout = QtWidgets.QVBoxLayout()
        
        # Providers management
        providers_mgmt_layout = QtWidgets.QHBoxLayout()
        
        self.providers_list = QtWidgets.QListWidget()
        self.providers_list.setMaximumHeight(120)
        self.providers_list.currentRowChanged.connect(self.onProviderSelected)
        providers_mgmt_layout.addWidget(self.providers_list)
        
        # Provider management buttons
        provider_buttons_layout = QtWidgets.QVBoxLayout()
        
        self.add_provider_btn = QtWidgets.QPushButton("Add")
        self.add_provider_btn.clicked.connect(self.addProvider)
        provider_buttons_layout.addWidget(self.add_provider_btn)
        
        self.remove_provider_btn = QtWidgets.QPushButton("Remove")
        self.remove_provider_btn.clicked.connect(self.removeProvider)
        provider_buttons_layout.addWidget(self.remove_provider_btn)
        
        self.duplicate_provider_btn = QtWidgets.QPushButton("Duplicate")
        self.duplicate_provider_btn.clicked.connect(self.duplicateProvider)
        provider_buttons_layout.addWidget(self.duplicate_provider_btn)
        
        provider_buttons_layout.addStretch()
        providers_mgmt_layout.addLayout(provider_buttons_layout)
        
        providers_layout.addLayout(providers_mgmt_layout)
        
        # Provider configuration form
        self.provider_config_group = QtWidgets.QGroupBox("Provider Configuration")
        self.provider_config_layout = QtWidgets.QFormLayout()
        
        # Initialize provider config widgets
        self.provider_name_edit = QtWidgets.QLineEdit()
        self.provider_type_combo = QtWidgets.QComboBox()
        self.provider_type_combo.addItems(['openai', 'anthropic', 'ollama', 'lm_studio', 'text_generation_webui', 'custom'])
        self.provider_url_edit = QtWidgets.QLineEdit()
        self.provider_key_edit = QtWidgets.QLineEdit()
        self.provider_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.provider_model_edit = QtWidgets.QLineEdit()
        self.provider_tokens_spin = QtWidgets.QSpinBox()
        self.provider_tokens_spin.setRange(1, 128*1024)
        self.provider_tokens_spin.setValue(16384)
        
        self.provider_config_layout.addRow("Name:", self.provider_name_edit)
        self.provider_config_layout.addRow("Type:", self.provider_type_combo)
        self.provider_config_layout.addRow("Base URL:", self.provider_url_edit)
        self.provider_config_layout.addRow("API Key:", self.provider_key_edit)
        self.provider_config_layout.addRow("Model:", self.provider_model_edit)
        self.provider_config_layout.addRow("Max Tokens:", self.provider_tokens_spin)
        
        # Save provider button
        save_provider_layout = QtWidgets.QHBoxLayout()
        save_provider_layout.addStretch()
        self.save_provider_btn = QtWidgets.QPushButton("Save Provider")
        self.save_provider_btn.clicked.connect(self.saveCurrentProvider)
        save_provider_layout.addWidget(self.save_provider_btn)
        
        self.provider_config_group.setLayout(self.provider_config_layout)
        providers_layout.addWidget(self.provider_config_group)
        providers_layout.addLayout(save_provider_layout)
        
        # Active provider selection
        active_provider_layout = QtWidgets.QHBoxLayout()
        active_provider_layout.addWidget(QtWidgets.QLabel("Active Provider:"))
        self.active_provider_combo = QtWidgets.QComboBox()
        self.active_provider_combo.currentTextChanged.connect(self.onActiveProviderChanged)
        active_provider_layout.addWidget(self.active_provider_combo)
        active_provider_layout.addStretch()
        providers_layout.addLayout(active_provider_layout)
        
        providers_group.setLayout(providers_layout)
        layout.addWidget(providers_group)

        # System Context Section
        system_context_group = QtWidgets.QGroupBox("System Context")
        system_context_layout = QtWidgets.QVBoxLayout()
        
        context_label = QtWidgets.QLabel("System instructions for the LLM:")
        system_context_layout.addWidget(context_label)
        
        self.system_context_edit = QtWidgets.QTextEdit()
        self.system_context_edit.setPlaceholderText("Enter system context/instructions for the LLM...")
        self.system_context_edit.setMaximumHeight(120)
        
        # Load default system prompt from LlmApi
        try:
            default_prompt = self.LlmApi.SYSTEM_PROMPT.strip()
            self.system_context_edit.setPlainText(default_prompt)
            self.original_system_context = default_prompt  # Store for revert
        except:
            self.original_system_context = ""
        
        system_context_layout.addWidget(self.system_context_edit)
        
        # Save and Revert buttons
        context_buttons_layout = QtWidgets.QHBoxLayout()
        context_buttons_layout.addStretch()
        
        save_context_button = QtWidgets.QPushButton("Save")
        save_context_button.clicked.connect(self.onSaveSystemContext)
        save_context_button.setMaximumWidth(80)
        
        revert_context_button = QtWidgets.QPushButton("Revert")
        revert_context_button.clicked.connect(self.onRevertSystemContext)
        revert_context_button.setMaximumWidth(80)
        
        context_buttons_layout.addWidget(save_context_button)
        context_buttons_layout.addWidget(revert_context_button)
        
        system_context_layout.addLayout(context_buttons_layout)
        system_context_group.setLayout(system_context_layout)
        layout.addWidget(system_context_group)

        # Analysis Options Section
        analysis_group = QtWidgets.QGroupBox("Analysis Options")
        analysis_layout = QtWidgets.QVBoxLayout()

        # Default IL Level
        il_level_layout = QtWidgets.QHBoxLayout()
        il_level_label = QtWidgets.QLabel("Default IL Level:")
        self.il_level_combo = QtWidgets.QComboBox()
        self.il_level_combo.addItems([
            "Assembly (ASM)",
            "Low Level IL (LLIL)", 
            "Medium Level IL (MLIL)",
            "High Level IL (HLIL)",
            "Pseudo-C"
        ])
        self.il_level_combo.setCurrentIndex(3)  # Default to HLIL
        il_level_layout.addWidget(il_level_label)
        il_level_layout.addWidget(self.il_level_combo)
        il_level_layout.addStretch()
        analysis_layout.addLayout(il_level_layout)

        # Context Extraction Settings
        context_layout = QtWidgets.QHBoxLayout()
        context_label = QtWidgets.QLabel("Context Lines:")
        self.context_spin = QtWidgets.QSpinBox()
        self.context_spin.setRange(0, 100)
        self.context_spin.setValue(10)
        self.context_spin.setToolTip("Number of surrounding lines to include for context")
        context_layout.addWidget(context_label)
        context_layout.addWidget(self.context_spin)
        context_layout.addStretch()
        analysis_layout.addLayout(context_layout)

        # Analysis Mode
        mode_layout = QtWidgets.QHBoxLayout()
        mode_label = QtWidgets.QLabel("Analysis Mode:")
        self.analysis_mode_combo = QtWidgets.QComboBox()
        self.analysis_mode_combo.addItems([
            "Conservative", 
            "Balanced", 
            "Aggressive"
        ])
        self.analysis_mode_combo.setCurrentIndex(1)  # Default to Balanced
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.analysis_mode_combo)
        mode_layout.addStretch()
        analysis_layout.addLayout(mode_layout)

        # Response Verbosity
        verbosity_layout = QtWidgets.QHBoxLayout()
        verbosity_label = QtWidgets.QLabel("Response Verbosity:")
        self.verbosity_combo = QtWidgets.QComboBox()
        self.verbosity_combo.addItems(["Concise", "Detailed", "Comprehensive"])
        self.verbosity_combo.setCurrentIndex(1)  # Default to Detailed
        verbosity_layout.addWidget(verbosity_label)
        verbosity_layout.addWidget(self.verbosity_combo)
        verbosity_layout.addStretch()
        analysis_layout.addLayout(verbosity_layout)

        # Checkboxes
        self.include_comments_check = QtWidgets.QCheckBox("Include existing comments in analysis")
        self.include_comments_check.setChecked(True)
        analysis_layout.addWidget(self.include_comments_check)

        self.include_imports_check = QtWidgets.QCheckBox("Include import/library information")
        self.include_imports_check.setChecked(True)
        analysis_layout.addWidget(self.include_imports_check)

        self.auto_apply_check = QtWidgets.QCheckBox("Auto-apply high-confidence suggestions")
        self.auto_apply_check.setChecked(False)
        analysis_layout.addWidget(self.auto_apply_check)

        self.syntax_highlight_check = QtWidgets.QCheckBox("Enable syntax highlighting in responses")
        self.syntax_highlight_check.setChecked(True)
        analysis_layout.addWidget(self.syntax_highlight_check)

        self.show_addresses_check = QtWidgets.QCheckBox("Show addresses in code snippets")
        self.show_addresses_check.setChecked(True)
        analysis_layout.addWidget(self.show_addresses_check)

        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        # General Settings Section
        general_group = QtWidgets.QGroupBox("General Settings")
        general_layout = QtWidgets.QVBoxLayout()

        # RAG Settings
        self.use_rag_check = QtWidgets.QCheckBox("Enable RAG (Retrieval Augmented Generation)")
        self.use_rag_check.setChecked(self.settings.get_bool('binassist.use_rag'))
        self.use_rag_check.stateChanged.connect(self.onUseRAGChanged)
        general_layout.addWidget(self.use_rag_check)

        # RAG Database Path
        rag_path_layout = QtWidgets.QHBoxLayout()
        rag_path_layout.addWidget(QtWidgets.QLabel("RAG Database Path:"))
        
        self.rag_path_edit = QtWidgets.QLineEdit()
        self.rag_path_edit.setText(self.settings.get_string('binassist.rag_db_path'))
        rag_path_layout.addWidget(self.rag_path_edit)
        
        rag_browse_btn = QtWidgets.QPushButton("Browse")
        rag_browse_btn.clicked.connect(self.browseRagPath)
        rag_browse_btn.setMaximumWidth(80)
        rag_path_layout.addWidget(rag_browse_btn)
        
        general_layout.addLayout(rag_path_layout)

        # RLHF Database Path
        rlhf_path_layout = QtWidgets.QHBoxLayout()
        rlhf_path_layout.addWidget(QtWidgets.QLabel("RLHF Database Path:"))
        
        self.rlhf_path_edit = QtWidgets.QLineEdit()
        self.rlhf_path_edit.setText(self.settings.get_string('binassist.rlhf_db'))
        rlhf_path_layout.addWidget(self.rlhf_path_edit)
        
        rlhf_browse_btn = QtWidgets.QPushButton("Browse")
        rlhf_browse_btn.clicked.connect(self.browseRlhfPath)
        rlhf_browse_btn.setMaximumWidth(80)
        rlhf_path_layout.addWidget(rlhf_browse_btn)
        
        general_layout.addLayout(rlhf_path_layout)

        general_group.setLayout(general_layout)
        layout.addWidget(general_group)

        # Initialize providers data
        self.providers = []
        self.current_provider_index = -1
        self.loadProvidersFromSettings()

        settings_widget.setLayout(layout)
        scroll_area.setWidget(settings_widget)
        
        # Create container widget
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout()
        container_layout.addWidget(scroll_area)
        container.setLayout(container_layout)
        
        return container

    def createRAGManagementTab(self) -> QtWidgets.QWidget:
        """
        Creates the tab for managing the RAG database.

        Returns:
            QWidget: A widget configured with RAG management functionalities.
        """
        layout = QtWidgets.QVBoxLayout()

        # Initialize RAG button
        self.rag_init_button = QtWidgets.QPushButton("Add Documents to RAG")
        self.rag_init_button.clicked.connect(self.onRAGInitClicked)
        layout.addWidget(self.rag_init_button)

        # Document list
        self.rag_document_list = QtWidgets.QListWidget()
        self.rag_document_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.rag_document_list)

        # Delete button
        delete_button = QtWidgets.QPushButton("Delete Selected")
        delete_button.clicked.connect(self.onDeleteRAGDocumentsClicked)
        layout.addWidget(delete_button)

        # Refresh button
        refresh_button = QtWidgets.QPushButton("Refresh List")
        refresh_button.clicked.connect(self.refreshRAGDocumentList)
        layout.addWidget(refresh_button)

        rag_management_widget = QtWidgets.QWidget()
        rag_management_widget.setLayout(layout)
        self.refreshRAGDocumentList()
        return rag_management_widget

    def _create_offset_layout(self) -> QtWidgets.QHBoxLayout:
        """
        Creates the layout displaying the current offset.

        Returns:
            QHBoxLayout: Layout containing the offset label and value.
        """
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Offset: "))
        layout.addWidget(self.offset)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        return layout

    def _create_explain_buttons_layout(self) -> QtWidgets.QHBoxLayout:
        """
        Creates the layout with buttons for explanation functionalities.

        Returns:
            QHBoxLayout: Layout containing buttons for explanation actions.
        """
        layout = QtWidgets.QHBoxLayout()
        
        explain_il_bt = QtWidgets.QPushButton("Explain Function", self)
        explain_line_bt = QtWidgets.QPushButton("Explain Line", self)
        clear_text_bt = QtWidgets.QPushButton("Clear", self)
        
        explain_il_bt.clicked.connect(self.onExplainILClicked)
        explain_line_bt.clicked.connect(self.onExplainLineClicked)
        clear_text_bt.clicked.connect(self.onClearTextClicked)
        
        layout.addWidget(explain_il_bt)
        layout.addWidget(explain_line_bt)
        layout.addWidget(clear_text_bt)
        
        return layout

    def _create_query_buttons_layout(self) -> QtWidgets.QHBoxLayout:
        """
        Creates the layout with buttons for query functionalities.

        Returns:
            QHBoxLayout: Layout containing buttons for query actions.
        """
        layout = QtWidgets.QHBoxLayout()
        submit_button = QtWidgets.QPushButton("Submit", self)
        clear_text_bt = QtWidgets.QPushButton("Clear", self)
        submit_button.clicked.connect(self.onSubmitQueryClicked)
        clear_text_bt.clicked.connect(self.onClearTextClicked)
        layout.addWidget(submit_button)
        layout.addWidget(clear_text_bt)
        return layout

    def get_func_text(self):
        """
        Determines the appropriate function to convert binary view data into text based on the current 
        intermediate language (IL) type set for the widget.

        Returns:
            callable: A function from LlmApi corresponding to the current IL type that converts binary data to text.
        """
        func = None
        if self.il_type.view_type == FunctionGraphType.NormalFunctionGraph:
            func = self.LlmApi.AsmToText
        if self.il_type.view_type == FunctionGraphType.LowLevelILFunctionGraph:
            func = self.LlmApi.LLILToText
        if self.il_type.view_type == FunctionGraphType.MediumLevelILFunctionGraph:
            func = self.LlmApi.MLILToText
        if self.il_type.view_type == FunctionGraphType.HighLevelILFunctionGraph:
            func = self.LlmApi.HLILToText
        if self.il_type.view_type == FunctionGraphType.HighLevelLanguageRepresentationFunctionGraph:
            func = self.LlmApi.PseudoCToText
        return func

    def get_line_text(self, bv, addr) -> str:
        """
        Returns the text of the currently selected line regardless of IL level.

        Returns:
            str: The text of the currently selected line.
        """
        if self.il_type == FunctionGraphType.NormalFunctionGraph:
            line = f"0x{addr:08x}  {bv.get_disassembly(addr)}"
        else:
            for inst in PythonScriptingInstance._registered_instances:
                break
            inst.interpreter.update_locals()
            inst.interpreter.update_magic_variables()
            line = PythonScriptingProvider.magic_variables['current_il_instruction'].get_value(inst)
        return line

    def onUseRAGChanged(self, state):
        self.settings.set_bool('binassist.use_rag', state == QtCore.Qt.Checked)

    def onRAGInitClicked(self):
        file_dialog = QtWidgets.QFileDialog()
        markdown_files, _ = file_dialog.getOpenFileNames(self, "Select Markdown Files", "", "Markdown Files (*.md)")
        if markdown_files:
            self.LlmApi.rag_init(markdown_files)
            QtWidgets.QMessageBox.information(self, "RAG Initialization", "RAG initialization complete.")
            self.refreshRAGDocumentList()

    def onDeleteRAGDocumentsClicked(self):
        """
        Handles the deletion of selected source documents from the RAG database.
        """
        selected_items = self.rag_document_list.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.information(self, "Delete Documents", "No documents selected.")
            return

        reply = QtWidgets.QMessageBox.question(self, 'Delete Documents', 
                                               f'Are you sure you want to delete {len(selected_items)} document(s) and all associated embeddings?',
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, 
                                               QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            documents_to_delete = [item.text() for item in selected_items]
            self.LlmApi.rag.delete_documents(documents_to_delete)
            self.refreshRAGDocumentList()
            QtWidgets.QMessageBox.information(self, "Delete Documents", f"{len(documents_to_delete)} document(s) and their associated embeddings deleted.")

    def refreshRAGDocumentList(self):
        """
        Refreshes the list of source documents in the RAG database, sorted alphabetically.
        """
        self.rag_document_list.clear()
        documents = self.LlmApi.rag.get_document_list()
        documents.sort()  # Sort the list alphabetically
        for doc in documents:
            self.rag_document_list.addItem(doc)

    def onExplainILClicked(self) -> None:
        """
        Handles the event when the 'Explain Function' button is clicked.
        Toggles the button between 'Explain' and 'Stop'.
        """
        self.submit_button = self.sender()
        
        if self.submit_button.text() == "Explain Function":
            self.submit_label = self.submit_button.text()
            # Start explanation
            datatype = self.datatype.split(':')[1]
            il_type = self.il_type.name
            func = self.get_func_text()
            self.text_box.clear()

            # Trigger LLM query and store request
            self.request = self.LlmApi.explain(self.bv, self.offset_addr, datatype, il_type, func, self.display_response)
            
            # Change the button text to "Stop"
            self.submit_button.setText("Stop")
        else:
            # Stop the running query
            self.LlmApi.stop_threads()
            
            # Revert the button back to "Explain"
            self.submit_button.setText(self.submit_label)

    def onExplainLineClicked(self) -> None:
        """
        Handles the event when the 'Explain Line' button is clicked.
        Toggles the button between 'Explain' and 'Stop'.
        """
        self.submit_button = self.sender()
        
        if self.submit_button.text() == "Explain Line":
            self.submit_label = self.submit_button.text()
            # Start explanation
            datatype = self.datatype.split(':')[1]
            il_type = self.il_type.name
            self.text_box.clear()
            
            # Trigger LLM query and store request
            self.request = self.LlmApi.explain(self.bv, self.offset_addr, datatype, il_type, self.get_line_text, self.display_response)
            
            # Change the button text to "Stop"
            self.submit_button.setText("Stop")
        else:
            # Stop the running query
            self.LlmApi.stop_threads()
            
            # Revert the button back to "Explain"
            self.submit_button.setText(self.submit_label)

    def onClearTextClicked(self) -> None:
        """
        Clears all text boxes when the 'Clear' button is clicked.
        """
        self.session_log.clear()  # Clear the session log
        self.text_box.clear()
        self.query_response_browser.clear()

    def onSubmitQueryClicked(self) -> None:
        """
        Submits the custom query or stops a running query based on the button state.
        """
        self.logger.info("onSubmitQueryClicked triggered")
        
        try:
            # Toggle functionality between Submit and Stop
            self.submit_button = self.sender()
            self.logger.debug(f"Submit button text: {self.submit_button.text()}")

            if self.submit_button.text() == "Submit":
                self.submit_label = self.submit_button.text()
                
                # Start a new query
                query = self.query_edit.toPlainText()
                self.logger.debug(f"Original query length: {len(query)}")
                
                query = self._process_custom_query(query)
                self.logger.debug(f"Processed query length: {len(query)}")
                
                self.session_log.append({"user": query, "assistant": "Awaiting response..."})

                # Prepend the session log to the query for context
                full_query = "\n".join([f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in self.session_log]) + f"\nUser: {query}"
                self.logger.debug(f"Full query length: {len(full_query)}")

                # Update system context if modified
                if hasattr(self, 'system_context_edit'):
                    current_context = self.system_context_edit.toPlainText().strip()
                    if current_context and current_context != self.LlmApi.SYSTEM_PROMPT:
                        self.logger.debug("Using custom system context for this query")
                        original_prompt = self.LlmApi.SYSTEM_PROMPT
                        self.LlmApi.SYSTEM_PROMPT = current_context
                        
                        # Store the running request
                        self.logger.debug("Calling LlmApi.query with custom context")
                        self.request = self.LlmApi.query(full_query, self.display_custom_response)
                        
                        # Restore original prompt
                        self.LlmApi.SYSTEM_PROMPT = original_prompt
                        self.logger.debug("LlmApi.query call completed with custom context")
                    else:
                        # Store the running request
                        self.logger.debug("Calling LlmApi.query")
                        self.request = self.LlmApi.query(full_query, self.display_custom_response)
                        self.logger.debug("LlmApi.query call completed")
                else:
                    # Store the running request
                    self.logger.debug("Calling LlmApi.query")
                    self.request = self.LlmApi.query(full_query, self.display_custom_response)
                    self.logger.debug("LlmApi.query call completed")

                # Update button to Stop
                self.submit_button.setText("Stop")
                self.logger.debug("Button text changed to Stop")
            else:
                self.logger.debug("Stopping running query")
                # Stop the running query
                self.LlmApi.stop_threads()
                # Revert button back to Submit
                self.submit_button.setText(self.submit_label)
                self.logger.debug("Query stopped and button reverted")
                
        except Exception as e:
            error_msg = f"Error in onSubmitQueryClicked: {type(e).__name__}: {e}"
            self.logger.error(error_msg)
            self.logger.exception("Full traceback:")
            
            # Show error to user
            try:
                QtWidgets.QMessageBox.critical(self, "Query Error", f"An error occurred: {str(e)}")
            except:
                pass  # Don't let error dialog crash us too
                
            # Reset button state
            try:
                if hasattr(self, 'submit_button') and hasattr(self, 'submit_label'):
                    self.submit_button.setText(self.submit_label)
            except:
                pass


    def onAnalyzeFunctionClicked(self) -> None:
        """
        Event for the 'Analyze Function' button.
        Toggles the button between 'Analyze Function' and 'Stop'.
        """
        self.submit_button = self.sender()

        if self.submit_button.text() == "Analyze Function":
            self.submit_label = self.submit_button.text()
            # Start analysis
            datatype = self.datatype.split(':')[1]
            il_type = self.il_type.name
            func = self.get_func_text()

            for fn_name, checkbox in self.filter_checkboxes.items():
                if checkbox.isChecked():
                    action = fn_name.split(':')[0].replace(' ', '_')
                    
                    # Trigger LLM query and store request
                    self.request = self.LlmApi.analyze_function(
                        action, self.bv, self.offset_addr, datatype, il_type, func, self.display_analyze_response
                    )

            # Change the button text to "Stop"
            self.submit_button.setText("Stop")
        else:
            # Stop the running query
            self.LlmApi.stop_threads()
            
            # Revert the button back to "Analyze Function"
            self.submit_button.setText(self.submit_label)

    def onAnalyzeClearClicked(self) -> None:
        """
        Event for the 'Analyze Clear' button.
        """
        self.actions_table.setRowCount(0)

    def loadProvidersFromSettings(self) -> None:
        """
        Load providers from Binary Ninja settings.
        """
        try:
            self.providers = []
            provider_names = []
            
            for i in range(1, 4):  # provider1, provider2, provider3
                try:
                    name = self.settings.get_string(f'binassist.provider{i}_name')
                    if name:
                        provider_data = {
                            'name': name,
                            'provider_type': self.settings.get_string(f'binassist.provider{i}_type'),
                            'base_url': self.settings.get_string(f'binassist.provider{i}_host'),
                            'api_key': self.settings.get_string(f'binassist.provider{i}_key'),
                            'model': self.settings.get_string(f'binassist.provider{i}_model'),
                            'max_tokens': self.settings.get_integer(f'binassist.provider{i}_max_tokens')
                        }
                        self.providers.append(provider_data)
                        provider_names.append(name)
                except:
                    continue
            
            # Update UI
            self.updateProvidersList()
            self.updateActiveProviderCombo()
            
            # Select first provider if available
            if self.providers:
                self.providers_list.setCurrentRow(0)
                
        except Exception as e:
            self.logger.error(f"Error loading providers from settings: {e}")

    def updateProvidersList(self) -> None:
        """
        Update the providers list widget.
        """
        self.providers_list.clear()
        for provider in self.providers:
            self.providers_list.addItem(f"{provider['name']} ({provider['provider_type']})")

    def updateActiveProviderCombo(self) -> None:
        """
        Update the active provider combo box.
        """
        current_text = self.active_provider_combo.currentText()
        self.active_provider_combo.clear()
        
        provider_names = [p['name'] for p in self.providers]
        self.active_provider_combo.addItems(provider_names)
        
        # Restore selection if possible
        if current_text in provider_names:
            index = provider_names.index(current_text)
            self.active_provider_combo.setCurrentIndex(index)
        elif provider_names:
            # Default to current active provider from settings
            active_provider = self.settings.get_string('binassist.active_provider')
            if active_provider in provider_names:
                index = provider_names.index(active_provider)
                self.active_provider_combo.setCurrentIndex(index)

    def onProviderSelected(self, row: int) -> None:
        """
        Handle provider selection in the list.
        """
        if 0 <= row < len(self.providers):
            self.current_provider_index = row
            provider = self.providers[row]
            
            # Load provider data into form
            self.provider_name_edit.setText(provider['name'])
            
            provider_type = provider['provider_type']
            index = self.provider_type_combo.findText(provider_type)
            if index >= 0:
                self.provider_type_combo.setCurrentIndex(index)
            
            self.provider_url_edit.setText(provider['base_url'])
            self.provider_key_edit.setText(provider['api_key'])
            self.provider_model_edit.setText(provider['model'])
            self.provider_tokens_spin.setValue(provider['max_tokens'])

    def addProvider(self) -> None:
        """
        Add a new provider.
        """
        new_provider = {
            'name': f'New Provider {len(self.providers) + 1}',
            'provider_type': 'openai',
            'base_url': 'https://api.openai.com/v1',
            'api_key': '',
            'model': 'gpt-4o-mini',
            'max_tokens': 16384
        }
        
        self.providers.append(new_provider)
        self.updateProvidersList()
        self.updateActiveProviderCombo()
        self.providers_list.setCurrentRow(len(self.providers) - 1)

    def removeProvider(self) -> None:
        """
        Remove the selected provider.
        """
        row = self.providers_list.currentRow()
        if 0 <= row < len(self.providers):
            # Confirm deletion
            provider_name = self.providers[row]['name']
            reply = QtWidgets.QMessageBox.question(
                self, 'Remove Provider', 
                f'Are you sure you want to remove "{provider_name}"?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, 
                QtWidgets.QMessageBox.No
            )
            
            if reply == QtWidgets.QMessageBox.Yes:
                self.providers.pop(row)
                self.updateProvidersList()
                self.updateActiveProviderCombo()
                
                # Select previous provider if possible
                if self.providers:
                    new_row = min(row, len(self.providers) - 1)
                    self.providers_list.setCurrentRow(new_row)
                else:
                    # Clear form if no providers left
                    self.clearProviderForm()

    def duplicateProvider(self) -> None:
        """
        Duplicate the selected provider.
        """
        row = self.providers_list.currentRow()
        if 0 <= row < len(self.providers):
            original = self.providers[row].copy()
            original['name'] = f"{original['name']} (Copy)"
            self.providers.append(original)
            self.updateProvidersList()
            self.updateActiveProviderCombo()
            self.providers_list.setCurrentRow(len(self.providers) - 1)

    def saveCurrentProvider(self) -> None:
        """
        Save the current provider configuration.
        """
        try:
            # Update current provider if one is selected
            if self.current_provider_index >= 0 and self.current_provider_index < len(self.providers):
                self.providers[self.current_provider_index] = {
                    'name': self.provider_name_edit.text(),
                    'provider_type': self.provider_type_combo.currentText(),
                    'base_url': self.provider_url_edit.text(),
                    'api_key': self.provider_key_edit.text(),
                    'model': self.provider_model_edit.text(),
                    'max_tokens': self.provider_tokens_spin.value()
                }
            
            # Save all providers to settings
            self.saveProvidersToSettings()
            self.updateProvidersList()
            self.updateActiveProviderCombo()
            
            # Refresh API provider
            self.LlmApi.api_provider = self.LlmApi.get_active_provider()
            
            QtWidgets.QMessageBox.information(self, "Success", "Provider saved successfully!")
            
        except Exception as e:
            self.logger.error(f"Error saving provider: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save provider: {str(e)}")

    def saveProvidersToSettings(self) -> None:
        """
        Save providers to Binary Ninja settings.
        """
        # Save up to 3 providers
        for i in range(3):
            if i < len(self.providers):
                provider = self.providers[i]
                self.settings.set_string(f'binassist.provider{i+1}_name', provider['name'])
                self.settings.set_string(f'binassist.provider{i+1}_type', provider['provider_type'])
                self.settings.set_string(f'binassist.provider{i+1}_host', provider['base_url'])
                self.settings.set_string(f'binassist.provider{i+1}_key', provider['api_key'])
                self.settings.set_string(f'binassist.provider{i+1}_model', provider['model'])
                self.settings.set_integer(f'binassist.provider{i+1}_max_tokens', provider['max_tokens'])
            else:
                # Clear unused provider slots
                self.settings.set_string(f'binassist.provider{i+1}_name', '')
        
        # Update active provider enum options in Binary Ninja settings
        if hasattr(self.settings, 'set_enum_values'):
            provider_names = [p['name'] for p in self.providers]
            try:
                self.settings.set_enum_values('binassist.active_provider', provider_names)
            except:
                pass  # Ignore if not supported

    def onActiveProviderChanged(self, provider_name: str) -> None:
        """
        Handle active provider change.
        """
        if provider_name:
            self.settings.set_string('binassist.active_provider', provider_name)
            # Refresh API provider
            self.LlmApi.api_provider = self.LlmApi.get_active_provider()
            self.logger.info(f"Active provider changed to: {provider_name}")

    def clearProviderForm(self) -> None:
        """
        Clear the provider configuration form.
        """
        self.provider_name_edit.clear()
        self.provider_type_combo.setCurrentIndex(0)
        self.provider_url_edit.clear()
        self.provider_key_edit.clear()
        self.provider_model_edit.clear()
        self.provider_tokens_spin.setValue(16384)

    def browseRagPath(self) -> None:
        """
        Browse for RAG database path.
        """
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select RAG Database Directory")
        if path:
            self.rag_path_edit.setText(path)
            self.settings.set_string('binassist.rag_db_path', path)

    def browseRlhfPath(self) -> None:
        """
        Browse for RLHF database path.
        """
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select RLHF Database File", "", "Database Files (*.db)")
        if path:
            self.rlhf_path_edit.setText(path)
            self.settings.set_string('binassist.rlhf_db', path)

    def onSaveSystemContext(self) -> None:
        """
        Save the current system context to the LlmApi.
        """
        try:
            new_context = self.system_context_edit.toPlainText()
            # Update the LlmApi system prompt
            self.LlmApi.SYSTEM_PROMPT = new_context
            self.original_system_context = new_context
            self.logger.info("System context saved successfully")
            
            # Show confirmation
            QtWidgets.QMessageBox.information(self, "Success", "System context saved successfully!")
            
        except Exception as e:
            self.logger.error(f"Error saving system context: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save system context: {str(e)}")

    def onRevertSystemContext(self) -> None:
        """
        Revert the system context to the original/saved version.
        """
        try:
            self.system_context_edit.setPlainText(self.original_system_context)
            self.logger.info("System context reverted")
        except Exception as e:
            self.logger.error(f"Error reverting system context: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to revert system context: {str(e)}")

    def onApplyActionsClicked(self) -> None:
        """
        Applies the selected actions from the actions table.
        """
        for row in range(self.actions_table.rowCount()):
            checkbox_widget = self.actions_table.cellWidget(row, 0)
            checkbox = checkbox_widget.findChild(QtWidgets.QCheckBox)
            if checkbox.isChecked():
                action_item = self.actions_table.item(row, 1)
                description_item = self.actions_table.item(row, 2)
                
                action = action_item.text()
                description = description_item.text()
                
                handler_name = f"handle_{action.replace(' ', '_')}"
                handler = getattr(ToolCalling, handler_name, None)
                
                if handler:
                    handler(self.bv, self.actions_table, self.offset_addr, description, row)
                else:
                    print(f"Unknown action: {action}")
                    self.actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Failed: Unknown action"))

        # Update the analysis database
        self.bv.update_analysis()

        # Resize columns to fit the content
        self.actions_table.resizeColumnsToContents()

    def display_response(self, response) -> None:
        """
        Displays the formatted response from the language model.

        Parameters:
            response (str): The response to be displayed.
        """
        html_resp = markdown.markdown(response["response"], extensions=['fenced_code'])
        html_resp += self._generate_feedback_buttons()
        self.response = response["response"]
        self.text_box.setHtml(html_resp)
        if(not self.LlmApi.isRunning()):
            # Revert the button back to "Explain"
            self.submit_button.setText(self.submit_label)


    def display_custom_response(self, response) -> None:
        """
        Displays the custom formatted response from the language model.

        Parameters:
            response (str): The custom response to be displayed.
        """
        # Update session log with response
        self.session_log[-1]["assistant"] = response["response"]
        # Rebuild and display the full conversation history
        full_conversation = "\n".join([f"---\n### User:\n{entry['user']}\n\n---\n### Assistant:\n{entry['assistant']}" for entry in self.session_log])

        html_resp = markdown.markdown(full_conversation, extensions=['fenced_code'])
        html_resp += self._generate_feedback_buttons()
        self.response = response["response"]
        self.query_response_browser.setHtml(html_resp)
        if(not self.LlmApi.isRunning()):
            # Revert the button back to "Explain"
            self.submit_button.setText(self.submit_label)

    def display_analyze_response(self, response) -> None:
        """
        Displays the custom formatted response from the language model in the actions table.

        Parameters:
            response (str): The JSON response to be displayed.
        """
        actions = {'tool_calls':[]}
        for it in response["response"]:
            actions['tool_calls'].append({'name':it.function.name, 'arguments':json.loads(it.function.arguments)})
            
        # Populate the table with the parsed actions
        for idx, action in enumerate(actions.get("tool_calls", [])):

            # Only populate tool calls we support.
            function_names = [fn_dict["function"]["name"] for fn_dict in ToolCalling.FN_TEMPLATES if fn_dict["type"] == "function"]
            if action["name"] not in function_names: continue

            self.actions_table.insertRow(idx)

            # Create a checkbox and center it in the "Select" column
            select_checkbox = QtWidgets.QCheckBox()
            select_widget = QtWidgets.QWidget()
            select_layout = QtWidgets.QHBoxLayout(select_widget)
            select_layout.addWidget(select_checkbox)
            select_layout.setAlignment(QtCore.Qt.AlignCenter)
            select_layout.setContentsMargins(0, 0, 0, 0)
            self.actions_table.setCellWidget(idx, 0, select_widget)

            # Insert the action description in the "Action" column
            action_item = QtWidgets.QTableWidgetItem(self._format_action(action))
            self.actions_table.setItem(idx, 1, action_item)

            # Insert the action description in the "Description" column
            action_item = QtWidgets.QTableWidgetItem(self._format_description(action))
            self.actions_table.setItem(idx, 2, action_item)

            # Leave the "Status" column empty for now
            status_item = QtWidgets.QTableWidgetItem("")
            self.actions_table.setItem(idx, 3, status_item)

        # Resize columns to fit the content
        self.actions_table.resizeColumnsToContents()
        if(not self.LlmApi.isRunning()):
            # Revert the button back to "Explain"
            self.submit_button.setText(self.submit_label)



    def _format_action(self, action: dict) -> str:
        return f"{action['name'].replace('_',' ')}"

    def _format_description(self, action: dict) -> str:
        if action['name'] == 'rename_function':
            return f"{action['arguments']['new_name']}"
        if action['name'] == 'rename_variable':
            return f"{action['arguments']['var_name']} -> {action['arguments']['new_name']}"
        if action['name'] == 'retype_variable':
            return f"{action['arguments']['var_name']} -> {action['arguments']['new_type']}"
        if action['name'] == 'auto_create_struct':
            return f"{action['arguments']['var_name']}"

    def _process_custom_query(self, query) -> str:
        """
        Processes the custom query by replacing placeholders with specific data such as the current line of 
        disassembly, the current function's name, or the current address, enhancing the query with contextual 
        information from the binary view.

        Parameters:
            query (str): The user's query with placeholders for dynamic data.

        Returns:
            str: The processed query with placeholders replaced by actual binary data.
        """
        func = self.get_func_text()

        line = self.get_line_text(self.bv, self.offset_addr)

        match = re.search(r'#range\(0x([0-9a-fA-F]+), 0x([0-9a-fA-F]+)\)', query)
        if match:
            range_start = int(match.group(1), 16)  # Convert hex to int
            range_end = int(match.group(2), 16)   # Convert hex to int
            range_end_offset = match.end()
            print(f"range_start: {range_start}, range_end: {range_end}, range_end_offset: {range_end_offset}")
            range_text = self.LlmApi.DataToText(self.bv, range_start, range_end)
            query = query[:range_end_offset] + f"\n```\n{range_text}\n```\n" + query[range_end_offset:]

        query = query.replace("#line", f'\n```\n{line}\n```\n')
        query = query.replace('#func', f'\n```\n{func(self.bv, self.offset_addr)}\n```\n')
        query = query.replace("#addr", hex(self.offset_addr) or "")
        return query

    def _generate_feedback_buttons(self) -> str:
        """
        Generates HTML content for feedback buttons.

        Returns:
            str: HTML string containing feedback buttons.
        """
        return """
            <div style="text-align: center; color: grey; font-size: 18px;">
                <a href="thumbs-up" style="color: grey; text-decoration: none;">👍</a>
                <a href="thumbs-down" style="color: grey; text-decoration: none;">👎</a>
            </div>
        """

    def onAnchorClicked(self, link) -> None:
        """
        Handles anchor clicks within the text browsers, specifically for feedback links.

        Parameters:
            link (QUrl): The URL that was clicked.
        """
        if link == QtCore.QUrl("thumbs-up"):
            self.handleThumbsUp()
        elif link == QtCore.QUrl("thumbs-down"):
            self.handleThumbsDown()

    def handleThumbsUp(self) -> None:
        """
        Handles the event when the 'Thumbs Up' feedback is given.
        """
        print("RLHF Upvote")
        self.LlmApi.store_feedback(self.settings.get_string('binassist.model'), self.request,self.response, 1)

    def handleThumbsDown(self) -> None:
        """
        Handles the event when the 'Thumbs Down' feedback is given.
        """
        print("RLHF Downvote")
        self.LlmApi.store_feedback(self.settings.get_string('binassist.model'), self.request, self.response, 0)

    def notifyOffsetChanged(self, offset) -> None:
        """
        Updates the displayed offset when it is changed in the binary view.

        Parameters:
            offset (int): The new offset value.
        """
        self.offset.setText(hex(offset))
        self.offset_addr = offset

    def notifyViewChanged(self, view_frame) -> None:
        """
        Updates the widget's context based on changes in the view frame.

        Parameters:
            view_frame (ViewFrame): The new view frame context.
        """
        if view_frame is None:
            self.il_type = None
            self.datatype = None
            self.bv = None
        else:
            self.il_type = view_frame.getCurrentViewInterface().getILViewType()
            self.datatype = view_frame.getCurrentView()
            self.bv = view_frame.getCurrentBinaryView()
            self.query_edit.setPlaceholderText(
                f"{self.datatype} - {self.il_type.name}\n" + \
                "#line to include the current disassembly line.\n" + \
                "#func to include current function disassembly.\n" + \
                "#addr to include the current hex address.\n" + \
                "#range(start, end) to include the linearview data in a given range."
                )

    def contextMenuEvent(self, event) -> None:
        self.m_contextMenuManager.show(self.m_menu, self.actionHandler)

class BinAssistWidgetType(SidebarWidgetType):
    """
    A SidebarWidgetType for creating instances of BinAssistWidget and managing its properties.
    """

    def __init__(self) -> None:
        """
        Initializes the BinAssistWidgetType with an icon and sets up its basic properties.
        """
        icon = QtGui.QImage(56, 56, QtGui.QImage.Format_RGB32)
        icon.fill(0)
        p = QtGui.QPainter()
        p.begin(icon)
        p.setFont(QtGui.QFont("Open Sans", 36))
        p.setPen(QtGui.QColor(255, 255, 255, 255))
        p.drawText(QtCore.QRectF(0, 0, 56, 56), QtCore.Qt.AlignCenter, "BA")
        p.end()
        super().__init__(icon, "BinAssist")

    def createWidget(self, frame, data) -> BinAssistWidget:
        """
        Factory method to create a new instance of BinAssistWidget.

        Parameters:
            frame (ViewFrame): The frame context for the widget.
            data: Additional data for the widget.

        Returns:
            BinAssistWidget: A new instance of BinAssistWidget.
        """
        return BinAssistWidget("BinAssist", frame, data)

    def defaultLocation(self) -> SidebarWidgetLocation:
        """
        Specifies the default location of the widget within the Binary Ninja UI.

        Returns:
            SidebarWidgetLocation: The default sidebar location.
        """
        return SidebarWidgetLocation.RightContent

    def contextSensitivity(self) -> SidebarContextSensitivity:
        """
        Defines the context sensitivity of the widget, indicating how it responds to context changes.

        Returns:
            SidebarContextSensitivity: The context sensitivity setting.
        """
        return SidebarContextSensitivity.SelfManagedSidebarContext
