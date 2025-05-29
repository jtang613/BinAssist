from binaryninja import *
from binaryninja import log
from binaryninjaui import SidebarWidget, SidebarWidgetType, UIActionHandler, SidebarWidgetLocation, \
    SidebarContextSensitivity, ViewFrame, ViewType, UIContext
from binaryninja import FunctionGraphType, PythonScriptingProvider, PythonScriptingInstance
from PySide6 import QtCore, QtGui, QtWidgets
import markdown
import json
import re

# BinAssist plugin uses Binary Ninja's native logging interface

# Import new settings system
from .core.settings import get_settings_manager, migrate_from_binary_ninja_settings

from .llm_api import LlmApi
from .core.services.tool_service import ToolService
from .core.analysis_db import AnalysisDB
from .utils.program_utils import get_program_hash, get_function_address_string, get_function_start_address

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
            
            # Use Binary Ninja's native logging
            log.log_info(f"[BinAssist] Initializing BinAssistWidget: {name}")
            
            # Initialize new settings system and perform migration if needed
            try:
                migrate_from_binary_ninja_settings(backup_existing=True)
            except Exception as e:
                log.log_warn(f"[BinAssist] Settings migration failed: {e}")
            
            self.settings = get_settings_manager()
            log.log_debug("[BinAssist] Settings initialized with SQLite backend")
            
            self.LlmApi = LlmApi()
            log.log_debug("[BinAssist] LlmApi initialized")
            
            self.tool_service = ToolService()
            log.log_debug("[BinAssist] ToolService initialized")
            
            # Initialize analysis database
            try:
                self.analysis_db = AnalysisDB()
                log.log_debug("[BinAssist] Analysis database initialized")
            except Exception as e:
                log.log_error(f"[BinAssist] Failed to initialize analysis database: {e}")
                self.analysis_db = None
            
            self.offset_addr = 0
            self.actionHandler = UIActionHandler()
            self.actionHandler.setupActionHandler(self)
            log.log_debug("[BinAssist] Action handler setup completed")
            
            self.bv = None
            self.datatype = None
            self.il_type = None
            self.request = None
            self.response = None
            self.session_log = []

            # Add timeout timer for button state recovery
            self.button_timeout_timer = QtCore.QTimer()
            self.button_timeout_timer.timeout.connect(self._handle_button_timeout)
            self.button_timeout_timer.setSingleShot(True)
            log.log_debug("[BinAssist] Button timeout timer initialized")

            self.offset = QtWidgets.QLabel(hex(0))

            self._init_ui()
            log.log_info("[BinAssist] BinAssistWidget initialization completed successfully")
            
        except Exception as e:
            error_msg = f"Critical error during BinAssistWidget initialization: {type(e).__name__}: {e}"
            print(error_msg)  # Ensure it gets to console even if logging fails
            try:
                log.log_error(f"[BinAssist] {error_msg}")
                log.log_error(f"[BinAssist] Full traceback: {e}")
            except:
                pass
            raise

    def __del__(self) -> None:
        """
        Destructor to properly clean up resources when the widget is destroyed.
        """
        try:
            if hasattr(self, 'analysis_db') and self.analysis_db:
                self.analysis_db.close()
                log.log_debug("[BinAssist] Analysis database connection closed during cleanup")
        except Exception as e:
            log.log_error(f"[BinAssist] Error during widget cleanup: {e}")

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
        self.use_rag_checkbox.setChecked(self.settings.get_boolean('use_rag'))
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
        tool_definitions = self.tool_service.get_tool_definitions()
        for fn_dict in tool_definitions:
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
        
        # Connect auto-save events to all provider fields
        self.provider_name_edit.editingFinished.connect(self.onProviderConfigChanged)
        self.provider_type_combo.currentTextChanged.connect(self.onProviderConfigChanged)
        self.provider_url_edit.editingFinished.connect(self.onProviderConfigChanged)
        self.provider_key_edit.editingFinished.connect(self.onProviderConfigChanged)
        self.provider_model_edit.editingFinished.connect(self.onProviderConfigChanged)
        self.provider_tokens_spin.valueChanged.connect(self.onProviderConfigChanged)
        
        # Provider action buttons
        provider_buttons_layout = QtWidgets.QHBoxLayout()
        provider_buttons_layout.addStretch()
        
        self.test_provider_btn = QtWidgets.QPushButton("Test")
        self.test_provider_btn.clicked.connect(self.testCurrentProvider)
        self.test_provider_btn.setMaximumWidth(80)
        provider_buttons_layout.addWidget(self.test_provider_btn)
        
        self.save_provider_btn = QtWidgets.QPushButton("Save Provider")
        self.save_provider_btn.clicked.connect(self.saveCurrentProvider)
        provider_buttons_layout.addWidget(self.save_provider_btn)
        
        save_provider_layout = provider_buttons_layout
        
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
        self.system_context_edit.setPlaceholderText("Enter custom system context/instructions for the LLM (leave empty to use default)...")
        self.system_context_edit.setMaximumHeight(120)
        
        # Load current system prompt from settings (empty means use default)
        current_prompt = self.settings.get_string('system_prompt', '')
        self.system_context_edit.setPlainText(current_prompt)
        
        # Auto-save when text changes (with debouncing)
        self.system_context_timer = QtCore.QTimer()
        self.system_context_timer.setSingleShot(True)
        self.system_context_timer.timeout.connect(self.onSystemContextChanged)
        self.system_context_edit.textChanged.connect(lambda: self.system_context_timer.start(1000))  # 1 second delay
        
        system_context_layout.addWidget(self.system_context_edit)
        system_context_group.setLayout(system_context_layout)
        layout.addWidget(system_context_group)

        # Analysis Options Section
        analysis_group = QtWidgets.QGroupBox("Analysis Options")
        analysis_layout = QtWidgets.QVBoxLayout()

        # Default IL Level
        il_level_layout = QtWidgets.QHBoxLayout()
        il_level_label = QtWidgets.QLabel("Default IL Level:")
        self.il_level_combo = QtWidgets.QComboBox()
        il_level_options = [
            "Assembly (ASM)",
            "Low Level IL (LLIL)", 
            "Medium Level IL (MLIL)",
            "High Level IL (HLIL)",
            "Pseudo-C"
        ]
        self.il_level_combo.addItems(il_level_options)
        
        # Load from settings
        saved_il_level = self.settings.get_string('default_il_level', 'HLIL')
        il_level_map = {
            'ASM': 0, 'LLIL': 1, 'MLIL': 2, 'HLIL': 3, 'Pseudo-C': 4
        }
        self.il_level_combo.setCurrentIndex(il_level_map.get(saved_il_level, 3))
        self.il_level_combo.currentTextChanged.connect(self.onAnalysisOptionChanged)
        
        il_level_layout.addWidget(il_level_label)
        il_level_layout.addWidget(self.il_level_combo)
        il_level_layout.addStretch()
        analysis_layout.addLayout(il_level_layout)

        # Context Extraction Settings
        context_layout = QtWidgets.QHBoxLayout()
        context_label = QtWidgets.QLabel("Context Lines:")
        self.context_spin = QtWidgets.QSpinBox()
        self.context_spin.setRange(0, 100)
        self.context_spin.setValue(self.settings.get_integer('context_lines', 10))
        self.context_spin.setToolTip("Number of surrounding lines to include for context")
        self.context_spin.valueChanged.connect(self.onAnalysisOptionChanged)
        context_layout.addWidget(context_label)
        context_layout.addWidget(self.context_spin)
        context_layout.addStretch()
        analysis_layout.addLayout(context_layout)

        # Analysis Mode
        mode_layout = QtWidgets.QHBoxLayout()
        mode_label = QtWidgets.QLabel("Analysis Mode:")
        self.analysis_mode_combo = QtWidgets.QComboBox()
        analysis_mode_options = ["Conservative", "Balanced", "Aggressive"]
        self.analysis_mode_combo.addItems(analysis_mode_options)
        
        # Load from settings
        saved_mode = self.settings.get_string('analysis_mode', 'Balanced')
        try:
            self.analysis_mode_combo.setCurrentIndex(analysis_mode_options.index(saved_mode))
        except ValueError:
            self.analysis_mode_combo.setCurrentIndex(1)  # Default to Balanced
        self.analysis_mode_combo.currentTextChanged.connect(self.onAnalysisOptionChanged)
        
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.analysis_mode_combo)
        mode_layout.addStretch()
        analysis_layout.addLayout(mode_layout)

        # Response Verbosity
        verbosity_layout = QtWidgets.QHBoxLayout()
        verbosity_label = QtWidgets.QLabel("Response Verbosity:")
        self.verbosity_combo = QtWidgets.QComboBox()
        verbosity_options = ["Concise", "Detailed", "Comprehensive"]
        self.verbosity_combo.addItems(verbosity_options)
        
        # Load from settings
        saved_verbosity = self.settings.get_string('response_verbosity', 'Detailed')
        try:
            self.verbosity_combo.setCurrentIndex(verbosity_options.index(saved_verbosity))
        except ValueError:
            self.verbosity_combo.setCurrentIndex(1)  # Default to Detailed
        self.verbosity_combo.currentTextChanged.connect(self.onAnalysisOptionChanged)
        
        verbosity_layout.addWidget(verbosity_label)
        verbosity_layout.addWidget(self.verbosity_combo)
        verbosity_layout.addStretch()
        analysis_layout.addLayout(verbosity_layout)

        # Checkboxes
        self.include_comments_check = QtWidgets.QCheckBox("Include existing comments in analysis")
        self.include_comments_check.setChecked(self.settings.get_boolean('include_comments', True))
        self.include_comments_check.stateChanged.connect(self.onAnalysisOptionChanged)
        analysis_layout.addWidget(self.include_comments_check)

        self.include_imports_check = QtWidgets.QCheckBox("Include import/library information")
        self.include_imports_check.setChecked(self.settings.get_boolean('include_imports', True))
        self.include_imports_check.stateChanged.connect(self.onAnalysisOptionChanged)
        analysis_layout.addWidget(self.include_imports_check)

        self.auto_apply_check = QtWidgets.QCheckBox("Auto-apply high-confidence suggestions")
        self.auto_apply_check.setChecked(self.settings.get_boolean('auto_apply_suggestions', False))
        self.auto_apply_check.stateChanged.connect(self.onAnalysisOptionChanged)
        analysis_layout.addWidget(self.auto_apply_check)

        self.syntax_highlight_check = QtWidgets.QCheckBox("Enable syntax highlighting in responses")
        self.syntax_highlight_check.setChecked(self.settings.get_boolean('syntax_highlighting', True))
        self.syntax_highlight_check.stateChanged.connect(self.onAnalysisOptionChanged)
        analysis_layout.addWidget(self.syntax_highlight_check)

        self.show_addresses_check = QtWidgets.QCheckBox("Show addresses in code snippets")
        self.show_addresses_check.setChecked(self.settings.get_boolean('show_addresses', True))
        self.show_addresses_check.stateChanged.connect(self.onAnalysisOptionChanged)
        analysis_layout.addWidget(self.show_addresses_check)

        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        # General Settings Section
        general_group = QtWidgets.QGroupBox("General Settings")
        general_layout = QtWidgets.QVBoxLayout()

        # RAG Settings
        self.use_rag_check = QtWidgets.QCheckBox("Enable RAG (Retrieval Augmented Generation)")
        self.use_rag_check.setChecked(self.settings.get_boolean('use_rag'))
        self.use_rag_check.stateChanged.connect(self.onUseRAGChanged)
        general_layout.addWidget(self.use_rag_check)

        # RAG Database Path
        rag_path_layout = QtWidgets.QHBoxLayout()
        rag_path_layout.addWidget(QtWidgets.QLabel("RAG Database Path:"))
        
        self.rag_path_edit = QtWidgets.QLineEdit()
        self.rag_path_edit.setText(self.settings.get_string('rag_db_path'))
        self.rag_path_edit.editingFinished.connect(self.onRagPathChanged)
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
        self.rlhf_path_edit.setText(self.settings.get_string('rlhf_db_path'))
        self.rlhf_path_edit.editingFinished.connect(self.onRlhfPathChanged)
        rlhf_path_layout.addWidget(self.rlhf_path_edit)
        
        rlhf_browse_btn = QtWidgets.QPushButton("Browse")
        rlhf_browse_btn.clicked.connect(self.browseRlhfPath)
        rlhf_browse_btn.setMaximumWidth(80)
        rlhf_path_layout.addWidget(rlhf_browse_btn)
        
        general_layout.addLayout(rlhf_path_layout)

        # Analysis Database Path
        analysis_path_layout = QtWidgets.QHBoxLayout()
        analysis_path_layout.addWidget(QtWidgets.QLabel("Analysis Cache Database Path:"))
        
        self.analysis_path_edit = QtWidgets.QLineEdit()
        self.analysis_path_edit.setText(self.settings.get_string('analysis_db_path', 'binassist_analysis.db'))
        self.analysis_path_edit.editingFinished.connect(self.onAnalysisPathChanged)
        analysis_path_layout.addWidget(self.analysis_path_edit)
        
        analysis_browse_btn = QtWidgets.QPushButton("Browse")
        analysis_browse_btn.clicked.connect(self.browseAnalysisPath)
        analysis_browse_btn.setMaximumWidth(80)
        analysis_path_layout.addWidget(analysis_browse_btn)
        
        general_layout.addLayout(analysis_path_layout)

        # Analysis Cache Management
        cache_management_layout = QtWidgets.QHBoxLayout()
        
        self.cache_stats_label = QtWidgets.QLabel("Cache: Loading...")
        cache_management_layout.addWidget(self.cache_stats_label)
        
        cache_refresh_btn = QtWidgets.QPushButton("Refresh Stats")
        cache_refresh_btn.clicked.connect(self.refreshCacheStats)
        cache_refresh_btn.setMaximumWidth(100)
        cache_management_layout.addWidget(cache_refresh_btn)
        
        cache_clear_btn = QtWidgets.QPushButton("Clear All Cache")
        cache_clear_btn.clicked.connect(self.clearAllCache)
        cache_clear_btn.setMaximumWidth(120)
        cache_clear_btn.setStyleSheet("QPushButton { background-color: #ff6b6b; color: white; }")
        cache_management_layout.addWidget(cache_clear_btn)
        
        cache_management_layout.addStretch()
        general_layout.addLayout(cache_management_layout)

        general_group.setLayout(general_layout)
        layout.addWidget(general_group)
        
        # Refresh cache stats on initialization
        QtCore.QTimer.singleShot(100, self.refreshCacheStats)

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
        self.settings.set_boolean('use_rag', state == QtCore.Qt.Checked)

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
            
            # Check for cached explanation first
            if self.analysis_db and self.bv:
                program_hash = get_program_hash(self.bv)
                function_start = get_function_start_address(self.bv, self.offset_addr)
                function_address = get_function_address_string(function_start)
                
                if program_hash:
                    log.log_debug(f"[BinAssist] Generated program hash: {program_hash[:16]}...")
                    cached_analysis = self.analysis_db.get_analysis(program_hash, function_address)
                    if cached_analysis:
                        log.log_info(f"[BinAssist] Found cached explanation for {function_address}")
                        self.display_cached_explanation(cached_analysis)
                        return
                    else:
                        log.log_debug(f"[BinAssist] No cached explanation found for {function_address}")
                else:
                    log.log_warn("[BinAssist] Could not generate program hash for cache lookup")
            
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
        Also deletes cached explanation for the current function.
        """
        self.session_log.clear()  # Clear the session log
        self.text_box.clear()
        self.query_response_browser.clear()
        
        # Delete cached explanation for current function
        if self.analysis_db and self.bv:
            try:
                program_hash = get_program_hash(self.bv)
                function_start = get_function_start_address(self.bv, self.offset_addr)
                function_address = get_function_address_string(function_start)
                
                if program_hash and function_address:
                    deleted = self.analysis_db.delete_analysis(program_hash, function_address)
                    if deleted:
                        log.log_info(f"[BinAssist] Deleted cached explanation for {function_address}")
                    else:
                        log.log_debug(f"[BinAssist] No cached explanation to delete for {function_address}")
                else:
                    log.log_warn("[BinAssist] Could not delete cached explanation: missing program hash or function address")
                    
            except Exception as e:
                log.log_error(f"[BinAssist] Error deleting cached explanation: {e}")

    def onSubmitQueryClicked(self) -> None:
        """
        Submits the custom query or stops a running query based on the button state.
        """
        log.log_info("[BinAssist] onSubmitQueryClicked triggered")
        
        try:
            # Toggle functionality between Submit and Stop
            self.submit_button = self.sender()
            log.log_debug(f"[BinAssist] Submit button text: {self.submit_button.text()}")

            if self.submit_button.text() == "Submit":
                self.submit_label = self.submit_button.text()
                
                # Start a new query
                query = self.query_edit.toPlainText()
                log.log_debug(f"[BinAssist] Original query length: {len(query)}")
                
                query = self._process_custom_query(query)
                log.log_debug(f"[BinAssist] Processed query length: {len(query)}")
                
                self.session_log.append({"user": query, "assistant": "Awaiting response..."})

                # Prepend the session log to the query for context
                full_query = "\n".join([f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in self.session_log]) + f"\nUser: {query}"
                log.log_debug(f"[BinAssist] Full query length: {len(full_query)}")

                # Update system context if modified
                if hasattr(self, 'system_context_edit'):
                    current_context = self.system_context_edit.toPlainText().strip()
                    if current_context and current_context != self.LlmApi.SYSTEM_PROMPT:
                        log.log_debug("[BinAssist] Using custom system context for this query")
                        original_prompt = self.LlmApi.SYSTEM_PROMPT
                        self.LlmApi.SYSTEM_PROMPT = current_context
                        
                        # Store the running request
                        log.log_debug("[BinAssist] Calling LlmApi.query with custom context")
                        self.request = self.LlmApi.query(full_query, self.display_custom_response)
                        
                        # Restore original prompt
                        self.LlmApi.SYSTEM_PROMPT = original_prompt
                        log.log_debug("[BinAssist] LlmApi.query call completed with custom context")
                    else:
                        # Store the running request
                        log.log_debug("[BinAssist] Calling LlmApi.query (else branch)")
                        self.request = self.LlmApi.query(full_query, self.display_custom_response)
                        log.log_debug("[BinAssist] LlmApi.query call completed (else branch)")
                else:
                    # Store the running request
                    log.log_debug("[BinAssist] Calling LlmApi.query (main else)")
                    self.request = self.LlmApi.query(full_query, self.display_custom_response)
                    log.log_debug("[BinAssist] LlmApi.query call completed (main else)")

                # Update button to Stop
                self.submit_button.setText("Stop")
                log.log_debug("[BinAssist] Button text changed to Stop")
                
                # Start timeout timer (5 minutes = 300000 ms)
                self.button_timeout_timer.start(300000)  # 5 minute timeout
                log.log_debug("[BinAssist] Button timeout timer started (5 minutes)")
            else:
                log.log_debug("[BinAssist] Stopping running query")
                # Stop the running query
                self.LlmApi.stop_threads()
                # Stop the timeout timer
                self.button_timeout_timer.stop()
                # Revert button back to Submit
                self.submit_button.setText(self.submit_label)
                log.log_debug("[BinAssist] Query stopped and button reverted")
                
        except Exception as e:
            error_msg = f"Error in onSubmitQueryClicked: {type(e).__name__}: {e}"
            log.log_error(f"[BinAssist] {error_msg}")
            log.log_error(f"[BinAssist] Full traceback: {e}")
            
            # Show error to user
            try:
                QtWidgets.QMessageBox.critical(self, "Query Error", f"An error occurred: {str(e)}")
            except:
                pass  # Don't let error dialog crash us too
                
            # Reset button state and stop timer
            try:
                if hasattr(self, 'button_timeout_timer'):
                    self.button_timeout_timer.stop()
                if hasattr(self, 'submit_button') and hasattr(self, 'submit_label'):
                    self.submit_button.setText(self.submit_label)
            except:
                pass


    def onAnalyzeFunctionClicked(self) -> None:
        """
        Event for the 'Analyze Function' button.
        Toggles the button between 'Analyze Function' and 'Stop'.
        """
        log.log_info("[BinAssist] onAnalyzeFunctionClicked triggered")
        
        try:
            self.submit_button = self.sender()
            log.log_debug(f"[BinAssist] Analyze button text: {self.submit_button.text()}")

            if self.submit_button.text() == "Analyze Function":
                self.submit_label = self.submit_button.text()
                log.log_debug("[BinAssist] Starting function analysis")
                
                # Start analysis
                datatype = self.datatype.split(':')[1]
                il_type = self.il_type.name
                func = self.get_func_text()

                for fn_name, checkbox in self.filter_checkboxes.items():
                    if checkbox.isChecked():
                        action = fn_name.split(':')[0].replace(' ', '_')
                        
                        log.log_debug(f"[BinAssist] Calling LlmApi.analyze_function for action: {action}")
                        # Store action for caching later
                        self.current_analysis_action = action
                        # Trigger LLM query and store request
                        self.request = self.LlmApi.analyze_function(
                            action, self.bv, self.offset_addr, datatype, il_type, func, self.display_analyze_response
                        )
                        log.log_debug("[BinAssist] LlmApi.analyze_function call completed")

                # Change the button text to "Stop"
                self.submit_button.setText("Stop")
                log.log_debug("[BinAssist] Analyze button text changed to Stop")
                
                # Start timeout timer (5 minutes = 300000 ms)
                self.button_timeout_timer.start(300000)  # 5 minute timeout
                log.log_debug("[BinAssist] Analyze button timeout timer started (5 minutes)")
                
            else:
                log.log_debug("[BinAssist] Stopping running analysis")
                # Stop the running query
                self.LlmApi.stop_threads()
                # Stop the timeout timer
                self.button_timeout_timer.stop()
                # Revert the button back to "Analyze Function"
                self.submit_button.setText(self.submit_label)
                log.log_debug("[BinAssist] Analysis stopped and button reverted")
                
        except Exception as e:
            error_msg = f"Error in onAnalyzeFunctionClicked: {type(e).__name__}: {e}"
            log.log_error(f"[BinAssist] {error_msg}")
            
            # Show error to user
            try:
                QtWidgets.QMessageBox.critical(self, "Analysis Error", f"An error occurred: {str(e)}")
            except:
                pass  # Don't let error dialog crash us too
                
            # Reset button state and stop timer
            try:
                if hasattr(self, 'button_timeout_timer'):
                    self.button_timeout_timer.stop()
                if hasattr(self, 'submit_button') and hasattr(self, 'submit_label'):
                    self.submit_button.setText(self.submit_label)
            except:
                pass

    def onAnalyzeClearClicked(self) -> None:
        """
        Event for the 'Analyze Clear' button.
        Clears the actions table and deletes cached analysis for the current function.
        """
        # Clear the actions table
        self.actions_table.setRowCount(0)
        
        # Delete cached analysis for current function
        if self.analysis_db and self.bv:
            try:
                program_hash = get_program_hash(self.bv)
                function_start = get_function_start_address(self.bv, self.offset_addr)
                function_address = get_function_address_string(function_start)
                
                if program_hash and function_address:
                    deleted = self.analysis_db.delete_analysis(program_hash, function_address)
                    if deleted:
                        log.log_info(f"[BinAssist] Deleted cached analysis for {function_address}")
                    else:
                        log.log_debug(f"[BinAssist] No cached analysis to delete for {function_address}")
                else:
                    log.log_warn("[BinAssist] Could not delete cached analysis: missing program hash or function address")
                    
            except Exception as e:
                log.log_error(f"[BinAssist] Error deleting cached analysis: {e}")

    def loadProvidersFromSettings(self) -> None:
        """
        Load providers from SQLite settings.
        """
        try:
            # Load providers from new JSON-based storage
            self.providers = self.settings.get_json('api_providers', [])
            
            # Update UI
            self.updateProvidersList()
            self.updateActiveProviderCombo()
            
            # Select first provider if available
            if self.providers:
                self.providers_list.setCurrentRow(0)
                
        except Exception as e:
            log.log_error(f"[BinAssist] Error loading providers from settings: {e}")

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
        # Temporarily disconnect signal to avoid triggering changes during setup
        self.active_provider_combo.currentTextChanged.disconnect()
        
        current_text = self.active_provider_combo.currentText()
        self.active_provider_combo.clear()
        
        provider_names = [p['name'] for p in self.providers]
        self.active_provider_combo.addItems(provider_names)
        
        # Always try to restore from settings first, then fall back to current selection
        active_provider = self.settings.get_string('active_provider', '')
        
        if active_provider and active_provider in provider_names:
            # Restore from settings
            index = provider_names.index(active_provider)
            self.active_provider_combo.setCurrentIndex(index)
            log.log_debug(f"[BinAssist] Restored active provider from settings: {active_provider}")
        elif current_text and current_text in provider_names:
            # Fall back to current selection
            index = provider_names.index(current_text)
            self.active_provider_combo.setCurrentIndex(index)
            log.log_debug(f"[BinAssist] Restored active provider from current: {current_text}")
        elif provider_names:
            # Default to first provider and save it as active
            self.active_provider_combo.setCurrentIndex(0)
            self.settings.set_string('active_provider', provider_names[0])
            log.log_debug(f"[BinAssist] Defaulted to first provider: {provider_names[0]}")
        
        # Reconnect signal
        self.active_provider_combo.currentTextChanged.connect(self.onActiveProviderChanged)
    
    def _disconnect_provider_auto_save(self) -> None:
        """Temporarily disconnect provider auto-save signals."""
        try:
            self.provider_name_edit.editingFinished.disconnect(self.onProviderConfigChanged)
            self.provider_type_combo.currentTextChanged.disconnect(self.onProviderConfigChanged)
            self.provider_url_edit.editingFinished.disconnect(self.onProviderConfigChanged)
            self.provider_key_edit.editingFinished.disconnect(self.onProviderConfigChanged)
            self.provider_model_edit.editingFinished.disconnect(self.onProviderConfigChanged)
            self.provider_tokens_spin.valueChanged.disconnect(self.onProviderConfigChanged)
        except Exception as e:
            # Signals might already be disconnected
            log.log_debug(f"[BinAssist] Note: Some auto-save signals were already disconnected: {e}")
    
    def _connect_provider_auto_save(self) -> None:
        """Reconnect provider auto-save signals."""
        self.provider_name_edit.editingFinished.connect(self.onProviderConfigChanged)
        self.provider_type_combo.currentTextChanged.connect(self.onProviderConfigChanged)
        self.provider_url_edit.editingFinished.connect(self.onProviderConfigChanged)
        self.provider_key_edit.editingFinished.connect(self.onProviderConfigChanged)
        self.provider_model_edit.editingFinished.connect(self.onProviderConfigChanged)
        self.provider_tokens_spin.valueChanged.connect(self.onProviderConfigChanged)

    def onProviderSelected(self, row: int) -> None:
        """
        Handle provider selection in the list.
        """
        if 0 <= row < len(self.providers):
            self.current_provider_index = row
            provider = self.providers[row]
            
            # Temporarily disconnect auto-save signals to prevent premature saves
            self._disconnect_provider_auto_save()
            
            try:
                log.log_debug(f"[BinAssist] Loading provider {row}: {provider['name']} ({provider['provider_type']}) - {provider['model']}")
                
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
                
            finally:
                # Reconnect auto-save signals
                self._connect_provider_auto_save()

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
            log.log_error(f"[BinAssist] Error saving provider: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save provider: {str(e)}")

    def saveProvidersToSettings(self) -> None:
        """
        Save providers to SQLite settings.
        """
        # Save providers as JSON array
        self.settings.set_json('api_providers', self.providers)

    def onActiveProviderChanged(self, provider_name: str) -> None:
        """
        Handle active provider change.
        """
        if provider_name:
            self.settings.set_string('active_provider', provider_name)
            # Refresh API provider
            self.LlmApi.api_provider = self.LlmApi.get_active_provider()
            log.log_info(f"[BinAssist] Active provider changed to: {provider_name}")

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
            self.onRagPathChanged()  # Trigger auto-save

    def browseRlhfPath(self) -> None:
        """
        Browse for RLHF database path.
        """
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select RLHF Database File", "", "Database Files (*.db)")
        if path:
            self.rlhf_path_edit.setText(path)
            self.onRlhfPathChanged()  # Trigger auto-save

    def onSystemContextChanged(self) -> None:
        """
        Auto-save system context when it changes.
        """
        try:
            new_context = self.system_context_edit.toPlainText().strip()
            
            # Save to settings (empty string means use default)
            self.settings.set_string('system_prompt', new_context)
            
            # Update the LlmApi system prompt
            if new_context:
                self.LlmApi.SYSTEM_PROMPT = new_context
            else:
                # Use default system prompt
                default_prompt = self.settings.get_setting_definition('system_prompt')['default_value']
                self.LlmApi.SYSTEM_PROMPT = default_prompt
            
            log.log_debug(f"[BinAssist] System context auto-saved: {'custom' if new_context else 'default'}")
            
        except Exception as e:
            log.log_error(f"[BinAssist] Error auto-saving system context: {e}")
    
    def onProviderConfigChanged(self) -> None:
        """
        Auto-save provider configuration when any field changes.
        """
        try:
            if hasattr(self, 'current_provider_index') and self.current_provider_index >= 0:
                # Update the current provider in memory
                if self.current_provider_index < len(self.providers):
                    updated_provider = {
                        'name': self.provider_name_edit.text(),
                        'provider_type': self.provider_type_combo.currentText(),
                        'base_url': self.provider_url_edit.text(),
                        'api_key': self.provider_key_edit.text(),
                        'model': self.provider_model_edit.text(),
                        'max_tokens': self.provider_tokens_spin.value(),
                        'timeout': 120,
                        'enabled': True
                    }
                    
                    log.log_debug(f"[BinAssist] Auto-saving provider {self.current_provider_index}: {updated_provider['name']} ({updated_provider['provider_type']}) - {updated_provider['model']}")
                    self.providers[self.current_provider_index] = updated_provider
                    
                    # Auto-save to settings
                    self.saveProvidersToSettings()
                    self.updateProvidersList()
                    self.updateActiveProviderCombo()
                    
                    # Refresh API provider if this is the active one
                    active_provider = self.settings.get_string('active_provider')
                    if self.providers[self.current_provider_index]['name'] == active_provider:
                        self.LlmApi.api_provider = self.LlmApi.get_active_provider()
                    
                    log.log_debug("[BinAssist] Provider configuration auto-saved")
                    
        except Exception as e:
            log.log_error(f"[BinAssist] Error auto-saving provider config: {e}")
    
    def onRagPathChanged(self) -> None:
        """Auto-save RAG database path."""
        try:
            path = self.rag_path_edit.text()
            self.settings.set_string('rag_db_path', path)
            log.log_debug(f"[BinAssist] RAG path auto-saved: {path}")
        except Exception as e:
            log.log_error(f"[BinAssist] Error auto-saving RAG path: {e}")
    
    def onRlhfPathChanged(self) -> None:
        """Auto-save RLHF database path."""
        try:
            path = self.rlhf_path_edit.text()
            self.settings.set_string('rlhf_db_path', path)
            log.log_debug(f"[BinAssist] RLHF path auto-saved: {path}")
        except Exception as e:
            log.log_error(f"[BinAssist] Error auto-saving RLHF path: {e}")

    def onAnalysisPathChanged(self) -> None:
        """Auto-save analysis database path."""
        try:
            path = self.analysis_path_edit.text()
            self.settings.set_string('analysis_db_path', path)
            log.log_debug(f"[BinAssist] Analysis database path auto-saved: {path}")
            
            # Reinitialize analysis database with new path
            if hasattr(self, 'analysis_db') and self.analysis_db:
                self.analysis_db.close()
            
            try:
                self.analysis_db = AnalysisDB()
                log.log_info("[BinAssist] Analysis database reinitialized with new path")
                self.refreshCacheStats()
            except Exception as db_error:
                log.log_error(f"[BinAssist] Failed to reinitialize analysis database: {db_error}")
                self.analysis_db = None
                
        except Exception as e:
            log.log_error(f"[BinAssist] Error auto-saving analysis database path: {e}")

    def browseAnalysisPath(self) -> None:
        """Browse for analysis database path."""
        try:
            current_path = self.analysis_path_edit.text()
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 
                "Select Analysis Database Path", 
                current_path,
                "SQLite Database (*.db);;All Files (*)"
            )
            if file_path:
                self.analysis_path_edit.setText(file_path)
                self.onAnalysisPathChanged()
        except Exception as e:
            log.log_error(f"[BinAssist] Error browsing analysis database path: {e}")

    def refreshCacheStats(self) -> None:
        """Refresh cache statistics display."""
        try:
            if self.analysis_db:
                analysis_count = self.analysis_db.get_analysis_count()
                context_count = self.analysis_db.get_context_count()
                self.cache_stats_label.setText(f"Cache: {analysis_count} analyses, {context_count} contexts")
                log.log_debug(f"[BinAssist] Cache stats refreshed: {analysis_count} analyses, {context_count} contexts")
            else:
                self.cache_stats_label.setText("Cache: Database not available")
        except Exception as e:
            log.log_error(f"[BinAssist] Error refreshing cache stats: {e}")
            self.cache_stats_label.setText("Cache: Error loading stats")

    def clearAllCache(self) -> None:
        """Clear all cached analysis and context data."""
        try:
            reply = QtWidgets.QMessageBox.question(
                self, 
                "Clear All Cache",
                "Are you sure you want to clear all cached analysis results and contexts?\n\nThis action cannot be undone.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            
            if reply == QtWidgets.QMessageBox.Yes:
                if self.analysis_db:
                    analysis_cleared = self.analysis_db.clear_all_analysis()
                    context_cleared = self.analysis_db.clear_all_contexts()
                    
                    if analysis_cleared and context_cleared:
                        log.log_info("[BinAssist] All cache data cleared successfully")
                        QtWidgets.QMessageBox.information(self, "Cache Cleared", "All cached data has been cleared successfully.")
                    else:
                        log.log_warn("[BinAssist] Some cache data may not have been cleared")
                        QtWidgets.QMessageBox.warning(self, "Cache Clear Warning", "Some cached data may not have been cleared. Check logs for details.")
                    
                    self.refreshCacheStats()
                else:
                    QtWidgets.QMessageBox.warning(self, "Cache Clear Error", "Analysis database is not available.")
        except Exception as e:
            log.log_error(f"[BinAssist] Error clearing cache: {e}")
            QtWidgets.QMessageBox.critical(self, "Cache Clear Error", f"An error occurred while clearing cache: {str(e)}")
    
    def onAnalysisOptionChanged(self) -> None:
        """Auto-save analysis options when they change."""
        try:
            # Save IL level
            il_level_text = self.il_level_combo.currentText()
            il_level_map = {
                'Assembly (ASM)': 'ASM',
                'Low Level IL (LLIL)': 'LLIL', 
                'Medium Level IL (MLIL)': 'MLIL',
                'High Level IL (HLIL)': 'HLIL',
                'Pseudo-C': 'Pseudo-C'
            }
            self.settings.set_string('default_il_level', il_level_map.get(il_level_text, 'HLIL'))
            
            # Save context lines
            self.settings.set_integer('context_lines', self.context_spin.value())
            
            # Save analysis mode
            self.settings.set_string('analysis_mode', self.analysis_mode_combo.currentText())
            
            # Save response verbosity
            self.settings.set_string('response_verbosity', self.verbosity_combo.currentText())
            
            # Save checkboxes
            self.settings.set_boolean('include_comments', self.include_comments_check.isChecked())
            self.settings.set_boolean('include_imports', self.include_imports_check.isChecked())
            self.settings.set_boolean('auto_apply_suggestions', self.auto_apply_check.isChecked())
            self.settings.set_boolean('syntax_highlighting', self.syntax_highlight_check.isChecked())
            self.settings.set_boolean('show_addresses', self.show_addresses_check.isChecked())
            
            log.log_debug("[BinAssist] Analysis options auto-saved")
            
        except Exception as e:
            log.log_error(f"[BinAssist] Error auto-saving analysis options: {e}")
    
    def testCurrentProvider(self) -> None:
        """
        Test the current provider configuration with a simple query.
        """
        log.log_info(f"[BinAssist] testCurrentProvider called")
        try:
            # Temporarily save current provider config
            current_config = {
                'name': self.provider_name_edit.text(),
                'provider_type': self.provider_type_combo.currentText(),
                'base_url': self.provider_url_edit.text(),
                'api_key': self.provider_key_edit.text(),
                'model': self.provider_model_edit.text(),
                'max_tokens': self.provider_tokens_spin.value()
            }
            
            log.log_debug(f"[BinAssist] Current provider config: {current_config}")
            
            # Validate required fields
            if not current_config['name'] or not current_config['base_url'] or not current_config['model']:
                QtWidgets.QMessageBox.warning(self, "Test Failed", "Please fill in Name, Base URL, and Model fields.")
                return
            
            if current_config['provider_type'] in ['openai', 'anthropic'] and not current_config['api_key']:
                QtWidgets.QMessageBox.warning(self, "Test Failed", "API Key is required for this provider type.")
                return
            
            # Disable test button and show progress
            self.test_provider_btn.setText("Testing...")
            self.test_provider_btn.setEnabled(False)
            QtWidgets.QApplication.processEvents()
            
            # Create a temporary client for testing
            self._test_provider_connectivity(current_config)
            
        except Exception as e:
            log.log_error(f"[BinAssist] Error testing provider: {e}")
            QtWidgets.QMessageBox.critical(self, "Test Error", f"Test failed with error: {str(e)}")
            
        finally:
            # Re-enable test button
            self.test_provider_btn.setText("Test")
            self.test_provider_btn.setEnabled(True)
    
    def _test_provider_connectivity(self, config: dict) -> None:
        """
        Test provider connectivity using the provider system.
        
        This method delegates to the provider's own test_connectivity method,
        following proper OOP principles instead of duplicating provider logic.
        """
        try:
            log.log_info(f"[BinAssist] _test_provider_connectivity starting for: {config}")
            
            from .core.api_provider.factory import provider_registry
            from .core.api_provider.config import APIProviderConfig, ProviderType
            
            log.log_debug(f"[BinAssist] Imported provider_registry: {provider_registry}")
            log.log_debug(f"[BinAssist] Registry type: {type(provider_registry)}")
            log.log_debug(f"[BinAssist] Registry factories: {getattr(provider_registry, '_factories', 'No _factories attr')}")
            
            # Convert dict config to APIProviderConfig
            log.log_debug(f"[BinAssist] Converting provider_type '{config['provider_type']}' to enum")
            provider_type = ProviderType(config['provider_type'])
            log.log_debug(f"[BinAssist] Provider type enum: {provider_type}")
            
            api_config = APIProviderConfig(
                name=config['name'],
                provider_type=provider_type,
                api_key=config['api_key'],
                base_url=config['base_url'],
                model=config['model'],
                max_tokens=config.get('max_tokens', 1000),
                timeout=config.get('timeout', 30)
            )
            
            # Create provider instance
            log.log_info(f"[BinAssist] Creating provider for testing: {config['name']}")
            log.log_debug(f"[BinAssist] Supported provider types: {provider_registry.get_supported_types()}")
            log.log_debug(f"[BinAssist] Is {provider_type} supported: {provider_registry.is_supported(provider_type)}")
            provider = provider_registry.create_provider(api_config)
            
            # Test connectivity using provider's method
            log.log_info(f"[BinAssist] Testing connectivity for provider: {config['name']}")
            response_text = provider.test_connectivity()
            
            # Cleanup
            provider.close()
            
            # Check response
            if response_text and response_text.strip():
                QtWidgets.QMessageBox.information(
                    self, "Test Successful", 
                    f"✅ Provider test successful!\n\nResponse: {response_text[:100]}..."
                )
                log.log_info(f"[BinAssist] Provider test successful: {config['name']}")
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Test Warning", 
                    "⚠️ Connection successful but received empty response."
                )
                
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "authentication" in error_msg.lower():
                QtWidgets.QMessageBox.critical(
                    self, "Test Failed", 
                    "❌ Authentication failed. Please check your API key."
                )
            elif "404" in error_msg or "not found" in error_msg.lower():
                QtWidgets.QMessageBox.critical(
                    self, "Test Failed", 
                    "❌ Model or endpoint not found. Please check the Base URL and Model name."
                )
            elif "timeout" in error_msg.lower():
                QtWidgets.QMessageBox.critical(
                    self, "Test Failed", 
                    "❌ Connection timed out. Please check the Base URL and network connection."
                )
            else:
                QtWidgets.QMessageBox.critical(
                    self, "Test Failed", 
                    f"❌ Test failed: {error_msg[:200]}..."
                )
            
            log.log_error(f"[BinAssist] Provider test failed: {e}")

    def onApplyActionsClicked(self) -> None:
        """
        Applies the selected actions from the actions table using ToolService.
        """
        log.log_info("[BinAssist] onApplyActionsClicked triggered")
        
        total_rows = self.actions_table.rowCount()
        log.log_debug(f"[BinAssist] Actions table has {total_rows} rows")
        
        if total_rows == 0:
            log.log_warn("[BinAssist] No actions available in table to apply")
            return
            
        checked_count = 0
        applied_count = 0
        
        for row in range(total_rows):
            log.log_debug(f"[BinAssist] Processing row {row}")
            
            checkbox_widget = self.actions_table.cellWidget(row, 0)
            if checkbox_widget is None:
                log.log_warn(f"[BinAssist] Row {row}: No checkbox widget found")
                continue
                
            checkbox = checkbox_widget.findChild(QtWidgets.QCheckBox)
            if checkbox is None:
                log.log_warn(f"[BinAssist] Row {row}: No checkbox found in widget")
                continue
                
            if checkbox.isChecked():
                checked_count += 1
                log.log_info(f"[BinAssist] Row {row}: Checkbox is checked, processing action")
                
                action_item = self.actions_table.item(row, 1)
                description_item = self.actions_table.item(row, 2)
                
                if action_item is None or description_item is None:
                    log.log_error(f"[BinAssist] Row {row}: Missing action or description item")
                    continue
                
                action = action_item.text()
                description = description_item.text()
                
                # Convert action name back to underscore format (UI displays with spaces for readability)
                action_name = action.replace(' ', '_')
                
                log.log_info(f"[BinAssist] Row {row}: Applying action '{action}' (converted to '{action_name}') with description '{description}'")
                
                # Use ToolService for action execution
                try:
                    log.log_debug(f"[BinAssist] Row {row}: Calling tool_service.handle_action_for_ui")
                    self.tool_service.handle_action_for_ui(
                        self.bv, self.actions_table, self.offset_addr, action_name, description, row
                    )
                    applied_count += 1
                    log.log_info(f"[BinAssist] Row {row}: Action applied successfully")
                except Exception as e:
                    log.log_error(f"[BinAssist] Row {row}: Action execution failed: {e}")
                    self.actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"Error: {str(e)}"))
            else:
                log.log_debug(f"[BinAssist] Row {row}: Checkbox not checked, skipping")

        log.log_info(f"[BinAssist] Apply Actions completed: {checked_count} actions checked, {applied_count} actions applied")

        # Update the analysis database
        log.log_debug("[BinAssist] Updating Binary Ninja analysis database")
        self.bv.update_analysis()
        log.log_debug("[BinAssist] Analysis database update completed")

        # Resize columns to fit the content
        self.actions_table.resizeColumnsToContents()
        log.log_debug("[BinAssist] Actions table columns resized")

    def display_response(self, response) -> None:
        """
        Displays the formatted response from the language model.

        Parameters:
            response (str): The response to be displayed.
        """
        try:
            # Check if this is the streaming completion signal
            if isinstance(response, dict) and response.get("streaming_complete"):
                log.log_info("[BinAssist] Received streaming completion signal in display_response - reverting button")
                
                # Cache the final response when streaming completes
                if self.analysis_db and self.bv and hasattr(self, 'response') and self.response:
                    try:
                        program_hash = get_program_hash(self.bv)
                        function_start = get_function_start_address(self.bv, self.offset_addr)
                        function_address = get_function_address_string(function_start)
                        
                        if program_hash and function_address:
                            # Create a query string for the explanation cache
                            query = "explain_function"
                            
                            # Store the final response text
                            success = self.analysis_db.upsert_analysis(program_hash, function_address, query, self.response)
                            if success:
                                log.log_info(f"[BinAssist] Cached explanation result for {function_address}")
                            else:
                                log.log_warn(f"[BinAssist] Failed to cache explanation result for {function_address}")
                        else:
                            log.log_warn("[BinAssist] Could not cache explanation: missing program hash or function address")
                            
                    except Exception as cache_error:
                        log.log_error(f"[BinAssist] Error caching explanation result: {cache_error}")
                
                self._revert_button_state("display_response_completion")
                return
                
            html_resp = markdown.markdown(response["response"], extensions=['fenced_code'])
            html_resp += self._generate_feedback_buttons()
            self.response = response["response"]
            self.text_box.setHtml(html_resp)
        except Exception as e:
            log.log_error(f"[BinAssist] Error displaying response: {e}")
            # Button reversion is now handled by specific completion signals only


    def display_custom_response(self, response) -> None:
        """
        Displays the custom formatted response from the language model.

        Parameters:
            response (str): The custom response to be displayed.
        """
        try:
            # Check if this is the streaming completion signal
            if isinstance(response, dict) and response.get("streaming_complete"):
                log.log_info("[BinAssist] Received streaming completion signal in display_custom_response - reverting button")
                self._revert_button_state("display_custom_response_completion")
                return
                
            # Update session log with response
            self.session_log[-1]["assistant"] = response["response"]
            # Rebuild and display the full conversation history
            full_conversation = "\n".join([f"---\n### User:\n{entry['user']}\n\n---\n### Assistant:\n{entry['assistant']}" for entry in self.session_log])

            html_resp = markdown.markdown(full_conversation, extensions=['fenced_code'])
            html_resp += self._generate_feedback_buttons()
            self.response = response["response"]
            self.query_response_browser.setHtml(html_resp)
        except Exception as e:
            log.log_error(f"[BinAssist] Error displaying custom response: {e}")
            # Button reversion is now handled by specific completion signals only

    def display_analyze_response(self, response) -> None:
        """
        Displays the custom formatted response from the language model in the actions table.

        Parameters:
            response (str): The JSON response to be displayed.
        """
        log.log_info("[BinAssist] display_analyze_response called")
        try:
            # Check if this is the streaming completion signal
            if isinstance(response, dict) and response.get("streaming_complete"):
                log.log_info("[BinAssist] Received streaming completion signal in display_analyze_response - reverting button")
                self._revert_button_state("display_analyze_response_completion")
                return
            
            log.log_debug("[BinAssist] Processing analyze response for actions table")
            actions = {'tool_calls':[]}
            for it in response["response"]:
                actions['tool_calls'].append({'name':it.function.name, 'arguments':json.loads(it.function.arguments)})
                
            # Populate the table with the parsed actions
            for idx, action in enumerate(actions.get("tool_calls", [])):

                # Only populate tool calls we support.
                tool_definitions = self.tool_service.get_tool_definitions()
                function_names = [fn_dict["function"]["name"] for fn_dict in tool_definitions if fn_dict["type"] == "function"]
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
            log.log_debug(f"[BinAssist] Actions table populated with {len(actions.get('tool_calls', []))} actions")
            
            
        except Exception as e:
            log.log_error(f"[BinAssist] Error displaying analyze response: {e}")
            # Only revert button on error, not on normal streaming updates

    def checkAndDisplayCachedAnalysis(self) -> None:
        """
        Check for cached analysis for the current function and display it if found.
        This is called when navigating between functions.
        """
        try:
            if not self.analysis_db or not self.bv:
                return
                
            program_hash = get_program_hash(self.bv)
            function_start = get_function_start_address(self.bv, self.offset_addr)
            function_address = get_function_address_string(function_start)
            
            if not program_hash or not function_address:
                log.log_debug("[BinAssist] Cannot check cache: missing program hash or function address")
                return
                
            log.log_debug(f"[BinAssist] Checking cache for function at {function_address}")
            
            # Look for any cached analysis for this function (regardless of action type)
            cached_analysis = self.analysis_db.get_analysis(program_hash, function_address)
            
            if cached_analysis:
                log.log_info(f"[BinAssist] Found cached analysis for function at {function_address}")
                
                # Check what type of cached analysis we have
                if cached_analysis.query == "explain_function":
                    # Auto-display cached explanation in the Explain tab
                    self.display_cached_explanation(cached_analysis)
                    log.log_debug("[BinAssist] Auto-displayed cached explanation during navigation")
                else:
                    # For function analysis, just clear the actions table without auto-displaying
                    self.actions_table.setRowCount(0)
                
            else:
                log.log_debug(f"[BinAssist] No cached analysis found for function at {function_address}")
                
                # Clear the response window and actions table when no cache available
                self.text_box.clear()
                self.actions_table.setRowCount(0)
                
        except Exception as e:
            log.log_error(f"[BinAssist] Error checking cached analysis: {e}")

    def display_cached_explanation(self, cached_analysis) -> None:
        """
        Display cached explanation results in the text box.
        
        Parameters:
            cached_analysis: Analysis object containing cached query and response
        """
        try:
            log.log_info(f"[BinAssist] Displaying cached explanation from {cached_analysis.timestamp}")
            
            # Validate cached response
            if not cached_analysis.response:
                log.log_error("[BinAssist] Cached analysis has empty response")
                return
            
            # The response is stored as plain text, no JSON parsing needed
            response_text = cached_analysis.response
            
            # Display in the text box
            import markdown
            html_resp = markdown.markdown(response_text, extensions=['fenced_code'])
            html_resp += self._generate_feedback_buttons()
            
            self.text_box.setHtml(html_resp)
            self.response = response_text
            
            log.log_debug("[BinAssist] Cached explanation displayed successfully")
            
        except Exception as e:
            log.log_error(f"[BinAssist] Error displaying cached explanation: {e}")

    def _revert_button_state(self, caller="unknown") -> None:
        """
        Safely revert button state back to original label.
        This method includes safety checks and error handling.
        """
        try:
            log.log_info(f"[BinAssist] _revert_button_state called by: {caller}")
            # Stop the timeout timer
            if hasattr(self, 'button_timeout_timer'):
                self.button_timeout_timer.stop()
                
            # Log thread status but don't block button revert
            if hasattr(self, 'LlmApi'):
                is_running = self.LlmApi.isRunning()
                log.log_debug(f"[BinAssist] Thread status check: isRunning={is_running}")
                # Don't return early - let button revert anyway since response was received
                
            # Safely revert button text
            if hasattr(self, 'submit_button') and hasattr(self, 'submit_label'):
                current_text = self.submit_button.text()
                target_text = self.submit_label
                log.log_debug(f"[BinAssist] Button state: current='{current_text}', target='{target_text}'")
                
                if current_text != target_text:
                    self.submit_button.setText(target_text)
                    log.log_info(f"[BinAssist] Button successfully reverted from '{current_text}' to '{target_text}'")
                else:
                    log.log_debug(f"[BinAssist] Button already in correct state: '{current_text}'")
                    
        except Exception as e:
            log.log_error(f"[BinAssist] Error reverting button state: {e}")
            # Last resort - try to set to common labels
            try:
                if hasattr(self, 'submit_button'):
                    current_text = self.submit_button.text()
                    if current_text == "Stop":
                        # Try to guess the original label
                        self.submit_button.setText("Submit")
                        log.log_debug("[BinAssist] Button reverted to default 'Submit'")
            except:
                pass  # Give up gracefully

    def _handle_button_timeout(self) -> None:
        """
        Handle button timeout - force revert button state if query takes too long.
        This prevents the plugin from hanging indefinitely.
        """
        log.log_warning("[BinAssist] Button timeout reached - forcing button state reset")
        
        try:
            # Force stop all threads
            if hasattr(self, 'LlmApi'):
                self.LlmApi.stop_threads()
                
            # Force revert button
            if hasattr(self, 'submit_button') and hasattr(self, 'submit_label'):
                self.submit_button.setText(self.submit_label)
                log.log_info("[BinAssist] Button forcibly reverted due to timeout")
                
        except Exception as e:
            log.log_error(f"[BinAssist] Error in timeout handler: {e}")

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
        # Get the current active provider's model
        active_provider_name = self.settings.get_string('active_provider')
        providers = self.settings.get_json('api_providers', [])
        current_model = 'unknown'
        for provider in providers:
            if provider.get('name') == active_provider_name:
                current_model = provider.get('model', 'unknown')
                break
        self.LlmApi.store_feedback(current_model, self.request, self.response, 1)

    def handleThumbsDown(self) -> None:
        """
        Handles the event when the 'Thumbs Down' feedback is given.
        """
        print("RLHF Downvote")
        # Get the current active provider's model
        active_provider_name = self.settings.get_string('active_provider')
        providers = self.settings.get_json('api_providers', [])
        current_model = 'unknown'
        for provider in providers:
            if provider.get('name') == active_provider_name:
                current_model = provider.get('model', 'unknown')
                break
        self.LlmApi.store_feedback(current_model, self.request, self.response, 0)

    def notifyOffsetChanged(self, offset) -> None:
        """
        Updates the displayed offset when it is changed in the binary view.

        Parameters:
            offset (int): The new offset value.
        """
        self.offset.setText(hex(offset))
        self.offset_addr = offset
        
        # Check for cached analysis when navigating to a new function
        self.checkAndDisplayCachedAnalysis()

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
            
            # Check for cached analysis when view changes (new binary loaded)
            self.checkAndDisplayCachedAnalysis()

    def contextMenuEvent(self, event) -> None:
        """Handle context menu event."""
        _ = event  # Event parameter required by Qt but not used
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
