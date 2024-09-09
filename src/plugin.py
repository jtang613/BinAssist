from binaryninja import *
from binaryninja.settings import Settings
from binaryninjaui import SidebarWidget, SidebarWidgetType, UIActionHandler, SidebarWidgetLocation, \
    SidebarContextSensitivity, ViewFrame, ViewType, UIContext
from binaryninja import FunctionGraphType, PythonScriptingProvider, PythonScriptingInstance
from PySide6 import QtCore, QtGui, QtWidgets
import markdown
from .llm_api import LlmApi


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
        super().__init__(name)
        self.settings = Settings()
        self.LlmApi = LlmApi()
        self.offset_addr = 0
        self.actionHandler = UIActionHandler()
        self.actionHandler.setupActionHandler(self)
        self.bv = None
        self.datatype = None
        self.il_type = None
        self.request = None
        self.response = None
        self.session_log = []

        self.offset = QtWidgets.QLabel(hex(0))

        self._init_ui()

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
        rag_management_tab = self.createRAGManagementTab()

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(explain_tab, "Explain")
        self.tabs.addTab(query_tab, "Custom Query")
        self.tabs.addTab(actions_tab, "Actions") 
        self.tabs.addTab(rag_management_tab, "RAG Management")

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
        self.query_edit = QtWidgets.QTextEdit()
        self.query_edit.setPlaceholderText("Enter your query here...")
        self.query_response_browser = QtWidgets.QTextBrowser()
        self.query_response_browser.setReadOnly(True)
        self.query_response_browser.setOpenLinks(False)
        self.query_response_browser.anchorClicked.connect(self.onAnchorClicked)

        layout = QtWidgets.QVBoxLayout()
        
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
        if self.il_type == FunctionGraphType.NormalFunctionGraph:
            func = self.LlmApi.AsmToText
        if self.il_type == FunctionGraphType.LowLevelILFunctionGraph:
            func = self.LlmApi.LLILToText
        if self.il_type == FunctionGraphType.MediumLevelILFunctionGraph:
            func = self.LlmApi.MLILToText
        if self.il_type == FunctionGraphType.HighLevelILFunctionGraph:
            func = self.LlmApi.HLILToText
        if self.il_type == FunctionGraphType.HighLevelLanguageRepresentationFunctionGraph:
            func = self.LlmApi.HLILToText
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
        """
        datatype = self.datatype.split(':')[1]
        il_type = self.il_type.name
        func = self.get_func_text()
        self.text_box.clear()
        self.request = self.LlmApi.explain(self.bv, self.offset_addr, datatype, il_type, func, self.display_response)

    def onExplainLineClicked(self) -> None:
        """
        Handles the event when the 'Explain Line' button is clicked.
        """
        datatype = self.datatype.split(':')[1]
        il_type = self.il_type.name
        self.text_box.clear()
        self.request = self.LlmApi.explain(self.bv, self.offset_addr, datatype, il_type, self.get_line_text, self.display_response)


    def onClearTextClicked(self) -> None:
        """
        Clears all text boxes when the 'Clear' button is clicked.
        """
        self.session_log.clear()  # Clear the session log
        self.text_box.clear()
        self.query_response_browser.clear()

    def onSubmitQueryClicked(self) -> None:
        """
        Submits the custom query entered by the user when the 'Submit' button is clicked.
        """
        query = self.query_edit.toPlainText()
        query = self.process_custom_query(query)
        self.session_log.append({"user": query, "assistant": "Awaiting response..."})

        # Prepend the session log to the query for context
        full_query = "\n".join([f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in self.session_log]) + f"\nUser: {query}"

        self.request = self.LlmApi.query(full_query, self.display_custom_response)

    def onAnalyzeFunctionClicked(self) -> None:
        """
        Event for the 'Analyze Function' button.
        """
        datatype = self.datatype.split(':')[1]
        il_type = self.il_type.name
        func = self.get_func_text()
        self.request = self.LlmApi.analyze_fn_names(self.bv, self.offset_addr, datatype, il_type, func, self.display_analyze_response)
        self.request = self.LlmApi.analyze_fn_names(self.bv, self.offset_addr, datatype, il_type, func, self.display_analyze_response)
        self.request = self.LlmApi.analyze_fn_vars(self.bv, self.offset_addr, datatype, il_type, func, self.display_analyze_response)

    def onAnalyzeClearClicked(self) -> None:
        """
        Event for the 'Analyze Clear' button.
        """
        print("Analyze Clear clicked")
        self.actions_table.setRowCount(0)

    def onApplyActionsClicked(self) -> None:
        """
        Applies the selected actions from the actions table.
        """
        for row in range(self.actions_table.rowCount()):
            # Check if the action is selected
            checkbox_widget = self.actions_table.cellWidget(row, 0)
            checkbox = checkbox_widget.findChild(QtWidgets.QCheckBox)
            if checkbox.isChecked():
                action_item = self.actions_table.item(row, 1)
                description_item = self.actions_table.item(row, 2)
                
                action = action_item.text()
                description = description_item.text()
                
                if action == "rename function":
                    # Parse the description to get address and new name
                    addr, new_name = description.split(' -> ')
                    addr = int(addr.replace('sub_',''), 16)  # Convert hex string to integer
                    self.bv.get_functions_containing(addr)[0].name = new_name
                    self.actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Applied"))
                
                elif action == "rename variable":
                    # Parse the description to get old name and new name
                    old_name, new_name = description.split(' -> ')
                    current_function = self.bv.get_functions_containing(self.offset_addr)[0]
                    if current_function:
                        for var in current_function.vars:
                            if var.name == old_name:
                                var.name = new_name
                                self.actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Applied"))
                                break
                        else:
                            self.actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Failed: Variable not found"))
                    else:
                        self.actions_table.setItem(row, 3, QtWidgets.QTableWidgetItem("Failed: Function not found"))
        
        # Update the analysis database
        self.bv.update_analysis()


    def display_response(self, response) -> None:
        """
        Displays the formatted response from the language model.

        Parameters:
            response (str): The response to be displayed.
        """
        html_resp = markdown.markdown(response["response"], extensions=['fenced_code'])
        html_resp += self._generate_feedback_buttons()
        self.response = response
        self.text_box.setHtml(html_resp)

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
        self.response = response
        self.query_response_browser.setHtml(html_resp)

    def display_analyze_response(self, response) -> None:
        """
        Displays the custom formatted response from the language model in the actions table.

        Parameters:
            response (str): The JSON response to be displayed.
        """
        actions = {'tool_calls':[]}
        if isinstance(response["response"], List):
            print(f"tool_calls: {response["response"]}")
            for it in response["response"]:
                print(f"{it.function.name} : {it.function.arguments}")
                print(f"type : {type(it.function.arguments)}")
                actions['tool_calls'].append({'name':it.function.name, 'arguments':json.loads(it.function.arguments)})
        else:
            try:
                # Parse the JSON response
                actions = json.loads(response["response"].replace('```json\n','').replace('```\n','').replace('\n```',''))
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
            
        # Populate the table with the parsed actions
        for idx, action in enumerate(actions.get("tool_calls", [])):
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


    def _format_action(self, action: dict) -> str:
        return f"{action['name'].replace('_',' ')}"

    def _format_description(self, action: dict) -> str:
        if action['name'] == 'rename_function':
            return f"{action['arguments']['addr']} -> {action['arguments']['name']}"
        if action['name'] == 'rename_variable':
            return f"{action['arguments']['var_name']} -> {action['arguments']['new_name']}"

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
                "#addr to include the current hex address."
                )

    def process_custom_query(self, query) -> str:
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

        query = query.replace("#line", f'\n```\n{line}\n```\n')
        query = query.replace('#func', f'\n```\n{func(self.bv, self.offset_addr)}\n```\n')
        query = query.replace("#addr", hex(self.offset_addr) or "")
        return query

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
