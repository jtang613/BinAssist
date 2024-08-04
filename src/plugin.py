from binaryninja.settings import Settings
from binaryninjaui import SidebarWidget, SidebarWidgetType, UIActionHandler, SidebarWidgetLocation, \
    SidebarContextSensitivity, ViewFrame, ViewType, UIContext
from binaryninja import FunctionGraphType
from PySide6 import QtCore, QtGui, QtWidgets
import markdown
import sqlite3
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

        self.offset = QtWidgets.QLabel(hex(0))

        self.text_box = QtWidgets.QTextBrowser()
        self.text_box.setReadOnly(True)
        self.text_box.setOpenLinks(False)
        self.text_box.anchorClicked.connect(self.onAnchorClicked)

        self.query_edit = QtWidgets.QTextEdit()
        self.query_edit.setPlaceholderText("Enter your query here...")
        self.query_response_browser = QtWidgets.QTextBrowser()
        self.query_response_browser.setReadOnly(True)
        self.query_response_browser.setOpenLinks(False)
        self.query_response_browser.anchorClicked.connect(self.onAnchorClicked)

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

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(explain_tab, "Explain")
        self.tabs.addTab(query_tab, "Custom Query")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def createExplainTab(self) -> QtWidgets.QWidget:
        """
        Creates the tab and layout for the explanation functionalities.

        Returns:
            QWidget: A widget configured with explanation functionalities.
        """
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
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self.query_edit)
        splitter.addWidget(self.query_response_browser)
        splitter.setSizes([200, 300])
        layout.addWidget(splitter)
        layout.addLayout(self._create_query_buttons_layout())
        query_widget = QtWidgets.QWidget()
        query_widget.setLayout(layout)
        return query_widget

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
        clear_text_bt = QtWidgets.QPushButton("Clear", self)
        explain_il_bt.clicked.connect(self.onExplainILClicked)
        clear_text_bt.clicked.connect(self.onClearTextClicked)
        layout.addWidget(explain_il_bt)
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

    def get_line_text(self):
        """
        Determines the appropriate function to convert binary view data into text based on the current 
        intermediate language (IL) type set for the widget.

        Returns:
            callable: A function from LlmApi corresponding to the current IL type that converts binary data to text.
        """
        line = None
        if self.il_type == FunctionGraphType.NormalFunctionGraph:
            func = self.LlmApi.AsmLineToText
        if self.il_type == FunctionGraphType.LowLevelILFunctionGraph:
            func = self.LlmApi.LLILLineToText
        if self.il_type == FunctionGraphType.MediumLevelILFunctionGraph:
            func = self.LlmApi.MLILLineToText
        if self.il_type == FunctionGraphType.HighLevelILFunctionGraph:
            func = self.LlmApi.HLILLineToText
        if self.il_type == FunctionGraphType.HighLevelLanguageRepresentationFunctionGraph:
            func = self.LlmApi.HLILLineToText
        return func

    def onExplainILClicked(self) -> None:
        """
        Handles the event when the 'Explain Function' button is clicked.
        """
        datatype = self.datatype.split(':')[1]
        il_type = self.il_type.name
        func = self.get_func_text()
        self.text_box.clear()
        self.request = self.LlmApi.explain(self.bv, self.offset_addr, datatype, il_type, func, self.display_response)

    def onClearTextClicked(self) -> None:
        """
        Clears all text boxes when the 'Clear' button is clicked.
        """
        self.text_box.clear()
        self.query_response_browser.clear()

    def onSubmitQueryClicked(self) -> None:
        """
        Submits the custom query entered by the user when the 'Submit' button is clicked.
        """
        query = self.query_edit.toPlainText()
        query = self.process_custom_query(query)
        self.query_response_browser.clear()
        self.request = self.LlmApi.query(query, self.display_custom_response)

    def display_response(self, response) -> None:
        """
        Displays the formatted response from the language model.

        Parameters:
            response (str): The response to be displayed.
        """
        html_resp = markdown.markdown(response, extensions=['fenced_code'])
        html_resp += self._generate_feedback_buttons()
        self.response = response
        self.text_box.setHtml(html_resp)

    def display_custom_response(self, response) -> None:
        """
        Displays the custom formatted response from the language model.

        Parameters:
            response (str): The custom response to be displayed.
        """
        html_resp = markdown.markdown(response, extensions=['fenced_code'])
        html_resp += self._generate_feedback_buttons()
        self.response = response
        self.query_response_browser.setHtml(html_resp)

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
        line = self.get_line_text()

        query = query.replace("#line", f'\n```\n{line(self.bv, self.offset_addr)}\n```\n')
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
