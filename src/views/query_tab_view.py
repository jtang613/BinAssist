#!/usr/bin/env python3

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                              QTextBrowser, QTextEdit, QLineEdit, QTableWidget, QTableWidgetItem,
                              QSplitter, QPlainTextEdit, QHeaderView, QAbstractItemView, QSizePolicy, QCheckBox)
from PySide6.QtCore import Signal, Qt, QDateTime
from PySide6.QtGui import QKeySequence
import markdown


class QueryTabView(QWidget):
    # Signals for controller communication
    submit_query_requested = Signal(str)  # query text
    stop_query_requested = Signal()
    new_chat_requested = Signal()
    delete_chats_requested = Signal(list)  # list of chat IDs
    chat_selected = Signal(int)  # chat ID
    chat_name_changed = Signal(int, str)  # chat ID, new name
    edit_mode_changed = Signal(bool)
    rag_enabled_changed = Signal(bool)
    mcp_enabled_changed = Signal(bool)
    agentic_enabled_changed = Signal(bool)
    # RLHF feedback signals
    rlhf_feedback_requested = Signal(bool)  # True for upvote, False for downvote
    
    def __init__(self):
        super().__init__()
        self.is_edit_mode = False
        self.markdown_content = "No chat selected. Click 'New' to start a conversation."
        self.chat_counter = 0
        self.current_chat_id = None
        self.query_running = False
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Top row - Current Offset and Edit/Save button
        self.create_top_row(layout)
        
        # Create splitter for resizable sections
        self.splitter = QSplitter(Qt.Vertical)
        
        # Main text widget - HTML browser/Markdown editor
        self.create_main_text_widget()
        
        # History table
        self.create_history_table()
        
        # Input text widget
        self.create_input_widget()
        
        # Add widgets to splitter
        self.splitter.addWidget(self.query_browser)
        self.splitter.addWidget(self.query_editor)
        self.splitter.addWidget(self.history_table)
        self.splitter.addWidget(self.input_widget)
        
        # Set initial splitter sizes (give more space to main text and input)
        self.splitter.setSizes([400, 400, 80, 100])
        
        layout.addWidget(self.splitter)
        
        # Bottom row - Action buttons
        self.create_bottom_row(layout)
        
        self.setLayout(layout)
    
    def create_top_row(self, parent_layout):
        top_row = QHBoxLayout()
        
        # Size-constrained and left-justified labels
        offset_label = QLabel("Current Offset:")
        offset_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        offset_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        self.current_offset_label = QLabel("0x0")
        self.current_offset_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.current_offset_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        # RAG and MCP checkboxes
        self.rag_checkbox = QCheckBox("RAG")
        self.rag_checkbox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.rag_checkbox.setChecked(False)  # Default disabled
        self.rag_checkbox.toggled.connect(self.rag_enabled_changed.emit)
        
        self.mcp_checkbox = QCheckBox("MCP")
        self.mcp_checkbox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.mcp_checkbox.setChecked(False)  # Default disabled
        self.mcp_checkbox.toggled.connect(self.mcp_enabled_changed.emit)

        # Agentic mode checkbox (ReAct autonomous agent)
        self.agentic_checkbox = QCheckBox("Agentic")
        self.agentic_checkbox.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.agentic_checkbox.setChecked(False)  # Default disabled
        self.agentic_checkbox.setToolTip("Enable ReAct autonomous agent for complex investigations")
        self.agentic_checkbox.toggled.connect(self.agentic_enabled_changed.emit)

        # Size-constrained Edit button
        self.edit_save_button = QPushButton("Edit")
        self.edit_save_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.edit_save_button.clicked.connect(self.toggle_edit_mode)
        
        top_row.addWidget(offset_label)
        top_row.addWidget(self.current_offset_label)
        top_row.addStretch()  # Push checkboxes and button to the right
        top_row.addWidget(self.rag_checkbox)
        top_row.addWidget(self.mcp_checkbox)
        top_row.addWidget(self.agentic_checkbox)
        top_row.addWidget(self.edit_save_button)
        
        parent_layout.addLayout(top_row)
    
    def create_main_text_widget(self):
        # HTML browser for read-only mode
        self.query_browser = QTextBrowser()
        self.query_browser.setHtml(self.markdown_to_html(self.markdown_content))
        self.query_browser.anchorClicked.connect(self._on_anchor_clicked)
        # Prevent default navigation for custom protocols
        self.query_browser.setOpenLinks(False)
        
        # Text editor for edit mode
        self.query_editor = QTextEdit()
        self.query_editor.setPlainText(self.markdown_content)
        self.query_editor.hide()  # Hidden by default
    
    def create_history_table(self):
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(2)
        self.history_table.setHorizontalHeaderLabels(["Description", "Timestamp"])
        
        # Configure table behavior
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.setSelectionMode(QAbstractItemView.ExtendedSelection)  # Allow multi-select
        self.history_table.itemSelectionChanged.connect(self.on_history_selection_changed)
        self.history_table.itemDoubleClicked.connect(self.on_history_item_double_clicked)
        self.history_table.itemChanged.connect(self.on_history_item_changed)
        
        # Enable sorting and set default sort order (timestamp descending - newest first)
        self.history_table.setSortingEnabled(True)
        self.history_table.sortByColumn(1, Qt.DescendingOrder)  # Sort by timestamp column, descending
        
        # Make table columns resizable
        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)  # Description column
        header.setSectionResizeMode(1, QHeaderView.Interactive)  # Timestamp column
        header.setStretchLastSection(True)  # Stretch last column to fill
        
        # Set initial column widths - wider description, narrower timestamp
        self.history_table.setColumnWidth(0, 300)  # Description column - wider for descriptive text
        self.history_table.setColumnWidth(1, 10)  # Timestamp column - just wide enough for YYYY-MM-DD HH:mm:ss
        
        # Set default height to show 3 rows
        row_height = self.history_table.verticalHeader().defaultSectionSize()
        header_height = self.history_table.horizontalHeader().height()
        self.history_table.setMinimumHeight(header_height + (row_height * 3))  # +10 for margins
    
    def create_input_widget(self):
        self.input_widget = QPlainTextEdit()
        self.input_widget.setPlaceholderText("Enter your query here...\nUse #func, #addr, #line, #range(start, end) for context macros.\n(Press Enter to submit, Ctrl+Enter for new line)")
        # self.input_widget.setMaximumHeight(100)  # Limit initial height
        
        # Handle key events for Enter and Ctrl+Enter
        self.input_widget.keyPressEvent = self.input_key_press_event
    
    def create_bottom_row(self, parent_layout):
        bottom_row = QHBoxLayout()
        
        self.submit_button = QPushButton("Submit")
        self.new_button = QPushButton("New")
        self.delete_button = QPushButton("Delete")
        
        # Connect button signals
        self.submit_button.clicked.connect(self.on_submit_clicked)
        self.new_button.clicked.connect(self.new_chat_requested.emit)
        self.delete_button.clicked.connect(self.on_delete_clicked)
        
        bottom_row.addWidget(self.submit_button)
        bottom_row.addWidget(self.new_button)
        bottom_row.addWidget(self.delete_button)
        
        parent_layout.addLayout(bottom_row)
    
    def input_key_press_event(self, event):
        """Handle Enter and Ctrl+Enter in input widget"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if event.modifiers() == Qt.ControlModifier:
                # Ctrl+Enter: Insert newline
                self.input_widget.insertPlainText("\n")
            else:
                # Enter: Submit query (or stop if running)
                self.on_submit_clicked()
        else:
            # Default behavior for other keys
            QPlainTextEdit.keyPressEvent(self.input_widget, event)
    
    def toggle_edit_mode(self):
        # Prevent edit mode changes during query execution
        if self.query_running:
            return
            
        self.is_edit_mode = not self.is_edit_mode
        
        if self.is_edit_mode:
            # Switch to edit mode
            self.query_browser.hide()
            self.query_editor.show()
            self.edit_save_button.setText("Save")
            # Copy current content to editor
            self.query_editor.setPlainText(self.markdown_content)
        else:
            # Switch to read mode (save)
            self.query_editor.hide()
            self.query_browser.show()
            self.edit_save_button.setText("Edit")
            # Save edited content
            self.markdown_content = self.query_editor.toPlainText()
            self.query_browser.setHtml(self.markdown_to_html(self.markdown_content))
        
        self.edit_mode_changed.emit(self.is_edit_mode)
    
    def on_submit_clicked(self):
        """Handle submit button click - toggles between submit and stop"""
        if self.query_running:
            self.stop_query_requested.emit()
        else:
            query_text = self.input_widget.toPlainText().strip()
            if query_text:
                # If no current chat exists, create a new one first
                if self.current_chat_id is None:
                    self.new_chat_requested.emit()
                
                self.submit_query_requested.emit(query_text)
                self.input_widget.clear()  # Clear input after submit
    
    def on_delete_clicked(self):
        selected_rows = []
        for item in self.history_table.selectedItems():
            if item.column() == 0:  # Only get one item per row
                selected_rows.append(item.row())
        
        if selected_rows:
            self.delete_chats_requested.emit(selected_rows)
    
    def on_history_selection_changed(self):
        selected_items = self.history_table.selectedItems()
        if len(selected_items) == 2:  # Single row selected (2 columns)
            row = selected_items[0].row()
            chat_id = self.history_table.item(row, 0).data(Qt.UserRole)  # Store chat ID in user data
            if chat_id is not None:
                self.current_chat_id = chat_id
                self.chat_selected.emit(chat_id)
    
    def on_history_item_double_clicked(self, item):
        if item.column() == 0:  # Description column
            # Enable editing for the description
            self.history_table.editItem(item)
    
    def on_history_item_changed(self, item):
        """Handle when a history table item is edited"""
        if item.column() == 0:  # Description column
            # Get the chat ID and new name
            chat_id = item.data(Qt.UserRole)
            new_name = item.text().strip()
            
            if chat_id is not None and new_name:
                # Emit signal to notify controller of the name change
                self.chat_name_changed.emit(chat_id, new_name)
    
    def add_chat_to_history(self, chat_id, description, timestamp=None):
        """Add a new chat entry to the history table"""
        if timestamp is None:
            timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        
        # Temporarily disable sorting and item change signals while adding the item
        was_sorting_enabled = self.history_table.isSortingEnabled()
        self.history_table.setSortingEnabled(False)
        
        # Temporarily disconnect itemChanged signal to prevent false triggers during loading
        self.history_table.itemChanged.disconnect()
        
        row_count = self.history_table.rowCount()
        self.history_table.insertRow(row_count)
        
        desc_item = QTableWidgetItem(description)
        desc_item.setData(Qt.UserRole, chat_id)  # Store chat ID
        timestamp_item = QTableWidgetItem(timestamp)
        timestamp_item.setFlags(timestamp_item.flags() & ~Qt.ItemIsEditable)  # Make timestamp read-only
        
        self.history_table.setItem(row_count, 0, desc_item)
        self.history_table.setItem(row_count, 1, timestamp_item)
        
        # Reconnect the itemChanged signal
        self.history_table.itemChanged.connect(self.on_history_item_changed)
        
        # Re-enable sorting if it was enabled before
        if was_sorting_enabled:
            self.history_table.setSortingEnabled(True)
            # Trigger a sort to ensure proper order (newest first)
            self.history_table.sortByColumn(1, Qt.DescendingOrder)
        
        # Find and select the new chat (it may have moved due to sorting)
        self._select_chat_by_id(chat_id)
        self.current_chat_id = chat_id
    
    def _select_chat_by_id(self, chat_id):
        """Find and select a chat by its ID in the history table"""
        for row in range(self.history_table.rowCount()):
            item = self.history_table.item(row, 0)  # Description column
            if item and item.data(Qt.UserRole) == chat_id:
                self.history_table.selectRow(row)
                return
        # If not found, select the first row as fallback
        if self.history_table.rowCount() > 0:
            self.history_table.selectRow(0)
    
    def remove_selected_chats(self):
        """Remove selected chats from history table"""
        selected_rows = []
        for item in self.history_table.selectedItems():
            if item.column() == 0:  # Only get one item per row
                selected_rows.append(item.row())
        
        # Remove rows in reverse order to maintain indices
        for row in sorted(selected_rows, reverse=True):
            self.history_table.removeRow(row)
        
        # Clear current chat if it was deleted
        if not self.history_table.rowCount():
            self.current_chat_id = None
            self.set_chat_content("No chat selected. Click 'New' to start a conversation.")
    
    def set_current_offset(self, offset_hex):
        """Update the displayed current offset"""
        self.current_offset_label.setText(offset_hex)
    
    def is_rag_enabled(self):
        """Get the current state of the RAG checkbox"""
        return self.rag_checkbox.isChecked()
    
    def is_mcp_enabled(self):
        """Get the current state of the MCP checkbox"""
        return self.mcp_checkbox.isChecked()
    
    def set_rag_enabled(self, enabled):
        """Set the RAG checkbox state"""
        self.rag_checkbox.setChecked(enabled)
    
    def set_mcp_enabled(self, enabled):
        """Set the MCP checkbox state"""
        self.mcp_checkbox.setChecked(enabled)

    def is_agentic_enabled(self):
        """Get the current state of the Agentic checkbox"""
        return self.agentic_checkbox.isChecked()

    def set_agentic_enabled(self, enabled):
        """Set the Agentic checkbox state"""
        self.agentic_checkbox.setChecked(enabled)

    def set_query_running(self, running: bool):
        """Update query running state and button text"""
        self.query_running = running
        if running:
            self.submit_button.setText("Stop")
            self.submit_button.setStyleSheet("background-color: #ff6b6b; color: white;")
            # Disable edit button during query to prevent conflicts
            self.edit_save_button.setEnabled(False)
            self.edit_save_button.setToolTip("Edit mode disabled during query execution")
        else:
            self.submit_button.setText("Submit")
            self.submit_button.setStyleSheet("")
            # Re-enable edit button
            self.edit_save_button.setEnabled(True)
            self.edit_save_button.setToolTip("Toggle edit mode")
    
    def set_chat_content(self, markdown_text):
        """Set the chat content (in Markdown format)"""
        self.markdown_content = markdown_text
        
        if not self.is_edit_mode:
            # Check if user is near the bottom before updating
            should_auto_scroll = self._should_auto_scroll_to_bottom()
            
            html_content = self.markdown_to_html(markdown_text)
            self.query_browser.setHtml(html_content)
            
            # Auto-scroll to bottom if user was following the conversation
            if should_auto_scroll:
                self._scroll_to_bottom()
        else:
            self.query_editor.setPlainText(markdown_text)
    
    def get_chat_content(self):
        """Get the current chat content"""
        if self.is_edit_mode:
            return self.query_editor.toPlainText()
        return self.markdown_content
    
    def get_next_chat_name(self):
        """Generate next chat name (Chat 1, Chat 2, etc.)"""
        self.chat_counter += 1
        return f"Chat {self.chat_counter}"
    
    def _should_auto_scroll_to_bottom(self):
        """Check if we should auto-scroll to bottom (user is following the conversation)"""
        if not hasattr(self, 'query_browser'):
            return True
            
        scrollbar = self.query_browser.verticalScrollBar()
        if not scrollbar:
            return True
            
        # Check if user is near the bottom (within 50 pixels)
        current_pos = scrollbar.value()
        max_pos = scrollbar.maximum()
        
        # If there's no content to scroll, always auto-scroll
        if max_pos <= 0:
            return True
            
        # User is considered "following" if they're within 50px of bottom
        return (max_pos - current_pos) <= 50
    
    def _scroll_to_bottom(self):
        """Scroll the text browser to the bottom"""
        if hasattr(self, 'query_browser'):
            scrollbar = self.query_browser.verticalScrollBar()
            if scrollbar:
                scrollbar.setValue(scrollbar.maximum())
    
    def markdown_to_html(self, markdown_text):
        """Convert Markdown text to HTML for display"""
        try:
            html = markdown.markdown(markdown_text, extensions=['codehilite', 'fenced_code', 'tables'])
            
            # Add RLHF feedback links at the bottom
            feedback_html = self._get_feedback_html()
            
            return f"""
            <div style='font-family: Arial, sans-serif; font-size: 12px;'>
                {html}
                {feedback_html}
            </div>
            """
        except:
            # Fallback if markdown parsing fails
            return f"<pre>{markdown_text}</pre>"
    
    def _get_feedback_html(self):
        """Generate HTML for RLHF feedback thumbs up/down links"""
        return """
        <div style='text-align: center; margin-top: 20px; padding-top: 10px; border-top: 1px solid #ddd;'>
            <a href='rlhf://upvote' style='text-decoration: none; color: #666; margin-right: 15px; font-size: 14px;'>👍</a>
            <a href='rlhf://downvote' style='text-decoration: none; color: #666; font-size: 14px;'>👎</a>
        </div>
        """
    
    def _on_anchor_clicked(self, url):
        """Handle clicks on HTML anchors (specifically RLHF feedback links)"""
        url_str = url.toString()
        if url_str == "rlhf://upvote":
            self.rlhf_feedback_requested.emit(True)
        elif url_str == "rlhf://downvote":
            self.rlhf_feedback_requested.emit(False)
