#!/usr/bin/env python3

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QPushButton, QTextBrowser, QTextEdit, QLineEdit, QSizePolicy, QCheckBox)
from PySide6.QtCore import Signal, Qt
import markdown


class ExplainTabView(QWidget):
    # Signals for controller communication
    explain_function_requested = Signal()
    explain_line_requested = Signal()
    stop_function_requested = Signal()
    stop_line_requested = Signal()
    clear_requested = Signal()
    edit_mode_changed = Signal(bool)
    rag_enabled_changed = Signal(bool)
    mcp_enabled_changed = Signal(bool)
    # RLHF feedback signals
    rlhf_feedback_requested = Signal(bool)  # True for upvote, False for downvote
    
    def __init__(self):
        super().__init__()
        self.is_edit_mode = False
        self.function_query_running = False
        self.line_query_running = False
        self.markdown_content = "No explanation available. Click 'Explain Function' or 'Explain Line' to generate content."
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Top row - Current Offset and Edit/Save button
        self.create_top_row(layout)
        
        # Main text widget - HTML browser/Markdown editor
        self.create_main_text_widget(layout)
        
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
        
        # Size-constrained Edit button
        self.edit_save_button = QPushButton("Edit")
        self.edit_save_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.edit_save_button.clicked.connect(self.toggle_edit_mode)
        
        top_row.addWidget(offset_label)
        top_row.addWidget(self.current_offset_label)
        top_row.addStretch()  # Push checkboxes and button to the right
        top_row.addWidget(self.rag_checkbox)
        top_row.addWidget(self.mcp_checkbox)
        top_row.addWidget(self.edit_save_button)
        
        parent_layout.addLayout(top_row)
    
    def create_main_text_widget(self, parent_layout):
        # HTML browser for read-only mode
        self.explain_browser = QTextBrowser()
        self.explain_browser.setHtml(self.markdown_to_html(self.markdown_content))
        self.explain_browser.anchorClicked.connect(self._on_anchor_clicked)
        # Prevent default navigation for custom protocols
        self.explain_browser.setOpenLinks(False)
        
        # Text editor for edit mode
        self.explain_editor = QTextEdit()
        self.explain_editor.setPlainText(self.markdown_content)
        self.explain_editor.hide()  # Hidden by default
        
        parent_layout.addWidget(self.explain_browser)
        parent_layout.addWidget(self.explain_editor)
    
    def create_bottom_row(self, parent_layout):
        bottom_row = QHBoxLayout()
        
        self.explain_function_button = QPushButton("Explain Function")
        self.explain_line_button = QPushButton("Explain Line")
        self.clear_button = QPushButton("Clear")
        
        # Connect button signals
        self.explain_function_button.clicked.connect(self._on_explain_function_clicked)
        self.explain_line_button.clicked.connect(self._on_explain_line_clicked)
        self.clear_button.clicked.connect(self.clear_requested.emit)
        
        bottom_row.addWidget(self.explain_function_button)
        bottom_row.addWidget(self.explain_line_button)
        bottom_row.addWidget(self.clear_button)
        
        parent_layout.addLayout(bottom_row)
    
    def toggle_edit_mode(self):
        # Prevent edit mode changes during query execution
        if self.function_query_running or self.line_query_running:
            return
            
        self.is_edit_mode = not self.is_edit_mode
        
        if self.is_edit_mode:
            # Switch to edit mode
            self.explain_browser.hide()
            self.explain_editor.show()
            self.edit_save_button.setText("Save")
            # Copy current content to editor
            self.explain_editor.setPlainText(self.markdown_content)
        else:
            # Switch to read mode (save)
            self.explain_editor.hide()
            self.explain_browser.show()
            self.edit_save_button.setText("Edit")
            # Save edited content
            self.markdown_content = self.explain_editor.toPlainText()
            self.explain_browser.setHtml(self.markdown_to_html(self.markdown_content))
        
        self.edit_mode_changed.emit(self.is_edit_mode)
    
    def set_current_offset(self, offset_hex):
        """Update the displayed current offset"""
        self.current_offset_label.setText(offset_hex)
    
    def set_explanation_content(self, markdown_text):
        """Set the explanation content (in Markdown format)"""
        self.markdown_content = markdown_text
        if not self.is_edit_mode:
            # Check if user is near the bottom before updating
            should_auto_scroll = self._should_auto_scroll_to_bottom()
            
            self.explain_browser.setHtml(self.markdown_to_html(markdown_text))
            
            # Auto-scroll to bottom if user was following the explanation
            if should_auto_scroll:
                self._scroll_to_bottom()
        else:
            self.explain_editor.setPlainText(markdown_text)
    
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
    
    def get_explanation_content(self):
        """Get the current explanation content"""
        if self.is_edit_mode:
            return self.explain_editor.toPlainText()
        return self.markdown_content
    
    def clear_content(self):
        """Clear the explanation content"""
        default_content = "No explanation available. Click 'Explain Function' or 'Explain Line' to generate content."
        self.set_explanation_content(default_content)
    
    def _should_auto_scroll_to_bottom(self):
        """Check if we should auto-scroll to bottom (user is following the explanation)"""
        if not hasattr(self, 'explain_browser'):
            return True
            
        scrollbar = self.explain_browser.verticalScrollBar()
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
        if hasattr(self, 'explain_browser'):
            scrollbar = self.explain_browser.verticalScrollBar()
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
    
    def _on_explain_function_clicked(self):
        """Handle explain function button click - toggles between explain and stop"""
        if self.function_query_running:
            self.stop_function_requested.emit()
        else:
            self.explain_function_requested.emit()
    
    def _on_explain_line_clicked(self):
        """Handle explain line button click - toggles between explain and stop"""
        if self.line_query_running:
            self.stop_line_requested.emit()
        else:
            self.explain_line_requested.emit()
    
    def set_function_query_running(self, running: bool):
        """Update function query running state and button text"""
        self.function_query_running = running
        if running:
            self.explain_function_button.setText("Stop")
            self.explain_function_button.setStyleSheet("background-color: #ff6b6b; color: white;")
            # Disable edit button during query to prevent conflicts
            self.edit_save_button.setEnabled(False)
            self.edit_save_button.setToolTip("Edit mode disabled during query execution")
        else:
            self.explain_function_button.setText("Explain Function")
            self.explain_function_button.setStyleSheet("")
            # Re-enable edit button if no other query is running
            if not self.line_query_running:
                self.edit_save_button.setEnabled(True)
                self.edit_save_button.setToolTip("Toggle edit mode")
    
    def set_line_query_running(self, running: bool):
        """Update line query running state and button text"""
        self.line_query_running = running
        if running:
            self.explain_line_button.setText("Stop")
            self.explain_line_button.setStyleSheet("background-color: #ff6b6b; color: white;")
            # Disable edit button during query to prevent conflicts
            self.edit_save_button.setEnabled(False)
            self.edit_save_button.setToolTip("Edit mode disabled during query execution")
        else:
            self.explain_line_button.setText("Explain Line")
            self.explain_line_button.setStyleSheet("")
            # Re-enable edit button if no other query is running
            if not self.function_query_running:
                self.edit_save_button.setEnabled(True)
                self.edit_save_button.setToolTip("Toggle edit mode")