#!/usr/bin/env python3

import os
from binaryninjaui import SidebarWidget, SidebarWidgetType, Sidebar, UIActionHandler, SidebarWidgetLocation, SidebarContextSensitivity
from PySide6 import QtCore
from PySide6.QtCore import Qt, QRectF
from PySide6.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QLabel, QWidget, QTabWidget
from PySide6.QtGui import QImage, QPixmap, QPainter, QFont, QColor
from PySide6.QtSvg import QSvgRenderer
from .explain_tab_view import ExplainTabView
from .query_tab_view import QueryTabView
from .actions_tab_view import ActionsTabView
from .rag_tab_view import RagTabView
from .settings_tab_view import SettingsTabView
from .semantic_graph_tab_view import SemanticGraphTabView
from ..controllers.settings_controller import SettingsController
from ..controllers.explain_controller import ExplainController
from ..controllers.query_controller import QueryController
from ..controllers.rag_controller import RAGController
from ..controllers.actions_controller import ActionsController
from ..controllers.semantic_graph_controller import SemanticGraphController


class BinAssistSidebarWidget(SidebarWidget):
    def __init__(self, name, frame, data):
        SidebarWidget.__init__(self, name)
        self.actionHandler = UIActionHandler()
        self.actionHandler.setupActionHandler(self)
        
        self.frame = frame
        self.data = data
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.create_explain_tab()
        self.create_query_tab()
        self.create_actions_tab()
        self.create_semantic_graph_tab()
        self.create_rag_tab()
        self.create_settings_tab()
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
    
    def create_explain_tab(self):
        self.explain_tab = ExplainTabView()
        
        # Create controller to manage explain functionality
        self.explain_controller = ExplainController(self.explain_tab, self.data, self.frame)
        
        # Connect RLHF feedback signal
        self.explain_tab.rlhf_feedback_requested.connect(self.explain_controller.handle_rlhf_feedback)
        
        self.tab_widget.addTab(self.explain_tab, "Explain")
    
    def create_query_tab(self):
        self.query_tab = QueryTabView()
        
        # Create controller to manage query functionality
        self.query_controller = QueryController(self.query_tab, self.data, self.frame)
        
        # Connect RLHF feedback signal
        self.query_tab.rlhf_feedback_requested.connect(self.query_controller.handle_rlhf_feedback)
        
        self.tab_widget.addTab(self.query_tab, "Query")
    
    def create_actions_tab(self):
        self.actions_tab = ActionsTabView()
        
        # Create controller to manage actions functionality
        self.actions_controller = ActionsController(self.actions_tab, self.data, self.frame)
        
        # Connect signals to controller methods
        self.actions_tab.analyse_function_requested.connect(self.actions_controller.analyze_current_function)
        self.actions_tab.apply_actions_requested.connect(self.actions_controller.apply_selected_actions)
        self.actions_tab.clear_actions_requested.connect(self.actions_controller.clear_actions)
        
        # Populate available actions in the view
        available_actions = self.actions_controller.get_available_actions()
        for action in available_actions:
            self.actions_tab.add_available_action(action['name'], action['description'])
        
        self.tab_widget.addTab(self.actions_tab, "Actions")
    
    def create_rag_tab(self):
        self.rag_tab = RagTabView()

        # Create controller to manage RAG functionality
        self.rag_controller = RAGController(self.rag_tab)

        self.tab_widget.addTab(self.rag_tab, "RAG")

    def create_semantic_graph_tab(self):
        self.semantic_graph_tab = SemanticGraphTabView()
        self.semantic_graph_controller = SemanticGraphController(self.semantic_graph_tab, self.data, self.frame)
        self.tab_widget.addTab(self.semantic_graph_tab, "Semantic Graph")
    
    def create_settings_tab(self):
        self.settings_tab = SettingsTabView()
        
        # Create controller to manage settings (replaces placeholder methods)
        self.settings_controller = SettingsController(self.settings_tab)
        
        self.tab_widget.addTab(self.settings_tab, "Settings")
    
    def notifyOffsetChanged(self, offset):
        # Update tabs with current offset
        offset_hex = hex(offset)
        if hasattr(self, 'explain_tab'):
            self.explain_tab.set_current_offset(offset_hex)
        if hasattr(self, 'query_tab'):
            self.query_tab.set_current_offset(offset_hex)
        if hasattr(self, 'actions_tab'):
            self.actions_tab.set_current_offset(offset_hex)
        
        # Update controllers with numeric offset for context service
        if hasattr(self, 'explain_controller'):
            self.explain_controller.set_current_offset(offset)
        if hasattr(self, 'query_controller'):
            self.query_controller.set_current_offset(offset)
        if hasattr(self, 'actions_controller'):
            self.actions_controller.set_current_offset(offset)
        if hasattr(self, 'semantic_graph_controller'):
            self.semantic_graph_controller.set_current_offset(offset)
    
    def notifyViewChanged(self, view_frame):
        if view_frame is None:
            self.data = None
            self.frame = None
        else:
            view = view_frame.getCurrentViewInterface()
            self.data = view.getData()
            self.frame = view_frame
        
        # Update controllers with new binary view and view frame
        if hasattr(self, 'explain_controller'):
            self.explain_controller.set_binary_view(self.data)
            self.explain_controller.set_view_frame(self.frame)
        if hasattr(self, 'query_controller'):
            self.query_controller.set_binary_view(self.data)
            self.query_controller.set_view_frame(self.frame)
        if hasattr(self, 'actions_controller'):
            if self.data is not None:
                self.actions_controller.set_binary_view(self.data)
            self.actions_controller.set_view_frame(self.frame)
        if hasattr(self, 'semantic_graph_controller'):
            self.semantic_graph_controller.set_binary_view(self.data)
            self.semantic_graph_controller.set_view_frame(self.frame)
    
    def contextMenuEvent(self, event):
        self.m_contextMenuManager.show(self.m_menu, self.actionHandler)
    
    
    # Placeholder signal handlers for RAG tab (will be moved to controller later)
    def on_add_documents(self):
        # TODO: Implement document selection and addition to RAG database
        pass
    
    def on_refresh_documents(self):
        # TODO: Implement refresh from RAG database
        if hasattr(self, 'rag_tab'):
            # For now, just repopulate sample data
            self.rag_tab.clear_documents()
            self.rag_tab.populate_sample_data()
    
    def on_delete_documents(self, selected_documents):
        # TODO: Implement deletion from RAG database
        if hasattr(self, 'rag_tab'):
            self.rag_tab.remove_selected_documents()
    


class BinAssistSidebarWidgetType(SidebarWidgetType):
    def __init__(self):
        # Load the robot SVG icon
        icon = QImage(56, 56, QImage.Format_RGB32)
        icon.fill(0)
        
        # Get the path to the SVG file
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        svg_path = os.path.join(plugin_dir, "res", "robot.svg")
        
        if os.path.exists(svg_path):
            # Render SVG to the icon with transparent background
            renderer = QSvgRenderer(svg_path)
            p = QPainter()
            p.begin(icon)
            # No background fill - keep transparent
            renderer.render(p, QRectF(4, 4, 48, 48))  # Render with some padding
            p.end()
        else:
            # Fallback to "BA" text if SVG not found
            p = QPainter()
            p.begin(icon)
            p.setFont(QFont("Open Sans", 20))
            p.setPen(QColor(255, 255, 255, 255))
            p.drawText(QRectF(0, 0, 56, 56), Qt.AlignCenter, "BA")
            p.end()
        
        SidebarWidgetType.__init__(self, icon, "BinAssist")
    
    def createWidget(self, frame, data):
        return BinAssistSidebarWidget("BinAssist", frame, data)
    
    def defaultLocation(self):
        return SidebarWidgetLocation.RightContent
    
    def contextSensitivity(self):
        return SidebarContextSensitivity.SelfManagedSidebarContext
