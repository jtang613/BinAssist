#!/usr/bin/env python3
"""
SymGraph.ai Tab View for BinAssist.

This view provides the UI for querying, pushing, and pulling
symbols and graph data from SymGraph.ai.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QLabel, QGroupBox, QRadioButton,
    QCheckBox, QButtonGroup, QSplitter, QFrame
)
from PySide6.QtCore import Signal, Qt

from ..services.models.symgraph_models import ConflictAction, ConflictEntry, PushScope


class SymGraphTabView(QWidget):
    """View for SymGraph.ai tab with query, push, and pull sections."""

    # Signals for controller communication
    query_requested = Signal()
    push_requested = Signal(str, bool, bool)  # scope, push_symbols, push_graph
    pull_preview_requested = Signal()
    apply_selected_requested = Signal(list)  # list of selected addresses
    select_all_requested = Signal()
    deselect_all_requested = Signal()
    invert_selection_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Set up the main UI layout."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Binary info header
        self.create_binary_info_section(layout)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Vertical)

        # Top panel - Query and Push sections
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 0)
        self.create_query_section(top_layout)
        self.create_push_section(top_layout)
        splitter.addWidget(top_panel)

        # Bottom panel - Pull section with conflict table
        bottom_panel = self.create_pull_section()
        splitter.addWidget(bottom_panel)

        # Set initial sizes (40% for query/push, 60% for pull)
        splitter.setSizes([400, 600])

        layout.addWidget(splitter)
        self.setLayout(layout)

    def create_binary_info_section(self, parent_layout):
        """Create the binary info header section."""
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(10, 10, 10, 10)

        # Binary name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Binary:"))
        self.binary_name_label = QLabel("<no binary loaded>")
        self.binary_name_label.setStyleSheet("font-weight: bold;")
        name_layout.addWidget(self.binary_name_label)
        name_layout.addStretch()
        info_layout.addLayout(name_layout)

        # SHA256 hash
        hash_layout = QHBoxLayout()
        hash_layout.addWidget(QLabel("SHA256:"))
        self.sha256_label = QLabel("<none>")
        self.sha256_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        hash_layout.addWidget(self.sha256_label)
        hash_layout.addStretch()
        info_layout.addLayout(hash_layout)

        parent_layout.addWidget(info_frame)

    def create_query_section(self, parent_layout):
        """Create the query section for checking SymGraph.ai status."""
        group = QGroupBox("Query Status")
        layout = QVBoxLayout()

        # Query button row
        button_row = QHBoxLayout()
        self.query_button = QPushButton("Check SymGraph.ai")
        self.query_button.clicked.connect(self.query_requested.emit)
        button_row.addWidget(self.query_button)
        button_row.addStretch()
        layout.addLayout(button_row)

        # Status label
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        self.status_label = QLabel("Not checked")
        self.status_label.setStyleSheet("color: gray;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # Stats frame (initially hidden)
        self.stats_frame = QFrame()
        self.stats_frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        stats_layout = QHBoxLayout(self.stats_frame)

        # Left column
        left_col = QVBoxLayout()
        self.symbols_stat = QLabel("Symbols: -")
        self.functions_stat = QLabel("Functions: -")
        left_col.addWidget(self.symbols_stat)
        left_col.addWidget(self.functions_stat)
        stats_layout.addLayout(left_col)

        # Right column
        right_col = QVBoxLayout()
        self.nodes_stat = QLabel("Graph Nodes: -")
        self.updated_stat = QLabel("Last Updated: -")
        right_col.addWidget(self.nodes_stat)
        right_col.addWidget(self.updated_stat)
        stats_layout.addLayout(right_col)

        self.stats_frame.setVisible(False)
        layout.addWidget(self.stats_frame)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_push_section(self, parent_layout):
        """Create the push section for uploading to SymGraph.ai."""
        group = QGroupBox("Push to SymGraph.ai")
        layout = QVBoxLayout()

        # Scope selection
        scope_layout = QHBoxLayout()
        scope_layout.addWidget(QLabel("Scope:"))

        self.scope_group = QButtonGroup(self)
        self.full_binary_radio = QRadioButton("Full Binary")
        self.current_function_radio = QRadioButton("Current Function")
        self.current_function_radio.setChecked(True)

        self.scope_group.addButton(self.full_binary_radio, 1)
        self.scope_group.addButton(self.current_function_radio, 2)

        scope_layout.addWidget(self.full_binary_radio)
        scope_layout.addWidget(self.current_function_radio)
        scope_layout.addStretch()
        layout.addLayout(scope_layout)

        # Data to push checkboxes
        data_layout = QHBoxLayout()
        data_layout.addWidget(QLabel("Data to Push:"))

        self.push_symbols_check = QCheckBox("Symbols (function names, variables, types)")
        self.push_symbols_check.setChecked(True)
        data_layout.addWidget(self.push_symbols_check)

        self.push_graph_check = QCheckBox("Graph (nodes, edges, summaries)")
        self.push_graph_check.setChecked(True)
        data_layout.addWidget(self.push_graph_check)

        data_layout.addStretch()
        layout.addLayout(data_layout)

        # Push button and status
        push_row = QHBoxLayout()
        self.push_button = QPushButton("Push to SymGraph.ai")
        self.push_button.clicked.connect(self.on_push_clicked)
        push_row.addWidget(self.push_button)

        self.push_status_label = QLabel("Status: Ready")
        self.push_status_label.setStyleSheet("color: gray;")
        push_row.addWidget(self.push_status_label)
        push_row.addStretch()
        layout.addLayout(push_row)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_pull_section(self):
        """Create the pull section with conflict resolution table."""
        group = QGroupBox("Pull from SymGraph.ai")
        layout = QVBoxLayout()

        # Pull preview button
        button_row = QHBoxLayout()
        self.pull_preview_button = QPushButton("Pull && Preview")
        self.pull_preview_button.clicked.connect(self.pull_preview_requested.emit)
        button_row.addWidget(self.pull_preview_button)
        button_row.addStretch()
        layout.addLayout(button_row)

        # Conflict resolution table
        self.create_conflict_table(layout)

        # Selection buttons
        selection_row = QHBoxLayout()
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self.on_select_all)
        selection_row.addWidget(self.select_all_button)

        self.deselect_all_button = QPushButton("Deselect All")
        self.deselect_all_button.clicked.connect(self.on_deselect_all)
        selection_row.addWidget(self.deselect_all_button)

        self.invert_selection_button = QPushButton("Invert Selection")
        self.invert_selection_button.clicked.connect(self.on_invert_selection)
        selection_row.addWidget(self.invert_selection_button)

        selection_row.addStretch()
        layout.addLayout(selection_row)

        # Apply and Cancel buttons
        action_row = QHBoxLayout()
        self.apply_button = QPushButton("Apply Selected")
        self.apply_button.clicked.connect(self.on_apply_clicked)
        action_row.addWidget(self.apply_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.clear_conflicts)
        action_row.addWidget(self.cancel_button)

        action_row.addStretch()

        self.pull_status_label = QLabel("")
        action_row.addWidget(self.pull_status_label)

        layout.addLayout(action_row)

        group.setLayout(layout)
        return group

    def create_conflict_table(self, parent_layout):
        """Create the conflict resolution table."""
        self.conflict_table = QTableWidget()
        self.conflict_table.setColumnCount(5)
        self.conflict_table.setHorizontalHeaderLabels([
            "Select", "Address", "Local Name", "Remote Name", "Action"
        ])

        # Configure table behavior
        self.conflict_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.conflict_table.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # Column sizing
        header = self.conflict_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)      # Select checkbox
        header.setSectionResizeMode(1, QHeaderView.Interactive)  # Address
        header.setSectionResizeMode(2, QHeaderView.Stretch)      # Local Name
        header.setSectionResizeMode(3, QHeaderView.Stretch)      # Remote Name
        header.setSectionResizeMode(4, QHeaderView.Fixed)        # Action

        self.conflict_table.setColumnWidth(0, 50)
        self.conflict_table.setColumnWidth(1, 100)
        self.conflict_table.setColumnWidth(4, 80)

        parent_layout.addWidget(self.conflict_table)

    # === Public methods for controller ===

    def set_binary_info(self, name: str, sha256: str):
        """Set the binary information display."""
        self.binary_name_label.setText(name or "<no binary loaded>")
        self.sha256_label.setText(sha256 or "<none>")

    def set_query_status(self, status: str, found: bool = False):
        """Set the query status display."""
        self.status_label.setText(status)
        if found:
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        elif "error" in status.lower() or "not found" in status.lower():
            self.status_label.setStyleSheet("color: red;")
        else:
            self.status_label.setStyleSheet("color: gray;")

    def set_stats(self, symbols: int, functions: int, nodes: int, last_updated: str):
        """Set the statistics display."""
        self.symbols_stat.setText(f"Symbols: {symbols:,}")
        self.functions_stat.setText(f"Functions: {functions:,}")
        self.nodes_stat.setText(f"Graph Nodes: {nodes:,}")
        self.updated_stat.setText(f"Last Updated: {last_updated or 'Unknown'}")
        self.stats_frame.setVisible(True)

    def hide_stats(self):
        """Hide the statistics display."""
        self.stats_frame.setVisible(False)

    def set_push_status(self, status: str, success: bool = None):
        """Set the push status display."""
        self.push_status_label.setText(f"Status: {status}")
        if success is True:
            self.push_status_label.setStyleSheet("color: green;")
        elif success is False:
            self.push_status_label.setStyleSheet("color: red;")
        else:
            self.push_status_label.setStyleSheet("color: gray;")

    def set_pull_status(self, status: str, success: bool = None):
        """Set the pull status display."""
        self.pull_status_label.setText(status)
        if success is True:
            self.pull_status_label.setStyleSheet("color: green;")
        elif success is False:
            self.pull_status_label.setStyleSheet("color: red;")
        else:
            self.pull_status_label.setStyleSheet("color: gray;")

    def populate_conflicts(self, conflicts: list):
        """Populate the conflict resolution table."""
        self.conflict_table.setRowCount(0)

        # Sort conflicts: CONFLICT first, then by address
        sorted_conflicts = sorted(
            conflicts,
            key=lambda x: (x.action != ConflictAction.CONFLICT, x.address)
        )

        for conflict in sorted_conflicts:
            self.add_conflict_row(conflict)

    def add_conflict_row(self, conflict: ConflictEntry):
        """Add a single conflict entry to the table."""
        row = self.conflict_table.rowCount()
        self.conflict_table.insertRow(row)

        # Checkbox for selection
        checkbox = QCheckBox()
        checkbox.setChecked(conflict.selected)
        checkbox_widget = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_widget)
        checkbox_layout.addWidget(checkbox)
        checkbox_layout.setAlignment(Qt.AlignCenter)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.conflict_table.setCellWidget(row, 0, checkbox_widget)

        # Address
        addr_item = QTableWidgetItem(f"0x{conflict.address:x}")
        addr_item.setFlags(addr_item.flags() & ~Qt.ItemIsEditable)
        addr_item.setData(Qt.UserRole, conflict.address)
        self.conflict_table.setItem(row, 1, addr_item)

        # Local Name
        local_item = QTableWidgetItem(conflict.local_name or "<none>")
        local_item.setFlags(local_item.flags() & ~Qt.ItemIsEditable)
        if conflict.local_name is None:
            local_item.setForeground(Qt.gray)
        self.conflict_table.setItem(row, 2, local_item)

        # Remote Name
        remote_item = QTableWidgetItem(conflict.remote_name or "<none>")
        remote_item.setFlags(remote_item.flags() & ~Qt.ItemIsEditable)
        if conflict.remote_name is None:
            remote_item.setForeground(Qt.gray)
        self.conflict_table.setItem(row, 3, remote_item)

        # Action type
        action_item = QTableWidgetItem(conflict.action.value.upper())
        action_item.setFlags(action_item.flags() & ~Qt.ItemIsEditable)

        # Color-code by action type
        if conflict.action == ConflictAction.CONFLICT:
            action_item.setForeground(Qt.red)
        elif conflict.action == ConflictAction.NEW:
            action_item.setForeground(Qt.darkGreen)
        elif conflict.action == ConflictAction.SAME:
            action_item.setForeground(Qt.darkGray)

        self.conflict_table.setItem(row, 4, action_item)

        # Store conflict reference for later retrieval
        addr_item.setData(Qt.UserRole + 1, conflict)

    def get_selected_addresses(self) -> list:
        """Get list of addresses for selected conflict entries."""
        selected = []
        for row in range(self.conflict_table.rowCount()):
            checkbox_widget = self.conflict_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    addr_item = self.conflict_table.item(row, 1)
                    if addr_item:
                        address = addr_item.data(Qt.UserRole)
                        selected.append(address)
        return selected

    def get_selected_conflicts(self) -> list:
        """Get list of ConflictEntry objects for selected rows."""
        selected = []
        for row in range(self.conflict_table.rowCount()):
            checkbox_widget = self.conflict_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    addr_item = self.conflict_table.item(row, 1)
                    if addr_item:
                        conflict = addr_item.data(Qt.UserRole + 1)
                        if conflict:
                            selected.append(conflict)
        return selected

    def clear_conflicts(self):
        """Clear the conflict resolution table."""
        self.conflict_table.setRowCount(0)
        self.pull_status_label.setText("")

    def set_buttons_enabled(self, enabled: bool):
        """Enable or disable all action buttons."""
        self.query_button.setEnabled(enabled)
        self.push_button.setEnabled(enabled)
        self.pull_preview_button.setEnabled(enabled)
        self.apply_button.setEnabled(enabled)

    def set_apply_button_text(self, text: str):
        """Update the apply button text (e.g., to show 'Stop' during operation)."""
        self.apply_button.setText(text)

    def set_pull_button_text(self, text: str):
        """Update the pull button text (e.g., to show 'Stop' during operation)."""
        self.pull_preview_button.setText(text)

    # === Private slot handlers ===

    def on_push_clicked(self):
        """Handle push button click."""
        scope = PushScope.FULL_BINARY.value if self.full_binary_radio.isChecked() else PushScope.CURRENT_FUNCTION.value
        push_symbols = self.push_symbols_check.isChecked()
        push_graph = self.push_graph_check.isChecked()

        if not push_symbols and not push_graph:
            self.set_push_status("Select at least one data type", success=False)
            return

        self.push_requested.emit(scope, push_symbols, push_graph)

    def on_apply_clicked(self):
        """Handle apply selected button click."""
        selected = self.get_selected_addresses()
        if selected:
            self.apply_selected_requested.emit(selected)
        else:
            self.set_pull_status("No items selected", success=False)

    def on_select_all(self):
        """Select all items in the conflict table."""
        for row in range(self.conflict_table.rowCount()):
            checkbox_widget = self.conflict_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.setChecked(True)

    def on_deselect_all(self):
        """Deselect all items in the conflict table."""
        for row in range(self.conflict_table.rowCount()):
            checkbox_widget = self.conflict_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.setChecked(False)

    def on_invert_selection(self):
        """Invert the selection in the conflict table."""
        for row in range(self.conflict_table.rowCount()):
            checkbox_widget = self.conflict_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.setChecked(not checkbox.isChecked())
