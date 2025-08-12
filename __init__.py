#!/usr/bin/env python3

from binaryninjaui import Sidebar
from .src.views.binassist_sidebar import BinAssistSidebarWidgetType

# Register the sidebar widget type with Binary Ninja
Sidebar.addSidebarWidgetType(BinAssistSidebarWidgetType())