#!/usr/bin/env python3

# Fix pywin32 paths on Windows (required for mcp library)
# Binary Ninja doesn't process .pth files, so we add the paths manually
import sys
import os
if sys.platform == 'win32':
    _site_packages = os.path.join(os.environ.get('APPDATA', ''), 'Binary Ninja', 'python310', 'site-packages')
    _win32_paths = [
        os.path.join(_site_packages, 'win32'),
        os.path.join(_site_packages, 'win32', 'lib'),
        os.path.join(_site_packages, 'Pythonwin'),
    ]
    for _p in _win32_paths:
        if _p not in sys.path and os.path.isdir(_p):
            sys.path.insert(0, _p)

from binaryninjaui import Sidebar
from .src.views.binassist_sidebar import BinAssistSidebarWidgetType

# Register the sidebar widget type with Binary Ninja
Sidebar.addSidebarWidgetType(BinAssistSidebarWidgetType())