from binaryninja import *
from binaryninjaui import Sidebar

from .src.settings import BinAssistSettings
from .src.plugin import BinAssistWidgetType

# Initialize settings
BinAssistSettings()

# Initialize and register the BinAssist widget
binAssistWidget = BinAssistWidgetType()
Sidebar.addSidebarWidgetType(binAssistWidget)
