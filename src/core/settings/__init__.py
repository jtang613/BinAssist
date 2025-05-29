"""
Core settings module for BinAssist.

This module provides a SQLite-based settings management system that replaces
Binary Ninja's restrictive native settings system.
"""

from .settings_manager import SettingsManager, SettingDefinition, get_settings_manager, reset_settings_manager
from .migration import migrate_from_binary_ninja_settings

__all__ = [
    'SettingsManager',
    'SettingDefinition', 
    'get_settings_manager',
    'reset_settings_manager',
    'migrate_from_binary_ninja_settings'
]