#!/usr/bin/env python3

"""
Migration: Add bypass_proxy field to llm_providers table
"""

import sqlite3

try:
    import binaryninja
    log = binaryninja.log.Logger(0, "BinAssist")
except ImportError:
    class MockLog:
        @staticmethod
        def log_info(msg): print(f"[BinAssist] INFO: {msg}")
        @staticmethod
        def log_error(msg): print(f"[BinAssist] ERROR: {msg}")
        @staticmethod
        def log_warn(msg): print(f"[BinAssist] WARN: {msg}")
        @staticmethod
        def log_debug(msg): print(f"[BinAssist] DEBUG: {msg}")
    log = MockLog()


def migrate_add_bypass_proxy(db_path: str):
    """
    Add bypass_proxy field to llm_providers table.
    Default is False so providers use the system proxy unless explicitly told not to.
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(llm_providers)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'bypass_proxy' not in columns:
            cursor.execute('''
                ALTER TABLE llm_providers
                ADD COLUMN bypass_proxy BOOLEAN DEFAULT 0
            ''')
            conn.commit()
            log.log_info("Added bypass_proxy column to llm_providers")
        else:
            log.log_debug("bypass_proxy column already exists, skipping migration")

    except Exception as e:
        conn.rollback()
        log.log_error(f"bypass_proxy migration failed: {e}")
        raise
    finally:
        conn.close()
