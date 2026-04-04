#!/usr/bin/env python3

"""
Migration: Add timeout field to llm_providers table.
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


def migrate_add_provider_timeout(db_path: str):
    """
    Add timeout field to llm_providers table.
    Default is 90 seconds for provider operations.
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(llm_providers)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'timeout' not in columns:
            cursor.execute('''
                ALTER TABLE llm_providers
                ADD COLUMN timeout INTEGER DEFAULT 90
            ''')
            conn.commit()
            log.log_info("Added timeout column to llm_providers")
        else:
            log.log_debug("timeout column already exists, skipping migration")

    except Exception as e:
        conn.rollback()
        log.log_error(f"provider timeout migration failed: {e}")
        raise
    finally:
        conn.close()
