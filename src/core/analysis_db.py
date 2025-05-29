"""
Analysis Database Module for BinAssist

Provides SQLite-based storage for function analysis results and program contexts.
Based on the GhidrAssist AnalysisDB implementation.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, NamedTuple
from binaryninja import log
from .settings import get_settings_manager


class Analysis(NamedTuple):
    """Container for analysis results."""
    query: str
    response: str
    timestamp: datetime


class AnalysisDB:
    """
    SQLite database for storing and retrieving function analysis results.
    
    Schema:
    - BNAnalysis: Stores analysis queries and responses keyed by program_hash + function_address
    - BNContext: Stores program-specific system contexts
    """

    def __init__(self):
        """Initialize the analysis database with default or configured path."""
        try:
            settings = get_settings_manager()
            db_path = settings.get_string('analysis_db_path', 'binassist_analysis.db')
            
            # Ensure directory exists
            db_dir = os.path.dirname(os.path.abspath(db_path))
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                log.log_info(f"[BinAssist] Created analysis database directory: {db_dir}")
            
            self.db_path = db_path
            self.connection = sqlite3.connect(db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            
            log.log_info(f"[BinAssist] Analysis database initialized: {db_path}")
            self._create_tables()
            
        except Exception as e:
            log.log_error(f"[BinAssist] Failed to initialize analysis database: {e}")
            raise

    def _create_tables(self):
        """Create the database tables if they don't exist."""
        try:
            # Create BNAnalysis table (equivalent to GHAnalysis)
            create_analysis_table = """
                CREATE TABLE IF NOT EXISTS BNAnalysis (
                    program_hash TEXT NOT NULL,
                    function_address TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (program_hash, function_address)
                )
            """
            
            # Create BNContext table (equivalent to GHContext)
            create_context_table = """
                CREATE TABLE IF NOT EXISTS BNContext (
                    program_hash TEXT PRIMARY KEY,
                    system_context TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            
            with self.connection:
                self.connection.execute(create_analysis_table)
                self.connection.execute(create_context_table)
                
            log.log_debug("[BinAssist] Analysis database tables created successfully")
            
        except sqlite3.Error as e:
            log.log_error(f"[BinAssist] Failed to create analysis database tables: {e}")
            raise

    def upsert_analysis(self, program_hash: str, function_address: str, query: str, response: str) -> bool:
        """
        Store or update analysis results for a program/function combination.
        
        Args:
            program_hash: SHA256 hash of the program
            function_address: Hexadecimal address of the function
            query: The analysis query that was sent
            response: The analysis response received
            
        Returns:
            True if successful, False otherwise
        """
        if not all([program_hash, function_address, query, response]):
            log.log_error("[BinAssist] Cannot upsert analysis: missing required parameters")
            return False
            
        upsert_sql = """
            INSERT INTO BNAnalysis (program_hash, function_address, query, response) 
            VALUES (?, ?, ?, ?) 
            ON CONFLICT(program_hash, function_address) 
            DO UPDATE SET 
                query = excluded.query, 
                response = excluded.response, 
                timestamp = CURRENT_TIMESTAMP
        """
        
        try:
            with self.connection:
                self.connection.execute(upsert_sql, (program_hash, function_address, query, response))
                
            log.log_debug(f"[BinAssist] Stored analysis for {program_hash[:8]}...@{function_address}")
            return True
            
        except sqlite3.Error as e:
            log.log_error(f"[BinAssist] Failed to store analysis: {e}")
            return False

    def get_analysis(self, program_hash: str, function_address: str) -> Optional[Analysis]:
        """
        Retrieve analysis results for a program/function combination.
        
        Args:
            program_hash: SHA256 hash of the program
            function_address: Hexadecimal address of the function
            
        Returns:
            Analysis object if found, None otherwise
        """
        if not all([program_hash, function_address]):
            log.log_error("[BinAssist] Cannot get analysis: missing required parameters")
            return None
            
        select_sql = """
            SELECT query, response, timestamp 
            FROM BNAnalysis 
            WHERE program_hash = ? AND function_address = ?
        """
        
        try:
            cursor = self.connection.execute(select_sql, (program_hash, function_address))
            row = cursor.fetchone()
            
            if row:
                # Parse timestamp string back to datetime
                timestamp_str = row['timestamp']
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                
                log.log_debug(f"[BinAssist] Retrieved cached analysis for {program_hash[:8]}...@{function_address}")
                return Analysis(
                    query=row['query'],
                    response=row['response'],
                    timestamp=timestamp
                )
            else:
                log.log_debug(f"[BinAssist] No cached analysis found for {program_hash[:8]}...@{function_address}")
                return None
                
        except sqlite3.Error as e:
            log.log_error(f"[BinAssist] Failed to retrieve analysis: {e}")
            return None
        except (ValueError, TypeError) as e:
            log.log_error(f"[BinAssist] Failed to parse analysis timestamp: {e}")
            return None

    def delete_analysis(self, program_hash: str, function_address: str) -> bool:
        """
        Delete analysis results for a program/function combination.
        
        Args:
            program_hash: SHA256 hash of the program
            function_address: Hexadecimal address of the function
            
        Returns:
            True if an entry was deleted, False otherwise
        """
        if not all([program_hash, function_address]):
            log.log_error("[BinAssist] Cannot delete analysis: missing required parameters")
            return False
            
        delete_sql = "DELETE FROM BNAnalysis WHERE program_hash = ? AND function_address = ?"
        
        try:
            with self.connection:
                cursor = self.connection.execute(delete_sql, (program_hash, function_address))
                rows_affected = cursor.rowcount
                
            log.log_info(f"[BinAssist] Deleted analysis for {program_hash[:8]}...@{function_address} (rows affected: {rows_affected})")
            return rows_affected > 0
            
        except sqlite3.Error as e:
            log.log_error(f"[BinAssist] Failed to delete analysis: {e}")
            return False

    def upsert_context(self, program_hash: str, context: str) -> bool:
        """
        Store or update program-specific system context.
        
        Args:
            program_hash: SHA256 hash of the program
            context: System context string
            
        Returns:
            True if successful, False otherwise
        """
        if not program_hash:
            log.log_error("[BinAssist] Cannot upsert context: missing program hash")
            return False
            
        if context is None:
            # If context is None, delete the entry to revert to default
            return self.delete_context(program_hash)
            
        upsert_sql = """
            INSERT INTO BNContext (program_hash, system_context) 
            VALUES (?, ?) 
            ON CONFLICT(program_hash) 
            DO UPDATE SET 
                system_context = excluded.system_context, 
                timestamp = CURRENT_TIMESTAMP
        """
        
        try:
            with self.connection:
                self.connection.execute(upsert_sql, (program_hash, context))
                
            log.log_debug(f"[BinAssist] Stored context for {program_hash[:8]}... ({len(context)} chars)")
            return True
            
        except sqlite3.Error as e:
            log.log_error(f"[BinAssist] Failed to store context: {e}")
            return False

    def get_context(self, program_hash: str) -> Optional[str]:
        """
        Retrieve program-specific system context.
        
        Args:
            program_hash: SHA256 hash of the program
            
        Returns:
            Context string if found, None otherwise
        """
        if not program_hash:
            log.log_error("[BinAssist] Cannot get context: missing program hash")
            return None
            
        select_sql = "SELECT system_context FROM BNContext WHERE program_hash = ?"
        
        try:
            cursor = self.connection.execute(select_sql, (program_hash,))
            row = cursor.fetchone()
            
            if row:
                context = row['system_context']
                log.log_debug(f"[BinAssist] Retrieved context for {program_hash[:8]}... ({len(context)} chars)")
                return context
            else:
                log.log_debug(f"[BinAssist] No context found for {program_hash[:8]}...")
                return None
                
        except sqlite3.Error as e:
            log.log_error(f"[BinAssist] Failed to retrieve context: {e}")
            return None

    def delete_context(self, program_hash: str) -> bool:
        """
        Delete program-specific system context.
        
        Args:
            program_hash: SHA256 hash of the program
            
        Returns:
            True if an entry was deleted, False otherwise
        """
        if not program_hash:
            log.log_error("[BinAssist] Cannot delete context: missing program hash")
            return False
            
        delete_sql = "DELETE FROM BNContext WHERE program_hash = ?"
        
        try:
            with self.connection:
                cursor = self.connection.execute(delete_sql, (program_hash,))
                rows_affected = cursor.rowcount
                
            log.log_info(f"[BinAssist] Deleted context for {program_hash[:8]}... (rows affected: {rows_affected})")
            return rows_affected > 0
            
        except sqlite3.Error as e:
            log.log_error(f"[BinAssist] Failed to delete context: {e}")
            return False

    def get_analysis_count(self) -> int:
        """
        Get the total number of cached analysis entries.
        
        Returns:
            Number of analysis entries in the database
        """
        try:
            cursor = self.connection.execute("SELECT COUNT(*) FROM BNAnalysis")
            return cursor.fetchone()[0]
        except sqlite3.Error as e:
            log.log_error(f"[BinAssist] Failed to get analysis count: {e}")
            return 0

    def get_context_count(self) -> int:
        """
        Get the total number of cached context entries.
        
        Returns:
            Number of context entries in the database
        """
        try:
            cursor = self.connection.execute("SELECT COUNT(*) FROM BNContext")
            return cursor.fetchone()[0]
        except sqlite3.Error as e:
            log.log_error(f"[BinAssist] Failed to get context count: {e}")
            return 0

    def clear_all_analysis(self) -> bool:
        """
        Clear all analysis entries from the database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.connection:
                cursor = self.connection.execute("DELETE FROM BNAnalysis")
                rows_affected = cursor.rowcount
                
            log.log_info(f"[BinAssist] Cleared all analysis entries (rows affected: {rows_affected})")
            return True
            
        except sqlite3.Error as e:
            log.log_error(f"[BinAssist] Failed to clear all analysis: {e}")
            return False

    def clear_all_contexts(self) -> bool:
        """
        Clear all context entries from the database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.connection:
                cursor = self.connection.execute("DELETE FROM BNContext")
                rows_affected = cursor.rowcount
                
            log.log_info(f"[BinAssist] Cleared all context entries (rows affected: {rows_affected})")
            return True
            
        except sqlite3.Error as e:
            log.log_error(f"[BinAssist] Failed to clear all contexts: {e}")
            return False

    def close(self):
        """Close the database connection."""
        try:
            if self.connection:
                self.connection.close()
                log.log_debug("[BinAssist] Analysis database connection closed")
        except sqlite3.Error as e:
            log.log_error(f"[BinAssist] Failed to close analysis database: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()