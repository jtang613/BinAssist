#!/usr/bin/env python3

import sqlite3
import os
import json
from typing import Dict, Any

# Setup BinAssist logger
try:
    import binaryninja
    log = binaryninja.log.Logger(0, "BinAssist")
except ImportError:
    # Fallback for testing outside Binary Ninja
    class MockLog:
        @staticmethod
        def log_info(msg): print(f"[BinAssist] INFO: {msg}")
        @staticmethod
        def log_error(msg): print(f"[BinAssist] ERROR: {msg}")
        @staticmethod
        def log_warn(msg): print(f"[BinAssist] WARN: {msg}")
    log = MockLog()


class DatabaseMigrations:
    """Database migration utilities for BinAssist services"""
    
    @staticmethod
    def get_schema_version(db_path: str) -> int:
        """Get current schema version"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if schema_version table exists
            cursor.execute('''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='schema_version'
            ''')
            
            if not cursor.fetchone():
                # No schema version table, assume version 0
                conn.close()
                return 0
            
            # Get current version
            cursor.execute('SELECT version FROM schema_version ORDER BY id DESC LIMIT 1')
            row = cursor.fetchone()
            version = row[0] if row else 0
            
            conn.close()
            return version
            
        except Exception as e:
            log.log_error(f"Failed to get schema version: {e}")
            return 0
    
    @staticmethod
    def set_schema_version(db_path: str, version: int) -> bool:
        """Set current schema version"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create schema_version table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS schema_version (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version INTEGER NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert new version
            cursor.execute('''
                INSERT INTO schema_version (version) VALUES (?)
            ''', (version,))
            
            conn.commit()
            conn.close()
            
            log.log_info(f"Set schema version to {version}")
            return True
            
        except Exception as e:
            log.log_error(f"Failed to set schema version: {e}")
            return False
    
    @staticmethod
    def migrate_analysis_db(db_path: str) -> bool:
        """Apply all migrations to analysis database"""
        try:
            current_version = DatabaseMigrations.get_schema_version(db_path)
            
            migrations = [
                (1, DatabaseMigrations._migration_001_initial_schema),
                (2, DatabaseMigrations._migration_002_add_indexes),
                (3, DatabaseMigrations._migration_003_native_message_storage),
                (4, DatabaseMigrations._migration_004_graphrag_schema),
                (5, DatabaseMigrations._migration_005_graphrag_columns),
                # Add future migrations here
            ]
            
            for version, migration_func in migrations:
                if current_version < version:
                    log.log_info(f"Applying migration {version}")
                    if migration_func(db_path):
                        DatabaseMigrations.set_schema_version(db_path, version)
                    else:
                        log.log_error(f"Migration {version} failed")
                        return False
            
            return True
            
        except Exception as e:
            log.log_error(f"Migration failed: {e}")
            return False
    
    @staticmethod
    def _migration_001_initial_schema(db_path: str) -> bool:
        """Migration 001: Initial schema creation"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # This migration is handled by the service initialization
            # Just mark as applied
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            log.log_error(f"Migration 001 failed: {e}")
            return False
    
    @staticmethod
    def _migration_002_add_indexes(db_path: str) -> bool:
        """Migration 002: Add additional indexes for performance"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Add any additional indexes that might be needed
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bnanalysis_created_at ON BNAnalysis(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bnchathistory_created_at ON BNChatHistory(created_at)')
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            log.log_error(f"Migration 002 failed: {e}")
            return False
    
    @staticmethod
    def _migration_003_native_message_storage(db_path: str) -> bool:
        """Migration 003: Add native message storage for provider-specific formats"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create new native message storage table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS BNChatMessages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    binary_hash TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    message_order INTEGER NOT NULL,
                    
                    -- Provider identity
                    provider_type TEXT NOT NULL,
                    
                    -- Native storage (exact provider format)
                    native_message_data TEXT NOT NULL,
                    
                    -- Normalized fields for UI/search (derived from native data)
                    role TEXT NOT NULL,
                    content_text TEXT,
                    message_type TEXT DEFAULT 'standard',
                    
                    -- Threading and relationships
                    parent_message_id INTEGER,
                    conversation_thread_id TEXT,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Constraints
                    UNIQUE(binary_hash, chat_id, message_order),
                    FOREIGN KEY(parent_message_id) REFERENCES BNChatMessages(id)
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_messages_lookup ON BNChatMessages(binary_hash, chat_id, message_order)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_messages_provider ON BNChatMessages(provider_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_messages_thread ON BNChatMessages(conversation_thread_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_messages_parent ON BNChatMessages(parent_message_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_messages_type ON BNChatMessages(message_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_messages_role ON BNChatMessages(role)')
            
            # Migrate existing chat history from old format (best effort)
            cursor.execute('''
                SELECT binary_hash, chat_id, message_order, role, content, metadata, created_at
                FROM BNChatHistory 
                ORDER BY binary_hash, chat_id, message_order
            ''')
            
            old_messages = cursor.fetchall()
            migrated_count = 0
            
            for old_msg in old_messages:
                binary_hash, chat_id, message_order, role, content, metadata, created_at = old_msg
                
                # Create a basic native message format (generic structure)
                native_message = {
                    "role": role,
                    "content": content
                }
                
                # Determine provider type (default to 'anthropic' for existing messages)
                provider_type = "anthropic"  # Default for existing data
                
                # Determine message type
                message_type = "standard"
                if role == "tool":
                    message_type = "tool_response"
                elif "tool_calls" in content.lower() or "function" in content.lower():
                    message_type = "tool_call"
                
                try:
                    cursor.execute('''
                        INSERT INTO BNChatMessages (
                            binary_hash, chat_id, message_order, provider_type,
                            native_message_data, role, content_text, message_type, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        binary_hash, chat_id, message_order, provider_type,
                        json.dumps(native_message), role, content, message_type, created_at
                    ))
                    migrated_count += 1
                except Exception as e:
                    log.log_warn(f"Failed to migrate message {message_order} in chat {chat_id}: {e}")
            
            # Rename old table to backup
            cursor.execute('ALTER TABLE BNChatHistory RENAME TO BNChatHistory_backup')
            
            conn.commit()
            conn.close()
            
            log.log_info(f"Migration 003 completed: Migrated {migrated_count} messages to native format")
            log.log_info("Old chat history backed up as BNChatHistory_backup")
            return True
            
        except Exception as e:
            log.log_error(f"Migration 003 failed: {e}")
            return False

    @staticmethod
    def _migration_004_graphrag_schema(db_path: str) -> bool:
        """Migration 004: GraphRAG schema (nodes, edges, optional FTS)"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS GraphNodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    binary_hash TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    address INTEGER,
                    name TEXT,
                    raw_code TEXT,
                    llm_summary TEXT,
                    security_flags TEXT,
                    network_apis TEXT,
                    file_io_apis TEXT,
                    ip_addresses TEXT,
                    urls TEXT,
                    file_paths TEXT,
                    domains TEXT,
                    registry_keys TEXT,
                    activity_profile TEXT,
                    risk_level TEXT,
                    is_stale INTEGER DEFAULT 1,
                    user_edited INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(binary_hash, node_type, address)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS GraphEdges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    binary_hash TEXT NOT NULL,
                    source_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    edge_type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(binary_hash, source_id, target_id, edge_type),
                    FOREIGN KEY(source_id) REFERENCES GraphNodes(id),
                    FOREIGN KEY(target_id) REFERENCES GraphNodes(id)
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_graph_nodes_lookup ON GraphNodes(binary_hash, node_type, address)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_graph_nodes_name ON GraphNodes(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON GraphEdges(source_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON GraphEdges(target_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_graph_edges_type ON GraphEdges(edge_type)')

            # Optional FTS5 index for semantic search
            try:
                cursor.execute('''
                    CREATE VIRTUAL TABLE IF NOT EXISTS GraphNodeFTS USING fts5(
                        name,
                        llm_summary,
                        security_flags,
                        content='GraphNodes',
                        content_rowid='id'
                    )
                ''')
                cursor.execute('''
                    CREATE TRIGGER IF NOT EXISTS graph_nodes_ai AFTER INSERT ON GraphNodes BEGIN
                        INSERT INTO GraphNodeFTS(rowid, name, llm_summary, security_flags)
                        VALUES (new.id, new.name, new.llm_summary, new.security_flags);
                    END;
                ''')
                cursor.execute('''
                    CREATE TRIGGER IF NOT EXISTS graph_nodes_ad AFTER DELETE ON GraphNodes BEGIN
                        INSERT INTO GraphNodeFTS(GraphNodeFTS, rowid, name, llm_summary, security_flags)
                        VALUES ('delete', old.id, old.name, old.llm_summary, old.security_flags);
                    END;
                ''')
                cursor.execute('''
                    CREATE TRIGGER IF NOT EXISTS graph_nodes_au AFTER UPDATE ON GraphNodes BEGIN
                        INSERT INTO GraphNodeFTS(GraphNodeFTS, rowid, name, llm_summary, security_flags)
                        VALUES ('delete', old.id, old.name, old.llm_summary, old.security_flags);
                        INSERT INTO GraphNodeFTS(rowid, name, llm_summary, security_flags)
                        VALUES (new.id, new.name, new.llm_summary, new.security_flags);
                    END;
                ''')
            except Exception as e:
                log.log_warn(f"FTS5 not available for GraphNodeFTS: {e}")

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            log.log_error(f"Migration 004 failed: {e}")
            return False

    @staticmethod
    def _migration_005_graphrag_columns(db_path: str) -> bool:
        """Migration 005: Add GraphRAG string reference columns."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            columns = [
                "ALTER TABLE GraphNodes ADD COLUMN ip_addresses TEXT",
                "ALTER TABLE GraphNodes ADD COLUMN urls TEXT",
                "ALTER TABLE GraphNodes ADD COLUMN file_paths TEXT",
                "ALTER TABLE GraphNodes ADD COLUMN domains TEXT",
                "ALTER TABLE GraphNodes ADD COLUMN registry_keys TEXT",
            ]
            for sql in columns:
                try:
                    cursor.execute(sql)
                except Exception:
                    pass

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            log.log_error(f"Migration 005 failed: {e}")
            return False


class DatabaseCleanup:
    """Database cleanup utilities"""
    
    @staticmethod
    def cleanup_expired_data(db_path: str) -> Dict[str, int]:
        """Clean up expired data from database"""
        stats = {"expired_contexts": 0, "old_chat_messages": 0}
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Clean up expired context cache
            cursor.execute('''
                DELETE FROM BNContext 
                WHERE expires_at IS NOT NULL AND expires_at <= CURRENT_TIMESTAMP
            ''')
            stats["expired_contexts"] = cursor.rowcount
            
            # Optionally clean up old chat messages (older than 90 days)
            # Disabled by default - users may want to keep chat history
            # cursor.execute('''
            #     DELETE FROM BNChatHistory 
            #     WHERE created_at < datetime('now', '-90 days')
            # ''')
            # stats["old_chat_messages"] = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if stats["expired_contexts"] > 0:
                log.log_info(f"Cleaned up {stats['expired_contexts']} expired context entries")
            
            return stats
            
        except Exception as e:
            log.log_error(f"Database cleanup failed: {e}")
            return stats
    
    @staticmethod
    def get_database_size_info(db_path: str) -> Dict[str, Any]:
        """Get database size and statistics"""
        info = {
            "file_size_mb": 0,
            "table_counts": {},
            "total_records": 0
        }
        
        try:
            # Get file size
            if os.path.exists(db_path):
                info["file_size_mb"] = os.path.getsize(db_path) / (1024 * 1024)
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get table counts
            tables = ["BNAnalysis", "BNContext", "BNChatHistory", "SystemPrompts"]
            for table in tables:
                try:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    count = cursor.fetchone()[0]
                    info["table_counts"][table] = count
                    info["total_records"] += count
                except:
                    info["table_counts"][table] = 0
            
            conn.close()
            return info
            
        except Exception as e:
            log.log_error(f"Failed to get database info: {e}")
            return info
    
    @staticmethod
    def vacuum_database(db_path: str) -> bool:
        """Optimize database by running VACUUM"""
        try:
            conn = sqlite3.connect(db_path)
            conn.execute('VACUUM')
            conn.close()
            
            log.log_info("Database vacuumed successfully")
            return True
            
        except Exception as e:
            log.log_error(f"Database vacuum failed: {e}")
            return False
