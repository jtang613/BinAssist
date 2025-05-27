"""
Feedback service for RLHF data collection.
"""

from typing import Optional, Dict, Any
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime

from .base_service import BaseService, ServiceError


@dataclass
class FeedbackEntry:
    """
    Represents a feedback entry.
    
    Attributes:
        model_name: Name of the model
        prompt_context: The prompt that was sent
        system_context: System context used
        response: The model response
        feedback: Feedback value (1 for positive, 0 for negative)
        timestamp: When the feedback was recorded
        metadata: Additional metadata
    """
    model_name: str
    prompt_context: str
    system_context: str
    response: str
    feedback: int
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class FeedbackService(BaseService):
    """
    Service for collecting and managing RLHF feedback data.
    
    This service handles the storage and retrieval of feedback
    for model responses to enable reinforcement learning.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the feedback service.
        
        Args:
            db_path: Path to the SQLite database
        """
        super().__init__("feedback_service")
        self.db_path = db_path
        self._lock = threading.Lock()
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize the SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        prompt_context TEXT NOT NULL,
                        system_context TEXT NOT NULL,
                        response TEXT NOT NULL,
                        feedback INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        metadata TEXT
                    )
                ''')
                
                # Create index on timestamp for faster queries
                c.execute('''
                    CREATE INDEX IF NOT EXISTS idx_feedback_timestamp 
                    ON feedback(timestamp)
                ''')
                
                conn.commit()
                
        except Exception as e:
            self.handle_error(e, "database initialization")
            raise ServiceError(f"Failed to initialize feedback database: {e}")
    
    def store_feedback(self, entry: FeedbackEntry) -> None:
        """
        Store a feedback entry.
        
        Args:
            entry: The feedback entry to store
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    c = conn.cursor()
                    
                    metadata_json = None
                    if entry.metadata:
                        import json
                        metadata_json = json.dumps(entry.metadata)
                    
                    c.execute('''
                        INSERT INTO feedback 
                        (model_name, prompt_context, system_context, response, 
                         feedback, timestamp, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        entry.model_name,
                        entry.prompt_context,
                        entry.system_context,
                        entry.response,
                        entry.feedback,
                        entry.timestamp.isoformat(),
                        metadata_json
                    ))
                    
                    conn.commit()
                    
        except Exception as e:
            self.handle_error(e, "feedback storage")
            raise ServiceError(f"Failed to store feedback: {e}")
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get feedback statistics.
        
        Returns:
            Dictionary with feedback statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                
                # Total count
                c.execute("SELECT COUNT(*) FROM feedback")
                total_count = c.fetchone()[0]
                
                # Positive/negative counts
                c.execute("SELECT feedback, COUNT(*) FROM feedback GROUP BY feedback")
                feedback_counts = dict(c.fetchall())
                
                # Model breakdown
                c.execute("""
                    SELECT model_name, COUNT(*) as count,
                           AVG(CAST(feedback as FLOAT)) as avg_feedback
                    FROM feedback 
                    GROUP BY model_name
                """)
                model_stats = [
                    {
                        "model": row[0],
                        "count": row[1],
                        "avg_feedback": row[2]
                    }
                    for row in c.fetchall()
                ]
                
                return {
                    "total_entries": total_count,
                    "positive_feedback": feedback_counts.get(1, 0),
                    "negative_feedback": feedback_counts.get(0, 0),
                    "model_breakdown": model_stats
                }
                
        except Exception as e:
            self.handle_error(e, "feedback statistics")
            return {
                "total_entries": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "model_breakdown": []
            }
    
    def export_feedback_data(self, output_path: str, 
                           model_name: Optional[str] = None) -> None:
        """
        Export feedback data to a file.
        
        Args:
            output_path: Path to save the exported data
            model_name: Optional model name filter
        """
        try:
            import json
            
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                
                query = """
                    SELECT model_name, prompt_context, system_context, 
                           response, feedback, timestamp, metadata
                    FROM feedback
                """
                params = []
                
                if model_name:
                    query += " WHERE model_name = ?"
                    params.append(model_name)
                
                query += " ORDER BY timestamp"
                
                c.execute(query, params)
                
                data = []
                for row in c.fetchall():
                    metadata = None
                    if row[6]:
                        try:
                            metadata = json.loads(row[6])
                        except:
                            pass
                    
                    data.append({
                        "model_name": row[0],
                        "prompt_context": row[1],
                        "system_context": row[2],
                        "response": row[3],
                        "feedback": row[4],
                        "timestamp": row[5],
                        "metadata": metadata
                    })
                
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
        except Exception as e:
            self.handle_error(e, "feedback export")
            raise ServiceError(f"Failed to export feedback data: {e}")
    
    def clear_feedback_data(self, model_name: Optional[str] = None) -> int:
        """
        Clear feedback data.
        
        Args:
            model_name: Optional model name filter
            
        Returns:
            Number of entries deleted
        """
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    c = conn.cursor()
                    
                    if model_name:
                        c.execute("DELETE FROM feedback WHERE model_name = ?", (model_name,))
                    else:
                        c.execute("DELETE FROM feedback")
                    
                    deleted_count = c.rowcount
                    conn.commit()
                    
                    return deleted_count
                    
        except Exception as e:
            self.handle_error(e, "feedback clearing")
            raise ServiceError(f"Failed to clear feedback data: {e}")