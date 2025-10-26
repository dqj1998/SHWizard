import sqlite3
import json
from datetime import datetime
from pathlib import Path, PurePath
from typing import List, Optional, Dict, Any, Tuple
from shwizard.utils.platform_utils import get_data_directory
from shwizard.utils.logger import get_logger

logger = get_logger(__name__)


class Database:
    def __init__(self, db_path: Optional[Path] = None):
        if isinstance(db_path, str) and db_path == ":memory:":
            self.db_path = db_path
            self.in_memory = True
            if self.in_memory:
                self._init_database()
        elif db_path is None:
            db_path = get_data_directory() / "history.db"
            self.db_path: Path = db_path
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_database()
            return
        self.in_memory = False
        db_path = get_data_directory() / "history.db"
        self.db_path: Path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS command_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_query TEXT NOT NULL,
                    generated_command TEXT NOT NULL,
                    executed BOOLEAN DEFAULT FALSE,
                    execution_result TEXT,
                    risk_level TEXT,
                    user_feedback INTEGER DEFAULT 0,
                    platform TEXT,
                    working_directory TEXT,
                    context_data TEXT,
                    execution_timestamps TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_command_timestamp 
                ON command_history(timestamp DESC)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_command_executed 
                ON command_history(executed)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_command_feedback 
                ON command_history(user_feedback)
            """)
            
            conn.commit()
            logger.debug(f"Initialized database at {self.db_path}")
    
    def add_command_history(
        self,
        user_query: str,
        generated_command: str,
        executed: bool = False,
        execution_result: Optional[str] = None,
        risk_level: Optional[str] = None,
        platform: Optional[str] = None,
        working_directory: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> int:
        context_json = json.dumps(context_data) if context_data else None
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if identical query+command already exists
            cursor = conn.execute("""
                SELECT id FROM command_history 
                WHERE user_query = ? AND generated_command = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (user_query, generated_command))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record instead of creating new one
                command_id = existing[0]
                
                conn.execute("""
                    UPDATE command_history 
                    SET timestamp = CURRENT_TIMESTAMP,
                        platform = ?,
                        working_directory = ?,
                        context_data = ?
                    WHERE id = ?
                """, (platform, working_directory, context_json, command_id))
                conn.commit()
                return command_id
            else:
                # Create new record (execution_timestamps will be added when executed)
                cursor = conn.execute("""
                    INSERT INTO command_history 
                    (user_query, generated_command, executed, execution_result, 
                     risk_level, platform, working_directory, context_data, execution_timestamps)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_query, generated_command, executed, execution_result,
                      risk_level, platform, working_directory, context_json, None))
                conn.commit()
                return cursor.lastrowid or 0
    
    def update_command_execution(
        self,
        command_id: int,
        executed: bool = True,
        execution_result: Optional[str] = None,
        user_feedback: int = 0
    ):
        with sqlite3.connect(self.db_path) as conn:
            # Get current execution_timestamps
            cursor = conn.execute("""
                SELECT execution_timestamps FROM command_history WHERE id = ?
            """, (command_id,))
            row = cursor.fetchone()
            
            timestamps = []
            if row and row[0]:
                try:
                    timestamps = json.loads(row[0])
                except (json.JSONDecodeError, TypeError):
                    timestamps = []
            
            # Add current execution timestamp
            current_time = datetime.now().isoformat()
            timestamps.append(current_time)
            
            conn.execute("""
                UPDATE command_history 
                SET executed = ?, execution_result = ?, user_feedback = ?,
                    execution_timestamps = ?, timestamp = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (executed, execution_result, user_feedback, 
                  json.dumps(timestamps), command_id))
            conn.commit()
    
    def get_command_history(
        self,
        limit: int = 100,
        executed_only: bool = False,
        min_feedback: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM command_history WHERE 1=1"
        params = []
        
        if executed_only:
            query += " AND executed = ?"
            params.append(True)
        
        if min_feedback is not None:
            query += " AND user_feedback >= ?"
            params.append(min_feedback)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                result = dict(row)
                if result.get("context_data"):
                    result["context_data"] = json.loads(result["context_data"])
                if result.get("execution_timestamps"):
                    result["execution_timestamps"] = json.loads(result["execution_timestamps"])
                    result["execution_count"] = len(result["execution_timestamps"])
                else:
                    result["execution_timestamps"] = []
                    # For old records without execution_timestamps, show 1 if executed, 0 otherwise
                    result["execution_count"] = 1 if result.get("executed") else 0
                results.append(result)
            
            return results
    
    def search_similar_commands(
        self,
        query: str,
        limit: int = 5,
        executed_only: bool = True
    ) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            sql = """
                SELECT * FROM command_history 
                WHERE (user_query LIKE ? OR generated_command LIKE ?)
            """
            params: List[Any] = [f"%{query}%", f"%{query}%"]
            
            if executed_only:
                sql += " AND executed = ?"
                params.append(1)
            
            sql += " ORDER BY timestamp DESC, user_feedback DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                result = dict(row)
                if result.get("context_data"):
                    result["context_data"] = json.loads(result["context_data"])
                if result.get("execution_timestamps"):
                    result["execution_timestamps"] = json.loads(result["execution_timestamps"])
                    result["execution_count"] = len(result["execution_timestamps"])
                else:
                    result["execution_timestamps"] = []
                    # For old records without execution_timestamps, show 1 if executed, 0 otherwise
                    result["execution_count"] = 1 if result.get("executed") else 0
                results.append(result)
            
            return results
    
    def get_command_statistics(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_commands,
                    SUM(CASE WHEN executed = 1 THEN 1 ELSE 0 END) as executed_commands,
                    SUM(CASE WHEN user_feedback > 0 THEN 1 ELSE 0 END) as positive_feedback
                FROM command_history
            """)
            row = cursor.fetchone()
            
            return {
                "total_commands": row[0] or 0,
                "executed_commands": row[1] or 0,
                "positive_feedback": row[2] or 0
            }
    
    def set_preference(self, key: str, value: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_preferences (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (key, value))
            conn.commit()
    
    def get_preference(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM user_preferences WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            return row[0] if row else default
    
    def cleanup_old_entries(self, max_entries: int = 10000):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                DELETE FROM command_history 
                WHERE id NOT IN (
                    SELECT id FROM command_history 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                )
            """, (max_entries,))
            conn.commit()
            logger.info(f"Cleaned up old entries, keeping latest {max_entries}")
