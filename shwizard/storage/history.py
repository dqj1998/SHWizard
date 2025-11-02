from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import shutil
from shwizard.storage.database import Database
from shwizard.utils.logger import get_logger
from shwizard.utils.platform_utils import get_home_directory

logger = get_logger(__name__)


class HistoryManager:
    def __init__(self, database: Optional[Database] = None):
        self.db = database or Database()
    
    def add_command(
        self,
        user_query: str,
        generated_command: str,
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        return self.db.add_command_history(
            user_query=user_query,
            generated_command=generated_command,
            executed=False,
            platform=context.get("os") if context else None,
            working_directory=context.get("cwd") if context else None,
            context_data=context
        )
    
    def mark_executed(
        self,
        command_id: int,
        success: bool = True,
        result: Optional[str] = None
    ):
        self.db.update_command_execution(
            command_id=command_id,
            executed=True,
            execution_result=result,
            user_feedback=success and 1 or 0
        )
    
    def add_feedback(self, command_id: int, feedback: int):
        if feedback >= 0:
            self.db.update_command_execution(
                command_id=command_id,
                executed=True,
                user_feedback=feedback
            )
    
    def search_relevant_commands(
        self,
        query: str,
        limit: int = 5,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        similar_commands = self.db.search_similar_commands(
            query=query,
            limit=limit * 2,
            executed_only=True
        )
        
        if not similar_commands:
            return []
        
        scored_commands = []
        now = datetime.now()
        
        for cmd_data in similar_commands:
            score = self._calculate_priority_score(cmd_data, now, context)
            scored_commands.append((score, cmd_data["generated_command"]))
        
        scored_commands.sort(reverse=True, key=lambda x: x[0])
        
        unique_commands = []
        seen = set()
        for _, command in scored_commands:
            if command not in seen:
                unique_commands.append(command)
                seen.add(command)
                if len(unique_commands) >= limit:
                    break
        
        return unique_commands
    
    def search_by_keywords(
        self,
        keywords: List[str],
        limit: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search command history by keywords and return results ranked by keyword match count.
        
        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results to return
            context: Optional context for additional scoring
        
        Returns:
            List of command history entries with keyword_match_count field
        """
        return self.db.search_by_keywords(
            keywords=keywords,
            limit=limit,
            executed_only=True
        )
    
    def _calculate_priority_score(
        self,
        cmd_data: Dict[str, Any],
        now: datetime,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        frequency_weight = 0.4
        recency_weight = 0.3
        success_weight = 0.3
        
        frequency_score = min(1.0, cmd_data.get("id", 0) / 100.0)
        
        try:
            cmd_time = datetime.fromisoformat(cmd_data["timestamp"])
            days_old = (now - cmd_time).days
            recency_score = max(0, 1.0 - (days_old / 365.0))
        except:
            recency_score = 0.0
        
        feedback = cmd_data.get("user_feedback", 0)
        if feedback > 0:
            success_score = 1.0
        else:
            success_score = 0.5 if cmd_data.get("executed") else 0.3
        
        total_score = (
            frequency_weight * frequency_score +
            recency_weight * recency_score +
            success_weight * success_score
        )
        
        if context and cmd_data.get("working_directory"):
            if cmd_data["working_directory"] == context.get("cwd"):
                total_score *= 1.2
        
        return total_score
    
    def get_recent_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.db.get_command_history(limit=limit)
    
    def get_statistics(self) -> Dict[str, Any]:
        return self.db.get_command_statistics()
    
    def set_preferred_language(self, lang: str):
        """Persist user's preferred language code (ISO 639-1) in user preferences."""
        try:
            self.db.set_preference("preferred_language", lang)
        except Exception as e:
            logger.warning(f"Failed to save preferred language: {e}")

    def get_preferred_language(self, default: Optional[str] = "en") -> str:
        """Load user's preferred language code from user preferences."""
        try:
            return self.db.get_preference("preferred_language", default) or default
        except Exception as e:
            logger.warning(f"Failed to load preferred language: {e}")
            return default

    def cleanup(self, max_entries: int = 10000):
        self.db.cleanup_old_entries(max_entries)
    
    def export_database(self, export_path: Optional[str] = None) -> Path:
        """
        Export the history database to a backup file.
        
        Args:
            export_path: Optional path where to export the database.
                        If not provided, exports to ~/shwizard_backup.db
        
        Returns:
            Path to the exported database file
        
        Raises:
            FileNotFoundError: If the source database doesn't exist
            PermissionError: If unable to write to the destination
            ValueError: If trying to export an in-memory database
        """
        # Get the source database path
        source_path = self.db.db_path
        
        # Check if source is in-memory (shouldn't be for export)
        if isinstance(source_path, str) and source_path == ":memory:":
            raise ValueError("Cannot export in-memory database")
        
        if self.db.in_memory:
            raise ValueError("Cannot export in-memory database")
        
        if export_path:
            dest_path = Path(export_path).expanduser().resolve()
        else:
            dest_path = get_home_directory() / "shwizard_backup.db"
        
        if not Path(source_path).exists():
            raise FileNotFoundError(f"Source database not found: {source_path}")
        
        # Create parent directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the database file
        try:
            shutil.copy2(source_path, dest_path)
            logger.info(f"Exported database from {source_path} to {dest_path}")
        except Exception as e:
            logger.error(f"Failed to export database: {e}")
            raise
        
        return dest_path
    
    def import_database(self, import_path: str, merge: bool = True) -> None:
        """
        Import a history database from a backup file.
        
        Args:
            import_path: Path to the database file to import
            merge: If True, merge with existing database. If False, replace entirely.
        
        Raises:
            FileNotFoundError: If the import file doesn't exist
            ValueError: If the import file is not a valid SQLite database or trying to import into in-memory database
        """
        source_path = Path(import_path).expanduser().resolve()
        
        if not source_path.exists():
            raise FileNotFoundError(f"Import file not found: {source_path}")
        
        # Get the destination database path
        dest_path = self.db.db_path
        
        if isinstance(dest_path, str) and dest_path == ":memory:":
            raise ValueError("Cannot import into in-memory database")
        
        if self.db.in_memory:
            raise ValueError("Cannot import into in-memory database")
        
        if merge:
            # Merge: Import all records from source into destination
            try:
                import sqlite3
                
                # Connect to both databases
                with sqlite3.connect(dest_path) as dest_conn:
                    # Attach source database
                    dest_conn.execute(f"ATTACH DATABASE '{source_path}' AS import_db")
                    
                    # Import command_history, avoiding duplicates based on query+command combination
                    # Use NOT EXISTS to skip records that already exist
                    dest_conn.execute("""
                        INSERT INTO command_history 
                        (timestamp, user_query, generated_command, executed, execution_result,
                         risk_level, user_feedback, platform, working_directory, context_data, execution_timestamps)
                        SELECT timestamp, user_query, generated_command, executed, execution_result,
                               risk_level, user_feedback, platform, working_directory, context_data, execution_timestamps
                        FROM import_db.command_history AS import_ch
                        WHERE NOT EXISTS (
                            SELECT 1 FROM command_history AS dest_ch
                            WHERE dest_ch.user_query = import_ch.user_query
                              AND dest_ch.generated_command = import_ch.generated_command
                        )
                    """)
                    
                    # Import user_preferences (update existing, insert new)
                    dest_conn.execute("""
                        INSERT OR REPLACE INTO user_preferences 
                        SELECT * FROM import_db.user_preferences
                    """)
                    
                    dest_conn.commit()
                    dest_conn.execute("DETACH DATABASE import_db")
                    
                logger.info(f"Merged database from {source_path} into {dest_path}")
            except Exception as e:
                logger.error(f"Failed to merge database: {e}")
                raise ValueError(f"Failed to import database (possibly invalid format): {e}")
        else:
            # Replace: Backup current DB and replace with imported one
            try:
                # Create backup of current database
                if Path(dest_path).exists():
                    backup_path = Path(dest_path).parent / f"{Path(dest_path).stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                    shutil.copy2(dest_path, backup_path)
                    logger.info(f"Backed up current database to {backup_path}")
                
                # Replace with imported database
                shutil.copy2(source_path, dest_path)
                logger.info(f"Replaced database with import from {source_path}")
            except Exception as e:
                logger.error(f"Failed to replace database: {e}")
                raise
