from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from shwizard.storage.database import Database
from shwizard.utils.logger import get_logger

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
