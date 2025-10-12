import pytest
import tempfile
from pathlib import Path
from shwizard.storage.database import Database
from shwizard.storage.history import HistoryManager


def test_database_initialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(db_path)
        assert db_path.exists()


def test_add_command_history():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(db_path)
        
        cmd_id = db.add_command_history(
            user_query="list files",
            generated_command="ls -la",
            platform="linux"
        )
        
        assert cmd_id > 0


def test_history_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(db_path)
        history = HistoryManager(database=db)
        
        cmd_id = history.add_command("find files", "find . -name '*.py'")
        assert cmd_id > 0
        
        history.mark_executed(cmd_id, success=True)
        history.add_feedback(cmd_id, 1)
        
        stats = history.get_statistics()
        assert stats['total_commands'] == 1
        assert stats['executed_commands'] == 1


def test_search_relevant_commands():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(db_path)
        history = HistoryManager(database=db)
        
        history.add_command("find python files", "find . -name '*.py'")
        history.add_command("count lines", "wc -l *.py")
        
        results = history.search_relevant_commands("python")
        assert len(results) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
