"""
Tests for database export/import functionality
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from shwizard.storage.history import HistoryManager
from shwizard.storage.database import Database


def test_export_default_location():
    """Test exporting to default location"""
    hm = HistoryManager()
    
    # Add a test command
    cmd_id = hm.add_command("test query", "test command", {"os": "linux"})
    hm.mark_executed(cmd_id, success=True)
    
    # Export to default location
    export_path = hm.export_database()
    
    assert export_path.exists()
    assert export_path.name == "shwizard_backup.db"
    assert export_path.stat().st_size > 0
    
    # Cleanup
    if export_path.exists():
        export_path.unlink()


def test_export_custom_location():
    """Test exporting to custom location"""
    hm = HistoryManager()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_path = Path(tmpdir) / "custom_backup.db"
        
        # Export to custom location
        export_path = hm.export_database(str(custom_path))
        
        # Use resolve() to normalize both paths for comparison (handles /var vs /private/var on macOS)
        assert export_path.resolve() == custom_path.resolve()
        assert export_path.exists()
        assert export_path.stat().st_size > 0


def test_export_creates_parent_directories():
    """Test that export creates parent directories if needed"""
    hm = HistoryManager()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = Path(tmpdir) / "nested" / "dir" / "backup.db"
        
        # Export to nested location
        export_path = hm.export_database(str(nested_path))
        
        assert export_path.exists()
        assert export_path.parent.exists()


def test_import_merge():
    """Test importing and merging databases"""
    # Create two separate history managers with different databases
    with tempfile.TemporaryDirectory() as tmpdir:
        # Database 1
        db1_path = Path(tmpdir) / "db1.db"
        db1 = Database(db1_path)
        hm1 = HistoryManager(db1)
        
        cmd1_id = hm1.add_command("query1", "command1", {"os": "linux"})
        hm1.mark_executed(cmd1_id, success=True)
        
        cmd2_id = hm1.add_command("query2", "command2", {"os": "linux"})
        hm1.mark_executed(cmd2_id, success=True)
        
        stats1 = hm1.get_statistics()
        assert stats1['total_commands'] == 2
        
        # Database 2
        db2_path = Path(tmpdir) / "db2.db"
        db2 = Database(db2_path)
        hm2 = HistoryManager(db2)
        
        cmd3_id = hm2.add_command("query3", "command3", {"os": "macos"})
        hm2.mark_executed(cmd3_id, success=True)
        
        stats2_before = hm2.get_statistics()
        assert stats2_before['total_commands'] == 1
        
        # Import db1 into db2 (merge)
        hm2.import_database(str(db1_path), merge=True)
        
        stats2_after = hm2.get_statistics()
        # Should have all unique commands from both databases
        assert stats2_after['total_commands'] == 3
        assert stats2_after['executed_commands'] == 3


def test_import_duplicate_handling():
    """Test that import handles duplicate entries correctly"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Database 1
        db1_path = Path(tmpdir) / "db1.db"
        db1 = Database(db1_path)
        hm1 = HistoryManager(db1)
        
        # Add same command twice to db1
        cmd1_id = hm1.add_command("find files", "find . -name '*.txt'", {"os": "linux"})
        hm1.mark_executed(cmd1_id, success=True)
        
        stats1 = hm1.get_statistics()
        assert stats1['total_commands'] == 1
        
        # Database 2
        db2_path = Path(tmpdir) / "db2.db"
        db2 = Database(db2_path)
        hm2 = HistoryManager(db2)
        
        # Add same command to db2
        cmd2_id = hm2.add_command("find files", "find . -name '*.txt'", {"os": "linux"})
        hm2.mark_executed(cmd2_id, success=True)
        
        # Add a different command to db2
        cmd3_id = hm2.add_command("list files", "ls -la", {"os": "linux"})
        hm2.mark_executed(cmd3_id, success=True)
        
        stats2_before = hm2.get_statistics()
        assert stats2_before['total_commands'] == 2
        
        # Import db1 into db2 (should not duplicate)
        hm2.import_database(str(db1_path), merge=True)
        
        stats2_after = hm2.get_statistics()
        # Should still have 2 commands (duplicate ignored)
        assert stats2_after['total_commands'] == 2
        assert stats2_after['executed_commands'] == 2


def test_import_nonexistent_file():
    """Test importing from nonexistent file raises error"""
    hm = HistoryManager()
    
    with pytest.raises(FileNotFoundError):
        hm.import_database("/nonexistent/path/database.db")


def test_export_in_memory_database():
    """Test that exporting in-memory database raises error"""
    db = Database(":memory:")
    hm = HistoryManager(db)
    
    with pytest.raises(ValueError, match="Cannot export in-memory database"):
        hm.export_database()


def test_import_in_memory_database():
    """Test that importing into in-memory database raises error"""
    db = Database(":memory:")
    hm = HistoryManager(db)
    
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        with pytest.raises(ValueError, match="Cannot import into in-memory database"):
            hm.import_database(tmp.name)


def test_roundtrip_export_import():
    """Test full roundtrip: create, export, import into new db"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create original database
        db1_path = Path(tmpdir) / "original.db"
        db1 = Database(db1_path)
        hm1 = HistoryManager(db1)
        
        # Add multiple commands with various properties
        commands = [
            ("list files", "ls -la", {"os": "linux"}),
            ("find python", "find . -name '*.py'", {"os": "linux"}),
            ("disk usage", "du -sh *", {"os": "macos"}),
        ]
        
        for query, cmd, ctx in commands:
            cmd_id = hm1.add_command(query, cmd, ctx)
            hm1.mark_executed(cmd_id, success=True)
        
        # Set a preference
        hm1.set_preferred_language("zh")
        
        stats_original = hm1.get_statistics()
        lang_original = hm1.get_preferred_language()
        
        # Export
        export_path = Path(tmpdir) / "export.db"
        hm1.export_database(str(export_path))
        
        # Create new database and import
        db2_path = Path(tmpdir) / "imported.db"
        db2 = Database(db2_path)
        hm2 = HistoryManager(db2)
        
        hm2.import_database(str(export_path), merge=True)
        
        stats_imported = hm2.get_statistics()
        lang_imported = hm2.get_preferred_language()
        
        # Verify everything was imported correctly
        assert stats_imported['total_commands'] == stats_original['total_commands']
        assert stats_imported['executed_commands'] == stats_original['executed_commands']
        assert lang_imported == lang_original
