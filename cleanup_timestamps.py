#!/usr/bin/env python3
"""
Cleanup script to reset execution_timestamps for all records.
This will clear the incorrect timestamps that were added during add_command.
"""

import sqlite3
from pathlib import Path
from shwizard.utils.platform_utils import get_data_directory

def cleanup_timestamps():
    """Reset execution_timestamps to NULL for all records."""
    db_path = get_data_directory() / "history.db"
    
    if not db_path.exists():
        print(f"No database found at {db_path}")
        return
    
    print(f"Cleaning up database at {db_path}")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Clear all execution_timestamps
        print("Resetting execution_timestamps to NULL...")
        cursor.execute("""
            UPDATE command_history 
            SET execution_timestamps = NULL
        """)
        
        conn.commit()
        print("✓ Cleanup completed successfully")
        
        # Show statistics
        cursor.execute("SELECT COUNT(*) FROM command_history")
        total = cursor.fetchone()[0]
        print(f"  Reset {total} records")
        print("\nNote: Execution counts will be rebuilt as you use SHWizard.")

if __name__ == "__main__":
    cleanup_timestamps()
