#!/usr/bin/env python3
"""
Database migration script to add execution_timestamps column to existing databases.
This script safely migrates existing command_history records to the new schema.
"""

import sqlite3
import json
from pathlib import Path
from shwizard.utils.platform_utils import get_data_directory

def migrate_database():
    """Add execution_timestamps column to command_history table if it doesn't exist."""
    db_path = get_data_directory() / "history.db"
    
    if not db_path.exists():
        print(f"No database found at {db_path}")
        return
    
    print(f"Migrating database at {db_path}")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("PRAGMA table_info(command_history)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'execution_timestamps' in columns:
            print("✓ execution_timestamps column already exists")
            return
        
        print("Adding execution_timestamps column...")
        
        # Add the new column
        cursor.execute("""
            ALTER TABLE command_history 
            ADD COLUMN execution_timestamps TEXT
        """)
        
        # Initialize execution_timestamps for existing records
        # Use the timestamp field as the initial execution time
        cursor.execute("""
            UPDATE command_history 
            SET execution_timestamps = json_array(timestamp)
            WHERE execution_timestamps IS NULL
        """)
        
        conn.commit()
        print("✓ Migration completed successfully")
        
        # Show statistics
        cursor.execute("SELECT COUNT(*) FROM command_history")
        total = cursor.fetchone()[0]
        print(f"  Updated {total} records")

if __name__ == "__main__":
    migrate_database()
