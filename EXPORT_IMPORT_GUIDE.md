# Database Export/Import Guide

## Overview

SHWizard now supports exporting and importing your command history database, making it easy to:
- **Backup** your command history
- **Migrate** your history to a new machine
- **Share** useful command patterns with your team
- **Restore** from backups

## Usage

### Export Command

Export your history database to a backup file:

```bash
# In interactive mode:
/export                      # Exports to ~/shwizard_backup.db (default)
/export ~/my_backup.db       # Exports to a custom location
/export /path/to/backup.db   # Exports to any specified path
```

The export command:
- Creates a complete copy of your history database
- Includes all command history, execution counts, and user preferences
- Creates parent directories automatically if they don't exist
- Reports the export location on success

### Import Command

Import a history database from a backup file:

```bash
# In interactive mode:
/import ~/my_backup.db       # Import and merge with existing history
/import /path/to/backup.db   # Import from any specified path
```

The import command:
- **Merges** the imported history with your existing database by default
- Avoids duplicate entries (same query + command combination)
- Updates user preferences from the imported database
- Preserves all execution timestamps and counts
- **Immediately refreshes** the prompt history for auto-completion
- All imported commands are instantly available for search and use

## Common Scenarios

### Backup Before System Migration

```bash
# On old machine:
shwizard -i
/export ~/shwizard_backup.db
# Copy the file to your new machine

# On new machine:
shwizard -i
/import ~/shwizard_backup.db
```

### Regular Backups

```bash
# Export with timestamp in filename
/export ~/backups/shwizard_$(date +%Y%m%d).db
```

### Sharing Command Patterns

```bash
# Export on one machine
/export ~/shared_commands.db

# Import on another machine (merges with existing)
/import ~/shared_commands.db
```

### Restore from Backup

```bash
# If you want to restore to a specific backup
/import ~/backups/shwizard_20251026.db
```

## Technical Details

### Database Location

The default history database is stored at:
- **macOS**: `~/Library/Application Support/shwizard/history.db`
- **Linux**: `~/.local/share/shwizard/history.db`
- **Windows**: `%LOCALAPPDATA%\shwizard\history.db`

### Export Behavior

- Creates an exact copy of the database file using `shutil.copy2`
- Preserves file metadata (timestamps, permissions)
- Default export location: `~/shwizard_backup.db`
- Validates that source database exists before export

### Import Behavior

- **Merge mode** (default): Combines imported history with existing
  - Uses `INSERT OR IGNORE` to prevent duplicate entries
  - Updates user preferences with `INSERT OR REPLACE`
  - Keeps all unique command history from both databases

### Error Handling

The commands handle various error conditions:
- Missing import file → Shows clear error message
- Invalid database format → Reports validation failure
- Permission issues → Reports access denied
- Invalid paths → Shows usage information

## Examples

### Example 1: Create Daily Backup

```bash
# In your shell:
shwizard -i
/export ~/backups/daily/shwizard_$(date +%Y%m%d).db
/quit
```

### Example 2: Migrate to New Machine

```bash
# Old machine:
shwizard -i
/export ~/transfer/my_history.db
/quit

# Transfer the file via scp, USB, cloud storage, etc.

# New machine:
shwizard -i
/import ~/transfer/my_history.db
/stats  # Verify the import
/quit
```

### Example 3: Share Team Commands

```bash
# Team member exports useful commands:
shwizard -i
/export ~/team_commands.db

# Other team members import:
shwizard -i
/import ~/Downloads/team_commands.db
/history  # Browse the imported commands
```

## Notes

- Imported commands are **immediately available** in history search, keyword search, and auto-completion
- The prompt session history is automatically refreshed after import (no restart needed!)
- The import command always merges by default to preserve your existing history
- Exported databases are standard SQLite files and can be examined with any SQLite tool
- The database includes both command history and user preferences (like language settings)

## Help

For more information about available commands:
```bash
shwizard -i
/help
```
