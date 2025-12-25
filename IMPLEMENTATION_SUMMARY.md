# Feature Implementation Summary

## Export/Import History Database Commands

### Overview
Added two new `/` commands to the SHWizard interactive mode to enable easy backup and migration of command history databases.

### New Commands

#### `/export [path]`
Exports the current history database to a backup file.
- **Default path**: `~/shwizard_backup.db`
- **Custom path**: Specify any path, e.g., `/export ~/my_backup.db`
- **Creates parent directories** automatically if they don't exist
- **Returns error** if trying to export in-memory database

#### `/import <path>`
Imports a history database from a backup file.
- **Required argument**: Path to the backup file to import
- **Merge behavior**: By default, merges with existing history (no duplicates)
- **Deduplication**: Uses query+command combination to avoid duplicates
- **User preferences**: Updates preferences from imported database
- **Auto-refresh**: Automatically refreshes prompt session history (no restart needed)
- **Immediate availability**: All imported commands are instantly searchable
- **Returns error** if:
  - Import file doesn't exist
  - Import file is not a valid SQLite database
  - Trying to import into in-memory database

### Files Modified

1. **shwizard/cli.py**
   - Added `/export` command handler in interactive mode
   - Added `/import` command handler in interactive mode
   - Updated help text to include new commands with usage examples

2. **shwizard/storage/history.py**
   - Added `export_database()` method to HistoryManager
   - Added `import_database()` method to HistoryManager
   - Added proper error handling and validation

3. **shwizard/storage/database.py**
   - Fixed Database `__init__` logic to properly handle custom paths
   - Added `in_memory` attribute for tracking database type

### Documentation

1. **EXPORT_IMPORT_GUIDE.md** (new)
   - Comprehensive guide on export/import functionality
   - Common scenarios and use cases
   - Technical details and error handling
   - Multiple usage examples

2. **README.md** (updated)
   - Added export/import commands to interactive mode examples
   - Added FAQ entry with backup/migration usage
   - Referenced EXPORT_IMPORT_GUIDE.md for detailed usage

### Tests

Created comprehensive test suite in **tests/test_export_import.py**:
- ✅ `test_export_default_location` - Export to default ~/shwizard_backup.db
- ✅ `test_export_custom_location` - Export to custom path
- ✅ `test_export_creates_parent_directories` - Auto-create directories
- ✅ `test_import_merge` - Import and merge databases
- ✅ `test_import_duplicate_handling` - Verify deduplication works
- ✅ `test_import_nonexistent_file` - Error handling for missing files
- ✅ `test_export_in_memory_database` - Prevent exporting in-memory DB
- ✅ `test_import_in_memory_database` - Prevent importing into in-memory DB
- ✅ `test_roundtrip_export_import` - Full export/import cycle

All 9 tests pass successfully.

### Usage Examples

#### Simple Backup
```bash
shwizard -i
/export                        # Exports to ~/shwizard_backup.db
```

#### Backup to Custom Location
```bash
shwizard -i
/export ~/backups/shwizard_$(date +%Y%m%d).db
```

#### Migrate to New Machine
```bash
# Old machine
shwizard -i
/export ~/transfer/my_history.db
/quit

# New machine (after transferring file)
shwizard -i
/import ~/transfer/my_history.db
/stats    # Verify import
```

### Key Features

1. **Merge by Default**: Import always merges with existing database, preserving your current history
2. **Deduplication**: Smart deduplication based on query+command combination
3. **Safe Operations**: Validates file existence, permissions, and database format
4. **Auto-create Directories**: Creates parent directories automatically when exporting
5. **Comprehensive Error Handling**: Clear error messages for common issues
6. **User Preferences Included**: Exports and imports user preferences (e.g., language settings)
7. **Execution History Preserved**: All execution timestamps and counts are maintained

### Database Location

Default database paths:
- **macOS**: `~/Library/Application Support/shwizard/history.db`
- **Linux**: `~/.local/share/shwizard/history.db`
- **Windows**: `%LOCALAPPDATA%\shwizard\history.db`

### Implementation Details

- Uses `shutil.copy2()` for export to preserve metadata
- Uses SQLite ATTACH DATABASE for efficient merging
- Uses NOT EXISTS clause to prevent duplicates during import
- Validates in-memory database operations to prevent errors
- Logs all operations for debugging

### Benefits

1. **Easy Backup**: Users can regularly backup their command history
2. **Migration Support**: Seamlessly move history between machines
3. **Team Sharing**: Share useful command patterns with team members
4. **Disaster Recovery**: Restore from backups if database is corrupted
5. **Testing**: Create test databases without affecting production data
