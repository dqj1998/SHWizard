# SHWizard AI Coding Assistant Guidelines

## Python env
venv: /Users/qingjie.du/HDD/my-prjs/myenv

## Architecture Overview

SHWizard is a natural language → shell command assistant with embedded local AI (Ollama). The architecture follows a pipeline pattern:

```
User Query → Context Collection → History Search → AI Generation → Safety Check → Execution → Feedback Storage
```

**Core Components:**
- `shwizard/core/` - AI service, context manager, command executor (execution engine)
- `shwizard/storage/` - SQLite database, config manager, history manager (persistence layer)
- `shwizard/safety/` - Multi-level dangerous command detection using regex rules
- `shwizard/llm/` - Ollama lifecycle management (download, install, server control)
- `shwizard/utils/` - I18n (LLM-powered translation), shell completion, logging

## Critical Data Flow Patterns

### 1. Command Deduplication via execution_timestamps
Commands are NOT duplicated in the database. The same query+command pair reuses the existing record and appends timestamps to `execution_timestamps` (JSON array). This powers the "execution count" feature.

**Example from `database.py:133-155`:**
```python
# Get current execution_timestamps and append new timestamp
cursor = conn.execute("SELECT execution_timestamps FROM command_history WHERE id = ?", (command_id,))
timestamps_json = cursor.fetchone()[0] or "[]"
timestamps = json.loads(timestamps_json)
timestamps.append(datetime.now().isoformat())
```

When displaying history, `execution_count = len(execution_timestamps)`.

### 2. Interactive Mode Command Prefilling
Interactive mode allows editing AI-generated commands before execution. The flow uses `prefill_command` state:

- User selects command with 'e' (edit) → returns command string to `process_query()`
- `process_query()` returns the command when `interactive_session=True`
- Main loop captures it as `prefill_command` and passes to `session.prompt(default=prefill_command)`

**See `cli.py:936-1100`** for the full interactive loop implementation.

### 3. Built-in cd Command Handling
`cd` is intercepted and handled by changing Python's `os.getcwd()` (NOT executed as shell command) because we need the process directory to persist across commands in interactive mode.

**Pattern in `cli.py:74-120`:**
```python
def handle_cd(command_str, tokens, ...):
    os.chdir(target)
    os.environ["OLDPWD"] = cwd_before
    PREV_DIR = cwd_before  # Global state for 'cd -'
```

## Development Workflows

### Testing
```bash
make test          # Run pytest suite
make lint          # flake8 with --max-line-length=100 --ignore=E203,W503
make format        # Black formatter
```

**Test structure:** In-memory SQLite databases (`:memory:`) are used extensively in tests to avoid filesystem state. See `tests/test_keyword_search.py` for patterns.

### Building Binary
```bash
make binary        # Uses PyInstaller, defined in build.py
```

The binary bundles Ollama manager and default model download URLs from `shwizard/data/default_config.yaml`.

## Project-Specific Conventions

### 1. Rich Console Output Everywhere
All user-facing text uses `rich.console.Console`. Examples:
- Tables: `Table(box=box.ROUNDED)` with `no_wrap=True, overflow="ellipsis"` for text columns
- Panels: `Panel.fit()` for section headers
- Status: `console.status("[bold green]...")` for loading states

### 2. Multi-Language Support via LLM Translation
Translation is NOT static dictionaries for most content—it uses the AI model itself:

**Pattern in `utils/i18n.py`:**
```python
def tr_llm(text: str, target_lang: str, translator: LLMTranslator) -> str:
    # Falls back to static UI_STRINGS only for core UI elements
    # Otherwise uses LLM for translation on-the-fly
```

This means safety rule descriptions, command explanations, etc. are translated by calling the AI model.

### 3. Safety Rules Structure
Defined in YAML (`shwizard/data/dangerous_commands.yaml`) with three risk levels:
- `high_risk` - Blocked or requires explicit confirmation (e.g., `rm -rf /`)
- `medium_risk` - Warning shown (e.g., `chmod 777`)
- `low_risk` - Informational (e.g., `sudo` operations)

**Rule matching:** Regex pattern matching in `safety/rules.py:check_command()`.

### 4. History Priority Scoring
`HistoryManager._calculate_priority_score()` uses weighted scoring:
- 40% frequency (execution count)
- 30% recency (days since last use)
- 30% success rate (user_feedback > 0)
- 20% bonus for same working directory

This powers the history-aware command suggestions.

### 5. Keyword Search with Match Counting
Recent feature: `search_by_keywords()` returns results with `keyword_match_count` field showing how many keywords matched. Used in interactive mode for direct command selection (1-8 keys).

**See `cli.py:273-330`** for keyword-based history display implementation.

## Integration Points

### Ollama Server Management
`llm/ollama_manager.py` handles:
- Auto-download from platform-specific URLs (macOS/Linux/Windows)
- Server lifecycle (start/stop on custom port 11435)
- Model availability checks and downloads
- Process cleanup via `atexit.register()`

**Critical:** Uses port 11435 (not default 11434) to avoid conflicts.

### Terminal Mouse Mode Control
VSCode integrated terminal requires explicit ANSI escape sequences to disable mouse reporting for text selection:

**Pattern in `cli.py:43-56`:**
```python
def set_mouse_mode(enabled: bool):
    if enabled:
        sys.stdout.write("\x1b[?1000h\x1b[?1006h")  # Enable
    else:
        sys.stdout.write("\x1b[?1000l\x1b[?1002l...")  # Disable ALL modes
```

Config option `ui.mouse_auto_mode` toggles this automatically when showing history/help.

## Common Gotchas

1. **Don't add new commands to database directly** - use `HistoryManager.add_command()` which handles deduplication
2. **SQLite text columns for JSON** - `execution_timestamps`, `context_data` are TEXT storing JSON strings; always `json.loads()` on read
3. **Interactive mode state** - `PREV_DIR` is a global for `cd -` support; reset on exit
4. **prompt_toolkit version compatibility** - Try/except blocks for `CompleteStyle`, `EditingMode` due to API changes across versions
5. **Rich table truncation** - Use `no_wrap=True` + manual truncation to 37 chars + "..." for 40-char columns (see `cli.py:830-837`)

## Key Files to Reference

- `cli.py:157-545` - Main `process_query()` orchestration with all error handling
- `storage/database.py:131-160` - Deduplication and timestamp appending logic
- `core/ai_service.py` - Prompt building and Ollama communication patterns
- `utils/prompt_utils.py` - AI prompt templates for command generation
- `IMPLEMENTATION_SUMMARY.md` - Export/import feature implementation details
