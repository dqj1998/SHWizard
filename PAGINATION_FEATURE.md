# Pagination Feature

## Overview

SHWizard now automatically paginates large command outputs to improve readability and prevent overwhelming the terminal with too much information at once.

## How It Works

### Automatic Detection

The pagination system automatically detects when command output is too large to display comfortably:

- **Line threshold**: More than 50 lines
- **Character threshold**: More than 5,000 characters

If either threshold is exceeded, pagination is automatically enabled.

### User Experience

When pagination is active:

1. **First Page Display**: The first 30 lines of output are shown
2. **Navigation Controls**: 
   - Press `Enter` or `Space` to view the next page
   - Press `b` to go back to the previous page
   - Press `q` to quit and return to the command prompt
3. **Progress Indicator**: Current page number and line range are shown in the panel title
   - Example: `Output - Page 2/5 (Lines 31-60/150)`

### Commands That Benefit

This feature is particularly useful for commands that produce large outputs:

- `cat large_file.txt` - Viewing file contents
- `ls -R` - Recursive directory listings
- `find /path` - Finding files
- `grep -r pattern` - Recursive searching
- `tail -n 1000 file.log` - Large log file excerpts
- `history` - Command history
- Any command with verbose output

## Technical Details

### Implementation

The pagination feature is implemented in `shwizard/utils/output_utils.py`:

- `should_paginate_output(output, threshold_lines=50, threshold_chars=5000)`: Determines if pagination is needed
- `display_paginated_output(output, console, title="Output", page_size=30)`: Displays output with pagination controls

### Integration Points

Pagination is integrated at all command output display points in `cli.py`:

1. Direct command execution
2. History command execution
3. AI-generated command execution

### Configuration

Currently, the thresholds are fixed in the code:
- **Line threshold**: 50 lines
- **Character threshold**: 5,000 characters
- **Page size**: 30 lines per page

Future enhancement: These could be made configurable via the config file.

## Examples

### Example 1: Viewing a Large File

```bash
$ shwizard -i
>>> cat /var/log/system.log
```

Output would be paginated if the log file has more than 50 lines.

### Example 2: Recursive Directory Listing

```bash
$ shwizard -i
>>> list all files in /usr recursively
```

The AI would generate `ls -R /usr` or `find /usr`, and the output would be paginated.

### Example 3: Large Find Results

```bash
$ shwizard -i
>>> find all python files
```

If there are many Python files, the output will be paginated automatically.

## Testing

Run the pagination test suite:

```bash
pytest tests/test_pagination.py -v
```

Create test files and try pagination:

```bash
python test_pagination.py
shwizard -i
```

Then use the commands displayed by the test script.

## Benefits

1. **Better Readability**: Large outputs are broken into manageable chunks
2. **Prevents Scrollback Loss**: Avoids filling the entire terminal buffer
3. **Navigation Control**: Users can review output at their own pace
4. **Automatic**: No need to pipe through `less` or `more` manually
5. **Consistent UX**: Same experience across all commands

## Future Enhancements

Potential improvements:

1. **Configurable Thresholds**: Allow users to set custom line/char thresholds
2. **Search Within Output**: Add `/` to search within paginated output
3. **Jump to Page**: Allow `g` to jump to a specific page number
4. **Copy Support**: Better integration with clipboard for selecting text
5. **Syntax Highlighting**: Apply syntax highlighting to code outputs
