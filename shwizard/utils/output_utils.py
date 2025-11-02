import re
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def sanitize_output(output: Optional[str]) -> str:
    """
    Normalize shell command output for safe rendering with Rich.

    Rationale:
    - Many CLI tools use carriage returns (\r) to update the same line (progress bars, spinners).
      When such raw output is embedded in a Rich Panel, \r can corrupt the layout and cause
      misaligned boxes or excessive indentation/spaces.
    - This function normalizes line endings and removes certain control sequences that can
      interfere with Rich rendering, while preserving color (ANSI SGR) codes.

    Behavior:
    - Convert CRLF to LF, and solitary CR to LF.
    - Remove null bytes.
    - Drop CSI K (erase line) sequences and cursor save/restore (CSI s/u) to avoid odd rewrites.
    - Collapse excessive spaces likely produced by \r-overwrites.
    - Trim trailing spaces on each line.
    - Collapse more than two consecutive blank lines.
    - Also handle inputs where control characters are provided as textual escape sequences
      (e.g. \"\\r\", \"\\n\", \"\\x1b[...\") instead of raw bytes.

    Note:
    - Keep ANSI color/style codes intact. Use Rich Text.from_ansi() to render them properly.
    """
    if output is None:
        return ""

    text = output

    # ---- Handle actual control characters first ----

    # Normalize line endings (actual)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\r(?!\n)", "\n", text)

    # Remove trailing null bytes that may appear in some outputs
    text = text.replace("\x00", "")

    # Remove control sequences that can cause unexpected rewrites inside panels
    # - CSI K: Erase in Line
    text = re.sub(r"\x1b\[\d*K", "", text)
    # - CSI s/u: save/restore cursor
    text = re.sub(r"\x1b\[[su]", "", text)

    # Collapse very long sequences of spaces (from overwritten progress bars)
    text = re.sub(r"[ ]{100,}", " ", text)

    # Trim trailing spaces per actual line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse more than two blank lines (actual)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # ---- Also handle textual escape sequences (literal backslashes) ----

    # Normalize escaped CRLF and solitary escaped CR to escaped LF
    text = text.replace("\\r\\n", "\\n")
    text = re.sub(r"\\r(?!\\n)", lambda m: "\\n", text)

    # Remove escaped control sequences (literal form)
    text = re.sub(r"\\x1b\[\d*K", "", text)  # ESC[...K
    text = re.sub(r"\\x1b\[[su]", "", text)  # ESC[s or ESC[u

    # Collapse long spaces again (just in case)
    text = re.sub(r"[ ]{100,}", " ", text)

    # Trim spaces around escaped newline tokens, trim trailing spaces per segment, and collapse blank lines to max 2
    if "\\n" in text:
        # Remove spaces around literal \n tokens
        text = re.sub(r"[ \t]*\\n[ \t]*", lambda m: "\\n", text)
        # Trim trailing spaces for each segment delimited by \n
        segments = [seg.rstrip() for seg in text.split("\\n")]
        text = "\\n".join(segments)
        # Collapse 3 or more consecutive escaped newlines to two
        text = re.sub(r"(?:\\n){3,}", lambda m: "\\n\\n", text)

    return text


def display_paginated_output(
    output: str, console: Console, title: str = "Output", page_size: int = 30
):
    """
    Display command output with pagination support for large outputs.

    Args:
        output: The command output to display
        console: Rich Console instance for output
        title: Title for the output panel
        page_size: Number of lines to display per page (default: 30)

    Features:
        - Automatically sanitizes output before display
        - Shows pagination controls at the bottom
        - Supports Enter (next page), q (quit), b (back)
        - Displays current page and total pages
    """
    if not output:
        console.print(f"[dim]No {title.lower()} to display[/dim]")
        return

    # Sanitize output first
    sanitized = sanitize_output(output)
    lines = sanitized.split("\n")
    total_lines = len(lines)

    # If output is small, just display it directly
    if total_lines <= page_size:
        text = Text.from_ansi(sanitized)
        console.print(Panel(text, title=title, expand=True))
        return

    # Paginate for large output
    total_pages = (total_lines + page_size - 1) // page_size
    current_page = 0

    while current_page < total_pages:
        # Display current page
        _display_page(lines, current_page, total_pages, page_size, total_lines, console, title)

        # Check if we're on the last page
        if _is_last_page(current_page, total_lines, page_size):
            console.print("[dim]End of output[/dim]")
            break

        # Get user input for navigation
        action = _get_page_action(console)

        if action == "quit":
            console.print("[yellow]Output display cancelled[/yellow]")
            break
        elif action == "back" and current_page > 0:
            current_page -= 1
        elif action in ("next", "unknown"):
            current_page += 1


def _display_page(lines, current_page, total_pages, page_size, total_lines, console, title):
    """Display a single page of output."""
    start_idx = current_page * page_size
    end_idx = min(start_idx + page_size, total_lines)
    page_lines = lines[start_idx:end_idx]
    page_text = "\n".join(page_lines)

    text = Text.from_ansi(page_text)
    page_info = f"{title} - Page {current_page + 1}/{total_pages} (Lines {start_idx + 1}-{end_idx}/{total_lines})"
    console.print(Panel(text, title=page_info, expand=True))


def _is_last_page(current_page, total_lines, page_size):
    """Check if current page is the last page."""
    end_idx = min((current_page + 1) * page_size, total_lines)
    return end_idx >= total_lines


def _get_page_action(console):
    """Get user action for page navigation. Returns: 'next', 'back', 'quit', or 'unknown'."""
    console.print(
        "[cyan]More output available[/cyan] " "[dim](Enter=next, b=back, q=quit)[/dim]", end=" "
    )

    try:
        # Use readchar if available, fallback to input
        try:
            import readchar as rc

            key = rc.readkey()
            console.print()  # New line after key press
        except ImportError:
            key = input().strip().lower() or "enter"

        if key.lower() in ("q", "Q"):
            return "quit"
        elif key.lower() in ("b", "B"):
            return "back"
        elif key in ("\r", "\n", " ", ""):
            return "next"
        else:
            return "unknown"  # Treat unknown keys as next

    except KeyboardInterrupt:
        console.print("\n[yellow]Output display cancelled[/yellow]")
        return "quit"


def should_paginate_output(
    output: str, threshold_lines: int = 50, threshold_chars: int = 5000
) -> bool:
    """
    Determine if output should be paginated based on size thresholds.

    Args:
        output: The command output to check
        threshold_lines: Maximum lines before pagination (default: 50)
        threshold_chars: Maximum characters before pagination (default: 5000)

    Returns:
        bool: True if output should be paginated
    """
    if not output:
        return False

    line_count = output.count("\n") + 1
    char_count = len(output)

    return line_count > threshold_lines or char_count > threshold_chars
