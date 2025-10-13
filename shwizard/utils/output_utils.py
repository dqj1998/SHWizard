import re
from typing import Optional


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
