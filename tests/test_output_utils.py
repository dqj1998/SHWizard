import pytest
from shwizard.utils.output_utils import sanitize_output


def test_sanitize_output_cr_and_crlf_to_lf():
    s = "line1\\rline2\\r\\nline3\\r"
    out = sanitize_output(s)
    assert out == "line1\\nline2\\nline3\\n"


def test_sanitize_output_removes_control_sequences():
    s = "\\x1b[2Kline\\x1b[sfoo\\x1b[u"
    out = sanitize_output(s)
    # CSI K (erase in line) removed, and cursor save/restore removed
    assert out == "linefoo"


def test_sanitize_output_collapse_excessive_spaces():
    s = "start " + (" " * 150) + " end"
    out = sanitize_output(s)
    # Long runs of spaces collapsed to a single space
    assert out == "start  end" or out == "start end"


def test_sanitize_output_trim_trailing_spaces_and_blank_lines():
    s = "abc   \\n   \\n\\n\\nxyz   "
    out = sanitize_output(s)
    # Trailing spaces trimmed, blank lines collapsed to max 2
    assert out == "abc\\n\\nxyz"
