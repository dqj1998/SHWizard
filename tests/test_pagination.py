"""Tests for pagination functionality in output_utils."""

import unittest
from shwizard.utils.output_utils import (
    should_paginate_output,
    sanitize_output,
    display_paginated_output,
)
from rich.console import Console
from io import StringIO


class TestPagination(unittest.TestCase):
    """Test pagination utilities."""

    def test_should_paginate_output_small_output(self):
        """Small output should not be paginated."""
        small_output = "Line\n" * 20
        self.assertFalse(should_paginate_output(small_output))

    def test_should_paginate_output_large_by_lines(self):
        """Output with many lines should be paginated."""
        large_output = "Line\n" * 100
        self.assertTrue(should_paginate_output(large_output))

    def test_should_paginate_output_large_by_chars(self):
        """Output with many characters should be paginated."""
        large_output = "x" * 10000
        self.assertTrue(should_paginate_output(large_output))

    def test_should_paginate_output_empty(self):
        """Empty output should not be paginated."""
        self.assertFalse(should_paginate_output(""))
        self.assertFalse(should_paginate_output(None))

    def test_should_paginate_output_custom_thresholds(self):
        """Custom thresholds should be respected."""
        output = "Line\n" * 30
        # With default thresholds (50 lines), should not paginate
        self.assertFalse(should_paginate_output(output))
        # With custom threshold (20 lines), should paginate
        self.assertTrue(should_paginate_output(output, threshold_lines=20))

    def test_sanitize_output_preserves_content(self):
        """Sanitize output should preserve actual content."""
        input_text = "Hello\nWorld\nTest"
        output = sanitize_output(input_text)
        self.assertEqual(output, input_text)

    def test_sanitize_output_normalizes_line_endings(self):
        """Sanitize output should normalize line endings."""
        input_text = "Hello\r\nWorld\rTest"
        output = sanitize_output(input_text)
        self.assertEqual(output, "Hello\nWorld\nTest")

    def test_sanitize_output_handles_none(self):
        """Sanitize output should handle None input."""
        output = sanitize_output(None)
        self.assertEqual(output, "")

    def test_display_paginated_output_empty(self):
        """Display paginated output should handle empty input."""
        console = Console(file=StringIO())
        # Should not raise an error
        display_paginated_output("", console, "Test")
        display_paginated_output(None, console, "Test")


if __name__ == "__main__":
    unittest.main()
