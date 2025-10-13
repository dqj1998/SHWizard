import pytest

import shwizard.cli as cli
from shwizard.storage.config import ConfigManager


class FakeLLMTranslator:
    """
    Test double for LLMTranslator that tracks detect() invocation count.
    """
    call_count = 0

    def __init__(self, ai_service):
        # No-op for tests
        pass

    def detect(self, text: str) -> str:
        FakeLLMTranslator.call_count += 1
        # Return a plausible language code regardless of input
        # Natural language case might be Chinese, but the exact value doesn't matter here
        return "en" if text.strip() == "ls" else "zh"

    def translate(self, text: str, target_lang: str, source_lang: str | None = None) -> str:
        # For UI string translation, simply return input; UI_STRINGS covers common cases
        return text


class FakeAIService:
    """
    Minimal stub for AIService to avoid real Ollama interactions during tests.
    """
    def __init__(self, model=None, base_url=None, timeout=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def initialize(self) -> bool:
        return True

    def generate_commands(self, query: str, context, relevant_commands):
        # Return a trivial safe command for natural language path
        return ["echo hello"]


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    """
    Patch cli module to use fakes for translator and AI service in all tests.
    """
    # Reset call count before each test
    FakeLLMTranslator.call_count = 0
    monkeypatch.setattr(cli, "LLMTranslator", FakeLLMTranslator)
    monkeypatch.setattr(cli, "AIService", FakeAIService)


def test_command_input_does_not_trigger_language_detection():
    """
    When the user input is a direct shell command (e.g., 'ls'),
    language detection must be skipped to avoid misclassification issues.
    """
    cfg = ConfigManager()
    # Dry-run to avoid executing anything; safety checks and history recording are okay
    cli.process_query("ls", cfg, dry_run=True, safety_enabled=True)
    assert FakeLLMTranslator.call_count == 0, "Language detection should be skipped for command inputs"


def test_natural_language_triggers_language_detection():
    """
    When the input is natural language (e.g., Chinese text),
    language detection should run to set UI language appropriately.
    """
    cfg = ConfigManager()
    # Use a clear natural-language input in Chinese
    cli.process_query("删除所有临时文件", cfg, dry_run=True, safety_enabled=True)
    assert FakeLLMTranslator.call_count == 1, "Language detection should run exactly once for natural language input"
