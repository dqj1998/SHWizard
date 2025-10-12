import pytest
from shwizard.safety.checker import SafetyChecker
from shwizard.safety.rules import RulesManager


def test_safety_checker_initialization():
    checker = SafetyChecker()
    assert checker.enabled == True


def test_dangerous_command_detection():
    checker = SafetyChecker()
    
    result = checker.check_command("rm -rf /")
    assert not result.is_safe or result.is_blocked()
    assert result.risk_level == "high_risk"
    
    result = checker.check_command("rm -rf /tmp/test")
    assert result.risk_level in ["medium_risk", "high_risk"]
    
    result = checker.check_command("ls -la")
    assert result.is_safe
    assert result.risk_level == "safe"


def test_disabled_safety_checker():
    checker = SafetyChecker(enabled=False)
    result = checker.check_command("rm -rf /")
    assert result.is_safe


def test_empty_command():
    checker = SafetyChecker()
    result = checker.check_command("")
    assert not result.is_safe
    assert "Empty" in result.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
