from typing import Optional, Tuple
from shwizard.safety.rules import RulesManager, DangerousCommandRule
from shwizard.utils.logger import get_logger

logger = get_logger(__name__)


class SafetyCheckResult:
    def __init__(
        self,
        is_safe: bool,
        risk_level: Optional[str] = None,
        rule: Optional[DangerousCommandRule] = None,
        message: Optional[str] = None
    ):
        self.is_safe = is_safe
        self.risk_level = risk_level or "safe"
        self.rule = rule
        self.message = message or "Command appears safe"
    
    def needs_confirmation(self) -> bool:
        return bool(self.rule and self.rule.confirmation_required)
    
    def is_blocked(self) -> bool:
        return bool(self.rule and self.rule.block)
    
    def needs_warning(self) -> bool:
        return bool(self.rule and self.rule.warning)


class SafetyChecker:
    def __init__(self, rules_manager: Optional[RulesManager] = None, enabled: bool = True):
        self.rules_manager = rules_manager or RulesManager()
        self.enabled = enabled
    
    def check_command(self, command: str) -> SafetyCheckResult:
        if not self.enabled:
            return SafetyCheckResult(is_safe=True)
        
        if not command or not command.strip():
            return SafetyCheckResult(is_safe=False, message="Empty command")
        
        rule = self.rules_manager.check_command(command)
        
        if rule is None:
            return SafetyCheckResult(is_safe=True)
        
        is_safe = not rule.block
        
        return SafetyCheckResult(
            is_safe=is_safe,
            risk_level=rule.risk_level,
            rule=rule,
            message=rule.description
        )
    
    def get_risk_level_color(self, risk_level: str) -> str:
        colors = {
            "high_risk": "red",
            "medium_risk": "yellow",
            "low_risk": "green",
            "safe": "green"
        }
        return colors.get(risk_level, "white")
    
    def get_risk_level_emoji(self, risk_level: str) -> str:
        emojis = {
            "high_risk": "🚨",
            "medium_risk": "⚠️",
            "low_risk": "ℹ️",
            "safe": "✅"
        }
        return emojis.get(risk_level, "")
