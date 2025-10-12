import re
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any
from shwizard.utils.logger import get_logger

logger = get_logger(__name__)


class DangerousCommandRule:
    def __init__(
        self,
        pattern: str,
        description: str,
        risk_level: str,
        confirmation_required: bool = False,
        warning: bool = False,
        info: bool = False,
        block: bool = False
    ):
        self.pattern = pattern
        self.regex = re.compile(pattern)
        self.description = description
        self.risk_level = risk_level
        self.confirmation_required = confirmation_required
        self.warning = warning
        self.info = info
        self.block = block
    
    def matches(self, command: str) -> bool:
        return self.regex.search(command) is not None


class RulesManager:
    def __init__(self, rules_file: Optional[Path] = None):
        self.rules: Dict[str, List[DangerousCommandRule]] = {
            "high_risk": [],
            "medium_risk": [],
            "low_risk": []
        }
        self._load_default_rules()
        
        if rules_file and rules_file.exists():
            self._load_custom_rules(rules_file)
    
    def _load_default_rules(self):
        default_rules_path = Path(__file__).parent.parent / "data" / "dangerous_commands.yaml"
        if default_rules_path.exists():
            try:
                with open(default_rules_path, "r", encoding="utf-8") as f:
                    rules_data = yaml.safe_load(f)
                    self._parse_rules(rules_data)
                logger.debug(f"Loaded default rules from {default_rules_path}")
            except Exception as e:
                logger.error(f"Error loading default rules: {e}")
    
    def _load_custom_rules(self, rules_file: Path):
        try:
            with open(rules_file, "r", encoding="utf-8") as f:
                rules_data = yaml.safe_load(f)
                
                if "custom_rules" in rules_data and rules_data["custom_rules"].get("enabled"):
                    custom_patterns = rules_data["custom_rules"].get("user_patterns", [])
                    for pattern_data in custom_patterns:
                        rule = DangerousCommandRule(
                            pattern=pattern_data["pattern"],
                            description=pattern_data.get("description", "Custom rule"),
                            risk_level=pattern_data.get("risk_level", "medium_risk"),
                            confirmation_required=pattern_data.get("confirmation_required", False),
                            warning=pattern_data.get("warning", True),
                            block=pattern_data.get("block", False)
                        )
                        self.rules[rule.risk_level].append(rule)
                
            logger.debug(f"Loaded custom rules from {rules_file}")
        except Exception as e:
            logger.error(f"Error loading custom rules: {e}")
    
    def _parse_rules(self, rules_data: Dict[str, Any]):
        for risk_level in ["high_risk", "medium_risk", "low_risk"]:
            if risk_level in rules_data:
                for rule_data in rules_data[risk_level]:
                    rule = DangerousCommandRule(
                        pattern=rule_data["pattern"],
                        description=rule_data["description"],
                        risk_level=risk_level,
                        confirmation_required=rule_data.get("confirmation_required", False),
                        warning=rule_data.get("warning", False),
                        info=rule_data.get("info", False),
                        block=rule_data.get("block", False)
                    )
                    self.rules[risk_level].append(rule)
    
    def check_command(self, command: str) -> Optional[DangerousCommandRule]:
        for risk_level in ["high_risk", "medium_risk", "low_risk"]:
            for rule in self.rules[risk_level]:
                if rule.matches(command):
                    return rule
        return None
    
    def get_all_rules(self) -> Dict[str, List[DangerousCommandRule]]:
        return self.rules
