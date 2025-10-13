import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from shwizard.utils.platform_utils import get_config_directory
from shwizard.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigManager:
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = get_config_directory() / "config.yaml"
        
        self.config_path: Path = config_path
        self.config: Dict[str, Any] = {}
        self._load_default_config()
        self._load_user_config()
    
    def _load_default_config(self):
        default_config_path = Path(__file__).parent.parent / "data" / "default_config.yaml"
        if default_config_path.exists():
            with open(default_config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
                logger.debug(f"Loaded default config from {default_config_path}")
        else:
            logger.warning(f"Default config not found at {default_config_path}")
            self.config = self._get_fallback_config()
    
    def _load_user_config(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
                    self._merge_config(self.config, user_config)
                    logger.debug(f"Loaded user config from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading user config: {e}")
        else:
            self._create_default_user_config()
    
    def _merge_config(self, base: Dict, override: Dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _create_default_user_config(self):
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Created default user config at {self.config_path}")
        except Exception as e:
            logger.error(f"Error creating default config: {e}")
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        return {
            "ollama": {
                "embedded": True,
                "auto_download": True,
                "base_url": "http://localhost:11435",
                "model": "gemma2:2b",
                "timeout": 60,
            },
            "safety": {
                "enabled": True,
                "confirm_high_risk": True,
                "warn_medium_risk": True,
            },
            "history": {
                "enabled": True,
                "max_entries": 10000,
                "priority_search": True,
            },
            "ui": {
                "color_enabled": True,
                "show_explanations": True,
                "confirm_execution": True,
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        keys = key.split(".")
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self.save()
    
    def save(self):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.debug(f"Saved config to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def reload(self):
        self._load_default_config()
        self._load_user_config()
