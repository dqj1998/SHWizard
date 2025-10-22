import yaml
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import shutil
from datetime import datetime
from shwizard.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigMigrator:
    """Handles migration from Ollama configuration to LLM_cpp configuration."""
    
    # Mapping from Ollama model names to GGUF equivalents
    OLLAMA_TO_GGUF_MAPPING = {
        "gemma3:270m": "gemma-3-270m-Q8_0.gguf",
        "gemma2:2b": "gemma-2-2b-it-q4_k_m.gguf",
        "gemma2:2b-instruct": "gemma-2-2b-it-q4_k_m.gguf",
        "llama3:8b": "llama-3.1-8b-instruct-q4_k_m.gguf",
        "llama3:8b-instruct": "llama-3.1-8b-instruct-q4_k_m.gguf",
        "llama3.1:8b": "llama-3.1-8b-instruct-q4_k_m.gguf",
        "llama3.1:8b-instruct": "llama-3.1-8b-instruct-q4_k_m.gguf",
        "llama3.2:3b": "llama-3.2-3b-instruct-q4_k_m.gguf",
        "llama3.2:3b-instruct": "llama-3.2-3b-instruct-q4_k_m.gguf",
        "phi3:mini": "phi-3-mini-4k-instruct-q4.gguf",
        "phi3:3.8b": "phi-3-mini-4k-instruct-q4.gguf",
        "qwen2:1.5b": "qwen2-1.5b-instruct-q4_k_m.gguf",
        "qwen2:1.5b-instruct": "qwen2-1.5b-instruct-q4_k_m.gguf",
        "codellama:7b": "llama-3.1-8b-instruct-q4_k_m.gguf",  # Fallback to general model
        "mistral:7b": "llama-3.1-8b-instruct-q4_k_m.gguf",   # Fallback to general model
    }
    
    def __init__(self):
        """Initialize the configuration migrator."""
        self.migration_log = []
    
    def migrate_config_file(self, config_path: Path, backup: bool = True) -> bool:
        """
        Migrate a configuration file from Ollama to LLM_cpp format.
        
        Args:
            config_path: Path to the configuration file
            backup: Whether to create a backup of the original file
            
        Returns:
            True if migration was successful, False otherwise
        """
        try:
            if not config_path.exists():
                logger.error(f"Configuration file not found: {config_path}")
                return False
            
            # Load existing configuration
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            # Check if migration is needed
            if not self._needs_migration(config):
                logger.info("Configuration already uses LLM_cpp format or no migration needed")
                return True
            
            # Create backup if requested
            if backup:
                backup_path = self._create_backup(config_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Perform migration
            migrated_config = self.migrate_config_dict(config)
            
            # Save migrated configuration
            with open(config_path, 'w') as f:
                yaml.dump(migrated_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration migrated successfully: {config_path}")
            self._log_migration_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate configuration file: {e}")
            return False
    
    def migrate_config_dict(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate a configuration dictionary from Ollama to LLM_cpp format.
        
        Args:
            config: Original configuration dictionary
            
        Returns:
            Migrated configuration dictionary
        """
        migrated_config = config.copy()
        self.migration_log = []
        
        # Add LLM_cpp configuration section
        if "llm" not in migrated_config:
            migrated_config["llm"] = {}
        
        llm_config = migrated_config["llm"]
        
        # Set backend to LLM_cpp
        llm_config["backend"] = "llmcpp"
        self._log_change("Set backend to 'llmcpp'")
        
        # Migrate Ollama settings to LLM_cpp
        if "ollama" in config:
            ollama_config = config["ollama"]
            self._migrate_ollama_settings(ollama_config, llm_config)
        
        # Set LLM_cpp specific defaults
        self._set_llmcpp_defaults(llm_config)
        
        # Preserve existing LLM_cpp settings if any
        self._preserve_existing_llmcpp_settings(config.get("llm", {}), llm_config)
        
        return migrated_config
    
    def _needs_migration(self, config: Dict[str, Any]) -> bool:
        """Check if configuration needs migration."""
        # If LLM section exists and backend is already llmcpp, no migration needed
        if "llm" in config:
            backend = config["llm"].get("backend", "")
            if backend == "llmcpp":
                return False
        
        # If Ollama section exists, migration is needed
        if "ollama" in config:
            return True
        
        # If no LLM section exists at all, migration is needed to add defaults
        if "llm" not in config:
            return True
        
        return False
    
    def _migrate_ollama_settings(self, ollama_config: Dict[str, Any], llm_config: Dict[str, Any]):
        """Migrate Ollama-specific settings to LLM_cpp format."""
        # Migrate model name
        ollama_model = ollama_config.get("model", "")
        if ollama_model:
            gguf_model = self._map_ollama_model_to_gguf(ollama_model)
            llm_config["model"] = gguf_model
            self._log_change(f"Migrated model: '{ollama_model}' -> '{gguf_model}'")
        
        # Migrate timeout and retries
        if "timeout" in ollama_config:
            llm_config["timeout"] = ollama_config["timeout"]
            self._log_change(f"Migrated timeout: {ollama_config['timeout']}")
        
        if "max_retries" in ollama_config:
            llm_config["max_retries"] = ollama_config["max_retries"]
            self._log_change(f"Migrated max_retries: {ollama_config['max_retries']}")
        
        # Migrate auto_download setting
        if "auto_download" in ollama_config:
            llm_config["auto_download"] = ollama_config["auto_download"]
            self._log_change(f"Migrated auto_download: {ollama_config['auto_download']}")
        
        # Migrate install path to model path
        if "download" in ollama_config and "install_path" in ollama_config["download"]:
            install_path = ollama_config["download"]["install_path"]
            # Convert Ollama install path to LLM_cpp model path
            if install_path.endswith("ollama"):
                model_path = install_path.replace("ollama", "models")
            else:
                model_path = str(Path(install_path).parent / "models")
            
            llm_config["model_path"] = model_path
            self._log_change(f"Migrated model path: '{install_path}' -> '{model_path}'")
    
    def _map_ollama_model_to_gguf(self, ollama_model: str) -> str:
        """Map Ollama model name to GGUF equivalent."""
        # Direct mapping
        if ollama_model in self.OLLAMA_TO_GGUF_MAPPING:
            return self.OLLAMA_TO_GGUF_MAPPING[ollama_model]
        
        # Pattern-based mapping
        model_lower = ollama_model.lower()
        
        # Gemma models
        if "gemma" in model_lower:
            if "270m" in model_lower:
                return "gemma-3-270m-Q8_0.gguf"
            elif "2b" in model_lower:
                return "gemma-2-2b-it-q4_k_m.gguf"
            else:
                return "gemma-2-2b-it-q4_k_m.gguf"  # Default to 2B
        
        # Llama models
        elif "llama" in model_lower:
            if "8b" in model_lower:
                return "llama-3.1-8b-instruct-q4_k_m.gguf"
            elif "3b" in model_lower:
                return "llama-3.2-3b-instruct-q4_k_m.gguf"
            else:
                return "llama-3.2-3b-instruct-q4_k_m.gguf"  # Default to 3B
        
        # Phi models
        elif "phi" in model_lower:
            return "phi-3-mini-4k-instruct-q4.gguf"
        
        # Qwen models
        elif "qwen" in model_lower:
            return "qwen2-1.5b-instruct-q4_k_m.gguf"
        
        # Default fallback
        else:
            logger.warning(f"Unknown Ollama model '{ollama_model}', using default GGUF model")
            return "gemma-3-270m-Q8_0.gguf"
    
    def _set_llmcpp_defaults(self, llm_config: Dict[str, Any]):
        """Set default LLM_cpp settings."""
        defaults = {
            # model_path is automatically determined by platform_utils.get_data_directory()
            "n_ctx": 2048,
            "n_gpu_layers": -1,
            "n_threads": 0,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "use_mmap": True,
            "use_mlock": False,
            "download_source": "huggingface",
            "fallback_models": [
                "gemma-3-270m-Q8_0.gguf",
                "gemma-2-2b-it-q4_k_m.gguf",
                "qwen2-1.5b-instruct-q4_k_m.gguf"
            ],
            "cache": {
                "max_size_mb": 10240,
                "max_age_days": 30,
                "cleanup_on_startup": False
            }
        }
        
        for key, value in defaults.items():
            if key not in llm_config:
                llm_config[key] = value
                self._log_change(f"Set default {key}: {value}")
    
    def _preserve_existing_llmcpp_settings(self, existing_llm_config: Dict[str, Any], new_llm_config: Dict[str, Any]):
        """Preserve any existing LLM_cpp settings."""
        for key, value in existing_llm_config.items():
            if key != "backend":  # Don't override backend setting
                if key in new_llm_config and new_llm_config[key] != value:
                    self._log_change(f"Preserved existing setting {key}: {value}")
                new_llm_config[key] = value
    
    def _create_backup(self, config_path: Path) -> Path:
        """Create a backup of the configuration file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = config_path.with_suffix(f".backup_{timestamp}{config_path.suffix}")
        shutil.copy2(config_path, backup_path)
        return backup_path
    
    def _log_change(self, message: str):
        """Log a migration change."""
        self.migration_log.append(message)
        logger.debug(f"Migration: {message}")
    
    def _log_migration_summary(self):
        """Log a summary of migration changes."""
        if self.migration_log:
            logger.info(f"Migration completed with {len(self.migration_log)} changes:")
            for change in self.migration_log:
                logger.info(f"  - {change}")
        else:
            logger.info("Migration completed with no changes needed")
    
    def get_migration_preview(self, config: Dict[str, Any]) -> List[str]:
        """
        Get a preview of changes that would be made during migration.
        
        Args:
            config: Configuration dictionary to preview
            
        Returns:
            List of change descriptions
        """
        # Temporarily store the original log
        original_log = self.migration_log.copy()
        
        # Perform migration to capture changes
        self.migrate_config_dict(config)
        
        # Get the preview
        preview = self.migration_log.copy()
        
        # Restore original log
        self.migration_log = original_log
        
        return preview
    
    def validate_migration(self, original_config: Dict[str, Any], migrated_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that migration was successful.
        
        Args:
            original_config: Original configuration
            migrated_config: Migrated configuration
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check that LLM section exists
        if "llm" not in migrated_config:
            issues.append("Missing 'llm' section in migrated configuration")
        else:
            llm_config = migrated_config["llm"]
            
            # Check backend is set
            if llm_config.get("backend") != "llmcpp":
                issues.append("Backend not set to 'llmcpp'")
            
            # Check model is set
            if not llm_config.get("model"):
                issues.append("No model specified in migrated configuration")
            
            # Check model path is set
            if not llm_config.get("model_path"):
                issues.append("No model_path specified in migrated configuration")
        
        # Check that original Ollama settings were preserved where appropriate
        if "ollama" in original_config:
            ollama_config = original_config["ollama"]
            llm_config = migrated_config.get("llm", {})
            
            # Check timeout migration
            if "timeout" in ollama_config and llm_config.get("timeout") != ollama_config["timeout"]:
                issues.append("Timeout setting not properly migrated")
            
            # Check max_retries migration
            if "max_retries" in ollama_config and llm_config.get("max_retries") != ollama_config["max_retries"]:
                issues.append("max_retries setting not properly migrated")
        
        return len(issues) == 0, issues
    
    def rollback_migration(self, config_path: Path, backup_path: Optional[Path] = None) -> bool:
        """
        Rollback a migration by restoring from backup.
        
        Args:
            config_path: Path to the current configuration file
            backup_path: Path to the backup file (if None, will try to find latest backup)
            
        Returns:
            True if rollback was successful, False otherwise
        """
        try:
            if backup_path is None:
                # Find the most recent backup
                backup_pattern = f"{config_path.stem}.backup_*{config_path.suffix}"
                backup_files = list(config_path.parent.glob(backup_pattern))
                
                if not backup_files:
                    logger.error("No backup files found for rollback")
                    return False
                
                # Sort by modification time and get the most recent
                backup_path = max(backup_files, key=lambda p: p.stat().st_mtime)
            
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Restore from backup
            shutil.copy2(backup_path, config_path)
            logger.info(f"Configuration rolled back from: {backup_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback migration: {e}")
            return False