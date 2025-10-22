from typing import Dict, Any, List, Optional
from pathlib import Path
import os
from shwizard.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigValidator:
    """Validates configuration settings for SHWizard."""
    
    @staticmethod
    def validate_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize LLM configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated and normalized configuration
        """
        validated_config = config.copy()
        
        # Ensure llm section exists
        if "llm" not in validated_config:
            validated_config["llm"] = {}
        
        llm_config = validated_config["llm"]
        
        # Validate backend
        backend = llm_config.get("backend", "llmcpp")
        if backend not in ["llmcpp", "ollama"]:
            logger.warning(f"Invalid backend '{backend}', defaulting to 'llmcpp'")
            backend = "llmcpp"
        llm_config["backend"] = backend
        
        # Validate model settings
        if backend == "llmcpp":
            ConfigValidator._validate_llmcpp_config(llm_config)
        else:
            ConfigValidator._validate_ollama_config(llm_config, validated_config.get("ollama", {}))
        
        return validated_config
    
    @staticmethod
    def _validate_llmcpp_config(llm_config: Dict[str, Any]):
        """Validate LLM_cpp specific configuration."""
        # Model settings
        if "model" not in llm_config:
            llm_config["model"] = "gemma-3-270m-Q8_0.gguf"
        
        # Validate model path - use platform-specific default if not specified
        from shwizard.utils.platform_utils import get_data_directory
        model_path = llm_config.get("model_path")
        if not model_path:
            model_path = get_data_directory() / "models"
        else:
            model_path = Path(model_path).expanduser()
        llm_config["model_path"] = str(model_path)
        
        # Validate numeric settings
        numeric_settings = {
            "timeout": (60, 1, 3600),  # (default, min, max)
            "max_retries": (3, 1, 10),
            "n_ctx": (2048, 512, 131072),
            "n_gpu_layers": (-1, -1, 1000),
            "n_threads": (0, 0, 64),
            "temperature": (0.7, 0.0, 2.0),
            "top_p": (0.9, 0.0, 1.0),
            "top_k": (40, 1, 200),
            "repeat_penalty": (1.1, 0.0, 2.0)
        }
        
        for setting, (default, min_val, max_val) in numeric_settings.items():
            value = llm_config.get(setting, default)
            try:
                value = float(value) if setting in ["temperature", "top_p", "repeat_penalty"] else int(value)
                if value < min_val or value > max_val:
                    logger.warning(f"LLM setting '{setting}' value {value} out of range [{min_val}, {max_val}], using default {default}")
                    value = default
                llm_config[setting] = value
            except (ValueError, TypeError):
                logger.warning(f"Invalid LLM setting '{setting}' value: {value}, using default {default}")
                llm_config[setting] = default
        
        # Validate boolean settings
        boolean_settings = {
            "auto_download": True,
            "use_mmap": True,
            "use_mlock": False
        }
        
        for setting, default in boolean_settings.items():
            value = llm_config.get(setting, default)
            if not isinstance(value, bool):
                try:
                    llm_config[setting] = str(value).lower() in ["true", "1", "yes", "on"]
                except:
                    llm_config[setting] = default
        
        # Validate cache settings
        if "cache" not in llm_config:
            llm_config["cache"] = {}
        
        cache_config = llm_config["cache"]
        cache_settings = {
            "max_size_mb": (10240, 1024, 102400),  # 1GB to 100GB
            "max_age_days": (30, 1, 365),
            "cleanup_on_startup": False
        }
        
        for setting, default in cache_settings.items():
            if isinstance(default, bool):
                value = cache_config.get(setting, default)
                if not isinstance(value, bool):
                    try:
                        cache_config[setting] = str(value).lower() in ["true", "1", "yes", "on"]
                    except:
                        cache_config[setting] = default
            else:
                default_val, min_val, max_val = default
                value = cache_config.get(setting, default_val)
                try:
                    value = int(value)
                    if value < min_val or value > max_val:
                        logger.warning(f"Cache setting '{setting}' value {value} out of range [{min_val}, {max_val}], using default {default_val}")
                        value = default_val
                    cache_config[setting] = value
                except (ValueError, TypeError):
                    logger.warning(f"Invalid cache setting '{setting}' value: {value}, using default {default_val}")
                    cache_config[setting] = default_val
        
        # Validate fallback models
        if "fallback_models" not in llm_config:
            llm_config["fallback_models"] = ["gemma-3-270m-Q8_0.gguf", "gemma-2-2b-it-q4_k_m.gguf", "qwen2-1.5b-instruct-q4_k_m.gguf"]
        elif not isinstance(llm_config["fallback_models"], list):
            logger.warning("fallback_models should be a list, using default")
            llm_config["fallback_models"] = ["gemma-3-270m-Q8_0.gguf", "gemma-2-2b-it-q4_k_m.gguf", "qwen2-1.5b-instruct-q4_k_m.gguf"]
        
        # Validate download source
        download_source = llm_config.get("download_source", "huggingface")
        if download_source not in ["huggingface", "local"]:
            logger.warning(f"Invalid download_source '{download_source}', using 'huggingface'")
            llm_config["download_source"] = "huggingface"
    
    @staticmethod
    def _validate_ollama_config(llm_config: Dict[str, Any], ollama_config: Dict[str, Any]):
        """Validate Ollama configuration for backward compatibility."""
        # Use Ollama settings if LLM settings are missing
        if "model" not in llm_config and "model" in ollama_config:
            llm_config["model"] = ollama_config["model"]
        
        if "timeout" not in llm_config and "timeout" in ollama_config:
            llm_config["timeout"] = ollama_config["timeout"]
        
        if "max_retries" not in llm_config and "max_retries" in ollama_config:
            llm_config["max_retries"] = ollama_config["max_retries"]
        
        # Set defaults for missing settings
        llm_config.setdefault("model", "gemma3:270m")
        llm_config.setdefault("timeout", 60)
        llm_config.setdefault("max_retries", 3)
    
    @staticmethod
    def validate_paths(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and expand file paths in configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with validated paths
        """
        validated_config = config.copy()
        
        # Validate LLM model path
        if "llm" in validated_config and "model_path" in validated_config["llm"]:
            model_path = Path(validated_config["llm"]["model_path"]).expanduser()
            validated_config["llm"]["model_path"] = str(model_path)
            
            # Create directory if it doesn't exist
            try:
                model_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create model directory {model_path}: {e}")
        
        # Validate other paths
        path_settings = [
            ("history", "database_path"),
            ("logging", "file"),
            ("safety", "custom_rules_file"),
            ("ollama", "download", "install_path")
        ]
        
        for path_setting in path_settings:
            try:
                current = validated_config
                for key in path_setting[:-1]:
                    if key in current:
                        current = current[key]
                    else:
                        break
                else:
                    if path_setting[-1] in current:
                        path_value = current[path_setting[-1]]
                        if isinstance(path_value, str):
                            expanded_path = Path(path_value).expanduser()
                            current[path_setting[-1]] = str(expanded_path)
            except Exception as e:
                logger.debug(f"Could not validate path {path_setting}: {e}")
        
        return validated_config
    
    @staticmethod
    def check_hardware_compatibility(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check hardware compatibility and adjust settings if needed.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration adjusted for hardware compatibility
        """
        try:
            from shwizard.utils.hardware_detector import HardwareDetector
            from shwizard.utils.platform_optimizer import PlatformOptimizer
            
            capabilities = HardwareDetector.detect_gpu_capabilities()
            platform_config = PlatformOptimizer.get_platform_config()
            
            if "llm" in config and config["llm"].get("backend") == "llmcpp":
                llm_config = config["llm"]
                
                # Auto-adjust GPU layers based on available hardware
                if llm_config.get("n_gpu_layers") == -1:
                    if not (capabilities.get("cuda") or capabilities.get("metal")):
                        llm_config["n_gpu_layers"] = 0
                        logger.info("No GPU acceleration available, using CPU-only inference")
                
                # Auto-adjust memory settings based on platform
                if platform_config.get("memory_optimization", {}).get("use_mlock"):
                    llm_config["use_mlock"] = True
                
                # Adjust context size for limited memory systems
                available_memory = capabilities.get("gpu_memory", 0)
                if available_memory > 0 and available_memory < 4000:  # Less than 4GB VRAM
                    if llm_config.get("n_ctx", 2048) > 2048:
                        llm_config["n_ctx"] = 2048
                        logger.info("Limited VRAM detected, reducing context size to 2048")
        
        except Exception as e:
            logger.debug(f"Could not check hardware compatibility: {e}")
        
        return config
    
    @staticmethod
    def validate_full_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform full configuration validation.
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            Fully validated configuration
        """
        # Step 1: Validate LLM configuration
        config = ConfigValidator.validate_llm_config(config)
        
        # Step 2: Validate and expand paths
        config = ConfigValidator.validate_paths(config)
        
        # Step 3: Check hardware compatibility
        config = ConfigValidator.check_hardware_compatibility(config)
        
        logger.debug("Configuration validation completed")
        return config
    
    @staticmethod
    def get_validation_errors(config: Dict[str, Any]) -> List[str]:
        """
        Get a list of validation errors without modifying the config.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check required sections
        if "llm" not in config:
            errors.append("Missing 'llm' configuration section")
        else:
            llm_config = config["llm"]
            
            # Check backend
            backend = llm_config.get("backend")
            if backend not in ["llmcpp", "ollama"]:
                errors.append(f"Invalid backend '{backend}', must be 'llmcpp' or 'ollama'")
            
            # Check model
            if not llm_config.get("model"):
                errors.append("Missing model specification in LLM configuration")
            
            # Check numeric ranges
            if backend == "llmcpp":
                numeric_checks = [
                    ("n_ctx", 512, 131072),
                    ("temperature", 0.0, 2.0),
                    ("top_p", 0.0, 1.0)
                ]
                
                for setting, min_val, max_val in numeric_checks:
                    value = llm_config.get(setting)
                    if value is not None:
                        try:
                            value = float(value)
                            if value < min_val or value > max_val:
                                errors.append(f"LLM setting '{setting}' value {value} out of range [{min_val}, {max_val}]")
                        except (ValueError, TypeError):
                            errors.append(f"LLM setting '{setting}' must be a number")
        
        return errors