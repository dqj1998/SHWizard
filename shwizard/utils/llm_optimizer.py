"""
LLM optimization utilities for shwizard.
Provides configuration and optimization settings for better LLM performance.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from shwizard.utils.logger import get_logger

logger = get_logger(__name__)


class LLMOptimizer:
    """Utility class for LLM optimization settings."""
    
    def __init__(self):
        self.config = self._load_optimization_config()
    
    def _load_optimization_config(self) -> Dict[str, Any]:
        """Load optimization configuration from YAML file."""
        try:
            config_path = Path(__file__).parent.parent / "data" / "llm_optimization.yaml"
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.debug("Loaded LLM optimization config")
                return config
            else:
                logger.warning("LLM optimization config not found, using defaults")
                return self._get_default_config()
                
        except Exception as e:
            logger.error(f"Error loading LLM optimization config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default optimization configuration."""
        return {
            "context": {
                "default_context": 4096,
                "min_context": 1024,
                "max_context": 8192,
                "warning_threshold": 0.8,
                "truncation_threshold": 0.9
            },
            "generation": {
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "max_tokens": 512,
                "min_tokens": 10,
                "stop_sequences": ["```", "---", "\n\n\n", "Explanation:", "Note:"]
            },
            "error_handling": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "exponential_backoff": True,
                "enable_context_management": True,
                "enable_cpu_fallback": True,
                "enable_reduced_parameters": True
            }
        }
    
    def get_optimal_context_size(self, available_memory_gb: float = 8.0) -> int:
        """Get optimal context size based on available memory."""
        context_config = self.config.get("context", {})
        
        if available_memory_gb < 4:
            return context_config.get("min_context", 1024)
        elif available_memory_gb < 8:
            return context_config.get("default_context", 4096) // 2
        else:
            return context_config.get("default_context", 4096)
    
    def get_generation_params(self, conservative: bool = False) -> Dict[str, Any]:
        """Get optimized generation parameters."""
        gen_config = self.config.get("generation", {})
        
        params = {
            "temperature": gen_config.get("temperature", 0.3),
            "top_p": gen_config.get("top_p", 0.9),
            "top_k": gen_config.get("top_k", 40),
            "repeat_penalty": gen_config.get("repeat_penalty", 1.1),
            "max_tokens": gen_config.get("max_tokens", 512),
            "stop": gen_config.get("stop_sequences", ["```", "---", "\n\n\n"])
        }
        
        if conservative:
            # More conservative settings for fallback scenarios
            params["temperature"] = 0.1
            params["max_tokens"] = min(params["max_tokens"], 128)
            params["top_p"] = 0.8
        
        return params
    
    def get_error_handling_config(self) -> Dict[str, Any]:
        """Get error handling configuration."""
        return self.config.get("error_handling", {})
    
    def get_platform_optimizations(self, platform: str = "metal") -> Dict[str, Any]:
        """Get platform-specific optimizations."""
        platform_config = self.config.get("platform", {})
        return platform_config.get(platform, {})
    
    def should_use_cpu_fallback(self, error: Exception) -> bool:
        """Determine if CPU fallback should be used for a given error."""
        error_str = str(error).lower()
        
        # GPU-related errors that benefit from CPU fallback
        gpu_error_patterns = [
            "cuda", "metal", "gpu", "device", "allocation",
            "out of memory", "memory", "vram"
        ]
        
        return any(pattern in error_str for pattern in gpu_error_patterns)
    
    def should_reduce_context(self, error: Exception) -> bool:
        """Determine if context reduction should be used for a given error."""
        error_str = str(error).lower()
        
        context_error_patterns = [
            "context", "length", "too long", "eof", "truncated"
        ]
        
        return any(pattern in error_str for pattern in context_error_patterns)
    
    def get_recommended_model(self, available_memory_gb: float) -> Optional[str]:
        """Get recommended model based on available memory."""
        models_config = self.config.get("models", {})
        
        if available_memory_gb < 4:
            models = models_config.get("low_memory", [])
        elif available_memory_gb < 8:
            models = models_config.get("medium_memory", [])
        else:
            models = models_config.get("high_memory", [])
        
        return models[0] if models else None
    
    def optimize_llm_config(
        self,
        base_config: Dict[str, Any],
        available_memory_gb: float = 8.0,
        platform: str = "metal"
    ) -> Dict[str, Any]:
        """
        Optimize LLM configuration based on system resources and platform.
        
        Args:
            base_config: Base LLM configuration
            available_memory_gb: Available system memory in GB
            platform: Target platform (metal, cuda, cpu)
            
        Returns:
            Optimized configuration dictionary
        """
        optimized_config = base_config.copy()
        
        # Optimize context size
        optimal_context = self.get_optimal_context_size(available_memory_gb)
        optimized_config["n_ctx"] = optimal_context
        
        # Apply platform-specific optimizations
        platform_opts = self.get_platform_optimizations(platform)
        
        if platform == "metal":
            # Conservative GPU layer count for Metal
            max_layers = platform_opts.get("max_gpu_layers", 50)
            if optimized_config.get("n_gpu_layers", -1) > max_layers:
                optimized_config["n_gpu_layers"] = max_layers
            
            # Reduce verbosity to minimize Metal warnings
            if platform_opts.get("reduce_warnings", True):
                optimized_config["verbose"] = False
        
        # Memory optimizations
        if available_memory_gb < 6:
            optimized_config["use_mlock"] = False
            optimized_config["n_gpu_layers"] = min(
                optimized_config.get("n_gpu_layers", 0), 20
            )
        
        logger.info(f"Optimized LLM config for {platform} with {available_memory_gb}GB memory")
        logger.debug(f"Context size: {optimized_config.get('n_ctx')}")
        logger.debug(f"GPU layers: {optimized_config.get('n_gpu_layers')}")
        
        return optimized_config


# Global optimizer instance
_optimizer = None

def get_llm_optimizer() -> LLMOptimizer:
    """Get the global LLM optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = LLMOptimizer()
    return _optimizer