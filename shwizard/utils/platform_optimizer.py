import platform
import os
from typing import Dict, Any, Optional
from pathlib import Path
from shwizard.utils.hardware_detector import HardwareDetector
from shwizard.utils.logger import get_logger

logger = get_logger(__name__)


class PlatformOptimizer:
    """Platform-specific optimizations for LLM_cpp performance."""
    
    @staticmethod
    def get_platform_config() -> Dict[str, Any]:
        """Get platform-specific configuration for optimal performance."""
        system = platform.system().lower()
        
        if system == "windows":
            return PlatformOptimizer._get_windows_config()
        elif system == "darwin":
            return PlatformOptimizer._get_macos_config()
        elif system == "linux":
            return PlatformOptimizer._get_linux_config()
        else:
            return PlatformOptimizer._get_default_config()
    
    @staticmethod
    def _get_windows_config() -> Dict[str, Any]:
        """Windows-specific optimizations."""
        config = {
            "platform": "windows",
            "file_extension": ".exe",
            "path_separator": "\\",
            "preferred_backend": "cuda" if HardwareDetector.detect_gpu_capabilities().get("cuda") else "cpu",
            "memory_optimization": {
                "use_mmap": True,
                "use_mlock": False,  # Windows doesn't support mlock well
                "numa_optimization": False
            },
            "threading": {
                "n_threads": min(os.cpu_count() or 4, 8),  # Limit threads on Windows
                "thread_affinity": False
            },
            "gpu_settings": {
                "cuda_visible_devices": None,  # Let CUDA handle device selection
                "gpu_split": None
            }
        }
        
        # Windows-specific CUDA optimizations
        if config["preferred_backend"] == "cuda":
            config["gpu_settings"].update({
                "cuda_memory_fraction": 0.8,  # Reserve some GPU memory for system
                "cuda_allow_growth": True
            })
        
        return config
    
    @staticmethod
    def _get_macos_config() -> Dict[str, Any]:
        """macOS-specific optimizations."""
        machine = platform.machine().lower()
        is_apple_silicon = "arm" in machine or "apple" in machine
        
        config = {
            "platform": "macos",
            "file_extension": "",
            "path_separator": "/",
            "preferred_backend": "metal" if is_apple_silicon else "cpu",
            "memory_optimization": {
                "use_mmap": True,
                "use_mlock": True,  # macOS supports mlock
                "numa_optimization": False
            },
            "threading": {
                "n_threads": os.cpu_count() or 4,
                "thread_affinity": False
            },
            "gpu_settings": {
                "metal_performance_shaders": is_apple_silicon,
                "unified_memory": is_apple_silicon
            }
        }
        
        # Apple Silicon specific optimizations
        if is_apple_silicon:
            config["memory_optimization"]["unified_memory_optimization"] = True
            config["gpu_settings"].update({
                "metal_memory_fraction": 0.7,  # Conservative for unified memory
                "metal_allow_fallback": True
            })
        
        return config
    
    @staticmethod
    def _get_linux_config() -> Dict[str, Any]:
        """Linux-specific optimizations."""
        capabilities = HardwareDetector.detect_gpu_capabilities()
        
        config = {
            "platform": "linux",
            "file_extension": "",
            "path_separator": "/",
            "preferred_backend": "cuda" if capabilities.get("cuda") else "cpu",
            "memory_optimization": {
                "use_mmap": True,
                "use_mlock": True,
                "numa_optimization": True  # Linux supports NUMA optimization
            },
            "threading": {
                "n_threads": os.cpu_count() or 4,
                "thread_affinity": True  # Linux supports thread affinity
            },
            "gpu_settings": {
                "cuda_visible_devices": None,
                "gpu_split": None
            }
        }
        
        # CUDA optimizations for Linux
        if config["preferred_backend"] == "cuda":
            config["gpu_settings"].update({
                "cuda_memory_fraction": 0.9,  # More aggressive on Linux
                "cuda_allow_growth": True,
                "cuda_memory_pool": True
            })
        
        # Check for ROCm support (AMD GPUs on Linux)
        if PlatformOptimizer._detect_rocm():
            config["gpu_settings"]["rocm_support"] = True
            if not capabilities.get("cuda"):
                config["preferred_backend"] = "rocm"
        
        return config
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Default configuration for unknown platforms."""
        return {
            "platform": "unknown",
            "file_extension": "",
            "path_separator": "/",
            "preferred_backend": "cpu",
            "memory_optimization": {
                "use_mmap": True,
                "use_mlock": False,
                "numa_optimization": False
            },
            "threading": {
                "n_threads": min(os.cpu_count() or 4, 4),
                "thread_affinity": False
            },
            "gpu_settings": {}
        }
    
    @staticmethod
    def _detect_rocm() -> bool:
        """Detect ROCm availability on Linux."""
        try:
            rocm_paths = [
                "/opt/rocm",
                "/usr/lib/x86_64-linux-gnu/rocm",
                "/usr/local/rocm"
            ]
            
            for path in rocm_paths:
                if Path(path).exists():
                    return True
            
            # Check for ROCm environment variables
            if os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"):
                return True
                
        except Exception:
            pass
        
        return False
    
    @staticmethod
    def optimize_for_model_size(model_size_mb: int, platform_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Optimize settings based on model size and platform."""
        if platform_config is None:
            platform_config = PlatformOptimizer.get_platform_config()
        
        hardware_settings = HardwareDetector.get_optimal_settings(model_size_mb)
        
        # Merge platform-specific optimizations with hardware-based settings
        optimized_config = {
            **platform_config,
            "model_optimization": {
                "model_size_mb": model_size_mb,
                "n_gpu_layers": hardware_settings["n_gpu_layers"],
                "n_ctx": hardware_settings["n_ctx"],
                "n_threads": hardware_settings["n_threads"],
                "use_mmap": hardware_settings["use_mmap"],
                "use_mlock": hardware_settings["use_mlock"]
            }
        }
        
        # Platform-specific adjustments
        if platform_config["platform"] == "windows" and model_size_mb > 4000:
            # Large models on Windows may need special handling
            optimized_config["model_optimization"]["use_mlock"] = False
            optimized_config["model_optimization"]["n_threads"] = min(
                optimized_config["model_optimization"]["n_threads"], 6
            )
        
        elif platform_config["platform"] == "macos" and platform_config["gpu_settings"].get("unified_memory"):
            # Unified memory on Apple Silicon needs careful management
            if model_size_mb > 2000:
                optimized_config["model_optimization"]["n_gpu_layers"] = min(
                    optimized_config["model_optimization"]["n_gpu_layers"], 30
                )
        
        return optimized_config
    
    @staticmethod
    def get_installation_command() -> str:
        """Get the appropriate installation command for the current platform."""
        build_variant = HardwareDetector.get_platform_specific_build()
        
        if build_variant == "llama-cpp-python":
            return "pip install llama-cpp-python"
        else:
            return f"pip install {build_variant}"
    
    @staticmethod
    def validate_platform_support() -> Dict[str, bool]:
        """Validate platform support and available features."""
        validation = {
            "platform_supported": True,
            "gpu_acceleration": False,
            "memory_optimization": False,
            "threading_optimization": False
        }
        
        platform_config = PlatformOptimizer.get_platform_config()
        
        # Check GPU acceleration
        if platform_config["preferred_backend"] in ["cuda", "metal", "rocm"]:
            validation["gpu_acceleration"] = True
        
        # Check memory optimization features
        if platform_config["memory_optimization"]["use_mlock"]:
            validation["memory_optimization"] = True
        
        # Check threading optimization
        if platform_config["threading"]["thread_affinity"]:
            validation["threading_optimization"] = True
        
        return validation