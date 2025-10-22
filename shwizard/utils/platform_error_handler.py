import platform
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from shwizard.utils.hardware_detector import HardwareDetector
from shwizard.utils.platform_optimizer import PlatformOptimizer
from shwizard.utils.logger import get_logger

logger = get_logger(__name__)


class PlatformErrorHandler:
    """Handles platform-specific errors and provides fallback solutions."""
    
    def __init__(self):
        """Initialize the platform error handler."""
        self.platform_config = PlatformOptimizer.get_platform_config()
        self.hardware_capabilities = HardwareDetector.detect_gpu_capabilities()
        self.platform_name = platform.system().lower()
    
    def handle_gpu_acceleration_failure(self, error: Exception, model_size_mb: int) -> Dict[str, Any]:
        """
        Handle GPU acceleration failures and provide fallback options.
        
        Args:
            error: The GPU acceleration error
            model_size_mb: Size of the model in MB
            
        Returns:
            Dict with fallback configuration and guidance
        """
        logger.warning(f"GPU acceleration failed: {error}")
        
        fallback_config = {
            "use_gpu": False,
            "n_gpu_layers": 0,
            "fallback_reason": str(error),
            "recommendations": [],
            "platform_specific_guidance": []
        }
        
        # Platform-specific fallback handling
        if self.platform_name == "darwin":  # macOS
            fallback_config.update(self._handle_macos_gpu_failure(error, model_size_mb))
        elif self.platform_name == "windows":
            fallback_config.update(self._handle_windows_gpu_failure(error, model_size_mb))
        elif self.platform_name == "linux":
            fallback_config.update(self._handle_linux_gpu_failure(error, model_size_mb))
        else:
            fallback_config["recommendations"].append("Use CPU-only inference")
        
        # General recommendations
        fallback_config["recommendations"].extend([
            "Consider using a smaller quantized model",
            "Reduce context window size (n_ctx) to save memory",
            "Close other applications to free up system resources"
        ])
        
        return fallback_config
    
    def _handle_macos_gpu_failure(self, error: Exception, model_size_mb: int) -> Dict[str, Any]:
        """Handle macOS-specific GPU failures."""
        config = {
            "platform_specific_guidance": [],
            "recommendations": []
        }
        
        error_str = str(error).lower()
        
        if "metal" in error_str:
            config["platform_specific_guidance"].extend([
                "Metal acceleration failed on macOS",
                "This may be due to insufficient GPU memory or Metal framework issues",
                "Falling back to CPU-only inference"
            ])
            
            # Check if it's Apple Silicon
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and "Apple" in result.stdout:
                    config["recommendations"].extend([
                        "On Apple Silicon, try reducing the model size",
                        "Consider using Q4_0 quantization instead of Q4_K_M",
                        "Monitor Activity Monitor for memory pressure"
                    ])
            except Exception:
                pass
        
        elif "memory" in error_str or "allocation" in error_str:
            config["platform_specific_guidance"].extend([
                "GPU memory allocation failed",
                "macOS unified memory may be under pressure"
            ])
            config["recommendations"].extend([
                "Close other applications to free unified memory",
                "Try a smaller model (2B parameters or less)",
                "Reduce n_ctx to 1024 or lower"
            ])
        
        return config
    
    def _handle_windows_gpu_failure(self, error: Exception, model_size_mb: int) -> Dict[str, Any]:
        """Handle Windows-specific GPU failures."""
        config = {
            "platform_specific_guidance": [],
            "recommendations": []
        }
        
        error_str = str(error).lower()
        
        if "cuda" in error_str:
            config["platform_specific_guidance"].extend([
                "CUDA acceleration failed on Windows",
                "This may be due to driver issues or insufficient VRAM"
            ])
            
            # Check CUDA availability
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    driver_version = result.stdout.strip()
                    config["recommendations"].extend([
                        f"NVIDIA driver version: {driver_version}",
                        "Ensure CUDA toolkit is properly installed",
                        "Try updating NVIDIA drivers to the latest version"
                    ])
                else:
                    config["recommendations"].extend([
                        "NVIDIA GPU not detected or drivers not installed",
                        "Install latest NVIDIA drivers from nvidia.com",
                        "Ensure GPU is properly seated and powered"
                    ])
            except Exception:
                config["recommendations"].append("Unable to check NVIDIA driver status")
        
        elif "memory" in error_str or "out of memory" in error_str:
            config["platform_specific_guidance"].extend([
                "GPU memory allocation failed",
                "Windows may have memory fragmentation issues"
            ])
            config["recommendations"].extend([
                "Close GPU-intensive applications (games, video editing)",
                "Restart the application to clear GPU memory",
                "Consider using MSI Afterburner to monitor GPU memory usage"
            ])
        
        elif "opencl" in error_str:
            config["platform_specific_guidance"].extend([
                "OpenCL acceleration failed",
                "This may be due to missing OpenCL runtime"
            ])
            config["recommendations"].extend([
                "Install Intel OpenCL runtime for CPU acceleration",
                "Update GPU drivers which include OpenCL support"
            ])
        
        return config
    
    def _handle_linux_gpu_failure(self, error: Exception, model_size_mb: int) -> Dict[str, Any]:
        """Handle Linux-specific GPU failures."""
        config = {
            "platform_specific_guidance": [],
            "recommendations": []
        }
        
        error_str = str(error).lower()
        
        if "cuda" in error_str:
            config["platform_specific_guidance"].extend([
                "CUDA acceleration failed on Linux",
                "This may be due to driver or CUDA toolkit issues"
            ])
            
            # Check CUDA installation
            cuda_paths = ["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda"]
            cuda_found = any(Path(p).exists() for p in cuda_paths)
            
            if cuda_found:
                config["recommendations"].extend([
                    "CUDA toolkit appears to be installed",
                    "Check CUDA version compatibility with your GPU",
                    "Verify LD_LIBRARY_PATH includes CUDA libraries",
                    "Try: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
                ])
            else:
                config["recommendations"].extend([
                    "CUDA toolkit not found in standard locations",
                    "Install CUDA toolkit from NVIDIA developer website",
                    "Ensure nvidia-driver and nvidia-cuda-toolkit packages are installed"
                ])
        
        elif "rocm" in error_str or "hip" in error_str:
            config["platform_specific_guidance"].extend([
                "ROCm acceleration failed",
                "This may be due to AMD GPU driver issues"
            ])
            config["recommendations"].extend([
                "Install ROCm drivers for AMD GPUs",
                "Check if your AMD GPU is supported by ROCm",
                "Verify ROCm installation: rocm-smi"
            ])
        
        elif "permission" in error_str or "access" in error_str:
            config["platform_specific_guidance"].extend([
                "GPU access permission denied",
                "This may be due to user permissions or driver issues"
            ])
            config["recommendations"].extend([
                "Add user to 'video' group: sudo usermod -a -G video $USER",
                "Check GPU device permissions: ls -la /dev/dri/",
                "Restart after adding user to video group"
            ])
        
        return config
    
    def handle_model_loading_failure(self, error: Exception, model_path: Path) -> Dict[str, Any]:
        """
        Handle model loading failures with platform-specific guidance.
        
        Args:
            error: The model loading error
            model_path: Path to the model file
            
        Returns:
            Dict with fallback options and guidance
        """
        logger.error(f"Model loading failed: {error}")
        
        fallback_config = {
            "success": False,
            "error": str(error),
            "fallback_options": [],
            "platform_guidance": [],
            "troubleshooting_steps": []
        }
        
        error_str = str(error).lower()
        
        # File-related errors
        if "no such file" in error_str or "not found" in error_str:
            fallback_config["fallback_options"].extend([
                "Download the model automatically",
                "Try a different model from the registry",
                "Check model path configuration"
            ])
            fallback_config["troubleshooting_steps"].extend([
                f"Verify model file exists: {model_path}",
                "Check file permissions",
                "Ensure model path is correctly configured"
            ])
        
        # Memory-related errors
        elif "memory" in error_str or "allocation" in error_str:
            fallback_config["fallback_options"].extend([
                "Try a smaller quantized model",
                "Reduce context window size",
                "Use CPU-only inference"
            ])
            
            # Platform-specific memory guidance
            if self.platform_name == "darwin":
                fallback_config["platform_guidance"].extend([
                    "macOS unified memory may be under pressure",
                    "Close other applications to free memory",
                    "Check Activity Monitor for memory usage"
                ])
            elif self.platform_name == "windows":
                fallback_config["platform_guidance"].extend([
                    "Windows may have memory fragmentation",
                    "Restart the application to clear memory",
                    "Check Task Manager for memory usage"
                ])
            else:  # Linux
                fallback_config["platform_guidance"].extend([
                    "Check available memory: free -h",
                    "Consider increasing swap space",
                    "Monitor memory usage: htop or top"
                ])
        
        # Corruption or format errors
        elif "corrupt" in error_str or "invalid" in error_str or "format" in error_str:
            fallback_config["fallback_options"].extend([
                "Re-download the model file",
                "Try a different model format",
                "Verify model file integrity"
            ])
            fallback_config["troubleshooting_steps"].extend([
                "Check model file size and compare with expected size",
                "Verify GGUF file format with file command",
                "Re-download from original source"
            ])
        
        return fallback_config
    
    def get_fallback_model_recommendations(self, failed_model: str, available_memory_mb: int) -> List[Dict[str, Any]]:
        """
        Get fallback model recommendations based on hardware constraints.
        
        Args:
            failed_model: Name of the model that failed to load
            available_memory_mb: Available system memory in MB
            
        Returns:
            List of recommended fallback models
        """
        try:
            from shwizard.llm.model_registry import ModelRegistry
            registry = ModelRegistry()
            
            # Get models that fit in available memory
            suitable_models = registry.list_models_by_size(available_memory_mb // 2)  # Use half available memory
            
            # Sort by size (smallest first for better compatibility)
            suitable_models.sort(key=lambda m: m.size_mb)
            
            recommendations = []
            for model in suitable_models[:3]:  # Top 3 recommendations
                compatibility = registry.get_model_compatibility(model.name)
                
                recommendations.append({
                    "name": model.name,
                    "display_name": model.display_name,
                    "size_mb": model.size_mb,
                    "description": model.description,
                    "compatible": compatibility["compatible"],
                    "reason": f"Smaller model ({model.size_mb}MB) that should fit in available memory"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get fallback recommendations: {e}")
            return [{
                "name": "gemma-3-270m-Q8_0.gguf",
                "display_name": "Gemma 3 270M (Fallback)",
                "size_mb": 1500,
                "description": "Small, reliable fallback model",
                "compatible": True,
                "reason": "Default fallback model for compatibility"
            }]
    
    def get_platform_troubleshooting_guide(self) -> Dict[str, List[str]]:
        """Get platform-specific troubleshooting guide."""
        guides = {
            "general": [
                "Ensure sufficient system memory (RAM) is available",
                "Close unnecessary applications to free resources",
                "Try restarting the application",
                "Check system logs for additional error details"
            ]
        }
        
        if self.platform_name == "darwin":  # macOS
            guides["macos"] = [
                "Check Activity Monitor for memory pressure",
                "Ensure Xcode Command Line Tools are installed",
                "For Apple Silicon: verify Metal framework is available",
                "Try running with Rosetta 2 if on Apple Silicon",
                "Check Console app for system-level errors"
            ]
        
        elif self.platform_name == "windows":
            guides["windows"] = [
                "Update NVIDIA/AMD drivers to latest version",
                "Install Microsoft Visual C++ Redistributables",
                "Check Windows Event Viewer for system errors",
                "Run Windows Memory Diagnostic tool",
                "Ensure Windows Defender isn't blocking the application"
            ]
        
        elif self.platform_name == "linux":
            guides["linux"] = [
                "Check dmesg for kernel-level errors",
                "Verify GPU drivers are properly installed",
                "Ensure user has proper permissions for GPU access",
                "Check library dependencies with ldd",
                "Monitor system resources with htop/top"
            ]
        
        return guides
    
    def create_error_report(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive error report for debugging.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            
        Returns:
            Comprehensive error report
        """
        report = {
            "timestamp": __import__("time").time(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "context": context
            },
            "hardware": self.hardware_capabilities,
            "platform_config": self.platform_config,
            "troubleshooting": self.get_platform_troubleshooting_guide(),
            "recommendations": []
        }
        
        # Add specific recommendations based on error type
        error_str = str(error).lower()
        
        if "gpu" in error_str or "cuda" in error_str or "metal" in error_str:
            gpu_fallback = self.handle_gpu_acceleration_failure(error, context.get("model_size_mb", 2000))
            report["recommendations"].extend(gpu_fallback["recommendations"])
        
        elif "memory" in error_str or "allocation" in error_str:
            available_memory = context.get("available_memory_mb", 8192)
            fallback_models = self.get_fallback_model_recommendations(
                context.get("model_name", "unknown"), available_memory
            )
            report["fallback_models"] = fallback_models
        
        return report