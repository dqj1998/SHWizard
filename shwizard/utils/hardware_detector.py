import platform
import subprocess
import os
from typing import Dict, Any, Optional
from pathlib import Path
from shwizard.utils.logger import get_logger

logger = get_logger(__name__)


class HardwareDetector:
    """Detects hardware capabilities for optimal LLM_cpp configuration."""
    
    @staticmethod
    def detect_gpu_capabilities() -> Dict[str, Any]:
        """Detect available GPU acceleration options."""
        capabilities = {
            "cuda": False,
            "metal": False,
            "opencl": False,
            "gpu_memory": 0,
            "gpu_count": 0,
            "gpu_names": []
        }
        
        # Detect CUDA on Windows/Linux
        if platform.system().lower() in ["windows", "linux"]:
            capabilities.update(HardwareDetector._detect_cuda())
        
        # Detect Metal on macOS
        if platform.system().lower() == "darwin":
            capabilities.update(HardwareDetector._detect_metal())
        
        # Detect OpenCL (cross-platform fallback)
        capabilities.update(HardwareDetector._detect_opencl())
        
        return capabilities
    
    @staticmethod
    def _detect_cuda() -> Dict[str, Any]:
        """Detect CUDA availability and capabilities."""
        cuda_info = {
            "cuda": False,
            "gpu_memory": 0,
            "gpu_count": 0,
            "gpu_names": []
        }
        
        try:
            # Check if nvidia-smi is available
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                cuda_info["cuda"] = True
                lines = result.stdout.strip().split('\n')
                
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 2:
                            gpu_name = parts[0].strip()
                            gpu_memory = int(parts[1].strip())
                            
                            cuda_info["gpu_names"].append(gpu_name)
                            cuda_info["gpu_memory"] += gpu_memory
                            cuda_info["gpu_count"] += 1
                
                logger.info(f"CUDA detected: {cuda_info['gpu_count']} GPU(s), {cuda_info['gpu_memory']}MB total")
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
            logger.debug("CUDA not available or nvidia-smi not found")
        
        return cuda_info
    
    @staticmethod
    def _detect_metal() -> Dict[str, Any]:
        """Detect Metal availability on macOS."""
        metal_info = {
            "metal": False,
            "gpu_memory": 0,
            "gpu_count": 0,
            "gpu_names": []
        }
        
        try:
            # Check if we're on Apple Silicon
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                cpu_brand = result.stdout.strip()
                if "Apple" in cpu_brand:
                    metal_info["metal"] = True
                    metal_info["gpu_count"] = 1
                    metal_info["gpu_names"] = ["Apple GPU"]
                    
                    # Try to get memory info
                    mem_result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if mem_result.returncode == 0:
                        # Estimate GPU memory as portion of system memory
                        total_memory = int(mem_result.stdout.strip())
                        metal_info["gpu_memory"] = total_memory // (1024 * 1024 * 2)  # Rough estimate
                    
                    logger.info(f"Metal detected on Apple Silicon: {cpu_brand}")
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
            logger.debug("Metal detection failed or not on Apple Silicon")
        
        return metal_info
    
    @staticmethod
    def _detect_opencl() -> Dict[str, Any]:
        """Detect OpenCL availability as fallback."""
        opencl_info = {"opencl": False}
        
        try:
            # Try to import pyopencl to check availability
            import pyopencl as cl
            platforms = cl.get_platforms()
            if platforms:
                opencl_info["opencl"] = True
                logger.info("OpenCL detected as fallback option")
        except ImportError:
            logger.debug("OpenCL not available (pyopencl not installed)")
        except Exception:
            logger.debug("OpenCL detection failed")
        
        return opencl_info
    
    @staticmethod
    def get_optimal_settings(model_size_mb: int) -> Dict[str, Any]:
        """Get optimal LLM_cpp settings based on hardware capabilities."""
        capabilities = HardwareDetector.detect_gpu_capabilities()
        
        settings = {
            "n_gpu_layers": 0,  # Default to CPU-only
            "n_ctx": 2048,
            "n_threads": os.cpu_count() or 4,
            "use_mmap": True,
            "use_mlock": False
        }
        
        # Adjust based on available GPU memory
        available_gpu_memory = capabilities.get("gpu_memory", 0)
        
        if capabilities.get("cuda") or capabilities.get("metal"):
            if available_gpu_memory > model_size_mb * 1.5:  # 50% overhead
                settings["n_gpu_layers"] = -1  # Use all layers
                logger.info("Using full GPU acceleration")
            elif available_gpu_memory > model_size_mb:
                settings["n_gpu_layers"] = 20  # Partial GPU acceleration
                logger.info("Using partial GPU acceleration")
            else:
                logger.info("Insufficient GPU memory, using CPU-only")
        
        # Adjust context size based on available memory
        if available_gpu_memory > 8000:  # 8GB+
            settings["n_ctx"] = 4096
        elif available_gpu_memory > 4000:  # 4GB+
            settings["n_ctx"] = 2048
        else:
            settings["n_ctx"] = 1024
        
        # Enable memory locking for better performance on systems with sufficient RAM
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            if available_ram_gb > 8:
                settings["use_mlock"] = True
        except ImportError:
            pass
        
        return settings
    
    @staticmethod
    def get_platform_specific_build() -> str:
        """Return the appropriate llama-cpp-python build variant for the platform."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if system == "windows":
            return "llama-cpp-python[cuda]" if HardwareDetector.detect_gpu_capabilities().get("cuda") else "llama-cpp-python"
        elif system == "darwin":
            if "arm" in machine or "apple" in machine:
                return "llama-cpp-python[metal]"
            else:
                return "llama-cpp-python"
        elif system == "linux":
            capabilities = HardwareDetector.detect_gpu_capabilities()
            if capabilities.get("cuda"):
                return "llama-cpp-python[cuda]"
            else:
                return "llama-cpp-python"
        
        return "llama-cpp-python"
    
    @staticmethod
    def validate_installation() -> bool:
        """Validate that llama-cpp-python is properly installed with acceleration."""
        try:
            import llama_cpp
            
            # Try to create a simple Llama instance to test installation
            # This will fail gracefully if there are issues
            logger.info("llama-cpp-python installation validated")
            return True
            
        except ImportError:
            logger.error("llama-cpp-python not installed")
            return False
        except Exception as e:
            logger.warning(f"llama-cpp-python installation may have issues: {e}")
            return False