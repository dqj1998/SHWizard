from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from shwizard.utils.hardware_detector import HardwareDetector
from shwizard.utils.platform_optimizer import PlatformOptimizer
from shwizard.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Information about a GGUF model."""
    name: str
    display_name: str
    repo_id: str
    filename: str
    size_mb: int
    quantization: str
    context_length: int
    architecture: str
    description: str
    min_ram_mb: int = 0
    min_vram_mb: int = 0
    recommended_gpu_layers: int = -1
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ModelRegistry:
    """Registry of available GGUF models with selection logic."""
    
    def __init__(self):
        """Initialize the model registry."""
        self.models = self._initialize_default_models()
        self.hardware_capabilities = HardwareDetector.detect_gpu_capabilities()
        self.platform_config = PlatformOptimizer.get_platform_config()
        
        logger.info(f"ModelRegistry initialized with {len(self.models)} models")
    
    def _initialize_default_models(self) -> Dict[str, ModelInfo]:
        """Initialize the registry with default models."""
        models = {}
        
        # Small models (good for low-end hardware)
        models["gemma-3-270m-q8_0"] = ModelInfo(
            name="gemma-3-270m-q8_0",
            display_name="Gemma 3 270M (Q8_0)",
            repo_id="ggml-org/gemma-3-270m-GGUF",
            filename="gemma-3-270m-Q8_0.gguf",
            size_mb=270,
            quantization="Q8_0",
            context_length=8192,
            architecture="gemma3",
            description="Ultra-lightweight 270M parameter model, perfect for resource-constrained environments",
            min_ram_mb=512,
            min_vram_mb=0,
            recommended_gpu_layers=16,
            tags=["tiny", "ultra-fast", "gemma", "default"]
        )
        
        models["gemma-2-2b-it-q4_k_m"] = ModelInfo(
            name="gemma-2-2b-it-q4_k_m",
            display_name="Gemma 2 2B Instruct (Q4_K_M)",
            repo_id="bartowski/gemma-2-2b-it-GGUF",
            filename="gemma-2-2b-it-Q4_K_M.gguf",
            size_mb=1500,
            quantization="Q4_K_M",
            context_length=8192,
            architecture="gemma2",
            description="Fast and efficient 2B parameter model, good for basic tasks",
            min_ram_mb=2048,
            min_vram_mb=0,
            recommended_gpu_layers=32,
            tags=["small", "fast", "instruct", "gemma"]
        )
        
        models["qwen2-1.5b-instruct-q4_k_m"] = ModelInfo(
            name="qwen2-1.5b-instruct-q4_k_m",
            display_name="Qwen2 1.5B Instruct (Q4_K_M)",
            repo_id="Qwen/Qwen2-1.5B-Instruct-GGUF",
            filename="qwen2-1_5b-instruct-q4_k_m.gguf",
            size_mb=1200,
            quantization="Q4_K_M",
            context_length=32768,
            architecture="qwen2",
            description="Very fast 1.5B parameter model with large context window",
            min_ram_mb=1536,
            min_vram_mb=0,
            recommended_gpu_layers=28,
            tags=["small", "fast", "instruct", "qwen", "large-context"]
        )
        
        # Medium models (balanced performance)
        models["llama-3.2-3b-instruct-q4_k_m"] = ModelInfo(
            name="llama-3.2-3b-instruct-q4_k_m",
            display_name="Llama 3.2 3B Instruct (Q4_K_M)",
            repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
            filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            size_mb=2000,
            quantization="Q4_K_M",
            context_length=131072,
            architecture="llama",
            description="Balanced 3B parameter model with excellent instruction following",
            min_ram_mb=2560,
            min_vram_mb=0,
            recommended_gpu_layers=36,
            tags=["medium", "balanced", "instruct", "llama", "large-context"]
        )
        
        models["phi-3-mini-4k-instruct-q4"] = ModelInfo(
            name="phi-3-mini-4k-instruct-q4",
            display_name="Phi-3 Mini 4K Instruct (Q4)",
            repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
            filename="Phi-3-mini-4k-instruct-q4.gguf",
            size_mb=2300,
            quantization="Q4_0",
            context_length=4096,
            architecture="phi3",
            description="Microsoft's efficient 3.8B parameter model optimized for instruction following",
            min_ram_mb=2816,
            min_vram_mb=0,
            recommended_gpu_layers=32,
            tags=["medium", "microsoft", "instruct", "phi"]
        )
        
        # Large models (high performance)
        models["llama-3.1-8b-instruct-q4_k_m"] = ModelInfo(
            name="llama-3.1-8b-instruct-q4_k_m",
            display_name="Llama 3.1 8B Instruct (Q4_K_M)",
            repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            size_mb=5000,
            quantization="Q4_K_M",
            context_length=131072,
            architecture="llama",
            description="High-quality 8B parameter model with excellent reasoning capabilities",
            min_ram_mb=6144,
            min_vram_mb=0,
            recommended_gpu_layers=32,
            tags=["large", "high-quality", "instruct", "llama", "large-context"]
        )
        
        # Add more quantization variants for popular models
        models["gemma-2-2b-it-q8_0"] = ModelInfo(
            name="gemma-2-2b-it-q8_0",
            display_name="Gemma 2 2B Instruct (Q8_0)",
            repo_id="bartowski/gemma-2-2b-it-GGUF",
            filename="gemma-2-2b-it-Q8_0.gguf",
            size_mb=2200,
            quantization="Q8_0",
            context_length=8192,
            architecture="gemma2",
            description="Higher quality 2B model with 8-bit quantization",
            min_ram_mb=2816,
            min_vram_mb=0,
            recommended_gpu_layers=32,
            tags=["small", "high-quality", "instruct", "gemma"]
        )
        
        return models
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self.models.get(model_name)
    
    def list_all_models(self) -> List[ModelInfo]:
        """List all available models."""
        return list(self.models.values())
    
    def list_models_by_tag(self, tag: str) -> List[ModelInfo]:
        """List models that have a specific tag."""
        return [model for model in self.models.values() if tag in model.tags]
    
    def list_models_by_size(self, max_size_mb: int) -> List[ModelInfo]:
        """List models that fit within a size limit."""
        return [model for model in self.models.values() if model.size_mb <= max_size_mb]
    
    def get_recommended_models(
        self,
        available_ram_mb: Optional[int] = None,
        available_vram_mb: Optional[int] = None,
        prefer_speed: bool = True,
        prefer_quality: bool = False,
        max_models: int = 3
    ) -> List[ModelInfo]:
        """
        Get recommended models based on hardware capabilities and preferences.
        
        Args:
            available_ram_mb: Available system RAM in MB
            available_vram_mb: Available GPU VRAM in MB
            prefer_speed: Prefer faster models
            prefer_quality: Prefer higher quality models
            max_models: Maximum number of models to return
            
        Returns:
            List of recommended models, sorted by suitability
        """
        # Auto-detect hardware if not provided
        if available_ram_mb is None or available_vram_mb is None:
            try:
                import psutil
                if available_ram_mb is None:
                    available_ram_mb = int(psutil.virtual_memory().available / (1024 * 1024))
            except ImportError:
                available_ram_mb = available_ram_mb or 8192  # Default assumption
            
            if available_vram_mb is None:
                available_vram_mb = self.hardware_capabilities.get("gpu_memory", 0)
        
        logger.info(f"Finding recommendations for RAM: {available_ram_mb}MB, VRAM: {available_vram_mb}MB")
        
        # Filter models that fit in available memory
        suitable_models = []
        for model in self.models.values():
            # Check if model fits in RAM
            if model.min_ram_mb > available_ram_mb:
                continue
            
            # Calculate suitability score
            score = self._calculate_model_score(
                model, available_ram_mb, available_vram_mb, prefer_speed, prefer_quality
            )
            
            suitable_models.append((model, score))
        
        # Sort by score (higher is better) and return top models
        suitable_models.sort(key=lambda x: x[1], reverse=True)
        
        recommended = [model for model, score in suitable_models[:max_models]]
        
        if recommended:
            logger.info(f"Recommended {len(recommended)} models: {[m.name for m in recommended]}")
        else:
            logger.warning("No suitable models found for current hardware")
        
        return recommended
    
    def _calculate_model_score(
        self,
        model: ModelInfo,
        available_ram_mb: int,
        available_vram_mb: int,
        prefer_speed: bool,
        prefer_quality: bool
    ) -> float:
        """Calculate a suitability score for a model."""
        score = 0.0
        
        # Base score based on model size efficiency
        ram_utilization = model.min_ram_mb / available_ram_mb
        if ram_utilization <= 0.5:
            score += 10  # Plenty of headroom
        elif ram_utilization <= 0.7:
            score += 7   # Good fit
        elif ram_utilization <= 0.9:
            score += 4   # Tight fit
        else:
            score += 1   # Very tight fit
        
        # GPU acceleration bonus
        if available_vram_mb > 0 and model.size_mb <= available_vram_mb:
            score += 5  # Can fit entirely in VRAM
        elif available_vram_mb > model.size_mb * 0.5:
            score += 3  # Can fit partially in VRAM
        
        # Speed preference
        if prefer_speed:
            if "fast" in model.tags or model.size_mb < 2000:
                score += 3
            if "small" in model.tags:
                score += 2
        
        # Quality preference
        if prefer_quality:
            if "high-quality" in model.tags or "Q8" in model.quantization:
                score += 3
            if model.size_mb > 3000:  # Larger models generally better quality
                score += 2
        
        # Architecture bonuses
        if model.architecture in ["llama", "gemma2"]:
            score += 1  # Well-supported architectures
        
        # Context length bonus
        if model.context_length >= 32768:
            score += 1  # Large context is useful
        
        # Instruction tuning bonus
        if "instruct" in model.tags:
            score += 2
        
        return score
    
    def get_fallback_model(self) -> Optional[ModelInfo]:
        """Get the most compatible fallback model for any hardware."""
        # Find the smallest, most compatible model
        fallback_candidates = [
            model for model in self.models.values()
            if ("small" in model.tags or "tiny" in model.tags) and model.min_ram_mb <= 2048
        ]
        
        if fallback_candidates:
            # Return the smallest one
            return min(fallback_candidates, key=lambda m: m.size_mb)
        
        # If no small models, return any model with minimal requirements
        all_models = list(self.models.values())
        if all_models:
            return min(all_models, key=lambda m: m.min_ram_mb)
        
        return None
    
    def select_optimal_model(
        self,
        task_type: str = "general",
        hardware_constraints: Optional[Dict[str, int]] = None
    ) -> Optional[ModelInfo]:
        """
        Select the optimal model for a specific task and hardware.
        
        Args:
            task_type: Type of task ("general", "coding", "reasoning", "fast")
            hardware_constraints: Dict with "max_ram_mb", "max_vram_mb" etc.
            
        Returns:
            Optimal model for the given constraints
        """
        constraints = hardware_constraints or {}
        max_ram = constraints.get("max_ram_mb", 16384)
        max_vram = constraints.get("max_vram_mb", self.hardware_capabilities.get("gpu_memory", 0))
        
        # Task-specific preferences
        task_preferences = {
            "general": {"prefer_speed": True, "prefer_quality": False},
            "coding": {"prefer_speed": False, "prefer_quality": True},
            "reasoning": {"prefer_speed": False, "prefer_quality": True},
            "fast": {"prefer_speed": True, "prefer_quality": False}
        }
        
        prefs = task_preferences.get(task_type, {"prefer_speed": True, "prefer_quality": False})
        
        recommended = self.get_recommended_models(
            available_ram_mb=max_ram,
            available_vram_mb=max_vram,
            prefer_speed=prefs["prefer_speed"],
            prefer_quality=prefs["prefer_quality"],
            max_models=1
        )
        
        return recommended[0] if recommended else self.get_fallback_model()
    
    def add_custom_model(self, model_info: ModelInfo):
        """Add a custom model to the registry."""
        self.models[model_info.name] = model_info
        logger.info(f"Added custom model: {model_info.name}")
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model from the registry."""
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Removed model from registry: {model_name}")
            return True
        return False
    
    def export_registry(self, file_path: Path):
        """Export the model registry to a JSON file."""
        try:
            registry_data = {
                "models": {
                    name: {
                        "name": model.name,
                        "display_name": model.display_name,
                        "repo_id": model.repo_id,
                        "filename": model.filename,
                        "size_mb": model.size_mb,
                        "quantization": model.quantization,
                        "context_length": model.context_length,
                        "architecture": model.architecture,
                        "description": model.description,
                        "min_ram_mb": model.min_ram_mb,
                        "min_vram_mb": model.min_vram_mb,
                        "recommended_gpu_layers": model.recommended_gpu_layers,
                        "tags": model.tags
                    }
                    for name, model in self.models.items()
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            logger.info(f"Registry exported to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export registry: {e}")
    
    def import_registry(self, file_path: Path):
        """Import models from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                registry_data = json.load(f)
            
            for name, model_data in registry_data.get("models", {}).items():
                model_info = ModelInfo(**model_data)
                self.models[name] = model_info
            
            logger.info(f"Registry imported from: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to import registry: {e}")
    
    def get_model_compatibility(self, model_name: str) -> Dict[str, Any]:
        """Get compatibility information for a model."""
        model = self.get_model_info(model_name)
        if not model:
            return {"compatible": False, "reason": "Model not found"}
        
        # Check hardware compatibility
        available_ram = 8192  # Default assumption
        try:
            import psutil
            available_ram = int(psutil.virtual_memory().available / (1024 * 1024))
        except ImportError:
            pass
        
        available_vram = self.hardware_capabilities.get("gpu_memory", 0)
        
        compatibility = {
            "compatible": True,
            "model": model.name,
            "requirements": {
                "min_ram_mb": model.min_ram_mb,
                "min_vram_mb": model.min_vram_mb,
                "model_size_mb": model.size_mb
            },
            "available": {
                "ram_mb": available_ram,
                "vram_mb": available_vram
            },
            "gpu_acceleration": available_vram >= model.size_mb * 0.5,
            "warnings": []
        }
        
        # Check compatibility issues
        if model.min_ram_mb > available_ram:
            compatibility["compatible"] = False
            compatibility["warnings"].append(f"Insufficient RAM: need {model.min_ram_mb}MB, have {available_ram}MB")
        
        if available_ram < model.min_ram_mb * 1.2:
            compatibility["warnings"].append("RAM usage will be very high, performance may be affected")
        
        if available_vram == 0:
            compatibility["warnings"].append("No GPU acceleration available, will use CPU-only inference")
        elif available_vram < model.size_mb:
            compatibility["warnings"].append("Model won't fit entirely in VRAM, will use partial GPU acceleration")
        
        return compatibility