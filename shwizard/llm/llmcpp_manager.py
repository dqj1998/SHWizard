import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import atexit

from shwizard.utils.platform_utils import get_data_directory
from shwizard.utils.hardware_detector import HardwareDetector
from shwizard.utils.platform_optimizer import PlatformOptimizer
from shwizard.utils.platform_error_handler import PlatformErrorHandler
from shwizard.utils.llm_optimizer import get_llm_optimizer
from shwizard.utils.logger import get_logger

logger = get_logger(__name__)


class LLMCppManager:
    """Manager for llama-cpp-python based LLM inference."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_name: str = "gemma-3-270m-Q8_0.gguf",
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize LLMCppManager.
        
        Args:
            model_path: Path to the model file or directory containing models
            model_name: Name of the model file to load
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 for auto)
            n_threads: Number of threads to use (None for auto-detect)
            verbose: Enable verbose logging
            **kwargs: Additional arguments passed to llama-cpp-python
        """
        self.model_name = model_name
        self.verbose = verbose
        self.llama_instance = None
        self._model_loaded = False
        
        # Set up model directory
        if model_path is None:
            model_path = get_data_directory() / "models"
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Get platform-specific optimizations
        self.platform_config = PlatformOptimizer.get_platform_config()
        self.error_handler = PlatformErrorHandler()
        self.llm_optimizer = get_llm_optimizer()
        
        # Auto-detect optimal settings if not specified
        if n_gpu_layers == -1 or n_threads is None:
            model_file_path = self.model_path / model_name
            model_size_mb = self._estimate_model_size(model_file_path)
            optimal_settings = HardwareDetector.get_optimal_settings(model_size_mb)
            
            if n_gpu_layers == -1:
                n_gpu_layers = optimal_settings["n_gpu_layers"]
            if n_threads is None:
                n_threads = optimal_settings["n_threads"]
        
        # Optimize context size based on available memory
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            optimal_context = self.llm_optimizer.get_optimal_context_size(available_memory_gb)
            if n_ctx == 2048 and optimal_context > 2048:
                n_ctx = optimal_context
                logger.info(f"Optimized context size to {n_ctx} based on available memory ({available_memory_gb:.1f}GB)")
        except ImportError:
            logger.debug("psutil not available, using default context size")
        
        # Store base configuration
        base_config = {
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_threads": n_threads,
            "verbose": verbose,
            **kwargs
        }
        
        # Apply optimizations
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            available_memory_gb = 8.0  # Default assumption
        
        platform_name = self.platform_config.get("preferred_backend", "cpu")
        self.config = self.llm_optimizer.optimize_llm_config(
            base_config, available_memory_gb, platform_name
        )
        
        # Apply additional platform-specific optimizations
        self._apply_platform_optimizations()
        
        # Register cleanup
        atexit.register(self.cleanup)
        
        logger.info(f"LLMCppManager initialized for model: {model_name}")
        logger.debug(f"Configuration: {self.config}")
    
    def _estimate_model_size(self, model_path: Path) -> int:
        """Estimate model size in MB."""
        if model_path.exists():
            return model_path.stat().st_size // (1024 * 1024)
        
        # Rough estimates based on model name patterns
        name_lower = self.model_name.lower()
        if "2b" in name_lower:
            return 1500  # ~1.5GB for 2B parameter models
        elif "3b" in name_lower:
            return 2000  # ~2GB for 3B parameter models
        elif "7b" in name_lower:
            return 4000  # ~4GB for 7B parameter models
        elif "8b" in name_lower:
            return 5000  # ~5GB for 8B parameter models
        elif "13b" in name_lower:
            return 8000  # ~8GB for 13B parameter models
        else:
            return 2000  # Default estimate
    
    def _apply_platform_optimizations(self):
        """Apply platform-specific optimizations to configuration."""
        memory_opts = self.platform_config.get("memory_optimization", {})
        threading_opts = self.platform_config.get("threading", {})
        
        # Apply memory optimizations
        if memory_opts.get("use_mmap", True):
            self.config["use_mmap"] = True
        if memory_opts.get("use_mlock", False):
            self.config["use_mlock"] = True
        
        # Apply threading optimizations
        if threading_opts.get("thread_affinity", False):
            # Note: llama-cpp-python doesn't directly support thread affinity
            # but we can optimize thread count
            pass
        
        # Apply GPU-specific settings
        gpu_settings = self.platform_config.get("gpu_settings", {})
        if self.platform_config.get("preferred_backend") == "metal":
            self.config["n_gpu_layers"] = min(self.config["n_gpu_layers"], 50)  # Conservative for Metal
            # Disable verbose logging to reduce Metal warnings
            if not self.verbose:
                self.config["verbose"] = False
        
        logger.debug(f"Applied platform optimizations: {self.platform_config['platform']}")
    
    def load_model(self, model_name: Optional[str] = None) -> bool:
        """
        Load a GGUF model file into memory.
        
        Args:
            model_name: Name of the model file to load (optional, uses default if None)
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if model_name:
            self.model_name = model_name
        
        model_file_path = self.model_path / self.model_name
        
        if not model_file_path.exists():
            logger.error(f"Model file not found: {model_file_path}")
            return False
        
        if not self._validate_gguf_file(model_file_path):
            logger.error(f"Invalid GGUF file: {model_file_path}")
            return False
        
        try:
            # Import llama_cpp here to avoid import errors if not installed
            import llama_cpp
            
            logger.info(f"Loading model: {model_file_path}")
            start_time = time.time()
            
            # Create Llama instance with configuration
            self.llama_instance = llama_cpp.Llama(
                model_path=str(model_file_path),
                **self.config
            )
            
            load_time = time.time() - start_time
            self._model_loaded = True
            
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            logger.info(f"Context size: {self.config['n_ctx']}, GPU layers: {self.config['n_gpu_layers']}")
            
            return True
            
        except ImportError:
            logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model_loaded = False
            
            # Handle model loading failure with platform-specific guidance
            fallback_info = self.error_handler.handle_model_loading_failure(e, model_file_path)
            
            # Log fallback options
            if fallback_info["fallback_options"]:
                logger.info("Fallback options available:")
                for option in fallback_info["fallback_options"]:
                    logger.info(f"  - {option}")
            
            # Log platform-specific guidance
            if fallback_info["platform_guidance"]:
                logger.info("Platform-specific guidance:")
                for guidance in fallback_info["platform_guidance"]:
                    logger.info(f"  - {guidance}")
            
            # Try fallback with CPU-only if GPU was attempted
            if self.config.get("n_gpu_layers", 0) > 0:
                logger.info("Attempting fallback to CPU-only inference...")
                return self._try_cpu_fallback(model_file_path)
            
            return False
    
    def _validate_gguf_file(self, file_path: Path) -> bool:
        """
        Validate that the file is a proper GGUF file.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            bool: True if valid GGUF file, False otherwise
        """
        try:
            # Check file extension
            if not file_path.suffix.lower() == '.gguf':
                logger.warning(f"File does not have .gguf extension: {file_path}")
                return False
            
            # Check file size (should be at least a few MB)
            file_size = file_path.stat().st_size
            if file_size < 1024 * 1024:  # Less than 1MB
                logger.error(f"File too small to be a valid model: {file_size} bytes")
                return False
            
            # Check GGUF magic number (first 4 bytes should be 'GGUF')
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                if magic != b'GGUF':
                    logger.error(f"Invalid GGUF magic number: {magic}")
                    return False
            
            logger.debug(f"GGUF file validation passed: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating GGUF file: {e}")
            return False
    
    def is_model_loaded(self) -> bool:
        """
        Check if a model is currently loaded and ready for inference.
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self._model_loaded and self.llama_instance is not None
    
    def unload_model(self):
        """Unload the current model and free memory."""
        if self.llama_instance:
            try:
                # llama-cpp-python doesn't have explicit unload, but we can delete the instance
                del self.llama_instance
                self.llama_instance = None
                self._model_loaded = False
                logger.info("Model unloaded successfully")
            except Exception as e:
                logger.error(f"Error unloading model: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dict containing model information
        """
        if not self.is_model_loaded():
            return {"loaded": False}
        
        model_file_path = self.model_path / self.model_name
        
        info = {
            "loaded": True,
            "model_name": self.model_name,
            "model_path": str(model_file_path),
            "file_size_mb": model_file_path.stat().st_size // (1024 * 1024),
            "context_size": self.config["n_ctx"],
            "gpu_layers": self.config["n_gpu_layers"],
            "threads": self.config["n_threads"],
            "platform": self.platform_config["platform"],
            "backend": self.platform_config.get("preferred_backend", "cpu")
        }
        
        # Add hardware-specific info
        if self.llama_instance:
            try:
                # Try to get additional info from llama instance if available
                pass  # llama-cpp-python doesn't expose much metadata
            except Exception:
                pass
        
        return info
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available GGUF models in the model directory.
        
        Returns:
            List of dictionaries containing model information
        """
        models = []
        
        try:
            for model_file in self.model_path.glob("*.gguf"):
                if self._validate_gguf_file(model_file):
                    model_info = {
                        "name": model_file.name,
                        "path": str(model_file),
                        "size_mb": model_file.stat().st_size // (1024 * 1024),
                        "modified": model_file.stat().st_mtime,
                        "loaded": model_file.name == self.model_name and self.is_model_loaded()
                    }
                    models.append(model_info)
            
            # Sort by modification time (newest first)
            models.sort(key=lambda x: x["modified"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
        
        return models
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text using the loaded model.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repeat_penalty: Repetition penalty
            stop: List of stop sequences
            stream: Whether to stream the response (not implemented yet)
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated text response
            
        Raises:
            RuntimeError: If no model is loaded
            ValueError: If prompt is too long for context window
        """
        if not self.is_model_loaded():
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        if not prompt.strip():
            logger.warning("Empty prompt provided")
            return ""
        
        # Check if prompt fits in context window
        prompt_length = len(prompt.split())  # Rough token estimate
        if prompt_length > self.config["n_ctx"] - max_tokens:
            # Truncate prompt to fit in context window
            max_prompt_tokens = self.config["n_ctx"] - max_tokens - 50  # Safety margin
            prompt_words = prompt.split()
            if len(prompt_words) > max_prompt_tokens:
                prompt = " ".join(prompt_words[-max_prompt_tokens:])
                logger.warning(f"Prompt truncated to fit context window: {max_prompt_tokens} tokens")
        
        try:
            logger.debug(f"Generating text with prompt length: {len(prompt)} chars")
            start_time = time.time()
            
            # Get optimized generation parameters
            optimized_params = self.llm_optimizer.get_generation_params()
            
            # Prepare generation parameters (user params override optimized ones)
            generation_params = {
                "max_tokens": max_tokens,
                "temperature": temperature if temperature != 0.7 else optimized_params.get("temperature", 0.3),
                "top_p": top_p if top_p != 0.9 else optimized_params.get("top_p", 0.9),
                "top_k": top_k if top_k != 40 else optimized_params.get("top_k", 40),
                "repeat_penalty": repeat_penalty if repeat_penalty != 1.1 else optimized_params.get("repeat_penalty", 1.1),
                "stop": stop or optimized_params.get("stop", []),
                **kwargs
            }
            
            # Generate response with error handling
            try:
                response = self.llama_instance(
                    prompt,
                    **generation_params
                )
            except Exception as gen_error:
                # Handle specific generation errors
                error_str = str(gen_error).lower()
                if self.llm_optimizer.should_reduce_context(gen_error):
                    logger.warning("EOF/truncation error, retrying with reduced parameters")
                    # Get conservative parameters from optimizer
                    conservative_params = self.llm_optimizer.get_generation_params(conservative=True)
                    conservative_params.update(kwargs)  # Preserve any additional kwargs
                    
                    # Use shorter prompt
                    short_prompt = prompt[-1000:] if len(prompt) > 1000 else prompt
                    response = self.llama_instance(short_prompt, **conservative_params)
                else:
                    raise gen_error
            
            generation_time = time.time() - start_time
            
            # Extract text from response
            if isinstance(response, dict):
                generated_text = response.get("choices", [{}])[0].get("text", "")
            else:
                generated_text = str(response)
            
            # Clean up the response
            generated_text = self._post_process_response(generated_text, prompt)
            
            logger.debug(f"Generated {len(generated_text)} chars in {generation_time:.2f}s")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            
            # Try to handle the error and provide fallback
            fallback_response = self.handle_inference_error(e, prompt)
            if fallback_response:
                logger.info("Fallback inference succeeded")
                return fallback_response
            
            raise RuntimeError(f"Text generation failed: {e}")
    
    def _post_process_response(self, response: str, original_prompt: str) -> str:
        """
        Post-process the generated response.
        
        Args:
            response: Raw response from the model
            original_prompt: Original input prompt
            
        Returns:
            str: Cleaned response text
        """
        if not response:
            return ""
        
        # Remove the original prompt if it's repeated in the response
        if response.startswith(original_prompt):
            response = response[len(original_prompt):].lstrip()
        
        # Clean up common artifacts
        response = response.strip()
        
        # Remove incomplete sentences at the end (optional)
        if response and not response.endswith(('.', '!', '?', '\n')):
            # Find the last complete sentence
            last_sentence_end = max(
                response.rfind('.'),
                response.rfind('!'),
                response.rfind('?'),
                response.rfind('\n')
            )
            if last_sentence_end > len(response) * 0.7:  # Only if we're not cutting too much
                response = response[:last_sentence_end + 1]
        
        return response
    
    def generate_with_context_management(
        self,
        prompt: str,
        max_tokens: int = 256,
        **kwargs
    ) -> str:
        """
        Generate text with automatic context length management.
        
        This method handles cases where the prompt is too long by implementing
        a sliding window approach or prompt compression.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated text response
        """
        # Estimate token count (rough approximation: 1 token ≈ 0.75 words)
        estimated_prompt_tokens = int(len(prompt.split()) * 1.33)
        available_context = self.config["n_ctx"] - max_tokens - 50  # Safety margin
        
        if estimated_prompt_tokens <= available_context:
            # Prompt fits, generate normally
            return self.generate(prompt, max_tokens=max_tokens, **kwargs)
        
        # Prompt is too long, implement sliding window
        logger.info(f"Prompt too long ({estimated_prompt_tokens} tokens), using context management")
        
        # Keep the most recent part of the prompt
        words = prompt.split()
        max_words = int(available_context * 0.75)  # Conservative estimate
        
        if len(words) > max_words:
            # Keep the beginning and end of the prompt
            keep_start = max_words // 4
            keep_end = max_words - keep_start
            
            truncated_prompt = (
                " ".join(words[:keep_start]) +
                " ... [content truncated] ... " +
                " ".join(words[-keep_end:])
            )
        else:
            truncated_prompt = prompt
        
        return self.generate(truncated_prompt, max_tokens=max_tokens, **kwargs)
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        
        Args:
            text: Input text
            
        Returns:
            int: Estimated number of tokens
        """
        # Simple estimation: roughly 1.33 tokens per word for English
        # This is a rough approximation and may not be accurate for all models
        words = len(text.split())
        return int(words * 1.33)
    
    def check_context_capacity(self, prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
        """
        Check if a prompt fits within the model's context window.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to reserve for generation
            
        Returns:
            Dict with capacity information
        """
        estimated_tokens = self.estimate_tokens(prompt)
        available_tokens = self.config["n_ctx"] - max_tokens
        
        return {
            "prompt_tokens": estimated_tokens,
            "max_generation_tokens": max_tokens,
            "available_tokens": available_tokens,
            "context_size": self.config["n_ctx"],
            "fits": estimated_tokens <= available_tokens,
            "utilization": estimated_tokens / self.config["n_ctx"],
            "needs_truncation": estimated_tokens > available_tokens
        }
    
    def _try_cpu_fallback(self, model_file_path: Path) -> bool:
        """
        Try loading the model with CPU-only configuration as fallback.
        
        Args:
            model_file_path: Path to the model file
            
        Returns:
            bool: True if CPU fallback succeeded, False otherwise
        """
        try:
            # Import llama_cpp here to avoid import errors if not installed
            import llama_cpp
            
            # Create CPU-only configuration
            cpu_config = self.config.copy()
            cpu_config["n_gpu_layers"] = 0
            
            # Reduce memory usage for CPU inference
            cpu_config["n_ctx"] = min(cpu_config.get("n_ctx", 2048), 1024)
            cpu_config["use_mlock"] = False
            
            logger.info(f"Attempting CPU-only fallback with reduced context: {cpu_config['n_ctx']}")
            
            # Create Llama instance with CPU-only configuration
            self.llama_instance = llama_cpp.Llama(
                model_path=str(model_file_path),
                **cpu_config
            )
            
            # Update configuration to reflect CPU-only mode
            self.config.update(cpu_config)
            self._model_loaded = True
            
            logger.info("CPU-only fallback successful")
            logger.warning("Running in CPU-only mode - inference will be slower")
            
            return True
            
        except Exception as fallback_error:
            logger.error(f"CPU fallback also failed: {fallback_error}")
            
            # Get fallback model recommendations
            try:
                import psutil
                available_memory = int(psutil.virtual_memory().available / (1024 * 1024))
            except ImportError:
                available_memory = 4096  # Conservative estimate
            
            recommendations = self.error_handler.get_fallback_model_recommendations(
                self.model_name, available_memory
            )
            
            if recommendations:
                logger.info("Consider trying these alternative models:")
                for rec in recommendations:
                    logger.info(f"  - {rec['display_name']}: {rec['reason']}")
            
            return False
    
    def handle_inference_error(self, error: Exception, prompt: str) -> Optional[str]:
        """
        Handle inference errors with platform-specific fallbacks.
        
        Args:
            error: The inference error
            prompt: The original prompt
            
        Returns:
            Fallback response or None if no fallback possible
        """
        logger.warning(f"Inference error: {error}")
        
        error_str = str(error).lower()
        
        # Context length exceeded
        if "context" in error_str or "length" in error_str:
            logger.info("Context length exceeded, trying with truncated prompt...")
            
            # Truncate prompt to fit in context
            max_prompt_length = self.config.get("n_ctx", 2048) // 2
            words = prompt.split()
            
            if len(words) > max_prompt_length:
                truncated_prompt = " ".join(words[-max_prompt_length:])
                logger.info(f"Truncated prompt from {len(words)} to {len(truncated_prompt.split())} words")
                
                try:
                    return self.generate(truncated_prompt, max_tokens=128)
                except Exception as retry_error:
                    logger.error(f"Truncated prompt also failed: {retry_error}")
        
        # Memory allocation errors
        elif "memory" in error_str or "allocation" in error_str:
            logger.info("Memory allocation error, trying with reduced parameters...")
            
            try:
                # Try with very conservative settings
                return self.generate(
                    prompt[:500],  # Very short prompt
                    max_tokens=64,  # Reduced output
                    n_ctx=512      # Minimal context
                )
            except Exception as retry_error:
                logger.error(f"Reduced parameter retry also failed: {retry_error}")
        
        # GPU-specific errors
        elif any(gpu_term in error_str for gpu_term in ["cuda", "metal", "gpu", "device"]):
            logger.info("GPU error detected, attempting CPU-only retry...")
            
            # If we haven't already fallen back to CPU, try it
            if self.config.get("n_gpu_layers", 0) > 0:
                original_layers = self.config["n_gpu_layers"]
                self.config["n_gpu_layers"] = 0
                
                try:
                    # Reload model with CPU-only
                    self.unload_model()
                    if self.load_model():
                        return self.generate(prompt, max_tokens=128)
                except Exception as cpu_retry_error:
                    logger.error(f"CPU retry failed: {cpu_retry_error}")
                    # Restore original configuration
                    self.config["n_gpu_layers"] = original_layers
        
        return None
    
    def cleanup(self):
        """Clean up resources."""
        if self.llama_instance:
            logger.debug("Cleaning up LLMCppManager resources")
            self.unload_model()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False