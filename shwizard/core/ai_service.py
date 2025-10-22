import time
import atexit
from typing import List, Optional, Dict, Any, Union
from shwizard.llm.llmcpp_manager import LLMCppManager
from shwizard.llm.model_downloader import ModelDownloader
from shwizard.llm.model_registry import ModelRegistry
from shwizard.utils.platform_error_handler import PlatformErrorHandler
from shwizard.utils.prompt_utils import build_command_prompt, build_explain_prompt, parse_command_response
from shwizard.utils.logger import get_logger

# Ollama support has been removed in favor of LLM_cpp
# This is kept for backward compatibility warnings
OLLAMA_AVAILABLE = False
OllamaManager = None

logger = get_logger(__name__)


class AIService:
    def __init__(
        self,
        llm_manager: Optional[Union[LLMCppManager, "OllamaManager"]] = None,
        model: str = "gemma-3-270m-q8_0.gguf",
        backend: str = "llmcpp",
        # Legacy Ollama parameters for backward compatibility
        ollama_manager: Optional["OllamaManager"] = None,
        base_url: str = "http://localhost:11435",
        timeout: int = 60,
        max_retries: int = 3,
        # LLM_cpp specific parameters
        auto_download: bool = True,
        **kwargs
    ):
        """
        Initialize AIService with LLM_cpp or Ollama backend.
        
        Args:
            llm_manager: LLMCppManager or OllamaManager instance
            model: Model name to use
            backend: Backend type ("llmcpp" or "ollama")
            ollama_manager: Legacy parameter for backward compatibility
            base_url: Ollama base URL (legacy)
            timeout: Request timeout
            max_retries: Maximum retry attempts
            auto_download: Automatically download models if not available
            **kwargs: Additional parameters passed to LLMCppManager
        """
        self.backend = backend
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.auto_download = auto_download
        self._initialized = False
        
        # Initialize model registry and downloader for LLM_cpp
        if backend == "llmcpp":
            self.model_registry = ModelRegistry()
            self.model_downloader = ModelDownloader()
            self.error_handler = PlatformErrorHandler()
            
            # Use provided LLMCppManager or create new one
            if isinstance(llm_manager, LLMCppManager):
                self.llm_manager = llm_manager
            else:
                self.llm_manager = LLMCppManager(
                    model_name=model,
                    **kwargs
                )
            
            # Legacy compatibility - if ollama_manager is provided, warn about deprecation
            if ollama_manager is not None:
                logger.warning("ollama_manager parameter is deprecated, use llm_manager with LLMCppManager instead")
        
        else:  # Ollama backend - no longer supported
            logger.warning("Ollama backend is no longer supported. Automatically switching to LLM_cpp backend.")
            logger.info("Your configuration will be automatically migrated to use LLM_cpp.")
            
            # Force switch to LLM_cpp backend
            self.backend = "llmcpp"
            self.model_registry = ModelRegistry()
            self.model_downloader = ModelDownloader()
            self.error_handler = PlatformErrorHandler()
            
            # Use provided LLMCppManager or create new one with migrated settings
            if isinstance(llm_manager, LLMCppManager):
                self.llm_manager = llm_manager
            else:
                # Create LLMCppManager with default settings
                self.llm_manager = LLMCppManager(
                    model_name="gemma-3-270m-q8_0.gguf",  # Default model
                    **kwargs
                )
        
        atexit.register(self.shutdown)
        logger.info(f"AIService initialized with {backend} backend, model: {model}")
    
    def initialize(self) -> bool:
        """Initialize the AI service and load the model."""
        if self._initialized:
            return True
        
        try:
            if self.backend == "llmcpp":
                return self._initialize_llmcpp()
            else:
                return self._initialize_ollama()
        except Exception as e:
            logger.error(f"Failed to initialize AI service: {e}")
            return False
    
    def _initialize_llmcpp(self) -> bool:
        """Initialize LLM_cpp backend."""
        logger.info("Initializing LLM_cpp backend...")
        
        # Check if model is already loaded
        if self.llm_manager.is_model_loaded():
            self._initialized = True
            logger.info("LLM_cpp model already loaded")
            return True
        
        # Check if model file exists locally
        model_path = self.llm_manager.model_path / self.model
        if not model_path.exists() and self.auto_download:
            logger.info(f"Model not found locally, attempting to download: {self.model}")
            
            # Try to download the model
            downloaded_path = self.model_downloader.download_model(self.model)
            if not downloaded_path:
                # Try to get a recommended fallback model
                logger.warning(f"Failed to download {self.model}, trying fallback model")
                fallback_model = self.model_registry.get_fallback_model()
                if fallback_model:
                    self.model = fallback_model.filename  # Use filename instead of name
                    downloaded_path = self.model_downloader.download_model(self.model)
                
                if not downloaded_path:
                    logger.error("Failed to download any compatible model")
                    
                    # Get platform-specific troubleshooting guidance
                    troubleshooting = self.error_handler.get_platform_troubleshooting_guide()
                    logger.info("Troubleshooting suggestions:")
                    for category, suggestions in troubleshooting.items():
                        logger.info(f"  {category.title()}:")
                        for suggestion in suggestions[:3]:  # Show top 3 suggestions
                            logger.info(f"    - {suggestion}")
                    
                    return False
        
        # Load the model
        if self.llm_manager.load_model(self.model):
            self._initialized = True
            logger.info("LLM_cpp service initialized successfully")
            
            # Log model info
            model_info = self.llm_manager.get_model_info()
            logger.info(f"Loaded model: {model_info.get('model_name', 'unknown')}")
            logger.info(f"Context size: {model_info.get('context_size', 'unknown')}")
            logger.info(f"GPU layers: {model_info.get('gpu_layers', 'unknown')}")
            
            return True
        else:
            logger.error("Failed to load model")
            
            # Provide fallback model recommendations
            try:
                import psutil
                available_memory = int(psutil.virtual_memory().available / (1024 * 1024))
            except ImportError:
                available_memory = 4096
            
            recommendations = self.error_handler.get_fallback_model_recommendations(
                self.model, available_memory
            )
            
            if recommendations:
                logger.info("Consider trying these alternative models:")
                for rec in recommendations:
                    logger.info(f"  - {rec['display_name']}: {rec['reason']}")
            
            return False
    
    def _initialize_ollama(self) -> bool:
        """Legacy Ollama initialization - redirects to LLM_cpp."""
        logger.warning("Ollama backend is deprecated. Redirecting to LLM_cpp initialization.")
        return self._initialize_llmcpp()
    
    def generate_commands(
        self,
        user_query: str,
        context: Dict[str, Any],
        history_commands: Optional[List[str]] = None
    ) -> List[str]:
        """Generate shell commands based on user query and context."""
        if not self._initialized:
            if not self.initialize():
                logger.error("AI service not initialized")
                return []
        
        prompt = build_command_prompt(
            user_query=user_query,
            os_type=context.get("os", "linux"),
            cwd=context.get("cwd", "~"),
            shell_type=context.get("shell", "bash"),
            installed_tools=context.get("installed_tools", []),
            history_commands=history_commands
        )
        
        for attempt in range(self.max_retries):
            try:
                if self.backend == "llmcpp":
                    response = self._call_llmcpp(prompt)
                else:
                    response = self._call_ollama(prompt)
                
                if response:
                    commands = parse_command_response(response)
                    if commands:
                        logger.info(f"Generated {len(commands)} command(s)")
                        return commands
                
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error generating commands (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
        
        logger.error("Failed to generate commands after all retries")
        return []
    
    def explain_command(self, command: str, context: Dict[str, Any]) -> Optional[str]:
        """Explain what a shell command does."""
        if not self._initialized:
            if not self.initialize():
                return None
        
        prompt = build_explain_prompt(
            command=command,
            os_type=context.get("os", "linux"),
            shell_type=context.get("shell", "bash")
        )
        
        try:
            if self.backend == "llmcpp":
                return self._call_llmcpp(prompt)
            else:
                return self._call_ollama(prompt)
        except Exception as e:
            logger.error(f"Error explaining command: {e}")
            return None
    
    def _call_llmcpp(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """Call LLM_cpp for text generation."""
        try:
            # Combine system prompt with user prompt if provided
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Debug: log the actual prompt being sent
            logger.debug(f"Sending prompt to LLM: {repr(full_prompt[:200])}...")
            
            # Check context capacity before generation
            capacity_info = self.llm_manager.check_context_capacity(full_prompt, max_tokens=512)
            
            if not capacity_info["fits"]:
                logger.warning(f"Prompt too long ({capacity_info['prompt_tokens']} tokens), using context management")
                # Use context management for long prompts
                response = self.llm_manager.generate_with_context_management(
                    prompt=full_prompt,
                    max_tokens=64,  # Reduced for long prompts
                    temperature=0.1,
                    top_p=0.9,
                    stop=["\n", "Explanation:"]
                )
            else:
                # Generate response using LLMCppManager
                response = self.llm_manager.generate(
                    prompt=full_prompt,
                    max_tokens=64,  # Reasonable limit for command generation
                    temperature=0.7,  # Higher temperature for better generation
                    top_p=0.9,
                    stop=[]  # No stop sequences
                )
            
            logger.debug(f"Raw LLM response: {repr(response)}")
            if response and response.strip():
                return response.strip()
            else:
                logger.warning("Empty response from LLM")
                return None
            
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "context" in error_msg or "length" in error_msg:
                logger.warning("Context length error, retrying with shorter prompt")
                try:
                    # Try with a much shorter prompt
                    short_prompt = full_prompt[-500:] if len(full_prompt) > 500 else full_prompt
                    response = self.llm_manager.generate(
                        prompt=short_prompt,
                        max_tokens=64,
                        temperature=0.1,
                        top_p=0.9,
                        stop=["\n", "Explanation:"]
                    )
                    return response.strip() if response else None
                except Exception:
                    pass
            
            logger.error(f"LLM_cpp runtime error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling LLM_cpp: {e}")
            return None
    
    def _call_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """Legacy Ollama support - redirects to LLM_cpp."""
        logger.warning("Ollama backend calls are being redirected to LLM_cpp")
        return self._call_llmcpp(prompt, system_prompt)
    
    def shutdown(self):
        """Shutdown the AI service and cleanup resources."""
        try:
            if self.backend == "llmcpp":
                if hasattr(self, 'llm_manager') and self.llm_manager:
                    self.llm_manager.cleanup()
                    logger.debug("LLM_cpp resources cleaned up")
            else:
                # Legacy Ollama cleanup - no longer needed
                logger.debug("Legacy Ollama cleanup skipped (using LLM_cpp)")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        if not self._initialized:
            # Try to initialize if not already done
            if not self.initialize():
                return {
                    "initialized": False,
                    "backend": self.backend,
                    "model": self.model,
                    "error": "Failed to initialize AI service"
                }
        
        if self.backend == "llmcpp":
            info = self.llm_manager.get_model_info()
            
            # Check if model is actually loaded
            if not info.get("loaded", False):
                return {
                    "initialized": False,
                    "backend": "llmcpp",
                    "model": self.model,
                    "error": "Model not loaded"
                }
            
            info["backend"] = "llmcpp"
            info["initialized"] = True
            return info
        else:
            # Legacy Ollama info
            return {
                "initialized": True,
                "backend": "ollama",
                "model": self.model,
                "base_url": getattr(self, 'base_url', 'unknown')
            }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List available models for the current backend."""
        if self.backend == "llmcpp":
            # Get cached models
            cached_models = self.model_downloader.list_cached_models()
            
            # Get registry models with compatibility info
            registry_models = []
            for model_info in self.model_registry.list_all_models():
                compatibility = self.model_registry.get_model_compatibility(model_info.name)
                registry_models.append({
                    "name": model_info.name,
                    "display_name": model_info.display_name,
                    "size_mb": model_info.size_mb,
                    "description": model_info.description,
                    "cached": any(m["name"] == model_info.name for m in cached_models),
                    "compatible": compatibility["compatible"],
                    "tags": model_info.tags
                })
            
            return registry_models
        else:
            # Legacy Ollama model listing
            if hasattr(self, 'ollama_manager'):
                return self.ollama_manager.list_models()
            return []
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model."""
        if self.backend == "llmcpp":
            try:
                # Check if model is available
                model_info = self.model_registry.get_model_info(model_name)
                if not model_info:
                    logger.error(f"Model not found in registry: {model_name}")
                    return False
                
                # Download model if not cached
                model_path = self.llm_manager.model_path / model_name
                if not model_path.exists() and self.auto_download:
                    logger.info(f"Downloading model: {model_name}")
                    downloaded_path = self.model_downloader.download_model(model_name)
                    if not downloaded_path:
                        logger.error(f"Failed to download model: {model_name}")
                        return False
                
                # Load the new model
                if self.llm_manager.load_model(model_name):
                    self.model = model_name
                    logger.info(f"Switched to model: {model_name}")
                    return True
                else:
                    logger.error(f"Failed to load model: {model_name}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error switching model: {e}")
                return False
        else:
            # Legacy Ollama model switching - redirect to LLM_cpp
            logger.warning("Ollama model switching redirected to LLM_cpp")
            # Force backend to LLM_cpp and try switching
            self.backend = "llmcpp"
            return self.switch_model(model_name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.shutdown()
        except Exception:
            pass
        return False
