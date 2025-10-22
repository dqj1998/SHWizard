from shwizard.llm.llmcpp_manager import LLMCppManager
from shwizard.llm.model_downloader import ModelDownloader
from shwizard.llm.model_registry import ModelRegistry

# Keep OllamaManager import for backward compatibility
try:
    from shwizard.llm.ollama_manager import OllamaManager
    __all__ = ["LLMCppManager", "ModelDownloader", "ModelRegistry", "OllamaManager"]
except ImportError:
    __all__ = ["LLMCppManager", "ModelDownloader", "ModelRegistry"]
