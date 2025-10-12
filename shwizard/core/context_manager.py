from typing import Dict, List, Any
from shwizard.utils.platform_utils import (
    get_os_type,
    get_shell_type,
    get_current_directory,
    get_system_info,
    get_installed_tools
)
from shwizard.utils.logger import get_logger

logger = get_logger(__name__)


class ContextManager:
    def __init__(self, collect_tools: bool = False):
        self.collect_tools = collect_tools
        self._cached_tools: List[str] = []
    
    def get_context(self) -> Dict[str, Any]:
        context = {
            "os": get_os_type(),
            "shell": get_shell_type(),
            "cwd": get_current_directory(),
        }
        
        if self.collect_tools:
            if not self._cached_tools:
                self._cached_tools = get_installed_tools()
            context["installed_tools"] = self._cached_tools
        else:
            context["installed_tools"] = []
        
        logger.debug(f"Current context: {context}")
        return context
    
    def get_full_system_info(self) -> Dict[str, Any]:
        return get_system_info()
    
    def refresh_tools_cache(self):
        self._cached_tools = get_installed_tools()
