import requests
import time
import atexit
from typing import List, Optional, Dict, Any
from shwizard.llm.ollama_manager import OllamaManager
from shwizard.utils.prompt_utils import build_command_prompt, build_explain_prompt, parse_command_response
from shwizard.utils.logger import get_logger

logger = get_logger(__name__)


class AIService:
    def __init__(
        self,
        ollama_manager: Optional[OllamaManager] = None,
        model: str = "gemma3:4b",
        base_url: str = "http://localhost:11435",
        timeout: int = 60,
        max_retries: int = 3
    ):
        self.ollama_manager = ollama_manager or OllamaManager(base_url=base_url)
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._initialized = False
        atexit.register(self.shutdown)
    
    def initialize(self) -> bool:
        if self._initialized:
            return True
        
        if not self.ollama_manager.is_server_running():
            logger.info("Starting Ollama server...")
            if not self.ollama_manager.start_ollama_server():
                logger.error("Failed to start Ollama server")
                return False
        
        if not self.ollama_manager.ensure_model_available(self.model):
            logger.error(f"Failed to ensure model {self.model} is available")
            return False
        
        self._initialized = True
        logger.info("AI service initialized successfully")
        return True
    
    def generate_commands(
        self,
        user_query: str,
        context: Dict[str, Any],
        history_commands: Optional[List[str]] = None
    ) -> List[str]:
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
                response, feedback = self._call_ollama(prompt)
                if response:
                    commands = parse_command_response(response)
                    if commands:
                        logger.info(f"Generated {len(commands)} command(s)")
                        # Update feedback in the history manager
                        command_id = context.get("command_id")
                        if command_id is not None:
                            history_manager = HistoryManager()
                            history_manager.add_feedback(command_id, feedback)
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
        if not self._initialized:
            if not self.initialize():
                return None
        
        prompt = build_explain_prompt(
            command=command,
            os_type=context.get("os", "linux"),
            shell_type=context.get("shell", "bash")
        )
        
        try:
            return self._call_ollama(prompt)
        except Exception as e:
            logger.error(f"Error explaining command: {e}")
            return None
    
    def _call_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            response = data.get("response", "").strip()
            # Determine feedback based on response content
            feedback = 1 if "error" not in response.lower() else -1
            return response, feedback
        except requests.exceptions.Timeout:
            logger.error("Request to Ollama timed out")
            return None, 0
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Ollama failed: {e}")
            return None, 0
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama: {e}")
            return None, 0
    
    def shutdown(self):
        if self.ollama_manager and self.ollama_manager.embedded:
            self.ollama_manager.stop_ollama_server()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.shutdown()
        except Exception:
            pass
        return False
