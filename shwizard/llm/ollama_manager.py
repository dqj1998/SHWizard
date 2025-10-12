import os
import subprocess
import time
import platform
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Optional
from shwizard.utils.platform_utils import get_data_directory
from shwizard.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaManager:
    OLLAMA_DOWNLOAD_URLS = {
        "linux": "https://github.com/ollama/ollama/releases/download/v0.3.12/ollama-linux-amd64",
        "macos": "https://github.com/ollama/ollama/releases/download/v0.3.12/ollama-darwin",
        "windows": "https://github.com/ollama/ollama/releases/download/v0.3.12/ollama-windows-amd64.zip"
    }
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        embedded: bool = True,
        auto_download: bool = True,
        install_path: Optional[Path] = None
    ):
        self.base_url = base_url
        self.embedded = embedded
        self.auto_download = auto_download
        
        if install_path is None:
            install_path = get_data_directory() / "ollama"
        self.install_path: Path = install_path
        self.install_path.mkdir(parents=True, exist_ok=True)
        
        self.ollama_process = None
        self.ollama_binary = self._get_ollama_binary_path()
    
    def _get_ollama_binary_path(self) -> Path:
        os_type = platform.system().lower()
        if os_type == "darwin":
            os_type = "macos"
        elif os_type not in ["linux", "windows"]:
            os_type = "linux"
        
        if os_type == "windows":
            return self.install_path / "ollama.exe"
        else:
            return self.install_path / "ollama"
    
    def _get_download_url(self) -> Optional[str]:
        os_type = platform.system().lower()
        if os_type == "darwin":
            os_type = "macos"
        elif os_type not in ["linux", "windows"]:
            os_type = "linux"
        
        return self.OLLAMA_DOWNLOAD_URLS.get(os_type)
    
    def is_ollama_installed(self) -> bool:
        if subprocess.run(["which", "ollama"], capture_output=True).returncode == 0:
            return True
        
        return self.ollama_binary.exists()
    
    def download_ollama(self) -> bool:
        if not self.auto_download:
            logger.warning("Auto-download is disabled")
            return False
        
        download_url = self._get_download_url()
        if not download_url:
            logger.error("Unsupported platform for Ollama download")
            return False
        
        logger.info(f"Downloading Ollama from {download_url}")
        
        try:
            response = requests.get(download_url, stream=True, timeout=300)
            response.raise_for_status()
            
            os_type = platform.system().lower()
            if os_type == "windows" or "darwin" in os_type:
                temp_file = self.install_path / "ollama_download.zip"
            else:
                temp_file = self.ollama_binary
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.debug(f"Download progress: {progress:.1f}%")
            
            if os_type != "linux":
                if zipfile.is_zipfile(temp_file):
                    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                        zip_ref.extractall(self.install_path)
                    temp_file.unlink()
            
            if self.ollama_binary.exists():
                os.chmod(self.ollama_binary, 0o755)
            
            logger.info("Ollama downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download Ollama: {e}")
            return False
    
    def start_ollama_server(self) -> bool:
        if self.is_server_running():
            logger.info("Ollama server is already running")
            return True
        
        if not self.embedded:
            logger.info("Using system Ollama installation")
            return False
        
        if not self.is_ollama_installed():
            logger.info("Ollama not found, attempting to download...")
            if not self.download_ollama():
                logger.error("Failed to download Ollama")
                return False
        
        try:
            if self.ollama_binary.exists():
                ollama_cmd = str(self.ollama_binary)
            else:
                ollama_cmd = "ollama"
            
            logger.info(f"Starting Ollama server with: {ollama_cmd}")
            
            self.ollama_process = subprocess.Popen(
                [ollama_cmd, "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "OLLAMA_HOST": self.base_url.replace("http://", "")}
            )
            
            time.sleep(3)
            
            for _ in range(10):
                if self.is_server_running():
                    logger.info("Ollama server started successfully")
                    return True
                time.sleep(1)
            
            logger.error("Ollama server failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start Ollama server: {e}")
            return False
    
    def is_server_running(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def stop_ollama_server(self):
        if self.ollama_process:
            try:
                self.ollama_process.terminate()
                self.ollama_process.wait(timeout=5)
                logger.info("Ollama server stopped")
            except:
                self.ollama_process.kill()
    
    def pull_model(self, model_name: str) -> bool:
        if not self.is_server_running():
            logger.error("Ollama server is not running")
            return False
        
        try:
            logger.info(f"Pulling model: {model_name}")
            
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600
            )
            
            for line in response.iter_lines():
                if line:
                    logger.debug(line.decode('utf-8'))
            
            logger.info(f"Model {model_name} pulled successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def list_models(self):
        if not self.is_server_running():
            return []
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        
        return []
    
    def ensure_model_available(self, model_name: str) -> bool:
        models = self.list_models()
        model_names = [m.get("name", "") for m in models]
        
        if model_name in model_names or any(model_name in name for name in model_names):
            logger.info(f"Model {model_name} is already available")
            return True
        
        logger.info(f"Model {model_name} not found, pulling...")
        return self.pull_model(model_name)
