import os
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from urllib.parse import urlparse
import requests

from shwizard.utils.platform_utils import get_data_directory
from shwizard.utils.logger import get_logger

# Import huggingface_hub conditionally
try:
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

    # Define dummy classes for type hints
    class RepositoryNotFoundError(Exception):
        pass

    class RevisionNotFoundError(Exception):
        pass


logger = get_logger(__name__)


class ModelDownloader:
    """Handles downloading and caching of GGUF model files."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize ModelDownloader.

        Args:
            cache_dir: Directory to cache downloaded models (default: ~/.shwizard/models/)
        """
        if cache_dir is None:
            cache_dir = get_data_directory() / "models"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file to track downloaded models
        self.metadata_file = self.cache_dir / "models_metadata.json"
        self.metadata = self._load_metadata()

        logger.info(f"ModelDownloader initialized with cache dir: {self.cache_dir}")

    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata from cache."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")

        return {"models": {}, "last_updated": time.time()}

    def _save_metadata(self):
        """Save model metadata to cache."""
        try:
            self.metadata["last_updated"] = time.time()
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def download_model(
        self,
        model_name: str,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        force_download: bool = False,
    ) -> Optional[Path]:
        """
        Download a GGUF model file.

        Args:
            model_name: Name/identifier for the model
            repo_id: Hugging Face repository ID (e.g., "microsoft/DialoGPT-medium")
            filename: Specific filename to download from the repo
            progress_callback: Optional callback for download progress (bytes_downloaded, total_bytes)
            force_download: Force re-download even if file exists

        Returns:
            Path to downloaded model file, or None if download failed
        """
        # Determine download parameters
        if repo_id is None:
            repo_id, filename = self._resolve_model_source(model_name)

        if filename is None:
            filename = model_name if model_name.endswith(".gguf") else f"{model_name}.gguf"

        local_path = self.cache_dir / filename

        # Check if model already exists and is valid
        if not force_download and self._is_model_cached(model_name, local_path):
            logger.info(f"Model already cached: {local_path}")
            return local_path

        logger.info(f"Downloading model: {repo_id}/{filename}")

        try:
            # Download from Hugging Face Hub
            downloaded_path = self._download_from_huggingface(
                repo_id, filename, local_path, progress_callback
            )

            if downloaded_path and self._verify_download(downloaded_path):
                # Update metadata
                self._update_model_metadata(
                    model_name,
                    {
                        "repo_id": repo_id,
                        "filename": filename,
                        "local_path": str(downloaded_path),
                        "download_time": time.time(),
                        "file_size": downloaded_path.stat().st_size,
                        "checksum": self._calculate_checksum(downloaded_path),
                    },
                )

                logger.info(f"Model downloaded successfully: {downloaded_path}")
                return downloaded_path
            else:
                logger.error("Download verification failed")
                return None

        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            return None

    def _resolve_model_source(self, model_name: str) -> tuple[str, str]:
        """
        Resolve model name to Hugging Face repo and filename.

        Args:
            model_name: Model name to resolve

        Returns:
            Tuple of (repo_id, filename)
        """
        # Default model mappings
        model_mappings = {
            "gemma-3-270m-q8_0.gguf": ("ggml-org/gemma-3-270m-GGUF", "gemma-3-270m-Q8_0.gguf"),
            "gemma-3-270m.gguf": ("ggml-org/gemma-3-270m-GGUF", "gemma-3-270m-Q8_0.gguf"),
            "gemma-2-2b-it-q4_k_m.gguf": (
                "bartowski/gemma-2-2b-it-GGUF",
                "gemma-2-2b-it-Q4_K_M.gguf",
            ),
            "llama-3.2-3b-instruct-q4_k_m.gguf": (
                "bartowski/Llama-3.2-3B-Instruct-GGUF",
                "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            ),
            "llama-3.1-8b-instruct-q4_k_m.gguf": (
                "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            ),
            "phi-3-mini-4k-instruct-q4.gguf": (
                "microsoft/Phi-3-mini-4k-instruct-gguf",
                "Phi-3-mini-4k-instruct-q4.gguf",
            ),
            "qwen2-1.5b-instruct-q4_k_m.gguf": (
                "Qwen/Qwen2-1.5B-Instruct-GGUF",
                "qwen2-1_5b-instruct-q4_k_m.gguf",
            ),
        }

        # Try direct lookup first (case-sensitive)
        if model_name in model_mappings:
            return model_mappings[model_name]
        
        # Try case-insensitive lookup
        model_name_lower = model_name.lower()
        for key, value in model_mappings.items():
            if key.lower() == model_name_lower:
                return value

        # Try to parse as repo_id/filename
        if "/" in model_name and model_name.count("/") >= 1:
            parts = model_name.split("/")
            if len(parts) >= 2:
                repo_id = "/".join(parts[:-1])
                filename = parts[-1]
                return repo_id, filename

        # Default fallback - use the first model in mappings (gemma-3-270m)
        logger.warning(f"Model {model_name} not found in mappings, using default fallback")
        return model_mappings["gemma-3-270m-q8_0.gguf"]

    def _download_from_huggingface(
        self,
        repo_id: str,
        filename: str,
        local_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Optional[Path]:
        """Download file from Hugging Face Hub."""
        if not HF_HUB_AVAILABLE:
            logger.error("huggingface_hub not available. Install with: pip install huggingface-hub")
            return None

        try:
            # Use huggingface_hub to download
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.cache_dir.parent),
                local_dir=str(self.cache_dir),
                local_dir_use_symlinks=False,
            )

            # Move to expected location if needed
            downloaded_path = Path(downloaded_path)
            if downloaded_path != local_path:
                if local_path.exists():
                    local_path.unlink()
                downloaded_path.rename(local_path)
                downloaded_path = local_path

            return downloaded_path

        except (RepositoryNotFoundError, RevisionNotFoundError) as e:
            logger.error(f"Repository or file not found: {e}")
            return None
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None

    def _is_model_cached(self, model_name: str, local_path: Path) -> bool:
        """Check if model is already cached and valid."""
        if not local_path.exists():
            return False

        # Check metadata
        if model_name in self.metadata.get("models", {}):
            cached_info = self.metadata["models"][model_name]
            cached_checksum = cached_info.get("checksum")

            if cached_checksum:
                current_checksum = self._calculate_checksum(local_path)
                if current_checksum == cached_checksum:
                    return True
                else:
                    logger.warning(f"Checksum mismatch for {model_name}, will re-download")

        # Basic validation - check if it's a valid GGUF file
        return self._verify_gguf_file(local_path)

    def _verify_download(self, file_path: Path) -> bool:
        """Verify that downloaded file is valid."""
        if not file_path.exists():
            return False

        # Check file size (should be at least 1MB)
        if file_path.stat().st_size < 1024 * 1024:
            logger.error(f"Downloaded file too small: {file_path.stat().st_size} bytes")
            return False

        # Verify GGUF format
        return self._verify_gguf_file(file_path)

    def _verify_gguf_file(self, file_path: Path) -> bool:
        """Verify that file is a valid GGUF file."""
        try:
            with open(file_path, "rb") as f:
                magic = f.read(4)
                return magic == b"GGUF"
        except Exception:
            return False

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum: {e}")
            return ""

    def _update_model_metadata(self, model_name: str, info: Dict[str, Any]):
        """Update metadata for a model."""
        if "models" not in self.metadata:
            self.metadata["models"] = {}

        self.metadata["models"][model_name] = info
        self._save_metadata()

    def list_cached_models(self) -> List[Dict[str, Any]]:
        """List all cached models with their information."""
        models = []

        # Get models from metadata
        for model_name, info in self.metadata.get("models", {}).items():
            local_path = Path(info.get("local_path", ""))
            if local_path.exists():
                models.append(
                    {
                        "name": model_name,
                        "path": str(local_path),
                        "size_mb": info.get("file_size", 0) // (1024 * 1024),
                        "download_time": info.get("download_time", 0),
                        "repo_id": info.get("repo_id", ""),
                        "checksum": info.get("checksum", ""),
                    }
                )

        # Also scan directory for any GGUF files not in metadata
        try:
            for gguf_file in self.cache_dir.glob("*.gguf"):
                if gguf_file.name not in [m["name"] for m in models]:
                    if self._verify_gguf_file(gguf_file):
                        models.append(
                            {
                                "name": gguf_file.name,
                                "path": str(gguf_file),
                                "size_mb": gguf_file.stat().st_size // (1024 * 1024),
                                "download_time": gguf_file.stat().st_mtime,
                                "repo_id": "unknown",
                                "checksum": "",
                            }
                        )
        except Exception as e:
            logger.error(f"Error scanning cache directory: {e}")

        return sorted(models, key=lambda x: x["download_time"], reverse=True)

    def remove_model(self, model_name: str) -> bool:
        """Remove a cached model."""
        try:
            # Remove from metadata
            if model_name in self.metadata.get("models", {}):
                model_info = self.metadata["models"][model_name]
                local_path = Path(model_info.get("local_path", ""))

                if local_path.exists():
                    local_path.unlink()
                    logger.info(f"Removed model file: {local_path}")

                del self.metadata["models"][model_name]
                self._save_metadata()
                return True

            # Try to find and remove by filename
            model_path = self.cache_dir / model_name
            if model_path.exists():
                model_path.unlink()
                logger.info(f"Removed model file: {model_path}")
                return True

            logger.warning(f"Model not found: {model_name}")
            return False

        except Exception as e:
            logger.error(f"Failed to remove model {model_name}: {e}")
            return False

    def get_cache_size(self) -> Dict[str, Any]:
        """Get information about cache usage."""
        total_size = 0
        model_count = 0

        try:
            for gguf_file in self.cache_dir.glob("*.gguf"):
                total_size += gguf_file.stat().st_size
                model_count += 1
        except Exception as e:
            logger.error(f"Error calculating cache size: {e}")

        return {
            "total_size_bytes": total_size,
            "total_size_mb": total_size // (1024 * 1024),
            "model_count": model_count,
            "cache_dir": str(self.cache_dir),
        }

    def cleanup_cache(
        self, max_size_mb: Optional[int] = None, max_age_days: Optional[int] = None
    ) -> int:
        """
        Clean up cache by removing old or excess models.

        Args:
            max_size_mb: Maximum cache size in MB
            max_age_days: Maximum age of models in days

        Returns:
            Number of models removed
        """
        removed_count = 0

        try:
            models = self.list_cached_models()

            # Remove models older than max_age_days
            if max_age_days:
                cutoff_time = time.time() - (max_age_days * 24 * 3600)
                for model in models:
                    if model["download_time"] < cutoff_time:
                        if self.remove_model(model["name"]):
                            removed_count += 1

            # Remove excess models if cache is too large
            if max_size_mb:
                current_size = self.get_cache_size()["total_size_mb"]
                if current_size > max_size_mb:
                    # Sort by download time (oldest first) and remove until under limit
                    models_by_age = sorted(models, key=lambda x: x["download_time"])

                    for model in models_by_age:
                        if current_size <= max_size_mb:
                            break

                        if self.remove_model(model["name"]):
                            current_size -= model["size_mb"]
                            removed_count += 1

            logger.info(f"Cache cleanup completed, removed {removed_count} models")

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

        return removed_count
