import platform
import os
import shutil
from typing import Dict, List, Optional
from pathlib import Path


def get_os_type() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    elif system == "windows":
        return "windows"
    else:
        return "linux"


def get_shell_type() -> str:
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        return "zsh"
    elif "bash" in shell:
        return "bash"
    elif "fish" in shell:
        return "fish"
    elif platform.system().lower() == "windows":
        return "powershell"
    else:
        return "sh"


def get_current_directory() -> str:
    return os.getcwd()


def get_system_info() -> Dict[str, str]:
    return {
        "os": get_os_type(),
        "shell": get_shell_type(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "cwd": get_current_directory(),
    }


def check_tool_installed(tool: str) -> bool:
    return shutil.which(tool) is not None


def get_installed_tools(tools_to_check: Optional[List[str]] = None) -> List[str]:
    if tools_to_check is None:
        tools_to_check = [
            "git", "docker", "kubectl", "npm", "python", "pip",
            "node", "java", "gcc", "make", "curl", "wget"
        ]
    
    installed = []
    for tool in tools_to_check:
        if check_tool_installed(tool):
            installed.append(tool)
    
    return installed


def get_home_directory() -> Path:
    return Path.home()


def get_config_directory() -> Path:
    if platform.system().lower() == "windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif platform.system().lower() == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    
    config_dir = base / "shwizard"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_data_directory() -> Path:
    if platform.system().lower() == "windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif platform.system().lower() == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    
    data_dir = base / "shwizard"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
