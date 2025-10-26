import subprocess
from typing import Optional, Tuple
from shwizard.utils.logger import get_logger

logger = get_logger(__name__)

class CommandExecutor:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
    
    def execute(self, command: str, timeout: Optional[int] = None) -> Tuple[bool, str]:
        if command.startswith('/'):
            return True, "Command not logged because it starts with '/'"
            
        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute: {command}")
            return True, "[DRY RUN] Command not actually executed"
        
        try:
            logger.info(f"Executing command: {command}")
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            
            if success:
                logger.info(f"Command executed successfully")
            else:
                logger.warning(f"Command failed with return code {result.returncode}")
            
            return success, output
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout} seconds")
            return False, f"Command timed out after {timeout} seconds"
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return False, str(e)
    
    def set_dry_run(self, enabled: bool):
        self.dry_run = enabled
        logger.info(f"Dry run mode: {'enabled' if enabled else 'disabled'}")

