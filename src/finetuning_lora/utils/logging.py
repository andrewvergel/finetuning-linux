"""Logging utilities for finetuning-lora."""
import logging
import os
from datetime import datetime
from typing import Optional

import torch

logger = logging.getLogger(__name__)

DEBUG_LOG_FILE = "debug.log"

def setup_logging(
    log_level: int = logging.INFO,
    log_dir: str = "logs",
    log_file: Optional[str] = None,
) -> str:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (default: logging.INFO)
        log_dir: Directory to store log files (default: "logs")
        log_file: Name of the log file (default: DEBUG_LOG_FILE)
        
    Returns:
        Path to the log file
    """
    log_file = log_file or DEBUG_LOG_FILE
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    # Remove existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    
    logging.info("Debug log initialized at %s", log_path)
    return log_path

def log_version_info(
    script_name: Optional[str] = None,
    script_version: Optional[str] = None,
) -> None:
    """Log version and system information.
    
    Args:
        script_name: Optional script name to log
        script_version: Optional script version to log
    """
    if script_name and script_version:
        logging.info("🚀 %s v%s", script_name, script_version)
    
    logging.info("📅 Started at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("💻 PyTorch: %s", torch.__version__)
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info("🖥️ Device: %s", device_name)
        logging.info("📊 VRAM: %.1fGB", vram_gb)
    else:
        logging.info("🖥️ Device: CPU")

