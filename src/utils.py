"""
Utility functions for the pillar design prediction model.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LOGS_DIR


def setup_logger(name: str) -> logging.Logger:
    """
    Setup a logger instance with file and console handlers.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler
    log_file = LOGS_DIR / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def print_separator(title: str = "", length: int = 80) -> None:
    """
    Print a formatted separator line with optional title.
    
    Args:
        title: Optional title to display
        length: Length of separator line
    """
    if title:
        print(f"\n{'=' * length}")
        print(f"  {title}")
        print(f"{'=' * length}\n")
    else:
        print(f"\n{'=' * length}\n")


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that DataFrame has all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if all required columns present, False otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    return True