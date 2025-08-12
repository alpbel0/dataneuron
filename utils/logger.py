"""
DataNeuron Project Centralized Logging Module
=============================================

This module provides a centralized logging system for the entire DataNeuron project.
It uses loguru for structured logging with both console (colored) and file (JSON) outputs.

Usage:
    from utils.logger import logger
    logger.info("This is an info message")
    logger.error("This is an error message")

Features:
- Colored console output for human readability
- JSON file output for production analysis
- Automatic file rotation and compression
- Log retention management
- Asynchronous file writing for performance
- Centralized configuration via settings
"""

import sys
import os
from pathlib import Path
from loguru import logger

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import LOG_LEVEL, LOG_FILE


# Remove the default loguru handler to have full control over configuration
logger.remove()


# ============================================================================
# CONSOLE HANDLER - Colored, human-readable format
# ============================================================================

console_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

logger.add(
    sys.stdout,
    format=console_format,
    level=LOG_LEVEL,
    colorize=True,
    backtrace=True,
    diagnose=True
)


# ============================================================================
# FILE HANDLER - JSON format with rotation and retention
# ============================================================================

logger.add(
    LOG_FILE,
    format="{time} | {level} | {name}:{function}:{line} | {message}",
    level=LOG_LEVEL,
    rotation="10 MB",        # Create new file when current reaches 10MB
    retention="2 days",     # Keep log files for 2 days
    compression="zip",       # Compress archived log files
    serialize=True,          # Output in JSON format
    enqueue=True,           # Asynchronous file writing for performance
    backtrace=True,
    diagnose=True
)


# ============================================================================
# LOGGER CONFIGURATION SUMMARY
# ============================================================================

def get_logger_info():
    """
    Returns information about the current logger configuration.
    Useful for debugging and verification.
    """
    return {
        "log_level": LOG_LEVEL,
        "log_file": str(LOG_FILE),
        "handlers": [
            {
                "type": "console",
                "format": "colored_human_readable",
                "level": LOG_LEVEL
            },
            {
                "type": "file", 
                "format": "json",
                "level": LOG_LEVEL,
                "rotation": "10 MB",
                "retention": "30 days",
                "compression": "zip",
                "async": True
            }
        ]
    }


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    """
    Test the logger with all different log levels.
    This helps verify that the configuration is working correctly.
    """
    
    print("=== DataNeuron Logger Test ===")
    print(f"Log Level: {LOG_LEVEL}")
    print(f"Log File: {LOG_FILE}")
    print("\nTesting all log levels:\n")
    
    # Test all log levels
    logger.trace("This is a TRACE message - most detailed level")
    logger.debug("This is a DEBUG message - detailed diagnostic info")
    logger.info("This is an INFO message - general information")
    logger.success("This is a SUCCESS message - operation completed successfully")
    logger.warning("This is a WARNING message - something unexpected happened")
    logger.error("This is an ERROR message - an error occurred")
    logger.critical("This is a CRITICAL message - serious error occurred")
    
    # Test structured logging with extra data
    logger.info("Structured log example", extra={
        "user_id": "12345", 
        "action": "file_upload",
        "file_size": 1024,
        "processing_time": 0.5
    })
    
    # Test exception logging
    try:
        result = 1 / 0
    except ZeroDivisionError as e:
        logger.exception("Exception occurred during division")
    
    print(f"\n=== Test Complete ===")
    print(f"Check console output above and log file at: {LOG_FILE}")
    print("If you see colored messages above and the file exists, logging is working correctly!")
    
    # Display configuration info
    print(f"\nLogger Configuration:")
    import json
    print(json.dumps(get_logger_info(), indent=2))