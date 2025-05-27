"""
Debug logging configuration for BinAssist.
"""

import logging
import sys
from datetime import datetime


def setup_debug_logging(level=logging.DEBUG, log_to_file=True):
    """
    Setup comprehensive debug logging for BinAssist.
    
    Args:
        level: Logging level (default: DEBUG)
        log_to_file: Whether to log to file as well as console
    """
    # Create root logger for binassist
    logger = logging.getLogger('binassist')
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if requested
    if log_to_file:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"binassist_debug_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Debug logging enabled - file: {log_file}")
        except Exception as e:
            logger.warning(f"Could not create log file: {e}")
    
    # Log important startup info
    logger.info("=" * 60)
    logger.info("BinAssist Debug Logging Started")
    logger.info(f"Log level: {logging.getLevelName(level)}")
    logger.info(f"Python version: {sys.version}")
    logger.info("=" * 60)
    
    return logger


def safe_call(func, *args, logger=None, operation="unknown", **kwargs):
    """
    Safely call a function with comprehensive error logging.
    
    Args:
        func: Function to call
        *args: Function arguments
        logger: Logger instance (optional)
        operation: Description of operation for logging
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or None if error
    """
    if logger is None:
        logger = logging.getLogger('binassist.safe_call')
    
    try:
        logger.debug(f"Starting {operation}")
        result = func(*args, **kwargs)
        logger.debug(f"Completed {operation} successfully")
        return result
    except Exception as e:
        logger.error(f"Error in {operation}: {type(e).__name__}: {e}")
        logger.exception(f"Full traceback for {operation}")
        return None


def log_function_entry(func):
    """
    Decorator to log function entry and exit.
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(f'binassist.{func.__module__}.{func.__name__}')
        logger.debug(f"Entering {func.__name__} with args={len(args)}, kwargs={list(kwargs.keys())}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__} successfully")
            return result
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {type(e).__name__}: {e}")
            raise
    
    return wrapper