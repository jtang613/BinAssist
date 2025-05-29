"""
Base service class for common functionality.
"""

from abc import ABC
from typing import Optional, Callable, Any
import threading
from binaryninja import log


class ServiceError(Exception):
    """Base exception for service errors."""
    pass


class BaseService(ABC):
    """
    Abstract base class for all services.
    
    Provides common functionality like logging, error handling,
    and cancellation support.
    """
    
    def __init__(self, name: str):
        """
        Initialize the service.
        
        Args:
            name: Name of the service for logging
        """
        self.name = name
        self._stop_event = threading.Event()
        self._error_handler: Optional[Callable[[Exception], None]] = None
    
    def set_error_handler(self, handler: Callable[[Exception], None]) -> None:
        """
        Set an error handler for the service.
        
        Args:
            handler: Function to call when errors occur
        """
        self._error_handler = handler
    
    def handle_error(self, error: Exception, operation: str = "unknown") -> None:
        """
        Handle an error that occurred during service operation.
        
        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
        """
        log.log_error(f"[BinAssist] Error during {operation}: {error}")
        
        if self._error_handler:
            try:
                self._error_handler(error)
            except Exception as handler_error:
                log.log_error(f"[BinAssist] Error in error handler: {handler_error}")
    
    def stop(self) -> None:
        """Stop any ongoing operations."""
        self._stop_event.set()
    
    def is_stopped(self) -> bool:
        """Check if the service has been stopped."""
        return self._stop_event.is_set()
    
    def reset(self) -> None:
        """Reset the service state."""
        self._stop_event.clear()