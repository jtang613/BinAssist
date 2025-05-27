"""
Retry handler for API requests.
"""

import time
import random
from typing import Callable, Any, Type, Tuple
from functools import wraps

from .exceptions import NetworkError, RateLimitError, APIProviderError


class RetryHandler:
    """
    Handles retry logic for API requests with exponential backoff.
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        """
        Initialize retry handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if an exception should trigger a retry.
        
        Args:
            exception: The exception that occurred
            attempt: Current attempt number (1-based)
            
        Returns:
            True if should retry
        """
        if attempt >= self.max_retries:
            return False
        
        # Retry on network errors and rate limits
        if isinstance(exception, (NetworkError, RateLimitError)):
            return True
        
        # Don't retry on authentication or other API errors
        return False
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for the given attempt.
        
        Args:
            attempt: Current attempt number (1-based)
            
        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add ±25% jitter
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def retry(self, func: Callable[[], Any]) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_exception = e
                
                if not self.should_retry(e, attempt):
                    raise
                
                if attempt < self.max_retries:
                    delay = self.get_delay(attempt)
                    print(f"Request failed (attempt {attempt}/{self.max_retries}), retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
        
        # If we get here, all retries failed
        raise last_exception


def with_retry(max_retries: int = 3, 
               base_delay: float = 1.0,
               max_delay: float = 60.0):
    """
    Decorator to add retry logic to a function.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_handler = RetryHandler(
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay
            )
            
            return retry_handler.retry(lambda: func(*args, **kwargs))
        
        return wrapper
    return decorator