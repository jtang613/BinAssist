import binaryninja
from binaryninja import log

class RegisterSettingsGroupException(Exception):
    """
    Exception raised when the registration of a settings group fails.
    """
    def __init__(self, message):
        super().__init__(message)
        log.log_error(f"Error registering settings group: {message}")

class RegisterSettingsKeyException(Exception):
    """
    Exception raised when the registration of a settings key fails.
    """
    def __init__(self, message):
        super().__init__(message)
        log.log_error(f"Error registering settings key: {message}")

class QueryFailedException(Exception):
    """
    Exception raised when a query to the LLM fails.
    """
    def __init__(self, message):
        super().__init__(message)
        log.log_error(f"Query failed: {message}")

class ResponseProcessingException(Exception):
    """
    Exception raised when processing the response from an LLM encounters an issue.
    """
    def __init__(self, message):
        super().__init__(message)
        log.log_error(f"Response processing error: {message}")

class DatabaseConnectionException(Exception):
    """
    Exception raised when a connection to the database cannot be established or fails during execution.
    """
    def __init__(self, message):
        super().__init__(message)
        log.log_error(f"Database connection error: {message}")

