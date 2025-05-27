"""
Exception classes for API providers.
"""


class APIProviderError(Exception):
    """Base exception for API provider errors."""
    pass


class AuthenticationError(APIProviderError):
    """Authentication-related errors."""
    pass


class RateLimitError(APIProviderError):
    """Rate limiting errors."""
    pass


class NetworkError(APIProviderError):
    """Network-related errors."""
    pass