"""
API Provider configuration classes.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from enum import Enum
import json


class ProviderType(Enum):
    """Enumeration of supported provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    TEXT_GENERATION_WEBUI = "text_generation_webui"
    CUSTOM = "custom"


@dataclass
class APIProviderConfig:
    """
    Configuration for an API provider.
    
    Attributes:
        name: Human-readable name for the provider
        provider_type: Type of provider
        base_url: Base URL for the API
        api_key: API key (optional for local providers)
        model: Model name to use
        max_tokens: Maximum tokens for responses
        timeout: Request timeout in seconds
        additional_params: Additional provider-specific parameters
    """
    name: str
    provider_type: ProviderType
    base_url: str
    model: str
    max_tokens: int = 4096
    api_key: Optional[str] = None
    timeout: int = 120
    additional_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.additional_params is None:
            self.additional_params = {}
            
        # Note: API key validation is handled at runtime when making requests
        # We allow empty API keys during configuration setup
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = asdict(self)
        result["provider_type"] = self.provider_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APIProviderConfig':
        """Create configuration from dictionary."""
        data = data.copy()
        data["provider_type"] = ProviderType(data["provider_type"])
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'APIProviderConfig':
        """Create configuration from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {"Content-Type": "application/json"}
        
        if self.api_key:
            if self.provider_type == ProviderType.OPENAI:
                headers["Authorization"] = f"Bearer {self.api_key}"
            elif self.provider_type == ProviderType.ANTHROPIC:
                headers["x-api-key"] = self.api_key
                headers["anthropic-version"] = "2023-06-01"
        
        return headers
    
    def validate(self) -> bool:
        """Validate the configuration."""
        try:
            # Check required fields
            if not self.name or not self.base_url or not self.model:
                return False
                
            # Provider-specific validation
            if self.provider_type in [ProviderType.OPENAI, ProviderType.ANTHROPIC]:
                if not self.api_key:
                    return False
                    
            # Validate numeric fields
            if self.max_tokens <= 0 or self.timeout <= 0:
                return False
                
            return True
        except Exception:
            return False