"""
Service for leveraging advanced provider capabilities.
"""

from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime

from .base_service import BaseService, ServiceError
from ..api_provider.factory import provider_registry
from ..api_provider.config import APIProviderConfig
from ..api_provider.capabilities import (
    VisionProvider, EmbeddingProvider, ModelListProvider, 
    ReasoningProvider, BatchProcessingProvider, UsageAnalyticsProvider,
    CodeAnalysisProvider, AdvancedProvider
)
from ..models.chat_message import ChatMessage


class AdvancedCapabilitiesService(BaseService):
    """
    Service for managing and utilizing advanced provider capabilities.
    
    This service provides a unified interface for accessing advanced features
    like vision analysis, embeddings, reasoning, and batch processing.
    """
    
    def __init__(self):
        """Initialize the advanced capabilities service."""
        super().__init__("advanced_capabilities")
        self._provider_cache: Dict[str, Any] = {}
        self._capability_matrix: Dict[str, List[str]] = {}
        self._refresh_capabilities()
    
    def _refresh_capabilities(self):
        """Refresh the capability matrix for available providers."""
        self._capability_matrix.clear()
        
        # This would normally scan registered providers
        # For now, we'll populate with known capabilities
        self._capability_matrix = {
            "openai": ["vision", "embeddings", "model_listing", "reasoning", "batch_processing", "usage_analytics"],
            "anthropic": ["chat", "function_calling"],  # Basic capabilities
            "ollama": ["chat", "embeddings"]  # Basic capabilities
        }
        
        self.logger.info(f"Refreshed capability matrix: {self._capability_matrix}")
    
    def get_provider_with_capability(self, capability: str, provider_config: APIProviderConfig) -> Optional[Any]:
        """
        Get a provider instance that supports a specific capability.
        
        Args:
            capability: The capability name (e.g., "vision", "embeddings")
            provider_config: Provider configuration
            
        Returns:
            Provider instance if available, None otherwise
        """
        try:
            provider_key = f"{provider_config.provider_type.value}_{provider_config.name}"
            
            # Get or create provider
            if provider_key not in self._provider_cache:
                provider = provider_registry.create_provider(provider_config)
                self._provider_cache[provider_key] = provider
            else:
                provider = self._provider_cache[provider_key]
            
            # Check if provider supports the capability
            if isinstance(provider, AdvancedProvider):
                if provider.supports_advanced_capability(capability):
                    return provider
            
            # Check specific capability interfaces
            if capability == "vision" and isinstance(provider, VisionProvider):
                return provider
            elif capability == "embeddings" and isinstance(provider, EmbeddingProvider):
                return provider
            elif capability == "reasoning" and isinstance(provider, ReasoningProvider):
                return provider
            elif capability == "batch_processing" and isinstance(provider, BatchProcessingProvider):
                return provider
            elif capability == "usage_analytics" and isinstance(provider, UsageAnalyticsProvider):
                return provider
            elif capability == "model_listing" and isinstance(provider, ModelListProvider):
                return provider
            elif capability == "code_analysis" and isinstance(provider, CodeAnalysisProvider):
                return provider
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get provider with capability {capability}: {e}")
            return None
    
    def analyze_image(self, image_data: bytes, prompt: str, provider_config: APIProviderConfig) -> Dict[str, Any]:
        """
        Analyze an image using a vision-capable provider.
        
        Args:
            image_data: Raw image data
            prompt: Analysis prompt
            provider_config: Provider configuration
            
        Returns:
            Analysis results
        """
        try:
            provider = self.get_provider_with_capability("vision", provider_config)
            if not provider:
                return {"error": "No vision-capable provider available"}
            
            result = provider.analyze_image(image_data, prompt)
            
            return {
                "success": True,
                "analysis": result,
                "provider": provider_config.name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Image analysis failed: {e}")
            return {"error": str(e), "success": False}
    
    def get_embeddings(self, texts: List[str], provider_config: APIProviderConfig) -> Dict[str, Any]:
        """
        Get embeddings for texts using an embedding-capable provider.
        
        Args:
            texts: List of texts to embed
            provider_config: Provider configuration
            
        Returns:
            Embedding results
        """
        try:
            provider = self.get_provider_with_capability("embeddings", provider_config)
            if not provider:
                return {"error": "No embedding-capable provider available"}
            
            embeddings = provider.get_embeddings(texts)
            dimension = provider.get_embedding_dimension()
            
            return {
                "success": True,
                "embeddings": embeddings,
                "dimension": dimension,
                "count": len(embeddings),
                "provider": provider_config.name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return {"error": str(e), "success": False}
    
    def create_reasoning_completion(self, messages: List[ChatMessage], 
                                  provider_config: APIProviderConfig) -> Dict[str, Any]:
        """
        Create a completion with reasoning traces.
        
        Args:
            messages: Chat messages
            provider_config: Provider configuration
            
        Returns:
            Reasoning completion results
        """
        try:
            provider = self.get_provider_with_capability("reasoning", provider_config)
            if not provider:
                return {"error": "No reasoning-capable provider available"}
            
            result = provider.create_reasoning_completion(messages)
            result["provider"] = provider_config.name
            result["timestamp"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Reasoning completion failed: {e}")
            return {"error": str(e), "success": False}
    
    def process_batch(self, batch_requests: List[Dict[str, Any]], 
                     provider_config: APIProviderConfig) -> Dict[str, Any]:
        """
        Process a batch of requests.
        
        Args:
            batch_requests: List of request dictionaries
            provider_config: Provider configuration
            
        Returns:
            Batch processing results
        """
        try:
            provider = self.get_provider_with_capability("batch_processing", provider_config)
            if not provider:
                return {"error": "No batch-processing-capable provider available"}
            
            results = provider.create_batch_completion(batch_requests)
            
            return {
                "success": True,
                "results": results,
                "total_requests": len(batch_requests),
                "provider": provider_config.name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return {"error": str(e), "success": False}
    
    def list_models(self, provider_config: APIProviderConfig) -> Dict[str, Any]:
        """
        List available models from a provider.
        
        Args:
            provider_config: Provider configuration
            
        Returns:
            Model listing results
        """
        try:
            provider = self.get_provider_with_capability("model_listing", provider_config)
            if not provider:
                return {"error": "No model-listing-capable provider available"}
            
            models = provider.list_models()
            
            return {
                "success": True,
                "models": models,
                "count": len(models),
                "provider": provider_config.name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Model listing failed: {e}")
            return {"error": str(e), "success": False}
    
    def get_usage_analytics(self, start_date: str, end_date: str, 
                           provider_config: APIProviderConfig) -> Dict[str, Any]:
        """
        Get usage analytics from a provider.
        
        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format) 
            provider_config: Provider configuration
            
        Returns:
            Usage analytics results
        """
        try:
            provider = self.get_provider_with_capability("usage_analytics", provider_config)
            if not provider:
                return {"error": "No analytics-capable provider available"}
            
            usage_stats = provider.get_usage_stats(start_date, end_date)
            cost_breakdown = provider.get_cost_breakdown(start_date, end_date)
            
            return {
                "success": True,
                "usage_stats": usage_stats,
                "cost_breakdown": cost_breakdown,
                "provider": provider_config.name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Usage analytics failed: {e}")
            return {"error": str(e), "success": False}
    
    def get_capability_matrix(self) -> Dict[str, List[str]]:
        """
        Get the current capability matrix for all providers.
        
        Returns:
            Dictionary mapping provider types to their capabilities
        """
        return self._capability_matrix.copy()
    
    def discover_capabilities(self, provider_config: APIProviderConfig) -> List[str]:
        """
        Discover what advanced capabilities a provider supports.
        
        Args:
            provider_config: Provider configuration
            
        Returns:
            List of supported capability names
        """
        try:
            provider = self.get_provider_with_capability("chat", provider_config)  # Get basic provider
            if isinstance(provider, AdvancedProvider):
                return provider.get_advanced_capabilities()
            
            # Manually check capabilities
            capabilities = []
            if isinstance(provider, VisionProvider):
                capabilities.append("vision")
            if isinstance(provider, EmbeddingProvider):
                capabilities.append("embeddings")
            if isinstance(provider, ModelListProvider):
                capabilities.append("model_listing")
            if isinstance(provider, ReasoningProvider):
                capabilities.append("reasoning")
            if isinstance(provider, BatchProcessingProvider):
                capabilities.append("batch_processing")
            if isinstance(provider, UsageAnalyticsProvider):
                capabilities.append("usage_analytics")
            if isinstance(provider, CodeAnalysisProvider):
                capabilities.append("code_analysis")
            
            return capabilities
            
        except Exception as e:
            self.logger.error(f"Capability discovery failed: {e}")
            return []
    
    def analyze_code_with_provider(self, code: str, language: str, analysis_type: str,
                                  provider_config: APIProviderConfig) -> Dict[str, Any]:
        """
        Analyze code using a code-analysis-capable provider.
        
        Args:
            code: Source code to analyze
            language: Programming language
            analysis_type: Type of analysis ("structure", "refactoring", "vulnerabilities")
            provider_config: Provider configuration
            
        Returns:
            Code analysis results
        """
        try:
            provider = self.get_provider_with_capability("code_analysis", provider_config)
            if not provider:
                # Fall back to using regular chat completion for code analysis
                provider = self.get_provider_with_capability("chat", provider_config)
                if not provider:
                    return {"error": "No provider available for code analysis"}
                
                # Create analysis prompt
                if analysis_type == "structure":
                    prompt = f"Analyze the structure and patterns in this {language} code:\n\n{code}"
                elif analysis_type == "refactoring":
                    prompt = f"Suggest refactoring improvements for this {language} code:\n\n{code}"
                elif analysis_type == "vulnerabilities":
                    prompt = f"Identify potential security vulnerabilities in this {language} code:\n\n{code}"
                else:
                    prompt = f"Analyze this {language} code:\n\n{code}"
                
                messages = [ChatMessage.user(prompt)]
                result = provider.create_chat_completion(messages)
                
                return {
                    "success": True,
                    "analysis": result,
                    "analysis_type": analysis_type,
                    "language": language,
                    "method": "chat_completion",
                    "provider": provider_config.name,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Use specialized code analysis methods
            if analysis_type == "structure":
                result = provider.analyze_code_structure(code, language)
            elif analysis_type == "refactoring":
                result = provider.suggest_refactoring(code, language)
            elif analysis_type == "vulnerabilities":
                result = provider.detect_vulnerabilities(code, language)
            else:
                result = provider.analyze_code_structure(code, language)  # Default
            
            return {
                "success": True,
                "analysis": result,
                "analysis_type": analysis_type,
                "language": language,
                "method": "specialized",
                "provider": provider_config.name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Code analysis failed: {e}")
            return {"error": str(e), "success": False}
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get statistics about the advanced capabilities service."""
        return {
            "cached_providers": len(self._provider_cache),
            "capability_matrix": self._capability_matrix,
            "available_capabilities": [
                "vision", "embeddings", "model_listing", "reasoning", 
                "batch_processing", "usage_analytics", "code_analysis"
            ]
        }