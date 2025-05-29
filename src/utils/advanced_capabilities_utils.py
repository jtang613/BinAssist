"""
Utilities for testing and integrating advanced provider capabilities.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from ..core.services.advanced_capabilities_service import AdvancedCapabilitiesService
from ..core.api_provider.config import APIProviderConfig, ProviderType
from ..core.models.chat_message import ChatMessage


class AdvancedCapabilitiesTester:
    """Test suite for advanced provider capabilities."""
    
    def __init__(self):
        self.service = AdvancedCapabilitiesService()
        self.logger = logging.getLogger("binassist.advanced_capabilities.tester")
    
    def test_all_capabilities(self, provider_config: APIProviderConfig) -> Dict[str, Any]:
        """
        Test all available capabilities for a provider.
        
        Args:
            provider_config: Provider configuration to test
            
        Returns:
            Test results dictionary
        """
        results = {
            "provider": provider_config.name,
            "provider_type": provider_config.provider_type.value,
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # Discover capabilities
        capabilities = self.service.discover_capabilities(provider_config)
        results["available_capabilities"] = capabilities
        
        # Test each capability
        for capability in capabilities:
            try:
                if capability == "vision":
                    results["tests"]["vision"] = self._test_vision(provider_config)
                elif capability == "embeddings":
                    results["tests"]["embeddings"] = self._test_embeddings(provider_config)
                elif capability == "model_listing":
                    results["tests"]["model_listing"] = self._test_model_listing(provider_config)
                elif capability == "reasoning":
                    results["tests"]["reasoning"] = self._test_reasoning(provider_config)
                elif capability == "batch_processing":
                    results["tests"]["batch_processing"] = self._test_batch_processing(provider_config)
                elif capability == "usage_analytics":
                    results["tests"]["usage_analytics"] = self._test_usage_analytics(provider_config)
                elif capability == "code_analysis":
                    results["tests"]["code_analysis"] = self._test_code_analysis(provider_config)
                    
            except Exception as e:
                results["tests"][capability] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def _test_vision(self, provider_config: APIProviderConfig) -> Dict[str, Any]:
        """Test vision capabilities."""
        try:
            # Create a simple test image (1x1 red pixel PNG)
            test_image = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xdd\xcc\xdb\x1d\x00\x00\x00\x00IEND\xaeB`\x82'
            
            result = self.service.analyze_image(
                test_image, 
                "Describe this test image",
                provider_config
            )
            
            return {
                "success": result.get("success", False),
                "has_analysis": bool(result.get("analysis")),
                "details": result
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_embeddings(self, provider_config: APIProviderConfig) -> Dict[str, Any]:
        """Test embedding capabilities."""
        try:
            test_texts = [
                "This is a test sentence for embeddings.",
                "Binary analysis and reverse engineering.",
                "Machine learning and artificial intelligence."
            ]
            
            result = self.service.get_embeddings(test_texts, provider_config)
            
            return {
                "success": result.get("success", False),
                "embedding_count": result.get("count", 0),
                "dimension": result.get("dimension", 0),
                "details": result
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_model_listing(self, provider_config: APIProviderConfig) -> Dict[str, Any]:
        """Test model listing capabilities."""
        try:
            result = self.service.list_models(provider_config)
            
            return {
                "success": result.get("success", False),
                "model_count": result.get("count", 0),
                "has_models": bool(result.get("models")),
                "details": result
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_reasoning(self, provider_config: APIProviderConfig) -> Dict[str, Any]:
        """Test reasoning capabilities."""
        try:
            messages = [
                ChatMessage.user("Solve this step by step: What is 2 + 2 * 3?")
            ]
            
            result = self.service.create_reasoning_completion(messages, provider_config)
            
            return {
                "success": not result.get("error"),
                "reasoning_available": result.get("reasoning_available", False),
                "has_content": bool(result.get("content")),
                "reasoning_tokens": result.get("reasoning_tokens", 0),
                "details": result
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_batch_processing(self, provider_config: APIProviderConfig) -> Dict[str, Any]:
        """Test batch processing capabilities."""
        try:
            batch_requests = [
                {
                    "id": "test_1",
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "kwargs": {}
                },
                {
                    "id": "test_2", 
                    "messages": [{"role": "user", "content": "Count to 3"}],
                    "kwargs": {}
                }
            ]
            
            result = self.service.process_batch(batch_requests, provider_config)
            
            return {
                "success": result.get("success", False),
                "processed_count": len(result.get("results", [])),
                "total_requests": result.get("total_requests", 0),
                "details": result
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_usage_analytics(self, provider_config: APIProviderConfig) -> Dict[str, Any]:
        """Test usage analytics capabilities."""
        try:
            start_date = "2024-01-01"
            end_date = "2024-12-31"
            
            result = self.service.get_usage_analytics(start_date, end_date, provider_config)
            
            return {
                "success": result.get("success", False),
                "has_usage_stats": bool(result.get("usage_stats")),
                "has_cost_breakdown": bool(result.get("cost_breakdown")),
                "details": result
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_code_analysis(self, provider_config: APIProviderConfig) -> Dict[str, Any]:
        """Test code analysis capabilities."""
        try:
            test_code = '''
def vulnerable_function(user_input):
    # Potential buffer overflow
    buffer = [0] * 10
    for i in range(len(user_input)):
        buffer[i] = ord(user_input[i])
    return buffer
'''
            
            result = self.service.analyze_code_with_provider(
                test_code, "python", "vulnerabilities", provider_config
            )
            
            return {
                "success": result.get("success", False),
                "has_analysis": bool(result.get("analysis")),
                "analysis_method": result.get("method"),
                "details": result
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class CapabilityRecommendationEngine:
    """Engine for recommending provider capabilities based on use cases."""
    
    @staticmethod
    def recommend_for_binary_analysis() -> Dict[str, List[str]]:
        """Recommend capabilities useful for binary analysis."""
        return {
            "essential": [
                "chat",  # Basic interaction
                "function_calling"  # Tool integration
            ],
            "highly_recommended": [
                "reasoning",  # Complex analysis tasks
                "code_analysis",  # Security analysis
                "embeddings"  # Similarity searches
            ],
            "useful": [
                "vision",  # Analyze screenshots/diagrams
                "batch_processing",  # Process multiple samples
                "model_listing"  # Model selection
            ],
            "optional": [
                "usage_analytics"  # Cost tracking
            ]
        }
    
    @staticmethod
    def recommend_for_malware_analysis() -> Dict[str, List[str]]:
        """Recommend capabilities useful for malware analysis."""
        return {
            "essential": [
                "chat",
                "function_calling",
                "reasoning"  # Critical for complex malware logic
            ],
            "highly_recommended": [
                "code_analysis",  # Vulnerability detection
                "batch_processing",  # Analyze multiple samples
                "embeddings"  # Similarity analysis
            ],
            "useful": [
                "vision",  # Analyze network diagrams
                "model_listing"
            ],
            "optional": [
                "usage_analytics"
            ]
        }
    
    @staticmethod
    def recommend_for_reverse_engineering() -> Dict[str, List[str]]:
        """Recommend capabilities useful for reverse engineering."""
        return {
            "essential": [
                "chat",
                "function_calling",
                "reasoning"  # Complex decompilation logic
            ],
            "highly_recommended": [
                "code_analysis",  # Structure analysis
                "embeddings"  # Pattern matching
            ],
            "useful": [
                "vision",  # Analyze UI screenshots
                "batch_processing",  # Process multiple functions
                "model_listing"
            ],
            "optional": [
                "usage_analytics"
            ]
        }
    
    @staticmethod
    def get_capability_descriptions() -> Dict[str, str]:
        """Get descriptions of each capability."""
        return {
            "chat": "Basic conversational AI for queries and explanations",
            "function_calling": "Ability to execute tools and functions during conversations",
            "vision": "Analyze images, screenshots, and visual content",
            "embeddings": "Generate vector representations for similarity search",
            "reasoning": "Advanced step-by-step reasoning for complex problems",
            "code_analysis": "Specialized code structure and vulnerability analysis",
            "batch_processing": "Process multiple requests efficiently",
            "model_listing": "Discover and select available AI models",
            "usage_analytics": "Track usage statistics and costs"
        }


def setup_advanced_capabilities_demo(provider_configs: List[APIProviderConfig]) -> Dict[str, Any]:
    """
    Set up a demonstration of advanced capabilities across providers.
    
    Args:
        provider_configs: List of provider configurations to test
        
    Returns:
        Demo results and recommendations
    """
    tester = AdvancedCapabilitiesTester()
    
    demo_results = {
        "timestamp": datetime.now().isoformat(),
        "providers_tested": len(provider_configs),
        "test_results": {},
        "capability_matrix": {},
        "recommendations": {}
    }
    
    # Test each provider
    for config in provider_configs:
        provider_key = f"{config.provider_type.value}_{config.name}"
        demo_results["test_results"][provider_key] = tester.test_all_capabilities(config)
        
        # Build capability matrix
        capabilities = demo_results["test_results"][provider_key].get("available_capabilities", [])
        demo_results["capability_matrix"][provider_key] = capabilities
    
    # Generate recommendations
    demo_results["recommendations"]["binary_analysis"] = CapabilityRecommendationEngine.recommend_for_binary_analysis()
    demo_results["recommendations"]["malware_analysis"] = CapabilityRecommendationEngine.recommend_for_malware_analysis()
    demo_results["recommendations"]["reverse_engineering"] = CapabilityRecommendationEngine.recommend_for_reverse_engineering()
    
    # Add capability descriptions
    demo_results["capability_descriptions"] = CapabilityRecommendationEngine.get_capability_descriptions()
    
    return demo_results