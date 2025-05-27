#!/usr/bin/env python3
"""
Test script for updated o* model handling in OpenAI provider.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.api_provider.config import APIProviderConfig, ProviderType
from core.api_provider.providers.openai_provider import OpenAIProvider
from core.models.chat_message import ChatMessage


def test_o_model_payload():
    """Test that o* models generate correct payloads."""
    print("Testing o* model payload generation...")
    
    # Test o3-mini
    config_o3 = APIProviderConfig(
        name="o3-mini-test",
        provider_type=ProviderType.OPENAI,
        base_url="https://api.openai.com/v1",
        model="o3-mini",
        max_tokens=4096,
        api_key="test-key"
    )
    
    provider_o3 = OpenAIProvider(config_o3)
    messages = [ChatMessage.user("Test message")]
    
    # Test chat payload for o3 model
    payload_o3 = provider_o3._prepare_payload(messages, stream=True)
    print(f"o3-mini payload: {payload_o3}")
    
    assert "max_completion_tokens" in payload_o3, "o3 model should use max_completion_tokens"
    assert "max_tokens" not in payload_o3, "o3 model should not use max_tokens"
    assert payload_o3["stream"] == True, "o3 model should allow streaming"
    assert payload_o3["max_completion_tokens"] == 4096, "Token count should match config"
    print("✓ o3-mini payload generated correctly")
    
    # Test o4-mini
    config_o4 = APIProviderConfig(
        name="o4-mini-test",
        provider_type=ProviderType.OPENAI,
        base_url="https://api.openai.com/v1", 
        model="o4-mini",
        max_tokens=4096,
        api_key="test-key"
    )
    
    provider_o4 = OpenAIProvider(config_o4)
    payload_o4 = provider_o4._prepare_payload(messages, stream=True)
    print(f"o4-mini payload: {payload_o4}")
    
    assert "max_completion_tokens" in payload_o4, "o4 model should use max_completion_tokens"
    assert "max_tokens" not in payload_o4, "o4 model should not use max_tokens"
    print("✓ o4-mini payload generated correctly")
    
    # Test with tools
    tools = [{"type": "function", "function": {"name": "test", "description": "test"}}]
    payload_o3_tools = provider_o3._prepare_payload(messages, tools=tools)
    print(f"o3-mini with tools payload: {payload_o3_tools}")
    assert "tools" in payload_o3_tools, "o3 model should include tools"
    print("✓ o3-mini includes tools correctly")


def test_regular_model_payload():
    """Test that regular models still work correctly."""
    print("\nTesting regular model payload generation...")
    
    config_gpt = APIProviderConfig(
        name="gpt-4o-test",
        provider_type=ProviderType.OPENAI,
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        max_tokens=4096,
        api_key="test-key"
    )
    
    provider_gpt = OpenAIProvider(config_gpt)
    messages = [ChatMessage.user("Test message")]
    
    payload_gpt = provider_gpt._prepare_payload(messages, stream=True)
    print(f"gpt-4o payload: {payload_gpt}")
    
    assert "max_tokens" in payload_gpt, "Regular model should use max_tokens"
    assert "max_completion_tokens" not in payload_gpt, "Regular model should not use max_completion_tokens"
    assert payload_gpt["stream"] == True, "Regular model should support streaming"
    print("✓ gpt-4o payload generated correctly")


if __name__ == "__main__":
    try:
        test_o_model_payload()
        test_regular_model_payload()
        print("\n🎉 All tests passed! Updated o* model handling is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)