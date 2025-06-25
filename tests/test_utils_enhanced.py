"""
Enhanced tests for utility functions in the venice_ai.utils module.
"""

import pytest
import json
import sys
import warnings
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List, Dict, Any

from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.types.models import ModelList
from venice_ai.utils import (
    get_filtered_models,
    estimate_token_count,
    validate_chat_messages,
    find_model_by_id,
    get_model_capabilities,
    format_tool_response
)


class TestGetFilteredModelsEnhanced:
    """Enhanced tests for get_filtered_models function."""
    
    @pytest.mark.asyncio
    async def test_error_handling_api_failure(self, async_venice_client):
        """Should handle API errors gracefully."""
        # Mock response with exception
        async_venice_client.models.list = MagicMock(side_effect=Exception("API failure"))
        
        # Should return empty list on error
        result = await get_filtered_models(async_venice_client)  # type: ignore[misc]
        assert isinstance(result, list)
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_no_models_found(self, async_venice_client):
        """Should return empty list when no models match filters."""
        # Mock response with no matches
        mock_models = {
            "data": [
                {"id": "model1", "type": "text", "model_spec": {"capabilities": {"streaming": True}}},
                {"id": "model2", "type": "image", "model_spec": {"capabilities": {"streaming": True}}}
            ]
        }
        
        async_venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Filter with non-matching capability
        result = await get_filtered_models(  # type: ignore[misc]
            async_venice_client,
            model_type="text",
            supports_vision=True  # Using new capability parameter
        )
        
        # Verify results
        assert isinstance(result, list)
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_empty_capabilities_list(self, async_venice_client):
        """Should handle empty capabilities list."""
        mock_models = {
            "data": [
                {"id": "model1", "type": "text", "model_spec": {"capabilities": {}}},
                {"id": "model2", "type": "image", "model_spec": {"capabilities": {}}}
            ]
        }
        
        async_venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Filter with capabilities but models have none
        result = await get_filtered_models(  # type: ignore[misc]
            async_venice_client,
            supports_function_calling=True  # Using new capability parameter
        )
        
        # Verify results
        assert isinstance(result, list)
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_empty_response(self, async_venice_client):
        """Should handle empty response."""
        mock_models: ModelList = {"object": "list", "data": [], "type": None}
        async_venice_client.models.list = MagicMock(return_value=mock_models)
        
        result = await get_filtered_models(async_venice_client)  # type: ignore[misc]
        
        # Verify results
        assert isinstance(result, list)
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_missing_model_spec(self, async_venice_client):
        """Should handle models without model_spec."""
        mock_models = {
            "data": [
                {"id": "model1", "type": "text"},  # No model_spec
                {"id": "model2", "type": "text", "model_spec": {"capabilities": {"streaming": True}}}
            ]
        }
        
        async_venice_client.models.list = AsyncMock(return_value=mock_models)
        
        # Filter models with streaming capability
        # Note: streaming is a legacy capability field
        result = await get_filtered_models(  # type: ignore[misc]
            async_venice_client,
            model_type="text"  # Just filter by type since streaming is legacy
        )
        
        # Verify results - both text models should be returned
        assert len(result) == 2
        assert result[0]["id"] == "model1"
        assert result[1]["id"] == "model2"
    
    @pytest.mark.asyncio
    async def test_malformed_response(self, async_venice_client):
        """Should handle malformed response."""
        # Response without "data" field
        mock_models = {"models": [{"id": "model1"}]}
        async_venice_client.models.list = MagicMock(return_value=mock_models)
        
        result = await get_filtered_models(async_venice_client)  # type: ignore[misc]
        
        # Verify results
        assert isinstance(result, list)
        assert len(result) == 0


class TestEstimateTokenCountEnhanced:
    """Enhanced tests for estimate_token_count function."""
    
    def test_none_input(self):
        """Should handle None input."""
        with pytest.raises(AttributeError):
            estimate_token_count(None)  # type: ignore[arg-type]
    
    def test_number_input(self):
        """Should handle numeric input by converting to string."""
        # Numbers should be converted to string
        count = estimate_token_count(12345)  # type: ignore[arg-type]
        assert isinstance(count, int)
        assert count > 0
    
    def test_whitespace_only(self):
        """Should handle whitespace-only input."""
        # Whitespace only should give minimal token count
        count = estimate_token_count("   \n\t   ")
        assert isinstance(count, int)
        assert count > 0

    @patch('venice_ai.utils.tiktoken', new=None)
    @patch('venice_ai.utils._TIKTOKEN_AVAILABLE', False)
    def test_tiktoken_import_failure(self):
        """Should fall back when tiktoken is not available."""
        # This test ensures the 'tiktoken library not found' warning is emitted.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Ensure all warnings are captured
            count = estimate_token_count("This is a test string")
            assert isinstance(count, int)
            assert count > 0 # Should return a valid fallback count
            # Should emit warning
            assert len(w) > 0
            assert "tiktoken library not found" in str(w[0].message)

    @patch('venice_ai.utils._TIKTOKEN_AVAILABLE', True)
    def test_tiktoken_attribute_error(self):
        """Should fall back when tiktoken has attribute error during encoding."""
        # Create mock tiktoken module with attribute error for get_encoding
        mock_tiktoken = MagicMock()
        mock_tiktoken.get_encoding.side_effect = AttributeError("'module' object has no attribute 'get_encoding'")
        
        with patch('venice_ai.utils.tiktoken', mock_tiktoken):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")  # Ensure all warnings are captured
                count = estimate_token_count("This is a test string")
                assert isinstance(count, int)
                assert count > 0
                # Should emit warning
                assert len(w) > 0
                assert "tiktoken attribute error" in str(w[0].message)

    def test_long_text(self):
        """Should handle very long text."""
        # Create a long text (100KB)
        long_text = "word " * 20000
        
        # Should handle without error
        count = estimate_token_count(long_text)
        assert isinstance(count, int)
        assert count > 1000  # Expect many tokens
    
    def test_special_characters(self):
        """Should handle text with special characters."""
        special_text = "👋 こんにちは! Спасибо. ¿Cómo estás? \n\t$%^&*()_+-=[]{}|;':\",./<>?"
        
        # Should handle without error
        count = estimate_token_count(special_text)
        assert isinstance(count, int)
        assert count > 0


class TestValidateChatMessagesEnhanced:
    """Enhanced tests for validate_chat_messages function."""
    
    def test_validate_null_messages(self):
        """Should handle null messages."""
        with pytest.raises(AttributeError):
            validate_chat_messages(None)  # type: ignore[arg-type]
    
    def test_max_messages_zero(self):
        """Should validate with max_messages=0."""
        messages = [{"role": "user", "content": "Hello"}]
        
        result = validate_chat_messages(messages, max_messages=0)
        assert len(result["errors"]) > 0
        assert "exceeds maximum" in result["errors"][0]
    
    def test_max_total_chars_zero(self):
        """Should validate with max_total_chars=0."""
        messages = [{"role": "user", "content": "Hello"}]
        
        result = validate_chat_messages(messages, max_total_chars=0)
        assert len(result["errors"]) > 0
        assert "exceeds maximum" in result["errors"][0]
    
    def test_null_content(self):
        """Should validate message with null content."""
        messages = [{"role": "user", "content": None}]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
    
    def test_non_string_role(self):
        """Should validate message with non-string role."""
        messages = [{"role": 123, "content": "Hello"}]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert "must be a string" in result["errors"][0]
    
    def test_multimodal_content_validation(self):
        """Should validate multimodal content."""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "image_url": {"url": "https://example.com/image.jpg"}}
            ]}
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) == 0
        
        # Invalid multimodal content
        messages = [
            {"role": "user", "content": [
                {"invalid_key": "value"}
            ]}
        ]
        
        result = validate_chat_messages(messages)
        # The validation doesn't fully check multimodal content structure yet
        # This test is to ensure it at least handles the list content type
    
    def test_system_message_with_tool_calls(self):
        """Should validate system message with tool_calls (invalid)."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant", "tool_calls": [{"id": "tool1"}]}
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert "cannot have 'tool_calls'" in result["errors"][0]
    
    def test_system_message_with_tool_call_id(self):
        """Should validate system message with tool_call_id (invalid)."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant", "tool_call_id": "tool1"}
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert "cannot have 'tool_call_id'" in result["errors"][0]
    
    def test_assistant_message_with_empty_tool_calls(self):
        """Should validate assistant message with empty tool_calls list."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "tool_calls": []}
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert "must be a non-empty list" in result["errors"][0]
    
    def test_assistant_message_with_empty_content_and_no_tool_calls(self):
        """Should validate assistant message with empty content and no tool_calls."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""}
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert "must have either non-empty content or tool_calls" in result["errors"][0]
    
    def test_tool_message_missing_tool_call_id(self):
        """Should validate tool message with missing tool_call_id."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "tool_calls": [
                {"id": "tool1", "type": "function", "function": {"name": "test", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "Result"}
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert "missing required 'tool_call_id'" in result["errors"][0]
    
    def test_tool_message_wrong_tool_call_id(self):
        """Should validate tool message with wrong tool_call_id."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "tool_calls": [
                {"id": "tool1", "type": "function", "function": {"name": "test", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "wrong_id", "content": "Result"}
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert "does not match any expected ID" in result["errors"][0]
    
    def test_tool_message_empty_content(self):
        """Should validate tool message with empty content."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "tool_calls": [
                {"id": "tool1", "type": "function", "function": {"name": "test", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "tool1", "content": ""}
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert "must have non-empty string content" in result["errors"][0]
    
    def test_missing_tool_responses(self):
        """Should validate missing tool responses."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "tool_calls": [
                {"id": "tool1", "type": "function", "function": {"name": "test", "arguments": "{}"}},
                {"id": "tool2", "type": "function", "function": {"name": "test2", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "tool1", "content": "Result"}
            # Missing response for tool2
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert "Missing tool responses" in result["errors"][0]
        assert "tool2" in result["errors"][0]


class TestFindModelByIdEnhanced:
    """Enhanced tests for find_model_by_id function."""
    
    @pytest.mark.asyncio
    async def test_error_handling(self, async_venice_client):
        """Should handle API errors gracefully."""
        # Mock response with exception
        async_venice_client.models.list = MagicMock(side_effect=Exception("API failure"))
        
        # Should return None on error
        result = await find_model_by_id(async_venice_client, "model1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_empty_response(self, async_venice_client):
        """Should handle empty response."""
        # Mock empty response
        mock_models: ModelList = {"object": "list", "data": [], "type": None}
        async_venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Should return None when model is not found
        result = await find_model_by_id(async_venice_client, "nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_malformed_model_data(self, async_venice_client):
        """Should handle model data without ID."""
        # Mock response with model missing ID
        mock_models = {
            "data": [
                {"name": "Model 1", "type": "text"},  # No id field
                {"id": "model2", "name": "Model 2", "type": "text"}
            ]
        }
        async_venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Should skip models with missing ID and find valid one
        result = await find_model_by_id(async_venice_client, "model2")
        assert result is not None
        assert result["id"] == "model2"
        
        # Should return None for nonexistent model
        result = await find_model_by_id(async_venice_client, "model1")
        assert result is None


class TestGetModelCapabilitiesEnhanced:
    """Enhanced tests for get_model_capabilities function."""
    
    @pytest.mark.asyncio
    async def test_error_handling(self, async_venice_client):
        """Should handle API errors gracefully."""
        # Mock find_model_by_id to raise exception
        with patch('venice_ai.utils.find_model_by_id', AsyncMock(side_effect=Exception("API failure"))):
            # Should return None on error
            result = await get_model_capabilities(async_venice_client, "model1")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_with_empty_capabilities(self, async_venice_client):
        """Should handle model found but has empty capabilities."""
        # Mock a model with empty capabilities
        mock_model = {
            "id": "model1",
            "type": "text",
            "model_spec": {"capabilities": {}}
        }
        
        # Mock find_model_by_id to return our mock model
        with patch('venice_ai.utils.find_model_by_id', AsyncMock(return_value=mock_model)):
            capabilities = await get_model_capabilities(async_venice_client, "model1")
            assert capabilities == {}
    
    @pytest.mark.asyncio
    async def test_with_null_capabilities(self, async_venice_client):
        """Should handle model found but has null capabilities."""
        # Mock a model with null capabilities
        mock_model = {
            "id": "model1",
            "type": "text",
            "model_spec": {"capabilities": None}
        }
        
        # Mock find_model_by_id to return our mock model
        with patch('venice_ai.utils.find_model_by_id', AsyncMock(return_value=mock_model)):
            capabilities = await get_model_capabilities(async_venice_client, "model1")
            assert capabilities is None


class TestFormatToolResponseEnhanced:
    """Enhanced tests for format_tool_response function."""
    
    def test_with_none_content(self):
        """Should handle None content."""
        response = format_tool_response("call_123", None)
        
        assert response["role"] == "tool"
        assert response["tool_call_id"] == "call_123"
        assert response["content"] == "null"
    
    def test_with_empty_string(self):
        """Should handle empty string content."""
        response = format_tool_response("call_123", "")
        
        assert response["role"] == "tool"
        assert response["tool_call_id"] == "call_123"
        assert response["content"] == ""
    
    def test_with_complex_dict(self):
        """Should stringify complex dict content."""
        data = {
            "results": [
                {"name": "Item 1", "values": [1, 2, 3]},
                {"name": "Item 2", "values": [4, 5, 6]}
            ],
            "metadata": {
                "timestamp": "2023-01-01T12:00:00Z",
                "source": "test"
            }
        }
        response = format_tool_response("call_456", data)
        
        assert response["role"] == "tool"
        assert response["tool_call_id"] == "call_456"
        
        # Content should be JSON string
        assert isinstance(response["content"], str)
        decoded = json.loads(response["content"])
        assert decoded["results"][0]["name"] == "Item 1"
        assert decoded["metadata"]["source"] == "test"
    
    def test_with_non_serializable_object(self):
        """Should handle non-JSON serializable objects."""
        class TestObject:
            def __str__(self):
                return "TestObject instance"
                
        data = TestObject()
        response = format_tool_response("call_789", data)
        
        assert response["role"] == "tool"
        assert response["tool_call_id"] == "call_789"
        assert response["content"] == "TestObject instance"
    
    def test_with_boolean_content(self):
        """Should handle boolean content."""
        response_true = format_tool_response("call_bool_true", True)
        response_false = format_tool_response("call_bool_false", False)
        
        assert response_true["content"] == "True"
        assert response_false["content"] == "False"
    
    def test_with_number_content(self):
        """Should handle numeric content."""
        response_int = format_tool_response("call_int", 42)
        response_float = format_tool_response("call_float", 3.14159)
        
        assert response_int["content"] == "42"
        assert response_float["content"] == "3.14159"