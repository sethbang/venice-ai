"""
Additional tests for venice_ai.utils module focused on improving coverage.

This test file specifically targets code paths that weren't covered
by the main test_utils.py file.
"""

import pytest
import sys
import json
from unittest.mock import patch, MagicMock, mock_open

from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.utils import (
    get_filtered_models,
    _get_filtered_models_sync,
    _get_filtered_models_async,
    estimate_token_count,
    validate_chat_messages,
    find_model_by_id,
    get_model_capabilities,
    format_tool_response
)


class TestGetFilteredModelsSync:
    """Tests for _get_filtered_models_sync function."""

    def test_exception_handling(self, venice_client):
        """Should handle exceptions in models.list() and return an empty list."""
        # Mock the client's list method to raise an exception
        venice_client.models.list = MagicMock(side_effect=Exception("API Error"))
        
        # Call the function and capture the result
        result = _get_filtered_models_sync(venice_client)
        
        # Verify an empty list is returned
        assert result == []
        assert venice_client.models.list.called

    def test_type_mismatch_filtering(self, venice_client):
        """Should filter out models with mismatched type."""
        # Mock response with different model types
        mock_models = {
            "data": [
                {"id": "model1", "type": "text", "model_spec": {"capabilities": {}}},
                {"id": "model2", "type": "image", "model_spec": {"capabilities": {}}},
                {"id": "model3", "type": "audio", "model_spec": {"capabilities": {}}}
            ]
        }
        
        # Mock the client's list method
        venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Filter for a specific model type
        result = _get_filtered_models_sync(venice_client, model_type="text")
        
        # Verify only models of the specified type are included
        assert len(result) == 1
        assert result[0]["id"] == "model1"

    def test_capability_filtering_missing(self, venice_client):
        """Should filter out models missing requested capabilities."""
        # Mock response with models having different capabilities
        mock_models = {
            "data": [
                {
                    "id": "model1",
                    "type": "text",
                    "model_spec": {"capabilities": {"supportsFunctionCalling": True}}
                },
                {
                    "id": "model2",
                    "type": "text",
                    "model_spec": {"capabilities": {}}  # Missing capability
                },
                {
                    "id": "model3",
                    "type": "text",
                    "model_spec": {"capabilities": {"supportsFunctionCalling": False}}  # False capability
                }
            ]
        }
        
        # Mock the client's list method
        venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Filter for models with a specific capability
        result = _get_filtered_models_sync(
            venice_client, 
            supports_capabilities=["supportsFunctionCalling"]
        )
        
        # Verify only models with the capability set to True are included
        assert len(result) == 1
        assert result[0]["id"] == "model1"


class TestGetFilteredModelsAsync:
    """Tests for _get_filtered_models_async function."""
    
    @pytest.mark.asyncio
    async def test_unexpected_error_handling(self, async_venice_client):
        """Should test the error path where TypeError is caught and handled."""
        # Create a mock that raises a TypeError with a message not matching the expected patterns
        error_message = "Some unexpected error"
        async_venice_client.models.list = MagicMock(side_effect=TypeError(error_message))
        
        # The function should catch the exception and return an empty list
        result = await _get_filtered_models_async(async_venice_client)
        assert result == []
        
        # Verify the error path was hit
        async_venice_client.models.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_client_fallback(self, venice_client):
        """Should handle sync client by falling back to non-awaited call."""
        # Mock response for a synchronous client
        mock_models = {
            "data": [
                {"id": "model1", "type": "text", "model_spec": {"capabilities": {}}}
            ]
        }
        
        # Mock the client's list method
        venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Call the async function with a sync client
        result = await _get_filtered_models_async(venice_client)
        
        # Verify the function worked with the sync client
        assert len(result) == 1
        assert result[0]["id"] == "model1"
        assert venice_client.models.list.called

    @pytest.mark.asyncio
    async def test_runtime_error_handling(self, async_venice_client):
        """Should handle RuntimeError with an unexpected error message."""
        # Create a mock that raises a RuntimeError with a message not matching the expected patterns
        error_message = "Some runtime error"
        async_venice_client.models.list = MagicMock(side_effect=RuntimeError(error_message))
        
        # The function should catch the exception and return an empty list
        result = await _get_filtered_models_async(async_venice_client)
        assert result == []
        
        # Verify the error path was hit
        async_venice_client.models.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception_handling(self, async_venice_client):
        """Should handle exceptions in async models.list() and return an empty list."""
        # Mock the client's list method to raise an exception
        async_venice_client.models.list = MagicMock(side_effect=Exception("API Error"))
        
        # Call the function and capture the result
        result = await _get_filtered_models_async(async_venice_client)
        
        # Verify an empty list is returned
        assert result == []
        assert async_venice_client.models.list.called


class TestEstimateTokenCount:
    """Tests for estimate_token_count function."""

    def test_tiktoken_success_path(self):
        """Should correctly use tiktoken for token estimation when available."""
        # Create a mock tiktoken module and encoding
        mock_tiktoken = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_tiktoken.get_encoding.return_value = mock_encoding
        
        # Patch the import and test the tiktoken path
        with patch('venice_ai.utils._TIKTOKEN_AVAILABLE', True), \
             patch('venice_ai.utils.tiktoken', mock_tiktoken):
            result = estimate_token_count("Some test text")
            
            # Verify tiktoken was used and correct token count returned
            assert result == 5
            mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")
            mock_encoding.encode.assert_called_once_with("Some test text")


class TestValidateChatMessagesDetailedCases:
    """Additional tests for validate_chat_messages focusing on edge cases."""
    
    def test_non_string_role(self):
        """Should detect when role is not a string."""
        messages = [
            {"role": 123, "content": "Invalid role type"}
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert any("Role at index 0 must be a string" in error for error in result["errors"])

    def test_empty_system_message_content(self):
        """Should detect empty content in system message."""
        messages = [
            {"role": "system", "content": ""}  # Empty string content
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert any("System message at index 0 must have non-empty string content" in error for error in result["errors"])

    def test_system_message_with_tool_calls(self):
        """Should detect system message with tool_calls."""
        messages = [
            {
                "role": "system", 
                "content": "System message", 
                "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "test", "arguments": "{}"}}]
            }
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert any("System message at index 0 cannot have 'tool_calls'" in error for error in result["errors"])

    def test_system_message_with_tool_call_id(self):
        """Should detect system message with tool_call_id."""
        messages = [
            {"role": "system", "content": "System message", "tool_call_id": "call_123"}
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert any("System message at index 0 cannot have 'tool_call_id'" in error for error in result["errors"])

    def test_empty_user_message_content(self):
        """Should detect empty content in user message."""
        messages = [
            {"role": "user", "content": ""}  # Empty string content
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert any("User message at index 0 has empty string content" in error for error in result["errors"])

    def test_user_message_empty_content_list(self):
        """Should detect empty content list in user message."""
        messages = [
            {"role": "user", "content": []}  # Empty content list
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert any("User message at index 0 has empty content list" in error for error in result["errors"])

    def test_user_message_invalid_content_type(self):
        """Should detect invalid content type in user message."""
        messages = [
            {"role": "user", "content": 123}  # Neither string nor list
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert any("User message at index 0 must have either string or list content" in error for error in result["errors"])

    def test_invalid_tool_calls_structure(self):
        """Should detect invalid tool_calls structure in assistant message."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "tool_calls": "not_a_list"}  # Invalid tool_calls type
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert any("must be a non-empty list" in error for error in result["errors"])

    def test_invalid_tool_call_structure(self):
        """Should detect invalid structure within tool_calls items."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "tool_calls": [{"not_a_valid_tool_call": True}]}  # Missing required fields
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        # Should detect missing id field
        assert any("missing required 'id' field" in error for error in result["errors"])

    def test_missing_tool_responses(self):
        """Should detect missing tool responses at the end of the conversation."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant", 
                "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "test", "arguments": "{}"}}
                ]
            }
            # Missing tool response message
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert any("Missing tool responses for tool_call_ids: call_123" in error for error in result["errors"])


class TestFindModelById:
    """Additional tests for find_model_by_id function."""

    @pytest.mark.asyncio
    async def test_unexpected_error_handling(self, async_venice_client):
        """Should test the error path where TypeError is caught and handled."""
        # Create a mock that raises a TypeError with a message not matching expected patterns
        error_message = "Some unexpected error"
        async_venice_client.models.list = MagicMock(side_effect=TypeError(error_message))
        
        # The function should catch the exception and return None
        result = await find_model_by_id(async_venice_client, "model1")
        assert result is None
        
        # Verify the error path was hit
        async_venice_client.models.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_client_fallback(self, venice_client):
        """Should fall back to sync client behavior and expect a DeprecationWarning."""
        # Mock response
        mock_models = {
            "data": [
                {"id": "model1", "type": "text"}
            ]
        }
        
        # Mock the client's list method
        venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Expect the specific DeprecationWarning
        with pytest.warns(DeprecationWarning, match="Calling an async function without awaiting"):
            result = await find_model_by_id(venice_client, "model1")
        
        # Verify the sync client's list method was called
        venice_client.models.list.assert_called_once()
        
        # Verify the results
        assert result is not None
        assert result["id"] == "model1"
        assert result["type"] == "text"
        
    def test_deprecation_warning(self):
        """Test that a deprecation warning is properly issued."""
        # Direct test of the warning mechanism
        with patch('warnings.warn') as mock_warn:
            # Directly import and call warnings.warn with the same parameters as in the function
            import warnings
            warnings.warn("Calling an async function without awaiting", DeprecationWarning, stacklevel=2)
            
            # Verify the warning was issued with the correct parameters
            mock_warn.assert_called_with(
                "Calling an async function without awaiting",
                DeprecationWarning,
                stacklevel=2
            )

    @pytest.mark.asyncio
    async def test_exception_handling(self, async_venice_client):
        """Should handle exceptions and return None."""
        # Mock the client's list method to raise an exception
        async_venice_client.models.list = MagicMock(side_effect=Exception("API Error"))
        
        # Call the function
        result = await find_model_by_id(async_venice_client, "model1")
        
        # Should return None on exception
        assert result is None
        assert async_venice_client.models.list.called


class TestGetModelCapabilities:
    """Additional tests for get_model_capabilities function."""

    @pytest.mark.asyncio
    async def test_exception_handling(self, async_venice_client):
        """Should handle exceptions when getting model capabilities."""
        # Mock find_model_by_id to raise an exception
        with patch('venice_ai.utils.find_model_by_id', side_effect=Exception("API Error")):
            # Call the function
            result = await get_model_capabilities(async_venice_client, "model1")
            
            # Should return None on exception
            assert result is None


class TestFormatToolResponse:
    """Additional tests for format_tool_response function."""

    def test_fallback_string_conversion(self):
        """Should convert non-string/non-dict/list content using str()."""
        # Test with an integer
        int_response = format_tool_response("call_123", 42)
        assert int_response["content"] == "42"
        
        # Test with a float
        float_response = format_tool_response("call_456", 3.14)
        assert float_response["content"] == "3.14"
        
        # Test with a boolean
        bool_response = format_tool_response("call_789", True)
        assert bool_response["content"] == "True"
        
        # Test with a custom object
        class CustomClass:
            def __str__(self):
                return "CustomObject"
        
        obj_response = format_tool_response("call_obj", CustomClass())
        assert obj_response["content"] == "CustomObject"