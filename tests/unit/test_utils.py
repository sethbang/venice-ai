import pytest
from typing import List, Dict, Any
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import io
import json

from venice_ai.utils import (
    get_filtered_models, _get_filtered_models_sync, _get_filtered_models_async,
    estimate_token_count, validate_chat_messages, find_model_by_id,
    get_model_capabilities, format_tool_response
)
from venice_ai._client import VeniceClient
from venice_ai._async_client import AsyncVeniceClient
from venice_ai.types.models import ModelType

class TestGetFilteredModels:
    @pytest.fixture
    def mock_sync_client(self):
        client = MagicMock(spec=VeniceClient)
        client.models = MagicMock()
        client.models.list.return_value = {"data": [
            {"id": "model1", "type": "text", "model_spec": {"capabilities": {"chat": True, "embeddings": False}}},
            {"id": "model2", "type": "image", "model_spec": {"capabilities": {"image_generation": True}}},
            {"id": "model3", "type": "text", "model_spec": {"capabilities": {"chat": True, "tool_calling": True}}},
            {"id": "model4", "type": "text", "model_spec": {}}, # Model without model_spec
            {"id": "model5", "type": "text", "model_spec": {"capabilities": {}}}, # Model with empty capabilities
        ]}
        return client

    @pytest.fixture
    def mock_async_client(self):
        client = MagicMock(spec=AsyncVeniceClient)
        client.models = MagicMock()
        client.models.list = AsyncMock(return_value={"data": [
            {"id": "model1-async", "type": "text", "model_spec": {"capabilities": {"chat": True, "embeddings": False}}},
            {"id": "model2-async", "type": "image", "model_spec": {"capabilities": {"image_generation": True}}},
            {"id": "model3-async", "type": "text", "model_spec": {"capabilities": {"chat": True, "tool_calling": True}}},
            {"id": "model4-async", "type": "text", "model_spec": {}}, # Model without model_spec
            {"id": "model5-async", "type": "text", "model_spec": {"capabilities": {}}}, # Model with empty capabilities
        ]})
        return client

    def test_get_filtered_models_sync_error_handling(self, mock_sync_client):
        """Test error handling in _get_filtered_models_sync."""
        mock_sync_client.models.list.side_effect = Exception("API Error")
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_stdout = io.StringIO()

        result = _get_filtered_models_sync(mock_sync_client)

        sys.stdout = old_stdout # Restore stdout
        
        assert result == []
        assert "Error fetching models: API Error" in captured_stdout.getvalue()

    @pytest.mark.asyncio
    async def test_get_filtered_models_async_error_handling(self, mock_async_client):
        """Test error handling in _get_filtered_models_async."""
        mock_async_client.models.list.side_effect = Exception("API Error Async")

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_stdout = io.StringIO()

        result = await _get_filtered_models_async(mock_async_client)

        sys.stdout = old_stdout # Restore stdout

        assert result == []
        assert "Error fetching models: API Error Async" in captured_stdout.getvalue()

class TestEstimateTokenCount:
    @patch("venice_ai.utils._TIKTOKEN_AVAILABLE", False)
    def test_fallback_without_tiktoken(self):
        """Test estimate_token_count fallback when tiktoken is not available."""
        with pytest.warns(UserWarning, match="tiktoken library not found"):
            text = "This is a test string for fallback."
            estimated_tokens = estimate_token_count(text)
            # Fallback is len(text) / 4, rounded down, min 1
            expected_tokens = max(1, int(len(text) / 4.0))
            assert estimated_tokens == expected_tokens

    @patch("venice_ai.utils.tiktoken")
    def test_fallback_with_tiktoken_attribute_error(self, mock_tiktoken):
        """Test estimate_token_count fallback when tiktoken import succeeds but attribute access fails."""
        mock_tiktoken.get_encoding.side_effect = AttributeError("Encoding not found")
        with pytest.warns(UserWarning, match="Warning: tiktoken attribute error occurred: Encoding not found. Using a simple character-based heuristic for token estimation."):
            text = "Another test string for fallback."
            estimated_tokens = estimate_token_count(text)
            expected_tokens = max(1, int(len(text) / 4.0))
            assert estimated_tokens == expected_tokens


class TestValidateChatMessages:
    def test_tool_message_missing_tool_call_id(self):
        """Test tool message validation with missing tool_call_id."""
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "func", "arguments": "{}"}}]},
            {"role": "tool", "content": "Tool output"}, # Missing tool_call_id
        ]
        result = validate_chat_messages(messages)
        assert "Tool message at index 1 missing required 'tool_call_id' field." in result["errors"]

    def test_tool_message_invalid_tool_call_id_type(self):
        """Test tool message validation with invalid tool_call_id type."""
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "func", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": 123, "content": "Tool output"}, # Invalid type
        ]
        result = validate_chat_messages(messages)
        assert "Tool message at index 1 has invalid 'tool_call_id' field." in result["errors"]

    def test_tool_message_empty_tool_call_id(self):
        """Test tool message validation with empty tool_call_id string."""
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "func", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "", "content": "Tool output"}, # Empty string
        ]
        result = validate_chat_messages(messages)
        assert "Tool message at index 1 has invalid 'tool_call_id' field." in result["errors"]

    def test_tool_message_unmatched_tool_call_id(self):
        """Test tool message validation with unmatched tool_call_id."""
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "func", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_456", "content": "Tool output"}, # Unmatched ID
        ]
        result = validate_chat_messages(messages)
        assert "Tool message at index 1 has 'tool_call_id': call_456 does not match any expected ID" in result["errors"]

    def test_tool_message_missing_content(self):
        """Test tool message validation with missing content."""
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "func", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_123"}, # Missing content
        ]
        result = validate_chat_messages(messages)
        assert "Tool message at index 1 is missing 'content'." in result["errors"]

    def test_tool_message_invalid_content_type(self):
        """Test tool message validation with invalid content type."""
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "func", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_123", "content": 123}, # Invalid type
        ]
        result = validate_chat_messages(messages)
        assert "Tool message at index 1 must have non-empty string content." in result["errors"]

    def test_tool_message_empty_content_string(self):
        """Test tool message validation with empty content string."""
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "func", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_123", "content": ""}, # Empty string
        ]
        result = validate_chat_messages(messages)
        assert "Tool message at index 1 must have non-empty string content." in result["errors"]

    def test_assistant_message_missing_content_and_tool_calls(self):
        """Test assistant message validation with missing content and tool_calls."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant"}, # Missing both
        ]
        result = validate_chat_messages(messages)
        assert "Assistant message at index 1 must have either non-empty content or tool_calls." in result["errors"]

    def test_assistant_message_empty_content_and_tool_calls_list(self):
        """Test assistant message validation with empty content string and empty tool_calls list."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "", "tool_calls": []}, # Empty content and empty list
        ]
        result = validate_chat_messages(messages)
        assert "Assistant message at index 1 must have either non-empty content or tool_calls." in result["errors"]

    def test_assistant_message_invalid_tool_calls_type(self):
        """Test assistant message validation with invalid tool_calls type."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "tool_calls": "not a list"}, # Invalid type
        ]
        result = validate_chat_messages(messages)
        assert "Assistant message at index 1 has 'tool_calls' that must be a non-empty list." in result["errors"]

    def test_assistant_message_tool_call_invalid_structure(self):
        """Test assistant message validation with invalid tool call structure."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function"}]}, # Missing function
        ]
        result = validate_chat_messages(messages)
        assert "Tool call at index 0 in message 1 missing required 'function' field." in result["errors"]

    def test_assistant_message_tool_call_invalid_function_structure(self):
        """Test assistant message validation with invalid tool call function structure."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "func"}}]}, # Missing arguments
        ]
        result = validate_chat_messages(messages)
        assert "Function call at index 0 in message 1 is missing 'arguments'." in result["errors"]

    def test_assistant_message_tool_call_invalid_function_name_type(self):
        """Test assistant message validation with invalid tool call function name type."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": 123, "arguments": "{}"}}]}, # Invalid name type
        ]
        result = validate_chat_messages(messages)
        assert "Function in tool call 0 in message 1 has invalid 'name' field." in result["errors"]

    def test_assistant_message_tool_call_invalid_function_arguments_type(self):
        """Test assistant message validation with invalid tool call function arguments type."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "func", "arguments": {}}}]}, # Invalid arguments type
        ]
        result = validate_chat_messages(messages)
        assert "Function call at index 0 in message 1 has invalid 'arguments' field type." in result["errors"]

    def test_assistant_message_with_tool_calls_and_non_string_content(self):
        """Test assistant message with tool_calls and non-string content."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": 123, "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "func", "arguments": "{}"}}]}, # Non-string content
        ]
        result = validate_chat_messages(messages)
        assert "Assistant message at index 1 with tool_calls has non-string content." in result["errors"]

    def test_user_message_invalid_content_type(self):
        """Test user message validation with invalid content type."""
        messages = [
            {"role": "user", "content": 123}, # Invalid type
        ]
        result = validate_chat_messages(messages)
        assert "User message at index 0 must have either string or list content." in result["errors"]

    def test_user_message_empty_content_list(self):
        """Test user message validation with empty content list."""
        messages = [
            {"role": "user", "content": []}, # Empty list
        ]
        result = validate_chat_messages(messages)
        assert "User message at index 0 has empty content list." in result["errors"]

    def test_system_message_invalid_content_type(self):
        """Test system message validation with invalid content type."""
        messages = [
            {"role": "system", "content": 123}, # Invalid type
        ]
        result = validate_chat_messages(messages)
        assert "System message at index 0 must have non-empty string content." in result["errors"]

    def test_system_message_empty_content_string(self):
        """Test system message validation with empty content string."""
        messages = [
            {"role": "system", "content": ""}, # Empty string
        ]
        result = validate_chat_messages(messages)
        assert "System message at index 0 must have non-empty string content." in result["errors"]

    def test_system_message_with_tool_calls(self):
        """Test system message validation with tool_calls."""
        messages = [
            {"role": "system", "content": "System message", "tool_calls": []}, # With tool_calls
        ]
        result = validate_chat_messages(messages)
        assert "System message at index 0 cannot have 'tool_calls'." in result["errors"]

    def test_system_message_with_tool_call_id(self):
        """Test system message validation with tool_call_id."""
        messages = [
            {"role": "system", "content": "System message", "tool_call_id": "call_123"}, # With tool_call_id
        ]
        result = validate_chat_messages(messages)
        assert "System message at index 0 cannot have 'tool_call_id'." in result["errors"]

    def test_message_invalid_role_type(self):
        """Test message validation with invalid role type."""
        messages = [
            {"role": 123, "content": "Hello"}, # Invalid type
        ]
        result = validate_chat_messages(messages)
        assert "Role at index 0 must be a string." in result["errors"]

    def test_message_missing_role(self):
        """Test message validation with missing role."""
        messages = [
            {"content": "Hello"}, # Missing role
        ]
        result = validate_chat_messages(messages)
        assert "Message at index 0 is missing required 'role' field." in result["errors"]

    def test_messages_not_a_list(self):
        """Test validation when messages is not a list."""
        messages = "not a list"
        result = validate_chat_messages(messages)  # type: ignore[arg-type]
        assert "Messages must be a list." in result["errors"]

    def test_messages_empty_list(self):
        """Test validation when messages is an empty list."""
        messages: List[Dict[str, Any]] = []
        result = validate_chat_messages(messages)
        assert "Messages list cannot be empty." in result["errors"]

    def test_message_count_exceeds_max(self):
        """Test validation when message count exceeds max_messages."""
        messages = [{"role": "user", "content": "1"}, {"role": "user", "content": "2"}]
        result = validate_chat_messages(messages, max_messages=1)
        assert "Message count (2) exceeds maximum allowed (1)." in result["errors"]

    def test_total_chars_exceeds_max(self):
        """Test validation when total character count exceeds max_total_chars."""
        messages = [{"role": "user", "content": "a" * 10}, {"role": "assistant", "content": "b" * 10}]
        result = validate_chat_messages(messages, max_total_chars=15)
        assert "Total character count (20) exceeds maximum allowed (15)." in result["errors"]

    def test_missing_tool_response(self):
        """Test validation when a tool call is not followed by a tool response."""
        messages = [
            {"role": "user", "content": "Call tool"},
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "func", "arguments": "{}"}}]},
            # Missing tool response for call_123
        ]
        result = validate_chat_messages(messages)
        assert "Missing tool responses for tool_call_ids: call_123" in result["errors"]

    def test_multiple_missing_tool_responses(self):
        """Test validation when multiple tool calls are not followed by responses."""
        messages = [
            {"role": "user", "content": "Call tools"},
            {"role": "assistant", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "func1", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "func2", "arguments": "{}"}},
            ]},
            # Missing responses for call_1 and call_2
        ]
        result = validate_chat_messages(messages)
        # Order might vary, check for both IDs
        assert "Missing tool responses for tool_call_ids: call_1, call_2" in result["errors"] or \
               "Missing tool responses for tool_call_ids: call_2, call_1" in result["errors"]

    def test_tool_response_after_non_assistant(self):
        """Test validation when a tool message does not follow an assistant message."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "tool_call_id": "call_123", "content": "Tool output"}, # Follows user
        ]
        result = validate_chat_messages(messages)
        assert "Tool message at index 1 must follow an assistant message." in result["errors"]

    def test_assistant_follows_assistant(self):
        """Test validation when an assistant message follows another assistant message."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "assistant", "content": "How are you?"}, # Follows assistant
        ]
        result = validate_chat_messages(messages)
        assert "Assistant message at index 2 should not directly follow another assistant message." in result["errors"]

    def test_user_follows_user(self):
        """Test validation when a user message follows another user message."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "How are you?"}, # Follows user
        ]
        result = validate_chat_messages(messages)
        assert "User message at index 1 cannot directly follow another user message." in result["errors"]

    def test_multiple_system_messages(self):
        """Test validation with multiple system messages."""
        messages = [
            {"role": "system", "content": "System 1"},
            {"role": "system", "content": "System 2"}, # Second system message
        ]
        result = validate_chat_messages(messages)
        assert "System message at index 1 must be the first message." in result["errors"]
        assert "Multiple system messages found. Only one is allowed." in result["errors"]

    def test_system_message_not_first(self):
        """Test validation when a system message is not the first."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "System message"}, # Not first
        ]
        result = validate_chat_messages(messages)
        assert "System message at index 1 must be the first message." in result["errors"]

class TestFindModelById:
    @pytest.fixture
    def mock_sync_client(self):
        client = MagicMock(spec=VeniceClient)
        client.models = MagicMock()
        client.models.list.return_value = {"data": [
            {"id": "model1", "name": "Model 1"},
            {"id": "model2", "name": "Model 2"},
        ]}
        return client

    @pytest.fixture
    def mock_async_client(self):
        client = MagicMock(spec=AsyncVeniceClient)
        client.models = MagicMock()
        client.models.list = AsyncMock(return_value={"data": [
            {"id": "model1-async", "name": "Model 1 Async"},
            {"id": "model2-async", "name": "Model 2 Async"},
        ]})
        return client

    @pytest.mark.asyncio
    async def test_find_model_by_id_sync_error_handling(self, mock_sync_client):
        """Test error handling in find_model_by_id sync."""
        mock_sync_client.models.list.side_effect = Exception("API Error Sync Find")

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_stdout = io.StringIO()

        result = find_model_by_id(mock_sync_client, "model1")

        sys.stdout = old_stdout # Restore stdout

        assert result is None
        assert "Error finding model by ID: API Error Sync Find" in captured_stdout.getvalue()

    @pytest.mark.asyncio
    async def test_find_model_by_id_async_error_handling(self, mock_async_client):
        """Test error handling in find_model_by_id async."""
        mock_async_client.models.list.side_effect = Exception("API Error Async Find")

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_stdout = io.StringIO()

        result = await find_model_by_id(mock_async_client, "model1-async")

        sys.stdout = old_stdout # Restore stdout

        assert result is None
        assert "Error finding model by ID: API Error Async Find" in captured_stdout.getvalue()

class TestGetModelCapabilities:
    @pytest.fixture
    def mock_sync_client(self):
        client = MagicMock(spec=VeniceClient)
        # Mock find_model_by_id to control its return value
        client.models = MagicMock() # Ensure models attribute exists
        return client

    @pytest.fixture
    def mock_async_client(self):
        client = MagicMock(spec=AsyncVeniceClient)
        # Mock find_model_by_id to control its return value
        client.models = MagicMock() # Ensure models attribute exists
        return client

    @patch("venice_ai.utils.find_model_by_id")
    @pytest.mark.asyncio
    async def test_get_model_capabilities_sync_error_handling(self, mock_find_model_by_id, mock_sync_client):
        """Test error handling in get_model_capabilities sync."""
        mock_find_model_by_id.side_effect = Exception("Find Model Error Sync")

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_stdout = io.StringIO()

        result = await get_model_capabilities(mock_sync_client, "model1")

        sys.stdout = old_stdout # Restore stdout

        assert result is None
        assert "Error getting model capabilities: Find Model Error Sync" in captured_stdout.getvalue()

    @pytest.mark.asyncio
    @patch("venice_ai.utils.find_model_by_id")
    async def test_get_model_capabilities_async_error_handling(self, mock_find_model_by_id, mock_async_client):
        """Test error handling in get_model_capabilities async."""
        mock_find_model_by_id.side_effect = Exception("Find Model Error Async")

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_stdout = io.StringIO()

        result = await get_model_capabilities(mock_async_client, "model1-async")

        sys.stdout = old_stdout # Restore stdout

        assert result is None
        assert "Error getting model capabilities: Find Model Error Async" in captured_stdout.getvalue()

class TestFormatToolResponse:
    def test_format_tool_response_non_string_non_dict_non_list(self):
        """Test format_tool_response with content that is not string, dict, or list."""
        tool_call_id = "call_123"
        content = 12345 # Integer content

        result = format_tool_response(tool_call_id, content)

        assert result == {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": "12345" # Should be stringified
        }

    def test_format_tool_response_none_content(self):
        """Test format_tool_response with None content."""
        tool_call_id = "call_456"
        content = None

        result = format_tool_response(tool_call_id, content)

        assert result == {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": "null" # Should be "null" string
        }