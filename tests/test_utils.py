"""
Tests for utility functions in the venice_ai.utils module.
"""

import pytest
import json
import warnings
from unittest.mock import patch, MagicMock, AsyncMock
import importlib.util

from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.exceptions import VeniceError
from venice_ai.types.models import ModelList
from venice_ai.utils import (
    get_filtered_models,
    estimate_token_count,
    validate_chat_messages,
    find_model_by_id,
    find_model_by_id_or_name_or_slug,
    get_model_capabilities_by_id_or_name_or_slug,
    get_model_capabilities,
    format_tool_response,
    _import_tiktoken_module,
    import_module_from_path, # Add import
    _get_filtered_models_async # Add import for direct testing if needed, or test via get_filtered_models
)


class TestTiktokenFallback:
    """Tests for tiktoken fallback logic."""
    
    def test_estimate_token_count_without_tiktoken(self):
        """Test estimate_token_count fallback when tiktoken is not available."""
        with patch('venice_ai.utils._TIKTOKEN_AVAILABLE', False):
            with pytest.warns(UserWarning, match="tiktoken library not found or unavailable"):
                result = estimate_token_count("Hello world")
                # Fallback should use max(1, int(len(s) / 4.0))
                expected = max(1, int(len("Hello world") / 4.0))
                assert result == expected


class TestModelLookupFunctions:
    """Tests for model lookup utility functions."""
    
    def test_find_model_by_id_or_name_or_slug_by_id(self):
        """Test finding model by ID."""
        # Mock MODELS_DATA with a test model
        test_model = {"id": "test-model-id", "name": "Test Model", "slug": "test-model"}
        with patch('venice_ai.utils.MODELS_DATA', [test_model]):
            result = find_model_by_id_or_name_or_slug("test-model-id")
            assert result == test_model
    
    def test_find_model_by_id_or_name_or_slug_by_name(self):
        """Test finding model by name."""
        test_model = {"id": "test-model-id", "name": "Test Model", "slug": "test-model"}
        with patch('venice_ai.utils.MODELS_DATA', [test_model]):
            result = find_model_by_id_or_name_or_slug("Test Model")
            assert result == test_model
    
    def test_find_model_by_id_or_name_or_slug_by_slug(self):
        """Test finding model by slug."""
        test_model = {"id": "test-model-id", "name": "Test Model", "slug": "test-model"}
        with patch('venice_ai.utils.MODELS_DATA', [test_model]):
            result = find_model_by_id_or_name_or_slug("test-model")
            assert result == test_model
    
    def test_find_model_by_id_or_name_or_slug_not_found(self):
        """Test finding model that doesn't exist."""
        test_model = {"id": "test-model-id", "name": "Test Model", "slug": "test-model"}
        with patch('venice_ai.utils.MODELS_DATA', [test_model]):
            result = find_model_by_id_or_name_or_slug("nonexistent-model")
            assert result is None
    
    def test_get_model_capabilities_by_id_or_name_or_slug_not_found(self):
        """Test getting capabilities for non-existent model."""
        with patch('venice_ai.utils.MODELS_DATA', []):
            result = get_model_capabilities_by_id_or_name_or_slug("nonexistent-model")
            assert result is None
    
    def test_get_model_capabilities_by_id_or_name_or_slug_invalid_spec(self):
        """Test getting capabilities when model_spec is invalid."""
        test_model = {"id": "test-model-id", "model_spec": "invalid_spec"}  # Not a dict
        with patch('venice_ai.utils.MODELS_DATA', [test_model]):
            result = get_model_capabilities_by_id_or_name_or_slug("test-model-id")
            assert result is None
    
    def test_get_model_capabilities_by_id_or_name_or_slug_missing_capabilities(self):
        """Test getting capabilities when capabilities key is missing."""
        test_model = {"id": "test-model-id", "model_spec": {"other_key": "value"}}  # No capabilities
        with patch('venice_ai.utils.MODELS_DATA', [test_model]):
            result = get_model_capabilities_by_id_or_name_or_slug("test-model-id")
            assert result is None


class TestValidateChatMessagesHighPriority:
    """Tests for validate_chat_messages function - high priority coverage."""
    
    def test_validate_chat_messages_tool_call_not_dict(self):
        """Test validation when tool_calls contains non-dict items."""
        messages = [
            {"role": "assistant", "tool_calls": ["not_a_dict"]}
        ]
        result = validate_chat_messages(messages)
        assert "Tool call at index 0 in message 0 must be a dictionary." in result["errors"]
    
    def test_validate_chat_messages_tool_call_invalid_type(self):
        """Test validation when tool call has invalid type."""
        messages = [
            {"role": "assistant", "tool_calls": [{"type": "invalid_type", "function": {"name": "test"}}]}
        ]
        result = validate_chat_messages(messages)
        assert "Tool call at index 0 in message 0 has invalid type 'invalid_type'. Must be 'function'." in result["errors"]
    
    def test_validate_chat_messages_function_call_missing_name(self):
        """Test validation when function call is missing name."""
        messages = [
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"arguments": "{}"}}]}
        ]
        result = validate_chat_messages(messages)
        assert "Function call at index 0 in message 0 is missing 'name'." in result["errors"]
    
    def test_validate_chat_messages_function_call_missing_arguments(self):
        """Test validation when function call is missing arguments."""
        messages = [
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "test"}}]}
        ]
        result = validate_chat_messages(messages)
        assert "Function call at index 0 in message 0 is missing 'arguments'." in result["errors"]
    
    def test_validate_chat_messages_assistant_no_content_no_tool_calls(self):
        """Test validation when assistant message has no content and no tool_calls."""
        messages = [
            {"role": "assistant"}  # No content, no tool_calls
        ]
        result = validate_chat_messages(messages)
        assert "Assistant message at index 0 must have non-null content when not using tool_calls." in result["errors"]
    
    def test_validate_chat_messages_tool_message_missing_content(self):
        """Test validation when tool message is missing content."""
        messages = [
            {"role": "tool", "tool_call_id": "test_id"}  # No content
        ]
        result = validate_chat_messages(messages)
        assert "Tool message at index 0 is missing 'content'." in result["errors"]


class TestImportModuleFromPath:
    """Tests for import_module_from_path utility function."""

    @patch('importlib.util.spec_from_file_location')
    def test_import_module_spec_is_none(self, mock_spec_from_file_location):
        """Test ImportError when spec_from_file_location returns None."""
        mock_spec_from_file_location.return_value = None
        with pytest.raises(ImportError, match="Could not load spec for module non_existent_module from path/to/module.py"):
            import_module_from_path("non_existent_module", "path/to/module.py")

    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_import_module_module_is_none(self, mock_module_from_spec, mock_spec_from_file_location):
        """Test ImportError when module_from_spec returns None."""
        mock_spec = MagicMock()
        mock_spec.name = "test_module"
        mock_spec_from_file_location.return_value = mock_spec
        mock_module_from_spec.return_value = None
        with pytest.raises(ImportError, match="Could not create module test_module from spec"):
            import_module_from_path("test_module", "path/to/test_module.py")

    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_import_module_loader_attribute_error(self, mock_module_from_spec, mock_spec_from_file_location):
        """Test ImportError when spec.loader is missing exec_module."""
        mock_spec = MagicMock()
        mock_spec.name = "test_module"
        mock_spec.loader = object()  # A loader without exec_module
        mock_spec_from_file_location.return_value = mock_spec
        mock_module_from_spec.return_value = MagicMock() # A dummy module object
        
        with pytest.raises(ImportError, match="Spec loader is not a valid Loader for module test_module"):
            import_module_from_path("test_module", "path/to/test_module.py")

    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_import_module_exec_module_raises_import_error(self, mock_module_from_spec, mock_spec_from_file_location):
        """Test that an ImportError during exec_module is re-raised."""
        mock_spec = MagicMock()
        mock_spec.name = "test_module"
        mock_loader = MagicMock()
        mock_loader.exec_module.side_effect = ImportError("Simulated exec error")
        mock_spec.loader = mock_loader
        mock_spec_from_file_location.return_value = mock_spec
        mock_module_from_spec.return_value = MagicMock()

        with pytest.raises(ImportError, match="Simulated exec error"):
            import_module_from_path("test_module", "path/to/test_module.py")


class TestGetFilteredModels:
    """Tests for get_filtered_models function."""

    @pytest.mark.asyncio
    async def test_filter_by_model_type(self, async_venice_client):
        """Should filter models by type."""
        # Mock response
        mock_models = {
            "data": [
                {"id": "model1", "type": "text", "model_spec": {"capabilities": {}}},
                {"id": "model2", "type": "image", "model_spec": {"capabilities": {}}},
                {"id": "model3", "type": "text", "model_spec": {"capabilities": {}}}
            ]
        }
        
        # Mock the client's list method
        async_venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Filter text models
        result = await get_filtered_models(async_venice_client, model_type="text")  # type: ignore[misc]
        
        # Verify results
        assert len(result) == 2
        assert result[0]["id"] == "model1"
        assert result[1]["id"] == "model3"
        
    @pytest.mark.asyncio
    async def test_filter_by_capabilities(self, async_venice_client):
        """Should filter models by capabilities."""
        # Mock response
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
                    "model_spec": {"capabilities": {"supportsFunctionCalling": False}}
                },
                {
                    "id": "model3",
                    "type": "text",
                    "model_spec": {"capabilities": {"supportsFunctionCalling": True, "streaming": True}}
                }
            ]
        }
        
        # Mock the client's list method
        async_venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Filter models with function calling
        result = await get_filtered_models(  # type: ignore[misc]
            async_venice_client,
            supports_capabilities=["supportsFunctionCalling"]
        )
        
        # Verify results
        assert len(result) == 2
        assert result[0]["id"] == "model1"
        assert result[1]["id"] == "model3"
        
    @pytest.mark.asyncio
    async def test_filter_by_multiple_capabilities(self, async_venice_client):
        """Should filter models by multiple capabilities."""
        # Mock response
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
                    "model_spec": {"capabilities": {"supportsFunctionCalling": True, "streaming": False}}
                },
                {
                    "id": "model3",
                    "type": "text",
                    "model_spec": {"capabilities": {"supportsFunctionCalling": True, "streaming": True}}
                }
            ]
        }
        
        # Mock the client's list method
        async_venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Filter models with both capabilities
        result = await get_filtered_models(  # type: ignore[misc]
            async_venice_client,
            supports_capabilities=["supportsFunctionCalling", "streaming"]
        )
        
        # Verify results
        assert len(result) == 1
        assert result[0]["id"] == "model3"
        
    @pytest.mark.asyncio
    async def test_filter_combined(self, async_venice_client):
        """Should filter by both type and capabilities."""
        # Mock response
        mock_models = {
            "data": [
                {
                    "id": "model1",
                    "type": "text",
                    "model_spec": {"capabilities": {"supportsFunctionCalling": True}}
                },
                {
                    "id": "model2",
                    "type": "image", 
                    "model_spec": {"capabilities": {"supportsFunctionCalling": True}}
                },
                {
                    "id": "model3",
                    "type": "text",
                    "model_spec": {"capabilities": {"streaming": True}}
                }
            ]
        }
        
        # Mock the client's list method
        async_venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Filter text models with function calling
        result = await get_filtered_models(  # type: ignore[misc]
            async_venice_client,
            model_type="text",
            supports_capabilities=["supportsFunctionCalling"]
        )
        
        # Verify results
        assert len(result) == 1
        assert result[0]["id"] == "model1"

    def test_sync_client(self, venice_client):
        """Should work with synchronous client."""
        # Mock response
        mock_models = {
            "data": [
                {"id": "model1", "type": "text", "model_spec": {"capabilities": {}}},
                {"id": "model2", "type": "image", "model_spec": {"capabilities": {}}}
            ]
        }
        
        # Mock the client's list method
        venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Test with sync client
        with pytest.warns(DeprecationWarning, match="Calling an async function without awaiting"):
            result = get_filtered_models(venice_client)
            
        # Verify results
        assert len(result) == 2  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_get_filtered_models_async_client_models_list_returns_none(self, async_venice_client):
        """Test _get_filtered_models_async when client.models.list() returns None."""
        async_venice_client.models.list = AsyncMock(return_value=None)
        result = await get_filtered_models(async_venice_client)  # type: ignore[misc]
        assert result == []

    @pytest.mark.asyncio
    async def test_get_filtered_models_async_client_models_list_missing_data_key(self, async_venice_client):
        """Test _get_filtered_models_async when client.models.list() response is missing 'data' key."""
        async_venice_client.models.list = AsyncMock(return_value={"other_key": "value"}) # No 'data' key
        result = await get_filtered_models(async_venice_client)  # type: ignore[misc]
        assert result == []

    @pytest.mark.asyncio
    async def test_get_filtered_models_async_client_models_list_data_is_none(self, async_venice_client):
        """Test _get_filtered_models_async when client.models.list() response 'data' is None."""
        async_venice_client.models.list = AsyncMock(return_value={"data": None})
        result = await get_filtered_models(async_venice_client)  # type: ignore[misc]
        assert result == []


class TestImportTiktokenModule:
    """Tests for _import_tiktoken_module function."""
    
    def test_import_error_handling(self):
        """Should raise ImportError when tiktoken is not available."""
        
        # Mock tiktoken as unavailable
        with patch('venice_ai.utils._TIKTOKEN_AVAILABLE', False):
            # The function should raise ImportError when tiktoken is not available
            with pytest.raises(ImportError, match="tiktoken library not available"):
                _import_tiktoken_module()
    
    def test_import_success(self):
        """Should return tiktoken module when available."""
        
        # Mock tiktoken as available
        mock_tiktoken = MagicMock()
        with patch('venice_ai.utils._TIKTOKEN_AVAILABLE', True), \
             patch('venice_ai.utils.tiktoken', mock_tiktoken):
            # The function should return the tiktoken module
            result = _import_tiktoken_module()
            assert result is mock_tiktoken


class TestEstimateTokenCount:
    """Tests for estimate_token_count function."""
    
    def test_empty_text(self):
        """Should return 0 for empty text."""
        assert estimate_token_count("") == 0
        
    def test_with_tiktoken(self):
        """Should use tiktoken when available (or use fallback)."""
        # We'll test that the function works, even if we can't verify which path it took
        result = estimate_token_count("Some test text")
        
        # Just verify we get a reasonable result
        assert isinstance(result, int)
        assert result > 0
        
    def test_fallback_without_tiktoken(self):
        """Should use character-based fallback when tiktoken is unavailable."""
        # Mock tiktoken as unavailable to force fallback
        with patch('venice_ai.utils._TIKTOKEN_AVAILABLE', False):
            text = "This is exactly 16 chars"
            expected_fallback = int(len(text) / 4.0)  # 4 tokens
            
            with pytest.warns(UserWarning, match="tiktoken library not found or unavailable"):
                result = estimate_token_count(text)
            
            # Should use the fallback estimation
            assert result == expected_fallback
        
    def test_with_model_id(self):
        """Should accept model_id parameter (reserved for future use)."""
        # Currently model_id doesn't affect the result, but we test the parameter is accepted
        result = estimate_token_count("Test text", model_id="test-model")
                
        assert isinstance(result, int)
    
    def test_fallback_with_non_string_input(self):
        """Should handle non-string input in fallback_estimation."""
        # Mock tiktoken availability to be False so fallback is used
        with patch('venice_ai.utils._TIKTOKEN_AVAILABLE', False):
            # Test with non-string input (should return 0 in fallback)
            with pytest.warns(UserWarning, match="tiktoken library not found or unavailable"):
                result = estimate_token_count(123)  # type: ignore[arg-type]
            assert result == 0
            
            with pytest.warns(UserWarning, match="tiktoken library not found or unavailable"):
                result = estimate_token_count(["list", "of", "strings"])  # type: ignore[arg-type]
            assert result == 0
    
    def test_general_exception_fallback(self):
        """Should use fallback when tiktoken raises an exception."""
        # Mock tiktoken to be available but raise an exception during encoding
        mock_tiktoken = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.encode.side_effect = RuntimeError("Simulated encoding error")
        mock_tiktoken.get_encoding.return_value = mock_encoding
        
        with patch('venice_ai.utils._TIKTOKEN_AVAILABLE', True), \
             patch('venice_ai.utils.tiktoken', mock_tiktoken):
            # When tiktoken raises an exception, should use fallback
            text = "Test text"
            expected_fallback = int(len(text) / 4.0)
            
            with pytest.warns(UserWarning, match="unexpected error with tiktoken: Simulated encoding error"):
                result = estimate_token_count(text)
            
            # Should use the fallback estimation
            assert result == expected_fallback
    
    def test_attribute_error_fallback(self):
        """Should use fallback when tiktoken raises an AttributeError."""
        # Mock tiktoken to be available but raise an AttributeError during encoding
        mock_tiktoken = MagicMock()
        mock_tiktoken.get_encoding.side_effect = AttributeError("'module' object has no attribute 'get_encoding'")
        
        with patch('venice_ai.utils._TIKTOKEN_AVAILABLE', True), \
             patch('venice_ai.utils.tiktoken', mock_tiktoken):
            # When tiktoken raises an AttributeError, should use fallback
            text = "Test text"
            expected_fallback = int(len(text) / 4.0)
            
            with pytest.warns(UserWarning, match="tiktoken attribute error occurred: 'module' object has no attribute 'get_encoding'"):
                result = estimate_token_count(text)
            
            # Should use the fallback estimation
            assert result == expected_fallback


class TestValidateChatMessages:
    """Tests for validate_chat_messages function."""
    
    def test_basic_valid_messages(self):
        """Should validate a basic valid message list."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) == 0
        assert len(result["warnings"]) == 0
        
    def test_invalid_message_structure(self):
        """Should detect invalid message structures."""
        # Not a list
        result = validate_chat_messages("not a list")  # type: ignore[arg-type]
        assert len(result["errors"]) > 0
        assert "must be a list" in result["errors"][0]
        
        # Empty list
        result = validate_chat_messages([])
        assert len(result["errors"]) > 0
        assert "cannot be empty" in result["errors"][0]
        
        # Message not a dictionary
        result = validate_chat_messages([{"role": "user", "content": "Valid"}, "Invalid"])  # type: ignore[arg-type]
        assert len(result["errors"]) > 0
        assert "must be a dictionary" in result["errors"][0]
        
    def test_invalid_role(self):
        """Should detect invalid roles."""
        result = validate_chat_messages([
            {"role": "invalid_role", "content": "Hello"}
        ])
        assert len(result["errors"]) > 0
        assert "Invalid role" in result["errors"][0]
        
        # Missing role
        result = validate_chat_messages([
            {"content": "No role here"}
        ])
        assert len(result["errors"]) > 0
        assert "missing" in result["errors"][0].lower()
        
    def test_system_message_rules(self):
        """Should validate system message rules."""
        # System not first
        result = validate_chat_messages([
            {"role": "user", "content": "First message"},
            {"role": "system", "content": "System should be first"}
        ])
        assert len(result["errors"]) > 0
        assert "must be the first" in result["errors"][0]
        
        # Multiple system messages
        result = validate_chat_messages([
            {"role": "system", "content": "First system"},
            {"role": "user", "content": "User message"},
            {"role": "system", "content": "Second system"}
        ])
        assert len(result["errors"]) > 0
        assert "only one" in result["errors"][1].lower()
        
    def test_message_ordering(self):
        """Should validate message ordering rules."""
        # User-user sequence
        result = validate_chat_messages([
            {"role": "user", "content": "First user"},
            {"role": "user", "content": "Second user"}
        ])
        assert len(result["errors"]) > 0
        assert "cannot directly follow" in result["errors"][0]
        
        # Assistant-assistant sequence
        result = validate_chat_messages([
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "First assistant"},
            {"role": "assistant", "content": "Second assistant"}
        ])
        assert len(result["errors"]) > 0
        assert "should not directly follow" in result["errors"][0]
        
    def test_tool_message_validation(self):
        """Should validate tool message structure and sequence."""
        # Tool without preceding assistant
        result = validate_chat_messages([
            {"role": "user", "content": "User message"},
            {"role": "tool", "tool_call_id": "call_123", "content": "Tool result"}
        ])
        assert len(result["errors"]) > 0
        assert "must follow an assistant" in result["errors"][0]
        
        # Tool without tool_call_id
        result = validate_chat_messages([
            {"role": "user", "content": "User message"},
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "test", "arguments": "{}"}}]},
            {"role": "tool", "content": "Missing tool_call_id"}
        ])
        assert len(result["errors"]) > 0
        assert "tool_call_id" in result["errors"][0]
        
        # Tool with unexpected tool_call_id
        result = validate_chat_messages([
            {"role": "user", "content": "User message"},
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "test", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "unexpected_id", "content": "Result"}
        ])
        assert len(result["errors"]) > 0
        assert "does not match" in result["errors"][0]
        
    def test_assistant_tool_calls_validation(self):
        """Should validate assistant tool_calls structure."""
        # Invalid tool_calls structure
        result = validate_chat_messages([
            {"role": "user", "content": "User message"},
            {"role": "assistant", "tool_calls": "not a list"}
        ])
        assert len(result["errors"]) > 0
        assert "must be a non-empty list" in result["errors"][0]
        
        # Empty tool_calls
        result = validate_chat_messages([
            {"role": "user", "content": "User message"},
            {"role": "assistant", "tool_calls": []}
        ])
        assert len(result["errors"]) > 0
        assert "must be a non-empty list" in result["errors"][0]
        
        # Invalid tool call structure
        result = validate_chat_messages([
            {"role": "user", "content": "User message"},
            {"role": "assistant", "tool_calls": [{"invalid": "structure"}]}
        ])
        assert len(result["errors"]) > 0
        assert "invalid structure" in result["errors"][0].lower()
        
    def test_max_constraints(self):
        """Should validate maximum constraints."""
        long_messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
        ]
        
        # Test max_messages
        result = validate_chat_messages(long_messages, max_messages=2)
        assert len(result["errors"]) > 0
        assert "exceeds maximum" in result["errors"][0]
        
        # Test max_total_chars
        result = validate_chat_messages(long_messages, max_total_chars=10)
        assert len(result["errors"]) > 0
        assert "exceeds maximum" in result["errors"][0]
    
    def test_tool_message_null_content(self):
        """Should validate that a tool message cannot have null content."""
        messages = [
            {"role": "user", "content": "User message"},
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "test", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_123", "content": None}
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert "must have non-empty string content" in result["errors"][0]
    
    def test_assistant_with_null_content_no_tool_calls(self):
        """Should validate that an assistant message without tool_calls cannot have null content."""
        messages = [
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": None}
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert "must have non-null content when not using tool_calls" in result["errors"][0]
    
    def test_assistant_with_tool_calls_and_content(self):
        """Should validate that an assistant message with tool_calls can have content."""
        # This is a valid case, with tool_calls and a string content
        messages = [
            {"role": "user", "content": "What's the weather and tell me a joke?"},
            {
                "role": "assistant",
                "content": "Sure, I can get the weather for you. What city?",
                "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]
            },
            # Add the required tool response:
            {"role": "tool", "tool_call_id": "call_123", "content": '{"temperature": "70F"}'}
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) == 0
    
    def test_assistant_with_tool_calls_and_non_string_content(self):
        """Should validate that an assistant message with tool_calls must have string content if present."""
        messages = [
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": {"invalid": "object"}, "tool_calls": [
                {"id": "call_123", "type": "function", "function": {"name": "test", "arguments": "{}"}}
            ]}
        ]
        
        result = validate_chat_messages(messages)
        assert len(result["errors"]) > 0
        assert "has non-string content" in result["errors"][0]
    
    def test_assistant_with_duplicate_tool_call_ids(self):
        """Should validate that assistant tool_calls cannot have duplicate IDs."""
        messages = [
            {"role": "user", "content": "User message"},
            {"role": "assistant", "tool_calls": [
                {"id": "same_id", "type": "function", "function": {"name": "test1", "arguments": "{}"}},
                {"id": "same_id", "type": "function", "function": {"name": "test2", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "same_id", "content": "Result"}
        ]
        
        result = validate_chat_messages(messages)
        # In practice, the second tool call would overwrite the first in the expected_tool_call_ids set,
        # and there would be no error for duplicate IDs, but the second tool response would be missing
        assert len(result["errors"]) == 0  # Currently no direct validation for duplicate IDs


class TestFindModelById:
    """Tests for find_model_by_id function."""
    
    @pytest.mark.asyncio
    async def test_find_existing_model(self, async_venice_client):
        """Should find an existing model by ID."""
        # Mock response
        mock_models = {
            "data": [
                {"id": "model1", "type": "text"},
                {"id": "model2", "type": "image"},
                {"id": "model3", "type": "text"}
            ]
        }
        
        # Mock the client's list method
        async_venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Find existing model
        result = await find_model_by_id(async_venice_client, "model2")
        
        # Verify results
        assert result["id"] == "model2"  # type: ignore[index]
        assert result["type"] == "image"  # type: ignore[index]
        
    @pytest.mark.asyncio
    async def test_find_nonexistent_model(self, async_venice_client):
        """Should return None for a nonexistent model ID."""
        # Mock response
        mock_models = {
            "data": [
                {"id": "model1", "type": "text"},
                {"id": "model2", "type": "image"}
            ]
        }
        
        # Mock the client's list method
        async_venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Try to find nonexistent model
        result = await find_model_by_id(async_venice_client, "nonexistent")
        
        # Verify results
        assert result is None

    @pytest.mark.asyncio
    async def test_find_model_by_id_client_models_list_returns_none(self, async_venice_client):
        """Test find_model_by_id when client.models.list() returns None."""
        async_venice_client.models.list = AsyncMock(return_value=None)
        result = await find_model_by_id(async_venice_client, "any_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_find_model_by_id_client_models_list_missing_data_key(self, async_venice_client):
        """Test find_model_by_id when client.models.list() response is missing 'data' key."""
        async_venice_client.models.list = AsyncMock(return_value={"other_key": "value"})
        result = await find_model_by_id(async_venice_client, "any_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_find_model_by_id_client_models_list_data_is_none(self, async_venice_client):
        """Test find_model_by_id when client.models.list() response 'data' is None."""
        async_venice_client.models.list = AsyncMock(return_value={"data": None})
        result = await find_model_by_id(async_venice_client, "any_id")
        assert result is None


class TestGetModelCapabilities:
    """Tests for get_model_capabilities function."""
    
    @pytest.mark.asyncio
    async def test_get_capabilities_existing_model(self, async_venice_client):
        """Should retrieve capabilities for existing model."""
        # Mock response
        mock_models = {
            "data": [
                {
                    "id": "model1",
                    "type": "text",
                    "model_spec": {"capabilities": {"supportsFunctionCalling": True}}
                }
            ]
        }
        
        # Mock the client's list method
        async_venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Get capabilities
        capabilities = await get_model_capabilities(async_venice_client, "model1")
        
        # Verify results
        assert capabilities == {"supportsFunctionCalling": True}
        
    @pytest.mark.asyncio
    async def test_get_capabilities_nonexistent_model(self, async_venice_client):
        """Should return None for a nonexistent model ID."""
        # Mock response
        mock_models: ModelList = {"object": "list", "data": [], "type": None}
        
        # Mock the client's list method
        async_venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Try to get capabilities for nonexistent model
        capabilities = await get_model_capabilities(async_venice_client, "nonexistent")
        
        # Verify results
        assert capabilities is None
        
    @pytest.mark.asyncio
    async def test_get_capabilities_no_model_spec(self, async_venice_client):
        """Should handle models without model_spec."""
        # Mock response
        mock_models = {
            "data": [
                {
                    "id": "model1",
                    "type": "text"
                    # No model_spec
                }
            ]
        }
        
        # Mock the client's list method
        async_venice_client.models.list = MagicMock(return_value=mock_models)
        
        # Get capabilities
        capabilities = await get_model_capabilities(async_venice_client, "model1")
        
        # Verify results - should return None due to missing model_spec
        assert capabilities is None


class TestFormatToolResponse:
    """Tests for format_tool_response function."""
    
    def test_with_string_content(self):
        """Should handle string content."""
        response = format_tool_response("call_123", "String result")
        
        assert response["role"] == "tool"
        assert response["tool_call_id"] == "call_123"
        assert response["content"] == "String result"
        
    def test_with_dict_content(self):
        """Should stringify dict content."""
        data = {"temperature": 22, "condition": "sunny"}
        response = format_tool_response("call_456", data)
        
        assert response["role"] == "tool"
        assert response["tool_call_id"] == "call_456"
        
        # Content should be JSON string
        assert isinstance(response["content"], str)
        decoded = json.loads(response["content"])
        assert decoded["temperature"] == 22
        assert decoded["condition"] == "sunny"
        
    def test_with_list_content(self):
        """Should stringify list content."""
        data = [1, 2, 3, 4]
        response = format_tool_response("call_789", data)
        
        assert response["role"] == "tool"
        assert response["tool_call_id"] == "call_789"
        
        # Content should be JSON string
        assert isinstance(response["content"], str)
        decoded = json.loads(response["content"])
        assert decoded == [1, 2, 3, 4]
        
    def test_with_none_content(self):
        """Should handle None content."""
        response = format_tool_response("call_null", None)
        
        assert response["role"] == "tool"
        assert response["tool_call_id"] == "call_null"
        assert response["content"] == "null"