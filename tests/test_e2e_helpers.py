"""
Tests for utility functions in the e2e_tests.utils.helpers module.
"""

import os
import json
import pytest
import pathlib
from unittest.mock import patch, MagicMock, AsyncMock

from venice_ai import VeniceClient, AsyncVeniceClient

# Import the helpers we want to test
from e2e_tests.utils.helpers import (
    get_test_model_id,
    generate_sample_messages,
    assert_chat_completion_structure,
    assert_tool_call_structure,
    load_test_data,
    create_temp_test_file,
    get_model_capabilities_for_test
)


class TestGetTestModelId:
    """Tests for get_test_model_id function."""
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"E2E_TEXT_MODEL_TOOL_CALLS": "env-model-id"})
    async def test_environment_variable_override(self, venice_client):
        """Should use environment variable if available."""
        with patch("e2e_tests.utils.helpers.get_filtered_models") as mock_filter:
            # Add additional debug output
            print(f"DEBUG: Testing with E2E_TEXT_MODEL_TOOL_CALLS={os.environ.get('E2E_TEXT_MODEL_TOOL_CALLS')}")
            
            # Call the function with text type
            model_id = await get_test_model_id(venice_client, model_type="text")
            
            # Print the actual result
            print(f"DEBUG: get_test_model_id returned: {model_id}")
            
            # Verify it used the environment variable value
            assert model_id == "env-model-id", f"Expected 'env-model-id' but got '{model_id}'"
            
            # Verify get_filtered_models was not called
            mock_filter.assert_not_called()
            
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"E2E_IMAGE_MODEL": "env-image-model"})
    async def test_environment_variable_match_type(self, venice_client):
        """Should only use environment variable that matches the type."""
        with patch("e2e_tests.utils.helpers.get_filtered_models") as mock_filter:
            mock_filter.return_value = [{"id": "fallback-model"}]
            
            # Call with text type (no matching env var)
            model_id = await get_test_model_id(venice_client, model_type="text")
            
            # Should use filtered models fallback
            assert model_id == "fallback-model"
            mock_filter.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_preferred_models_match(self, venice_client):
        """Should use first matching preferred model."""
        # Create side effect functions that handle both sync and async contexts
        async def async_compatible_find(client, model_id): # Renamed to async_compatible_find
            if model_id == "model1":
                return {"id": "model1", "type": "image"}
            elif model_id == "model2":
                return {"id": "model2", "type": "text"}
            elif model_id == "model3":
                return {"id": "model3", "type": "text"}
            return None
            
        async def async_compatible_capabilities(client, model_id): # Renamed to async_compatible_capabilities
            if model_id == "model2":
                return {"supportsFunctionCalling": True}
            return None
        
        # Create mocks that work for both sync and async
        with patch("e2e_tests.utils.helpers.find_model_by_id", new_callable=AsyncMock if isinstance(venice_client, AsyncVeniceClient) else MagicMock) as mock_find:
            with patch("e2e_tests.utils.helpers.get_model_capabilities", new_callable=AsyncMock if isinstance(venice_client, AsyncVeniceClient) else MagicMock) as mock_capabilities:
                # Set up mocks that work in both contexts
                mock_find.side_effect = async_compatible_find # Changed to async_compatible_find
                mock_capabilities.side_effect = async_compatible_capabilities # Changed to async_compatible_capabilities
                
                # Call with preferred models
                model_id = await get_test_model_id( # Added await
                    venice_client,
                    model_type="text",
                    required_capabilities=["supportsFunctionCalling"],
                    preferred_models=["model1", "model2", "model3"]
                )
                
                # Should return the first valid preferred model
                assert model_id == "model2"
                
                # Should have checked both models
                mock_find.assert_any_call(venice_client, "model1")
                mock_find.assert_any_call(venice_client, "model2")
                
                # Should have checked capabilities only for model2
                mock_capabilities.assert_called_once_with(venice_client, "model2")
                
    @pytest.mark.asyncio
    async def test_preferred_models_no_match(self, venice_client):
        """Should fall back to filtered models if no preferred models match."""
        # Create side effect functions that handle both sync and async contexts
        async def async_compatible_find(client, model_id): # Renamed to async_compatible_find
            if model_id == "model1":
                return {"id": "model1", "type": "image"}
            elif model_id == "model2":
                return {"id": "model2", "type": "text"}
            return None
            
        async def async_compatible_capabilities(client, model_id): # Renamed to async_compatible_capabilities
            return {"supportsFunctionCalling": False}
            
        async def async_compatible_filter(client, model_type=None, supports_capabilities=None): # Renamed to async_compatible_filter
            # Make sure we're returning the expected value for the test
            print(f"DEBUG: Mock filter called with {model_type}, {supports_capabilities}")
            return [{"id": "filtered-model"}]
        
        with patch("e2e_tests.utils.helpers.find_model_by_id", new_callable=AsyncMock if isinstance(venice_client, AsyncVeniceClient) else MagicMock) as mock_find:
            with patch("e2e_tests.utils.helpers.get_model_capabilities", new_callable=AsyncMock if isinstance(venice_client, AsyncVeniceClient) else MagicMock) as mock_capabilities:
                with patch("e2e_tests.utils.helpers.get_filtered_models", new_callable=AsyncMock if isinstance(venice_client, AsyncVeniceClient) else MagicMock) as mock_filter:
                    # Set up mocks that work in both contexts
                    mock_find.side_effect = async_compatible_find # Changed to async_compatible_find
                    mock_capabilities.side_effect = async_compatible_capabilities # Changed to async_compatible_capabilities
                    mock_filter.side_effect = async_compatible_filter # Changed to async_compatible_filter
                    
                    # Call with preferred models
                    model_id = await get_test_model_id( # Added await
                        venice_client,
                        model_type="text",
                        required_capabilities=["supportsFunctionCalling"],
                        preferred_models=["model1", "model2"]
                    )
                    
                    # Should fall back to filtered model
                    assert model_id == "filtered-model"
                    
                    # Should call get_filtered_models with correct params
                    mock_filter.assert_called_once_with(
                        venice_client,
                        "text",
                        ["supportsFunctionCalling"]
                    )
                    
    @pytest.mark.asyncio
    async def test_dynamic_selection(self, venice_client):
        """Should use get_filtered_models if no env var or preferred models."""
        with patch("e2e_tests.utils.helpers.get_filtered_models") as mock_filter:
            mock_filter.return_value = [{"id": "dynamic-model"}]
            
            # Call without env vars or preferred models
            model_id = await get_test_model_id(venice_client, model_type="text") # Added await
            
            # Should use filtered model
            assert model_id == "dynamic-model"
            mock_filter.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_no_matching_models(self, venice_client):
        """Should skip test if no matching models are found."""
        with patch("e2e_tests.utils.helpers.get_filtered_models") as mock_filter:
            mock_filter.return_value = []  # No models found
            
            # Should raise pytest.skip
            with pytest.raises(pytest.skip.Exception):
                await get_test_model_id(
                    venice_client,
                    model_type="text",
                    required_capabilities=["nonExistentCapability"]
                )


class TestGenerateSampleMessages:
    """Tests for generate_sample_messages function."""
    
    def test_single_user_message(self):
        """Should generate a single user message by default."""
        messages = generate_sample_messages()
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Test message 1"
        
    def test_multiple_messages(self):
        """Should generate multiple user messages when count > 1."""
        messages = generate_sample_messages(count=3)
        
        assert len(messages) == 3
        for i, msg in enumerate(messages, 1):
            assert msg["role"] == "user"
            assert msg["content"] == f"Test message {i}"
            
    def test_with_system_prompt(self):
        """Should include system prompt when requested."""
        messages = generate_sample_messages(count=2, include_system_prompt=True)
        
        assert len(messages) == 3  # system + 2 user
        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"].lower()
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "user"
        
    def test_custom_content_prefix(self):
        """Should use custom content prefix when provided."""
        messages = generate_sample_messages(count=2, content_prefix="Custom prefix")
        
        assert len(messages) == 2
        assert messages[0]["content"] == "Custom prefix 1"
        assert messages[1]["content"] == "Custom prefix 2"


class TestAssertChatCompletionStructure:
    """Tests for assert_chat_completion_structure function."""
    
    def test_full_response_structure(self):
        """Should validate a valid full response structure."""
        # Create a valid full response
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1615221692,
            "model": "model-id-1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello there!"
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        
        # This should not raise an assertion error
        assert_chat_completion_structure(response)
        
        # Test with usage
        response["usage"] = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
        assert_chat_completion_structure(response)
        
    def test_full_response_with_tool_calls(self):
        """Should validate a full response with tool calls."""
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1615221692,
            "model": "model-id-1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'll check the weather.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"location\":\"New York\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ]
        }
        
        # This should not raise an assertion error
        assert_chat_completion_structure(response)
        
    def test_streaming_chunk_structure(self):
        """Should validate a valid streaming chunk structure."""
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1615221692,
            "model": "model-id-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "Hello"
                    },
                    "finish_reason": None
                }
            ]
        }
        
        # This should not raise an assertion error
        assert_chat_completion_structure(chunk, is_streaming_chunk=True)
        
    def test_first_streaming_chunk_with_role(self):
        """Should validate a first streaming chunk with role."""
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1615221692,
            "model": "model-id-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant"
                    },
                    "finish_reason": None
                }
            ]
        }
        
        # This should not raise an assertion error
        assert_chat_completion_structure(chunk, is_streaming_chunk=True)
        
    def test_streaming_chunk_with_empty_delta_and_finish_reason(self):
        """Should validate a streaming chunk with empty delta and finish reason."""
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1615221692,
            "model": "model-id-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        
        # This should not raise an assertion error
        assert_chat_completion_structure(chunk, is_streaming_chunk=True)
        
    def test_final_usage_chunk(self):
        """Should validate a final usage chunk with empty choices."""
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1615221692,
            "model": "model-id-1",
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        
        # This should not raise an assertion error
        assert_chat_completion_structure(chunk, is_streaming_chunk=True, is_final_usage_chunk=True)
        
    def test_streaming_chunk_with_tool_calls(self):
        """Should validate a streaming chunk with tool_calls."""
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1615221692,
            "model": "model-id-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"lo"
                                }
                            }
                        ]
                    },
                    "finish_reason": None
                }
            ]
        }
        
        # This should not raise an assertion error
        assert_chat_completion_structure(chunk, is_streaming_chunk=True)
        
    def test_invalid_response_structure(self):
        """Should raise assertion errors for invalid structures."""
        # Missing required fields
        invalid_response = {
            "id": "chatcmpl-123",
            # missing 'object'
            "created": 1615221692,
            "choices": []
        }
        
        # This should raise an assertion error
        with pytest.raises(AssertionError):
            assert_chat_completion_structure(invalid_response)
            
        # Invalid object type
        invalid_response = {
            "id": "chatcmpl-123",
            "object": "wrong_type",  # not chat.completion
            "created": 1615221692,
            "model": "model-id-1",
            "choices": []
        }
        
        with pytest.raises(AssertionError):
            assert_chat_completion_structure(invalid_response)


class TestAssertToolCallStructure:
    """Tests for assert_tool_call_structure function."""
    
    def test_valid_full_tool_call(self):
        """Should validate a valid full tool call structure."""
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\":\"New York\"}"
            }
        }
        
        # This should not raise an assertion error
        assert_tool_call_structure(tool_call)
        
    def test_valid_delta_tool_call(self):
        """Should validate a valid delta tool call structure."""
        tool_call_delta = {
            "index": 0,
            "id": "call_123",
            "function": {
                "name": "get_weather"
            }
        }
        
        # This should not raise an assertion error
        assert_tool_call_structure(tool_call_delta, is_delta=True)
        
        # Minimal delta with just index
        minimal_delta = {"index": 0}
        assert_tool_call_structure(minimal_delta, is_delta=True)
        
    def test_invalid_full_tool_call(self):
        """Should raise assertion errors for invalid full tool call structures."""
        # Missing id
        invalid_tool_call = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{}"
            }
        }
        
        with pytest.raises(AssertionError):
            assert_tool_call_structure(invalid_tool_call)
            
        # Wrong type
        invalid_tool_call = {
            "id": "call_123",
            "type": "not_function",
            "function": {
                "name": "get_weather",
                "arguments": "{}"
            }
        }
        
        with pytest.raises(AssertionError):
            assert_tool_call_structure(invalid_tool_call)
            
        # Missing function name
        invalid_tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {
                "arguments": "{}"
            }
        }
        
        with pytest.raises(AssertionError):
            assert_tool_call_structure(invalid_tool_call)
            
    def test_invalid_delta_tool_call(self):
        """Should raise assertion errors for invalid delta tool call structures."""
        # Missing index
        invalid_delta = {
            "id": "call_123"
        }
        
        with pytest.raises(AssertionError):
            assert_tool_call_structure(invalid_delta, is_delta=True)


class TestLoadTestData:
    """Tests for load_test_data function."""
    
    @patch('pathlib.Path.read_bytes')
    def test_load_binary_data(self, mock_read_bytes):
        """Should load binary data with default mode."""
        mock_read_bytes.return_value = b'binary data'
        
        data = load_test_data("sample_file.bin")
        
        assert data == b'binary data'
        mock_read_bytes.assert_called_once()
        
    @patch('pathlib.Path.read_text')
    def test_load_text_data(self, mock_read_text):
        """Should load text data with text mode."""
        mock_read_text.return_value = 'text data'
        
        data = load_test_data("sample_file.txt", mode="r")
        
        assert data == 'text data'
        mock_read_text.assert_called_once()
        
    @patch('pathlib.Path.exists')
    def test_file_not_found(self, mock_exists):
        """Should raise FileNotFoundError if file doesn't exist."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            load_test_data("nonexistent_file.txt")
            
    @patch('pathlib.Path')
    def test_custom_data_dir(self, mock_path):
        """Should use custom data directory if provided."""
        # Set up mock
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.__truediv__.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True
        mock_path_instance.read_bytes.return_value = b'data'
        
        # Call with custom data_dir
        load_test_data("file.txt", data_dir="custom/dir")
        
        # Verify Path was called with custom dir
        mock_path.assert_called_once_with("custom/dir")
        mock_path_instance.__truediv__.assert_called_once_with("file.txt")


class TestCreateTempTestFile:
    """Tests for create_temp_test_file function."""
    
    def test_create_text_file(self, tmp_path):
        """Should create a text file with string content."""
        file_path = create_temp_test_file(tmp_path, "test.txt", "Hello, world!")
        
        assert file_path.exists()
        assert file_path.read_text() == "Hello, world!"
        
    def test_create_binary_file(self, tmp_path):
        """Should create a binary file with bytes content."""
        data = b'\x00\x01\x02\x03'
        file_path = create_temp_test_file(tmp_path, "test.bin", data)
        
        assert file_path.exists()
        assert file_path.read_bytes() == data
        
    def test_create_with_subdirectories(self, tmp_path):
        """Should create file with path containing subdirectories."""
        file_path = create_temp_test_file(tmp_path, "subdir/test.txt", "Test content")
        
        assert file_path.exists()
        assert file_path.read_text() == "Test content"
        
    def test_custom_encoding(self, tmp_path):
        """Should use custom encoding for text files."""
        text = "Special characters: é ñ ß"
        file_path = create_temp_test_file(tmp_path, "encoded.txt", text, encoding="latin-1")
        
        assert file_path.exists()
        # Read it back with the same encoding
        assert file_path.read_text(encoding="latin-1") == text


class TestGetModelCapabilitiesForTest:
    """Tests for get_model_capabilities_for_test function."""
    
    @pytest.mark.asyncio
    async def test_get_capabilities_success(self, async_venice_client):
        """Should successfully get capabilities."""
        # Mock get_model_capabilities
        with patch("e2e_tests.utils.helpers.get_model_capabilities") as mock_get_caps:
            mock_get_caps.return_value = {"supportsFunctionCalling": True}
            
            # Call the function
            capabilities = await get_model_capabilities_for_test(async_venice_client, "test-model")
            
            # Verify results
            assert capabilities == {"supportsFunctionCalling": True}
            mock_get_caps.assert_called_once_with(async_venice_client, "test-model")
            
    @pytest.mark.asyncio
    async def test_get_capabilities_failure(self, async_venice_client):
        """Should fail test if capabilities are unexpectedly None."""
        # Mock get_model_capabilities
        with patch("e2e_tests.utils.helpers.get_model_capabilities") as mock_get_caps:
            mock_get_caps.return_value = None
            
            # Should raise pytest.fail
            with pytest.raises(pytest.fail.Exception):
                await get_model_capabilities_for_test(async_venice_client, "nonexistent-model")