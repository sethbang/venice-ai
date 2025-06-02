import pytest
from typing import List, Dict, Any
from unittest.mock import patch
from venice_ai.utils import validate_chat_messages


class TestNumTokensFromMessages:
    """Test cases for num_tokens_from_messages function (targeting lines 15-17 effect)"""
    
    def test_num_tokens_from_messages_tiktoken_unavailable_effect(self):
        """
        Test Case 1.1: Simulate the state where tiktoken failed to import
        (covering the effect of lines 15-17) and verify num_tokens_from_messages returns 0.
        """
        # Note: Since num_tokens_from_messages doesn't exist in the current codebase,
        # this test is implemented as specified but will need the function to be added
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch('venice_ai.utils._TIKTOKEN_AVAILABLE', False), \
             patch('venice_ai.utils.tiktoken', None):
            # This would test the function when it exists
            # assert num_tokens_from_messages(messages, "gpt-3.5-turbo-0613") == 0
            pass  # Placeholder until function is implemented
    
    def test_num_tokens_from_messages_tiktoken_available_model_not_in_tiktoken_models(self):
        """
        Test Case 1.2: Test behavior when tiktoken is available but the model is not in TIKTOKEN_MODELS.
        """
        messages = [{"role": "user", "content": "Hello"}]
        
        # Assuming tiktoken is actually available in the test environment,
        # or mock it to be available but ensure TIKTOKEN_MODELS doesn't include the test model.
        with patch('venice_ai.utils._TIKTOKEN_AVAILABLE', True):
             # Mock tiktoken.get_encoding to ensure it's not called for an unknown model
             with patch('venice_ai.utils.tiktoken.get_encoding', side_effect=Exception("Should not be called")) as mock_get_encoding:
                 # This would test the function when it exists
                 # assert num_tokens_from_messages(messages, "some-other-model") == 0
                 # mock_get_encoding.assert_not_called()
                 pass  # Placeholder until function is implemented


class TestValidateChatMessages:
    """Test cases for validate_chat_messages function (targeting specific missed lines)"""
    
    def test_validate_chat_messages_invalid_tool_call_function_type(self):
        """
        Test Case 2.1: Cover lines 381-382.
        """
        messages = [{"role": "assistant", "tool_calls": [{"id": "tc1", "type": "function", "function": "not_a_dict"}]}]
        
        result = validate_chat_messages(messages)
        assert "errors" in result
        assert len(result["errors"]) > 0
        # Check that the error message contains information about invalid function field type
        error_found = any("invalid 'function' field type" in error for error in result["errors"])
        assert error_found, f"Expected error about invalid function field type, got: {result['errors']}"
    
    def test_validate_chat_messages_invalid_tool_call_function_arguments_type(self):
        """
        Test Case 2.2: Cover line 414 (arguments validation).
        """
        messages = [{"role": "assistant", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "f1", "arguments": 123}}]}]
        
        result = validate_chat_messages(messages)
        assert "errors" in result
        assert len(result["errors"]) > 0
        # Check that the error message contains information about invalid arguments field type
        error_found = any("invalid 'arguments' field type" in error for error in result["errors"])
        assert error_found, f"Expected error about invalid arguments field type, got: {result['errors']}"
    
    def test_validate_chat_messages_invalid_tool_message_tool_call_id_type(self):
        """
        Test Case 2.3: Cover line 448 (tool_call_id validation).
        """
        messages = [{"role": "tool", "tool_call_id": 123, "content": "result"}]
        
        result = validate_chat_messages(messages)
        assert "errors" in result
        assert len(result["errors"]) > 0
        # Check that the error message contains information about invalid tool_call_id field
        error_found = any("invalid 'tool_call_id' field" in error for error in result["errors"])
        assert error_found, f"Expected error about invalid tool_call_id field, got: {result['errors']}"
    
    def test_validate_chat_messages_invalid_tool_message_content_type(self):
        """
        Test Case 2.4: Cover line 462 (tool content validation).
        """
        messages = [{"role": "tool", "tool_call_id": "tc1", "content": 123}]
        
        result = validate_chat_messages(messages)
        assert "errors" in result
        assert len(result["errors"]) > 0
        # Check that the error message contains information about non-empty string content requirement
        error_found = any("must have non-empty string content" in error for error in result["errors"])
        assert error_found, f"Expected error about non-empty string content, got: {result['errors']}"
    
    def test_validate_chat_messages_assistant_missing_content_and_tool_calls(self):
        """
        Test Case 2.5: Cover line 436 (assistant must have content or tool_calls).
        """
        messages = [{"role": "assistant"}]
        
        result = validate_chat_messages(messages)
        assert "errors" in result
        assert len(result["errors"]) > 0
        # Check that the error message contains information about missing content or tool_calls
        error_found = any("must have either non-empty content or tool_calls" in error for error in result["errors"])
        assert error_found, f"Expected error about missing content or tool_calls, got: {result['errors']}"
    
    def test_validate_chat_messages_tool_missing_tool_call_id(self):
        """
        Test Case 2.6: Cover line 446 (tool missing tool_call_id).
        """
        messages = [{"role": "tool", "content": "result"}]
        
        result = validate_chat_messages(messages)
        assert "errors" in result
        assert len(result["errors"]) > 0
        # Check that the error message contains information about missing tool_call_id field
        error_found = any("missing required 'tool_call_id' field" in error for error in result["errors"])
        assert error_found, f"Expected error about missing tool_call_id field, got: {result['errors']}"
    
    def test_validate_chat_messages_invalid_role_value(self):
        """
        Test Case 2.7: Cover line 311 (invalid role value).
        """
        messages = [{"role": "invalid_role", "content": "c"}]
        
        result = validate_chat_messages(messages)
        assert "errors" in result
        assert len(result["errors"]) > 0
        # Check that the error message contains information about invalid role
        error_found = any("Invalid role 'invalid_role'" in error for error in result["errors"])
        assert error_found, f"Expected error about invalid role, got: {result['errors']}"
    
    def test_validate_chat_messages_invalid_content_type_for_user_role(self):
        """
        Test Case 2.8: Cover line 359 (user content validation).
        """
        messages = [{"role": "user", "content": 123}]
        
        result = validate_chat_messages(messages)
        assert "errors" in result
        assert len(result["errors"]) > 0
        # Check that the error message contains information about content type requirement
        error_found = any("must have either string or list content" in error for error in result["errors"])
        assert error_found, f"Expected error about content type, got: {result['errors']}"
    
    def test_validate_chat_messages_invalid_role_type(self):
        """
        Test Case 2.9: Cover line 306 (role type validation).
        """
        messages = [{"role": 123, "content": "c"}]
        
        result = validate_chat_messages(messages)
        assert "errors" in result
        assert len(result["errors"]) > 0
        # Check that the error message contains information about role being a string
        error_found = any("must be a string" in error for error in result["errors"])
        assert error_found, f"Expected error about role being a string, got: {result['errors']}"
    
    def test_validate_chat_messages_message_not_a_dict(self):
        """
        Test Case 2.10: Cover line 294 (message must be dictionary).
        """
        messages = ["not_a_dict"]
        
        result = validate_chat_messages(messages)  # type: ignore[arg-type]
        assert "errors" in result
        assert len(result["errors"]) > 0
        # Check that the error message contains information about message being a dictionary
        error_found = any("must be a dictionary" in error for error in result["errors"])
        assert error_found, f"Expected error about message being a dictionary, got: {result['errors']}"
    
    def test_validate_chat_messages_invalid_tool_call_id_type(self):
        """
        Test Case 2.11: Cover line 389 (tool_call id validation).
        """
        messages = [{"role": "assistant", "tool_calls": [{"id": 123, "type": "function", "function": {"name": "f1", "arguments": "{}"}}]}]
        
        result = validate_chat_messages(messages)
        assert "errors" in result
        assert len(result["errors"]) > 0
        # Check that the error message contains information about invalid id field
        error_found = any("invalid 'id' field" in error for error in result["errors"])
        assert error_found, f"Expected error about invalid id field, got: {result['errors']}"
    
    def test_validate_chat_messages_empty_messages_list(self):
        """
        Test Case 2.12: Ensure an empty list returns validation errors (covers lines 282-284).
        """
        messages: List[Dict[str, Any]] = []
        
        result = validate_chat_messages(messages)
        assert "errors" in result
        assert len(result["errors"]) > 0
        # Check that the error message contains information about empty messages list
        error_found = any("Messages list cannot be empty" in error for error in result["errors"])
        assert error_found, f"Expected error about empty messages list, got: {result['errors']}"
    
    def test_validate_chat_messages_valid_complex_case(self):
        """
        Test Case 2.13: Ensure a valid, complex structure passes.
        """
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User query"},
            {"role": "assistant", "tool_calls": [
                {"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "Boston"}'}}
            ]},
            {"role": "tool", "tool_call_id": "call_123", "content": '{"temperature": "70F"}'},
            {"role": "assistant", "content": "The weather in Boston is 70F."}
        ]
        
        result = validate_chat_messages(messages)
        assert "errors" in result
        assert "warnings" in result
        # Valid messages should have no errors
        assert len(result["errors"]) == 0, f"Expected no errors for valid messages, got: {result['errors']}"