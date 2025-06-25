import pytest
import httpx
import json
from unittest.mock import patch, MagicMock, PropertyMock
from pydantic import BaseModel, ValidationError
from typing import Optional, List

from venice_ai._client import VeniceClient
from venice_ai.exceptions import APIResponseProcessingError
from venice_ai.types.chat import ChatCompletionChunk


class MockModel(BaseModel):
    """Mock Pydantic model for testing."""
    id: str
    value: int
    required_field: str


class StrictMockModel(BaseModel):
    """Strict mock model that will fail validation easily."""
    id: int  # Expecting int, but API might return string
    strict_value: float
    nested: dict


class TestClientPydanticCastingErrors:
    """Test Pydantic model casting error handling in VeniceClient."""
    
    @pytest.fixture
    def client(self):
        return VeniceClient(api_key="test-api-key")
    
    @pytest.fixture
    def mock_httpx_client(self, client):
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.headers = httpx.Headers({
            "Authorization": "Bearer test-api-key",
            "Accept": "application/json"
        })
        client._client = mock_client
        return mock_client
    
    def test_request_cast_to_validation_error(self, client, mock_httpx_client):
        """Test _request with cast_to that fails validation."""
        # Mock response with data that won't validate
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "123",
            "value": "not_an_int",  # This will fail int validation
            "missing_required_field": True  # Missing required_field
        }
        mock_httpx_client.request.return_value = mock_response
        
        with pytest.raises(APIResponseProcessingError) as exc_info:
            client._request("GET", "test", cast_to=MockModel)
        
        error = exc_info.value
        assert "Failed to cast response to" in str(error)
        assert "MockModel" in str(error)
        assert error.response == mock_response
        assert isinstance(error.original_error, ValidationError)
    
    def test_request_cast_to_type_error(self, client, mock_httpx_client):
        """Test _request with cast_to that raises TypeError."""
        # Mock response with completely wrong structure
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = "not a dict"  # String instead of dict
        mock_httpx_client.request.return_value = mock_response
        
        with pytest.raises(APIResponseProcessingError) as exc_info:
            client._request("GET", "test", cast_to=MockModel)
        
        error = exc_info.value
        assert "Failed to cast response to" in str(error)
        assert error.response == mock_response
    
    def test_post_with_cast_to_error(self, client, mock_httpx_client):
        """Test post method with cast_to that fails."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "should_be_int",  # StrictMockModel expects int
            "strict_value": "not_a_float",
            "nested": "not_a_dict"
        }
        mock_httpx_client.request.return_value = mock_response
        
        with pytest.raises(APIResponseProcessingError) as exc_info:
            client.post("endpoint", json_data={"key": "value"}, cast_to=StrictMockModel)
        
        error = exc_info.value
        assert "Failed to cast response to" in str(error)
        assert "StrictMockModel" in str(error)
    
    def test_get_with_cast_to_error(self, client, mock_httpx_client):
        """Test get method with cast_to that fails."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"incomplete": "data"}
        mock_httpx_client.request.return_value = mock_response
        
        with pytest.raises(APIResponseProcessingError) as exc_info:
            client.get("endpoint", cast_to=MockModel)
        
        error = exc_info.value
        assert "Failed to cast response to" in str(error)
    
    def test_delete_with_cast_to_error(self, client, mock_httpx_client):
        """Test delete method with cast_to that fails."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = None  # None response
        mock_httpx_client.request.return_value = mock_response
        
        with pytest.raises(APIResponseProcessingError) as exc_info:
            client.delete("endpoint", cast_to=MockModel)
        
        error = exc_info.value
        assert "Failed to cast response to" in str(error)
    
    def test_request_multipart_cast_to_error(self, client, mock_httpx_client):
        """Test _request_multipart with cast_to that fails validation."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "id": 123,  # Wrong type for MockModel (expects string)
            "value": 456,
            "required_field": None  # None for required field
        }
        mock_httpx_client.request.return_value = mock_response
        
        files = {"file": ("test.txt", b"content", "text/plain")}
        
        with pytest.raises(APIResponseProcessingError) as exc_info:
            client._request_multipart("POST", "upload", files=files, cast_to=MockModel)
        
        error = exc_info.value
        assert "Failed to cast multipart response to" in str(error)
        assert "MockModel" in str(error)
        assert error.response == mock_response
        assert isinstance(error.original_error, Exception)
    
    def test_request_multipart_cast_to_attribute_error(self, client, mock_httpx_client):
        """Test _request_multipart with cast_to that raises AttributeError."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = []  # List instead of dict
        mock_httpx_client.request.return_value = mock_response
        
        files = {"file": ("test.jpg", b"image_data", "image/jpeg")}
        
        with pytest.raises(APIResponseProcessingError) as exc_info:
            client._request_multipart("POST", "upload", files=files, cast_to=StrictMockModel)
        
        error = exc_info.value
        assert "Failed to cast multipart response to" in str(error)
        assert "StrictMockModel" in str(error)
    
    def test_stream_request_cast_to_error(self, client, mock_httpx_client):
        """Test _stream_request with cast_to that fails for SSE chunks."""
        # Mock the stream context manager
        mock_stream = MagicMock()
        mock_httpx_client.stream.return_value.__enter__.return_value = mock_stream
        mock_stream.status_code = 200
        
        # Mock SSE data that won't validate against the model
        mock_stream.iter_lines.return_value = [
            b"data: {\"id\": \"123\", \"value\": \"not_int\", \"missing_field\": true}",
            b"data: {\"id\": \"456\", \"value\": null}",  # null for required int
            b"data: [DONE]"
        ]
        
        # Collect results - the generator should skip invalid chunks
        results = list(client._stream_request("POST", "stream", json_data={}, cast_to=MockModel))
        
        # Should have processed lines but skipped invalid ones
        assert len(results) == 0  # All chunks failed validation and were skipped
    
    def test_stream_request_cast_to_with_valid_and_invalid(self, client, mock_httpx_client):
        """Test _stream_request with mix of valid and invalid chunks."""
        # Mock the stream context manager
        mock_stream = MagicMock()
        mock_httpx_client.stream.return_value.__enter__.return_value = mock_stream
        mock_stream.status_code = 200
        
        # Mock SSE data with mix of valid and invalid
        mock_stream.iter_lines.return_value = [
            b"data: {\"id\": \"valid1\", \"value\": 100, \"required_field\": \"present\"}",  # Valid
            b"data: {\"id\": \"invalid\", \"value\": \"not_int\"}",  # Invalid - missing required_field
            b"data: {\"id\": \"valid2\", \"value\": 200, \"required_field\": \"also_present\"}",  # Valid
            b"data: [DONE]"
        ]
        
        # Collect results
        results = list(client._stream_request("POST", "stream", json_data={}, cast_to=MockModel))
        
        # Should have only valid chunks
        assert len(results) == 2
        assert results[0].id == "valid1"
        assert results[0].value == 100
        assert results[1].id == "valid2"
        assert results[1].value == 200
    
    def test_stream_request_default_cast_behavior(self, client, mock_httpx_client):
        """Test _stream_request without cast_to defaults to ChatCompletionChunk."""
        # Mock the stream context manager
        mock_stream = MagicMock()
        mock_httpx_client.stream.return_value.__enter__.return_value = mock_stream
        mock_stream.status_code = 200
        
        # Mock SSE data
        mock_stream.iter_lines.return_value = [
            b"data: {\"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}",
            b"data: {\"choices\": [{\"delta\": {\"content\": \" world\"}}]}",
            b"data: [DONE]"
        ]
        
        # Collect results - should be cast as ChatCompletionChunk by default
        results = list(client._stream_request("POST", "stream", json_data={}))
        
        assert len(results) == 2
        # Results should be ChatCompletionChunk objects when no cast_to is provided
        from venice_ai.types.chat import ChatCompletionChunk
        assert isinstance(results[0], ChatCompletionChunk)
        assert hasattr(results[0], 'choices')
        assert results[0].choices[0].delta.content == "Hello"
        assert results[1].choices[0].delta.content == " world"
    
    def test_stream_request_json_decode_error_in_chunk(self, client, mock_httpx_client):
        """Test _stream_request handling of malformed JSON in SSE chunk."""
        # Mock the stream context manager
        mock_stream = MagicMock()
        mock_httpx_client.stream.return_value.__enter__.return_value = mock_stream
        mock_stream.status_code = 200
        
        # Mock SSE data with invalid JSON
        mock_stream.iter_lines.return_value = [
            b'data: {"valid": "json", "id": "1", "value": 1, "required_field": "ok"}',
            b'data: {invalid json}',  # Malformed JSON
            b'data: {"another": "valid", "id": "2", "value": 2, "required_field": "ok"}',
            b'data: [DONE]'
        ]
        
        # Collect results - should skip the malformed JSON
        results = list(client._stream_request("POST", "stream", json_data={}, cast_to=MockModel))
        
        assert len(results) == 2  # Only valid JSON chunks
        assert results[0].id == "1"
        assert results[1].id == "2"
    
    def test_request_successful_cast(self, client, mock_httpx_client):
        """Test successful casting to ensure error handling doesn't break valid cases."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "success",
            "value": 42,
            "required_field": "present"
        }
        mock_httpx_client.request.return_value = mock_response
        
        result = client._request("GET", "test", cast_to=MockModel)
        
        assert isinstance(result, MockModel)
        assert result.id == "success"
        assert result.value == 42
        assert result.required_field == "present"
    
    def test_request_multipart_successful_cast(self, client, mock_httpx_client):
        """Test successful multipart casting to ensure error handling doesn't break valid cases."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "id": "multipart_success",
            "value": 100,
            "required_field": "uploaded"
        }
        mock_httpx_client.request.return_value = mock_response
        
        files = {"file": ("test.txt", b"content", "text/plain")}
        result = client._request_multipart("POST", "upload", files=files, cast_to=MockModel)
        
        assert isinstance(result, MockModel)
        assert result.id == "multipart_success"
        assert result.value == 100
    
    def test_stream_request_successful_cast(self, client, mock_httpx_client):
        """Test successful stream casting to ensure error handling doesn't break valid cases."""
        # Mock the stream context manager
        mock_stream = MagicMock()
        mock_httpx_client.stream.return_value.__enter__.return_value = mock_stream
        mock_stream.status_code = 200
        
        # Mock valid SSE data
        mock_stream.iter_lines.return_value = [
            b"data: {\"id\": \"stream1\", \"value\": 10, \"required_field\": \"first\"}",
            b"data: {\"id\": \"stream2\", \"value\": 20, \"required_field\": \"second\"}",
            b"data: [DONE]"
        ]
        
        results = list(client._stream_request("POST", "stream", json_data={}, cast_to=MockModel))
        
        assert len(results) == 2
        assert all(isinstance(r, MockModel) for r in results)
        assert results[0].id == "stream1"
        assert results[1].id == "stream2"


class TestClientPydanticCastingEdgeCases:
    """Test edge cases for Pydantic casting errors."""
    
    @pytest.fixture
    def client(self):
        return VeniceClient(api_key="test-api-key")
    
    @pytest.fixture
    def mock_httpx_client(self, client):
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.headers = httpx.Headers({
            "Authorization": "Bearer test-api-key",
            "Accept": "application/json"
        })
        client._client = mock_client
        return mock_client
    
    def test_cast_to_with_custom_exception(self, client, mock_httpx_client):
        """Test cast_to that raises a custom exception type."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "value"}
        mock_httpx_client.request.return_value = mock_response
        
        # Create a model that raises a custom exception
        class CustomError(Exception):
            pass
        
        class ProblematicModel(BaseModel):
            @classmethod
            def model_validate(cls, obj, *, strict=None, from_attributes=None, context=None, by_alias=False, by_name=False):
                raise CustomError("Custom validation error")
        
        with pytest.raises(APIResponseProcessingError) as exc_info:
            client._request("GET", "test", cast_to=ProblematicModel)
        
        error = exc_info.value
        assert "Failed to cast response to" in str(error)
        assert isinstance(error.original_error, CustomError)
    
    def test_multipart_cast_to_none_response(self, client, mock_httpx_client):
        """Test multipart request with None JSON response."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 204  # No content
        mock_response.headers = {}
        mock_response.json.return_value = None
        mock_httpx_client.request.return_value = mock_response
        
        files = {"file": ("empty.txt", b"", "text/plain")}
        
        with pytest.raises(APIResponseProcessingError) as exc_info:
            client._request_multipart("POST", "upload", files=files, cast_to=MockModel)
        
        error = exc_info.value
        assert "Failed to cast multipart response to" in str(error)
    
    def test_stream_empty_data_lines(self, client, mock_httpx_client):
        """Test stream with empty data lines."""
        mock_stream = MagicMock()
        mock_httpx_client.stream.return_value.__enter__.return_value = mock_stream
        mock_stream.status_code = 200
        
        # Mock SSE data with empty lines and spaces
        mock_stream.iter_lines.return_value = [
            b"",  # Empty line
            b"   ",  # Whitespace only
            b"data: ",  # Data prefix with no content
            b"data: {\"id\": \"valid\", \"value\": 1, \"required_field\": \"ok\"}",
            b"data: [DONE]"
        ]
        
        results = list(client._stream_request("POST", "stream", json_data={}, cast_to=MockModel))
        
        # Should only process the valid data line
        assert len(results) == 1
        assert results[0].id == "valid"