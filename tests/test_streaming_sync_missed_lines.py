import pytest
from unittest.mock import Mock, patch, MagicMock
from httpx import Response, HTTPStatusError, Request
from venice_ai.streaming import Stream
from venice_ai._client import VeniceClient
from venice_ai.exceptions import InvalidRequestError # Corrected import


class TestStreamSyncMissedLines:
    """Test cases for Stream (synchronous) to cover missed lines in streaming.py"""

    def test_stream_close_iterator_close_raises_exception(self):
        """
        Test Case 1.1: Cover line 40 (exception handling in Stream.close when self._iterator.close() raises an error).
        
        Objective: Cover line 40 (exception handling in Stream.close when self._iterator.close() raises an error).
        Lines to cover: 40
        """
        # Setup
        mock_iterator = Mock()
        mock_iterator.close.side_effect = RuntimeError("Iterator close failed")
        mock_response = Mock(spec=Response)
        mock_client = Mock(spec=VeniceClient)
        stream = Stream(iterator=mock_iterator, client=mock_client)
        
        # Action
        stream.close()
        
        # Assertions
        mock_iterator.close.assert_called_once()
        # The test itself should complete without an unhandled RuntimeError

    def test_stream_iteration_http_status_error(self):
        """
        Test Case: Cover line 67 (httpx.HTTPStatusError during iteration in Stream.__next__).
        Objective: Ensure APIStatusError is raised when httpx.HTTPStatusError occurs during stream iteration.
        Lines to cover: 67
        """
        # Setup
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "https://example.com/stream"
        
        mock_response_for_error = MagicMock(spec=Response)
        mock_response_for_error.status_code = 400
        mock_response_for_error.request = mock_request

        # This iterator will raise HTTPStatusError when next() is called
        def error_raising_iterator():
            yield {"choices": [{"delta": {"content": "chunk1"}}]} # First chunk is fine
            raise HTTPStatusError(
                message="Bad Request during iteration",
                request=mock_request,
                response=mock_response_for_error
            )

        mock_iterator = iter(error_raising_iterator())
        
        mock_client = Mock(spec=VeniceClient)
        # Mock the _translate_httpx_error_to_api_error method to return a proper exception
        mock_client._translate_httpx_error_to_api_error = Mock(
            return_value=InvalidRequestError(
                "HTTP Status 400: API error 400 for POST https://example.com/stream: Bad Request during iteration",
                request=mock_request,
                response=mock_response_for_error
            )
        )
        
        stream = Stream(iterator=mock_iterator, client=mock_client)
        
        # Action & Assertions
        # First item should be yielded correctly
        first_chunk = next(stream)
        assert first_chunk["choices"][0]["delta"]["content"] == "chunk1"
        
        # Second call to next() should raise InvalidRequestError for a 400 status
        with pytest.raises(InvalidRequestError) as exc_info: # Corrected expected exception
            next(stream)
        
        assert exc_info.value.status_code == 400
        assert "Bad Request during iteration" in str(exc_info.value)