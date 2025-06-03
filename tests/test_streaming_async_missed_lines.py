import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from httpx import Response, HTTPStatusError, Request
from venice_ai.streaming import AsyncStream
from venice_ai._async_client import AsyncVeniceClient
from venice_ai.exceptions import InvalidRequestError # Corrected import


class TestAsyncStreamMissedLines:
    """Test cases for AsyncStream (asynchronous) to cover missed lines in streaming.py"""

    async def test_async_stream_close_iterator_aclose_exists_and_raises(self):
        """
        Test Case 2.1: Cover line 73 (exception handling when self._iterator.aclose() exists and raises an error).
        
        Objective: Cover line 73 (exception handling when self._iterator.aclose() exists and raises an error).
        Lines to cover: 73
        """
        # Setup
        mock_async_iterator = AsyncMock()
        mock_async_iterator.aclose = AsyncMock(side_effect=RuntimeError("Iterator aclose failed"))
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        async_stream = AsyncStream(iterator=mock_async_iterator, client=mock_client)
        
        # Action
        await async_stream.close()
        
        # Assertions
        mock_async_iterator.aclose.assert_called_once()
        # The test itself should complete without an unhandled RuntimeError

    async def test_async_stream_close_iterator_no_aclose_method(self):
        """
        Test Case 2.2: Cover line 70 (path where self._iterator does not have aclose method).
        
        Objective: Cover line 70 (path where hasattr check for aclose returns False).
        Lines to cover: 70
        """
        # Setup
        mock_async_iterator = AsyncMock()
        # Remove aclose method to simulate iterator without aclose
        if hasattr(mock_async_iterator, 'aclose'):
            delattr(mock_async_iterator, 'aclose')
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        async_stream = AsyncStream(iterator=mock_async_iterator, client=mock_client)
        
        # Action
        await async_stream.close()
        
        # Assertions
        # Since there's no aclose method, nothing should be called on the iterator
        # The method should complete without error
        assert True  # Test passes if no exception is raised

    @pytest.mark.asyncio
    async def test_async_stream_iteration_http_status_error(self):
        """
        Test Case: Cover line 160 (httpx.HTTPStatusError during iteration in AsyncStream.__anext__).
        Objective: Ensure InvalidRequestError (or appropriate APIError subclass) is raised.
        Lines to cover: 160
        """
        # Setup
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "https://example.com/async_stream"
        
        mock_response_for_error = MagicMock(spec=Response) # Sync mock for response attributes
        mock_response_for_error.status_code = 400
        mock_response_for_error.request = mock_request

        # This async iterator will raise HTTPStatusError when __anext__() is called
        async def error_raising_async_iterator():
            yield {"choices": [{"delta": {"content": "chunk1"}}]} # First chunk is fine
            raise HTTPStatusError(
                message="Bad Request during async iteration",
                request=mock_request,
                response=mock_response_for_error
            )

        mock_iterator = error_raising_async_iterator() # Get the async generator
        
        mock_client = AsyncMock(spec=AsyncVeniceClient)
        # Mock the _translate_httpx_error_to_api_error method to return a proper exception
        mock_client._translate_httpx_error_to_api_error = AsyncMock(
            return_value=InvalidRequestError(
                "HTTP Status 400: API error 400 for POST https://example.com/async_stream: Bad Request during async iteration",
                request=mock_request,
                response=mock_response_for_error
            )
        )
        
        async_stream = AsyncStream(iterator=mock_iterator, client=mock_client)
        
        # Action & Assertions
        # First item should be yielded correctly
        first_chunk = await async_stream.__anext__()
        assert first_chunk["choices"][0]["delta"]["content"] == "chunk1"
        
        # Second call to __anext__() should raise InvalidRequestError for a 400 status
        with pytest.raises(InvalidRequestError) as exc_info:
            await async_stream.__anext__()
        
        assert exc_info.value.status_code == 400
        assert "Bad Request during async iteration" in str(exc_info.value)