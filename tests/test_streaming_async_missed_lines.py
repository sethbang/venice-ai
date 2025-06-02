import pytest
from unittest.mock import Mock, AsyncMock, patch
from httpx import Response
from venice_ai.streaming import AsyncStream
from venice_ai._async_client import AsyncVeniceClient


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