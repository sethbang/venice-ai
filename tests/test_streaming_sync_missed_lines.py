import pytest
from unittest.mock import Mock, patch
from httpx import Response
from venice_ai.streaming import Stream
from venice_ai._client import VeniceClient


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