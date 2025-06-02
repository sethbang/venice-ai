"""
Test cases for ChatResource and AsyncChatResource in src/venice_ai/resources/chat.py.

This module implements test cases to ensure proper initialization and basic functionality
of the chat resource classes.
"""

import pytest
from unittest.mock import Mock, patch

from venice_ai._client import VeniceClient
from venice_ai._async_client import AsyncVeniceClient
from venice_ai.resources.chat import ChatResource, AsyncChatResource
from venice_ai.resources.chat.completions import ChatCompletions, AsyncChatCompletions


def test_chat_resource_initialization():
    """Test initialization of ChatResource with a VeniceClient."""
    # Setup mock client
    mock_client = Mock(spec=VeniceClient)
    
    # Action: Initialize ChatResource
    chat_resource = ChatResource(mock_client)
    
    # Assertions
    assert chat_resource._client is mock_client
    assert isinstance(chat_resource.completions, ChatCompletions)
    assert chat_resource.completions._client is mock_client
    print("ChatResource initialization test passed, coverage should include src/venice_ai/resources/chat.py lines 12-31")


@pytest.mark.asyncio
async def test_async_chat_resource_initialization():
    """Test initialization of AsyncChatResource with an AsyncVeniceClient."""
    # Setup mock async client
    mock_async_client = Mock(spec=AsyncVeniceClient)
    
    # Action: Initialize AsyncChatResource
    async_chat_resource = AsyncChatResource(mock_async_client)
    
    # Assertions
    assert async_chat_resource._client is mock_async_client
    assert isinstance(async_chat_resource.completions, AsyncChatCompletions)
    assert async_chat_resource.completions._client is mock_async_client
    print("AsyncChatResource initialization test passed, coverage should include src/venice_ai/resources/chat.py lines 33-51")