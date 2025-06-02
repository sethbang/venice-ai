"""
Test coverage for src/venice_ai/resources/chat.py

This module provides comprehensive test coverage for both ChatResource (synchronous)
and AsyncChatResource (asynchronous) classes, targeting missed lines 2-51.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from venice_ai.resources.chat import ChatResource, AsyncChatResource
from venice_ai.resources.chat.completions import ChatCompletions, AsyncChatCompletions
from venice_ai._client import VeniceClient
from venice_ai._async_client import AsyncVeniceClient


class TestChatResourceCoverage:
    """Test coverage for ChatResource (synchronous) class."""

    def test_chat_resource_initialization(self):
        """
        Test Case 1.1: Cover lines 12-30 (definition and initialization of ChatResource).
        
        Objective: Ensure ChatResource initializes correctly with proper client assignment
        and ChatCompletions instance creation.
        Lines covered: 12-30
        """
        # Setup
        mock_venice_client = Mock(spec=VeniceClient)
        
        # Action
        chat_resource = ChatResource(client=mock_venice_client)
        
        # Assertions
        assert chat_resource._client is mock_venice_client
        assert chat_resource.completions is not None
        assert isinstance(chat_resource.completions, ChatCompletions)
        assert chat_resource.completions._client is mock_venice_client


class TestAsyncChatResourceCoverage:
    """Test coverage for AsyncChatResource (asynchronous) class."""

    def test_async_chat_resource_initialization(self):
        """
        Test Case 2.1: Cover lines 33-51 (definition and initialization of AsyncChatResource).
        
        Objective: Ensure AsyncChatResource initializes correctly with proper client assignment
        and AsyncChatCompletions instance creation.
        Lines covered: 33-51
        """
        # Setup
        mock_async_venice_client = AsyncMock(spec=AsyncVeniceClient)
        
        # Action
        async_chat_resource = AsyncChatResource(client=mock_async_venice_client)
        
        # Assertions
        assert async_chat_resource._client is mock_async_venice_client
        assert async_chat_resource.completions is not None
        assert isinstance(async_chat_resource.completions, AsyncChatCompletions)
        assert async_chat_resource.completions._client is mock_async_venice_client