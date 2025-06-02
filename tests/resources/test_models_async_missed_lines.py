"""
Test cases for AsyncModels (asynchronous) missed lines coverage.

This module implements test cases to cover specific missed lines in the
AsyncModels class from src/venice_ai/resources/models.py, targeting lines:
126-138
"""

import pytest
from unittest.mock import AsyncMock
from typing import Dict, Any

from venice_ai._async_client import AsyncVeniceClient
from venice_ai.resources.models import AsyncModels
from venice_ai.types.models import ModelList


class TestAsyncModelsMissedLines:
    """Test class for AsyncModels (asynchronous) missed lines coverage."""

    async def test_async_models_list_with_specific_type(self):
        """
        Test Case 1: Cover one of the elif branches for the type parameter (e.g., type="embedding"),
        ensuring the type is correctly added to the query parameters. This covers lines like 128.
        
        Objective: Cover part of 126-138 (specifically, a matching elif type == "embedding": path).
        Lines to cover: Part of 126-138 (specifically, a matching elif path).
        """
        mock_async_venice_client = AsyncMock(spec=AsyncVeniceClient)
        async_models_resource = AsyncModels(mock_async_venice_client)
        mock_async_venice_client.get = AsyncMock()
        
        await async_models_resource.list(type="embedding")
        
        mock_async_venice_client.get.assert_called_once()
        args, kwargs = mock_async_venice_client.get.call_args
        assert args[0] == "models"
        expected_params = {"type": "embedding"}
        assert kwargs.get("params") == expected_params

    async def test_async_models_list_with_type_none(self):
        """
        Test Case 2: Cover the path where type is None (default), ensuring the type parameter
        is set to "all" in the query parameters from the type handling logic. This covers the
        if type is not None: being false and the else branch.
        
        Objective: Cover part of 126-138 (specifically, the if type is not None: condition being false).
        Lines to cover: Part of 126-138 (specifically, the if type is not None: condition being false).
        """
        mock_async_venice_client = AsyncMock(spec=AsyncVeniceClient)
        async_models_resource = AsyncModels(mock_async_venice_client)
        mock_async_venice_client.get = AsyncMock()
        
        await async_models_resource.list(type=None)
        
        mock_async_venice_client.get.assert_called_once()
        args, kwargs = mock_async_venice_client.get.call_args
        assert args[0] == "models"
        expected_params: Dict[str, Any] = {"type": "all"}
        assert kwargs.get("params") == expected_params

    async def test_async_models_list_with_unknown_type(self):
        """
        Test Case 3: Cover the path where type is provided but does not match any of the known
        literal values, ensuring no type key is added to the query parameters from this specific
        if/elif block. This covers the if type is not None: being true, but none of the elif
        conditions matching.
        
        Objective: Cover part of 126-138 (specifically, if type is not None: is true, but no elif matches).
        Lines to cover: Part of 126-138 (specifically, if type is not None: is true, but no elif matches).
        """
        mock_async_venice_client = AsyncMock(spec=AsyncVeniceClient)
        async_models_resource = AsyncModels(mock_async_venice_client)
        mock_async_venice_client.get = AsyncMock()
        
        await async_models_resource.list(type="some_unknown_value")  # type: ignore
        
        mock_async_venice_client.get.assert_called_once()
        args, kwargs = mock_async_venice_client.get.call_args
        assert args[0] == "models"
        expected_params: Dict[str, Any] = {}
        assert kwargs.get("params") == expected_params