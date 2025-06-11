"""
Additional tests for improving coverage of characters.py
Specifically targeting the return statements in list() methods.
"""

import pytest
import httpx
from typing import Dict, Any

from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.types.characters import CharacterList
from venice_ai.resources.characters import Characters, AsyncCharacters


def test_characters_list_return_coverage(monkeypatch):
    """
    Test specifically targeting the return statement in Characters.list
    by mocking the _client.get directly instead of using httpx_mock.
    """
    # Create a mock client with a get method that returns a valid response
    class MockClient:
        def get(self, path: str, **kwargs) -> Dict[str, Any]:
            assert path == "characters"
            return {
                "data": [
                    {
                        "adult": False,
                        "createdAt": "2025-01-01T00:00:00Z",
                        "description": "Coverage test character",
                        "name": "Coverage Test",
                        "shareUrl": "https://example.com/coverage",
                        "slug": "coverage-test",
                        "stats": {"imports": 5},
                        "tags": ["coverage", "test"],
                        "updatedAt": "2025-01-01T00:00:00Z",
                        "webEnabled": True,
                    }
                ],
                "object": "list"
            }

    # Create Characters instance with mock client
    characters = Characters(MockClient())
    
    # Call list() method - this should hit the return statement we want to cover
    result = characters.list()
    
    # Verify the returned object is correct
    assert isinstance(result, CharacterList)
    assert result.object == "list"
    assert len(result.data) == 1
    assert result.data[0].name == "Coverage Test"
    assert result.data[0].slug == "coverage-test"


@pytest.mark.asyncio
async def test_async_characters_list_return_coverage(monkeypatch):
    """
    Test specifically targeting the return statement in AsyncCharacters.list
    by mocking the _client.get directly instead of using httpx_mock.
    """
    # Create a mock client with an async get method that returns a valid response
    class MockAsyncClient:
        async def get(self, path: str, **kwargs) -> Dict[str, Any]:
            assert path == "characters"
            return {
                "data": [
                    {
                        "adult": False,
                        "createdAt": "2025-01-01T00:00:00Z",
                        "description": "Async coverage test character",
                        "name": "Async Coverage Test",
                        "shareUrl": "https://example.com/async-coverage",
                        "slug": "async-coverage-test",
                        "stats": {"imports": 10},
                        "tags": ["async", "coverage", "test"],
                        "updatedAt": "2025-01-01T00:00:00Z",
                        "webEnabled": True,
                    }
                ],
                "object": "list"
            }

    # Create AsyncCharacters instance with mock client
    async_characters = AsyncCharacters(MockAsyncClient())
    
    # Call list() method - this should hit the return statement we want to cover
    result = await async_characters.list()
    
    # Verify the returned object is correct
    assert isinstance(result, CharacterList)
    assert result.object == "list"
    assert len(result.data) == 1
    assert result.data[0].name == "Async Coverage Test"
    assert result.data[0].slug == "async-coverage-test"