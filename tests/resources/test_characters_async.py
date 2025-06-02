"""
Tests for the asynchronous AsyncCharacters resource.
"""

import pytest
import pytest_asyncio
import httpx
from typing import List, Literal, TypedDict, Optional

from venice_ai import AsyncVeniceClient
from venice_ai.types.characters import CharacterList, Character
from venice_ai.exceptions import APIError, AuthenticationError


# Define mock response structures for testing
class MockStats(TypedDict):
    imports: int

class MockCharacter(TypedDict):
    id: str
    name: str
    description: Optional[str]
    avatarUrl: str
    adult: bool
    createdAt: str
    shareUrl: Optional[str]
    slug: str
    stats: MockStats
    tags: List[str]
    updatedAt: str
    webEnabled: bool


class MockCharacterList(TypedDict):
    data: List[MockCharacter]
    object: Literal["list"]


@pytest.mark.asyncio
async def test_list_success_async(httpx_mock):
    """Tests successful asynchronous retrieval of characters list."""
    mock_response_data: MockCharacterList = {
        "data": [
            {
                "id": "char_async_1",
                "avatarUrl": "https://example.com/avatar1.png",
                "adult": False,
                "createdAt": "2025-01-01T00:00:00Z",
                "description": "A test character.",
                "name": "Test Character 1",
                "shareUrl": "https://example.com/share1",
                "slug": "test-character-1",
                "stats": {"imports": 10},
                "tags": ["test", "character"],
                "updatedAt": "2025-01-01T00:00:00Z",
                "webEnabled": True,
            },
            {
                "id": "char_async_2",
                "avatarUrl": "https://example.com/avatar2.png",
                "adult": True,
                "createdAt": "2025-01-02T00:00:00Z",
                "description": "Another test character.",
                "name": "Test Character 2",
                "shareUrl": "https://example.com/share2",
                "slug": "test-character-2",
                "stats": {"imports": 20},
                "tags": ["another", "test"],
                "updatedAt": "2025-01-02T00:00:00Z",
                "webEnabled": False,
            },
        ],
        "object": "list",
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/characters", # Corrected URL to match client request
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        response = await client.characters.list()

    assert isinstance(response, CharacterList)
    assert response.object == "list"
    assert isinstance(response.data, list)
    assert len(response.data) == 2
    assert isinstance(response.data[0], Character)
    # Removed assertion on 'id' as it's not in the Pydantic model
    assert response.data[0].name == "Test Character 1"
    assert response.data[1].name == "Test Character 2"


@pytest.mark.asyncio
async def test_list_api_error_async(httpx_mock):
    """Tests asynchronous API error handling for characters list retrieval."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/characters", # Corrected URL to match client request
        status_code=401,
        json={"error": {"message": "Unauthorized", "type": "authentication_error"}},
    )

    async with AsyncVeniceClient(api_key="invalid-key") as client:
        with pytest.raises(AuthenticationError) as excinfo:
            await client.characters.list()

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 401
    assert "Unauthorized" in str(excinfo.value)