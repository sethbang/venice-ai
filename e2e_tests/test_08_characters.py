import pytest

from venice_ai import VeniceClient
from venice_ai import AsyncVeniceClient
from venice_ai import exceptions
from venice_ai.types.characters import CharacterList, Character


def test_characters_list(httpx_mock):
    """
    Test listing characters using the synchronous client with mocked response.
    """
    mock_response_data = {
        "data": [
            {
                "slug": "test-char",
                "name": "Test Character",
                "description": "A test character for e2e testing",
                "system_prompt": "You are a test character",
                "user_prompt": "Hello, I'm a test character",
                "vision_enabled": False,
                "image_url": "https://example.com/image.png",
                "voice_id": "test-voice-id",
                "category_tags": ["test", "character"],
                "created_at": "2025-01-01T12:00:00Z",
                "updated_at": "2025-01-01T12:00:00Z"
            }
        ],
        "object": "list"
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/characters",
        json=mock_response_data,
        status_code=200,
    )

    client = VeniceClient(api_key="test-key")
    characters = client.characters.list()
    
    assert characters is not None
    assert isinstance(characters, CharacterList)
    assert hasattr(characters, 'data')
    assert isinstance(characters.data, list)
    assert hasattr(characters, 'object')
    assert characters.object == "list"
    
    # Check the structure of the first character
    first_character = characters.data[0]
    assert isinstance(first_character, Character)
    assert isinstance(first_character.name, str)
    assert isinstance(first_character.slug, str)
    assert first_character.created_at is not None
    assert first_character.updated_at is not None
    # Check optional fields
    if first_character.description is not None:
        assert isinstance(first_character.description, str)
    if first_character.system_prompt is not None:
        assert isinstance(first_character.system_prompt, str)
    if first_character.user_prompt is not None:
        assert isinstance(first_character.user_prompt, str)
    if first_character.image_url is not None:
        assert isinstance(first_character.image_url, str)
    if first_character.voice_id is not None:
        assert isinstance(first_character.voice_id, str)
    if first_character.category_tags is not None:
        assert isinstance(first_character.category_tags, list)


@pytest.mark.asyncio
async def test_characters_list_async(httpx_mock):
    """
    Test listing characters using the asynchronous client with mocked response.
    """
    mock_response_data = {
        "data": [
            {
                "slug": "test-char-async",
                "name": "Test Character Async",
                "description": "A test character for async e2e testing",
                "system_prompt": "You are an async test character",
                "user_prompt": "Hello, I'm an async test character",
                "vision_enabled": True,
                "image_url": "https://example.com/async-image.png",
                "voice_id": "async-test-voice-id",
                "category_tags": ["test", "character", "async"],
                "created_at": "2025-01-01T12:00:00Z",
                "updated_at": "2025-01-01T12:00:00Z"
            }
        ],
        "object": "list"
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/characters",
        json=mock_response_data,
        status_code=200,
    )

    client = AsyncVeniceClient(api_key="test-key")
    characters = await client.characters.list()
    
    assert characters is not None
    assert isinstance(characters, CharacterList)
    assert hasattr(characters, 'data')
    assert isinstance(characters.data, list)
    assert hasattr(characters, 'object')
    assert characters.object == "list"
    
    # Check the structure of the first character
    first_character = characters.data[0]
    assert isinstance(first_character, Character)
    assert isinstance(first_character.name, str)
    assert isinstance(first_character.slug, str)
    assert first_character.created_at is not None
    assert first_character.updated_at is not None
    # Check optional fields
    if first_character.description is not None:
        assert isinstance(first_character.description, str)
    if first_character.system_prompt is not None:
        assert isinstance(first_character.system_prompt, str)
    if first_character.user_prompt is not None:
        assert isinstance(first_character.user_prompt, str)
    if first_character.image_url is not None:
        assert isinstance(first_character.image_url, str)
    if first_character.voice_id is not None:
        assert isinstance(first_character.voice_id, str)
    if first_character.category_tags is not None:
        assert isinstance(first_character.category_tags, list)
    
    await client.close()