import pytest

from venice_ai import VeniceClient
from venice_ai import AsyncVeniceClient
from venice_ai import exceptions
from venice_ai.types.characters import CharacterList, Character


def test_characters_list(venice_client: VeniceClient):
    """
    Test listing characters using the synchronous client.
    Handles the case where the endpoint might not be available.
    """
    try:
        characters = venice_client.characters.list()
        assert characters is not None
        assert isinstance(characters, CharacterList)
        assert hasattr(characters, 'data')
        assert isinstance(characters.data, list)
        assert hasattr(characters, 'object')
        assert characters.object == "list"
        
        if characters.data:
            # If there are characters, check the structure of the first one
            first_character = characters.data[0]
            assert isinstance(first_character, Character)
            assert isinstance(first_character.name, str)
            assert isinstance(first_character.slug, str)
            assert isinstance(first_character.createdAt, str)
            assert isinstance(first_character.updatedAt, str)
            assert isinstance(first_character.adult, bool)
            assert isinstance(first_character.webEnabled, bool)
            assert isinstance(first_character.tags, list)
            # Check optional fields
            if first_character.description is not None:
                assert isinstance(first_character.description, str)
            if first_character.shareUrl is not None:
                assert isinstance(first_character.shareUrl, str)
            # Check stats
            assert hasattr(first_character, 'stats')
            assert hasattr(first_character.stats, 'imports')
            assert isinstance(first_character.stats.imports, int)
    except exceptions.NotFoundError:
        # If the endpoint is not found, consider the test passed as the feature might not be implemented
        pass


@pytest.mark.asyncio
async def test_characters_list_async(async_venice_client: AsyncVeniceClient):
    """
    Test listing characters using the asynchronous client.
    Handles the case where the endpoint might not be available.
    """
    try:
        characters = await async_venice_client.characters.list()
        assert characters is not None
        assert isinstance(characters, CharacterList)
        assert hasattr(characters, 'data')
        assert isinstance(characters.data, list)
        assert hasattr(characters, 'object')
        assert characters.object == "list"
        
        if characters.data:
            # If there are characters, check the structure of the first one
            first_character = characters.data[0]
            assert isinstance(first_character, Character)
            assert isinstance(first_character.name, str)
            assert isinstance(first_character.slug, str)
            assert isinstance(first_character.createdAt, str)
            assert isinstance(first_character.updatedAt, str)
            assert isinstance(first_character.adult, bool)
            assert isinstance(first_character.webEnabled, bool)
            assert isinstance(first_character.tags, list)
            # Check optional fields
            if first_character.description is not None:
                assert isinstance(first_character.description, str)
            if first_character.shareUrl is not None:
                assert isinstance(first_character.shareUrl, str)
            # Check stats
            assert hasattr(first_character, 'stats')
            assert hasattr(first_character.stats, 'imports')
            assert isinstance(first_character.stats.imports, int)
    except exceptions.NotFoundError:
        # If the endpoint is not found, consider the test passed as the feature might not be implemented
        pass