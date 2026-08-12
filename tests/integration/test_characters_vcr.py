"""
VCRpy-based integration tests for Characters Resource.

This module uses VCRpy to record and replay real API interactions with the
Venice.ai Characters endpoint.

Tests use @pytest.mark.vcr decorator for automatic cassette recording.
"""

import os

import pytest
import pytest_asyncio

from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.types.api.characters import (
    CharacterResponse,
    CharactersListResponse,
)


@pytest_asyncio.fixture
async def venice_client(backend_instance):
    """Create a Venice client for VCR testing with shared rate limit coordination."""
    api_key = os.getenv("VENICE_API_KEY", "test-api-key-for-recording")

    client = create_test_venice_client(
        api_key=api_key,
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=True,
    )
    try:
        yield client
    finally:
        await client.close()


# ============================================================================
# Characters List Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.vcr
async def test_characters_list_basic(venice_client, vcr_cassette):
    """
    Test basic characters list retrieval using VCRpy.
    """
    with vcr_cassette:
        response = await venice_client.characters.list()

    # Validate response structure
    assert response is not None
    assert isinstance(response, CharactersListResponse)
    assert hasattr(response, "data")
    assert isinstance(response.data, list)

    # If we have characters, validate structure
    if len(response.data) > 0:
        character = response.data[0]
        assert hasattr(character, "slug")
        assert hasattr(character, "name")
        assert hasattr(character, "description")


@pytest.mark.integration
@pytest.mark.vcr
async def test_characters_list_with_extra_headers(venice_client, vcr_cassette):
    """
    Test characters list with extra headers.

    This test exercises the extra_headers branch.
    """
    with vcr_cassette:
        response = await venice_client.characters.list(
            extra_headers={"X-Custom-Header": "test-value"}
        )

    assert response is not None
    assert isinstance(response, CharactersListResponse)


@pytest.mark.integration
@pytest.mark.vcr
async def test_characters_list_with_extra_query(venice_client, vcr_cassette):
    """
    Test characters list with extra query parameters.

    This test exercises the extra_query branch.
    """
    with vcr_cassette:
        response = await venice_client.characters.list(extra_query={"custom_param": "value"})

    assert response is not None
    assert isinstance(response, CharactersListResponse)


@pytest.mark.integration
@pytest.mark.vcr
async def test_characters_list_with_timeout(venice_client, vcr_cassette):
    """
    Test characters list with custom timeout.

    This test covers the timeout parameter of characters.py.
    """
    with vcr_cassette:
        response = await venice_client.characters.list(timeout=60.0)

    assert response is not None
    assert isinstance(response, CharactersListResponse)


@pytest.mark.integration
@pytest.mark.vcr
async def test_characters_list_with_all_params(venice_client, vcr_cassette):
    """
    Test characters list with all optional parameters.

    This test ensures all branches (headers, params, timeout) are covered.
    """
    with vcr_cassette:
        response = await venice_client.characters.list(
            extra_headers={"X-Test": "header"},
            extra_query={"test": "query"},
            timeout=45.0,
        )

    assert response is not None
    assert isinstance(response, CharactersListResponse)


# ============================================================================
# Characters Get Tests (Lines 207-222)
# ============================================================================


@pytest.mark.integration
@pytest.mark.vcr
async def test_characters_get_basic(venice_client, vcr_cassette):
    """
    Test basic character retrieval by slug using VCRpy.

    First gets a character list to find a valid slug, then retrieves that character.
    """
    with vcr_cassette:
        # First get list to find a valid character slug
        list_response = await venice_client.characters.list()

        # Skip test if no characters available
        if len(list_response.data) == 0:
            pytest.skip("No characters available for testing")

        # Get first character's slug
        test_slug = list_response.data[0].slug

        # Now test the get method
        response = await venice_client.characters.get(test_slug)

    # Validate response structure
    assert response is not None
    assert isinstance(response, CharacterResponse)
    assert hasattr(response, "data")
    assert response.data.slug == test_slug


@pytest.mark.integration
@pytest.mark.vcr
async def test_characters_get_with_extra_headers(venice_client, vcr_cassette):
    """
    Test character get with extra headers.

    This test exercises the extra_headers branch.
    """
    with vcr_cassette:
        # Get a list first to find a valid slug
        list_response = await venice_client.characters.list()

        if len(list_response.data) == 0:
            pytest.skip("No characters available")

        test_slug = list_response.data[0].slug

        # Test with extra headers
        response = await venice_client.characters.get(
            test_slug, extra_headers={"X-Custom-Header": "test-value"}
        )

    assert response is not None
    assert isinstance(response, CharacterResponse)


@pytest.mark.integration
@pytest.mark.vcr
async def test_characters_get_with_extra_query(venice_client, vcr_cassette):
    """
    Test character get with extra query parameters.

    This test exercises the extra_query branch.
    """
    with vcr_cassette:
        # Get a list first
        list_response = await venice_client.characters.list()

        if len(list_response.data) == 0:
            pytest.skip("No characters available")

        test_slug = list_response.data[0].slug

        # Test with extra query parameters
        response = await venice_client.characters.get(
            test_slug, extra_query={"include_details": "true"}
        )

    assert response is not None
    assert isinstance(response, CharacterResponse)


@pytest.mark.integration
@pytest.mark.vcr
async def test_characters_get_with_timeout(venice_client, vcr_cassette):
    """
    Test character get with custom timeout.

    This test covers the timeout parameter of characters.get().
    """
    with vcr_cassette:
        list_response = await venice_client.characters.list()

        if len(list_response.data) == 0:
            pytest.skip("No characters available")

        test_slug = list_response.data[0].slug

        response = await venice_client.characters.get(test_slug, timeout=60.0)

    assert response is not None
    assert isinstance(response, CharacterResponse)


@pytest.mark.integration
@pytest.mark.vcr
async def test_characters_get_with_all_params(venice_client, vcr_cassette):
    """
    Test character get with all optional parameters.

    This test ensures all branches (headers, params, timeout) are covered.
    """
    with vcr_cassette:
        list_response = await venice_client.characters.list()

        if len(list_response.data) == 0:
            pytest.skip("No characters available")

        test_slug = list_response.data[0].slug

        # Test with all parameters
        response = await venice_client.characters.get(
            test_slug,
            extra_headers={"X-Test": "header"},
            extra_query={"details": "full"},
            timeout=45.0,
        )

    assert response is not None
    assert isinstance(response, CharacterResponse)
    assert response.data.slug == test_slug


@pytest.mark.integration
@pytest.mark.vcr
async def test_characters_get_nonexistent(venice_client, vcr_cassette):
    """
    Test character get with non-existent slug to test error handling.

    This tests the error path when a character is not found.
    """
    from venice_ai.exceptions import VeniceError

    with vcr_cassette, pytest.raises(VeniceError):  # Will be VeniceModelException or similar
        # Try to get a character that doesn't exist
        await venice_client.characters.get("nonexistent-character-slug-12345")


@pytest.mark.integration
@pytest.mark.vcr
async def test_characters_response_structure(venice_client, vcr_cassette):
    """
    Test detailed character response structure.

    This test validates that all expected fields are present in the response.
    """
    with vcr_cassette:
        list_response = await venice_client.characters.list()

        if len(list_response.data) == 0:
            pytest.skip("No characters available")

        test_slug = list_response.data[0].slug
        response = await venice_client.characters.get(test_slug)

    # Validate response structure
    assert response.data is not None

    # Check expected fields
    character = response.data
    assert hasattr(character, "slug")
    assert hasattr(character, "name")
    assert hasattr(character, "description")
    assert hasattr(character, "modelId")
    assert hasattr(character, "stats")
    assert hasattr(character, "tags")

    # Validate types
    assert isinstance(character.slug, str)
    assert isinstance(character.name, str)
    if character.description:
        assert isinstance(character.description, str)
    if character.tags:
        assert isinstance(character.tags, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
