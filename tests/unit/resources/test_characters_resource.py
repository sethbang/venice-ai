# tests/unit/resources/test_characters_resource.py
from unittest.mock import AsyncMock

import pytest

from venice_ai.resources.characters import Characters
from venice_ai.types.api.characters import (
    Character,
    CharacterReview,
    CharacterReviewsPagination,
    CharacterReviewsResponse,
    CharacterReviewsSummary,
    CharactersListResponse,
    CharacterStats,
)


@pytest.mark.asyncio
async def test_characters_list():
    """Test the Characters.list() method."""
    mock_client = AsyncMock()
    test_character = Character(
        slug="test",
        name="Test Character",
        description="Test character description",
        shareUrl="https://example.com/test",
        photoUrl="https://example.com/photo.jpg",
        adult=False,
        webEnabled=True,
        createdAt="2024-01-01T00:00:00Z",
        updatedAt="2024-01-01T00:00:00Z",
        tags=["test"],
        stats=CharacterStats(imports=0),
        modelId="venice-uncensored",
    )
    mock_client.get = AsyncMock(
        return_value=CharactersListResponse(object="list", data=[test_character])
    )

    characters = Characters(mock_client)
    result = await characters.list()

    mock_client.get.assert_called_once_with(
        "characters",
        headers=None,
        params=None,
        timeout=None,
        cast_to=CharactersListResponse,
    )
    assert result.data[0].slug == "test"


@pytest.mark.asyncio
async def test_characters_list_with_params():
    """Test Characters.list() with extra parameters."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=CharactersListResponse(object="list", data=[]))

    characters = Characters(mock_client)
    await characters.list(
        extra_headers={"X-Custom": "header"},
        extra_query={"filter": "active"},
        timeout=30.0,
    )

    mock_client.get.assert_called_once_with(
        "characters",
        headers={"X-Custom": "header"},
        params={"filter": "active"},
        timeout=30.0,
        cast_to=CharactersListResponse,
    )


@pytest.mark.asyncio
async def test_characters_list_with_filters():
    """list() threads the typed filter kwargs into query params."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=CharactersListResponse(object="list", data=[]))

    characters = Characters(mock_client)
    await characters.list(
        categories=["roleplay", "philosophy"],
        is_adult=False,
        is_pro=True,
        is_web_enabled=True,
        limit=25,
        model_id=["llama-3.3-70b", "venice-uncensored"],
        offset=10,
        search="assistant",
        sort_by="highlyRated",
        sort_order="desc",
        tags=["helpful"],
    )

    mock_client.get.assert_called_once_with(
        "characters",
        headers=None,
        params={
            "categories": "roleplay,philosophy",
            "isAdult": "false",
            "isPro": "true",
            "isWebEnabled": "true",
            "limit": 25,
            "modelId": "llama-3.3-70b,venice-uncensored",
            "offset": 10,
            "search": "assistant",
            "sortBy": "highlyRated",
            "sortOrder": "desc",
            "tags": "helpful",
        },
        timeout=None,
        cast_to=CharactersListResponse,
    )


def test_character_model_accepts_new_fields():
    """Character model parses the fields added by the April 2026 expansion."""
    char = Character.model_validate(
        {
            "slug": "alan-watts",
            "name": "Alan Watts",
            "description": None,
            "shareUrl": None,
            "photoUrl": None,
            "adult": False,
            "webEnabled": True,
            "createdAt": "2024-12-20T21:28:08.934Z",
            "updatedAt": "2025-02-09T03:23:53.708Z",
            "tags": [],
            "stats": {
                "imports": 112,
                "averageRating": 4.7,
                "ratingCount": 24,
                "ratingSum": 113,
                "userRating": 5,
            },
            "modelId": "venice-uncensored",
            "id": "2f460055-7595-4640-9cb6-c442c4c869b0",
            "author": "k3x9q",
            "featured": False,
            "isOwner": False,
        }
    )

    assert char.id == "2f460055-7595-4640-9cb6-c442c4c869b0"
    assert char.author == "k3x9q"
    assert char.featured is False
    assert char.isOwner is False
    assert char.stats.averageRating == 4.7
    assert char.stats.ratingCount == 24
    assert char.stats.userRating == 5


def test_character_model_backward_compatible_without_new_fields():
    """Construction without the new fields still succeeds."""
    char = Character(
        slug="t",
        name="t",
        description=None,
        shareUrl=None,
        photoUrl=None,
        adult=False,
        webEnabled=True,
        createdAt="2024-01-01T00:00:00Z",
        updatedAt="2024-01-01T00:00:00Z",
        tags=[],
        stats=CharacterStats(imports=0),
        modelId="venice-uncensored",
    )
    assert char.id is None
    assert char.author is None
    assert char.isOwner is None
    assert char.stats.averageRating is None


@pytest.mark.asyncio
async def test_characters_reviews_no_pagination():
    """reviews() without page/pageSize sends no extra query params."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(
        return_value=CharacterReviewsResponse(
            object="list",
            data=[],
            pagination=CharacterReviewsPagination(page=1, pageSize=20, total=0, totalPages=0),
            summary=CharacterReviewsSummary(averageRating=0.0, totalReviews=0),
        )
    )

    characters = Characters(mock_client)
    await characters.reviews("alan-watts")

    mock_client.get.assert_called_once_with(
        "characters/alan-watts/reviews",
        headers=None,
        params=None,
        timeout=None,
        cast_to=CharacterReviewsResponse,
    )


@pytest.mark.asyncio
async def test_characters_reviews_with_pagination():
    """reviews() maps page_size kwarg to the API's pageSize param."""
    mock_client = AsyncMock()
    review = CharacterReview(
        id="r1",
        characterId="c1",
        createdAt="2025-02-09T00:00:00Z",
        rating=5,
        message="Great",
        locale="en",
        username="user",
        userAvatarUrl=None,
        isOwner=False,
    )
    mock_client.get = AsyncMock(
        return_value=CharacterReviewsResponse(
            object="list",
            data=[review],
            pagination=CharacterReviewsPagination(page=2, pageSize=50, total=87, totalPages=2),
            summary=CharacterReviewsSummary(averageRating=4.7, totalReviews=87),
        )
    )

    characters = Characters(mock_client)
    result = await characters.reviews("alan-watts", page=2, page_size=50)

    mock_client.get.assert_called_once_with(
        "characters/alan-watts/reviews",
        headers=None,
        params={"page": 2, "pageSize": 50},
        timeout=None,
        cast_to=CharacterReviewsResponse,
    )
    assert result.summary.averageRating == 4.7
    assert result.data[0].rating == 5
