"""
Tests for cli/commands/characters.py

Covers:
- characters() click group command
- list_characters() click command entrypoint
- _list_characters_async() core logic with all branches
- info_character() click command entrypoint
- _info_character_async() core logic with all branches
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from venice_ai.cli.commands.characters import (
    _info_character_async,
    _list_characters_async,
    _reviews_character_async,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(plain: bool = False):
    """Return a minimal mock Click context."""
    ctx = MagicMock()
    ctx.obj = {"config": {"api": {"key": "test-key"}}, "plain": plain}
    return ctx


def _setup_client_patch(MockVeniceClient, mock_client):
    """Configure the MockVeniceClient to act as an async context manager returning mock_client."""
    MockVeniceClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    MockVeniceClient.return_value.__aexit__ = AsyncMock(return_value=None)


_SENTINEL = object()


def _make_character(
    slug="test-char",
    name="Test Character",
    description=_SENTINEL,
    modelId="llama-3.3-70b",
    tags=_SENTINEL,
    adult=False,
    shareUrl=_SENTINEL,
    photoUrl="https://venice.ai/photo/test-char.jpg",
    webEnabled=True,
    createdAt="2024-01-01T00:00:00Z",
    updatedAt="2024-06-01T00:00:00Z",
    stats=_SENTINEL,
    id="char-uuid-1",
    author="someuser",
    featured=False,
    isOwner=None,
):
    """Create a mock character SimpleNamespace."""
    if description is _SENTINEL:
        description = "A test character"
    if tags is _SENTINEL:
        tags = ["helpful", "coding"]
    if shareUrl is _SENTINEL:
        shareUrl = "https://venice.ai/chat/test-char"
    if stats is _SENTINEL:
        stats = SimpleNamespace(
            imports=42,
            averageRating=4.5,
            ratingCount=10,
            ratingSum=45,
            userRating=None,
        )
    return SimpleNamespace(
        slug=slug,
        name=name,
        description=description,
        modelId=modelId,
        tags=tags,
        adult=adult,
        shareUrl=shareUrl,
        photoUrl=photoUrl,
        webEnabled=webEnabled,
        createdAt=createdAt,
        updatedAt=updatedAt,
        stats=stats,
        id=id,
        author=author,
        featured=featured,
        isOwner=isOwner,
    )


def _make_review(
    rating=5,
    createdAt="2024-05-01T00:00:00Z",
    message="Great character!",
    id="review-uuid-1",
    characterId="char-uuid-1",
):
    """Create a mock CharacterReview SimpleNamespace."""
    return SimpleNamespace(
        id=id,
        characterId=characterId,
        rating=rating,
        createdAt=createdAt,
        message=message,
    )


def _make_reviews_response(reviews=None, averageRating=4.5, totalReviews=None):
    """Create a mock CharacterReviewsResponse SimpleNamespace."""
    if reviews is None:
        reviews = [_make_review()]
    if totalReviews is None:
        totalReviews = len(reviews)
    return SimpleNamespace(
        data=reviews,
        summary=SimpleNamespace(averageRating=averageRating, totalReviews=totalReviews),
        pagination=SimpleNamespace(page=1, pageSize=20, total=totalReviews, totalPages=1),
    )


# ---------------------------------------------------------------------------
# _list_characters_async tests — JSON output
# ---------------------------------------------------------------------------


class TestListCharactersAsyncJsonOutput:
    """Tests for _list_characters_async in JSON output mode."""

    @pytest.mark.asyncio
    async def test_list_json_output_with_characters(self):
        """JSON output mode returns character list as JSON."""
        mock_client = AsyncMock()
        char1 = _make_character(slug="char-1", name="Character One", modelId="model-a")
        char2 = _make_character(slug="char-2", name="Character Two", modelId="model-b")
        mock_response = SimpleNamespace(data=[char1, char2])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _list_characters_async(ctx, output_json=True, search=None)

            mock_echo.assert_called_once()
            data = json.loads(mock_echo.call_args[0][0])
            assert len(data) == 2
            assert data[0]["slug"] == "char-1"
            assert data[1]["slug"] == "char-2"

    @pytest.mark.asyncio
    async def test_list_json_output_empty(self):
        """JSON output mode with no characters returns empty list."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(data=[])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _list_characters_async(ctx, output_json=True, search=None)

            mock_echo.assert_called_once()
            data = json.loads(mock_echo.call_args[0][0])
            assert data == []

    @pytest.mark.asyncio
    async def test_list_json_includes_all_fields(self):
        """JSON output includes slug, name, description, tags, modelId, adult."""
        mock_client = AsyncMock()
        char = _make_character(
            slug="alan-watts",
            name="Alan Watts",
            description="Philosopher",
            modelId="llama-3.3-70b",
            tags=["philosophy", "zen"],
            adult=False,
        )
        mock_response = SimpleNamespace(data=[char])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _list_characters_async(ctx, output_json=True, search=None)

            data = json.loads(mock_echo.call_args[0][0])
            assert data[0]["slug"] == "alan-watts"
            assert data[0]["name"] == "Alan Watts"
            assert data[0]["description"] == "Philosopher"
            assert data[0]["modelId"] == "llama-3.3-70b"
            assert data[0]["tags"] == ["philosophy", "zen"]
            assert data[0]["adult"] is False


# ---------------------------------------------------------------------------
# _list_characters_async tests — search filter
# ---------------------------------------------------------------------------


class TestListCharactersAsyncSearch:
    """Tests for _list_characters_async server-side search pass-through (CHAR-05)."""

    @pytest.mark.asyncio
    async def test_search_passed_to_server(self):
        """--search is forwarded straight to client.characters.list(search=...)."""
        mock_client = AsyncMock()
        char = _make_character(slug="alan-watts", name="Alan Watts")
        mock_response = SimpleNamespace(data=[char])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _list_characters_async(ctx, output_json=True, search="alan")

            mock_client.characters.list.assert_awaited_once()
            assert mock_client.characters.list.call_args.kwargs.get("search") == "alan"

    @pytest.mark.asyncio
    async def test_no_local_filtering_returns_server_results_verbatim(self):
        """The command no longer filters locally; all server rows are returned."""
        mock_client = AsyncMock()
        char1 = _make_character(slug="alan-watts", name="Alan Watts")
        char2 = _make_character(slug="coding-asst", name="Coding Assistant")
        mock_response = SimpleNamespace(data=[char1, char2])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _list_characters_async(ctx, output_json=True, search="alan")

            data = json.loads(mock_echo.call_args[0][0])
            # Both rows pass through unfiltered — server is the source of truth.
            assert len(data) == 2

    @pytest.mark.asyncio
    async def test_no_search_omits_search_kwarg(self):
        """When --search is not set, search is not passed to list()."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(data=[_make_character()])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _list_characters_async(ctx, output_json=True, search=None)

            assert "search" not in mock_client.characters.list.call_args.kwargs

    @pytest.mark.asyncio
    async def test_search_no_matches_plain(self):
        """Search set + empty server results shows specific message in plain mode."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(data=[])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _list_characters_async(ctx, output_json=False, search="nomatch")

            mock_om.warning.assert_called_once()
            assert "nomatch" in mock_om.warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_search_no_matches_rich(self):
        """Search set + empty server results shows warning in rich mode."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(data=[])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _list_characters_async(ctx, output_json=False, search="nomatch")

            mock_om.warning.assert_called_once()


# ---------------------------------------------------------------------------
# _list_characters_async tests — filter pass-through (CHAR-06)
# ---------------------------------------------------------------------------


class TestListCharactersAsyncFilters:
    """Tests that CHAR-06 filter flags map to list() kwargs (only when set)."""

    @pytest.mark.asyncio
    async def test_all_filters_passed_through(self):
        """All set filters map to the correct list() kwargs."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(data=[_make_character()])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _list_characters_async(
                ctx,
                output_json=True,
                search=None,
                sort_by="highlyRated",
                sort_order="desc",
                tags=("philosophy", "zen"),
                categories=("education", "lifestyle"),
                limit=20,
                offset=5,
                is_adult=False,
                is_pro=True,
                is_web_enabled=True,
                model_id="llama-3.3-70b",
            )

            kwargs = mock_client.characters.list.call_args.kwargs
            assert kwargs["sort_by"] == "highlyRated"
            assert kwargs["sort_order"] == "desc"
            assert kwargs["tags"] == ["philosophy", "zen"]
            assert kwargs["categories"] == ["education", "lifestyle"]
            assert kwargs["limit"] == 20
            assert kwargs["offset"] == 5
            assert kwargs["is_adult"] is False
            assert kwargs["is_pro"] is True
            assert kwargs["is_web_enabled"] is True
            assert kwargs["model_id"] == ["llama-3.3-70b"]

    @pytest.mark.asyncio
    async def test_unset_filters_omitted(self):
        """Filters left at default are not passed to list()."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(data=[_make_character()])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _list_characters_async(ctx, output_json=True, search=None)

            kwargs = mock_client.characters.list.call_args.kwargs
            for key in (
                "sort_by",
                "sort_order",
                "tags",
                "categories",
                "limit",
                "offset",
                "is_adult",
                "is_pro",
                "is_web_enabled",
                "model_id",
            ):
                assert key not in kwargs

    @pytest.mark.asyncio
    async def test_categories_filter_forwarded_as_list(self):
        """CLI-B-05: --categories (repeatable) maps to list(categories=[...])."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(data=[_make_character()])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _list_characters_async(
                ctx,
                output_json=True,
                search=None,
                categories=("education",),
            )

            kwargs = mock_client.characters.list.call_args.kwargs
            # Mirrors --tags: the CLI passes a list; the SDK does the comma-join.
            assert kwargs["categories"] == ["education"]

    def test_categories_cli_flag_forwarded(self):
        """End-to-end CliRunner: repeated --categories reaches list()."""
        from venice_ai.cli.cli import cli

        mock_client = AsyncMock()
        mock_response = SimpleNamespace(data=[_make_character()])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
        ):
            _setup_client_patch(MockClient, mock_client)

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "characters",
                    "list",
                    "--categories",
                    "education",
                    "--categories",
                    "lifestyle",
                    "--json",
                ],
            )

        assert result.exit_code == 0, result.output
        kwargs = mock_client.characters.list.call_args.kwargs
        assert kwargs["categories"] == ["education", "lifestyle"]


# ---------------------------------------------------------------------------
# _list_characters_async tests — empty list display
# ---------------------------------------------------------------------------


class TestListCharactersAsyncEmpty:
    """Tests for _list_characters_async with empty character list."""

    @pytest.mark.asyncio
    async def test_empty_list_plain(self):
        """Empty character list shows 'No characters found.' in plain mode."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(data=[])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _list_characters_async(ctx, output_json=False, search=None)

            mock_om.warning.assert_called_once()
            assert "No characters found" in mock_om.warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_empty_list_rich(self):
        """Empty character list shows yellow warning in rich mode."""
        mock_client = AsyncMock()
        mock_response = SimpleNamespace(data=[])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _list_characters_async(ctx, output_json=False, search=None)

            mock_om.warning.assert_called_once()


# ---------------------------------------------------------------------------
# _list_characters_async tests — plain display
# ---------------------------------------------------------------------------


class TestListCharactersAsyncPlainDisplay:
    """Tests for _list_characters_async in plain display mode."""

    @pytest.mark.asyncio
    async def test_plain_display_shows_header_and_characters(self):
        """Plain mode shows header, character rows, and total count."""
        mock_client = AsyncMock()
        char = _make_character(slug="test-char", name="Test Character", description="A test")
        mock_response = SimpleNamespace(data=[char])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _list_characters_async(ctx, output_json=False, search=None)

            mock_om.table.assert_called_once()
            call_kwargs = mock_om.table.call_args
            headers = call_kwargs[1]["headers"]
            rows = call_kwargs[1]["rows"]
            assert "Slug" in headers
            assert "Name" in headers
            assert rows[0][0] == "test-char"
            assert rows[0][1] == "Test Character"

    @pytest.mark.asyncio
    async def test_plain_display_truncates_description(self):
        """Plain mode truncates description to 50 chars."""
        mock_client = AsyncMock()
        long_desc = "A" * 80
        char = _make_character(slug="test-char", name="Test", description=long_desc)
        mock_response = SimpleNamespace(data=[char])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _list_characters_async(ctx, output_json=False, search=None)

            rows = mock_om.table.call_args[1]["rows"]
            # The full 80 char desc should not appear in the row
            assert long_desc not in rows[0][2]

    @pytest.mark.asyncio
    async def test_plain_display_handles_none_description(self):
        """Plain mode handles None description without error."""
        mock_client = AsyncMock()
        char = _make_character(slug="test-char", name="Test", description=None)
        mock_response = SimpleNamespace(data=[char])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _list_characters_async(ctx, output_json=False, search=None)

            rows = mock_om.table.call_args[1]["rows"]
            assert rows[0][0] == "test-char"

    @pytest.mark.asyncio
    async def test_plain_display_multiple_characters(self):
        """Plain mode shows all characters in table."""
        mock_client = AsyncMock()
        chars = [_make_character(slug=f"char-{i}", name=f"Char {i}") for i in range(3)]
        mock_response = SimpleNamespace(data=chars)
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _list_characters_async(ctx, output_json=False, search=None)

            rows = mock_om.table.call_args[1]["rows"]
            assert len(rows) == 3
            for i in range(3):
                assert rows[i][0] == f"char-{i}"


# ---------------------------------------------------------------------------
# _list_characters_async tests — rich display
# ---------------------------------------------------------------------------


class TestListCharactersAsyncRichDisplay:
    """Tests for _list_characters_async in rich display mode."""

    @pytest.mark.asyncio
    async def test_rich_display_prints_table_and_total(self):
        """Rich mode prints a table via OutputManager."""
        mock_client = AsyncMock()
        chars = [_make_character(slug=f"char-{i}", name=f"Char {i}") for i in range(2)]
        mock_response = SimpleNamespace(data=chars)
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _list_characters_async(ctx, output_json=False, search=None)

            mock_om.table.assert_called_once()
            rows = mock_om.table.call_args[1]["rows"]
            assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_rich_display_handles_none_description(self):
        """Rich mode handles None description gracefully."""
        mock_client = AsyncMock()
        char = _make_character(slug="test-char", name="Test", description=None)
        mock_response = SimpleNamespace(data=[char])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _list_characters_async(ctx, output_json=False, search=None)

            mock_om.table.assert_called_once()


# ---------------------------------------------------------------------------
# _list_characters_async tests — ctx.obj edge cases
# ---------------------------------------------------------------------------


class TestListCharactersAsyncCtxEdgeCases:
    """Tests for _list_characters_async ctx edge cases."""

    @pytest.mark.asyncio
    async def test_ctx_obj_none_defaults_to_rich(self):
        """ctx.obj is None → plain defaults to False (rich mode)."""
        mock_client = AsyncMock()
        char = _make_character()
        mock_response = SimpleNamespace(data=[char])
        mock_client.characters.list = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = MagicMock()
            ctx.obj = None
            await _list_characters_async(ctx, output_json=False, search=None)

            mock_om.table.assert_called_once()


# ---------------------------------------------------------------------------
# _info_character_async tests — JSON output
# ---------------------------------------------------------------------------


class TestInfoCharacterAsyncJsonOutput:
    """Tests for _info_character_async in JSON output mode."""

    @pytest.mark.asyncio
    async def test_info_json_output_full(self):
        """JSON output mode returns character details as JSON."""
        mock_client = AsyncMock()
        char = _make_character(
            slug="alan-watts",
            name="Alan Watts",
            description="Philosopher",
            modelId="llama-3.3-70b",
            tags=["philosophy", "zen"],
            adult=False,
            shareUrl="https://venice.ai/chat/alan-watts",
            photoUrl="https://venice.ai/photo/alan-watts.jpg",
            webEnabled=True,
            createdAt="2024-01-01T00:00:00Z",
            updatedAt="2024-06-01T00:00:00Z",
            stats=SimpleNamespace(imports=100),
        )
        mock_response = SimpleNamespace(data=char)
        mock_client.characters.get = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _info_character_async(ctx, slug="alan-watts", output_json=True)

            mock_echo.assert_called_once()
            data = json.loads(mock_echo.call_args[0][0])
            assert data["slug"] == "alan-watts"
            assert data["name"] == "Alan Watts"
            assert data["description"] == "Philosopher"
            assert data["modelId"] == "llama-3.3-70b"
            assert data["tags"] == ["philosophy", "zen"]
            assert data["adult"] is False
            assert data["shareUrl"] == "https://venice.ai/chat/alan-watts"
            assert data["photoUrl"] == "https://venice.ai/photo/alan-watts.jpg"
            assert data["webEnabled"] is True
            assert data["stats"]["imports"] == 100

    @pytest.mark.asyncio
    async def test_info_json_includes_identity_and_full_stats(self):
        """CHAR-08: info --json includes id/author/featured/isOwner + full stats."""
        mock_client = AsyncMock()
        char = _make_character(
            slug="alan-watts",
            id="char-uuid-99",
            author="venice-team",
            featured=True,
            isOwner=False,
            stats=SimpleNamespace(
                imports=100,
                averageRating=4.8,
                ratingCount=50,
                ratingSum=240,
                userRating=5,
            ),
        )
        mock_response = SimpleNamespace(data=char)
        mock_client.characters.get = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _info_character_async(ctx, slug="alan-watts", output_json=True)

            data = json.loads(mock_echo.call_args[0][0])
            assert data["id"] == "char-uuid-99"
            assert data["author"] == "venice-team"
            assert data["featured"] is True
            assert data["isOwner"] is False
            assert data["stats"]["imports"] == 100
            assert data["stats"]["averageRating"] == 4.8
            assert data["stats"]["ratingCount"] == 50
            assert data["stats"]["ratingSum"] == 240
            assert data["stats"]["userRating"] == 5

    @pytest.mark.asyncio
    async def test_info_json_output_without_share_url(self):
        """JSON output mode with no shareUrl returns None for shareUrl."""
        mock_client = AsyncMock()
        char = _make_character(shareUrl=None)
        mock_response = SimpleNamespace(data=char)
        mock_client.characters.get = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _info_character_async(ctx, slug="test-char", output_json=True)

            data = json.loads(mock_echo.call_args[0][0])
            assert data["shareUrl"] is None


# ---------------------------------------------------------------------------
# _info_character_async tests — plain display
# ---------------------------------------------------------------------------


class TestInfoCharacterAsyncPlainDisplay:
    """Tests for _info_character_async in plain display mode."""

    @pytest.mark.asyncio
    async def test_info_plain_with_all_fields(self):
        """Plain mode displays all character fields."""
        mock_client = AsyncMock()
        char = _make_character(
            slug="alan-watts",
            name="Alan Watts",
            description="Philosopher",
            modelId="llama-3.3-70b",
            tags=["philosophy", "zen"],
            adult=False,
            shareUrl="https://venice.ai/chat/alan-watts",
            webEnabled=True,
            createdAt="2024-01-01T00:00:00Z",
            updatedAt="2024-06-01T00:00:00Z",
            stats=SimpleNamespace(imports=100),
        )
        mock_response = SimpleNamespace(data=char)
        mock_client.characters.get = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _info_character_async(ctx, slug="alan-watts", output_json=False)

            calls = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Alan Watts" in calls
            assert "alan-watts" in calls
            assert "llama-3.3-70b" in calls
            assert "philosophy" in calls
            assert "zen" in calls
            assert "100" in calls
            assert "2024-01-01T00:00:00Z" in calls
            assert "https://venice.ai/chat/alan-watts" in calls

    @pytest.mark.asyncio
    async def test_info_plain_with_description(self):
        """Plain mode shows description when present."""
        mock_client = AsyncMock()
        char = _make_character(description="A deep philosopher")
        mock_response = SimpleNamespace(data=char)
        mock_client.characters.get = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _info_character_async(ctx, slug="test-char", output_json=False)

            calls = "".join(str(c) for c in mock_echo.call_args_list)
            assert "A deep philosopher" in calls
            assert "Description" in calls

    @pytest.mark.asyncio
    async def test_info_plain_without_description(self):
        """Plain mode shows nothing when description is empty."""
        mock_client = AsyncMock()
        char = _make_character(description=None)
        mock_response = SimpleNamespace(data=char)
        mock_client.characters.get = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _info_character_async(ctx, slug="test-char", output_json=False)

            # Should not crash, just doesn't print description
            assert mock_echo.called

    @pytest.mark.asyncio
    async def test_info_plain_without_share_url(self):
        """Plain mode skips Share URL line when shareUrl is None."""
        mock_client = AsyncMock()
        char = _make_character(shareUrl=None)
        mock_response = SimpleNamespace(data=char)
        mock_client.characters.get = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _info_character_async(ctx, slug="test-char", output_json=False)

            calls = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Share URL" not in calls

    @pytest.mark.asyncio
    async def test_info_plain_with_empty_tags(self):
        """Plain mode shows 'None' for empty tags list."""
        mock_client = AsyncMock()
        char = _make_character(tags=[])
        mock_response = SimpleNamespace(data=char)
        mock_client.characters.get = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _info_character_async(ctx, slug="test-char", output_json=False)

            calls = "".join(str(c) for c in mock_echo.call_args_list)
            assert "None" in calls


# ---------------------------------------------------------------------------
# _info_character_async tests — rich display
# ---------------------------------------------------------------------------


class TestInfoCharacterAsyncRichDisplay:
    """Tests for _info_character_async in rich display mode."""

    @pytest.mark.asyncio
    async def test_info_rich_display_with_all_fields(self):
        """Rich mode displays character in a panel with all fields."""
        mock_client = AsyncMock()
        char = _make_character(
            slug="alan-watts",
            name="Alan Watts",
            description="A philosopher",
            modelId="llama-3.3-70b",
            tags=["philosophy"],
            adult=False,
            shareUrl="https://venice.ai/chat/alan-watts",
            webEnabled=True,
            stats=SimpleNamespace(imports=42),
        )
        mock_response = SimpleNamespace(data=char)
        mock_client.characters.get = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _info_character_async(ctx, slug="alan-watts", output_json=False)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_info_rich_display_without_description(self):
        """Rich mode displays panel without description block when description is None."""
        mock_client = AsyncMock()
        char = _make_character(description=None)
        mock_response = SimpleNamespace(data=char)
        mock_client.characters.get = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _info_character_async(ctx, slug="test-char", output_json=False)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_info_rich_display_without_share_url(self):
        """Rich mode displays panel without share URL when shareUrl is None."""
        mock_client = AsyncMock()
        char = _make_character(shareUrl=None)
        mock_response = SimpleNamespace(data=char)
        mock_client.characters.get = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _info_character_async(ctx, slug="test-char", output_json=False)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_info_rich_display_with_empty_tags(self):
        """Rich mode shows 'None' for empty tags list."""
        mock_client = AsyncMock()
        char = _make_character(tags=None)
        char.tags = None  # Override to test None tags
        mock_response = SimpleNamespace(data=char)
        mock_client.characters.get = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _info_character_async(ctx, slug="test-char", output_json=False)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_info_rich_display_adult_character(self):
        """Rich mode shows 'Yes' for adult character."""
        mock_client = AsyncMock()
        char = _make_character(adult=True)
        mock_response = SimpleNamespace(data=char)
        mock_client.characters.get = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _info_character_async(ctx, slug="test-char", output_json=False)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_info_rich_display_web_disabled(self):
        """Rich mode shows 'Disabled' when webEnabled is False."""
        mock_client = AsyncMock()
        char = _make_character(webEnabled=False)
        mock_response = SimpleNamespace(data=char)
        mock_client.characters.get = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _info_character_async(ctx, slug="test-char", output_json=False)

            assert mock_console.print.called

    @pytest.mark.asyncio
    async def test_info_rich_ctx_obj_none_defaults_to_rich(self):
        """ctx.obj is None → plain defaults to False (rich mode)."""
        mock_client = AsyncMock()
        char = _make_character()
        mock_response = SimpleNamespace(data=char)
        mock_client.characters.get = AsyncMock(return_value=mock_response)

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.console") as mock_console,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = MagicMock()
            ctx.obj = None
            await _info_character_async(ctx, slug="test-char", output_json=False)

            assert mock_console.print.called


# ---------------------------------------------------------------------------
# CLI entrypoint tests — list_characters
# ---------------------------------------------------------------------------


class TestListCharactersCLI:
    """Tests for the list_characters() click command entrypoint."""

    def test_list_help(self):
        """Check --help output for list command."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["characters", "list", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output.lower() or "character" in result.output.lower()

    def test_list_invokes_asyncio_run(self):
        """list_characters() calls asyncio.run with _list_characters_async coroutine."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.characters.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["characters", "list"])
            assert mock_run.called

    def test_list_json_flag(self):
        """list --json passes output_json=True."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.characters.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(cli, ["characters", "list", "--json"])
            assert result.exit_code == 0

    def test_list_search_flag(self):
        """list --search passes search term correctly."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.characters.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(cli, ["characters", "list", "--search", "coding"])
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# CLI entrypoint tests — info_character
# ---------------------------------------------------------------------------


class TestInfoCharacterCLI:
    """Tests for the info_character() click command entrypoint."""

    def test_info_help(self):
        """Check --help output for info command."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["characters", "info", "--help"])
        assert result.exit_code == 0

    def test_info_invokes_asyncio_run(self):
        """info_character() calls asyncio.run with _info_character_async coroutine."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.characters.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["characters", "info", "alan-watts"])
            assert mock_run.called

    def test_info_json_flag(self):
        """info --json passes output_json=True."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.characters.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(cli, ["characters", "info", "alan-watts", "--json"])
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# characters() group tests
# ---------------------------------------------------------------------------


class TestCharactersGroup:
    """Tests for the characters() click group."""

    def test_characters_group_help(self):
        """Check --help output for the characters group."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["characters", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "info" in result.output
        assert "reviews" in result.output


# ---------------------------------------------------------------------------
# _reviews_character_async tests (CHAR-07)
# ---------------------------------------------------------------------------


class TestReviewsCharacterAsync:
    """Tests for _reviews_character_async core logic."""

    @pytest.mark.asyncio
    async def test_reviews_calls_resource_with_pagination(self):
        """reviews subcommand calls client.characters.reviews(slug, page=, page_size=)."""
        mock_client = AsyncMock()
        mock_client.characters.reviews = AsyncMock(return_value=_make_reviews_response())

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _reviews_character_async(
                ctx, slug="alan-watts", output_json=False, page=2, page_size=50
            )

            mock_client.characters.reviews.assert_awaited_once()
            args = mock_client.characters.reviews.call_args
            assert args.args[0] == "alan-watts"
            assert args.kwargs["page"] == 2
            assert args.kwargs["page_size"] == 50

    @pytest.mark.asyncio
    async def test_reviews_plain_prints_message(self):
        """Plain mode prints each review's message, rating, and createdAt."""
        mock_client = AsyncMock()
        review = _make_review(rating=4, createdAt="2024-05-02T00:00:00Z", message="Loved it")
        mock_client.characters.reviews = AsyncMock(
            return_value=_make_reviews_response(reviews=[review])
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo") as mock_echo,
            patch("venice_ai.cli.utils.output.OutputManager"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _reviews_character_async(
                ctx, slug="alan-watts", output_json=False, page=None, page_size=None
            )

            calls = "".join(str(c) for c in mock_echo.call_args_list)
            assert "Loved it" in calls
            assert "4/5" in calls
            assert "2024-05-02T00:00:00Z" in calls

    @pytest.mark.asyncio
    async def test_reviews_json_output(self):
        """JSON output includes summary and each review's rating/createdAt/message."""
        mock_client = AsyncMock()
        review = _make_review(rating=3, message="Okay")
        mock_client.characters.reviews = AsyncMock(
            return_value=_make_reviews_response(reviews=[review], averageRating=3.0, totalReviews=1)
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.commands.characters.click.echo") as mock_echo,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _reviews_character_async(
                ctx, slug="alan-watts", output_json=True, page=None, page_size=None
            )

            data = json.loads(mock_echo.call_args[0][0])
            assert data["summary"]["averageRating"] == 3.0
            assert data["summary"]["totalReviews"] == 1
            assert data["reviews"][0]["rating"] == 3
            assert data["reviews"][0]["message"] == "Okay"
            assert "createdAt" in data["reviews"][0]

    @pytest.mark.asyncio
    async def test_reviews_empty_shows_warning(self):
        """No reviews shows a warning (non-JSON)."""
        mock_client = AsyncMock()
        mock_client.characters.reviews = AsyncMock(
            return_value=_make_reviews_response(reviews=[], totalReviews=0)
        )

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx()
            await _reviews_character_async(
                ctx, slug="alan-watts", output_json=False, page=None, page_size=None
            )

            mock_om.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_reviews_plain_enables_plain_mode(self):
        """Plain ctx enables plain mode so OutputManager output is not rich-formatted."""
        mock_client = AsyncMock()
        mock_client.characters.reviews = AsyncMock(return_value=_make_reviews_response())

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.console.enable_plain_mode") as mock_enable,
            patch("venice_ai.cli.utils.console.is_plain_mode", return_value=False),
            patch("venice_ai.cli.utils.output.OutputManager"),
            patch("venice_ai.cli.commands.characters.click.echo"),
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=True)
            await _reviews_character_async(
                ctx, slug="alan-watts", output_json=False, page=None, page_size=None
            )

            mock_enable.assert_called_once()

    @pytest.mark.asyncio
    async def test_reviews_rich_table(self):
        """Rich mode renders a table of reviews."""
        mock_client = AsyncMock()
        mock_client.characters.reviews = AsyncMock(return_value=_make_reviews_response())

        with (
            patch("venice_ai.VeniceClient") as MockClient,
            patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
            patch("venice_ai.cli.utils.output.OutputManager") as mock_om,
        ):
            _setup_client_patch(MockClient, mock_client)

            ctx = _make_ctx(plain=False)
            await _reviews_character_async(
                ctx, slug="alan-watts", output_json=False, page=None, page_size=None
            )

            mock_om.table.assert_called_once()


class TestReviewsCharacterCLI:
    """Tests for the reviews_character() click command entrypoint."""

    def test_reviews_help(self):
        """Check --help output for reviews command."""
        from venice_ai.cli.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["characters", "reviews", "--help"])
        assert result.exit_code == 0

    def test_reviews_invokes_asyncio_run(self):
        """reviews_character() calls asyncio.run."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch(
            "venice_ai.cli.commands.characters.asyncio.run", side_effect=consume_coro
        ) as mock_run:
            runner.invoke(cli, ["characters", "reviews", "alan-watts"])
            assert mock_run.called

    def test_reviews_pagination_flags(self):
        """reviews --page/--page-size parse without error."""
        from venice_ai.cli.cli import cli

        def consume_coro(coro):
            coro.close()
            return None

        runner = CliRunner()
        with patch("venice_ai.cli.commands.characters.asyncio.run", side_effect=consume_coro):
            result = runner.invoke(
                cli,
                ["characters", "reviews", "alan-watts", "--page", "2", "--page-size", "10"],
            )
            assert result.exit_code == 0
