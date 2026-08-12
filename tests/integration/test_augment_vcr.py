"""
VCRpy-based integration tests for the Augment resource.

Covers POST ``/augment/scrape``, ``/augment/search``, and ``/augment/text-parser``
against live Venice endpoints, replaying via cassettes on subsequent runs.
"""

import os

import pytest
import pytest_asyncio

from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.exceptions import APIError, VeniceError
from venice_ai.types.api.augment import (
    AugmentScrapeResponse,
    AugmentSearchResponse,
    AugmentTextParserResponse,
)


@pytest_asyncio.fixture
async def venice_client(backend_instance):
    api_key = os.getenv("VENICE_API_KEY")
    if not api_key:
        pytest.skip("VENICE_API_KEY environment variable required for integration tests")

    client = create_test_venice_client(
        api_key=api_key,
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=True,
    )
    try:
        yield client
    finally:
        await client.close()


@pytest.mark.integration
async def test_augment_scrape(venice_client, vcr_cassette):
    """Scraping example.com returns a markdown response envelope."""
    with vcr_cassette:
        try:
            result = await venice_client.augment.scrape(url="https://example.com")
        except (VeniceError, APIError) as e:
            pytest.skip(f"Augment scrape unavailable: {e}")

        assert isinstance(result, AugmentScrapeResponse)
        assert result.url == "https://example.com"
        assert result.format == "markdown"
        assert isinstance(result.content, str)
        assert len(result.content) > 0


@pytest.mark.integration
async def test_augment_search(venice_client, vcr_cassette):
    """Running a simple search returns structured results."""
    with vcr_cassette:
        try:
            result = await venice_client.augment.search(
                query="Venice AI",
                limit=3,
                search_provider="brave",
            )
        except (VeniceError, APIError) as e:
            pytest.skip(f"Augment search unavailable: {e}")

        assert isinstance(result, AugmentSearchResponse)
        assert isinstance(result.query, str)
        assert isinstance(result.results, list)


@pytest.mark.integration
async def test_augment_parse_text(venice_client, vcr_cassette, tmp_path):
    """Text parser on a plain text file returns extracted text + token count."""
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("Venice AI is a privacy-first inference platform.\n")

    with vcr_cassette:
        try:
            result = await venice_client.augment.parse_text(file=str(txt_path))
        except (VeniceError, APIError) as e:
            pytest.skip(f"Augment text-parser unavailable: {e}")

        assert isinstance(result, AugmentTextParserResponse)
        assert "Venice AI" in result.text
        assert result.tokens >= 1
