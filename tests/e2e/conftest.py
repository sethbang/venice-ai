"""
E2E Test Configuration.

Configuration-driven model selection with offline fallback for E2E tests
against the live Venice API.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.test_models import TEST_MODELS
from venice_ai import VeniceClient
from venice_ai.exceptions import APIError, RateLimitError

# Model selection via environment or config file (NOT hardcoded in source)
E2E_TEST_MODEL = os.environ.get("VENICE_E2E_TEST_MODEL")  # Explicit override
E2E_PREFERRED_TIER = os.environ.get("VENICE_E2E_TIER", "XS")  # Default to cheapest

# Fallback models sourced from the shared test model config.
# When a model cache exists (via ``pytest --refresh-models``), these resolve
# to dynamically discovered models.  Without a cache, the hardcoded defaults
# in test_models.py are used — no API call required at import time.
#
# ``FASTEST_TEXT_MODEL`` is first because it's a plain (non-reasoning) chat
# model — several e2e tests (e.g. ``test_streaming_token_accuracy``) assert
# on ``delta.content`` and will see empty content when the model routes output
# into ``reasoning_content`` instead.
FALLBACK_MODELS = [
    TEST_MODELS.FASTEST_TEXT_MODEL,
    TEST_MODELS.SMALL_TEXT_MODEL,
]

# Cache validity period
CACHE_MAX_AGE_HOURS = 24

# Tolerance margins for state verification
REMAINING_MARGIN = 5  # Allow ±5 requests difference
TOKEN_MARGIN = 10000  # Allow ±10K tokens difference
RESET_TIME_MARGIN = 10  # Allow ±10 seconds drift

# Cost budget limits
E2E_MONTHLY_BUDGET = {
    "max_requests": 10000,  # ~333/day for daily runs
    "max_tokens": 5000000,  # 5M tokens/month
    "tier_allowed": "XS",  # Use XS tier for cost efficiency
}


def get_e2e_test_model_sync() -> str:
    """
    Get test model WITHOUT using rate-limited API calls.

    This avoids circular dependency where model selection itself
    could be rate-limited, causing test setup failures.

    Priority:
    1. Explicit environment variable (VENICE_E2E_TEST_MODEL)
    2. Valid cached model from recent discovery run
    3. Fallback model list (NO API call required)
    4. Fail fast with clear error message
    """
    # Priority 1: Explicit configuration
    if E2E_TEST_MODEL:
        return E2E_TEST_MODEL

    # Priority 2: Cached from previous run (with staleness check)
    cache_file = Path(".e2e_test_model_cache")
    if cache_file.exists():
        try:
            cache_data = json.loads(cache_file.read_text())
            cached_model = cache_data.get("model")
            cached_at_str = cache_data.get("cached_at", "1970-01-01")
            cached_at = datetime.fromisoformat(cached_at_str.replace("Z", "+00:00"))

            # Check if cache is fresh (less than 24 hours old)
            now = datetime.now(cached_at.tzinfo) if cached_at.tzinfo else datetime.now()
            if now - cached_at < timedelta(hours=CACHE_MAX_AGE_HOURS):
                if cached_model:
                    return cached_model
            else:
                print(f"Warning: Model cache is stale ({cached_at}), using fallback")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Could not parse model cache: {e}")

    # Priority 3: Fallback models (NO API CALL - breaks circular dependency)
    for model in FALLBACK_MODELS:
        print(f"Using fallback model: {model} (no API call required)")
        return model

    raise RuntimeError(
        "E2E test model not configured. Set VENICE_E2E_TEST_MODEL environment variable "
        "or update FALLBACK_MODELS list in conftest.py"
    )


class MarginConfig:
    """Configuration for tolerance margins in state verification."""

    REMAINING_MARGIN = REMAINING_MARGIN
    TOKEN_MARGIN = TOKEN_MARGIN
    RESET_TIME_MARGIN = RESET_TIME_MARGIN


@pytest.fixture
def margin_config() -> MarginConfig:
    """Provide margin configuration for state verification."""
    return MarginConfig()


@pytest.fixture
def e2e_retry_config() -> dict[str, Any]:
    """
    Retry configuration for E2E tests.

    Provides settings for handling transient failures in live API tests.
    """
    return {
        "max_retries": 3,
        "retry_delay": 2.0,
        "retry_on": [APIError, TimeoutError, RateLimitError],
    }


@pytest.fixture
async def e2e_client() -> AsyncGenerator[VeniceClient]:
    """
    Create a VeniceClient for E2E tests.

    Uses the VENICE_API_KEY environment variable.
    """
    api_key = os.environ.get("VENICE_API_KEY")
    if not api_key:
        pytest.skip("VENICE_API_KEY environment variable not set")

    client = VeniceClient(api_key=api_key)
    yield client
    await client.close()


@pytest.fixture
def e2e_test_model() -> str:
    """Provide the E2E test model name."""
    return get_e2e_test_model_sync()
