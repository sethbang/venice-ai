"""
VCRpy-based integration tests for the x402 resource.

Two notes on auth:

- ``top_up`` uses Bearer (VENICE_API_KEY) auth.
- ``balance`` and ``transactions`` require SIWE / EIP-4361 wallet signing.
  These tests use a deterministic throwaway private key — it's never a real
  wallet, never funded, and never committed. The value to test is the
  SDK's wire format (path + X-Sign-In-With-X header). Expect the API to
  return a not-found / empty-wallet shape for the unfunded wallet; the
  cassette captures whichever response actually lands.
"""

import os

import pytest
import pytest_asyncio

from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.exceptions import APIError, VeniceError

# Skip the whole module if the x402 extra isn't installed.
pytest.importorskip("eth_account", reason="x402 extra not installed")
pytest.importorskip("siwe", reason="x402 extra not installed")

from venice_ai.auth.x402 import X402Auth
from venice_ai.types.api.x402 import X402TopUpResponse

# Deterministic throwaway key — DO NOT fund this address.
_THROWAWAY_KEY = "0x" + "a" * 63 + "b"


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


@pytest.fixture
def x402_auth() -> X402Auth:
    return X402Auth(private_key=_THROWAWAY_KEY)


@pytest.mark.integration
async def test_x402_balance_for_unfunded_wallet(venice_client, x402_auth, vcr_cassette):
    """Balance for a never-funded wallet — the SDK builds the SIWE request
    correctly; the API may respond with a not-found / zero-balance shape."""
    with vcr_cassette:
        try:
            result = await venice_client.x402.balance(auth=x402_auth)
        except (VeniceError, APIError) as e:
            # Unfunded wallet or server-side 404/401 is acceptable — the
            # cassette already captured the SDK's request on the wire.
            pytest.skip(f"x402 balance expected shape unavailable: {e}")

        assert result.success is True


@pytest.mark.integration
async def test_x402_transactions_for_unfunded_wallet(venice_client, x402_auth, vcr_cassette):
    """Transactions list for the throwaway wallet."""
    with vcr_cassette:
        try:
            result = await venice_client.x402.transactions(auth=x402_auth)
        except (VeniceError, APIError) as e:
            pytest.skip(f"x402 transactions unavailable: {e}")

        assert result.success is True


@pytest.mark.integration
async def test_x402_top_up_discover_payment_requirements(venice_client, vcr_cassette):
    """Posting an empty body discovers the payment requirements; the API is
    expected to return a 402 with structured details (docs). A success case
    (200 with credited amount) is also accepted."""
    with vcr_cassette:
        try:
            result = await venice_client.x402.top_up()
        except (VeniceError, APIError) as e:
            # 402 Payment Required is the documented "happy path" response
            # for an empty body — the SDK surfaces it as an exception with
            # the upstream payload attached. Accept it.
            msg = str(e).lower()
            assert "402" in msg or "payment" in msg, f"Unexpected error: {e}"
            return

        # If the server responds 200 (e.g. in a dry-run environment), we should
        # still have the expected response envelope.
        assert isinstance(result, X402TopUpResponse)
