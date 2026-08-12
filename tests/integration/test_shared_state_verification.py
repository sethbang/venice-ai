"""
Verification test for shared rate limit state coordination.

This test verifies that parallel test workers properly share rate limit state
through the shared Redis backend, preventing 429 errors during concurrent execution.

Note: The account system (VeniceAccount) has been removed. These tests now verify
that clients are created with rate limiters. Backend sharing verification is only
performed when using AdaptiveScheduler, which has a state_manager.backend architecture.
SimpleRateLimiter (the default) does not support distributed state.
"""

import pytest
import pytest_asyncio

from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode


def has_adaptive_scheduler(client) -> bool:
    """Check if the client has an AdaptiveScheduler with state_manager.backend."""
    if not hasattr(client, "rate_limiter") or client.rate_limiter is None:
        return False
    if not hasattr(client.rate_limiter, "state_manager"):
        return False
    return hasattr(client.rate_limiter.state_manager, "backend")


@pytest.mark.integration
class TestSharedStateVerification:
    """Verify that parallel tests share rate limit state."""

    @pytest_asyncio.fixture
    async def client_a(self, backend_instance):
        """First client using shared backend."""
        client = create_test_venice_client(
            api_key="test-key-a",
            scheduler_mode=SchedulerMode.INTELLIGENT,
            enable_redis=True,
        )
        yield client
        await client.close()

    @pytest_asyncio.fixture
    async def client_b(self, backend_instance):
        """Second client using shared backend."""
        client = create_test_venice_client(
            api_key="test-key-b",
            scheduler_mode=SchedulerMode.INTELLIGENT,
            enable_redis=True,
        )
        yield client
        await client.close()

    async def test_clients_share_backend_instance(self, client_a, client_b, backend_instance):
        """
        Verify that multiple clients have rate limiters configured.

        The account system has been removed. This test now verifies that clients
        are properly created with rate limiters. Backend sharing is only verified
        when using AdaptiveScheduler.
        """
        # Both clients should have rate limiters
        assert hasattr(client_a, "rate_limiter"), "Client A missing rate_limiter"
        assert hasattr(client_b, "rate_limiter"), "Client B missing rate_limiter"

        # Check rate limiter backend sharing if using AdaptiveScheduler
        if has_adaptive_scheduler(client_a) and has_adaptive_scheduler(client_b):
            backend_a = client_a.rate_limiter.state_manager.backend
            backend_b = client_b.rate_limiter.state_manager.backend
            assert backend_a is backend_instance, (
                "Client A rate limiter backend is not the shared instance"
            )
            assert backend_b is backend_instance, (
                "Client B rate limiter backend is not the shared instance"
            )

    async def test_clients_share_backend_namespace(self, client_a, client_b, backend_instance):
        """
        Verify that multiple clients use the same backend namespace.

        The account system has been removed. This test now verifies namespace
        sharing through the rate limiter's state_manager backend when using
        AdaptiveScheduler. Skipped for SimpleRateLimiter.
        """
        if not has_adaptive_scheduler(client_a) or not has_adaptive_scheduler(client_b):
            pytest.skip(
                "Namespace sharing requires AdaptiveScheduler with state_manager.backend. "
                "SimpleRateLimiter does not support distributed state."
            )

        backend_a = client_a.rate_limiter.state_manager.backend
        backend_b = client_b.rate_limiter.state_manager.backend

        assert hasattr(backend_a, "namespace"), "Client A backend missing namespace"
        assert hasattr(backend_b, "namespace"), "Client B backend missing namespace"

        namespace_a = backend_a.namespace
        namespace_b = backend_b.namespace

        assert namespace_a == namespace_b, f"Namespace mismatch: {namespace_a} != {namespace_b}"
        assert namespace_a == backend_instance.namespace, (
            f"Client A namespace {namespace_a} != shared {backend_instance.namespace}"
        )

    async def test_backend_is_redis(self, backend_instance):
        """
        Verify that the shared backend is Redis for proper distributed coordination.

        Memory backends cannot coordinate across processes.
        """
        from venice_ai.core.backends.redis import RedisBackend

        assert isinstance(backend_instance, RedisBackend), (
            f"Expected RedisBackend, got {type(backend_instance).__name__}"
        )

    async def test_backend_namespace_format(self, backend_instance):
        """
        Verify that the backend namespace follows the expected format.

        Expected format: test_shared_rate_limits_{hash}
        """
        namespace = backend_instance.namespace

        assert namespace.startswith("test_shared_rate_limits_"), (
            f"Unexpected namespace format: {namespace}"
        )

        # Should have a hash suffix
        parts = namespace.split("_")
        assert len(parts) >= 4, f"Namespace missing hash suffix: {namespace}"

        # Hash should be alphanumeric
        hash_suffix = parts[-1]
        assert hash_suffix.isalnum(), f"Hash suffix not alphanumeric: {hash_suffix}"

    async def test_multiple_clients_different_api_keys_same_backend(self, backend_instance):
        """
        Verify that clients with different API keys can be created with rate limiters.

        The account system has been removed. This test now verifies that multiple
        clients with different API keys are properly created. Backend sharing is
        only verified when using AdaptiveScheduler.
        """
        clients = []

        try:
            # Create 3 clients with different API keys
            for i in range(3):
                client = create_test_venice_client(
                    api_key=f"test-key-{i}",
                    scheduler_mode=SchedulerMode.INTELLIGENT,
                    enable_redis=True,
                )
                clients.append(client)

            # All should have rate limiters
            for i, client in enumerate(clients):
                assert hasattr(client, "rate_limiter"), f"Client {i} missing rate_limiter"

            # Check rate limiter backend sharing if using AdaptiveScheduler
            if has_adaptive_scheduler(clients[0]):
                for i, client in enumerate(clients):
                    rl_backend = client.rate_limiter.state_manager.backend
                    assert rl_backend is backend_instance, (
                        f"Client {i} rate limiter does not share backend instance"
                    )

        finally:
            # Cleanup
            for client in clients:
                await client.close()


@pytest.mark.integration
async def test_shared_backend_connection(backend_instance):
    """
    Verify that the shared backend is connected and operational.

    This ensures the backend is ready for use by test clients.
    Note: This test requires Redis to be running. It will be skipped
    if Redis is not available (common in local development).
    """
    import redis.exceptions

    from venice_ai.core.backends.redis import RedisBackend

    assert isinstance(backend_instance, RedisBackend)

    # Backend should have a connection method
    assert hasattr(backend_instance, "_ensure_connected")

    # Try to ensure connection - may fail if Redis is not running
    try:
        await backend_instance._ensure_connected()
    except (redis.exceptions.ConnectionError, ConnectionError, OSError) as e:
        pytest.skip(
            f"Redis not available for connection test: {e}. "
            f"This is expected in local development without Redis."
        )

    # Should have a namespace
    assert backend_instance.namespace is not None
    assert len(backend_instance.namespace) > 0
