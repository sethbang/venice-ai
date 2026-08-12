"""
Testing configuration preset for Venice AI SDK.

This preset is optimized for automated testing with:
- Memory backend for test isolation
- Relaxed rate limits (10x multiplier)
- Basic scheduler for predictable test execution
- Minimal overhead for fast test runs
- Circuit breaker disabled for consistent testing
"""

from ..core.config import (
    BackendConfig,
    BackendType,
    CircuitBreakerConfig,
    HttpClientConfig,
    SchedulerConfig,
    SchedulerMode,
    VeniceAIConfig,
)


def create_testing_config(
    test_rate_multiplier: float = 10.0,
    enable_circuit_breaker: bool = False,
) -> VeniceAIConfig:
    """
    Create a testing-optimized configuration.

    This configuration is designed for automated testing and provides:
    - Memory backend for test isolation (no shared state)
    - Basic scheduler for deterministic execution
    - 10x rate limit multiplier for faster tests
    - Circuit breaker disabled by default (can be enabled for testing CB logic)
    - Minimal timeouts for fast test execution
    - No metrics collection overhead

    Args:
        test_rate_multiplier: Multiplier for rate limits (default: 10.0 = 10x more lenient)
        enable_circuit_breaker: Enable circuit breaker for testing CB logic (default: False)

    Note:
        "Circuit breaker" here refers to the failure-recovery feature of the
        extracted ``adaptive-rate-limiter`` package's scheduler. It only takes
        effect when ``RateLimiterMode.ADAPTIVE`` (or ``SchedulerMode.INTELLIGENT``)
        is in use; under SIMPLE or DISABLED modes the configuration is inert.

    Returns:
        VeniceAIConfig configured for testing

    Example:
        >>> from venice_ai.presets import create_testing_config
        >>> from venice_ai import VeniceClient
        >>>
        >>> # In your test fixtures
        >>> @pytest.fixture
        >>> async def test_client():
        ...     config = create_testing_config()
        ...     async with VeniceClient(config=config, api_key="test-key") as client:
        ...         yield client

    Best Practices:
        - Use this preset in your pytest fixtures
        - Don't use real API keys in tests
        - Use VCR.py or mocks for external API calls
        - Tests should be isolated and deterministic
    """
    return VeniceAIConfig(
        environment="test",
        debug=False,  # Less noise in test output
        # Memory backend for test isolation
        backend=BackendConfig(
            backend_type=BackendType.MEMORY,
        ),
        # Fast HTTP settings for tests
        http_client=HttpClientConfig(
            timeout=10.0,  # Short timeout for fast failure
            max_connections=10,  # Low concurrency for tests
            max_keepalive_connections=3,
            max_retries=1,  # Minimal retries for faster tests
        ),
        # Basic scheduler for deterministic tests
        scheduler=SchedulerConfig(
            mode=SchedulerMode.BASIC,
            max_concurrent_executions=5,  # Low for predictable tests
            max_queue_size=50,
            enable_rate_limiting=True,
            rate_limit_buffer_ratio=0.9,
            overflow_policy="reject",
            metrics_enabled=False,  # No metrics overhead
            enable_performance_tracking=False,
            test_rate_multiplier=test_rate_multiplier,  # 10x more lenient
        ),
        # Circuit breaker configuration
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=100 if not enable_circuit_breaker else 5,
            reset_timeout=1.0,  # Fast recovery in tests
            success_threshold=1,
        )
        if enable_circuit_breaker
        else CircuitBreakerConfig(
            failure_threshold=999,  # Effectively disabled
            reset_timeout=1.0,
            success_threshold=1,
        ),
    )


def create_testing_config_with_intelligent_scheduler(
    test_rate_multiplier: float = 10.0,
) -> VeniceAIConfig:
    """
    Create testing config with intelligent scheduler.

    Use this when you want to test the intelligent scheduler behavior
    including rate limiting, queue management, etc.

    Args:
        test_rate_multiplier: Rate limit multiplier (default: 10.0)

    Returns:
        VeniceAIConfig with intelligent scheduler for testing
    """
    return VeniceAIConfig(
        environment="test",
        debug=False,
        backend=BackendConfig(
            backend_type=BackendType.MEMORY,
        ),
        http_client=HttpClientConfig(
            timeout=15.0,
            max_connections=20,
            max_keepalive_connections=5,
            max_retries=2,
        ),
        # Use intelligent scheduler for testing its behavior
        scheduler=SchedulerConfig(
            mode=SchedulerMode.INTELLIGENT,
            max_concurrent_executions=10,
            max_queue_size=100,
            enable_rate_limiting=True,
            rate_limit_buffer_ratio=0.9,
            overflow_policy="reject",
            metrics_enabled=True,  # Enable for testing metrics
            enable_performance_tracking=True,
            test_rate_multiplier=test_rate_multiplier,
        ),
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=999,  # Disabled
            reset_timeout=1.0,
            success_threshold=1,
        ),
    )


def create_testing_config_for_circuit_breaker(
    failure_threshold: int = 5,
    reset_timeout: float = 5.0,
) -> VeniceAIConfig:
    """
    Create testing config specifically for testing circuit breaker logic.

    Args:
        failure_threshold: Number of failures before opening circuit (default: 5)
        reset_timeout: Seconds before testing recovery (default: 5.0)

    Note:
        "Circuit breaker" here refers to the failure-recovery feature of the
        extracted ``adaptive-rate-limiter`` package's scheduler. It only takes
        effect when ``RateLimiterMode.ADAPTIVE`` (or ``SchedulerMode.INTELLIGENT``)
        is in use; under SIMPLE or DISABLED modes the configuration is inert.

    Returns:
        VeniceAIConfig configured for circuit breaker testing
    """
    return VeniceAIConfig(
        environment="test",
        debug=True,  # Enable debug for CB testing
        backend=BackendConfig(
            backend_type=BackendType.MEMORY,
        ),
        http_client=HttpClientConfig(
            timeout=5.0,
            max_connections=5,
            max_keepalive_connections=2,
            max_retries=0,  # No retries for testing CB
        ),
        scheduler=SchedulerConfig(
            mode=SchedulerMode.BASIC,
            max_concurrent_executions=3,
            max_queue_size=20,
            enable_rate_limiting=False,
            metrics_enabled=False,
            enable_performance_tracking=False,
        ),
        # Circuit breaker with test-friendly settings
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            reset_timeout=reset_timeout,
            success_threshold=1,
        ),
    )
