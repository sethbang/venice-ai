"""
Unit tests for configuration preset factory functions.

Tests cover:
- Development preset variants
- Production preset variants
- Testing preset variants
- Preset configuration validation
"""

from venice_ai.core.config import (
    BackendType,
    SchedulerMode,
)
from venice_ai.presets import (
    create_development_config,
    create_production_config,
    create_testing_config,
)
from venice_ai.presets.development import (
    create_development_config_with_rate_limiting,
)
from venice_ai.presets.production import (
    create_production_config_conservative,
    create_production_config_high_throughput,
)
from venice_ai.presets.testing import (
    create_testing_config_for_circuit_breaker,
    create_testing_config_with_intelligent_scheduler,
)


class TestDevelopmentPresets:
    """Test development configuration presets."""

    def test_create_development_config_defaults(self):
        """Test default development config."""
        config = create_development_config()

        assert config.environment == "development"
        assert config.debug is True
        assert config.backend.backend_type == BackendType.MEMORY
        assert config.scheduler.mode == SchedulerMode.BASIC
        assert config.scheduler.enable_rate_limiting is False
        assert config.http_client.timeout == 60.0

    def test_create_development_config_custom_debug(self):
        """Test development config with debug disabled."""
        config = create_development_config(enable_debug=False)

        assert config.debug is False
        assert config.environment == "development"

    def test_create_development_config_custom_timeout(self):
        """Test development config with custom timeout."""
        config = create_development_config(timeout=120.0)

        assert config.http_client.timeout == 120.0

    def test_create_development_config_with_rate_limiting(self):
        """Test development config with rate limiting."""
        config = create_development_config_with_rate_limiting()

        assert config.environment == "development"
        assert config.scheduler.mode == SchedulerMode.INTELLIGENT
        assert config.scheduler.enable_rate_limiting is True
        assert config.scheduler.metrics_enabled is True
        assert config.backend.backend_type == BackendType.MEMORY

    def test_create_development_config_with_rate_limiting_debug_disabled(self):
        """Test development config with rate limiting and debug off."""
        config = create_development_config_with_rate_limiting(enable_debug=False)

        assert config.debug is False
        assert config.scheduler.enable_rate_limiting is True


class TestProductionPresets:
    """Test production configuration presets."""

    def test_create_production_config_defaults(self):
        """Test default production config."""
        config = create_production_config(
            redis_url="redis://localhost:6379", _allow_localhost_for_testing=True
        )

        assert config.environment == "production"
        assert config.debug is False
        assert config.backend.backend_type == BackendType.REDIS
        assert config.backend.redis is not None
        assert config.backend.redis.redis_url == "redis://localhost:6379"
        assert config.backend.redis.key_prefix == "venice:prod:"
        assert config.scheduler.mode == SchedulerMode.INTELLIGENT
        assert config.scheduler.enable_rate_limiting is True
        assert config.scheduler.max_concurrent_executions == 100
        assert config.scheduler.max_queue_size == 5000
        assert config.scheduler.metrics_enabled is True
        assert config.circuit_breaker is not None

    def test_create_production_config_custom_redis(self):
        """Test production config with custom Redis URL."""
        config = create_production_config(
            redis_url="redis://prod-server:6379",
            redis_key_prefix="myapp:prod:",
        )

        assert config.backend.redis is not None
        assert config.backend.redis.redis_url == "redis://prod-server:6379"
        assert config.backend.redis.key_prefix == "myapp:prod:"

    def test_create_production_config_custom_concurrency(self):
        """Test production config with custom concurrency."""
        config = create_production_config(
            redis_url="redis://localhost:6379",
            max_concurrent_executions=200,
            max_queue_size=10000,
            _allow_localhost_for_testing=True,
        )

        assert config.scheduler.max_concurrent_executions == 200
        assert config.scheduler.max_queue_size == 10000

    def test_create_production_config_metrics_disabled(self):
        """Test production config with metrics disabled."""
        config = create_production_config(
            redis_url="redis://localhost:6379",
            enable_metrics=False,
            _allow_localhost_for_testing=True,
        )

        assert config.scheduler.metrics_enabled is False
        assert config.scheduler.enable_performance_tracking is False

    def test_create_production_config_high_throughput(self):
        """Test high-throughput production config."""
        config = create_production_config_high_throughput(
            redis_url="redis://localhost:6379",
            _allow_localhost_for_testing=True,
        )

        assert config.environment == "production"
        assert config.backend.backend_type == BackendType.REDIS
        assert config.scheduler.max_concurrent_executions == 300
        assert config.scheduler.max_queue_size == 10000
        assert config.http_client.max_connections == 500
        assert config.scheduler.rate_limit_buffer_ratio == 0.95

    def test_create_production_config_high_throughput_custom_redis(self):
        """Test high-throughput config with custom Redis."""
        config = create_production_config_high_throughput(
            redis_url="redis://fast-redis:6379",
            redis_key_prefix="fast:",
        )

        assert config.backend.redis is not None
        assert config.backend.redis.redis_url == "redis://fast-redis:6379"
        assert config.backend.redis.key_prefix == "fast:"

    def test_create_production_config_conservative(self):
        """Test conservative production config."""
        config = create_production_config_conservative(
            redis_url="redis://localhost:6379",
            _allow_localhost_for_testing=True,
        )

        assert config.environment == "production"
        assert config.scheduler.max_concurrent_executions == 50
        assert config.scheduler.rate_limit_buffer_ratio == 0.8
        assert config.http_client.timeout == 60.0
        assert config.http_client.max_retries == 5
        assert config.backend.redis is not None
        assert config.backend.redis.max_retries == 5

    def test_create_production_config_conservative_custom_redis(self):
        """Test conservative config with custom Redis."""
        config = create_production_config_conservative(
            redis_url="redis://reliable:6379",
            redis_key_prefix="safe:",
        )

        assert config.backend.redis is not None
        assert config.backend.redis.redis_url == "redis://reliable:6379"
        assert config.backend.redis.key_prefix == "safe:"


class TestTestingPresets:
    """Test testing configuration presets."""

    def test_create_testing_config_defaults(self):
        """Test default testing config."""
        config = create_testing_config()

        assert config.environment == "test"
        assert config.debug is False
        assert config.backend.backend_type == BackendType.MEMORY
        assert config.scheduler.mode == SchedulerMode.BASIC
        assert config.scheduler.enable_rate_limiting is True
        assert config.scheduler.test_rate_multiplier == 10.0
        assert config.scheduler.metrics_enabled is False
        assert config.circuit_breaker.failure_threshold == 999  # Effectively disabled

    def test_create_testing_config_custom_multiplier(self):
        """Test testing config with custom rate multiplier."""
        config = create_testing_config(test_rate_multiplier=5.0)

        assert config.scheduler.test_rate_multiplier == 5.0

    def test_create_testing_config_circuit_breaker_enabled(self):
        """Test testing config with circuit breaker enabled."""
        config = create_testing_config(enable_circuit_breaker=True)

        assert config.circuit_breaker.failure_threshold == 5
        assert config.circuit_breaker.reset_timeout == 1.0

    def test_create_testing_config_circuit_breaker_disabled(self):
        """Test testing config with circuit breaker disabled (default)."""
        config = create_testing_config(enable_circuit_breaker=False)

        assert config.circuit_breaker.failure_threshold == 999

    def test_create_testing_config_with_intelligent_scheduler(self):
        """Test testing config with intelligent scheduler."""
        config = create_testing_config_with_intelligent_scheduler()

        assert config.environment == "test"
        assert config.scheduler.mode == SchedulerMode.INTELLIGENT
        assert config.scheduler.enable_rate_limiting is True
        assert config.scheduler.metrics_enabled is True
        assert config.scheduler.test_rate_multiplier == 10.0
        assert config.circuit_breaker.failure_threshold == 999

    def test_create_testing_config_with_intelligent_scheduler_custom_multiplier(self):
        """Test intelligent scheduler config with custom multiplier."""
        config = create_testing_config_with_intelligent_scheduler(test_rate_multiplier=20.0)

        assert config.scheduler.test_rate_multiplier == 20.0
        assert config.scheduler.mode == SchedulerMode.INTELLIGENT

    def test_create_testing_config_for_circuit_breaker(self):
        """Test circuit breaker testing config."""
        config = create_testing_config_for_circuit_breaker()

        assert config.environment == "test"
        assert config.debug is True
        assert config.circuit_breaker.failure_threshold == 5
        assert config.circuit_breaker.reset_timeout == 5.0
        assert config.scheduler.enable_rate_limiting is False
        assert config.http_client.max_retries == 0

    def test_create_testing_config_for_circuit_breaker_custom_params(self):
        """Test circuit breaker config with custom parameters."""
        config = create_testing_config_for_circuit_breaker(
            failure_threshold=3,
            reset_timeout=10.0,
        )

        assert config.circuit_breaker.failure_threshold == 3
        assert config.circuit_breaker.reset_timeout == 10.0


class TestPresetConsistency:
    """Test consistency across different presets."""

    def test_all_production_presets_use_redis(self):
        """Verify all production presets use Redis backend."""
        configs = [
            create_production_config(
                redis_url="redis://localhost:6379", _allow_localhost_for_testing=True
            ),
            create_production_config_high_throughput(
                redis_url="redis://localhost:6379", _allow_localhost_for_testing=True
            ),
            create_production_config_conservative(
                redis_url="redis://localhost:6379", _allow_localhost_for_testing=True
            ),
        ]

        for config in configs:
            assert config.backend.backend_type == BackendType.REDIS
            assert config.backend.redis is not None
            assert config.environment == "production"
            assert config.debug is False

    def test_all_production_presets_enable_rate_limiting(self):
        """Verify all production presets enable rate limiting."""
        configs = [
            create_production_config(
                redis_url="redis://localhost:6379", _allow_localhost_for_testing=True
            ),
            create_production_config_high_throughput(
                redis_url="redis://localhost:6379", _allow_localhost_for_testing=True
            ),
            create_production_config_conservative(
                redis_url="redis://localhost:6379", _allow_localhost_for_testing=True
            ),
        ]

        for config in configs:
            assert config.scheduler.enable_rate_limiting is True
            assert config.scheduler.mode == SchedulerMode.INTELLIGENT

    def test_all_testing_presets_use_memory(self):
        """Verify all testing presets use memory backend."""
        configs = [
            create_testing_config(),
            create_testing_config_with_intelligent_scheduler(),
            create_testing_config_for_circuit_breaker(),
        ]

        for config in configs:
            assert config.backend.backend_type == BackendType.MEMORY
            assert config.environment == "test"

    def test_development_presets_use_memory(self):
        """Verify development presets use memory backend."""
        configs = [
            create_development_config(),
            create_development_config_with_rate_limiting(),
        ]

        for config in configs:
            assert config.backend.backend_type == BackendType.MEMORY
            assert config.environment == "development"


class TestPresetValidation:
    """Test that presets produce valid configurations."""

    def test_production_presets_are_valid(self):
        """Test all production presets pass validation."""
        from venice_ai.validation import validate_config

        configs = [
            create_production_config(
                redis_url="redis://localhost:6379", _allow_localhost_for_testing=True
            ),
            create_production_config_high_throughput(
                redis_url="redis://localhost:6379", _allow_localhost_for_testing=True
            ),
            create_production_config_conservative(
                redis_url="redis://localhost:6379", _allow_localhost_for_testing=True
            ),
        ]

        for config in configs:
            result = validate_config(config)
            # May have warnings/recommendations but should be valid
            assert result.is_valid

    def test_development_presets_are_valid(self):
        """Test development presets pass validation."""
        from venice_ai.validation import validate_config

        configs = [
            create_development_config(),
            create_development_config_with_rate_limiting(),
        ]

        for config in configs:
            result = validate_config(config)
            assert result.is_valid

    def test_testing_presets_are_valid(self):
        """Test testing presets pass validation."""
        from venice_ai.validation import validate_config

        configs = [
            create_testing_config(),
            create_testing_config_with_intelligent_scheduler(),
            create_testing_config_for_circuit_breaker(),
        ]

        for config in configs:
            result = validate_config(config)
            assert result.is_valid
