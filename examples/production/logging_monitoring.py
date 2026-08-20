"""
Venice AI SDK - Production Logging and Monitoring

This example demonstrates production-ready logging and monitoring patterns
for the Venice AI SDK, including:

1. Structured logging setup
2. Request/response logging
3. Error tracking and alerting
4. Performance monitoring
5. Custom log formatters
6. Integration patterns with external monitoring systems

Requirements:
    pip install venice-py
    export VENICE_API_KEY="your-api-key"

Optional monitoring tools (shown conceptually):
    pip install sentry-sdk  # For error tracking
    pip install python-json-logger  # For structured JSON logs
"""

import asyncio
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from venice_ai import VeniceClient
from venice_ai.core.config import VeniceAIConfig
from venice_ai.exceptions import VeniceError
from venice_ai.factory import VeniceClientFactory
from venice_ai.types.api.requests import UserMessage

# Resolve results dir relative to this file's location so log files land under
# examples/results/ instead of polluting whatever directory the example is run
# from.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Custom Log Formatters
# =============================================================================


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with context."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured context."""
        # Base log data
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context from record (using getattr for custom attributes)
        if hasattr(record, "request_id"):
            log_data["request_id"] = getattr(record, "request_id", None)
        if hasattr(record, "model"):
            log_data["model"] = getattr(record, "model", None)
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = getattr(record, "duration_ms", None)
        if hasattr(record, "status_code"):
            log_data["status_code"] = getattr(record, "status_code", None)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Format as key=value pairs for easy parsing
        parts = [f"{k}={v}" for k, v in log_data.items()]
        return " ".join(parts)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for terminal output (development)."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for terminal."""
        original_levelname = record.levelname
        color = self.COLORS.get(original_levelname, self.RESET)
        record.levelname = f"{color}{original_levelname}{self.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


# =============================================================================
# Logging Setup Functions
# =============================================================================


def setup_development_logging() -> logging.Logger:
    """
    Setup logging for development environment.

    Features:
    - Colored output for terminal
    - DEBUG level logging
    - Detailed formatting
    """
    logger = logging.getLogger("venice_ai")
    logger.setLevel(logging.DEBUG)

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


def setup_production_logging(log_file: str | None = None) -> logging.Logger:
    """
    Setup logging for production environment.

    Features:
    - Structured logging to file
    - INFO level (less verbose)
    - JSON-like formatting for log aggregation
    - Separate error log file

    Args:
        log_file: Path to main log file. Defaults to ``examples/results/venice_ai.log``
            so the example never writes into the caller's current directory.
    """
    main_path = Path(log_file) if log_file is not None else RESULTS_DIR / "venice_ai.log"
    error_path = RESULTS_DIR / "venice_ai_errors.log"

    logger = logging.getLogger("venice_ai")
    logger.setLevel(logging.INFO)

    # Main log file handler (all logs)
    file_handler = logging.FileHandler(main_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(StructuredFormatter())

    # Error log file handler (errors only)
    error_handler = logging.FileHandler(error_path)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(StructuredFormatter())

    # Console handler for critical issues
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger


# =============================================================================
# Request Logging Wrapper
# =============================================================================


class RequestLogger:
    """Wrapper for logging Venice AI requests with context."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._request_counter = 0

    def _get_request_id(self) -> str:
        """Generate unique request ID."""
        self._request_counter += 1
        timestamp = int(time.time() * 1000)
        return f"req_{timestamp}_{self._request_counter}"

    async def log_chat_request(
        self,
        client: VeniceClient,
        model: str,
        messages: list,
        **kwargs: Any,
    ) -> Any:
        """
        Log a chat completion request with timing and response metadata.

        Args:
            client: Venice AI client
            model: Model to use
            messages: Chat messages
            **kwargs: Additional chat completion parameters

        Returns:
            Chat completion response
        """
        request_id = self._get_request_id()
        start_time = time.time()

        # Log request start
        self.logger.info(
            "Starting chat request",
            extra={
                "request_id": request_id,
                "model": model,
                "message_count": len(messages),
            },
        )

        try:
            # Make request
            response = await client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )

            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Surface Venice production signals carried on response headers.
            # balance_info exposes remaining diem/usd credit; response_rate_limits
            # exposes the request/token budget windows. Both are None when the
            # corresponding headers are absent (e.g. recorded fixtures), so guard.
            balance = response.balance_info
            rate_limits = response.response_rate_limits

            # Log success
            self.logger.info(
                "Chat request completed",
                extra={
                    "request_id": request_id,
                    "model": model,
                    "duration_ms": duration_ms,
                    "finish_reason": response.choices[0].finish_reason,
                    "tokens_used": (response.usage.total_tokens if response.usage else None),
                    "balance_diem": balance.diem if balance else None,
                    "balance_usd": balance.usd if balance else None,
                    "remaining_requests": (rate_limits.remaining_requests if rate_limits else None),
                    "remaining_tokens": (rate_limits.remaining_tokens if rate_limits else None),
                },
            )

            return response

        except VeniceError as e:
            # Calculate duration even on error
            duration_ms = int((time.time() - start_time) * 1000)

            # Log error with full context
            self.logger.error(
                f"Chat request failed: {str(e)}",
                extra={
                    "request_id": request_id,
                    "model": model,
                    "duration_ms": duration_ms,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise


# =============================================================================
# Performance Monitoring
# =============================================================================


class PerformanceMonitor:
    """Monitor and log performance metrics."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics: dict[str, Any] = {
            "request_durations": [],
            "token_usage": [],
            "error_counts": {},
        }

    def record_request(
        self, duration_ms: int, tokens: int | None = None, error: str | None = None
    ) -> None:
        """Record metrics for a request."""
        self.metrics["request_durations"].append(duration_ms)

        if tokens:
            self.metrics["token_usage"].append(tokens)

        if error:
            error_type = error.__class__.__name__ if hasattr(error, "__class__") else str(error)
            self.metrics["error_counts"][error_type] = (
                self.metrics["error_counts"].get(error_type, 0) + 1
            )

    def log_summary(self) -> None:
        """Log summary statistics."""
        if not self.metrics["request_durations"]:
            self.logger.info("No requests recorded")
            return

        durations = self.metrics["request_durations"]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)

        self.logger.info(
            "Performance summary",
            extra={
                "total_requests": len(durations),
                "avg_duration_ms": int(avg_duration),
                "max_duration_ms": max_duration,
                "min_duration_ms": min_duration,
            },
        )

        if self.metrics["token_usage"]:
            tokens = self.metrics["token_usage"]
            self.logger.info(
                "Token usage summary",
                extra={
                    "total_tokens": sum(tokens),
                    "avg_tokens": int(sum(tokens) / len(tokens)),
                },
            )

        if self.metrics["error_counts"]:
            self.logger.warning(
                "Error summary",
                extra={"errors": self.metrics["error_counts"]},
            )


# =============================================================================
# Error Tracking Integration (Conceptual)
# =============================================================================


class ErrorTracker:
    """
    Conceptual integration with error tracking services (e.g., Sentry).

    In production, you would integrate with actual services:
    - Sentry: sentry_sdk.capture_exception()
    - Datadog: statsd.increment('venice.errors')
    - CloudWatch: cloudwatch.put_metric_data()
    """

    def __init__(self, logger: logging.Logger, enable_external: bool = False):
        self.logger = logger
        self.enable_external = enable_external

    async def track_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Track an error with context.

        Args:
            error: The exception that occurred
            context: Additional context (model, request_id, etc.)
        """
        # Log locally
        self.logger.error(
            f"Error tracked: {str(error)}",
            extra=context or {},
            exc_info=True,
        )

        # Send to external service (if enabled)
        if self.enable_external:
            # Example Sentry integration:
            # import sentry_sdk
            # with sentry_sdk.push_scope() as scope:
            #     for key, value in (context or {}).items():
            #         scope.set_extra(key, value)
            #     sentry_sdk.capture_exception(error)

            self.logger.debug("Error sent to external tracking service")


# =============================================================================
# Example Usage
# =============================================================================


async def example_development_logging() -> bool:
    """Example: Development logging setup."""
    print("=" * 60)
    print("Development Logging Example")
    print("=" * 60)

    # Setup development logging
    logger = setup_development_logging()

    # Create client with debug logging enabled
    from venice_ai.core.config import SchedulerConfig, SchedulerMode

    config = VeniceAIConfig(
        debug=True,
        scheduler=SchedulerConfig(mode=SchedulerMode.BASIC),
    )

    client = VeniceClientFactory.create_client(config=config)

    try:
        # Create request logger
        request_logger = RequestLogger(logger)

        # Make request with logging
        model = await client.models.resolve_chat()
        response = await request_logger.log_chat_request(
            client=client,
            model=model,
            messages=[UserMessage(content="Say 'Hello from Venice AI!'")],
            max_completion_tokens=50,
        )

        print(f"\n✅ Response: {response.text}")
        return bool(response.text)

    finally:
        await client.close()


async def example_production_logging() -> bool:
    """Example: Production logging with monitoring."""
    print("\n" + "=" * 60)
    print("Production Logging Example")
    print("=" * 60)

    main_log = RESULTS_DIR / "venice_ai.log"
    error_log = RESULTS_DIR / "venice_ai_errors.log"

    # Setup production logging (writes under examples/results/, never the cwd)
    logger = setup_production_logging(str(main_log))

    # Create client with production config
    from venice_ai.core.config import SchedulerConfig, SchedulerMode

    config = VeniceAIConfig(
        scheduler=SchedulerConfig(mode=SchedulerMode.BASIC),
    )

    client = VeniceClientFactory.create_client(config=config)

    succeeded = 0
    try:
        # Initialize monitoring
        request_logger = RequestLogger(logger)
        perf_monitor = PerformanceMonitor(logger)
        error_tracker = ErrorTracker(logger, enable_external=False)

        # Make multiple requests against a dynamically resolved model
        model = await client.models.resolve_chat()
        for i in range(2):
            try:
                start = time.time()

                response = await request_logger.log_chat_request(
                    client=client,
                    model=model,
                    messages=[UserMessage(content=f"Request #{i + 1}: Quick response")],
                    max_completion_tokens=20,
                )

                duration_ms = int((time.time() - start) * 1000)
                tokens = response.usage.total_tokens if response.usage else None

                # Record metrics
                perf_monitor.record_request(duration_ms, tokens)

                # Surface the Venice header-derived production signals.
                balance = response.balance_info
                rate_limits = response.response_rate_limits
                balance_str = (
                    f"{balance.diem} diem / ${balance.usd} usd"
                    if balance
                    else "n/a (no balance headers)"
                )
                rl_str = (
                    f"{rate_limits.remaining_requests} req / "
                    f"{rate_limits.remaining_tokens} tok remaining"
                    if rate_limits
                    else "n/a (no rate-limit headers)"
                )
                succeeded += 1
                print(f"✅ Request {i + 1} completed in {duration_ms}ms")
                print(f"   balance: {balance_str}")
                print(f"   rate limits: {rl_str}")

            except VeniceError as e:
                # Track error
                await error_tracker.track_error(
                    e,
                    context={
                        "model": model,
                        "request_number": i + 1,
                    },
                )
                perf_monitor.record_request(0, error=str(e))

        # Log performance summary
        perf_monitor.log_summary()

        print("\n✅ Logs written to:")
        print(f"   - {main_log} (all logs)")
        print(f"   - {error_log} (errors only)")

    finally:
        await client.close()

    return succeeded > 0


async def example_structured_logging():
    """Example: Structured logging for log aggregation."""
    print("\n" + "=" * 60)
    print("Structured Logging Example")
    print("=" * 60)

    # Setup with structured formatter
    logger = logging.getLogger("venice_ai.structured")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)

    # Demonstrate structured logging
    logger.info(
        "Client initialized",
        extra={
            "environment": "production",
            "api_version": "v1",
            "features": ["rate_limiting", "retry"],
        },
    )

    logger.info(
        "Request completed",
        extra={
            "request_id": "req_123456",
            "model": "llama-3.3-70b",
            "duration_ms": 1234,
            "tokens_used": 150,
        },
    )

    print("\n💡 Structured logs are easily parsed by log aggregation tools")
    print("   (e.g., ELK Stack, Splunk, CloudWatch Logs Insights)")


async def example_monitoring_best_practices():
    """Display monitoring best practices."""
    print("\n" + "=" * 60)
    print("Monitoring Best Practices")
    print("=" * 60)

    practices = [
        (
            "Log Levels",
            [
                "DEBUG: Development only (very verbose)",
                "INFO: Normal operations (requests, responses)",
                "WARNING: Unusual but handled situations",
                "ERROR: Errors that need attention",
                "CRITICAL: System-critical failures",
            ],
        ),
        (
            "What to Log",
            [
                "✅ Request/response metadata (model, tokens, duration)",
                "✅ Errors with full context and stack traces",
                "✅ Performance metrics (latency, throughput)",
                "✅ Rate limit status and warnings",
                "❌ Never log API keys or sensitive data",
                "❌ Don't log full request/response content in production",
            ],
        ),
        (
            "Log Rotation",
            [
                "Use rotating file handlers for disk space management",
                "Example: RotatingFileHandler(maxBytes=10MB, backupCount=5)",
                "Or use TimedRotatingFileHandler for daily rotation",
            ],
        ),
        (
            "External Services",
            [
                "Sentry/Rollbar: Real-time error tracking with alerting",
                "Datadog/New Relic: Application performance monitoring",
                "ELK Stack: Centralized log aggregation and search",
                "CloudWatch/Stackdriver: Cloud-native monitoring",
            ],
        ),
        (
            "Alerting Rules",
            [
                "Error rate threshold: >1% errors in 5 minutes",
                "Latency threshold: p95 > 2 seconds",
                "Rate limit warnings: >80% capacity used",
                "Circuit breaker trips: Immediate alert",
            ],
        ),
    ]

    for category, items in practices:
        print(f"\n📋 {category}:")
        for item in items:
            print(f"   {item}")


async def main() -> int:
    """Run all logging examples."""
    print("=" * 60)
    print("Venice AI SDK - Production Logging & Monitoring")
    print("=" * 60)

    # Run examples. The two live examples are tracked honestly so the process
    # exit code reflects whether the Venice calls actually succeeded; the
    # structured-logging and best-practices demos are informational only.
    results: list[tuple[str, bool]] = []
    results.append(("Development Logging", await example_development_logging()))
    results.append(("Production Logging", await example_production_logging()))
    await example_structured_logging()
    await example_monitoring_best_practices()

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    if failed == 0:
        print(f"✅ All {passed}/{len(results)} live logging examples succeeded!")
    else:
        print(f"⚠️ {passed}/{len(results)} live examples succeeded; {failed} failed")
        for name, ok in results:
            status = "✓" if ok else "✗"
            print(f"   {status} {name}")
    print("=" * 60)

    print("\n🔑 Key Takeaways:")
    print("   1. Use DEBUG level in development, INFO in production")
    print("   2. Implement structured logging for better searchability")
    print("   3. Monitor performance metrics (latency, tokens, errors)")
    print("   4. Integrate with external monitoring services")
    print("   5. Set up alerts for critical issues")
    print("   6. Never log sensitive data (API keys, PII)")
    print("   7. Use log rotation to manage disk space")
    print("   8. Include request context in all logs")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(exit_code)
