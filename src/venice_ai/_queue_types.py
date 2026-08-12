"""
Queue-Based Rate Limiting Core Types
===================================

This module defines the core data structures and types for Venice AI's intelligent
queue-based rate limiting system. It provides the foundational classes and enums
used throughout the rate limiting and scheduling components.

The queue system implements sophisticated rate limiting that goes beyond simple
request throttling, providing intelligent queuing, request classification, and
adaptive scheduling based on model capabilities and current load conditions.

Core Type Categories:
    * **Resource Types**: Classification of different API resource types
    * **Rate Limit Types**: Different kinds of rate limits (RPM, RPD, TPM)
    * **Request Metadata**: Rich metadata for request classification and routing
    * **Queue Data Structures**: Core queue and scheduling data structures
    * **Exception Types**: Queue-specific error conditions

Key Components:
    * **ResourceType**: Enum for API resource classification
    * **RateLimitType**: Types of rate limits enforced by the API
    * **RequestMetadata**: Comprehensive request classification data
    * **FailedRequestCounter**: Failed request tracking for rate limiting

Design Principles:
    * **Type Safety**: Strong typing for all queue operations
    * **Extensibility**: Easy to add new resource types and limits
    * **Performance**: Lightweight data structures for high throughput
    * **Debugging**: Rich metadata for troubleshooting and monitoring

Example:
    >>> from venice_ai._queue_types import ResourceType, RequestMetadata
    >>>
    >>> # Create request metadata for classification
    >>> metadata = RequestMetadata(
    ...     request_id="req_123",
    ...     model_id="llama-3.3-70b",
    ...     resource_type=ResourceType.LLM,
    ...     estimated_tokens=150,
    ...     priority=0
    ... )
    >>>
    >>> print(f"Request for {metadata.resource_type.value} resource")
    >>> print(f"Estimated cost: {metadata.estimated_tokens} tokens")
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class ResourceType(Enum):
    """
    Classification of Venice AI API resource types with distinct rate limiting behaviors.

    Each resource type represents a different category of API operation with its own
    rate limiting characteristics, pricing structure, and performance requirements.
    This classification enables the intelligent scheduler to apply appropriate
    queuing and rate limiting strategies.

    Resource Categories:
        * **LLM**: Large Language Model operations (chat, completions)
        * **IMAGE**: Image generation and manipulation
        * **AUDIO**: Audio processing (TTS, transcription, translation)
        * **EMBEDDING**: Text embedding generation
        * **API_MANAGEMENT**: API key and account management operations
        * **BILLING**: Billing and usage information queries
        * **CHARACTERS**: Character-specific model operations
        * **VIDEO**: Async video generation (queue/retrieve/complete)
        * **MUSIC**: Async music generation via the /audio/* queue family

    Rate Limiting Implications:
        * Each resource type may have different RPM/RPD limits
        * Token-based limits (TPM) typically apply only to LLM resources
        * Management operations often have lower limits but higher priority
    """

    LLM = "llm"
    IMAGE = "image"
    AUDIO = "audio"
    EMBEDDING = "embedding"
    API_MANAGEMENT = "api_management"
    BILLING = "billing"
    CHARACTERS = "characters"
    VIDEO = "video"
    MUSIC = "music"


class RateLimitType(Enum):
    """
    Types of rate limits enforced by the Venice AI API.

    The Venice AI API implements multiple types of rate limiting to ensure fair
    usage and system stability. Each limit type has different characteristics
    and reset behaviors that affect queuing and scheduling decisions.

    Limit Types:
        * **RPM**: Requests Per Minute - Short-term burst protection
        * **RPD**: Requests Per Day - Long-term usage quotas
        * **TPM**: Tokens Per Minute - Content-based limiting for LLM operations

    Reset Behaviors:
        * RPM limits reset every minute (sliding or fixed window)
        * RPD limits reset daily (typically at UTC midnight)
        * TPM limits reset every minute and are model-specific
    """

    RPM = "RPM"  # Requests Per Minute
    RPD = "RPD"  # Requests Per Day
    TPM = "TPM"  # Tokens Per Minute


@dataclass
class RequestMetadata:
    """
    Comprehensive metadata for request classification, routing, and queue management.

    This dataclass contains all the information needed to classify, prioritize, route,
    and track requests through the intelligent queue system. It serves as the primary
    data structure for request lifecycle management.

    The metadata enables sophisticated request handling including:
    * Resource type classification for appropriate queue selection
    * Token estimation for rate limit calculations
    * Priority-based scheduling and queue ordering
    * Timeout handling and request lifecycle management
    * Client-specific tracking and debugging

    Attributes:
        request_id: Unique identifier for this request instance
        model_id: Venice AI model identifier for the target model
        resource_type: Classification of the API resource being accessed
        estimated_tokens: Estimated token consumption for LLM requests (None for non-LLM)
        priority: Request priority level (higher numbers = higher priority)
        submitted_at: UTC timestamp when the request was submitted to the queue
        timeout: Maximum time to wait for request completion (seconds)
        client_id: Optional client identifier for multi-tenant scenarios
        endpoint: API endpoint path for debugging and metrics
        requires_model: Whether this request requires a specific model (vs. generic endpoint)

    Priority Levels:
        * 0: Normal priority (default)
        * 1-9: Higher priority (for important operations)
        * Negative: Lower priority (for background operations)
    """

    request_id: str
    model_id: str
    resource_type: ResourceType
    estimated_tokens: int | None = None  # For LLM requests
    priority: int = 0  # Higher = more important
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    timeout: float | None = 60.0
    client_id: str | None = None
    endpoint: str | None = None
    requires_model: bool = True


@dataclass
class RateLimitConfig:
    """Configuration for a specific model/resource."""

    model_id: str
    resource_type: ResourceType
    rpm_limit: int
    rpd_limit: int | None = None
    tpm_limit: int | None = None  # Tokens per minute for LLMs


@dataclass
class QueueInfo:
    """
    Information about a queue for scheduling decisions with safe priority tracking.

    This dataclass contains metadata for queue management and scheduling, including
    thread-safe priority tracking that eliminates the need for unsafe queue peeking.

    Priority Tracking:
        Instead of accessing the internal _queue attribute (which causes race conditions),
        this class tracks priority information through metadata updates during enqueue
        and dequeue operations. This provides safe, accurate priority information for
        scheduling decisions without risking IndexError or corruption.

    Thread Safety:
        All priority tracking metadata updates are protected by an asyncio.Lock to ensure
        atomicity of compound operations (+=, etc.). This prevents race conditions under
        concurrent access from multiple coroutines.
    """

    queue_key: str
    model_id: str
    resource_type: ResourceType
    queue: "asyncio.PriorityQueue[Any]"
    rate_config: RateLimitConfig
    queue_depth: int = 0
    last_request_time: datetime | None = None

    # Priority tracking metadata (protected by _metadata_lock)
    current_priority: float = 0.0  # Most recent priority value enqueued
    priority_sum: float = 0.0  # Running sum for average calculation
    total_enqueued: int = 0  # Total items ever enqueued
    total_dequeued: int = 0  # Total items ever dequeued
    last_enqueue_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_dequeue_time: datetime | None = None

    # Priority statistics for monitoring and debugging
    max_priority_seen: float = float("-inf")
    min_priority_seen: float = float("inf")

    # Lock for protecting metadata updates
    _metadata_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @property
    def avg_priority(self) -> float:
        """
        Calculate average priority of items enqueued.

        This provides a representative priority value for the queue that can be used
        in scheduling decisions without accessing the queue's internal structure.

        Returns:
            float: Average priority of all items ever enqueued, or 0.0 if no items
        """
        return self.priority_sum / self.total_enqueued if self.total_enqueued > 0 else 0.0

    @property
    def current_size(self) -> int:
        """
        Safe access to queue size using built-in qsize().

        Uses the queue's thread-safe qsize() method instead of accessing internal state.

        Returns:
            int: Current number of items in the queue
        """
        return self.queue.qsize()

    @property
    def is_empty(self) -> bool:
        """
        Check if queue is empty safely.

        Uses the queue's thread-safe empty() method instead of checking length.

        Returns:
            bool: True if queue is empty, False otherwise
        """
        return self.queue.empty()

    @property
    def items_pending(self) -> int:
        """
        Number of items still in queue (enqueued - dequeued).

        This provides an alternative measure of queue depth based on the difference
        between total enqueued and dequeued items.

        Returns:
            int: Number of items currently pending in the queue
        """
        return self.total_enqueued - self.total_dequeued

    async def update_on_enqueue(self, priority: float) -> None:
        """
        Update metadata when item is enqueued.

        This method must be called whenever an item is added to the queue to maintain
        accurate priority tracking. It updates all relevant statistics atomically.

        Thread Safety:
            All compound operations (+=, max, min) are protected by an asyncio.Lock
            to prevent race conditions under concurrent access. This ensures atomic
            read-modify-write operations and prevents lost updates.

        Args:
            priority: The priority value of the item being enqueued (higher = more urgent)
        """
        async with self._metadata_lock:
            self.current_priority = priority
            self.priority_sum += priority
            self.total_enqueued += 1
            self.last_enqueue_time = datetime.now(UTC)
            self.max_priority_seen = max(self.max_priority_seen, priority)
            self.min_priority_seen = min(self.min_priority_seen, priority)

    async def update_on_dequeue(self) -> None:
        """
        Update metadata when item is dequeued.

        This method must be called whenever an item is removed from the queue to maintain
        accurate tracking of queue activity.

        Thread Safety:
            All compound operations (+=) are protected by an asyncio.Lock to prevent
            race conditions under concurrent access. This ensures atomic read-modify-write
            operations and prevents lost updates.
        """
        async with self._metadata_lock:
            self.total_dequeued += 1
            self.last_dequeue_time = datetime.now(UTC)

    def get_priority_for_scheduling(self) -> float:
        """
        Get priority value for scheduling decisions.

        This is the primary method that scheduling strategies should use to obtain
        priority information. It returns the average priority, which represents the
        typical urgency of items in this queue without requiring unsafe queue access.

        Returns:
            float: Average priority value for scheduling (higher = more urgent)
        """
        return self.avg_priority


@dataclass
class FailedRequestCounter:
    """Tracks failed requests for the 20/30s limit."""

    count: int = 0
    window_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    max_failures: int = 20  # Venice AI API limit for failed requests
    window_seconds: int = 30  # Venice AI API time window in seconds

    def increment(self) -> int:
        """Increment failed request count and return current count."""
        now = datetime.now(UTC)

        # Reset window if expired
        if (now - self.window_start).total_seconds() > self.window_seconds:
            self.count = 0
            self.window_start = now

        self.count += 1
        return self.count

    def is_limit_exceeded(self) -> bool:
        """Check if failed request limit is exceeded."""
        now = datetime.now(UTC)

        # Reset window if expired
        if (now - self.window_start).total_seconds() > self.window_seconds:
            self.count = 0
            self.window_start = now
            return False

        return self.count >= self.max_failures


class SchedulerStoppedError(Exception):
    """Raised when a request is rejected because the scheduler is stopping."""

    pass
