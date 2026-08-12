"""
Comprehensive tests for safe queue metadata tracking.

This test suite verifies the thread-safe priority tracking system that eliminates
the need for unsafe queue peeking operations. It ensures that:
1. Metadata is updated correctly during enqueue/dequeue
2. Priority calculations are accurate
3. Thread-safety is maintained
4. No internal queue access (_queue) is used
5. Empty queues are handled safely
"""

import asyncio
from unittest.mock import Mock

import pytest

from venice_ai._queue_types import (
    QueueInfo,
    RateLimitConfig,
    ResourceType,
)


class TestQueueInfoMetadata:
    """Test QueueInfo metadata tracking functionality."""

    def create_queue_info(self):
        """Create a QueueInfo instance for testing."""
        queue = asyncio.PriorityQueue()
        config = RateLimitConfig(
            model_id="test-model",
            resource_type=ResourceType.LLM,
            rpm_limit=100,
        )
        return QueueInfo(
            queue_key="test-queue",
            model_id="test-model",
            resource_type=ResourceType.LLM,
            queue=queue,
            rate_config=config,
        )

    @pytest.fixture
    def queue_info(self):
        """Create a QueueInfo instance for testing."""
        return self.create_queue_info()

    @pytest.mark.asyncio
    async def test_metadata_updates_on_enqueue(self, queue_info):
        """Test that metadata is correctly updated when items are enqueued."""
        # Enqueue items with different priorities
        priorities = [5.0, 10.0, 3.0, 8.0]

        for priority in priorities:
            await queue_info.update_on_enqueue(priority)

        # Verify current priority is the last enqueued
        assert queue_info.current_priority == 8.0

        # Verify total enqueued count
        assert queue_info.total_enqueued == 4

        # Verify priority sum
        assert queue_info.priority_sum == sum(priorities)

        # Verify average priority
        expected_avg = sum(priorities) / len(priorities)
        assert queue_info.avg_priority == expected_avg

        # Verify max/min tracking
        assert queue_info.max_priority_seen == 10.0
        assert queue_info.min_priority_seen == 3.0

        # Verify enqueue time was updated
        assert queue_info.last_enqueue_time is not None

    @pytest.mark.asyncio
    async def test_metadata_updates_on_dequeue(self, queue_info):
        """Test that metadata is correctly updated when items are dequeued."""
        # Enqueue some items first
        for priority in [5.0, 10.0, 3.0]:
            await queue_info.update_on_enqueue(priority)

        # Dequeue items
        await queue_info.update_on_dequeue()
        await queue_info.update_on_dequeue()

        # Verify dequeue count
        assert queue_info.total_dequeued == 2

        # Verify dequeue time was updated
        assert queue_info.last_dequeue_time is not None

        # Verify items_pending property
        assert queue_info.items_pending == 1  # 3 enqueued - 2 dequeued

    @pytest.mark.asyncio
    async def test_average_priority_calculation(self, queue_info):
        """Test that average priority is calculated correctly."""
        # Test with no items
        assert queue_info.avg_priority == 0.0

        # Test with single item
        await queue_info.update_on_enqueue(7.5)
        assert queue_info.avg_priority == 7.5

        # Test with multiple items
        await queue_info.update_on_enqueue(2.5)
        await queue_info.update_on_enqueue(5.0)
        assert queue_info.avg_priority == 5.0  # (7.5 + 2.5 + 5.0) / 3

    @pytest.mark.asyncio
    async def test_priority_statistics(self, queue_info):
        """Test that priority statistics (min/max) are tracked correctly."""
        # Enqueue items with various priorities
        priorities = [10.0, 3.0, 7.0, 1.0, 15.0, 5.0]

        for priority in priorities:
            await queue_info.update_on_enqueue(priority)

        # Verify max and min
        assert queue_info.max_priority_seen == 15.0
        assert queue_info.min_priority_seen == 1.0

    @pytest.mark.asyncio
    async def test_get_priority_for_scheduling(self, queue_info):
        """Test get_priority_for_scheduling method."""
        # Enqueue items
        await queue_info.update_on_enqueue(8.0)
        await queue_info.update_on_enqueue(12.0)
        await queue_info.update_on_enqueue(4.0)

        # Verify it returns average priority
        expected = (8.0 + 12.0 + 4.0) / 3
        assert queue_info.get_priority_for_scheduling() == expected

    @pytest.mark.asyncio
    async def test_safe_queue_properties(self):
        """Test safe queue access properties (is_empty, current_size)."""
        queue_info = self.create_queue_info()

        # Test empty queue
        assert queue_info.is_empty is True
        assert queue_info.current_size == 0

        # Add items to queue (simulating actual queue operations)
        await queue_info.queue.put((1, "item1"))
        await queue_info.queue.put((2, "item2"))

        # Test non-empty queue
        assert queue_info.is_empty is False
        assert queue_info.current_size == 2

    @pytest.mark.asyncio
    async def test_items_pending_property(self, queue_info):
        """Test items_pending property calculation."""
        assert queue_info.items_pending == 0

        # Enqueue 5 items
        for i in range(5):
            await queue_info.update_on_enqueue(float(i))
        assert queue_info.items_pending == 5

        # Dequeue 2 items
        for _ in range(2):
            await queue_info.update_on_dequeue()
        assert queue_info.items_pending == 3

    @pytest.mark.asyncio
    async def test_concurrent_metadata_access(self):
        """Test that metadata updates are safe under concurrent access."""
        queue_info = self.create_queue_info()

        # Simulate concurrent enqueue operations
        async def enqueue_items(start_priority, count):
            for i in range(count):
                await queue_info.update_on_enqueue(start_priority + i)
                await asyncio.sleep(0.001)  # Small delay to encourage interleaving

        # Run multiple concurrent enqueue tasks
        tasks = [
            enqueue_items(0.0, 10),
            enqueue_items(10.0, 10),
            enqueue_items(20.0, 10),
        ]
        await asyncio.gather(*tasks)

        # Verify total count is correct
        assert queue_info.total_enqueued == 30

        # Verify priority sum is correct
        expected_sum = sum(range(30))
        assert queue_info.priority_sum == expected_sum

    @pytest.mark.asyncio
    async def test_empty_queue_safety(self, queue_info):
        """Test that empty queue operations don't cause errors."""
        # These should not raise errors on empty queue
        assert queue_info.avg_priority == 0.0
        assert queue_info.get_priority_for_scheduling() == 0.0
        assert queue_info.is_empty is True
        assert queue_info.current_size == 0
        assert queue_info.items_pending == 0

        # Dequeue on empty shouldn't cause calculation errors
        await queue_info.update_on_dequeue()
        assert queue_info.total_dequeued == 1
        assert queue_info.items_pending == -1  # Can go negative

    def test_no_queue_internal_access(self, queue_info):
        """Verify that QueueInfo doesn't access queue._queue attribute."""
        # The is_empty and current_size properties should use safe methods
        # This test verifies they work even if _queue is not accessible

        # Mock the queue to not have _queue attribute
        mock_queue = Mock(spec=["empty", "qsize", "put", "get"])
        mock_queue.empty.return_value = False
        mock_queue.qsize.return_value = 5

        queue_info.queue = mock_queue

        # These should still work without accessing _queue
        assert queue_info.is_empty is False
        assert queue_info.current_size == 5

        # Verify the safe methods were called
        mock_queue.empty.assert_called_once()
        mock_queue.qsize.assert_called_once()


class TestMetadataIntegration:
    """Integration tests for metadata tracking in mode strategies."""

    @pytest.mark.asyncio
    async def test_enqueue_dequeue_integration(self):
        """Test that enqueue/dequeue updates metadata correctly in integration."""
        from collections import deque

        # Simulate the mode strategy pattern
        queue = deque()
        priority_queue = asyncio.PriorityQueue()
        config = RateLimitConfig(
            model_id="test-model",
            resource_type=ResourceType.LLM,
            rpm_limit=100,
        )
        queue_info = QueueInfo(
            queue_key="test-queue",
            model_id="test-model",
            resource_type=ResourceType.LLM,
            queue=priority_queue,
            rate_config=config,
        )

        # Simulate enqueue with metadata update
        priorities = [5.0, 10.0, 3.0]
        for priority in priorities:
            queue.append(f"item-{priority}")
            await queue_info.update_on_enqueue(priority)

        # Verify metadata after enqueue
        assert queue_info.total_enqueued == 3
        assert queue_info.avg_priority == 6.0  # (5+10+3)/3

        # Simulate dequeue with metadata update
        queue.popleft()
        await queue_info.update_on_dequeue()

        # Verify metadata after dequeue
        assert queue_info.total_dequeued == 1
        assert queue_info.items_pending == 2

        # Average priority should remain the same (based on all enqueued)
        assert queue_info.avg_priority == 6.0
