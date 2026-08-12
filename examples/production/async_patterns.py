"""
Venice AI SDK - Production Async Patterns

This example demonstrates production-ready asynchronous patterns for the Venice AI SDK:

1. Concurrent request handling
2. Async context managers
3. Task cancellation and cleanup
4. Error handling in async code
5. Streaming with async iteration
6. Connection pooling optimization

Requirements:
    pip install venice-ai
    export VENICE_API_KEY="your-api-key"
"""

import asyncio
import sys

from venice_ai import VeniceClient
from venice_ai.types.api.chat import ChatCompletionResponse
from venice_ai.types.api.requests import UserMessage

# =============================================================================
# Pattern 1: Concurrent Request Handling
# =============================================================================


async def example_concurrent_requests():
    """
    Demonstrate efficient concurrent API request handling.

    Best practices:
    - Use asyncio.gather() for concurrent requests
    - Handle individual task failures gracefully
    - Limit concurrency to avoid overwhelming the API
    """
    print("=" * 60)
    print("Pattern 1: Concurrent Request Handling")
    print("=" * 60)

    async with VeniceClient() as client:
        model = await client.models.resolve_chat()
        # Define multiple prompts to process concurrently
        prompts = [
            "What is Python?",
            "What is async/await?",
            "What is Venice AI?",
        ]

        # Create tasks for concurrent execution
        tasks = [
            client.chat.completions.create(
                model=model,
                messages=[UserMessage(content=prompt)],
                max_completion_tokens=50,
            )
            for prompt in prompts
        ]

        # Execute concurrently and gather results
        print(f"\n🚀 Executing {len(tasks)} requests concurrently...")
        responses: list[ChatCompletionResponse] = await asyncio.gather(*tasks)

        # Process results
        for i, response in enumerate(responses):
            print(f"\n✅ Response {i + 1}:")
            print(f"   Prompt: {prompts[i]}")
            content = response.text or ""
            print(f"   Answer: {content[:100]}...")
            if response.usage:
                print(f"   Tokens: {response.usage.total_tokens}")


# =============================================================================
# Pattern 2: Async Context Managers
# =============================================================================


async def example_context_managers():
    """
    Demonstrate proper resource management with async context managers.

    Best practices:
    - Always use async with for automatic cleanup
    - Ensure connections are properly closed
    - Handle exceptions within context
    """
    print("\n" + "=" * 60)
    print("Pattern 2: Async Context Managers")
    print("=" * 60)

    print("\n✅ Using async context manager for automatic cleanup:")

    # The client will automatically close when exiting the context
    async with VeniceClient() as client:
        model = await client.models.resolve_chat()
        response = await client.chat.completions.create(
            model=model,
            messages=[UserMessage(content="Hello!")],
            max_completion_tokens=20,
        )
        print(f"   Response: {response.text}")

    print("   ✓ Client automatically closed")

    # Manual resource management (not recommended)
    print("\n⚠️  Manual management (for comparison):")
    client = VeniceClient()
    try:
        model = await client.models.resolve_chat()
        response = await client.chat.completions.create(
            model=model,
            messages=[UserMessage(content="Hi!")],
            max_completion_tokens=20,
        )
        print(f"   Response: {response.text}")
    finally:
        await client.close()
        print("   ✓ Client manually closed")


# =============================================================================
# Pattern 3: Task Cancellation and Cleanup
# =============================================================================


async def example_task_cancellation():
    """
    Demonstrate proper task cancellation and cleanup.

    Best practices:
    - Use asyncio.timeout() for timeouts
    - Handle CancelledError appropriately
    - Clean up resources on cancellation
    """
    print("\n" + "=" * 60)
    print("Pattern 3: Task Cancellation & Cleanup")
    print("=" * 60)

    async with VeniceClient() as client:
        model = await client.models.resolve_chat()
        print("\n🔄 Testing timeout handling...")

        try:
            # Set a timeout that will likely succeed
            async with asyncio.timeout(30):
                response = await client.chat.completions.create(
                    model=model,
                    messages=[UserMessage(content="Quick response")],
                    max_completion_tokens=20,
                )
                print(f"✅ Completed: {response.text}")

        except TimeoutError:
            print("⏱️  Request timed out (would handle cleanup here)")
        except asyncio.CancelledError:
            print("🚫 Task was cancelled (would handle cleanup here)")
            raise


# =============================================================================
# Pattern 4: Error Handling in Async Code
# =============================================================================


async def example_async_error_handling():
    """
    Demonstrate comprehensive error handling in async code.

    Best practices:
    - Catch specific exceptions
    - Use try/except/finally for cleanup
    - Handle errors per-task in gather()
    """
    print("\n" + "=" * 60)
    print("Pattern 4: Async Error Handling")
    print("=" * 60)

    async with VeniceClient() as client:
        model = await client.models.resolve_chat()
        print("\n🛡️  Demonstrating error recovery:")

        # A deliberately bogus model id so one task genuinely fails — this is a
        # failure-trigger fixture, not a usage example, so a literal is correct
        # here (you cannot resolve_*() a model that does not exist).
        invalid_model = "venice-nonexistent-model"

        # Using return_exceptions=True so one failure doesn't abort the batch
        tasks = [
            client.chat.completions.create(
                model=model,
                messages=[UserMessage(content="Valid request")],
                max_completion_tokens=20,
            ),
            client.chat.completions.create(
                model=invalid_model,
                messages=[UserMessage(content="This call should fail")],
                max_completion_tokens=20,
            ),
        ]

        # Gather with error handling
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and errors
        successes = 0
        failures = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failures += 1
                print(f"❌ Task {i + 1} failed (recovered): {type(result).__name__}")
            elif hasattr(result, "choices"):
                successes += 1
                print(f"✅ Task {i + 1} succeeded")
                content = result.text or ""  # type: ignore[union-attr]
                print(f"   Response: {content[:50]}...")

        print(f"\n   📊 Handled {failures} failure(s), {successes} success(es)")


# =============================================================================
# Pattern 5: Streaming with Async Iteration
# =============================================================================


async def example_async_streaming():
    """
    Demonstrate async streaming with proper iteration.

    Best practices:
    - Use async for to iterate over streams
    - Handle stream interruption
    - Process chunks incrementally
    """
    print("\n" + "=" * 60)
    print("Pattern 5: Async Streaming")
    print("=" * 60)

    async with VeniceClient() as client:
        model = await client.models.resolve_chat()
        print("\n📡 Streaming response:")

        stream = await client.chat.completions.create(
            model=model,
            messages=[UserMessage(content="Count to 5")],
            max_completion_tokens=50,
            stream=True,
        )

        print("   ", end="", flush=True)
        async for chunk in stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()  # New line after streaming


# =============================================================================
# Pattern 6: Semaphore for Rate Limiting
# =============================================================================


async def example_semaphore_pattern():
    """
    Demonstrate ``client.gather()`` for bounded-concurrency request batching.

    Best practices:
    - Use ``client.gather(awaitables, max_concurrency=N)`` instead of
      hand-rolled ``asyncio.Semaphore`` + ``asyncio.gather()`` loops
    - It accepts awaitables across any modality (chat, image, embeddings…)
    - ``return_exceptions=True`` (default) keeps one failure from aborting the batch
    """
    print("\n" + "=" * 60)
    print("Pattern 6: Bounded-Concurrency with client.gather()")
    print("=" * 60)

    async with VeniceClient() as client:
        model = await client.models.resolve_chat()
        print("\n🚦 Executing 5 requests with max 3 concurrent:")

        responses = await client.gather(
            [
                client.chat.completions.create(
                    model=model,
                    messages=[UserMessage(content=f"Request {i + 1}")],
                    max_completion_tokens=20,
                )
                for i in range(5)
            ],
            max_concurrency=3,
        )
        print(f"\n✅ All {len(responses)} requests completed")


# =============================================================================
# Pattern 7: Background Tasks
# =============================================================================


async def example_background_tasks():
    """
    Demonstrate running background tasks alongside main work.

    Best practices:
    - Use asyncio.create_task() for background work
    - Track and cancel background tasks
    - Handle background task errors
    """
    print("\n" + "=" * 60)
    print("Pattern 7: Background Tasks")
    print("=" * 60)

    async def background_processor(client: VeniceClient):
        """Simulated background task."""
        for i in range(3):
            await asyncio.sleep(1)
            print(f"   📊 Background task tick {i + 1}")

    async with VeniceClient() as client:
        model = await client.models.resolve_chat()
        print("\n🔄 Starting background task...")

        # Create background task
        bg_task = asyncio.create_task(background_processor(client))

        # Main work
        print("   🚀 Executing main request...")
        response = await client.chat.completions.create(
            model=model,
            messages=[UserMessage(content="Hello")],
            max_completion_tokens=20,
        )
        content = response.text or ""
        print(f"   ✅ Main request done: {content[:30]}...")

        # Wait for background task
        await bg_task
        print("   ✅ Background task completed")


# =============================================================================
# Pattern 8: Retry with Exponential Backoff
# =============================================================================


async def example_retry_pattern():
    """
    Demonstrate retry pattern with exponential backoff.

    Best practices:
    - Use exponential backoff for retries
    - Set maximum retry attempts
    - Log retry attempts
    """
    print("\n" + "=" * 60)
    print("Pattern 8: Retry with Exponential Backoff")
    print("=" * 60)

    async def request_with_retry(
        client: VeniceClient,
        model: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> ChatCompletionResponse:
        """Make a request with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                print(f"   🔄 Attempt {attempt + 1}/{max_retries}")
                response = await client.chat.completions.create(
                    model=model,
                    messages=[UserMessage(content="Test")],
                    max_completion_tokens=20,
                )
                print(f"   ✅ Success on attempt {attempt + 1}")
                return response
            except Exception:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2**attempt)
                print(f"   ⚠️  Failed, retrying in {delay}s...")
                await asyncio.sleep(delay)

        raise RuntimeError("Max retries exceeded")

    async with VeniceClient() as client:
        model = await client.models.resolve_chat()
        print("\n🔄 Testing retry pattern:")
        response = await request_with_retry(client, model)
        content = response.text or ""
        print(f"   Final response: {content[:40]}...")


# =============================================================================
# Best Practices Summary
# =============================================================================


async def show_best_practices():
    """Display async programming best practices."""
    print("\n" + "=" * 60)
    print("Async Programming Best Practices")
    print("=" * 60)

    practices = [
        (
            "Resource Management",
            [
                "✅ Always use async with for VeniceClient",
                "✅ Ensure proper cleanup in finally blocks",
                "✅ Cancel tasks on shutdown",
            ],
        ),
        (
            "Concurrency",
            [
                "✅ Use asyncio.gather() for multiple requests",
                "✅ Bound concurrency with client.gather(max_concurrency=N)",
                "✅ Handle individual task failures",
            ],
        ),
        (
            "Error Handling",
            [
                "✅ Use return_exceptions=True in gather()",
                "✅ Implement exponential backoff for retries",
                "✅ Log all errors with context",
            ],
        ),
        (
            "Streaming",
            [
                "✅ Use async for to iterate streams",
                "✅ Handle stream interruption gracefully",
                "✅ Process chunks incrementally",
            ],
        ),
        (
            "Performance",
            [
                "✅ Reuse client connections",
                "✅ Use connection pooling",
                "✅ Batch similar requests",
                "✅ Monitor async task overhead",
            ],
        ),
    ]

    for category, items in practices:
        print(f"\n📋 {category}:")
        for item in items:
            print(f"   {item}")


# =============================================================================
# Main Example Runner
# =============================================================================


async def main():
    """Run all async pattern examples."""
    print("=" * 60)
    print("Venice AI SDK - Production Async Patterns")
    print("=" * 60)

    await example_concurrent_requests()
    await example_context_managers()
    await example_task_cancellation()
    await example_async_error_handling()
    await example_async_streaming()
    await example_semaphore_pattern()
    await example_background_tasks()
    await example_retry_pattern()
    await show_best_practices()

    print("\n" + "=" * 60)
    print("✅ All async patterns demonstrated!")
    print("=" * 60)

    print("\n🔑 Key Takeaways:")
    print("   1. Always use async with for automatic resource cleanup")
    print("   2. Use client.gather() for bounded-concurrency batching")
    print("   3. Implement proper error handling with return_exceptions")
    print("   4. Use client.with_retries() for scoped retry policies")
    print("   5. Handle task cancellation and timeouts")
    print("   6. Use async for to iterate over streams")
    print("   7. Monitor and manage background tasks")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
