#!/usr/bin/env python3
"""
Venice AI SDK - Prompt Caching
==============================

This example demonstrates how to use Venice AI's prompt caching features
to reduce costs (up to 90% discount on cached tokens) and latency for
repeated or multi-turn conversations.

Key features covered:
- prompt_cache_key: Routing hint for cache affinity across requests
- cache_control: Fine-grained markers on content blocks for caching
- Monitoring cache performance via response usage metrics
"""

import asyncio
import sys
import time

from venice_ai import VeniceClient
from venice_ai.types.api import AssistantMessage, ChatUsage, SystemMessage, UserMessage
from venice_ai.types.api.requests.common import TextContent

# Type alias for message lists accepted by the API
Message = UserMessage | AssistantMessage | SystemMessage

# ---------------------------------------------------------------------------
# Long system prompt used across examples to make caching worthwhile.
# In production you'd use your own domain-specific system instructions.
# ---------------------------------------------------------------------------
LONG_SYSTEM_PROMPT = """\
You are a senior software architect specializing in distributed systems design.

Your expertise covers the following areas in depth:

1. **Microservices Architecture**: You understand service decomposition, bounded contexts,
   API gateway patterns, service mesh topologies, and inter-service communication strategies
   including synchronous REST/gRPC and asynchronous event-driven messaging via Kafka, RabbitMQ,
   and NATS.

2. **Database Design**: You are proficient in relational modeling (PostgreSQL, MySQL),
   document stores (MongoDB, CouchDB), key-value stores (Redis, DynamoDB), wide-column stores
   (Cassandra, ScyllaDB), and graph databases (Neo4j, ArangoDB). You can advise on schema
   design, indexing strategies, partitioning, replication, and consistency trade-offs (CAP theorem,
   PACELC).

3. **Cloud-Native Infrastructure**: You have hands-on experience with Kubernetes orchestration,
   Helm chart authoring, Terraform IaC, CI/CD pipelines (GitHub Actions, GitLab CI, ArgoCD),
   observability stacks (Prometheus, Grafana, OpenTelemetry, Jaeger), and multi-cloud deployment
   strategies across AWS, GCP, and Azure.

4. **Performance & Reliability**: You can design for low-latency, high-throughput workloads using
   caching layers (Redis, Memcached, CDN edge caching), load balancing (L4/L7, consistent hashing),
   retry strategies, bulkheads, retries with exponential back-off, and chaos engineering practices.

5. **Security**: You follow zero-trust principles, understand OAuth 2.0 / OIDC flows, service-to-
   service mTLS, secret management (Vault, AWS Secrets Manager), and supply-chain security (SBOM,
   Sigstore, SLSA).

When answering questions, always:
- Provide concrete, production-ready recommendations rather than theoretical overviews.
- Cite trade-offs explicitly so the user can make informed decisions.
- Include code snippets or configuration examples when helpful.
- Keep answers concise but thorough. Aim for the right level of detail.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_usage(usage: ChatUsage | None, *, label: str = "") -> None:
    """Pretty-print token usage from a response. No-op if usage is missing."""
    prefix = f"   📊 [{label}] " if label else "   📊 "
    if usage is None:
        print(f"{prefix}Usage data not provided by the API.")
        return
    print(
        f"{prefix}Tokens — prompt: {usage.prompt_tokens}, "
        f"completion: {usage.completion_tokens}, total: {usage.total_tokens}"
    )

    if usage.prompt_tokens_details is not None:
        cached = usage.prompt_tokens_details.cached_tokens
        if cached is not None:
            print(f"{prefix}Cached tokens: {cached}")


def _indent(text: str, spaces: int = 6) -> str:
    """Indent every line of *text* by *spaces* spaces."""
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


async def basic_prompt_caching() -> bool:
    """Demonstrate basic prompt_cache_key usage with a long system prompt.

    Returns True on success, False on failure.
    """
    print("💾 Basic Prompt Caching with prompt_cache_key")
    print("-" * 50)

    async with VeniceClient() as client:
        try:
            chat_model = await client.models.resolve_chat()
            print(f"   🤖 Selected model: {chat_model}")

            cache_key = "demo-architect-session-001"
            # Pad the system prompt to ~1500+ tokens so it crosses the typical
            # ~1024-token cache-prefix threshold most providers enforce.
            padded_system = (
                LONG_SYSTEM_PROMPT
                + "\n\nADDITIONAL REFERENCE NOTES:\n"
                + (
                    "- This is reference material reused verbatim across requests "
                    "to demonstrate prompt-prefix caching. " * 60
                )
            )
            messages: list[Message] = [
                SystemMessage(content=padded_system),
                UserMessage(
                    content="What are the main trade-offs between REST and gRPC for inter-service communication?"
                ),
            ]

            # --- First request: populates the cache ---
            print(f"\n📤 Request 1 (cold cache) — cache key: {cache_key}")
            start = time.perf_counter()
            response1 = await client.chat.completions.create(
                model=chat_model,
                messages=messages,
                max_completion_tokens=200,
                prompt_cache_key=cache_key,
            )
            elapsed1 = time.perf_counter() - start

            content1 = str(response1.text or "")
            print(f"   ⏱️  Latency: {elapsed1:.2f}s")
            print(f"   📝 Response: {content1[:120]}...")
            _print_usage(response1.usage, label="Request 1")

            # --- Second request: same cache key, should benefit from cache ---
            print(f"\n📤 Request 2 (warm cache) — same cache key: {cache_key}")
            start = time.perf_counter()
            response2 = await client.chat.completions.create(
                model=chat_model,
                messages=messages,
                max_completion_tokens=200,
                prompt_cache_key=cache_key,
            )
            elapsed2 = time.perf_counter() - start

            content2 = str(response2.text or "")
            print(f"   ⏱️  Latency: {elapsed2:.2f}s")
            print(f"   📝 Response: {content2[:120]}...")
            _print_usage(response2.usage, label="Request 2")

            # --- Compare ---
            if elapsed1 > 0:
                # Positive value means request 2 was faster (i.e. cache helped).
                speedup = ((elapsed1 - elapsed2) / elapsed1) * 100
                if speedup > 0:
                    print(f"\n⚡ Request 2 was {speedup:.1f}% faster than request 1")
                else:
                    print(
                        f"\n⚡ Request 2 was {-speedup:.1f}% slower than request 1 — no cache benefit observed"
                    )
            print(
                "   💡 Tip: Speedup depends on model, load, and whether content was already cached server-side."
            )
            print(
                "   💡 Caching usually requires >1024 tokens of identical prefix; this example pads the prompt to that range."
            )
            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False


async def multi_turn_caching() -> bool:
    """Show how prompt_cache_key helps in multi-turn conversations.

    Returns True on success, False on failure.
    """
    print("\n\n🔄 Multi-Turn Conversation Caching")
    print("-" * 50)

    async with VeniceClient() as client:
        try:
            chat_model = await client.models.resolve_chat()
            print(f"   🤖 Selected model: {chat_model}")

            # Use a stable cache key for the whole conversation so the growing
            # message history can benefit from prefix caching.
            cache_key = "demo-multiturn-arch-review"

            conversation: list[Message] = [
                SystemMessage(content=LONG_SYSTEM_PROMPT),
            ]

            questions = [
                "How should I partition a PostgreSQL database handling 50k writes/sec?",
                "What replication strategy would you recommend for that setup?",
                "How do I handle failover without data loss?",
            ]

            for i, question in enumerate(questions, 1):
                conversation.append(UserMessage(content=question))

                print(f"\n📤 Turn {i}: {question[:70]}...")
                start = time.perf_counter()
                response = await client.chat.completions.create(
                    model=chat_model,
                    messages=conversation,
                    max_completion_tokens=200,
                    prompt_cache_key=cache_key,
                )
                elapsed = time.perf_counter() - start

                assistant_text = str(response.text or "")
                # Keep the assistant reply in history for the next turn
                conversation.append(AssistantMessage.from_response(response))

                print(f"   ⏱️  Latency: {elapsed:.2f}s")
                print(f"   📝 Response: {assistant_text[:100]}...")
                _print_usage(response.usage, label=f"Turn {i}")

            print("\n💡 With prompt_cache_key the system prompt and earlier turns are")
            print("   increasingly likely to be served from cache as the conversation grows.")
            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False


async def cache_control_markers() -> bool:
    """Use TextContent with cache_control to mark specific blocks for caching.

    Returns True on success, False on failure.
    """
    print("\n\n🏷️  Cache Control Markers (cache_control on TextContent)")
    print("-" * 50)

    async with VeniceClient() as client:
        try:
            chat_model = await client.models.resolve_chat()
            print(f"   🤖 Selected model: {chat_model}")

            # Build the system message using TextContent blocks with cache_control.
            # The long reference material is marked as cacheable; the short instruction
            # prefix is not, since it may change more frequently.
            system_message = SystemMessage(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            "You are a helpful assistant that answers questions about "
                            "the following reference material."
                        ),
                    ),
                    TextContent(
                        type="text",
                        text=LONG_SYSTEM_PROMPT,
                        cache_control={"type": "ephemeral"},
                    ),
                ],
            )

            user_message = UserMessage(
                content=[
                    TextContent(
                        type="text",
                        text="Summarize the key areas of expertise in a bullet list.",
                    ),
                ],
            )

            print("📤 Sending request with cache_control markers...")
            print('   📌 System context marked with cache_control={"type": "ephemeral"}')

            start = time.perf_counter()
            response = await client.chat.completions.create(
                model=chat_model,
                messages=[system_message, user_message],
                max_completion_tokens=300,
            )
            elapsed = time.perf_counter() - start

            content = str(response.text or "")
            print(f"   ⏱️  Latency: {elapsed:.2f}s")
            print(f"   📝 Response:\n{_indent(content[:300])}")
            _print_usage(response.usage, label="cache_control request")

            print('\n💡 cache_control={"type": "ephemeral"} tells the API which content')
            print("   blocks are safe to cache. Combine with prompt_cache_key for best results.")
            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False


async def monitor_cache_performance() -> bool:
    """Show how to inspect response.usage for cache-related metrics.

    Returns True on success, False on failure.
    """
    print("\n\n📊 Monitoring Cache Performance")
    print("-" * 50)

    async with VeniceClient() as client:
        try:
            chat_model = await client.models.resolve_chat()
            print(f"   🤖 Selected model: {chat_model}")

            cache_key = "demo-monitoring-session"
            messages: list[Message] = [
                SystemMessage(content=LONG_SYSTEM_PROMPT),
                UserMessage(content="Give a one-sentence summary of your expertise."),
            ]

            # Make two requests so the second can potentially hit cache
            for i in range(1, 3):
                label = "cold" if i == 1 else "warm"
                print(f"\n📤 Request {i} ({label} cache)")

                response = await client.chat.completions.create(
                    model=chat_model,
                    messages=messages,
                    max_completion_tokens=100,
                    prompt_cache_key=cache_key,
                )

                usage = response.usage
                if usage is None:
                    print("   ⚠️  No usage data returned by the API.")
                    continue
                print(f"   📈 prompt_tokens:      {usage.prompt_tokens}")
                print(f"   📈 completion_tokens:   {usage.completion_tokens}")
                print(f"   📈 total_tokens:        {usage.total_tokens}")

                # Check for detailed cache metrics
                if usage.prompt_tokens_details:
                    details = usage.prompt_tokens_details
                    cached = details.cached_tokens
                    if cached is not None:
                        print(f"   ✅ cached_tokens:       {cached}")
                        if usage.prompt_tokens > 0:
                            hit_rate = (cached / usage.prompt_tokens) * 100
                            print(f"   📊 cache hit rate:      {hit_rate:.1f}%")
                    else:
                        print("   ℹ️  cached_tokens:       not reported by model")

                    audio = details.audio_tokens
                    if audio is not None:
                        print(f"   🔊 audio_tokens:        {audio}")
                else:
                    print("   ℹ️  prompt_tokens_details not present in this response")

            print("\n💡 Not all models populate prompt_tokens_details or cached_tokens.")
            print("   When available, use cached_tokens to verify cache effectiveness.")
            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False


async def best_practices() -> bool:
    """Print best-practice tips for prompt caching (no API calls needed).

    Returns True (informational; cannot fail).
    """
    print("\n\n📚 Prompt Caching Best Practices")
    print("-" * 50)

    tips = [
        (
            "🔑 Choose stable cache keys",
            "Use session IDs, user IDs, or conversation IDs as cache keys.\n"
            "      Requests sharing the same key are routed for cache affinity.",
        ),
        (
            "📏 Cache long, static content",
            "System prompts, reference docs, and few-shot examples benefit most.\n"
            "      Short prompts have less room for savings.",
        ),
        (
            "🏷️  Use cache_control for fine-grained control",
            'Mark cacheable TextContent blocks with cache_control={"type": "ephemeral"}.\n'
            "      Leave frequently-changing content unmarked.",
        ),
        (
            "🔄 Reuse keys across turns",
            "In multi-turn chats, keep the same prompt_cache_key for the entire\n"
            "      conversation so the growing prefix stays cached.",
        ),
        (
            "📊 Monitor usage metrics",
            "Check response.usage.prompt_tokens_details.cached_tokens to measure\n"
            "      actual cache hit rates and estimate cost savings.",
        ),
        (
            "💰 Estimate savings",
            "Cached tokens can be up to 90% cheaper. For a 2000-token system prompt\n"
            "      reused 100 times, that's ~180k tokens worth of savings.",
        ),
        (
            "⚠️  Don't over-key",
            "Using unique cache keys per request defeats the purpose.\n"
            "      Group related requests under the same key.",
        ),
        (
            "🔀 Combine both features",
            "Use prompt_cache_key for routing affinity AND cache_control markers\n"
            "      on content blocks for maximum caching benefit.",
        ),
    ]

    for title, detail in tips:
        print(f"\n   {title}")
        print(f"      {detail}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all prompt caching examples.

    Returns 0 if every section succeeded, 1 if any failed.
    """
    print("🚀 Venice AI Prompt Caching Examples")
    print("=" * 60)

    results: list[tuple[str, bool]] = []
    results.append(("Basic Prompt Caching", await basic_prompt_caching()))
    results.append(("Multi-Turn Caching", await multi_turn_caching()))
    results.append(("Cache Control Markers", await cache_control_markers()))
    results.append(("Monitor Cache Performance", await monitor_cache_performance()))
    results.append(("Best Practices", await best_practices()))

    print("\n\n✨ Prompt caching examples completed!")
    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    if failed:
        print(f"⚠️ {passed}/{len(results)} sections succeeded; {failed} failed")
        for name, ok in results:
            status = "✓" if ok else "✗"
            print(f"   {status} {name}")
    print("\n💡 Key concepts demonstrated:")
    print("   - prompt_cache_key for routing affinity (top-level create() param)")
    print("   - cache_control markers on TextContent blocks")
    print("   - Monitoring cached_tokens in response usage")
    print("   - Multi-turn conversation caching patterns")
    print("   - Best practices for maximizing cache hit rates")
    print("\n📚 Next Steps:")
    print("   - Integrate prompt_cache_key into your conversation flows")
    print("   - Add cache_control to long system prompts and reference docs")
    print("   - Monitor prompt_tokens_details.cached_tokens in production")

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
