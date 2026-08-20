"""
Venice AI SDK - Production Cost Management

This example demonstrates how to track, monitor, and optimize costs when using
the Venice AI SDK in production:

1. Token usage tracking via the built-in :class:`venice_ai.CostTracker`
2. Cost estimation per request
3. Usage analytics and reporting
4. Budget management via :class:`venice_ai.BudgetManager`
5. Cost optimization strategies

Requirements:
    pip install venice-py
    export VENICE_API_KEY="your-api-key"
"""

import asyncio
import sys
from decimal import Decimal

from venice_ai import BudgetManager, CostTracker, VeniceClient
from venice_ai.exceptions import RateLimitError
from venice_ai.types.api.requests import UserMessage

# =============================================================================
# Example Patterns
# =============================================================================


async def example_basic_cost_tracking(client: VeniceClient, tracker: CostTracker) -> bool:
    print("=" * 60)
    print("Pattern 1: Basic Cost Tracking — manual tracker.track()")
    print("=" * 60)

    try:
        print("\n💰 Tracking costs for multiple requests:\n")

        model = await client.models.resolve_chat()

        prompts = [
            "Explain Python in one sentence",
            "What is async/await?",
            "Quick fact about Venice AI",
        ]

        for prompt in prompts:
            response = await client.chat.completions.create(
                model=model,
                messages=[UserMessage(content=prompt)],
                max_completion_tokens=50,
            )
            cost = await tracker.track(response, metadata={"prompt": prompt})

            print(f"Model: {model}")
            print(f"   Tokens: {response.usage.total_tokens if response.usage else 0}")
            print(f"   Cost: ${cost:.6f}")
            print()

        summary = await tracker.summary()
        print("📊 Summary:")
        print(f"   Total requests: {summary.total_requests}")
        print(f"   Total cost: ${summary.total_cost_usd:.6f}")
        print(f"   Total tokens: {summary.total_tokens}")
        print(f"   Avg cost/request: ${summary.average_cost_usd:.6f}")
        return True
    except Exception as e:
        print(f"\n❌ Pattern 1 failed: {e}")
        return False


async def example_budget_management(client: VeniceClient, tracker: CostTracker) -> bool:
    print("\n" + "=" * 60)
    print("Pattern 2: Budget Management — BudgetManager + can_afford()")
    print("=" * 60)

    try:
        # BudgetManager wraps an existing tracker; both daily and monthly
        # caps are optional but at least one is required.
        budget = BudgetManager(
            tracker=tracker,
            daily_usd=Decimal("1.00"),
            monthly_usd=Decimal("30.00"),
        )

        print(f"\n💵 Daily budget: ${budget.daily_usd}")
        print(f"📅 Monthly budget: ${budget.monthly_usd}\n")

        model = await client.models.resolve_chat()

        # Pre-flight estimate so we ask the budget meaningfully.
        estimate = await client.chat.completions.estimate_cost(
            model=model,
            messages=[UserMessage(content="Request 1")],
            expected_completion_tokens=20,
        )

        for i in range(3):
            if not await budget.can_afford(estimate.total_cost_usd):
                print("⚠️  Budget limit reached!")
                break

            response = await client.chat.completions.create(
                model=model,
                messages=[UserMessage(content=f"Request {i + 1}")],
                max_completion_tokens=20,
            )
            cost = await tracker.track(response)
            remaining = await budget.remaining()

            print(f"✅ Request {i + 1} completed")
            print(f"   Cost: ${cost:.6f}")
            if remaining.daily_remaining_usd is not None and remaining.daily_used_pct is not None:
                print(
                    f"   Remaining: ${remaining.daily_remaining_usd:.6f} "
                    f"({100 - remaining.daily_used_pct:.1f}%)\n"
                )
        return True
    except Exception as e:
        print(f"\n❌ Pattern 2 failed: {e}")
        return False


async def _pick_two_distinct_chat_models(client: VeniceClient) -> list[str]:
    """Choose two distinct chat models likely to have different pricing/output."""
    primary = await client.models.resolve_chat(require_function_calling=True)
    secondary = await client.models.resolve_chat(require_reasoning=True, exclude_models=[primary])

    if secondary == primary:
        catalog = await client.models.list(type="chat")
        for entry in catalog.data:
            if entry.id != primary and entry.model_spec and entry.model_spec.pricing:
                secondary = entry.id
                break

    return [primary, secondary]


async def example_cost_optimization(client: VeniceClient, tracker: CostTracker) -> bool:
    print("\n" + "=" * 60)
    print("Pattern 3: Cost Optimization")
    print("=" * 60)

    try:
        print("\n🎯 Optimization Strategies:\n")

        print("1️⃣  Model Selection")
        print("   ✅ Use smaller models for simple tasks")
        print("   ✅ Reserve large models for complex reasoning\n")

        prompt = "Please tell me a compelling short story."

        models_to_compare = await _pick_two_distinct_chat_models(client)

        if len(set(models_to_compare)) < 2:
            print(
                "   ⚠️ Could not find two distinct chat models in the catalog — "
                "comparison degraded to a single model.\n"
            )
        else:
            print(f"   Comparing: {models_to_compare[0]} vs {models_to_compare[1]}\n")

        per_model_cost: dict[str, Decimal] = {}
        for model in models_to_compare:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[UserMessage(content=prompt)],
                    max_completion_tokens=100,
                )
                cost = await tracker.track(response)
                per_model_cost[model] = cost
                print(f"   {model}: ${cost:.6f}")
            except RateLimitError:
                print(f"   ⏱️ Rate limited on {model}, skipping...")
                continue

        if len(per_model_cost) >= 2:
            ids = list(per_model_cost.keys())
            delta = abs(per_model_cost[ids[0]] - per_model_cost[ids[1]])
            print(f"\n   📐 Cost delta: ${delta:.6f} between {ids[0]} and {ids[1]}")
        elif not per_model_cost:
            print("   ⚠️ No cost data collected (all comparison calls failed).")

        print()

        print("2️⃣  Token Management")
        print("   ✅ Set max_completion_tokens to limit response length")
        print("   ✅ Use shorter prompts when possible")
        print("   ✅ Implement prompt caching for repeated queries\n")

        print("3️⃣  Batch Processing")
        print("   ✅ Group similar requests")
        print("   ✅ Use concurrent requests efficiently")
        print("   ✅ Implement request queuing\n")
        return True
    except Exception as e:
        print(f"\n❌ Pattern 3 failed: {e}")
        return False


async def example_cost_analytics(client: VeniceClient, tracker: CostTracker) -> bool:
    print("\n" + "=" * 60)
    print("Pattern 4: Cost Analytics")
    print("=" * 60)

    try:
        print("\n📈 Generating cost analytics:\n")

        model = await client.models.resolve_chat()

        prompts_tokens = [
            ("Complex analysis task", 100),
            ("Medium task", 50),
            ("Simple task", 20),
            ("Another simple task", 20),
        ]

        for prompt, tokens in prompts_tokens:
            response = await client.chat.completions.create(
                model=model,
                messages=[UserMessage(content=prompt)],
                max_completion_tokens=tokens,
            )
            await tracker.track(response, metadata={"scenario": prompt})

        print("📊 Cost Breakdown by Model:")
        for model_id, cost in (await tracker.by_model()).items():
            print(f"   {model_id}: ${cost:.6f}")

        print()
        summary = await tracker.summary()
        print("📈 Overall Statistics:")
        print(f"   Total cost: ${summary.total_cost_usd:.6f}")
        print(f"   Average per request: ${summary.average_cost_usd:.6f}")
        print(f"   Total tokens: {summary.total_tokens}")
        return True
    except Exception as e:
        print(f"\n❌ Pattern 4 failed: {e}")
        return False


async def show_best_practices(client: VeniceClient) -> bool:
    print("\n" + "=" * 60)
    print("Cost Management Best Practices")
    print("=" * 60)

    try:
        default_chat = await client.models.resolve_chat()
        try:
            fn_calling = await client.models.resolve_chat(require_function_calling=True)
        except Exception:
            fn_calling = default_chat
        try:
            reasoning = await client.models.resolve_chat(require_reasoning=True)
        except Exception:
            reasoning = default_chat
        try:
            code_optimized = await client.models.resolve_chat(require_code_optimization=True)
        except Exception:
            code_optimized = default_chat
        try:
            vision = await client.models.resolve_chat(require_vision=True)
        except Exception:
            vision = default_chat

        practices = [
            (
                "Model Selection (resolved at runtime)",
                [
                    f"✅ General chat: {default_chat} (client.models.resolve_chat())",
                    f"✅ Function calling / tools: {fn_calling}",
                    f"✅ Reasoning-heavy tasks: {reasoning}",
                    f"✅ Code generation: {code_optimized}",
                    f"✅ Multimodal / vision: {vision}",
                    "✅ Always resolve dynamically — never hardcode IDs",
                ],
            ),
            (
                "Token Management",
                [
                    "✅ Set max_completion_tokens based on actual needs",
                    "✅ Use stop sequences to limit generation",
                    "✅ Optimize prompts to be concise",
                    "✅ Cache common responses",
                ],
            ),
            (
                "Monitoring",
                [
                    "✅ Track costs per request via venice_ai.costs.calculate_completion_cost",
                    "✅ Set up budget alerts",
                    "✅ Monitor usage patterns",
                    "✅ Review costs regularly",
                ],
            ),
            (
                "Optimization",
                [
                    "✅ Batch similar requests",
                    "✅ Implement rate limiting",
                    "✅ Use streaming for long responses",
                    "✅ Consider prompt engineering to reduce tokens",
                ],
            ),
            (
                "Budget Control",
                [
                    "✅ Set daily and monthly limits",
                    "✅ Implement retry strategies",
                    "✅ Alert on unusual usage",
                    "✅ Review and adjust budgets periodically",
                ],
            ),
        ]

        for category, items in practices:
            print(f"\n📋 {category}:")
            for item in items:
                print(f"   {item}")
        return True
    except Exception as e:
        print(f"\n❌ Best Practices section failed: {e}")
        return False


async def main() -> int:
    print("=" * 60)
    print("Venice AI SDK - Production Cost Management")
    print("=" * 60)

    async with VeniceClient() as client:
        # CostTracker.from_client pre-populates the pricing map from
        # client.models.list(type="chat") so callers don't have to.
        tracker = await CostTracker.from_client(client)
        if not tracker.pricing_map:
            print("\n❌ Could not load chat-model pricing from the catalog.", file=sys.stderr)
            return 1

        results: list[tuple[str, bool]] = []
        results.append(("Basic Cost Tracking", await example_basic_cost_tracking(client, tracker)))
        await tracker.reset()
        results.append(("Budget Management", await example_budget_management(client, tracker)))
        await tracker.reset()
        results.append(("Cost Optimization", await example_cost_optimization(client, tracker)))
        await tracker.reset()
        results.append(("Cost Analytics", await example_cost_analytics(client, tracker)))
        results.append(("Best Practices", await show_best_practices(client)))

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    if failed == 0:
        print(f"✅ All {passed}/{len(results)} cost management patterns succeeded!")
    else:
        print(f"⚠️ {passed}/{len(results)} patterns succeeded; {failed} failed")
        for name, ok in results:
            status = "✓" if ok else "✗"
            print(f"   {status} {name}")
    print("=" * 60)

    print("\n🔑 Key Takeaways:")
    print("   1. Track token usage and costs for every request")
    print("   2. Choose appropriate model sizes for tasks")
    print("   3. Set and monitor budgets (daily/monthly)")
    print("   4. Optimize prompts and max_completion_tokens settings")
    print("   5. Use cost analytics to identify optimization opportunities")
    print("   6. Implement alerts for unusual usage patterns")
    print("   7. Regular review and adjustment of budgets")

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
