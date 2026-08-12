#!/usr/bin/env python3
"""
Venice AI SDK - Reasoning Effort
=================================

This example demonstrates how to use Venice AI's ``reasoning_effort`` parameter
to control thinking depth on reasoning models. Different effort levels trade off
response quality against speed and token usage.

As of the March 2026 API update the effort enum has been widened from the
original three tiers to seven:
``none | minimal | low | medium | high | xhigh | max``. This example exercises a
representative subset of those tiers (to keep the demo fast); the full enum is
documented above and any tier accepted by the SDK can be passed.

Key features covered:
- reasoning_effort: top-level ``create()`` parameter across all seven tiers
- The nested ``reasoning`` object with ``effort`` and ``summary`` controls
- Performance vs quality tradeoffs (latency and token usage per tier)
- Combining reasoning_effort with venice_parameters
"""

import asyncio
import sys
import time
from typing import Any

from venice_ai import (
    InternalServerError,
    RateLimitError,
    ReasoningConfig,
    ReasoningEffortLevel,
    ReasoningSummary,
    VeniceClient,
)
from venice_ai.types.api import ChatUsage, UserMessage, VeniceParameters

EXCLUDED_MODELS = ["venice-uncensored", "venice-uncensored-1-2"]
PREFERRED_REASONING_MODELS = [
    "qwen3-235b-a22b-thinking-2507",
    "qwen-3-6-max-preview",
    "qwen-3-6-plus",
    "kimi-k2-5",
]
TIER_RESULTS: dict[str, dict[str, object]] = {}

# Keep completion budgets modest: reasoning models spend most of their budget on
# hidden thinking, so a few thousand tokens is plenty to demonstrate behavior
# while keeping the (sequential) demo fast enough to finish well under a couple
# of minutes.
MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(*args: Any, **kwargs: Any) -> None:
    """``print`` that always flushes, so output survives an early process kill."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def _print_usage(usage: ChatUsage | None, *, label: str = "") -> None:
    """Pretty-print token usage from a response. No-op if usage is missing."""
    prefix = f"   📊 [{label}] " if label else "   📊 "
    if usage is None:
        log(f"{prefix}Usage data not provided by the API.")
        return
    log(
        f"{prefix}Tokens — prompt: {usage.prompt_tokens}, "
        f"completion: {usage.completion_tokens}, total: {usage.total_tokens}"
    )


def _truncate(text: str, length: int = 200) -> str:
    """Truncate text to a given length with ellipsis."""
    if len(text) <= length:
        return text
    return text[:length] + "..."


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


async def basic_reasoning_effort() -> bool:
    """Run the same prompt across a representative set of effort tiers.

    Records per-tier outcomes in ``TIER_RESULTS``. A tier that is rate-limited or
    that the API rejects with a server 500 is recorded as informational (not a
    failure) — not every model honors every tier, and a generic 500 does not
    prove the tier is unsupported. Returns ``True`` if at least one tier produced
    a valid response, ``False`` only if every tier failed.
    """
    log("🧠 Basic Reasoning Effort — Comparing Effort Tiers")
    log("-" * 60)

    async with VeniceClient() as client:
        reasoning_model = await client.models.resolve_chat(
            require_reasoning=True,
            preferred_models=PREFERRED_REASONING_MODELS,
            exclude_models=EXCLUDED_MODELS,
        )
        log(f"   🤖 Using reasoning model: {reasoning_model}")
        log(
            "   ℹ️  Note: not every reasoning model honors every tier — some collapse "
            "'minimal' / 'xhigh' / 'max' to nearby tiers internally."
        )

        prompt = "In one short paragraph, why do a triangle's angles sum to 180°?"
        # A representative subset of the 7-tier enum spanning the range. The full
        # set is none|minimal|low|medium|high|xhigh|max; exercising every tier
        # sequentially is slow, so we sample the span here.
        levels: list[ReasoningEffortLevel] = ["none", "low", "max"]

        for level in levels:
            log(f'\n   ⚙️  reasoning_effort = "{level}"')
            start = time.time()
            try:
                response = await client.chat.completions.create(
                    model=reasoning_model,
                    messages=[UserMessage(content=prompt)],
                    reasoning_effort=level,
                    max_completion_tokens=MAX_TOKENS,
                )
            except RateLimitError as e:
                elapsed = time.time() - start
                log(f"   ⏱️  Latency: {elapsed:.2f}s")
                log(f"   ⚠️  Rate-limited on tier '{level}' (transient): {e}")
                TIER_RESULTS[level] = {
                    "success": False,
                    "completion_tokens": None,
                    "error": f"RateLimitError: {e}",
                }
                continue
            except InternalServerError as e:
                elapsed = time.time() - start
                log(f"   ⏱️  Latency: {elapsed:.2f}s")
                log(
                    f"   ⚠️  Server error on tier '{level}': {e} "
                    f"— may be transient or this tier may not be honored (informational)"
                )
                TIER_RESULTS[level] = {
                    "success": False,
                    "completion_tokens": None,
                    "error": f"InternalServerError: {e}",
                }
                continue

            elapsed = time.time() - start
            content = str(response.text or "")
            reasoning = getattr(response.choices[0].message, "reasoning_content", None)
            if reasoning and not content:
                content = str(reasoning)
            usage = response.usage
            completion_tokens = usage.completion_tokens if usage else None

            log(f"   ⏱️  Latency: {elapsed:.2f}s")
            log(f"   📝 Response: {_truncate(content, 150)}")
            _print_usage(usage, label=level)

            TIER_RESULTS[level] = {
                "success": True,
                "completion_tokens": completion_tokens,
                "error": None,
            }

        log("\n💡 Higher effort → more thorough reasoning, more tokens, potentially slower.")

    return any(r.get("success") is True for r in TIER_RESULTS.values())


async def performance_vs_quality() -> bool:
    """Time each request to show how effort level affects speed and depth.

    Returns ``True`` if at least one tier produced a response; a per-tier
    rate-limit / server-500 is recorded but does not fail the demo.
    """
    log("\n\n⚡ Performance vs Quality Tradeoff")
    log("-" * 60)

    async with VeniceClient() as client:
        reasoning_model = await client.models.resolve_chat(
            require_reasoning=True,
            preferred_models=PREFERRED_REASONING_MODELS,
            exclude_models=EXCLUDED_MODELS,
        )

        prompt = (
            "A farmer has 100 meters of fencing. What rectangular pen dimensions "
            "maximize the enclosed area? Show your work briefly."
        )

        results = []
        levels: list[ReasoningEffortLevel] = ["low", "high"]
        for level in levels:
            log(f'\n   ⚙️  reasoning_effort = "{level}"')
            start = time.time()
            try:
                response = await client.chat.completions.create(
                    model=reasoning_model,
                    messages=[UserMessage(content=prompt)],
                    reasoning_effort=level,
                    max_completion_tokens=MAX_TOKENS,
                )
            except (RateLimitError, InternalServerError) as e:
                elapsed = time.time() - start
                log(f"   ⏱️  Latency: {elapsed:.2f}s")
                log(f"   ⚠️  Tier '{level}' skipped ({type(e).__name__}, informational): {e}")
                continue
            elapsed = time.time() - start

            usage = response.usage
            content = str(response.text or "")
            reasoning = getattr(response.choices[0].message, "reasoning_content", None)
            if reasoning and not content:
                content = str(reasoning)
            results.append(
                {
                    "level": level,
                    "elapsed": elapsed,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                    "response_length": len(content),
                }
            )

            log(f"   ⏱️  Latency: {elapsed:.2f}s")
            log(f"   📏 Response length: {len(content)} chars")
            _print_usage(usage, label=level)

        if results:
            log("\n   📊 Summary Comparison:")
            log(f"   {'Level':<8} {'Time':>8} {'Tokens':>8} {'Chars':>8}")
            log(f"   {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
            for r in results:
                log(
                    f"   {r['level']:<8} {r['elapsed']:>7.2f}s "
                    f"{r['total_tokens']:>7} {r['response_length']:>7}"
                )

        log("\n💡 Lower effort is faster but may skip steps; higher effort is thorough.")

    return bool(results)


async def combining_with_venice_parameters() -> bool:
    """Show reasoning_effort alongside venice_parameters like strip_thinking_response.

    Returns ``True`` if at least one variant produced a response; a per-call
    rate-limit / server-500 is recorded but does not fail the demo.
    """
    log("\n\n🔗 Combining reasoning_effort with Venice Parameters")
    log("-" * 60)

    ok = False
    async with VeniceClient() as client:
        reasoning_model = await client.models.resolve_chat(
            require_reasoning=True,
            preferred_models=PREFERRED_REASONING_MODELS,
            exclude_models=EXCLUDED_MODELS,
        )

        prompt = "Explain the halting problem in simple terms (briefly)."

        log("\n   🧠 High effort + thinking visible (default)")
        start = time.time()
        try:
            response_visible = await client.chat.completions.create(
                model=reasoning_model,
                messages=[UserMessage(content=prompt)],
                reasoning_effort="high",
                max_completion_tokens=MAX_TOKENS,
            )
        except (RateLimitError, InternalServerError) as e:
            log(f"   ⚠️  Visible variant skipped ({type(e).__name__}, informational): {e}")
        else:
            elapsed_visible = time.time() - start
            content_visible = str(response_visible.text or "")
            reasoning = getattr(response_visible.choices[0].message, "reasoning_content", None)
            log(f"   ⏱️  Latency: {elapsed_visible:.2f}s")
            if reasoning:
                log(f"   🤔 Reasoning content: {_truncate(str(reasoning), 100)}")
            log(f"   📝 Response: {_truncate(content_visible, 150)}")
            _print_usage(response_visible.usage, label="high + visible")
            ok = True

        log("\n   🧠 High effort + strip_thinking_response=True")
        start = time.time()
        try:
            response_stripped = await client.chat.completions.create(
                model=reasoning_model,
                messages=[UserMessage(content=prompt)],
                reasoning_effort="high",
                max_completion_tokens=MAX_TOKENS,
                venice_parameters=VeniceParameters.model_validate(
                    {"strip_thinking_response": True}
                ),
            )
        except (RateLimitError, InternalServerError) as e:
            log(f"   ⚠️  Stripped variant skipped ({type(e).__name__}, informational): {e}")
        else:
            elapsed_stripped = time.time() - start
            content_stripped = str(response_stripped.text or "")
            reasoning_stripped = getattr(
                response_stripped.choices[0].message, "reasoning_content", None
            )
            log(f"   ⏱️  Latency: {elapsed_stripped:.2f}s")
            if reasoning_stripped:
                log(f"   🤔 Reasoning content: {_truncate(str(reasoning_stripped), 100)}")
            else:
                log("   🤔 Reasoning content: (stripped)")
            log(f"   📝 Response: {_truncate(content_stripped, 150)}")
            _print_usage(response_stripped.usage, label="high + stripped")
            ok = True

        log("\n💡 Key distinctions:")
        log("   • reasoning_effort controls HOW DEEPLY the model thinks (low/medium/high)")
        log("   • strip_thinking_response hides the <think> blocks from the output")
        log("   • disable_thinking completely turns off reasoning (different from low effort)")
        log("   • These parameters can be combined for fine-grained control")

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def nested_reasoning_object() -> bool:
    """Demonstrate the nested ``reasoning`` object for effort + summary control.

    The nested config accepts both ``effort`` and a ``summary`` verbosity
    (``"auto"`` / ``"concise"`` / ``"detailed"``). When both ``reasoning_effort``
    and ``reasoning.effort`` are set, ``reasoning_effort`` takes precedence.

    Returns ``True`` if the call produced a response; a rate-limit / server-500
    is recorded but does not fail the demo.
    """
    log("\n\n🧩 Nested reasoning Config — effort + summary")
    log("-" * 60)

    ok = False
    async with VeniceClient() as client:
        reasoning_model = await client.models.resolve_chat(
            require_reasoning=True,
            preferred_models=PREFERRED_REASONING_MODELS,
            exclude_models=EXCLUDED_MODELS,
        )

        prompt = "Briefly explain why prime numbers matter in cryptography."

        summary: ReasoningSummary = "concise"
        log(f"\n   🧾 summary = {summary!r}")
        try:
            response = await client.chat.completions.create(
                model=reasoning_model,
                messages=[UserMessage(content=prompt)],
                reasoning=ReasoningConfig(effort="high", summary=summary),
                max_completion_tokens=MAX_TOKENS,
            )
        except (RateLimitError, InternalServerError) as e:
            log(f"   ⚠️  Skipped ({type(e).__name__}, informational): {e}")
        else:
            content = str(response.text or "")
            log(f"   📝 Response: {_truncate(content, 180)}")
            _print_usage(response.usage, label=summary)
            ok = True

        log("\n💡 Use reasoning.summary to shape how the model narrates its own thinking.")

    return ok


async def main() -> int:
    """Run all reasoning effort examples and return an exit code.

    Exit semantics (per the README note that not every model honors every tier):
    a tier that the API rejects with a generic server 500, or that is
    rate-limited, is treated as *informational*, not a failure. The process
    exits 0 as long as at least one effort tier / demo produced a valid
    response. A genuine error — an authentication failure or any unexpected
    exception — propagates to ``__main__`` and surfaces as a non-zero exit. The
    only path that returns non-zero here is "every demo failed".
    """
    log("🚀 Venice AI Reasoning Effort Examples")
    log("=" * 60)

    # Each demo returns True if it produced at least one valid response.
    demo_results: list[tuple[str, bool]] = [
        ("basic_reasoning_effort", await basic_reasoning_effort()),
        ("performance_vs_quality", await performance_vs_quality()),
        ("combining_with_venice_parameters", await combining_with_venice_parameters()),
        ("nested_reasoning_object", await nested_reasoning_object()),
    ]

    # Per-tier report from basic_reasoning_effort (the tiers it exercised).
    exercised_tiers = list(TIER_RESULTS.keys())
    tier_successes = sum(1 for r in TIER_RESULTS.values() if r.get("success") is True)

    log(
        f"\n\n📈 Per-tier outcome: {tier_successes}/{len(exercised_tiers)} "
        f"exercised effort tiers produced a response"
    )
    for tier in exercised_tiers:
        result = TIER_RESULTS[tier]
        if result["success"]:
            log(f"   • {tier}: ✅ completion_tokens={result['completion_tokens']}")
        else:
            log(f"   • {tier}: ⚠️  {result['error']} (informational)")

    successful_tiers = [
        (tier, int(TIER_RESULTS[tier]["completion_tokens"]))  # type: ignore[arg-type]
        for tier in exercised_tiers
        if TIER_RESULTS[tier].get("success") is True
        and TIER_RESULTS[tier].get("completion_tokens") is not None
    ]
    monotonic = True
    for i in range(1, len(successful_tiers)):
        prev_tier, prev_tokens = successful_tiers[i - 1]
        curr_tier, curr_tokens = successful_tiers[i]
        if curr_tokens < prev_tokens:
            monotonic = False
            log(
                f"   ℹ️  tier {curr_tier} ({curr_tokens} tokens) returned fewer tokens "
                f"than tier {prev_tier} ({prev_tokens} tokens) — model may not strictly "
                f"honor effort (informational)"
            )
    if monotonic and len(successful_tiers) >= 2:
        log("   ✅ Completion tokens are monotonically non-decreasing across tiers.")

    # Demo-level rollup (for reporting only — gating is on at-least-one success).
    failed_demos = [name for name, ok in demo_results if not ok]
    any_success = any(ok for _, ok in demo_results)

    if not any_success:
        # Every demo failed against the live API: that is a genuine failure.
        log("\n❌ Reasoning effort examples failed: no demo produced a response.")
        return 1

    if failed_demos:
        log(
            f"\n✨ Reasoning effort examples completed "
            f"(some tiers/demos were skipped as informational: {', '.join(failed_demos)})."
        )
    else:
        log("\n✨ Reasoning effort examples completed!")

    log("\n💡 Key concepts demonstrated:")
    log("   - reasoning_effort is a top-level create() parameter, not in venice_parameters")
    log("   - Seven levels: none / minimal / low / medium / high / xhigh / max")
    log("   - Nested reasoning={'effort', 'summary'} gives finer control")
    log("   - Higher effort → more tokens, deeper analysis, potentially slower")
    log("   - Combine with strip_thinking_response to hide reasoning from output")
    log("   - Match effort level to task complexity for optimal cost/quality tradeoff")
    log("\n📚 Next Steps:")
    log("   - Use client.models.resolve_chat(require_reasoning=True) to pick a model")
    log("   - Default to 'medium' for most workloads, 'max' for critical analysis")
    log("   - Monitor token usage to optimize costs across effort levels")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!", flush=True)
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr, flush=True)
        print(
            "Check that your API key is valid and that a reasoning-capable model is available.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
