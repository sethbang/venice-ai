"""
Cost Calculation and Estimation Utilities
=========================================

This module provides comprehensive cost calculation and estimation utilities for
Venice AI API usage. It supports real-time cost tracking, pre-request estimation,
and detailed breakdown of usage costs across different pricing models.

All calculations are based on actual token usage and current model pricing information.

Key Features:
    * **Real-time Cost Calculation**: Calculate actual costs from API responses
    * **Pre-request Estimation**: Estimate costs before making API calls
    * **Token-based Pricing**: Accurate cost calculation based on token consumption
    * **Model-specific Pricing**: Different pricing tiers for different models

Pricing Models:
    * **Input Tokens**: Cost for processing input text (prompts, messages)
    * **Output Tokens**: Cost for generating output text (completions, responses)
    * **Flat Rate Models**: Simple per-request pricing for some operations
    * **Tiered Pricing**: Volume-based pricing with different rates

Cost Types:
    * **USD**: Traditional US Dollar pricing for enterprise billing
    * **DIEM**: Platform currency where 1 DIEM = $1 USD

Example:
    >>> from venice_ai.costs import calculate_completion_cost, estimate_completion_cost
    >>>
    >>> # Calculate actual cost from a completion
    >>> completion = await client.chat.completions.create(...)
    >>> entry = await client.models.get(completion.model)
    >>> model_pricing = entry.model_spec.pricing
    >>> cost = calculate_completion_cost(completion, model_pricing)
    >>> print(f"Cost: ${cost['usd']:.6f} USD")
    >>>
    >>> # Estimate cost before making request
    >>> estimated_cost = estimate_completion_cost(
    ...     prompt="Your prompt here",
    ...     estimated_completion_tokens=500,
    ...     model_pricing=model_pricing
    ... )
    >>> print(f"Estimated: ${estimated_cost['usd']:.6f} USD")
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .types.api.chat import ChatCompletionResponse as ChatCompletion
from .types.api.embeddings import EmbeddingsResponse
from .types.api.models import LLMModelPricing as ModelPricing

if TYPE_CHECKING:
    from ._client import VeniceClient


class ChatCostEstimate(BaseModel):
    """Pre-flight cost estimate for a chat completion request.

    Returned by :meth:`client.chat.completions.estimate_cost`. Token counts
    are heuristic word-count approximations (the same approach the
    :func:`estimate_completion_cost` helper uses); ``total_cost_usd`` is
    therefore an estimate, not a guarantee.
    """

    model: str = Field(..., description="Model id used for the estimate")
    prompt_tokens: int = Field(..., description="Estimated input token count")
    expected_completion_tokens: int = Field(
        ..., description="Caller-provided completion token budget"
    )
    prompt_cost_usd: Decimal = Field(..., description="Estimated USD cost for prompt tokens")
    completion_cost_usd: Decimal = Field(
        ..., description="Estimated USD cost for completion tokens"
    )
    total_cost_usd: Decimal = Field(..., description="Sum of prompt + completion USD cost")


def calculate_completion_cost(
    completion: ChatCompletion, model_pricing: ModelPricing | None
) -> dict[str, Decimal]:
    """
    Calculate the actual cost of a completed chat completion request.

    This function analyzes a ChatCompletion response and calculates the precise
    cost based on actual token usage reported by the API. It handles both input
    and output token pricing.

    The calculation uses the token usage data from the completion response and
    applies the current pricing structure for the specific model used. This
    provides accurate post-request cost tracking for billing and analytics.

    Token Cost Calculation:
        * **Input Cost**: prompt_tokens × input_cost_per_million_tokens
        * **Output Cost**: completion_tokens × output_cost_per_million_tokens
        * **Total Cost**: Input Cost + Output Cost

    Args:
        completion: The completed ChatCompletion response containing actual
                   token usage data from the API. Must include usage information
                   with prompt_tokens and completion_tokens counts.
        model_pricing: Current pricing information for the model that was used.
                      Contains input and output costs per million tokens. If None,
                      returns zero cost.

    Returns:
        Dictionary with cost breakdown containing:
        - 'usd': Total cost in US Dollars as a Decimal with exact precision

    Note:
        If the completion lacks usage data or model_pricing is None, the function
        returns zero cost rather than raising an exception to maintain robust
        operation in production environments.

    Example:
        >>> from venice_ai import VeniceClient
        >>> from venice_ai.costs import calculate_completion_cost
        >>>
        >>> client = VeniceClient(api_key="your-api-key")
        >>>
        >>> # Create a chat completion
        >>> completion = await client.chat.completions.create(
        ...     model=await client.models.resolve_chat(),
        ...     messages=[{"role": "user", "content": "Hello world!"}]
        ... )
        >>>
        >>> # Get current model pricing
        >>> entry = await client.models.get(completion.model)
        >>> model_pricing = entry.model_spec.pricing
        >>>
        >>> # Calculate actual costs
        >>> costs = calculate_completion_cost(completion, model_pricing)
        >>> print(f"Cost: ${costs['usd']:.6f} USD")
        >>> print(f"Tokens: {completion.usage.total_tokens} total")
    """
    if not model_pricing or not hasattr(completion, "usage") or not completion.usage:
        return {"usd": Decimal("0.00")}

    # Extract token counts
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens

    # Initialize costs using Decimal for exact precision
    usd_cost = Decimal("0.00")

    # New pricing structure uses nested PricingTier objects
    # Convert to Decimal for exact monetary calculations
    input_usd = Decimal(str(model_pricing.input.usd or 0.0))
    output_usd = Decimal(str(model_pricing.output.usd or 0.0))

    # Calculate USD cost with exact decimal precision
    usd_cost += (Decimal(str(prompt_tokens)) / Decimal("1000000")) * input_usd
    usd_cost += (Decimal(str(completion_tokens)) / Decimal("1000000")) * output_usd

    return {"usd": usd_cost}


def calculate_embedding_cost(
    embedding_response: Any, model_pricing: ModelPricing | None
) -> dict[str, Decimal]:
    """
    Calculate the actual cost of a completed embedding request.

    This function analyzes an embedding response and calculates the cost based on
    the total tokens processed during the embedding generation. Unlike chat
    completions, embeddings typically use only input token pricing since they
    don't generate variable-length outputs.

    Embedding Cost Calculation:
        * **Input Processing**: total_tokens × input_cost_per_million_tokens
        * **Fixed Output**: Embeddings have fixed output dimensions
        * **Total Cost**: Primarily based on input token processing

    Args:
        embedding_response: The completed embedding response containing usage
                           data. Must include a usage object with total_tokens
                           count from the embedding operation.
        model_pricing: Current pricing information for the embedding model.
                      Contains input costs per million tokens. If None,
                      returns zero cost.

    Returns:
        Dictionary with cost breakdown containing:
        - 'usd': Total cost in US Dollars as a Decimal with exact precision

    Example:
        >>> from venice_ai import VeniceClient
        >>> from venice_ai.costs import calculate_embedding_cost
        >>>
        >>> client = VeniceClient(api_key="your-api-key")
        >>>
        >>> # Create embeddings
        >>> response = await client.embeddings.create(
        ...     model=await client.models.resolve_embedding(),
        ...     input="Hello, world! This is a sample text."
        ... )
        >>>
        >>> # Get current model pricing
        >>> entry = await client.models.get(response.model)
        >>> model_pricing = entry.model_spec.pricing
        >>>
        >>> # Calculate actual costs
        >>> costs = calculate_embedding_cost(response, model_pricing)
        >>> print(f"Cost: ${costs['usd']:.6f} USD")
        >>> print(f"Tokens processed: {response.usage.total_tokens}")
    """
    if (
        not model_pricing
        or not hasattr(embedding_response, "usage")
        or not embedding_response.usage
    ):
        return {"usd": Decimal("0.00")}

    total_tokens = embedding_response.usage.total_tokens

    # New pricing structure uses nested PricingTier objects
    # Convert to Decimal for exact monetary calculations
    input_usd = Decimal(str(model_pricing.input.usd or 0.0))

    # Calculate USD cost with exact decimal precision
    usd_cost = (Decimal(str(total_tokens)) / Decimal("1000000")) * input_usd

    return {"usd": usd_cost}


def estimate_completion_cost(
    prompt: str,
    estimated_completion_tokens: int,
    model_pricing: ModelPricing | None,
    tokens_per_word: float = 1.3,
) -> dict[str, Decimal]:
    """
    Estimate the cost of a chat completion before making the API request.

    This function provides pre-request cost estimation based on prompt analysis
    and expected completion length. It uses heuristic token counting to estimate
    input costs and user-provided estimates for output costs, enabling budget
    planning and cost-aware request optimization.

    The estimation is particularly useful for:
    * Budget planning and cost control
    * Optimizing prompts for cost efficiency
    * Batch processing cost estimation
    * User-facing cost previews

    Estimation Methodology:
        * **Input Tokens**: Estimated from word count using configurable ratio
        * **Output Tokens**: User-provided estimate based on expected response length
        * **Pricing**: Applied using current model pricing structure
        * **Accuracy**: Approximation only - actual costs may vary

    Args:
        prompt: The input text to estimate token costs for. This is analyzed
               for word count and converted to estimated tokens using the
               tokens_per_word ratio.
        estimated_completion_tokens: Expected number of tokens in the model's
                                   response. This should be estimated based on
                                   the desired response length and complexity.
        model_pricing: Current pricing information for the target model.
                      Contains input and output costs per million tokens.
                      If None, returns zero cost.
        tokens_per_word: Conversion ratio from words to tokens. Default of 1.3
                        is optimized for English text. Adjust for other contexts:
                        - English text: ~1.3 tokens/word (default)
                        - Japanese/Chinese: ~2.0 tokens/word
                        - Code/technical: ~1.5-2.0 tokens/word
                        - Mixed content: Adjust based on composition

    Returns:
        Dictionary with estimated cost breakdown containing:
        - 'usd': Estimated total cost in US Dollars as a Decimal with exact precision

    Accuracy Notes:
        * Token estimation is heuristic and may not match exact tokenization
        * Actual costs depend on precise tokenizer behavior
        * Output token count is user-estimated and may vary significantly
        * Different models may have different tokenization patterns

    Example:
        >>> from venice_ai import VeniceClient
        >>> from venice_ai.costs import estimate_completion_cost
        >>>
        >>> client = VeniceClient(api_key="your-api-key")
        >>>
        >>> # Get current model pricing for the model you plan to call
        >>> model_id = await client.models.resolve_chat()
        >>> model_pricing = (await client.models.get(model_id)).model_spec.pricing
        >>>
        >>> # Estimate costs for different scenarios
        >>> prompt = "Write a detailed explanation of quantum computing"
        >>>
        >>> # Short response estimate
        >>> short_cost = estimate_completion_cost(
        ...     prompt=prompt,
        ...     estimated_completion_tokens=200,
        ...     model_pricing=model_pricing
        ... )
        >>>
        >>> # Long response estimate
        >>> long_cost = estimate_completion_cost(
        ...     prompt=prompt,
        ...     estimated_completion_tokens=1000,
        ...     model_pricing=model_pricing
        ... )
        >>>
        >>> print(f"Short response: ${short_cost['usd']:.6f} USD")
        >>> print(f"Long response: ${long_cost['usd']:.6f} USD")
        >>> print(f"Cost difference: ${long_cost['usd'] - short_cost['usd']:.6f} USD")
    """
    if not model_pricing:
        return {"usd": Decimal("0.00")}

    # Estimate prompt tokens based on word count
    word_count = len(prompt.split())
    estimated_prompt_tokens = int(word_count * tokens_per_word)

    # Initialize costs using Decimal for exact precision
    usd_cost = Decimal("0.00")

    # New pricing structure uses nested PricingTier objects
    # Convert to Decimal for exact monetary calculations
    input_usd = Decimal(str(model_pricing.input.usd or 0.0))
    output_usd = Decimal(str(model_pricing.output.usd or 0.0))

    # Calculate USD cost with exact decimal precision
    usd_cost += (Decimal(str(estimated_prompt_tokens)) / Decimal("1000000")) * input_usd
    usd_cost += (Decimal(str(estimated_completion_tokens)) / Decimal("1000000")) * output_usd

    return {"usd": usd_cost}


# ---------------------------------------------------------------------------
# Stateful cost tracking — CostTracker / BudgetManager
# ---------------------------------------------------------------------------


class CostRecord(BaseModel):
    """One per-request cost-tracking entry."""

    timestamp: datetime = Field(..., description="UTC timestamp when the request was tracked")
    model: str = Field(..., description="Model id used for the request")
    prompt_tokens: int = Field(..., description="Input/prompt token count from the response")
    completion_tokens: int = Field(
        ..., description="Output/completion token count (0 for embeddings)"
    )
    total_tokens: int = Field(..., description="Total tokens billed for the request")
    cost_usd: Decimal = Field(..., description="Computed cost in USD")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Free-form metadata supplied by the caller"
    )


class CostSummary(BaseModel):
    """Aggregate stats produced by :meth:`CostTracker.summary`."""

    total_requests: int
    total_cost_usd: Decimal
    total_tokens: int
    average_cost_usd: Decimal = Field(
        ..., description="Mean USD cost per tracked request (0 when none)"
    )
    average_tokens: float = Field(
        ..., description="Mean tokens per tracked request (0.0 when none)"
    )


class BudgetRemaining(BaseModel):
    """Remaining-budget snapshot returned by :meth:`BudgetManager.remaining`."""

    daily_remaining_usd: Decimal | None = Field(
        None, description="USD remaining against the daily cap (None if no daily cap)"
    )
    daily_used_pct: float | None = Field(
        None, description="Daily-cap usage as a 0–100 percentage (None if no daily cap)"
    )
    monthly_remaining_usd: Decimal | None = Field(
        None, description="USD remaining against the monthly cap (None if no monthly cap)"
    )
    monthly_used_pct: float | None = Field(
        None, description="Monthly-cap usage as a 0–100 percentage (None if no monthly cap)"
    )


class CostTracker:
    """Stateful, async-safe accumulator for per-request API costs.

    Wraps the existing :func:`calculate_completion_cost` and
    :func:`calculate_embedding_cost` helpers. Three integration paths:

    * **Manual** — call :meth:`track` on each response yourself.
    * **Wired-on-client** — pass to ``VeniceClient(cost_tracker=tracker)``;
      the SDK calls :meth:`track` automatically on every chat / embeddings
      response.
    * **From-client factory** — :meth:`from_client` builds a tracker
      pre-populated with the live pricing map.

    All mutating operations take a single :class:`asyncio.Lock` so concurrent
    in-flight requests can update state safely.
    """

    def __init__(self, pricing_map: dict[str, ModelPricing] | None = None) -> None:
        """:param pricing_map: ``{model_id: LLMModelPricing}``. Models absent
        from the map produce zero-cost records (the underlying helpers
        gracefully handle missing pricing)."""
        self.pricing_map: dict[str, ModelPricing] = dict(pricing_map or {})
        self.requests: list[CostRecord] = []
        self.total_cost_usd: Decimal = Decimal("0.00")
        self.total_tokens: int = 0
        self._lock = asyncio.Lock()

    @classmethod
    async def from_client(cls, client: VeniceClient) -> CostTracker:
        """Build a tracker pre-populated with the live chat-pricing map."""
        catalog = await client.models.list(type="chat")
        pricing_map: dict[str, ModelPricing] = {}
        for entry in catalog.data:
            spec = entry.model_spec
            if spec and spec.pricing and isinstance(spec.pricing, ModelPricing):
                pricing_map[entry.id] = spec.pricing
        return cls(pricing_map=pricing_map)

    async def track(
        self,
        response: ChatCompletion | EmbeddingsResponse,
        *,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Decimal:
        """Record one response and return its USD cost.

        :param response: A :class:`ChatCompletionResponse` or
            :class:`EmbeddingsResponse`.
        :param model: Override the model id used to look up pricing. Defaults
            to ``response.model``.
        :param metadata: Free-form metadata stored on the resulting
            :class:`CostRecord`.
        :raises TypeError: For unsupported response types.
        """
        if isinstance(response, ChatCompletion):
            model_id = model or response.model
            pricing = self.pricing_map.get(model_id)
            cost = calculate_completion_cost(response, pricing)["usd"]
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0
        elif isinstance(response, EmbeddingsResponse):
            model_id = model or response.model
            pricing = self.pricing_map.get(model_id)
            cost = calculate_embedding_cost(response, pricing)["usd"]
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = 0
            total_tokens = response.usage.total_tokens
        else:
            raise TypeError(
                f"CostTracker.track() does not support {type(response).__name__}; "
                f"expected ChatCompletionResponse or EmbeddingsResponse."
            )

        record = CostRecord(
            timestamp=datetime.now(UTC),
            model=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            metadata=dict(metadata or {}),
        )
        async with self._lock:
            self.requests.append(record)
            self.total_cost_usd += cost
            self.total_tokens += total_tokens
        return cost

    async def summary(self) -> CostSummary:
        """Aggregate stats across all tracked requests."""
        async with self._lock:
            n = len(self.requests)
            if n == 0:
                return CostSummary(
                    total_requests=0,
                    total_cost_usd=Decimal("0.00"),
                    total_tokens=0,
                    average_cost_usd=Decimal("0.00"),
                    average_tokens=0.0,
                )
            return CostSummary(
                total_requests=n,
                total_cost_usd=self.total_cost_usd,
                total_tokens=self.total_tokens,
                average_cost_usd=self.total_cost_usd / Decimal(n),
                average_tokens=self.total_tokens / n,
            )

    async def by_model(self) -> dict[str, Decimal]:
        """USD cost grouped by model id."""
        async with self._lock:
            costs: dict[str, Decimal] = {}
            for rec in self.requests:
                costs[rec.model] = costs.get(rec.model, Decimal("0.00")) + rec.cost_usd
            return costs

    async def reset(self) -> None:
        """Clear all tracked state."""
        async with self._lock:
            self.requests.clear()
            self.total_cost_usd = Decimal("0.00")
            self.total_tokens = 0


class BudgetManager:
    """Daily / monthly USD-cap enforcement layered on a :class:`CostTracker`.

    Either ``daily_usd`` or ``monthly_usd`` may be ``None`` to disable that
    cap. The tracker is shared, not owned — ``BudgetManager`` does not call
    :meth:`CostTracker.reset`; callers manage rollover themselves.
    """

    def __init__(
        self,
        *,
        tracker: CostTracker,
        daily_usd: Decimal | None = None,
        monthly_usd: Decimal | None = None,
    ) -> None:
        if daily_usd is None and monthly_usd is None:
            raise ValueError("BudgetManager needs at least one of daily_usd or monthly_usd")
        self.tracker = tracker
        self.daily_usd = daily_usd
        self.monthly_usd = monthly_usd

    async def can_afford(self, estimated_cost_usd: Decimal) -> bool:
        """``True`` if adding *estimated_cost_usd* keeps both caps satisfied."""
        summary = await self.tracker.summary()
        projected = summary.total_cost_usd + estimated_cost_usd
        return not (
            (self.daily_usd is not None and projected > self.daily_usd)
            or (self.monthly_usd is not None and projected > self.monthly_usd)
        )

    async def remaining(self) -> BudgetRemaining:
        """Snapshot of remaining headroom and usage percentages."""
        summary = await self.tracker.summary()
        spent = summary.total_cost_usd

        daily_remaining: Decimal | None = None
        daily_pct: float | None = None
        if self.daily_usd is not None:
            daily_remaining = max(self.daily_usd - spent, Decimal("0"))
            daily_pct = float(spent / self.daily_usd * 100) if self.daily_usd > 0 else 0.0

        monthly_remaining: Decimal | None = None
        monthly_pct: float | None = None
        if self.monthly_usd is not None:
            monthly_remaining = max(self.monthly_usd - spent, Decimal("0"))
            monthly_pct = float(spent / self.monthly_usd * 100) if self.monthly_usd > 0 else 0.0

        return BudgetRemaining(
            daily_remaining_usd=daily_remaining,
            daily_used_pct=daily_pct,
            monthly_remaining_usd=monthly_remaining,
            monthly_used_pct=monthly_pct,
        )


async def _maybe_track_response(tracker: CostTracker, response: Any) -> None:
    """Internal hook: feed a parsed response into *tracker* if it's a tracked type.

    Called from :class:`VeniceClient._request` when a ``cost_tracker`` is wired
    on the client. Silently ignores untracked response types and swallows any
    tracking-side exception so an observability bug never masks a successful
    request.
    """
    if not isinstance(response, ChatCompletion | EmbeddingsResponse):
        return
    try:
        await tracker.track(response)
    except Exception:  # noqa: BLE001 — observability must never break the request
        import logging

        logging.getLogger(__name__).warning("cost_tracker.track() raised; ignoring", exc_info=True)
