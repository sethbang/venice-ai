"""
Model selection strategies for Venice AI testing.

This module provides reusable selection strategies for the DynamicModelSelector,
particularly useful for test suites that need cost optimization and load distribution.

Functions:
    get_model_price: Extract comparable price from a model dictionary
    random_cheap_strategy: Select randomly from the cheapest tier of models

Example:
    >>> from venice_ai import create_model_selector
    >>> from venice_ai.test_support.strategies import random_cheap_strategy
    >>>
    >>> selector = create_model_selector(client, default_selector=random_cheap_strategy)
    >>> model = await selector.select_chat_model()  # Randomly picks from cheap models
"""

import random
from typing import Any


def get_model_price(model: dict[str, Any]) -> float | None:
    """
    Extract comparable price from a model dictionary.

    Handles different pricing structures for different model types:
    - LLM models: input + output token prices
    - Image models: generation price
    - Audio/TTS models: input price

    Args:
        model: Model dictionary from the cache (contains model_spec with pricing)

    Returns:
        Comparable price as float, or None if pricing unavailable.
        For LLM models, returns the sum of input and output prices.
        For image models, returns the generation price.
        For audio models, returns the input price.
    """
    pricing = model.get("model_spec", {}).get("pricing")
    if pricing is None:
        return None

    # LLM pricing: input + output token prices
    if "input" in pricing and "output" in pricing:
        input_usd = pricing["input"].get("usd", 0) if pricing["input"] else 0
        output_usd = pricing["output"].get("usd", 0) if pricing["output"] else 0
        if input_usd is None:
            input_usd = 0
        if output_usd is None:
            output_usd = 0
        return float(input_usd) + float(output_usd)

    # Image pricing: generation cost
    if "generation" in pricing:
        gen = pricing["generation"]
        if gen and gen.get("usd") is not None:
            return float(gen["usd"])
        return None

    # Audio/TTS pricing: input cost only
    if "input" in pricing:
        inp = pricing["input"]
        if inp and inp.get("usd") is not None:
            return float(inp["usd"])
        return None

    return None


def random_cheap_strategy(
    candidates: list[dict[str, Any]],
    price_cliff_multiplier: float = 5.0,
) -> str:
    """
    Select a random model from the cheapest tier.

    This strategy is designed for test suites to:
    1. Optimize cost by prioritizing cheap models
    2. Distribute load across multiple models to avoid rate limits
    3. Handle models without pricing gracefully

    Algorithm:
    1. Partition models into priced and unpriced pools
    2. If priced models exist, find the cheapest price
    3. Build a "cheap pool" of models within 5x of the minimum price
    4. Randomly select from this cheap pool
    5. If no priced models, fall back to random selection from unpriced pool

    Args:
        candidates: List of model dictionaries from the cache
        price_cliff_multiplier: Multiplier for the price cliff threshold (default 5.0).
            Models within (min_price * multiplier) are considered "cheap".

    Returns:
        Selected model ID string

    Raises:
        ValueError: If no candidates are available for selection

    Example:
        >>> models = [
        ...     {"id": "cheap-model", "model_spec": {"pricing": {"input": {"usd": 0.001}, "output": {"usd": 0.002}}}},
        ...     {"id": "expensive-model", "model_spec": {"pricing": {"input": {"usd": 0.1}, "output": {"usd": 0.2}}}},
        ... ]
        >>> selected = random_cheap_strategy(models)
        >>> # Will likely return "cheap-model" since it's in the cheap tier
    """
    if not candidates:
        raise ValueError("No candidates available for selection")

    # Partition into priced and unpriced
    priced_models: list[tuple[dict[str, Any], float]] = []
    unpriced_models: list[dict[str, Any]] = []

    for model in candidates:
        price = get_model_price(model)
        if price is not None and price >= 0:  # Exclude negative/invalid prices
            priced_models.append((model, price))
        else:
            unpriced_models.append(model)

    # If we have priced models, use price cliff logic
    if priced_models:
        # Sort by price ascending
        priced_models.sort(key=lambda x: x[1])
        min_price = priced_models[0][1]

        # Handle zero-price edge case (free models)
        # Include models up to a small threshold if cheapest is free
        threshold = 0.001 if min_price == 0 else min_price * price_cliff_multiplier

        cheap_pool = [m for m, p in priced_models if p <= threshold]

        # Validate pool is non-empty (should always be true if priced_models is non-empty)
        if cheap_pool:
            # Non-cryptographic: spreading test/load across cheap candidates.
            selected = random.choice(cheap_pool)  # nosec B311
            return selected["id"]

    # Fallback: randomly select from unpriced models
    if unpriced_models:
        # Non-cryptographic: spreading test/load across unpriced candidates.
        selected = random.choice(unpriced_models)  # nosec B311
        return selected["id"]

    # Final fallback: shouldn't reach here, but return first candidate
    return candidates[0]["id"]


def cheapest_model_strategy(candidates: list[dict[str, Any]]) -> str:
    """
    Select the single cheapest model (no randomization).

    Useful for production scenarios where cost optimization is the primary goal
    and load distribution is not needed.

    Args:
        candidates: List of model dictionaries from the cache

    Returns:
        Selected model ID string (the cheapest priced model, or first unpriced)

    Raises:
        ValueError: If no candidates are available for selection
    """
    if not candidates:
        raise ValueError("No candidates available for selection")

    best_model = None
    best_price = float("inf")

    for model in candidates:
        price = get_model_price(model)
        if price is not None and price >= 0 and price < best_price:
            best_price = price
            best_model = model

    if best_model:
        return best_model["id"]

    # No priced models, return first candidate
    return candidates[0]["id"]


def first_available_strategy(candidates: list[dict[str, Any]]) -> str:
    """
    Select the first available model (deterministic).

    Simple strategy that always returns the first candidate. Useful as a
    baseline or for scenarios requiring deterministic behavior.

    Args:
        candidates: List of model dictionaries from the cache

    Returns:
        Selected model ID string

    Raises:
        ValueError: If no candidates are available for selection
    """
    if not candidates:
        raise ValueError("No candidates available for selection")

    return candidates[0]["id"]


def random_strategy(candidates: list[dict[str, Any]]) -> str:
    """
    Select a random model from all candidates (uniform distribution).

    Simple randomization without considering pricing. Useful for maximum
    load distribution when cost is not a concern.

    Args:
        candidates: List of model dictionaries from the cache

    Returns:
        Selected model ID string

    Raises:
        ValueError: If no candidates are available for selection
    """
    if not candidates:
        raise ValueError("No candidates available for selection")

    # Non-cryptographic: spreading test/load across all candidates.
    return random.choice(candidates)["id"]  # nosec B311
