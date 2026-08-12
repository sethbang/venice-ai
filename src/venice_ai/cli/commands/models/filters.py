"""
Filtering logic for Venice AI models
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class FilterOptions:
    """Container for all filter options"""

    types: list[str] | None = None
    capabilities: list[str] | None = None
    traits: list[str] | None = None
    max_input_price: float | None = None
    max_output_price: float | None = None
    max_gen_price: float | None = None
    budget: float | None = None
    beta: bool | None = None
    online: bool | None = None
    search_query: str | None = None


class ModelFilter:
    """Handles all model filtering logic"""

    # Capability mapping from CLI flags to model attributes
    CAPABILITY_MAP = {
        "function-calling": "supportsFunctionCalling",
        "vision": "supportsVision",
        "reasoning": "supportsReasoning",
        "web-search": "supportsWebSearch",
        "code": "optimizedForCode",
        "response-schema": "supportsResponseSchema",
        "logprobs": "supportsLogProbs",
    }

    @staticmethod
    def filter_by_type(models: list[Any], types: list[str] | None) -> list[Any]:
        """Filter models by type(s)"""
        if not types or "all" in types:
            return models
        return [m for m in models if m.type in types]

    @staticmethod
    def filter_by_capabilities(models: list[Any], caps: list[str]) -> list[Any]:
        """Filter models by required capabilities"""
        if not caps:
            return models

        result = []
        for model in models:
            if not hasattr(model.model_spec, "capabilities"):
                continue

            model_caps = model.model_spec.capabilities
            has_all_caps = True

            for cap in caps:
                attr = ModelFilter.CAPABILITY_MAP.get(cap.lower())
                if attr and not getattr(model_caps, attr, False):
                    has_all_caps = False
                    break

            if has_all_caps:
                result.append(model)

        return result

    @staticmethod
    def filter_by_price(models: list[Any], options: FilterOptions) -> list[Any]:
        """Filter models by price constraints"""
        result = []

        for model in models:
            if not hasattr(model.model_spec, "pricing"):
                # Include models without pricing if no price filters
                if not any(
                    [
                        options.max_input_price,
                        options.max_output_price,
                        options.max_gen_price,
                        options.budget,
                    ]
                ):
                    result.append(model)
                continue

            pricing = model.model_spec.pricing

            # Check input price
            if (
                options.max_input_price is not None
                and hasattr(pricing, "input")
                and pricing.input
                and hasattr(pricing.input, "usd")
                and pricing.input.usd > options.max_input_price
            ):
                continue

            # Check output price
            if (
                options.max_output_price is not None
                and hasattr(pricing, "output")
                and pricing.output
                and hasattr(pricing.output, "usd")
                and pricing.output.usd > options.max_output_price
            ):
                continue

            # Check generation price (for images)
            if (
                options.max_gen_price is not None
                and hasattr(pricing, "generation")
                and pricing.generation
                and hasattr(pricing.generation, "usd")
                and pricing.generation.usd > options.max_gen_price
            ):
                continue

            # Check budget (average of input/output)
            if (
                options.budget is not None
                and hasattr(pricing, "input")
                and hasattr(pricing, "output")
                and pricing.input
                and pricing.output
                and hasattr(pricing.input, "usd")
                and hasattr(pricing.output, "usd")
            ):
                avg = (pricing.input.usd + pricing.output.usd) / 2
                if avg > options.budget:
                    continue

            result.append(model)

        return result

    @staticmethod
    def filter_by_traits(models: list[Any], traits: list[str]) -> list[Any]:
        """Filter models by trait tags"""
        if not traits:
            return models

        return [
            m
            for m in models
            if hasattr(m.model_spec, "traits")
            and m.model_spec.traits
            and any(t in m.model_spec.traits for t in traits)
        ]

    @staticmethod
    def filter_by_status(models: list[Any], options: FilterOptions) -> list[Any]:
        """Filter models by status (beta, online)"""
        result = models

        if options.beta is not None:
            result = [m for m in result if getattr(m.model_spec, "beta", False) == options.beta]

        if options.online is not None:
            result = [
                m for m in result if getattr(m.model_spec, "offline", False) != options.online
            ]

        return result

    @staticmethod
    def search_models(models: list[Any], query: str) -> list[Any]:
        """Search models by name or ID"""
        if not query:
            return models

        query = query.lower()
        return [
            m
            for m in models
            if query in m.id.lower()
            or (
                hasattr(m.model_spec, "name")
                and m.model_spec.name
                and query in m.model_spec.name.lower()
            )
        ]

    @classmethod
    def apply_all_filters(cls, models: list[Any], options: FilterOptions) -> list[Any]:
        """Apply all filters in sequence"""
        result = models

        if options.types:
            result = cls.filter_by_type(result, options.types)

        if options.capabilities:
            result = cls.filter_by_capabilities(result, options.capabilities)

        if options.traits:
            result = cls.filter_by_traits(result, options.traits)

        result = cls.filter_by_price(result, options)
        result = cls.filter_by_status(result, options)

        if options.search_query:
            result = cls.search_models(result, options.search_query)

        return result
