"""
Sorting functionality for Venice AI models
"""

from typing import Any


class ModelSorter:
    """Handles model sorting"""

    @staticmethod
    def sort_models(models: list[Any], criterion: str) -> list[Any]:
        """Sort models by specified criterion"""
        if not models:
            return models

        if criterion == "name":
            return sorted(
                models,
                key=lambda m: (
                    getattr(m.model_spec, "name", m.id) if hasattr(m.model_spec, "name") else m.id
                ).lower(),
            )

        elif criterion == "id":
            return sorted(models, key=lambda m: m.id.lower())

        elif criterion == "price-asc":

            def get_input_price(m):
                if (
                    hasattr(m.model_spec, "pricing")
                    and m.model_spec.pricing
                    and hasattr(m.model_spec.pricing, "input")
                    and m.model_spec.pricing.input
                ):
                    return (
                        m.model_spec.pricing.input.usd
                        if hasattr(m.model_spec.pricing.input, "usd")
                        else float("inf")
                    )
                return float("inf")

            return sorted(models, key=get_input_price)

        elif criterion == "price-desc":

            def get_input_price_desc(m):
                if (
                    hasattr(m.model_spec, "pricing")
                    and m.model_spec.pricing
                    and hasattr(m.model_spec.pricing, "input")
                    and m.model_spec.pricing.input
                ):
                    return (
                        m.model_spec.pricing.input.usd
                        if hasattr(m.model_spec.pricing.input, "usd")
                        else 0
                    )
                return 0

            return sorted(models, key=get_input_price_desc, reverse=True)

        elif criterion == "context":

            def get_context(m):
                return (
                    getattr(m.model_spec, "availableContextTokens", 0)
                    if hasattr(m.model_spec, "availableContextTokens")
                    else 0
                )

            return sorted(models, key=get_context, reverse=True)

        elif criterion == "created":
            return sorted(models, key=lambda m: getattr(m, "created", 0), reverse=True)

        else:
            # Default: sort by ID
            return sorted(models, key=lambda m: m.id.lower())
