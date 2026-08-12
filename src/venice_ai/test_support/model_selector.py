"""
Model selector re-exports for Venice AI test support.

This module provides convenient imports for model selection utilities,
delegating to the canonical location in the models package.

The types are exported from:
- ``venice_ai.models.selection``: DynamicModelSelector, create_model_selector

Example usage:
    >>> from venice_ai.test_support.model_selector import DynamicModelSelector
    >>> selector = DynamicModelSelector(client, cache_ttl=300.0)
    >>> model = await selector.select_chat_model()
"""

# Re-export model selection utilities from canonical location
from venice_ai.models.selection import (
    DynamicModelSelector,
    ModelCache,
    create_model_selector,
    get_chat_model,
    get_embedding_model,
    get_multiple_models,
)

__all__ = [
    "DynamicModelSelector",
    "create_model_selector",
    "get_chat_model",
    "get_embedding_model",
    "get_multiple_models",
    "ModelCache",
]
