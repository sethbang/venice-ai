"""
Adapter to bridge Venice's RequestClassifier to the core library's ClassifierProtocol.

This module provides the translation layer between Venice SDK's enum-based
request classification and the core library's string-based interface.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # During type-checking, always import the real classes so Pylance can resolve
    # base classes and type annotations correctly.
    from adaptive_rate_limiter.protocols import (
        ClassifierProtocol,
    )
    from adaptive_rate_limiter.protocols import (
        RequestMetadata as CoreRequestMetadata,
    )

try:
    from adaptive_rate_limiter.protocols import (
        ClassifierProtocol as _ClassifierProtocolRuntime,
    )
    from adaptive_rate_limiter.protocols import (
        RequestMetadata as _CoreRequestMetadataRuntime,
    )

    _ADAPTIVE_AVAILABLE = True
except ImportError:
    _ADAPTIVE_AVAILABLE = False
    _ClassifierProtocolRuntime = object  # type: ignore[assignment,misc]
    _CoreRequestMetadataRuntime = None  # type: ignore[assignment,misc]

# At runtime use the sentinel names; at type-check time the TYPE_CHECKING block wins.
if not TYPE_CHECKING:
    ClassifierProtocol = _ClassifierProtocolRuntime
    CoreRequestMetadata = _CoreRequestMetadataRuntime

from .._queue_types import RequestMetadata as VeniceRequestMetadata
from .._request_classifier import RequestClassifier

logger = logging.getLogger(__name__)


class VeniceClassifierAdapter(ClassifierProtocol):
    """
    Adapts Venice's RequestClassifier to the core library's ClassifierProtocol.

    This adapter handles the conversion between:
    - Venice's RequestMetadata (with ResourceType enum)
    - Core library's RequestMetadata (with string-based resource_type)

    The adapter pattern allows the core library to remain API-agnostic
    while Venice SDK maintains backward compatibility with its existing
    enum-based resource types.
    """

    def __init__(self, classifier: RequestClassifier):
        """
        Initialize the adapter with a Venice RequestClassifier.

        Args:
            classifier: Venice SDK's RequestClassifier instance
        """
        if not _ADAPTIVE_AVAILABLE:
            raise ImportError(
                "adaptive-rate-limiter package is required for VeniceClassifierAdapter. "
                "Install with: pip install venice-ai[adaptive]"
            )
        self._classifier = classifier

    async def classify(self, request: dict[str, Any]) -> CoreRequestMetadata:
        """
        Classify a request and return metadata in core library format.

        This method:
        1. Calls the Venice RequestClassifier to get Venice RequestMetadata
        2. Converts the ResourceType enum to a string
        3. Returns a core library RequestMetadata object

        Args:
            request: Raw request dictionary

        Returns:
            CoreRequestMetadata with string-based resource_type
        """
        # Call Venice classifier (returns VeniceRequestMetadata with ResourceType enum)
        venice_meta: VeniceRequestMetadata = await self._classifier.classify(request)

        # Convert ResourceType enum to string
        resource_type_str = (
            venice_meta.resource_type.value
            if hasattr(venice_meta.resource_type, "value")
            else str(venice_meta.resource_type)
        )

        # Create core library metadata with string-based resource_type
        return CoreRequestMetadata(
            request_id=venice_meta.request_id,
            model_id=venice_meta.model_id,
            resource_type=resource_type_str,
            estimated_tokens=venice_meta.estimated_tokens or 0,
            priority=venice_meta.priority or 0,
            timeout=venice_meta.timeout,
            client_id=venice_meta.client_id,
            endpoint=venice_meta.endpoint,
            requires_model=venice_meta.requires_model,
        )
