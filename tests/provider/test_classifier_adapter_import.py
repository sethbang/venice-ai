"""Tests for VeniceClassifierAdapter ImportError path."""

from unittest.mock import MagicMock

import pytest


class TestClassifierAdapterImportError:
    """Test that VeniceClassifierAdapter raises ImportError when adaptive-rate-limiter is missing."""

    def test_raises_import_error_when_not_available(self):
        """When _ADAPTIVE_AVAILABLE is False, __init__ should raise ImportError."""
        # We need to test the runtime behavior when adaptive-rate-limiter is not installed.
        # Patch the module-level flag directly.
        from venice_ai.provider import classifier_adapter

        original = classifier_adapter._ADAPTIVE_AVAILABLE
        try:
            classifier_adapter._ADAPTIVE_AVAILABLE = False

            mock_classifier = MagicMock()
            with pytest.raises(ImportError, match="adaptive-rate-limiter"):
                classifier_adapter.VeniceClassifierAdapter(classifier=mock_classifier)
        finally:
            classifier_adapter._ADAPTIVE_AVAILABLE = original
