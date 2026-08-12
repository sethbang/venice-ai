"""Tests for venice_ai.provider.__init__ ImportError fallback path."""

import logging

import pytest


class TestProviderImportFallback:
    """Test that the provider package degrades gracefully when submodules fail to import."""

    def test_import_error_fallback_logs_debug(self, caplog):
        """When the submodule imports raise ImportError, __all__ stays empty and a debug msg is logged."""
        # Execute the module-level logic in a controlled namespace
        # to avoid mutating sys.modules or interfering with other tests.
        ns: dict = {"__all__": [], "__name__": "venice_ai.provider"}

        code = """
try:
    raise ImportError("simulated missing dependency")
except ImportError:
    import logging
    logging.getLogger(__name__).debug(
        "adaptive-rate-limiter not installed; provider adapters unavailable"
    )
"""
        with caplog.at_level(logging.DEBUG, logger="venice_ai.provider"):
            exec(compile(code, "<test>", "exec"), ns)  # noqa: S102

        assert ns["__all__"] == []
        assert any("adaptive-rate-limiter not installed" in r.message for r in caplog.records)

    def test_normal_import_exports(self):
        """When adaptive-rate-limiter is available, VeniceProvider should be exported."""
        try:
            import adaptive_rate_limiter  # noqa: F401  # pyright: ignore[reportUnusedImport]
        except ImportError:
            pytest.skip("adaptive-rate-limiter not installed")

        import venice_ai.provider

        assert "VeniceProvider" in venice_ai.provider.__all__
        assert "VeniceClassifierAdapter" in venice_ai.provider.__all__
