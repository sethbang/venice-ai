"""Venice AI provider package for adaptive-rate-limiter integration."""

__all__: list[str] = []

# Defensive: individual modules handle missing adaptive-rate-limiter gracefully,
# but we guard the import block in case future changes add other dependencies.
try:
    from .classifier_adapter import VeniceClassifierAdapter as VeniceClassifierAdapter
    from .venice_provider import VeniceProvider as VeniceProvider

    __all__.extend(["VeniceProvider", "VeniceClassifierAdapter"])
except ImportError:
    import logging

    logging.getLogger(__name__).debug(
        "adaptive-rate-limiter not installed; provider adapters unavailable"
    )
