"""
Pytest configuration and fixtures for Venice CLI tests
"""

from types import SimpleNamespace

import pytest
from rich.highlighter import NullHighlighter

from venice_ai.cli import config as _cli_config
from venice_ai.cli.utils.console import console as _cli_console
from venice_ai.cli.utils.console import disable_plain_mode


@pytest.fixture(autouse=True)
def _reset_active_config_path():
    """Reset the process-global CLI --config path between tests.

    ``venice``'s root callback writes a module-global active config path
    (``set_active_config_path``). Without a reset a test that invokes the CLI
    with ``--config`` could leak that path into a later test's
    ``get_api_key`` / ``get_base_url`` resolution under xdist. Reset before and
    after every CLI test for deterministic isolation.
    """
    _cli_config.set_active_config_path(None)
    yield
    _cli_config.set_active_config_path(None)


@pytest.fixture(autouse=True)
def _deterministic_console():
    """Pin the CLI console to ANSI-free output between tests.

    ``venice --plain`` reconfigures the process-global console in
    ``cli.utils.console``. Without a reset, whether a test sees colorized output
    depends on which tests happened to run before it in the same xdist worker,
    which makes output assertions order-dependent. CLI tests compare
    human-readable text, so colour and highlighting are pinned off and the
    plain-mode flag is restored around every test.
    """

    def _pin() -> None:
        disable_plain_mode()
        _cli_console.no_color = True
        _cli_console.highlighter = NullHighlighter()

    _pin()
    yield
    _pin()


@pytest.fixture
def mock_model_text():
    """Create a mock text model using SimpleNamespace"""
    return SimpleNamespace(
        id="test-text-model",
        type="text",
        created=1745903059,
        model_spec=SimpleNamespace(
            name="Test Text Model",
            traits=["default"],
            offline=False,
            beta=False,
            availableContextTokens=131072,
            pricing=SimpleNamespace(
                input=SimpleNamespace(usd=0.5, diem=0.5),
                output=SimpleNamespace(usd=2.0, diem=2.0),
            ),
            capabilities=SimpleNamespace(
                supportsFunctionCalling=True,
                supportsVision=False,
                supportsReasoning=True,
                supportsWebSearch=True,
                optimizedForCode=False,
                supportsResponseSchema=True,
                supportsLogProbs=True,
                quantization="fp8",
            ),
            constraints=SimpleNamespace(
                temperature=SimpleNamespace(default=0.7),
                top_p=SimpleNamespace(default=0.95),
            ),
        ),
    )


@pytest.fixture
def mock_model_image():
    """Create a mock image model using SimpleNamespace"""
    return SimpleNamespace(
        id="test-image-model",
        type="image",
        created=1726851920,
        model_spec=SimpleNamespace(
            name="Test Image Model",
            traits=["highest_quality"],
            offline=False,
            beta=False,
            pricing=SimpleNamespace(generation=SimpleNamespace(usd=0.01, diem=0.01)),
            constraints=SimpleNamespace(steps=SimpleNamespace(default=25, max=30)),
        ),
    )


@pytest.fixture
def mock_model_upscale():
    """Create a mock upscale model using SimpleNamespace"""
    return SimpleNamespace(
        id="upscaler",
        type="upscale",
        created=1744453050,
        model_spec=SimpleNamespace(
            name="Upscaler",
            traits=[],
            offline=False,
            beta=False,
            pricing=SimpleNamespace(
                generation=SimpleNamespace(usd=0.01, diem=0.01),
                upscale={
                    "2x": SimpleNamespace(usd=0.02, diem=0.02),
                    "4x": SimpleNamespace(usd=0.08, diem=0.08),
                },
            ),
        ),
    )


@pytest.fixture
def mock_model_embedding():
    """Create a mock embedding model using SimpleNamespace"""
    return SimpleNamespace(
        id="test-embedding",
        type="embedding",
        created=1741924661,
        model_spec=SimpleNamespace(
            name="Test Embedding",
            traits=[],
            offline=False,
            beta=False,
            pricing=SimpleNamespace(
                input=SimpleNamespace(usd=0.15, diem=0.15),
                output=SimpleNamespace(usd=0.6, diem=0.6),
            ),
        ),
    )


@pytest.fixture
def sample_models(mock_model_text, mock_model_image, mock_model_upscale, mock_model_embedding):
    """Collection of sample models for testing"""
    return [mock_model_text, mock_model_image, mock_model_upscale, mock_model_embedding]
