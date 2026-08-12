"""
Venice AI CLI - Command Line Interface for Venice AI

A modern, feature-rich CLI for interacting with Venice AI services.
"""

from __future__ import annotations

from venice_ai import __version__

__author__ = "Venice AI CLI Contributors"


# Optional runtime deps declared under ``[tool.poetry.extras] cli`` in
# pyproject.toml. Kept here so the install-hint stays in sync if the extra
# changes; tests assert one entry per dep.
_CLI_OPTIONAL_DEPS: frozenset[str] = frozenset(
    {
        "rich",
        "questionary",
        "click",
        "aiofiles",
        "PIL",  # Pillow imports as PIL
        "yaml",  # pyyaml imports as yaml
        "dotenv",  # python-dotenv imports as dotenv
        "filelock",
    }
)


def _maybe_hint_cli_extra(exc: ModuleNotFoundError) -> ModuleNotFoundError:
    """Return a friendlier ``ModuleNotFoundError`` if ``exc.name`` is one of
    the CLI optional deps; otherwise return ``exc`` unchanged.

    Without this, a fresh ``pip install venice-ai`` user who runs ``venice``
    sees ``No module named 'rich'`` and burns three install cycles before
    finding the ``[cli]`` extra. We only rewrap names we recognize so a real
    bug (typo'd internal import, broken third-party dep) is not masked.
    """
    if exc.name is None or exc.name not in _CLI_OPTIONAL_DEPS:
        return exc
    return ModuleNotFoundError(
        f"Missing CLI dependency '{exc.name}'. The Venice CLI ships as an "
        f"optional install extra to keep the core SDK lean.\n\n"
        f"    pip install 'venice-ai[cli]'\n",
        name=exc.name,
    )


try:
    from .cli import cli
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via subprocess in CI
    raise _maybe_hint_cli_extra(exc) from exc

__all__ = ["cli", "__version__"]
