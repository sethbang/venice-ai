"""Tests for the friendly ``pip install venice-py[cli]`` hint when an optional
CLI dependency (rich, questionary, etc.) is missing.

Without this hint, a fresh ``pip install venice-py`` user who runs ``venice-py``
gets a bare ``ModuleNotFoundError: No module named 'rich'`` — three failed
``pip install`` cycles before they discover the ``[cli]`` extra. Outside-agent
feedback called this out specifically.

We test the rewrap helper directly rather than mucking with ``sys.modules`` to
force a real import failure — the helper is pure, the integration point in
``venice_ai.cli.__init__`` is one line, and a unit test on the helper is both
faster and more durable than a re-import dance.
"""

from __future__ import annotations

import pytest


def test_rewrap_known_cli_dep_includes_install_hint() -> None:
    """A ModuleNotFoundError for a known CLI extra dep gets a hint added."""
    from venice_ai.cli import _maybe_hint_cli_extra

    original = ModuleNotFoundError("No module named 'rich'", name="rich")
    rewrapped = _maybe_hint_cli_extra(original)

    assert isinstance(rewrapped, ModuleNotFoundError)
    assert rewrapped.name == "rich"
    assert "venice-py[cli]" in str(rewrapped)
    assert "rich" in str(rewrapped)


@pytest.mark.parametrize(
    "missing",
    ["rich", "questionary", "click", "aiofiles", "PIL", "yaml", "dotenv", "filelock"],
)
def test_rewrap_covers_all_cli_extra_deps(missing: str) -> None:
    """Every dep declared in ``[tool.poetry.extras] cli`` should trigger the hint."""
    from venice_ai.cli import _maybe_hint_cli_extra

    rewrapped = _maybe_hint_cli_extra(
        ModuleNotFoundError(f"No module named {missing!r}", name=missing)
    )
    assert "venice-py[cli]" in str(rewrapped), (
        f"Expected install hint for CLI dep {missing!r}; got: {rewrapped}"
    )


def test_rewrap_passes_through_unrelated_module_errors() -> None:
    """A ModuleNotFoundError for a non-CLI dep must NOT be rewrapped — that
    would mask real bugs (e.g. a typo'd internal import) behind a misleading
    'install the cli extra' message."""
    from venice_ai.cli import _maybe_hint_cli_extra

    original = ModuleNotFoundError(
        "No module named 'venice_ai.does_not_exist'",
        name="venice_ai.does_not_exist",
    )
    rewrapped = _maybe_hint_cli_extra(original)
    assert rewrapped is original


def test_rewrap_passes_through_when_name_is_none() -> None:
    """Some weird import paths produce ``name=None`` — must not crash."""
    from venice_ai.cli import _maybe_hint_cli_extra

    original = ModuleNotFoundError("mystery import failure")
    rewrapped = _maybe_hint_cli_extra(original)
    assert rewrapped is original
