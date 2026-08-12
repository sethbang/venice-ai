"""Tests for ``tools/skills/check_skill_symbols.py``.

The symbol checker is the safety net against skill markdown referencing SDK
symbols / keyword args / attributes that do not exist. The other skill checkers
(examples-path, size, lint) never resolve a symbol against the installed
``venice_ai`` surface, so a construct like ``RetryOptions(max_retries=)`` would
otherwise pass undetected.

These tests pin BOTH halves of the contract:

* It FLAGS these known-bad constructs:
  ``RetryOptions(max_retries=3)``, ``from venice_ai import NoSuchSymbol``,
  ``client.image.create(negative_prompt="x")``, ``VoiceDetail(...).name``.
* It does NOT flag their valid counterparts: ``RetryOptions(max_attempts=3)``,
  a real import, a real method kwarg, and an arbitrary kwarg on an
  ``extra='allow'`` pydantic model.

Fixtures are written to a temp skills tree and the checker is pointed at it via
monkeypatched ``SKILLS_ROOT`` / ``REPO_ROOT`` (same pattern as
``test_check_skill_examples.py``).
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = REPO_ROOT / "tools" / "skills" / "check_skill_symbols.py"
_MODULE_NAME = "_check_skill_symbols"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so that the module's own
    # `@dataclass` (with `from __future__ import annotations`) can resolve its
    # string annotations via `sys.modules[__name__]`. Without this, dataclass
    # processing raises AttributeError on a None module.
    sys.modules[_MODULE_NAME] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(_MODULE_NAME, None)
        raise
    return mod


def _write_skill(tmp_path: Path, body: str) -> Path:
    """Create a temp skills tree containing one SKILL.md with ``body`` as a
    fenced python block, and return the skills root."""
    skills_root = tmp_path / "tools" / "skills"
    skill_dir = skills_root / "venice-ai-demo"
    skill_dir.mkdir(parents=True)
    md = "# Demo\n\n```python\n" + textwrap.dedent(body).strip("\n") + "\n```\n"
    (skill_dir / "SKILL.md").write_text(md, encoding="utf-8")
    return skills_root


def _run(
    mod: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    skills_root: Path,
) -> int:
    # REPO_ROOT is used both for sys.path (already applied at import) and for
    # the relative_to() in output; pointing it at tmp_path keeps the printed
    # paths relative and avoids touching the real repo tree.
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "SKILLS_ROOT", skills_root)
    return int(mod.main())


# ---------------------------------------------------------------------------
# MUST FLAG: the four constructs the audit fixed by hand.
# ---------------------------------------------------------------------------
def test_flags_constructor_bad_kwarg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # RetryOptions' real field is `max_attempts`, not `max_retries`.
    skills_root = _write_skill(
        tmp_path,
        """
        from venice_ai.middleware.retry import RetryOptions

        opts = RetryOptions(max_retries=3)
        """,
    )
    assert _run(_load_module(), monkeypatch, tmp_path, skills_root) == 1
    out = capsys.readouterr().out
    assert "max_retries" in out
    assert "RetryOptions" in out


def test_flags_nonexistent_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    skills_root = _write_skill(
        tmp_path,
        """
        from venice_ai import NoSuchSymbol
        """,
    )
    assert _run(_load_module(), monkeypatch, tmp_path, skills_root) == 1
    out = capsys.readouterr().out
    assert "NoSuchSymbol" in out


def test_flags_method_removed_kwarg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `negative_prompt` was removed from image.create.
    skills_root = _write_skill(
        tmp_path,
        """
        from venice_ai import VeniceClient

        client = VeniceClient(api_key="x")
        client.image.create(prompt="a cat", negative_prompt="dog")
        """,
    )
    assert _run(_load_module(), monkeypatch, tmp_path, skills_root) == 1
    out = capsys.readouterr().out
    assert "negative_prompt" in out
    assert "image.create" in out


def test_flags_bad_attribute_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Use a strict (extra="ignore") request model: EmbeddingsRequest has no
    # `.nonexistent_attr`. (Response models now use extra="allow" for
    # forward-compat, which by design exempts them from attribute checking —
    # the checker only flags unknown attributes on non-allow models.)
    skills_root = _write_skill(
        tmp_path,
        """
        from venice_ai.types import EmbeddingsRequest

        r = EmbeddingsRequest(input="hi", model="m")
        print(r.nonexistent_attr)
        """,
    )
    assert _run(_load_module(), monkeypatch, tmp_path, skills_root) == 1
    out = capsys.readouterr().out
    assert "EmbeddingsRequest" in out
    assert "nonexistent_attr" in out


# ---------------------------------------------------------------------------
# MUST NOT FLAG: the valid counterparts.
# ---------------------------------------------------------------------------
def test_accepts_constructor_good_kwarg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skills_root = _write_skill(
        tmp_path,
        """
        from venice_ai.middleware.retry import RetryOptions

        opts = RetryOptions(max_attempts=3)
        """,
    )
    assert _run(_load_module(), monkeypatch, tmp_path, skills_root) == 0


def test_accepts_real_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skills_root = _write_skill(
        tmp_path,
        """
        from venice_ai import VeniceClient, RetryOptions
        from venice_ai.types.api import UserMessage
        """,
    )
    assert _run(_load_module(), monkeypatch, tmp_path, skills_root) == 0


def test_accepts_real_method_kwarg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skills_root = _write_skill(
        tmp_path,
        """
        from venice_ai import VeniceClient

        client = VeniceClient(api_key="x")
        client.image.create(model="m", prompt="a cat", width=1024, height=1024)
        """,
    )
    assert _run(_load_module(), monkeypatch, tmp_path, skills_root) == 0


def test_accepts_extra_allow_arbitrary_kwarg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # RequestEcho has model_config extra='allow' — arbitrary kwargs are valid
    # and MUST NOT be flagged.
    from venice_ai.core.models.common import RequestEcho  # noqa: F401  (guards the assumption)

    assert RequestEcho.model_config.get("extra") == "allow"
    skills_root = _write_skill(
        tmp_path,
        """
        from venice_ai.core.models.common import RequestEcho

        echo = RequestEcho(model="m", totally_made_up_field=123)
        """,
    )
    assert _run(_load_module(), monkeypatch, tmp_path, skills_root) == 0


def test_accepts_kwargs_method_arbitrary_kwarg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # chat.completions.create takes **kwargs — passing an unusual kwarg must
    # NOT be flagged (the checker can't and shouldn't reason about **kwargs).
    skills_root = _write_skill(
        tmp_path,
        """
        from venice_ai import VeniceClient

        client = VeniceClient(api_key="x")
        client.chat.completions.create(model="m", messages=[], some_passthrough=1)
        """,
    )
    assert _run(_load_module(), monkeypatch, tmp_path, skills_root) == 0
