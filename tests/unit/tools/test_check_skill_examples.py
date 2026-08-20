"""Tests for ``tools/skills/check_skill_examples.py``.

Focus on the ``## Scripts`` validation: ``scripts/<file>.py`` references under a
``## Scripts`` heading must resolve against the owning skill's ``scripts/``
directory. (The old implementation only validated ``examples/`` refs and silently
ignored Scripts refs, so a dangling script reference would have passed CI.)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = REPO_ROOT / "tools" / "skills" / "check_skill_examples.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_check_skill_examples", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_skill_tree(
    tmp_path: Path,
    *,
    skill_md: str,
    script_files: tuple[str, ...] = (),
) -> tuple[Path, Path]:
    """Create a minimal skills tree and return (skills_root, examples_root)."""
    skills_root = tmp_path / "tools" / "skills"
    examples_root = tmp_path / "examples"
    examples_root.mkdir(parents=True)

    skill_dir = skills_root / "venice-py-demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    for name in script_files:
        (scripts_dir / name).write_text("# stub\n", encoding="utf-8")

    return skills_root, examples_root


def _run(
    mod: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    skills_root: Path,
    examples_root: Path,
) -> int:
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "SKILLS_ROOT", skills_root)
    monkeypatch.setattr(mod, "EXAMPLES_ROOT", examples_root)
    result: int = mod.main()
    return result


def test_existing_script_reference_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    md = "# Demo\n\n## Scripts\n\n- `scripts/topup.py` — does a thing.\n"
    skills_root, examples_root = _build_skill_tree(
        tmp_path, skill_md=md, script_files=("topup.py",)
    )
    assert _run(mod, monkeypatch, tmp_path, skills_root, examples_root) == 0


def test_missing_script_reference_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    md = "# Demo\n\n## Scripts\n\n- `scripts/does_not_exist.py` — dangling ref.\n"
    skills_root, examples_root = _build_skill_tree(tmp_path, skill_md=md)
    # The new Scripts validation must catch the dangling reference.
    assert _run(mod, monkeypatch, tmp_path, skills_root, examples_root) == 1


def test_script_reference_with_trailing_args_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `scripts/lint.py <path>` — trailing args inside the backticks must not
    # break path extraction.
    mod = _load_module()
    md = "# Demo\n\n## Scripts\n\n- `scripts/lint.py <path>` — scans a dir.\n"
    skills_root, examples_root = _build_skill_tree(tmp_path, skill_md=md, script_files=("lint.py",))
    assert _run(mod, monkeypatch, tmp_path, skills_root, examples_root) == 0


def test_script_reference_outside_scripts_heading_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A `scripts/...` ref in prose (NOT under a "## Scripts" heading) is not
    # validated, matching the heading-scoped contract.
    mod = _load_module()
    md = "# Demo\n\nSee `scripts/ghost.py` somewhere.\n\n## Other\n\nnope.\n"
    skills_root, examples_root = _build_skill_tree(tmp_path, skill_md=md)
    assert _run(mod, monkeypatch, tmp_path, skills_root, examples_root) == 0


def test_empty_skills_root_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A skill-dir glob that matches nothing must fail loudly. Reporting "0 skills
    # checked" and exiting 0 would let a renamed skills tree pass CI unvalidated.
    mod = _load_module()
    skills_root = tmp_path / "src" / "venice_ai" / "skills"
    skills_root.mkdir(parents=True)
    examples_root = tmp_path / "examples"
    examples_root.mkdir()
    assert _run(mod, monkeypatch, tmp_path, skills_root, examples_root) == 1
