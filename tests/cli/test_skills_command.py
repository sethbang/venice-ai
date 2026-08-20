"""Tests for the ``venice-py skills`` CLI command group."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from venice_ai.cli.cli import cli
from venice_ai.cli.commands.skills import (
    available_skills,
    scope_dir,
    skill_description,
)

EXPECTED = {
    "venice-py",
    "venice-py-multimodal",
    "venice-py-production",
    "venice-py-x402",
}


def test_available_skills_lists_the_four_bundled_skills() -> None:
    assert set(available_skills()) == EXPECTED


def test_skill_description_is_nonempty_for_each() -> None:
    for name in available_skills():
        assert skill_description(name).strip()


def test_scope_dir_project_is_cwd_relative(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert scope_dir(False) == tmp_path / ".claude" / "skills"


def test_scope_dir_global_is_home_relative(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert scope_dir(True) == tmp_path / ".claude" / "skills"


def test_skills_list_shows_all_four() -> None:
    result = CliRunner().invoke(cli, ["skills", "list"])
    assert result.exit_code == 0, result.output
    for name in EXPECTED:
        assert name in result.output


def test_install_project_scope_copies_skill_trees(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["skills", "install"])
    assert result.exit_code == 0, result.output
    for name in EXPECTED:
        assert (tmp_path / ".claude" / "skills" / name / "SKILL.md").is_file()


def test_install_named_subset_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["skills", "install", "venice-py"])
    assert result.exit_code == 0, result.output
    base = tmp_path / ".claude" / "skills"
    assert (base / "venice-py" / "SKILL.md").is_file()
    assert not (base / "venice-py-x402").exists()


def test_install_unknown_skill_errors(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["skills", "install", "nope"])
    assert result.exit_code != 0
    assert "nope" in result.output


def test_install_global_scope(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)  # keep project dir clean
    result = CliRunner().invoke(cli, ["skills", "install", "venice-py", "--global"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / ".claude" / "skills" / "venice-py" / "SKILL.md").is_file()


def test_install_dry_run_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["skills", "install", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".claude" / "skills").exists()


def test_install_existing_prompts_and_skips_on_no(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["skills", "install", "venice-py"])
    sentinel = tmp_path / ".claude" / "skills" / "venice-py" / "MARKER"
    sentinel.write_text("keep")
    result = runner.invoke(cli, ["skills", "install", "venice-py"], input="n\n")
    assert result.exit_code == 0, result.output
    assert sentinel.exists()  # skipped, not overwritten


def test_install_force_overwrites(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["skills", "install", "venice-py"])
    sentinel = tmp_path / ".claude" / "skills" / "venice-py" / "MARKER"
    sentinel.write_text("stale")
    result = runner.invoke(cli, ["skills", "install", "venice-py", "--force"])
    assert result.exit_code == 0, result.output
    assert not sentinel.exists()  # overwritten clean


def test_install_excludes_eval_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    CliRunner().invoke(cli, ["skills", "install", "venice-py"])
    installed = tmp_path / ".claude" / "skills" / "venice-py"
    assert (installed / "SKILL.md").is_file()
    assert not (installed / "evals").exists()  # eval harness must not ship to users


def test_uninstall_removes_installed_skill(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["skills", "install", "venice-py"])
    base = tmp_path / ".claude" / "skills"
    assert (base / "venice-py").exists()
    result = runner.invoke(cli, ["skills", "uninstall", "venice-py"])
    assert result.exit_code == 0, result.output
    assert not (base / "venice-py").exists()


def test_uninstall_all_when_no_names(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["skills", "install"])
    result = runner.invoke(cli, ["skills", "uninstall"])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".claude" / "skills" / "venice-py").exists()


def test_uninstall_missing_is_noop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["skills", "uninstall", "venice-py"])
    assert result.exit_code == 0, result.output


def test_uninstall_rejects_path_traversal_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["skills", "uninstall", "../../etc"])
    assert result.exit_code != 0
    assert "unknown skill" in result.output


def test_install_dry_run_after_install_does_not_prompt_or_fail(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["skills", "install", "venice-py"])  # now installed
    # No input= provided: if it tried to prompt, click would abort with exit 1.
    result = runner.invoke(cli, ["skills", "install", "venice-py", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "would overwrite" in result.output
    # still exactly the originally-installed content, nothing rewritten/removed
    assert (tmp_path / ".claude" / "skills" / "venice-py" / "SKILL.md").is_file()


# ---------------------------------------------------------------------------
# Pre-rename installs. A user who installed before the distribution was renamed
# has venice-ai* directories on disk; both generations would trigger in Claude
# Code until they are cleared.
# ---------------------------------------------------------------------------

LEGACY = {
    "venice-ai",
    "venice-ai-multimodal",
    "venice-ai-production",
    "venice-ai-x402",
}

_PROVENANCE = "> _Unofficial, community-maintained — not affiliated with or endorsed by Venice AI._"


def _plant_legacy(base: Path, name: str, *, provenance: bool = True) -> Path:
    """Write a pre-rename skill directory the way an old install left it."""
    skill_dir = base / name
    skill_dir.mkdir(parents=True)
    body = f"---\nname: {name}\ndescription: legacy copy.\n---\n\n# Legacy\n"
    if provenance:
        body += f"\n{_PROVENANCE}\n"
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")
    return skill_dir


def test_install_removes_superseded_legacy_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    base = tmp_path / ".claude" / "skills"
    legacy = _plant_legacy(base, "venice-ai")
    result = CliRunner().invoke(cli, ["skills", "install", "venice-py"])
    assert result.exit_code == 0, result.output
    assert not legacy.exists()
    assert (base / "venice-py" / "SKILL.md").is_file()


def test_install_all_clears_every_legacy_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    base = tmp_path / ".claude" / "skills"
    for name in sorted(LEGACY):
        _plant_legacy(base, name)
    result = CliRunner().invoke(cli, ["skills", "install"])
    assert result.exit_code == 0, result.output
    assert {p.name for p in base.iterdir()} == EXPECTED


def test_install_subset_leaves_untargeted_legacy_dirs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    base = tmp_path / ".claude" / "skills"
    _plant_legacy(base, "venice-ai")
    kept = _plant_legacy(base, "venice-ai-x402")
    result = CliRunner().invoke(cli, ["skills", "install", "venice-py"])
    assert result.exit_code == 0, result.output
    assert not (base / "venice-ai").exists()
    assert kept.exists()  # its replacement was not installed, so it is not superseded yet


def test_install_keeps_same_named_dir_that_is_not_ours(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    base = tmp_path / ".claude" / "skills"
    foreign = _plant_legacy(base, "venice-ai", provenance=False)
    (foreign / "KEEP").write_text("someone else's skill", encoding="utf-8")
    result = CliRunner().invoke(cli, ["skills", "install", "venice-py"])
    assert result.exit_code == 0, result.output
    assert (foreign / "KEEP").exists()


def test_install_keeps_same_named_dir_without_skill_md(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    base = tmp_path / ".claude" / "skills"
    foreign = base / "venice-ai"
    foreign.mkdir(parents=True)
    (foreign / "notes.txt").write_text("unrelated", encoding="utf-8")
    result = CliRunner().invoke(cli, ["skills", "install", "venice-py"])
    assert result.exit_code == 0, result.output
    assert (foreign / "notes.txt").exists()


def test_install_leaves_legacy_symlink_alone(tmp_path: Path, monkeypatch) -> None:
    # `tools/skills/install.sh --symlink` setups predate the rename; rmtree
    # cannot remove a symlink, so the sweep must skip it rather than crash.
    monkeypatch.chdir(tmp_path)
    base = tmp_path / ".claude" / "skills"
    base.mkdir(parents=True)
    target = _plant_legacy(tmp_path / "elsewhere", "venice-ai")
    link = base / "venice-ai"
    link.symlink_to(target, target_is_directory=True)
    result = CliRunner().invoke(cli, ["skills", "install", "venice-py"])
    assert result.exit_code == 0, result.output
    assert link.is_symlink()
    assert (target / "SKILL.md").is_file()


def test_install_dry_run_reports_legacy_without_removing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    base = tmp_path / ".claude" / "skills"
    legacy = _plant_legacy(base, "venice-ai")
    result = CliRunner().invoke(cli, ["skills", "install", "venice-py", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "venice-ai" in result.output
    assert legacy.exists()


def test_uninstall_removes_superseded_legacy_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["skills", "install", "venice-py"])
    base = tmp_path / ".claude" / "skills"
    legacy = _plant_legacy(base, "venice-ai")
    result = runner.invoke(cli, ["skills", "uninstall", "venice-py"])
    assert result.exit_code == 0, result.output
    assert not legacy.exists()
    assert not (base / "venice-py").exists()


def test_uninstall_keeps_same_named_dir_that_is_not_ours(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    base = tmp_path / ".claude" / "skills"
    foreign = _plant_legacy(base, "venice-ai", provenance=False)
    result = CliRunner().invoke(cli, ["skills", "uninstall", "venice-py"])
    assert result.exit_code == 0, result.output
    assert foreign.exists()


def test_list_flags_superseded_install(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _plant_legacy(tmp_path / ".claude" / "skills", "venice-ai")
    result = CliRunner().invoke(cli, ["skills", "list"])
    assert result.exit_code == 0, result.output
    assert "Superseded" in result.output
    assert "venice-ai" in result.output
    # Detection only — list must never delete.
    assert (tmp_path / ".claude" / "skills" / "venice-ai").exists()
