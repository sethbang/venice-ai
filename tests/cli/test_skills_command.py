"""Tests for the ``venice skills`` CLI command group."""

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
    "venice-ai",
    "venice-ai-multimodal",
    "venice-ai-production",
    "venice-ai-x402",
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
    result = CliRunner().invoke(cli, ["skills", "install", "venice-ai"])
    assert result.exit_code == 0, result.output
    base = tmp_path / ".claude" / "skills"
    assert (base / "venice-ai" / "SKILL.md").is_file()
    assert not (base / "venice-ai-x402").exists()


def test_install_unknown_skill_errors(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["skills", "install", "nope"])
    assert result.exit_code != 0
    assert "nope" in result.output


def test_install_global_scope(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)  # keep project dir clean
    result = CliRunner().invoke(cli, ["skills", "install", "venice-ai", "--global"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / ".claude" / "skills" / "venice-ai" / "SKILL.md").is_file()


def test_install_dry_run_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["skills", "install", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".claude" / "skills").exists()


def test_install_existing_prompts_and_skips_on_no(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["skills", "install", "venice-ai"])
    sentinel = tmp_path / ".claude" / "skills" / "venice-ai" / "MARKER"
    sentinel.write_text("keep")
    result = runner.invoke(cli, ["skills", "install", "venice-ai"], input="n\n")
    assert result.exit_code == 0, result.output
    assert sentinel.exists()  # skipped, not overwritten


def test_install_force_overwrites(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["skills", "install", "venice-ai"])
    sentinel = tmp_path / ".claude" / "skills" / "venice-ai" / "MARKER"
    sentinel.write_text("stale")
    result = runner.invoke(cli, ["skills", "install", "venice-ai", "--force"])
    assert result.exit_code == 0, result.output
    assert not sentinel.exists()  # overwritten clean


def test_install_excludes_eval_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    CliRunner().invoke(cli, ["skills", "install", "venice-ai"])
    installed = tmp_path / ".claude" / "skills" / "venice-ai"
    assert (installed / "SKILL.md").is_file()
    assert not (installed / "evals").exists()  # eval harness must not ship to users


def test_uninstall_removes_installed_skill(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["skills", "install", "venice-ai"])
    base = tmp_path / ".claude" / "skills"
    assert (base / "venice-ai").exists()
    result = runner.invoke(cli, ["skills", "uninstall", "venice-ai"])
    assert result.exit_code == 0, result.output
    assert not (base / "venice-ai").exists()


def test_uninstall_all_when_no_names(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["skills", "install"])
    result = runner.invoke(cli, ["skills", "uninstall"])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".claude" / "skills" / "venice-ai").exists()


def test_uninstall_missing_is_noop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["skills", "uninstall", "venice-ai"])
    assert result.exit_code == 0, result.output


def test_uninstall_rejects_path_traversal_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["skills", "uninstall", "../../etc"])
    assert result.exit_code != 0
    assert "unknown skill" in result.output


def test_install_dry_run_after_install_does_not_prompt_or_fail(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["skills", "install", "venice-ai"])  # now installed
    # No input= provided: if it tried to prompt, click would abort with exit 1.
    result = runner.invoke(cli, ["skills", "install", "venice-ai", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "would overwrite" in result.output
    # still exactly the originally-installed content, nothing rewritten/removed
    assert (tmp_path / ".claude" / "skills" / "venice-ai" / "SKILL.md").is_file()
