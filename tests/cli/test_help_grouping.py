"""Tests for the curated ``venice --help`` command-group layout."""

from __future__ import annotations

from click.testing import CliRunner

from venice_ai.cli.cli import cli


def test_help_shows_grouped_sections() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    out = result.output
    # Each curated group should render as its own section header.
    for section in (
        "Commands (Generate)",
        "Commands (Discover)",
        "Commands (Account)",
        "Commands (Develop)",
    ):
        assert section in out, f"Missing section: {section}\n\n{out}"


def test_help_lists_lint_under_develop() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    # ``lint`` should appear *after* the Develop header.
    out = result.output
    develop_idx = out.find("Commands (Develop)")
    assert develop_idx != -1
    lint_idx = out.find("lint", develop_idx)
    assert lint_idx != -1, "lint should appear in the Develop section"


def test_help_lists_image_under_generate() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    out = result.output
    generate_idx = out.find("Commands (Generate)")
    assert generate_idx != -1
    image_idx = out.find("image", generate_idx)
    assert image_idx != -1


def test_help_top_text_mentions_featured_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    out = result.output
    assert "venice models resolve" in out
    assert "venice lint" in out
