"""``venice skills`` — install the SDK's bundled Claude Code skills.

The four skills ship as package data under ``venice_ai/skills/<name>/`` and are
copied into a project-local (default) or global ``.claude/skills/`` directory so
Claude Code auto-loads them.
"""

from __future__ import annotations

import shutil
from importlib.resources import as_file, files
from pathlib import Path

import click

from ..utils.output import OutputManager

_PACKAGE = "venice_ai"
_SUBDIR = "skills"


def _root():
    """Traversable for the packaged ``skills`` directory."""
    return files(_PACKAGE) / _SUBDIR


def available_skills() -> list[str]:
    """Names of bundled skills (dirs containing a ``SKILL.md``), sorted."""
    out: list[str] = []
    for entry in _root().iterdir():
        if entry.is_dir() and (entry / "SKILL.md").is_file():
            out.append(entry.name)
    return sorted(out)


def skill_description(name: str) -> str:
    """First sentence of a skill's frontmatter ``description:`` (best-effort)."""
    try:
        text: str = (_root() / name / "SKILL.md").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""
    in_frontmatter = False
    for line in text.splitlines():
        if line.strip() == "---":
            if in_frontmatter:
                break
            in_frontmatter = True
            continue
        if in_frontmatter and line.startswith("description:"):
            desc = line.split(":", 1)[1].strip()
            return (desc[:117] + "...") if len(desc) > 120 else desc
    return ""


def scope_dir(global_: bool) -> Path:
    """Target ``.claude/skills`` dir for the chosen scope."""
    base = Path.home() if global_ else Path.cwd()
    return base / ".claude" / "skills"


_SCOPE_OPTIONS = [
    click.option(
        "--global",
        "global_",
        is_flag=True,
        help="Target ~/.claude/skills/ instead of the current project.",
    ),
]


def _add_scope_options(func):
    for option in reversed(_SCOPE_OPTIONS):
        func = option(func)
    return func


@click.group("skills")
def skills() -> None:
    """Install, list, and remove the bundled Venice Claude Code skills."""


def _install_one(name: str, dest_root: Path, *, force: bool, dry_run: bool) -> str:
    src = _root() / name
    dest = dest_root / name
    if dry_run:
        return "would-overwrite" if dest.exists() else "would-install"
    if dest.exists() and not force:
        return "exists"
    if dest.exists():
        shutil.rmtree(dest)
    dest_root.mkdir(parents=True, exist_ok=True)
    with as_file(src) as src_path:
        # Exclude eval/test artifacts and caches — users get the skill, not its harness.
        shutil.copytree(
            src_path,
            dest,
            ignore=shutil.ignore_patterns("evals", "__pycache__", "*.pyc", ".DS_Store"),
        )
    return "installed"


@skills.command("list")
@_add_scope_options
def list_skills(global_: bool) -> None:
    """List the bundled skills and whether each is installed in the active scope."""
    dest_root = scope_dir(global_)
    for name in available_skills():
        marker = "installed" if (dest_root / name).exists() else "-"
        click.echo(f"{name:28} [{marker}]  {skill_description(name)}")


@skills.command("install")
@click.argument("names", nargs=-1)
@_add_scope_options
@click.option("--force", is_flag=True, help="Overwrite existing skills without prompting.")
@click.option("--dry-run", is_flag=True, help="Print what would happen; write nothing.")
def install_skills(names: tuple[str, ...], global_: bool, force: bool, dry_run: bool) -> None:
    """Install the bundled skills into .claude/skills/ (project scope by default).

    With no NAMES, installs all bundled skills. Pass one or more skill names to
    install a subset. Use --global for ~/.claude/skills/.

    Examples::

        venice skills install
        venice skills install venice-ai venice-ai-x402 --global
        venice skills install --force
    """
    bundled = available_skills()
    targets = list(names) if names else bundled
    unknown = [n for n in targets if n not in bundled]
    if unknown:
        raise click.BadParameter(
            f"unknown skill(s): {', '.join(unknown)}. Available: {', '.join(bundled)}"
        )

    dest_root = scope_dir(global_)
    for name in targets:
        result = _install_one(name, dest_root, force=force, dry_run=dry_run)
        if result == "exists":
            if click.confirm(f"  {dest_root / name} exists. Overwrite?", default=False):
                _install_one(name, dest_root, force=True, dry_run=dry_run)
                click.echo(f"  overwrote {name}")
            else:
                click.echo(f"  skipped {name}")
        elif result in ("would-install", "would-overwrite"):
            verb = "overwrite" if result == "would-overwrite" else "install"
            click.echo(f"  [dry-run] would {verb} {name} -> {dest_root / name}")
        else:
            click.echo(f"  installed {name}")

    if not dry_run:
        OutputManager.success(f"Skills target: {dest_root}")


@skills.command("uninstall")
@click.argument("names", nargs=-1)
@_add_scope_options
def uninstall_skills(names: tuple[str, ...], global_: bool) -> None:
    """Remove bundled skills from .claude/skills/ (project scope by default).

    With no NAMES, removes all bundled skills from the chosen scope.
    """
    targets = list(names) if names else available_skills()
    bundled = available_skills()
    unknown = [n for n in targets if n not in bundled]
    if unknown:
        raise click.BadParameter(
            f"unknown skill(s): {', '.join(unknown)}. Available: {', '.join(bundled)}"
        )
    dest_root = scope_dir(global_)
    removed = 0
    for name in targets:
        dest = dest_root / name
        if dest.exists():
            shutil.rmtree(dest)
            click.echo(f"  removed {name}")
            removed += 1
    OutputManager.success(f"Removed {removed} skill(s) from {dest_root}")
