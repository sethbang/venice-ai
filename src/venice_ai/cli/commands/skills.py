"""``venice-py skills`` — install the SDK's bundled Claude Code skills.

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

# Directory each bundled skill occupied before the distribution was renamed.
# Both sets trigger in Claude Code if a pre-rename install is left on disk, so
# installing or uninstalling a skill also clears its superseded counterpart.
_LEGACY_DIR_FOR = {
    "venice-py": "venice-ai",
    "venice-py-multimodal": "venice-ai-multimodal",
    "venice-py-production": "venice-ai-production",
    "venice-py-x402": "venice-ai-x402",
}

# Provenance line carried by every bundled SKILL.md. Paired with a matching
# frontmatter ``name:`` it is what tells a superseded copy of ours apart from an
# unrelated user skill that happens to occupy the same directory name.
_PROVENANCE_MARKER = "Unofficial, community-maintained"


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


def _frontmatter_value(text: str, field: str) -> str:
    """Value of a ``field:`` line in a SKILL.md's frontmatter, ``""`` if absent."""
    in_frontmatter = False
    prefix = f"{field}:"
    for line in text.splitlines():
        if line.strip() == "---":
            if in_frontmatter:
                break
            in_frontmatter = True
            continue
        if in_frontmatter and line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def skill_description(name: str) -> str:
    """First sentence of a skill's frontmatter ``description:`` (best-effort)."""
    try:
        text: str = (_root() / name / "SKILL.md").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""
    desc = _frontmatter_value(text, "description")
    return (desc[:117] + "...") if len(desc) > 120 else desc


def _is_superseded_install(path: Path, expected_name: str) -> bool:
    """True when ``path`` holds a pre-rename install of one of our own skills.

    Requires a real directory — never a symlink, which ``rmtree`` cannot remove
    and which a developer may have pointed at a source tree — containing a
    ``SKILL.md`` whose frontmatter ``name:`` is ``expected_name`` and which
    carries the bundled skills' provenance line. An unrelated user skill sitting
    at the same directory name fails both checks and is left alone.
    """
    if path.is_symlink() or not path.is_dir():
        return False
    skill_md = path / "SKILL.md"
    if not skill_md.is_file():
        return False
    try:
        text = skill_md.read_text(encoding="utf-8")
    except OSError:
        return False
    return _frontmatter_value(text, "name") == expected_name and _PROVENANCE_MARKER in text


def superseded_installs(names: list[str], dest_root: Path) -> list[str]:
    """Pre-rename directories in ``dest_root`` belonging to the named skills."""
    found: list[str] = []
    for name in names:
        legacy = _LEGACY_DIR_FOR.get(name)
        if legacy is not None and _is_superseded_install(dest_root / legacy, legacy):
            found.append(legacy)
    return found


def _remove_superseded(names: list[str], dest_root: Path, *, dry_run: bool) -> int:
    """Delete the named skills' pre-rename directories. Returns the count removed."""
    removed = 0
    for legacy in superseded_installs(names, dest_root):
        path = dest_root / legacy
        if dry_run:
            click.echo(f"  [dry-run] would remove superseded {legacy} -> {path}")
            continue
        shutil.rmtree(path)
        click.echo(f"  removed superseded {legacy}")
        removed += 1
    return removed


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
    bundled = available_skills()
    for name in bundled:
        marker = "installed" if (dest_root / name).exists() else "-"
        click.echo(f"{name:28} [{marker}]  {skill_description(name)}")

    superseded = superseded_installs(bundled, dest_root)
    if superseded:
        click.echo()
        click.echo(
            f"Superseded skill(s) under {dest_root}: {', '.join(superseded)}. "
            "They trigger alongside the current ones; 'venice-py skills install' clears them."
        )


@skills.command("install")
@click.argument("names", nargs=-1)
@_add_scope_options
@click.option("--force", is_flag=True, help="Overwrite existing skills without prompting.")
@click.option("--dry-run", is_flag=True, help="Print what would happen; write nothing.")
def install_skills(names: tuple[str, ...], global_: bool, force: bool, dry_run: bool) -> None:
    """Install the bundled skills into .claude/skills/ (project scope by default).

    With no NAMES, installs all bundled skills. Pass one or more skill names to
    install a subset. Use --global for ~/.claude/skills/.

    Any pre-rename copy of a targeted skill is removed from the same directory,
    so the two generations never trigger side by side.

    Examples::

        venice-py skills install
        venice-py skills install venice-py venice-py-x402 --global
        venice-py skills install --force
    """
    bundled = available_skills()
    targets = list(names) if names else bundled
    unknown = [n for n in targets if n not in bundled]
    if unknown:
        raise click.BadParameter(
            f"unknown skill(s): {', '.join(unknown)}. Available: {', '.join(bundled)}"
        )

    dest_root = scope_dir(global_)
    # Command level, not inside _install_one: an accepted overwrite prompt calls
    # that helper a second time, which would repeat the sweep.
    _remove_superseded(targets, dest_root, dry_run=dry_run)
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

    With no NAMES, removes all bundled skills from the chosen scope. Pre-rename
    copies of the named skills are removed alongside them.
    """
    targets = list(names) if names else available_skills()
    bundled = available_skills()
    unknown = [n for n in targets if n not in bundled]
    if unknown:
        raise click.BadParameter(
            f"unknown skill(s): {', '.join(unknown)}. Available: {', '.join(bundled)}"
        )
    dest_root = scope_dir(global_)
    removed = _remove_superseded(targets, dest_root, dry_run=False)
    for name in targets:
        dest = dest_root / name
        if dest.exists():
            shutil.rmtree(dest)
            click.echo(f"  removed {name}")
            removed += 1
    OutputManager.success(f"Removed {removed} skill(s) from {dest_root}")
