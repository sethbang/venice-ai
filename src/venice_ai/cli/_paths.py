"""Per-user data directory for the Venice AI CLI.

The CLI keeps its config file, saved chat transcripts and image presets under
``~/.venice-py/``. That directory used to be ``~/.venice/``, which collides with
the official Venice CLI — a separate Node tool whose binary is ``venice``. Both
can be installed side by side, so this one moved out of the way.

Because the directory holds user-authored content — presets someone tuned by
hand, months of conversation history — the move is backed by a one-time copy
from the old location, performed lazily the first time anything actually reads
the directory (see :func:`ensure_migrated`).
"""

from __future__ import annotations

import errno
import os
import shutil
import stat
import sys
import tempfile
import threading
from pathlib import Path

APP_DIR_NAME = ".venice-py"
LEGACY_APP_DIR_NAME = ".venice"

# Subpaths this CLI owns, each with the permission mask its copy is narrowed to
# (``None`` keeps whatever the source had). The legacy directory may now hold
# files belonging to the official ``venice`` CLI, so the migration copies these
# by name rather than cloning the whole tree.
_OWNED_FILES: tuple[tuple[str, int | None], ...] = (
    ("config.yaml", 0o600),  # may hold a plaintext API key
)
_OWNED_DIRS: tuple[tuple[str, int | None], ...] = (
    ("conversations", 0o700),  # transcripts contain prompt / response text
    ("presets", None),
)

_migration_lock = threading.Lock()
_migration_done = False


def _home() -> Path:
    """Return the user's home directory.

    Indirected through a function so tests can point the migration at a
    temporary home instead of reaching the real one.
    """
    return Path.home()


def app_dir() -> Path:
    """Return ``~/.venice-py``."""
    return _home() / APP_DIR_NAME


def legacy_app_dir() -> Path:
    """Return the pre-rename ``~/.venice``."""
    return _home() / LEGACY_APP_DIR_NAME


def _echo_stderr(message: str) -> None:
    """Write a message to stderr, preferring click when it is importable."""
    try:
        import click
    except ModuleNotFoundError:  # pragma: no cover - click is a ``[cli]`` extra
        print(message, file=sys.stderr)
    else:
        click.echo(message, err=True)


def _restrict(path: Path, mask: int | None) -> None:
    """Narrow ``path``'s permission bits to ``mask``, never widening them."""
    if mask is None:
        return
    current = stat.S_IMODE(path.stat().st_mode)
    narrowed = current & mask
    if narrowed != current:
        os.chmod(path, narrowed)


def _stage_legacy_contents(legacy: Path, stage: Path) -> bool:
    """Copy the subpaths we own from ``legacy`` into ``stage``.

    Returns ``True`` if anything was copied. ``copy2`` / ``copytree`` carry the
    source's mode bits across; :func:`_restrict` then clamps the sensitive ones.
    """
    copied = False

    for name, mask in _OWNED_FILES:
        source = legacy / name
        if not source.is_file():
            continue
        destination = stage / name
        shutil.copy2(source, destination)
        _restrict(destination, mask)
        copied = True

    for name, mask in _OWNED_DIRS:
        source = legacy / name
        if not source.is_dir():
            continue
        destination = stage / name
        shutil.copytree(source, destination)
        _restrict(destination, mask)
        copied = True

    return copied


def ensure_migrated() -> None:
    """Populate ``~/.venice-py`` from ``~/.venice`` once, if it is not there yet.

    Called from the entry points that first touch the directory, so an
    invocation that never reads config, presets or conversations never pays for
    it and no consumer has to know the migration exists.

    The old directory is left untouched: the official ``venice`` CLI may
    legitimately own that path now, and deleting another tool's data would be
    far worse than leaving a stale copy behind.

    Contents are staged in a sibling temporary directory and moved into place
    with a single :func:`os.rename`, which is atomic and fails when the
    destination is already populated. Concurrent CLI invocations therefore race
    for the rename instead of interleaving writes: one wins, the losers discard
    their staged copy and read what the winner published. That is cheaper than a
    lock file, which would need stale-lock recovery of its own.
    """
    global _migration_done
    with _migration_lock:
        if _migration_done:
            return
        try:
            _migrate()
        finally:
            _migration_done = True


def _migrate() -> None:
    target = app_dir()
    legacy = legacy_app_dir()

    # An existing target means either a normal install or an already-completed
    # migration. Never copy over it: the user may have edited it since, and a
    # stale legacy file must not clobber a newer one.
    if target.exists() or not legacy.is_dir():
        return

    stage: str | None = None
    try:
        # mkdtemp creates at 0o700 — the mode the config directory wants anyway,
        # since the staged directory becomes ~/.venice-py verbatim.
        stage = tempfile.mkdtemp(prefix=f"{APP_DIR_NAME}.migrating-", dir=_home())
        if not _stage_legacy_contents(legacy, Path(stage)):
            # The legacy directory exists but holds nothing of ours — most
            # likely it belongs to the official CLI. Leave the target absent so
            # its normal consumers create it on demand.
            return

        try:
            os.rename(stage, target)
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST, errno.ENOTDIR):
                raise
            # Another process published the directory first. Its copy came from
            # the same source, so drop ours without a word.
            return
        stage = None

        _echo_stderr(
            f"Venice CLI data has moved to {target}: copied config, conversations "
            f"and presets from {legacy}.\n"
            f"{legacy} was left in place — the official 'venice' CLI may use it."
        )
    except OSError as exc:
        # Migration is a convenience, never a precondition. A failure here must
        # not stop the command the user actually asked for.
        _echo_stderr(f"Warning: could not copy Venice CLI data from {legacy} to {target}: {exc}")
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
