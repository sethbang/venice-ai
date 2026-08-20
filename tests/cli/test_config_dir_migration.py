"""Tests for the one-time ``~/.venice`` → ``~/.venice-py`` data migration.

The CLI's per-user directory moved because ``~/.venice`` belongs to the
official Venice CLI. It holds user-authored content — hand-tuned image presets,
chat history, a configured base URL — so the constant change is paired with a
copy that runs the first time anything reads the directory. These tests pin the
properties that make that copy safe: it never deletes the old directory, never
overwrites the new one, never runs twice, and never widens permissions.

Every test drives a temporary home via ``_paths._home``; nothing here touches
the real ``~/.venice`` or ``~/.venice-py``.
"""

from __future__ import annotations

import errno
import json
import os
import stat

import pytest

from venice_ai.cli import _paths
from venice_ai.cli import config as cli_config
from venice_ai.cli import conversation as cli_conversation
from venice_ai.cli import presets as cli_presets

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway home directory that ``_paths`` resolves against."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(_paths, "_home", lambda: fake_home)
    return fake_home


def _make_legacy(home, *, config: bool = True, conversations: bool = True, presets: bool = True):
    """Populate a pre-rename ``~/.venice`` and return it."""
    legacy = home / _paths.LEGACY_APP_DIR_NAME
    legacy.mkdir(exist_ok=True)

    if config:
        (legacy / "config.yaml").write_text("api:\n  key: legacy-key\n")
    if conversations:
        convs = legacy / "conversations"
        convs.mkdir()
        (convs / "conv1.json").write_text(
            json.dumps({"id": "conv1", "title": "old chat", "messages": []})
        )
    if presets:
        presets_dir = legacy / "presets"
        presets_dir.mkdir()
        (presets_dir / "my-look.json").write_text(json.dumps({"name": "my-look", "config": {}}))

    return legacy


def _run_migration():
    """Run ``ensure_migrated`` as a fresh process would.

    The real latch is per-process; clearing it first models a second invocation
    of ``venice-py`` rather than a second call inside one.
    """
    _paths._migration_done = False
    _paths.ensure_migrated()


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _stage_dirs(home) -> list:
    """Leftover staging directories, which a clean run must not produce."""
    return [p for p in home.iterdir() if p.name.startswith(f"{_paths.APP_DIR_NAME}.migrating-")]


# ---------------------------------------------------------------------------
# Fresh install
# ---------------------------------------------------------------------------


class TestFreshInstall:
    def test_no_directories_are_created_when_neither_exists(self, home, capsys):
        """A first-time user has no legacy data; nothing should be invented."""
        _run_migration()

        assert not (home / _paths.APP_DIR_NAME).exists()
        assert not (home / _paths.LEGACY_APP_DIR_NAME).exists()
        assert capsys.readouterr().err == ""

    def test_legacy_without_any_owned_subpaths_is_not_migrated(self, home, capsys):
        """``~/.venice`` may belong purely to the official CLI.

        Copying an unrelated tool's files — or creating ``~/.venice-py`` as a
        side effect of noticing them — would be wrong on both counts.
        """
        legacy = home / _paths.LEGACY_APP_DIR_NAME
        legacy.mkdir()
        (legacy / "credentials.json").write_text("{}")

        _run_migration()

        assert not (home / _paths.APP_DIR_NAME).exists()
        assert (legacy / "credentials.json").exists()
        assert capsys.readouterr().err == ""
        assert _stage_dirs(home) == []


# ---------------------------------------------------------------------------
# The migration itself
# ---------------------------------------------------------------------------


class TestLegacyIsCopied:
    def test_owned_subpaths_are_copied(self, home):
        """Config, conversations and presets all land in the new directory."""
        _make_legacy(home)

        _run_migration()

        target = home / _paths.APP_DIR_NAME
        assert (target / "config.yaml").read_text() == "api:\n  key: legacy-key\n"
        assert json.loads((target / "conversations" / "conv1.json").read_text())["id"] == "conv1"
        assert json.loads((target / "presets" / "my-look.json").read_text())["name"] == "my-look"

    def test_legacy_directory_is_left_intact(self, home):
        """Copy, never move — the official ``venice`` CLI may own that path."""
        legacy = _make_legacy(home)

        _run_migration()

        assert legacy.is_dir()
        assert (legacy / "config.yaml").exists()
        assert (legacy / "conversations" / "conv1.json").exists()
        assert (legacy / "presets" / "my-look.json").exists()

    def test_unowned_legacy_files_are_not_copied(self, home):
        """Only our three subpaths move; the rest is somebody else's data."""
        legacy = _make_legacy(home)
        (legacy / "credentials.json").write_text("{}")
        (legacy / "sessions").mkdir()

        _run_migration()

        target = home / _paths.APP_DIR_NAME
        assert not (target / "credentials.json").exists()
        assert not (target / "sessions").exists()

    def test_partial_legacy_directory_is_copied(self, home):
        """A user who only ever ran ``configure`` still gets their config."""
        _make_legacy(home, conversations=False, presets=False)

        _run_migration()

        target = home / _paths.APP_DIR_NAME
        assert (target / "config.yaml").exists()
        assert not (target / "conversations").exists()

    def test_no_staging_directory_is_left_behind(self, home):
        _make_legacy(home)

        _run_migration()

        assert _stage_dirs(home) == []

    def test_message_names_both_directories(self, home, capsys):
        legacy = _make_legacy(home)

        _run_migration()

        err = capsys.readouterr().err
        assert str(home / _paths.APP_DIR_NAME) in err
        assert str(legacy) in err
        assert "left in place" in err


# ---------------------------------------------------------------------------
# Idempotency / no clobbering
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_existing_target_is_never_overwritten(self, home, capsys):
        """Both directories present: the newer one wins, untouched."""
        _make_legacy(home)
        target = home / _paths.APP_DIR_NAME
        target.mkdir()
        (target / "config.yaml").write_text("api:\n  key: current-key\n")

        _run_migration()

        assert (target / "config.yaml").read_text() == "api:\n  key: current-key\n"
        # A stale legacy conversation must not appear alongside it either.
        assert not (target / "conversations").exists()
        assert capsys.readouterr().err == ""

    def test_second_run_does_not_recopy_edited_files(self, home):
        """The user edits a migrated file; the next run must respect that."""
        _make_legacy(home)
        _run_migration()

        target = home / _paths.APP_DIR_NAME
        (target / "config.yaml").write_text("api:\n  key: edited-key\n")
        (target / "presets" / "my-look.json").unlink()

        _run_migration()

        assert (target / "config.yaml").read_text() == "api:\n  key: edited-key\n"
        assert not (target / "presets" / "my-look.json").exists()

    def test_message_is_not_repeated_on_the_second_run(self, home, capsys):
        _make_legacy(home)

        _run_migration()
        assert "left in place" in capsys.readouterr().err

        _run_migration()
        assert capsys.readouterr().err == ""

    def test_repeated_calls_within_one_process_run_once(self, home, capsys):
        """The in-process latch keeps a multi-command session from re-checking."""
        _make_legacy(home)

        _paths._migration_done = False
        _paths.ensure_migrated()
        first = capsys.readouterr().err

        _paths.ensure_migrated()
        second = capsys.readouterr().err

        assert "left in place" in first
        assert second == ""


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------


class TestPermissions:
    def test_conversations_dir_is_not_left_group_or_world_accessible(self, home):
        """Transcripts hold prompt/response text; 0o755 on the source must not
        survive the copy."""
        legacy = _make_legacy(home)
        os.chmod(legacy / "conversations", 0o755)

        _run_migration()

        mode = _mode(home / _paths.APP_DIR_NAME / "conversations")
        assert mode & 0o077 == 0, oct(mode)
        assert mode == 0o700, oct(mode)

    def test_narrower_source_permissions_are_preserved(self, home):
        """Clamping must never widen: a 0o500 source stays 0o500."""
        legacy = _make_legacy(home)
        os.chmod(legacy / "conversations", 0o500)

        _run_migration()

        assert _mode(home / _paths.APP_DIR_NAME / "conversations") == 0o500

    def test_config_file_is_not_left_world_readable(self, home):
        """``config.yaml`` may hold a plaintext API key."""
        legacy = _make_legacy(home)
        os.chmod(legacy / "config.yaml", 0o644)

        _run_migration()

        assert _mode(home / _paths.APP_DIR_NAME / "config.yaml") == 0o600

    def test_new_directory_is_owner_only(self, home):
        _make_legacy(home)

        _run_migration()

        assert _mode(home / _paths.APP_DIR_NAME) == 0o700


# ---------------------------------------------------------------------------
# Concurrency and failure handling
# ---------------------------------------------------------------------------


class TestConcurrencyAndFailures:
    def test_losing_the_rename_race_is_silent_and_cleans_up(self, home, capsys, monkeypatch):
        """Two CLI processes race; the loser discards its staged copy.

        Simulated by failing the rename the way the kernel does when another
        process has already published the directory.
        """
        _make_legacy(home)

        real_rename = os.rename

        def _busy(src, dst):
            if str(dst) == str(home / _paths.APP_DIR_NAME):
                raise OSError(errno.ENOTEMPTY, "Directory not empty")
            return real_rename(src, dst)

        monkeypatch.setattr(os, "rename", _busy)

        _run_migration()

        assert _stage_dirs(home) == []
        assert capsys.readouterr().err == ""

    def test_unexpected_copy_failure_warns_and_does_not_raise(self, home, capsys, monkeypatch):
        """Migration is a convenience; it must never abort the user's command."""
        _make_legacy(home)

        def _boom(*args, **kwargs):
            raise OSError(errno.EACCES, "Permission denied")

        monkeypatch.setattr(_paths.shutil, "copy2", _boom)

        _run_migration()

        assert "could not copy Venice CLI data" in capsys.readouterr().err
        assert _stage_dirs(home) == []


# ---------------------------------------------------------------------------
# Wiring into the consumers
# ---------------------------------------------------------------------------


class TestConsumerHooks:
    """Each consumer must trigger the migration before it reads its directory."""

    @pytest.fixture
    def wired_home(self, home, monkeypatch):
        """A temporary home with legacy data, with the import-time path
        constants pointed at it the way a real install has them."""
        _make_legacy(home)
        app_dir = home / _paths.APP_DIR_NAME
        monkeypatch.setattr(cli_config, "DEFAULT_CONFIG_PATH", app_dir / "config.yaml")
        monkeypatch.setattr(cli_presets, "DEFAULT_PRESETS_DIR", app_dir / "presets")
        monkeypatch.setattr(cli_conversation, "CONVERSATIONS_DIR", str(app_dir / "conversations"))
        _paths._migration_done = False
        return app_dir

    @pytest.mark.parametrize(
        "entry_point",
        [
            pytest.param(lambda: cli_config.load_config(), id="load_config"),
            pytest.param(lambda: cli_presets.get_presets_dir(), id="get_presets_dir"),
            pytest.param(lambda: cli_conversation._ensure_dir(), id="_ensure_dir"),
            pytest.param(lambda: cli_conversation.list_conversations(), id="list_conversations"),
            # load/delete never call _ensure_dir, so they need their own hook —
            # without it a pre-rename conversation silently looks deleted.
            pytest.param(
                lambda: cli_conversation.load_conversation("conv1"), id="load_conversation"
            ),
            pytest.param(
                lambda: cli_conversation.delete_conversation("nope"), id="delete_conversation"
            ),
        ],
    )
    def test_entry_point_triggers_migration(self, wired_home, entry_point):
        entry_point()

        assert (wired_home / "conversations" / "conv1.json").exists()
        assert (wired_home / "presets" / "my-look.json").exists()

    def test_save_config_migrates_before_creating_the_directory(self, wired_home):
        """``venice-py configure`` as the very first command must not orphan
        the legacy data by creating ``~/.venice-py`` ahead of the copy."""
        cli_config.save_config({"api": {"key": "brand-new"}})

        assert (wired_home / "conversations" / "conv1.json").exists()
        assert (wired_home / "presets" / "my-look.json").exists()
        assert cli_config.load_config()["api"]["key"] == "brand-new"

    def test_load_config_reads_the_migrated_file(self, wired_home):
        """The end-to-end point of the migration: a pre-rename API key still
        resolves after the directory moves."""
        assert cli_config.load_config().get("api", {}).get("key") == "legacy-key"

    def test_load_conversation_finds_a_pre_rename_transcript(self, wired_home):
        assert cli_conversation.load_conversation("conv1")["title"] == "old chat"
