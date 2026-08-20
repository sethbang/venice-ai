"""
Tests for venice_ai.cli.conversation - conversation persistence module.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from venice_ai.cli.conversation import (
    _ensure_dir,
    _safe_conv_path,
    delete_conversation,
    get_last_conversation_id,
    list_conversations,
    load_conversation,
    save_conversation,
)

# ---------------------------------------------------------------------------
# Helper / fixtures
# ---------------------------------------------------------------------------


def _patch_dir(tmp_path):
    """Return a context manager that redirects CONVERSATIONS_DIR to tmp_path."""
    return patch("venice_ai.cli.conversation.CONVERSATIONS_DIR", str(tmp_path))


# ---------------------------------------------------------------------------
# _ensure_dir
# ---------------------------------------------------------------------------


class TestEnsureDir:
    def test_creates_directory_if_missing(self, tmp_path):
        target = tmp_path / "new_dir"
        with patch("venice_ai.cli.conversation.CONVERSATIONS_DIR", str(target)):
            _ensure_dir()
        assert target.is_dir()

    def test_does_not_raise_if_directory_exists(self, tmp_path):
        with _patch_dir(tmp_path):
            _ensure_dir()  # first call creates it
            _ensure_dir()  # second call should be a no-op (exist_ok=True)
        assert tmp_path.is_dir()


# ---------------------------------------------------------------------------
# _safe_conv_path
# ---------------------------------------------------------------------------


class TestSafeConvPath:
    def test_normal_id_returns_path(self, tmp_path):
        with _patch_dir(tmp_path):
            path = _safe_conv_path("abc-123")
        assert path.endswith("abc-123.json")

    def test_underscore_and_dash_are_allowed(self, tmp_path):
        with _patch_dir(tmp_path):
            path = _safe_conv_path("conv_id-01")
        assert "conv_id-01.json" in path

    def test_path_traversal_raises(self, tmp_path):
        with _patch_dir(tmp_path), pytest.raises(ValueError, match="Invalid conversation ID"):
            _safe_conv_path("../../../etc/passwd")

    def test_id_with_slash_raises(self, tmp_path):
        with _patch_dir(tmp_path), pytest.raises(ValueError, match="Invalid conversation ID"):
            _safe_conv_path("some/id")

    def test_id_with_spaces_raises(self, tmp_path):
        with _patch_dir(tmp_path), pytest.raises(ValueError, match="Invalid conversation ID"):
            _safe_conv_path("my id")

    def test_empty_id_raises(self, tmp_path):
        with _patch_dir(tmp_path), pytest.raises(ValueError, match="Invalid conversation ID"):
            _safe_conv_path("")

    def test_id_with_dot_raises(self, tmp_path):
        with _patch_dir(tmp_path), pytest.raises(ValueError, match="Invalid conversation ID"):
            _safe_conv_path("file.json")

    def test_path_includes_conversations_dir(self, tmp_path):
        with _patch_dir(tmp_path):
            path = _safe_conv_path("myconv")
        assert str(tmp_path) in path


# ---------------------------------------------------------------------------
# File / directory permissions
# ---------------------------------------------------------------------------


class TestConversationPermissions:
    def test_saved_file_mode_is_0600(self, tmp_path):
        import stat

        convs_dir = tmp_path / ".venice-py" / "conversations"
        with patch("venice_ai.cli.conversation.CONVERSATIONS_DIR", str(convs_dir)):
            filepath = save_conversation("conv1", "model", [{"role": "user", "content": "hi"}])
        mode = stat.S_IMODE(Path(filepath).stat().st_mode)
        assert mode == 0o600, oct(mode)

    def test_conversations_dir_mode_is_0700(self, tmp_path):
        import stat

        # Point at a fresh, non-existent subpath so directory creation runs.
        convs_dir = tmp_path / ".venice-py" / "conversations"
        with patch("venice_ai.cli.conversation.CONVERSATIONS_DIR", str(convs_dir)):
            _ensure_dir()
        mode = stat.S_IMODE(convs_dir.stat().st_mode)
        assert mode == 0o700, oct(mode)


# ---------------------------------------------------------------------------
# save_conversation
# ---------------------------------------------------------------------------


class TestSaveConversation:
    def test_new_conversation_creates_file(self, tmp_path):
        with _patch_dir(tmp_path):
            filepath = save_conversation("conv1", "gpt-4", [{"role": "user", "content": "Hello"}])
        assert Path(filepath).exists()

    def test_returns_filepath_string(self, tmp_path):
        with _patch_dir(tmp_path):
            result = save_conversation("conv1", "gpt-4", [])
        assert isinstance(result, str)
        assert result.endswith(".json")

    def test_auto_title_from_first_user_dict_message(self, tmp_path):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Tell me about Python"},
        ]
        with _patch_dir(tmp_path):
            filepath = save_conversation("conv1", "model", messages)
            with open(filepath) as f:
                data = json.load(f)
        assert data["title"] == "Tell me about Python"

    def test_auto_title_truncated_at_60_chars(self, tmp_path):
        long_content = "A" * 80
        messages = [{"role": "user", "content": long_content}]
        with _patch_dir(tmp_path):
            filepath = save_conversation("conv1", "model", messages)
            with open(filepath) as f:
                data = json.load(f)
        assert data["title"] == "A" * 60 + "..."

    def test_auto_title_short_message_no_ellipsis(self, tmp_path):
        messages = [{"role": "user", "content": "Short"}]
        with _patch_dir(tmp_path):
            filepath = save_conversation("conv1", "model", messages)
            with open(filepath) as f:
                data = json.load(f)
        assert data["title"] == "Short"
        assert "..." not in data["title"]

    def test_auto_title_from_message_object_with_content_attr(self, tmp_path):
        msg = MagicMock()
        msg.role = "user"
        msg.content = "Object message content"
        with _patch_dir(tmp_path):
            filepath = save_conversation("conv1", "model", [msg])
            with open(filepath) as f:
                data = json.load(f)
        assert data["title"] == "Object message content"

    def test_explicit_title_overrides_auto(self, tmp_path):
        messages = [{"role": "user", "content": "Some user message"}]
        with _patch_dir(tmp_path):
            filepath = save_conversation("conv1", "model", messages, title="My Custom Title")
            with open(filepath) as f:
                data = json.load(f)
        assert data["title"] == "My Custom Title"

    def test_fallback_title_when_no_user_messages(self, tmp_path):
        messages = [{"role": "system", "content": "System only"}]
        with _patch_dir(tmp_path):
            filepath = save_conversation("conv1", "model", messages)
            with open(filepath) as f:
                data = json.load(f)
        assert data["title"] == "Untitled"

    def test_fallback_title_when_empty_messages(self, tmp_path):
        with _patch_dir(tmp_path):
            filepath = save_conversation("conv1", "model", [])
            with open(filepath) as f:
                data = json.load(f)
        assert data["title"] == "Untitled"

    def test_messages_as_dicts_serialized(self, tmp_path):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        with _patch_dir(tmp_path):
            filepath = save_conversation("conv1", "model", messages)
            with open(filepath) as f:
                data = json.load(f)
        assert data["messages"] == messages

    def test_messages_as_objects_serialized(self, tmp_path):
        msg1 = MagicMock(spec=[])  # spec=[] prevents auto-attribute creation
        msg1.role = "user"
        msg1.content = "Hello from obj"

        with _patch_dir(tmp_path):
            filepath = save_conversation("conv1", "model", [msg1])
            with open(filepath) as f:
                data = json.load(f)
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hello from obj"

    def test_preserves_created_at_on_update(self, tmp_path):
        messages = [{"role": "user", "content": "Hello"}]
        with _patch_dir(tmp_path):
            # First save
            filepath = save_conversation("conv1", "model", messages)
            with open(filepath) as f:
                original = json.load(f)
            original_created_at = original["created_at"]

            # Second save (update)
            filepath2 = save_conversation(
                "conv1", "model", messages + [{"role": "assistant", "content": "Hi"}]
            )
            with open(filepath2) as f:
                updated = json.load(f)

        assert updated["created_at"] == original_created_at

    def test_updated_at_is_set(self, tmp_path):
        with _patch_dir(tmp_path):
            filepath = save_conversation("conv1", "model", [])
            with open(filepath) as f:
                data = json.load(f)
        assert "updated_at" in data
        assert data["updated_at"]

    def test_created_at_is_set(self, tmp_path):
        with _patch_dir(tmp_path):
            filepath = save_conversation("conv1", "model", [])
            with open(filepath) as f:
                data = json.load(f)
        assert "created_at" in data
        assert data["created_at"]

    def test_model_stored(self, tmp_path):
        with _patch_dir(tmp_path):
            filepath = save_conversation("conv1", "venice-gpt4", [])
            with open(filepath) as f:
                data = json.load(f)
        assert data["model"] == "venice-gpt4"

    def test_id_stored(self, tmp_path):
        with _patch_dir(tmp_path):
            filepath = save_conversation("myconv", "model", [])
            with open(filepath) as f:
                data = json.load(f)
        assert data["id"] == "myconv"

    def test_invalid_id_raises(self, tmp_path):
        with _patch_dir(tmp_path), pytest.raises(ValueError):
            save_conversation("bad/id", "model", [])

    def test_object_message_missing_attrs_uses_defaults(self, tmp_path):
        """Message object without role/content attributes uses getattr defaults."""
        msg = object()  # plain object - getattr will use default
        with _patch_dir(tmp_path):
            filepath = save_conversation("conv1", "model", [msg])
            with open(filepath) as f:
                data = json.load(f)
        assert data["messages"][0]["role"] == "unknown"
        assert data["messages"][0]["content"] == ""

    def test_user_message_with_non_string_content(self, tmp_path):
        """Non-string content in user message gets converted to str for title."""
        messages = [{"role": "user", "content": 12345}]
        with _patch_dir(tmp_path):
            filepath = save_conversation("conv1", "model", messages)
            with open(filepath) as f:
                data = json.load(f)
        assert data["title"] == "12345"


# ---------------------------------------------------------------------------
# load_conversation
# ---------------------------------------------------------------------------


class TestLoadConversation:
    def test_load_existing_conversation(self, tmp_path):
        messages = [{"role": "user", "content": "Hello"}]
        with _patch_dir(tmp_path):
            save_conversation("conv1", "model", messages)
            result = load_conversation("conv1")
        assert result is not None
        assert result["id"] == "conv1"
        assert result["messages"] == messages

    def test_load_nonexistent_returns_none(self, tmp_path):
        with _patch_dir(tmp_path):
            result = load_conversation("nonexistent")
        assert result is None

    def test_load_invalid_id_raises(self, tmp_path):
        with _patch_dir(tmp_path), pytest.raises(ValueError):
            load_conversation("../bad")

    def test_load_returns_dict(self, tmp_path):
        with _patch_dir(tmp_path):
            save_conversation("conv1", "model", [])
            result = load_conversation("conv1")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# list_conversations
# ---------------------------------------------------------------------------


class TestListConversations:
    def test_empty_directory_returns_empty_list(self, tmp_path):
        with _patch_dir(tmp_path):
            result = list_conversations()
        assert result == []

    def test_lists_all_conversations(self, tmp_path):
        with _patch_dir(tmp_path):
            save_conversation("conv1", "model", [{"role": "user", "content": "A"}])
            save_conversation("conv2", "model", [{"role": "user", "content": "B"}])
            result = list_conversations()
        assert len(result) == 2
        ids = {c["id"] for c in result}
        assert ids == {"conv1", "conv2"}

    def test_sorted_by_updated_at_descending(self, tmp_path):
        """Conversations should be sorted newest first."""
        with _patch_dir(tmp_path):
            save_conversation("older", "model", [{"role": "user", "content": "Old"}])
            save_conversation("newer", "model", [{"role": "user", "content": "New"}])
            result = list_conversations()
        # The last saved one should appear first (it has the latest updated_at)
        assert result[0]["id"] == "newer"

    def test_invalid_json_files_skipped_gracefully(self, tmp_path):
        """Invalid JSON files should be silently skipped."""
        with _patch_dir(tmp_path):
            # Write a bad JSON file
            bad_file = tmp_path / "bad.json"
            bad_file.write_text("{{{not valid json")

            save_conversation("ok", "model", [{"role": "user", "content": "Good"}])
            result = list_conversations()
        # Only the valid conversation should appear
        assert len(result) == 1
        assert result[0]["id"] == "ok"

    def test_non_json_files_ignored(self, tmp_path):
        """Non-.json files in the directory should be ignored."""
        with _patch_dir(tmp_path):
            # Write a non-json file
            (tmp_path / "somefile.txt").write_text("hello")
            save_conversation("conv1", "model", [])
            result = list_conversations()
        assert len(result) == 1

    def test_mix_of_valid_and_invalid_files(self, tmp_path):
        with _patch_dir(tmp_path):
            save_conversation("good1", "model", [{"role": "user", "content": "G1"}])
            save_conversation("good2", "model", [{"role": "user", "content": "G2"}])
            # Two bad files
            (tmp_path / "broken1.json").write_text("NOT JSON")
            (tmp_path / "broken2.json").write_text("")
            result = list_conversations()
        assert len(result) == 2
        ids = {c["id"] for c in result}
        assert "good1" in ids
        assert "good2" in ids

    def test_ioerror_file_skipped(self, tmp_path):
        """IOError when reading a file should be silently skipped."""
        import builtins

        original_open = builtins.open

        with _patch_dir(tmp_path):
            save_conversation("conv1", "model", [])

            # Patch open to raise IOError only for a specific filename
            def fake_open(name, *args, **kwargs):
                if "conv1" in str(name):
                    raise OSError("simulated read error")
                return original_open(name, *args, **kwargs)

            with patch("builtins.open", side_effect=fake_open):
                result = list_conversations()

        assert result == []


# ---------------------------------------------------------------------------
# delete_conversation
# ---------------------------------------------------------------------------


class TestDeleteConversation:
    def test_delete_existing_returns_true(self, tmp_path):
        with _patch_dir(tmp_path):
            save_conversation("conv1", "model", [])
            result = delete_conversation("conv1")
        assert result is True

    def test_delete_removes_file(self, tmp_path):
        with _patch_dir(tmp_path):
            filepath = save_conversation("conv1", "model", [])
            delete_conversation("conv1")
        assert not Path(filepath).exists()

    def test_delete_nonexistent_returns_false(self, tmp_path):
        with _patch_dir(tmp_path):
            result = delete_conversation("doesnotexist")
        assert result is False

    def test_delete_invalid_id_raises(self, tmp_path):
        with _patch_dir(tmp_path), pytest.raises(ValueError):
            delete_conversation("bad/id")


# ---------------------------------------------------------------------------
# get_last_conversation_id
# ---------------------------------------------------------------------------


class TestGetLastConversationId:
    def test_returns_none_when_no_conversations(self, tmp_path):
        with _patch_dir(tmp_path):
            result = get_last_conversation_id()
        assert result is None

    def test_returns_most_recent_id(self, tmp_path):
        with _patch_dir(tmp_path):
            save_conversation("first", "model", [{"role": "user", "content": "A"}])
            save_conversation("second", "model", [{"role": "user", "content": "B"}])
            result = get_last_conversation_id()
        assert result == "second"

    def test_returns_string(self, tmp_path):
        with _patch_dir(tmp_path):
            save_conversation("myconv", "model", [])
            result = get_last_conversation_id()
        assert isinstance(result, str)

    def test_single_conversation(self, tmp_path):
        with _patch_dir(tmp_path):
            save_conversation("only", "model", [])
            result = get_last_conversation_id()
        assert result == "only"
