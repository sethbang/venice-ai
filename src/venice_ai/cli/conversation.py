"""Conversation persistence for Venice AI CLI."""

import json
import os
import re
from datetime import datetime
from typing import Any

from . import _paths

CONVERSATIONS_DIR = str(_paths.app_dir() / "conversations")


def _ensure_dir():
    _paths.ensure_migrated()
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
    # Conversation transcripts may contain sensitive prompt/response text;
    # restrict the directory to the owner only (0o700).
    os.chmod(CONVERSATIONS_DIR, 0o700)


def _safe_conv_path(conv_id: str) -> str:
    """Build a safe file path for a conversation ID, preventing path traversal."""
    # Reads that skip _ensure_dir (load / delete) still need the migration to
    # have run, or a conversation saved before the rename looks like it is gone.
    _paths.ensure_migrated()
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", conv_id)
    if not safe_id or safe_id != conv_id:
        raise ValueError(f"Invalid conversation ID: {conv_id!r}")
    return os.path.join(CONVERSATIONS_DIR, f"{safe_id}.json")


def save_conversation(conv_id: str, model: str, messages: list, title: str | None = None) -> str:
    """Save a conversation to disk. Returns the file path."""
    _ensure_dir()
    if not title and messages:
        # Auto-generate title from first user message
        for msg in messages:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if role == "user" and content:
                content_str = content if isinstance(content, str) else str(content)
                title = content_str[:60] + ("..." if len(content_str) > 60 else "")
                break

    # Serialize messages to plain dicts
    serializable_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            serializable_messages.append(msg)
        else:
            serializable_messages.append(
                {
                    "role": getattr(msg, "role", "unknown"),
                    "content": getattr(msg, "content", ""),
                }
            )

    data = {
        "id": conv_id,
        "title": title or "Untitled",
        "model": model,
        "messages": serializable_messages,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    filepath = _safe_conv_path(conv_id)

    # If file exists, preserve created_at
    if os.path.exists(filepath):
        with open(filepath) as f:
            existing = json.load(f)
            data["created_at"] = existing.get("created_at", data["created_at"])

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    # Transcript may contain sensitive content; restrict to owner read/write.
    os.chmod(filepath, 0o600)

    return filepath


def load_conversation(conv_id: str) -> dict[str, Any] | None:
    """Load a conversation by ID."""
    filepath = _safe_conv_path(conv_id)
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        result: dict[str, Any] = json.load(f)
        return result


def list_conversations() -> list:
    """List all saved conversations, sorted by updated_at descending."""
    _ensure_dir()
    conversations = []
    for filename in os.listdir(CONVERSATIONS_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(CONVERSATIONS_DIR, filename)
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    conversations.append(data)
            except (OSError, json.JSONDecodeError):
                continue

    conversations.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
    return conversations


def delete_conversation(conv_id: str) -> bool:
    """Delete a conversation. Returns True if deleted."""
    filepath = _safe_conv_path(conv_id)
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False


def get_last_conversation_id() -> str | None:
    """Get the ID of the most recently updated conversation."""
    convs = list_conversations()
    return convs[0]["id"] if convs else None
