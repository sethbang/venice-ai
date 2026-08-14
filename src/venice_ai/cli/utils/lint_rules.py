"""AST-based linter for v1 / OpenAI-style / non-idiomatic Venice patterns.

Walks Python source files and reports patterns that won't work (or work
sub-optimally) on Venice SDK v2. Used by the ``venice-py lint`` CLI subcommand
and importable as a library for tooling integrations (e.g., a future
``ruff`` plugin).

Rule codes
----------

V100 — ``AsyncVeniceClient`` import / call (gone in v2 — use ``VeniceClient``)
V101 — ``client.image.generate(...)`` (gone — use ``client.image.create(...)``)
V102 — ``client.audio.generate_music(...)`` (not a v2 method — music is its own resource, ``client.music.run(...)``)
V103 — ``client.audio.generate_speech(...)`` (not a v2 method — use ``client.audio.create_speech(...)``)
V104 — ``client.video.queue(...)`` (not a v2 method — use ``client.video.run/submit(...)``)
V105 — ``client.video.complete(job_id)`` (not a v2 method — use ``client.video.cancel(job_id)``)
V106 — ``client.embeddings.generate(...)`` (not a v2 method — use ``client.embeddings.create(...)``)
V107 — ``client.audio.transcribe(audio=...)`` (wrong kwarg — use ``file=``)
V200 — ``max_tokens=`` kwarg (removed in v2 — use ``max_completion_tokens=``)
V300 — Hardcoded model string on ``model=`` kwarg (use ``client.models.resolve_*()``)
V401 — ``client.chat.completions.create(stream=True)`` (works, but bypasses the
       v2 streaming idiom — prefer ``client.chat.completions.stream(...)`` +
       ``async with stream:``)
V501 — ``BudgetManager(limit=...)`` (wrong kwarg — use ``daily_usd=`` /
       ``monthly_usd=``)
V502 — ``tracker.total`` accessor (real attr is ``total_cost_usd``)
V503 — ``tracker.calls`` accessor (use ``len(tracker.requests)`` or
       ``total_tokens``)
V601 — ``e.payment_instructions`` access on ``PaymentRequiredError`` (real
       attr is ``e.body``)
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Finding",
    "V1UsageVisitor",
    "lint_file",
    "lint_path",
]


@dataclass
class Finding:
    """A single lint result."""

    path: Path
    line: int
    col: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: {self.code} {self.message}"


# Patterns that look like Venice / OpenAI / Anthropic chat-model IDs. Heuristic.
_MODEL_ID_PATTERN = re.compile(
    r"^("
    r"gpt-\d|"  # gpt-3.5, gpt-4, gpt-4o, etc.
    r"claude-(opus|sonnet|haiku|3|4)|"  # claude-3-5-sonnet, etc.
    r"llama|llama-\d|"  # llama-2-70b, etc.
    r"mistral|mixtral|"
    r"deepseek|qwen|"
    r"venice-|"  # venice-uncensored-1-2, etc.
    r"zai-org|"  # zai-org-glm-4.7
    r"flux|sd-?\d|stable-diffusion|"  # image
    r"tts-|whisper|parakeet|"  # audio
    r"seedance|seedream"  # video
    r")"
)

# Common identifier strings that LOOK like model IDs but aren't real models.
# Adding entries here suppresses V300 false positives.
_NOT_A_MODEL: frozenset[str] = frozenset({"text-davinci-003", "demo-model", "test", "fake-model"})


# Codes considered "informational" — flagged but not errors unless --strict.
INFORMATIONAL_CODES: frozenset[str] = frozenset({"V401"})


# (callee_suffix, code, message) for removed-method matchers.
_REMOVED_METHOD_RULES: tuple[tuple[str, str, str], ...] = (
    ("image.generate", "V101", "client.image.generate(...) → client.image.create(...)"),
    ("audio.generate_music", "V102", "client.audio.generate_music(...) → client.music.run(...)"),
    (
        "audio.generate_speech",
        "V103",
        "client.audio.generate_speech(...) → client.audio.create_speech(...)",
    ),
    ("video.queue", "V104", "client.video.queue(...) → client.video.run(...) or .submit(...)"),
    ("video.complete", "V105", "client.video.complete(job_id) → client.video.cancel(job_id)"),
    (
        "embeddings.generate",
        "V106",
        "client.embeddings.generate(...) → client.embeddings.create(...)",
    ),
)


class V1UsageVisitor(ast.NodeVisitor):
    """AST visitor that records v1 / non-idiomatic usage patterns."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[Finding] = []

    # -- Helpers ------------------------------------------------------------

    def _add(self, node: ast.AST, code: str, message: str) -> None:
        self.findings.append(
            Finding(
                path=self.path,
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                code=code,
                message=message,
            )
        )

    @staticmethod
    def _attr_path(node: ast.AST) -> str:
        """Return dot-joined attribute path for ``a.b.c.d`` → ``"a.b.c.d"``."""
        parts: list[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))

    # -- Imports ------------------------------------------------------------

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.startswith("venice_ai"):
            for alias in node.names:
                if alias.name == "AsyncVeniceClient":
                    self._add(
                        alias,
                        "V100",
                        "AsyncVeniceClient does not exist in v2. "
                        "Use VeniceClient (async by default).",
                    )
        self.generic_visit(node)

    # -- Calls --------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:  # noqa: C901 — single dispatch site
        callee = (
            self._attr_path(node.func) if isinstance(node.func, (ast.Attribute, ast.Name)) else ""
        )

        # AsyncVeniceClient(...)
        if callee.endswith("AsyncVeniceClient"):
            self._add(node, "V100", "AsyncVeniceClient ctor — use VeniceClient (async default).")

        # Removed verbs / methods
        for tail, code, msg in _REMOVED_METHOD_RULES:
            if callee.endswith(tail):
                self._add(node, code, msg)
                break

        # transcribe(audio=...) — kwarg renamed to file=
        if callee.endswith("audio.transcribe"):
            for kw in node.keywords:
                if kw.arg == "audio":
                    self._add(
                        kw,
                        "V107",
                        "client.audio.transcribe(audio=...) — kwarg is `file=`, not `audio=`.",
                    )

        # max_tokens= kwarg
        for kw in node.keywords:
            if kw.arg == "max_tokens":
                self._add(
                    kw,
                    "V200",
                    "max_tokens= was removed in v2. Use max_completion_tokens=.",
                )

        # Hardcoded model="..." — flag string literals on model= kwargs.
        for kw in node.keywords:
            if (
                kw.arg == "model"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                value = kw.value.value
                if value in _NOT_A_MODEL:
                    continue
                if _MODEL_ID_PATTERN.match(value):
                    self._add(
                        kw,
                        "V300",
                        f"hardcoded model={value!r} — use client.models.resolve_*() instead.",
                    )

        # client.chat.completions.create(stream=True)
        if callee.endswith("chat.completions.create"):
            for kw in node.keywords:
                if (
                    kw.arg == "stream"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    self._add(
                        kw,
                        "V401",
                        "create(stream=True) bypasses the v2 streaming idiom. "
                        "Prefer client.chat.completions.stream(...) + async with stream:.",
                    )

        # BudgetManager(limit=...)
        if callee.endswith("BudgetManager"):
            for kw in node.keywords:
                if kw.arg == "limit":
                    self._add(
                        kw,
                        "V501",
                        "BudgetManager(limit=...) — use daily_usd=Decimal(...) "
                        "and/or monthly_usd=Decimal(...).",
                    )

        self.generic_visit(node)

    # -- Attribute accesses -------------------------------------------------

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Tracker accessors: tracker.total / tracker.calls are wrong attr names.
        # We can't infer the variable's type without flow analysis, so we
        # heuristically match on names containing "tracker" or "cost".
        if isinstance(node.value, ast.Name):
            var_name = node.value.id
            lower = var_name.lower()
            if "tracker" in lower or "cost" in lower:
                if node.attr == "total":
                    self._add(
                        node,
                        "V502",
                        f"{var_name}.total — should be {var_name}.total_cost_usd "
                        f"(USD spend) or {var_name}.total_tokens (token count) "
                        f"or len({var_name}.requests) (call count).",
                    )
                if node.attr == "calls":
                    self._add(
                        node,
                        "V503",
                        f"{var_name}.calls — should be len({var_name}.requests) "
                        f"(or {var_name}.total_tokens for tokens).",
                    )

        # PaymentRequiredError.payment_instructions doesn't exist in v2.
        if node.attr == "payment_instructions":
            self._add(
                node,
                "V601",
                "PaymentRequiredError.payment_instructions doesn't exist in v2. "
                "Use e.body for structured payment requirements.",
            )

        self.generic_visit(node)


def lint_file(path: Path) -> list[Finding]:
    """Lint a single ``.py`` file and return the list of findings."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [
            Finding(path, exc.lineno or 0, exc.offset or 0, "V000", f"could not parse: {exc.msg}")
        ]
    visitor = V1UsageVisitor(path)
    visitor.visit(tree)
    return visitor.findings


# Directories to skip when walking a tree.
_SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {"__pycache__", ".venv", "venv", ".git", "node_modules", "build", "dist", ".tox", ".mypy_cache"}
)


def lint_path(target: Path, codes: Iterable[str] | None = None) -> list[Finding]:
    """Lint a file or directory.

    Args:
        target: A ``.py`` file or a directory tree to walk.
        codes: Optional iterable of rule codes to keep; everything else is
            filtered out. ``None`` keeps all findings.

    Returns:
        Flat list of findings (filtered by ``codes`` if supplied), in
        file-then-line order.
    """
    findings: list[Finding] = []
    if target.is_file():
        if target.suffix == ".py":
            findings.extend(lint_file(target))
    else:
        for path in sorted(target.rglob("*.py")):
            if any(part in _SKIP_DIR_NAMES for part in path.parts):
                continue
            findings.extend(lint_file(path))

    if codes is not None:
        wanted = set(codes)
        findings = [f for f in findings if f.code in wanted]
    return findings
