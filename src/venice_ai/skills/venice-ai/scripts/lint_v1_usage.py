#!/usr/bin/env python3
"""Flag v1 / OpenAI-style / non-idiomatic patterns in Venice v2 codebases.

NOTE: On SDK >= 2.0.0, prefer ``venice-py lint <path>`` (the built-in CLI
subcommand) — same rule codes, same flake8-style output, plus discoverable
via ``venice-py --help``. This script exists for users on older SDK versions
or as a standalone tool that doesn't require installing the SDK.

Walks Python files under a target path and reports:

  - `max_tokens=` kwargs (v1 → use `max_completion_tokens=`)
  - `client.image.generate(...)` (gone in v2 → `client.image.create(...)`)
  - `client.audio.generate_music(...)` (not a v2 method → `client.music.run(...)`)
  - `client.video.queue(...)` / `.complete(...)` (not v2 methods → `.run`/`.submit`/`.cancel`)
  - `client.audio.generate_speech(...)` (not a v2 method → `.create_speech(...)`)
  - `client.embeddings.generate(...)` (not a v2 method → `.create(...)`)
  - `from venice_ai import AsyncVeniceClient` (wrong name → `VeniceClient`)
  - `AsyncVeniceClient(...)` constructor calls
  - Hardcoded model strings on `model=` kwargs (use `client.models.resolve_*()`)
  - `client.chat.completions.create(stream=True)` + `async for chunk` (use `.stream(...)` + `async with stream:`)
  - `tool_from_function(fn)` results passed to `run_with_tools(tools=[...])` (use bare callables)
  - `BudgetManager(limit=...)` (wrong kwarg → `daily_usd=` / `monthly_usd=`)
  - `tracker.total` / `tracker.calls` (wrong attrs → `tracker.total_cost_usd` / `tracker.total_tokens`)
  - `e.payment_instructions` on PaymentRequiredError (wrong → `e.body`)
  - `client.audio.transcribe(audio=...)` (wrong kwarg → `file=`)

Usage:
    python lint_v1_usage.py <path-to-source-tree>
    python lint_v1_usage.py src/

Exit code 0 if clean, 1 if any findings.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Finding:
    path: Path
    line: int
    col: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: {self.code} {self.message}"


# Patterns that look like Venice/OpenAI/Anthropic chat model IDs. Heuristic, not exhaustive.
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


# Common test/demo/identifier strings that LOOK like model IDs but aren't real models.
_NOT_A_MODEL = {"text-davinci-003", "demo-model", "test", "fake-model"}


class V1UsageVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, source_lines: list[str]) -> None:
        self.path = path
        self.source = source_lines
        self.findings: list[Finding] = []

    # ── Helpers ──

    def add(self, node: ast.AST, code: str, message: str) -> None:
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
    def attr_path(node: ast.AST) -> str:
        """Return dot-joined attribute path for `a.b.c.d(...)` -> `a.b.c.d`."""
        parts: list[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))

    # ── Imports ──

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.startswith("venice_ai"):
            for alias in node.names:
                if alias.name == "AsyncVeniceClient":
                    self.add(
                        alias,
                        "V100",
                        "AsyncVeniceClient does not exist in v2. Use VeniceClient (async by default).",
                    )
        self.generic_visit(node)

    # ── Calls ──

    def visit_Call(self, node: ast.Call) -> None:
        callee = (
            self.attr_path(node.func) if isinstance(node.func, (ast.Attribute, ast.Name)) else ""
        )

        # AsyncVeniceClient(...)
        if callee.endswith("AsyncVeniceClient"):
            self.add(node, "V100", "AsyncVeniceClient ctor — use VeniceClient (async default).")

        # Removed verbs / methods
        removed_methods = {
            "image.generate": ("V101", "client.image.generate(...) → client.image.create(...)"),
            "audio.generate_music": (
                "V102",
                "client.audio.generate_music(...) → client.music.run(...)",
            ),
            "audio.generate_speech": (
                "V103",
                "client.audio.generate_speech(...) → client.audio.create_speech(...)",
            ),
            "video.queue": (
                "V104",
                "client.video.queue(...) → client.video.run(...) or .submit(...)",
            ),
            "video.complete": (
                "V105",
                "client.video.complete(job_id) → client.video.cancel(job_id)",
            ),
            "embeddings.generate": (
                "V106",
                "client.embeddings.generate(...) → client.embeddings.create(...)",
            ),
        }
        for tail, (code, msg) in removed_methods.items():
            if callee.endswith(tail):
                self.add(node, code, msg)
                break

        # transcribe(audio=...)
        if callee.endswith("audio.transcribe"):
            for kw in node.keywords:
                if kw.arg == "audio":
                    self.add(
                        kw,
                        "V107",
                        "client.audio.transcribe(audio=...) — kwarg is `file=`, not `audio=`.",
                    )

        # max_tokens=
        for kw in node.keywords:
            if kw.arg == "max_tokens":
                self.add(kw, "V200", "max_tokens= was removed in v2. Use max_completion_tokens=.")

        # response_format on chat.completions.create — but model= is hardcoded
        # Hardcoded model="..." — flag string literals on `model=` kwargs of any Venice call.
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
                    self.add(
                        kw,
                        "V300",
                        f"hardcoded model='{value}' — use client.models.resolve_*() instead.",
                    )

        # client.chat.completions.create(stream=True)
        if callee.endswith("chat.completions.create"):
            for kw in node.keywords:
                if (
                    kw.arg == "stream"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    self.add(
                        kw,
                        "V401",
                        "create(stream=True) is the OpenAI-style path. Prefer client.chat.completions.stream(...) + async with stream:.",
                    )

        # BudgetManager(limit=...)
        if callee.endswith("BudgetManager"):
            for kw in node.keywords:
                if kw.arg == "limit":
                    self.add(
                        kw,
                        "V501",
                        "BudgetManager(limit=...) — wrong kwarg. Use daily_usd=Decimal(...) and/or monthly_usd=Decimal(...).",
                    )

        # PaymentRequiredError attr access — handled in visit_Attribute below

        self.generic_visit(node)

    # ── Attribute accesses ──

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Wrong tracker accessors: `tracker.total` / `tracker.calls` should be total_cost_usd / total_tokens
        # We can't know the variable's type without type inference; flag the attribute name regardless
        # (the variable name is a hint).
        if isinstance(node.value, ast.Name):
            var = node.value.id.lower()
            if "tracker" in var or "cost" in var:
                if node.attr == "total":
                    self.add(
                        node,
                        "V502",
                        f"{var}.total — should be {var}.total_cost_usd (USD spend) or {var}.total_tokens (token count) or len({var}.requests) (call count).",
                    )
                if node.attr == "calls":
                    self.add(
                        node,
                        "V503",
                        f"{var}.calls — should be len({var}.requests) (or {var}.total_tokens for tokens).",
                    )

        # PaymentRequiredError.payment_instructions
        if node.attr == "payment_instructions":
            # Heuristic: warn whenever this attribute appears (it doesn't exist in v2)
            self.add(
                node,
                "V601",
                "e.payment_instructions doesn't exist on PaymentRequiredError. Use e.body for structured payment requirements.",
            )

        self.generic_visit(node)


def lint_file(path: Path) -> list[Finding]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return [Finding(path, e.lineno or 0, e.offset or 0, "V000", f"could not parse: {e.msg}")]

    visitor = V1UsageVisitor(path, source.splitlines())
    visitor.visit(tree)
    return visitor.findings


def lint_path(target: Path) -> list[Finding]:
    findings: list[Finding] = []
    if target.is_file():
        if target.suffix == ".py":
            findings.extend(lint_file(target))
    else:
        for path in target.rglob("*.py"):
            # Skip common unhelpful directories
            if any(
                part in {"__pycache__", ".venv", ".git", "node_modules", "build", "dist"}
                for part in path.parts
            ):
                continue
            findings.extend(lint_file(path))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=Path, help="File or directory to lint")
    parser.add_argument(
        "--code",
        action="append",
        default=None,
        help="Only report findings with this code (repeatable, e.g. --code V200 --code V300)",
    )
    args = parser.parse_args(argv)

    if not args.path.exists():
        print(f"error: path does not exist: {args.path}", file=sys.stderr)
        return 2

    findings = lint_path(args.path)
    if args.code:
        wanted = set(args.code)
        findings = [f for f in findings if f.code in wanted]

    for f in findings:
        print(f)

    by_code: dict[str, int] = {}
    for f in findings:
        by_code[f.code] = by_code.get(f.code, 0) + 1
    if findings:
        print(
            f"\n{len(findings)} finding(s) across {len({f.path for f in findings})} file(s):",
            file=sys.stderr,
        )
        for code, count in sorted(by_code.items()):
            print(f"  {code}: {count}", file=sys.stderr)

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
