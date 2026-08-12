#!/usr/bin/env python3
"""Run ``venice lint`` against every Python code block in SKILL.md / references/*.md.

Extracts ```python fenced code blocks from each markdown file under
``src/venice_ai/skills/<skill>/`` and pipes the body through the lint visitor. CI
fails if any extracted block triggers a lint finding — that means the skill
is teaching a non-idiomatic pattern.

Skips informational findings (V401) by default; pass ``--strict`` to treat
them as errors.

Exit 0 on clean, 1 on findings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / "src" / "venice_ai" / "skills"

# Make the SDK's own venice_ai package importable so we can call the visitor
# directly without a subprocess.
sys.path.insert(0, str(REPO_ROOT / "src"))

from venice_ai.cli.utils.lint_rules import (  # noqa: E402
    INFORMATIONAL_CODES,
    Finding,
    V1UsageVisitor,
)


def _extract_python_blocks(md_text: str) -> list[tuple[int, str]]:
    """Return (start_line, body) for each ```python fenced block."""
    blocks: list[tuple[int, str]] = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```python") or line.strip() == "```py":
            start = i + 1
            body_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body_lines.append(lines[i])
                i += 1
            blocks.append((start + 1, "\n".join(body_lines)))
        i += 1
    return blocks


def _lint_block(body: str, source_path: Path, start_line: int) -> list[Finding]:
    import ast

    try:
        tree = ast.parse(body)
    except SyntaxError:
        # Skill examples are sometimes intentionally fragmentary (no top-level
        # `async def` wrapper, etc.). Don't fail on parse errors — that's not
        # what the lint rules are for.
        return []
    visitor = V1UsageVisitor(source_path)
    visitor.visit(tree)
    # Adjust line numbers to point into the markdown file.
    for f in visitor.findings:
        f.line += start_line - 1
    return visitor.findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict", action="store_true", help="Treat informational findings as errors."
    )
    args = parser.parse_args()

    all_findings: list[Finding] = []
    blocks_checked = 0
    files_checked = 0

    for skill_dir in sorted(SKILLS_ROOT.glob("venice-ai*")):
        if not skill_dir.is_dir():
            continue
        for md in skill_dir.rglob("*.md"):
            files_checked += 1
            text = md.read_text(encoding="utf-8")
            for start_line, body in _extract_python_blocks(text):
                blocks_checked += 1
                findings = _lint_block(body, md, start_line)
                if args.strict:
                    all_findings.extend(findings)
                else:
                    all_findings.extend(f for f in findings if f.code not in INFORMATIONAL_CODES)

    print(f"Checked {blocks_checked} python code block(s) across {files_checked} markdown file(s).")
    if all_findings:
        print()
        for f in all_findings:
            print(f"{f.path.relative_to(REPO_ROOT)}:{f.line}:{f.col}: {f.code} {f.message}")
        print()
        print(
            f"{len(all_findings)} finding(s); fix the SKILL.md / reference example to use the canonical v2 pattern."
        )
        return 1
    print("All code blocks teach idiomatic v2 patterns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
