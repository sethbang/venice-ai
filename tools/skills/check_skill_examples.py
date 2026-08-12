#!/usr/bin/env python3
"""Verify that examples/* and scripts/* paths referenced in SKILL.md files exist.

Walks every ``src/venice_ai/skills/<skill>/SKILL.md`` and ``references/*.md`` for
markdown inline-code paths that look like ``examples/foo/bar.py``. Verifies
each one resolves under the SDK repo's ``examples/`` directory.

Also validates ``scripts/<file>`` references appearing under a ``## Scripts``
heading: each one must resolve under the owning skill's ``scripts/`` directory.

Used by CI to catch reference drift when example or script files get renamed
or removed without updating the skills.

Exit 0 on clean, 1 on any missing reference.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / "src" / "venice_ai" / "skills"
EXAMPLES_ROOT = REPO_ROOT / "examples"

# Two patterns we check for:
#   1. Backticks with leading `examples/` — explicit path under SDK repo root.
#      Example: `examples/chat/streaming_chat.py`
#   2. Backticks with relative path under an "Examples to read" section header —
#      these are implicit refs to examples/. Example: `chat/streaming_chat.py`
_PATH_RE_EXPLICIT = re.compile(r"`(examples/[^\s`]+\.py)`")
_PATH_RE_RELATIVE = re.compile(r"`([a-zA-Z_][a-zA-Z0-9_/-]*/[a-zA-Z0-9_]+\.py)`")
_EXAMPLES_HEADING_RE = re.compile(r"^##\s+Examples to read", re.IGNORECASE)
_SCRIPTS_HEADING_RE = re.compile(r"^##\s+Scripts\b", re.IGNORECASE)
# scripts/<file>.py references — resolved relative to the owning skill dir.
_PATH_RE_SCRIPTS = re.compile(r"`(scripts/[a-zA-Z0-9_/-]+\.py)\b")


def _check_path(rel: str, base: Path = EXAMPLES_ROOT) -> bool:
    return (base / rel).exists() if base == EXAMPLES_ROOT else (REPO_ROOT / rel).exists()


def main() -> int:
    if not EXAMPLES_ROOT.exists():
        print(f"error: {EXAMPLES_ROOT} does not exist", file=sys.stderr)
        return 1

    missing: list[tuple[Path, int, str]] = []
    checked = 0

    for skill_dir in sorted(SKILLS_ROOT.glob("venice-ai*")):
        if not skill_dir.is_dir():
            continue
        for md in skill_dir.rglob("*.md"):
            in_examples_section = False
            in_scripts_section = False
            for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
                # Track which section we're in by the most recent "## " heading.
                if line.startswith("## "):
                    in_examples_section = bool(_EXAMPLES_HEADING_RE.match(line))
                    in_scripts_section = bool(_SCRIPTS_HEADING_RE.match(line))

                # Always check explicit `examples/...` references.
                for match in _PATH_RE_EXPLICIT.finditer(line):
                    rel = match.group(1)
                    checked += 1
                    if not (REPO_ROOT / rel).exists():
                        missing.append((md, lineno, rel))

                # In an Examples section, also check bare relative paths (treated
                # as relative to examples/).
                if in_examples_section and not line.startswith("##"):
                    for match in _PATH_RE_RELATIVE.finditer(line):
                        rel = match.group(1)
                        # Skip explicit examples/... already handled above
                        if rel.startswith("examples/"):
                            continue
                        checked += 1
                        if not (EXAMPLES_ROOT / rel).exists():
                            missing.append((md, lineno, f"examples/{rel}"))

                # In a Scripts section, check `scripts/<file>.py` refs against the
                # owning skill's scripts/ directory.
                if in_scripts_section and not line.startswith("##"):
                    for match in _PATH_RE_SCRIPTS.finditer(line):
                        rel = match.group(1)
                        checked += 1
                        if not (skill_dir / rel).exists():
                            missing.append((md, lineno, f"{skill_dir.name}/{rel}"))

    print(
        f"Checked {checked} examples/* and scripts/* file references "
        "across SKILL.md and references/"
    )
    if missing:
        print()
        for path, lineno, rel in missing:
            print(f"{path.relative_to(REPO_ROOT)}:{lineno}: missing -> {rel}")
        print()
        print(f"{len(missing)} broken reference(s); update the SKILL.md or restore the example.")
        return 1

    print("All references resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
