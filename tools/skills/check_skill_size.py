#!/usr/bin/env python3
"""Enforce the ≤500-line SKILL.md guidance from the skill-creator skill.

The skill-creator's SKILL.md authoring guide recommends ≤500 lines per
SKILL.md, with deeper material pushed into ``references/``. This check
fails CI if any SKILL.md crosses that ceiling, with a hint to extract.

Exit 0 on clean, 1 on any oversize SKILL.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / "src" / "venice_ai" / "skills"
LIMIT = 500


def main() -> int:
    over: list[tuple[Path, int]] = []
    checked = 0

    for skill_md in sorted(SKILLS_ROOT.glob("venice-py*/SKILL.md")):
        line_count = sum(1 for _ in skill_md.open(encoding="utf-8"))
        checked += 1
        rel = skill_md.relative_to(REPO_ROOT)
        marker = "  " if line_count <= LIMIT else "❌"
        print(f"{marker} {line_count:>4} {rel}")
        if line_count > LIMIT:
            over.append((skill_md, line_count))

    print()
    if not checked:
        print(f"error: no SKILL.md files found under {SKILLS_ROOT}", file=sys.stderr)
        return 1
    if over:
        print(f"{len(over)} SKILL.md file(s) exceed {LIMIT} lines:")
        for path, n in over:
            print(f"  {path.relative_to(REPO_ROOT)}: {n} lines")
        print()
        print("Move deeper material into references/ files; SKILL.md is the index/router.")
        return 1
    print(f"All {checked} SKILL.md files within the {LIMIT}-line guidance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
