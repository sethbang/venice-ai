"""Unit tests for the examples swallow-to-exit-0 checker.

The checker (``tools/examples/check_no_swallow.py``) must catch the
swallow-to-exit-0 anti-pattern the live audit found, while staying zero
false-positive on the remediated tree. Three high-precision rules:

- A: a *broad* except (``Exception`` / ``BaseException`` / bare) returning a
  truthy literal — returning success from an error handler.
- B: a broad except whose body is pure print/log + fall-through (no ``raise``,
  ``sys.exit``, assignment, or value-returning ``return``).
- C: if ``main`` is annotated ``-> int``, its result must reach ``sys.exit``.

Narrow typed excepts (``except ValueError`` / ``except (VeniceError, APIError)``)
are intentional error-demos and are never flagged.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "examples" / "check_no_swallow.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_no_swallow", TOOL_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass field-type resolution can find the module.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


find_violations = _load().find_violations


def _rules(src: str) -> set[str]:
    return {v.rule for v in find_violations("snippet.py", textwrap.dedent(src))}


# ---- clean cases (must NOT flag) ----


def test_ok_tracking_return_false_is_clean():
    assert (
        _rules(
            """
        async def demo() -> bool:
            try:
                await thing()
                return True
            except Exception as e:
                print(f"err: {e}")
                return False
        """
        )
        == set()
    )


def test_assignment_ok_flag_is_clean():
    assert (
        _rules(
            """
        async def demo() -> bool:
            ok = True
            for x in items:
                try:
                    await thing(x)
                except Exception as e:
                    print(e)
                    ok = False
                    continue
            return ok
        """
        )
        == set()
    )


def test_reraise_is_clean():
    assert (
        _rules(
            """
        try:
            work()
        except Exception as e:
            print(e)
            raise
        """
        )
        == set()
    )


def test_narrow_typed_except_is_carved_out():
    # Intentional error-demo: returning True from a *typed* catch is fine.
    assert (
        _rules(
            """
        async def error_handling() -> bool:
            try:
                await scrape(bad_url)
            except (VeniceError, APIError) as e:
                print(f"handled as designed: {e}")
                return True
            return True
        """
        )
        == set()
    )


# ---- Rule A: broad except returns success ----


def test_rule_a_broad_except_returns_true():
    assert "A" in _rules(
        """
        async def demo() -> bool:
            try:
                await thing()
            except Exception as e:
                print(e)
                return True
        """
    )


# ---- Rule B: broad except swallows (print + fall-through) ----


def test_rule_b_print_then_fallthrough():
    assert "B" in _rules(
        """
        async def demo():
            try:
                await thing()
            except Exception as e:
                print(f"error: {e}")
        """
    )


def test_rule_b_print_then_continue():
    assert "B" in _rules(
        """
        async def demo():
            for x in items:
                try:
                    await thing(x)
                except Exception as e:
                    print(e)
                    continue
        """
    )


# ---- Rule C: main -> int result must reach sys.exit ----


def test_rule_c_discarded_int_main_flagged():
    assert "C" in _rules(
        """
        async def main() -> int:
            return 0

        if __name__ == "__main__":
            asyncio.run(main())
        """
    )


def test_rule_c_inline_sys_exit_is_clean():
    assert "C" not in _rules(
        """
        async def main() -> int:
            return 0

        if __name__ == "__main__":
            sys.exit(asyncio.run(main()))
        """
    )


def test_rule_c_assigned_then_exit_is_clean():
    assert "C" not in _rules(
        """
        async def main() -> int:
            return 0

        if __name__ == "__main__":
            rc = asyncio.run(main())
            sys.exit(rc)
        """
    )


def test_rule_c_none_main_raise_based_is_clean():
    # main() -> None that raises on failure; __main__ exits via except. Not flagged.
    assert "C" not in _rules(
        """
        async def main() -> None:
            await work()

        if __name__ == "__main__":
            try:
                asyncio.run(main())
            except Exception as e:
                print(e, file=sys.stderr)
                sys.exit(1)
        """
    )


# ---- the real audit fixture: the pre-remediation swallow must be caught ----


def test_pre_remediation_swallow_fixture_is_flagged():
    """The pre-Wave1 tool_calling.py (a real swallow) must be flagged; if it
    isn't, the checker doesn't actually work."""
    import subprocess

    old = subprocess.run(
        ["git", "show", "ec1be8a:examples/chat/tool_calling.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if old.returncode != 0:
        pytest.skip("pre-Wave1 fixture commit not available in this checkout")
    violations = find_violations("examples/chat/tool_calling.py", old.stdout)
    assert violations, "pre-remediation tool_calling.py must trip the swallow checker"


# ---- hard gate vs advisory + zero-FP on the real tree ----

_mod = _load()


def test_examples_tree_has_no_hard_violations():
    """The remediated examples/ tree must be clean of hard (A/C) findings —
    the zero-false-positive bar for the CI gate."""
    hard: list[str] = []
    for path in sorted((REPO_ROOT / "examples").rglob("*.py")):
        for v in find_violations(str(path), path.read_text(encoding="utf-8")):
            if v.rule in _mod.HARD_RULES:
                hard.append(f"{path}:{v.line} [{v.rule}] {v.message}")
    assert not hard, "hard (A/C) findings on the current tree:\n" + "\n".join(hard)


def test_main_default_passes_on_current_tree():
    """Default gate (A/C hard, B advisory) passes on the remediated tree."""
    assert _mod.main([str(REPO_ROOT / "examples")]) == 0


def test_main_strict_fails_when_advisory_b_present():
    """--strict promotes Rule B to fatal; the tree has known B patterns, so it
    must fail under --strict (proves the advisory path is wired)."""
    assert _mod.main([str(REPO_ROOT / "examples"), "--strict"]) == 1
