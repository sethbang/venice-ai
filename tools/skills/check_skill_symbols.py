#!/usr/bin/env python3
"""Static check that SKILL.md code blocks reference SDK symbols that EXIST.

The other skill checkers verify file references (``check_skill_examples``),
SKILL.md size (``check_skill_size``), and anti-patterns (``lint_skill_code``).
None of them verify that an imported name, a constructor keyword, or an
attribute the skill teaches actually exists on the installed ``venice_ai``
package. That gap let real bugs ship clean through ``make skills-check``:

  * ``RetryOptions(max_retries=...)``       — real field is ``max_attempts``
  * ``image.create(negative_prompt=...)``   — kwarg was removed
  * ``VoiceDetail(...).name`` / ``.locale`` — fields don't exist
  * ``from venice_ai... import NoSuchName`` — name doesn't exist

This checker parses every ```python fenced block under ``src/venice_ai/skills/**/*.md``
with ``ast`` and, for the constructs it can *confidently* resolve against the
importable SDK, verifies the symbol exists.

Design priority: **zero false positives.** A checker that breaks valid code is
worse than no checker. Whenever a symbol's type can't be resolved with
confidence — a dynamic value, ``**kwargs``, an external library, a pydantic
model with ``extra='allow'`` (arbitrary kwargs are valid there), a method whose
signature takes ``**kwargs`` — the construct is SKIPPED, not flagged.

Checks performed
----------------
1. **Imports** — ``from venice_ai... import X`` / ``import (...)``: every name
   must exist in the (importable) target module, OR be an importable submodule.
   If the module can't be imported because of a *third-party* optional
   dependency (``ModuleNotFoundError`` for a non-``venice_ai`` package, e.g.
   ``eth_account``), the block's imports are skipped — absence of the optional
   dep is not proof the symbol is wrong. A missing ``venice_ai.*`` submodule,
   however, IS a real structural finding.
2. **Constructor keyword args** — ``Foo(bar=...)`` where ``Foo`` resolves
   (via an in-block ``from venice_ai`` import) to a pydantic ``BaseModel``,
   a dataclass, or a plain class with an introspectable ``__init__``: each
   ``bar`` must be a real field / parameter (or alias). Models with
   ``extra='allow'`` and signatures with ``**kwargs`` accept arbitrary
   keywords and are skipped.
3. **Method keyword args** — ``client.<chain>.<method>(bar=...)`` where the
   receiver ``client`` was assigned ``VeniceClient(...)`` in the same block.
   The chain is resolved against a single offline ``VeniceClient`` oracle
   instance (it instantiates without network I/O). Methods whose signature
   takes ``**kwargs`` (e.g. ``chat.completions.create``) are skipped.
4. **Attribute access** — ``v.attr`` where ``v`` was assigned a resolvable
   pydantic/dataclass constructor in the same block (e.g.
   ``v = VoiceDetail(...); v.name``). Restricted to pydantic/dataclass
   instances so we never touch plain classes (whose resource attributes are
   set on the instance, not the class) — that keeps ``client.chat`` and
   friends out of scope.

Output is flake8-style ``file:line: message``. Exit 0 clean, 1 on findings.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import sys
from pathlib import Path

import pydantic

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / "src" / "venice_ai" / "skills"

# Make the in-tree SDK importable without an editable install.
sys.path.insert(0, str(REPO_ROOT / "src"))

# Attributes that exist dynamically on every Venice response model (attached at
# runtime, not declared as fields) or come from pydantic's BaseModel surface.
# Accessing these is always valid — never flag them.
_DYNAMIC_ATTR_WHITELIST = {
    "_response",  # VeniceBaseModel attaches the raw httpx response here
    "model_config",
    "model_fields",
    "model_extra",
    "model_fields_set",
    "model_computed_fields",
}


# ---------------------------------------------------------------------------
# Resolution oracle: one offline VeniceClient used to resolve `client.x.y`
# method chains. Instantiating with a placeholder key performs no network I/O.
# ---------------------------------------------------------------------------
def _build_client_oracle() -> object | None:
    try:
        from venice_ai import VeniceClient

        return VeniceClient(api_key="placeholder-for-static-analysis")
    except Exception:
        # If the client can't be constructed offline, we simply skip the
        # method-kwarg sub-check rather than crash.
        return None


_CLIENT_ORACLE = _build_client_oracle()


@dataclasses.dataclass
class Finding:
    path: Path
    line: int
    message: str


def _extract_python_blocks(md_text: str) -> list[tuple[int, str]]:
    """Return (markdown_line_of_first_body_line, body) for each ```python block.

    Mirrors the extraction in ``lint_skill_code.py`` so line numbers line up
    with what the other checkers report.
    """
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


def _try_import(module: str) -> object | None:
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        # A missing *third-party* optional dependency (e.g. `redis`,
        # `eth_account`) is not evidence the venice symbol is wrong — the SDK
        # gates those behind extras. Re-raise only for missing venice_ai.*
        # submodules, which ARE real structural problems.
        missing = (exc.name or "")
        if missing == "venice_ai" or missing.startswith("venice_ai."):
            raise
        return None
    except Exception:
        # Any other import-time failure (e.g. the optional dep is installed but
        # errors on import) — be conservative and skip.
        return None


def _is_importable_submodule(qualified: str) -> bool:
    """Best-effort probe: is ``qualified`` an importable submodule?

    Used to avoid flagging ``from pkg import submodule``. Never raises — a
    failed probe simply means "not a submodule", so the caller falls through
    to a missing-symbol finding.
    """
    try:
        return importlib.import_module(qualified) is not None
    except Exception:
        return False


def _resolvable_kwargs(obj: object) -> tuple[set[str] | None, bool]:
    """Return (valid_kwarg_names, accepts_arbitrary).

    ``valid_kwarg_names`` is ``None`` when the object's accepted keywords can't
    be determined with confidence (caller must then SKIP). ``accepts_arbitrary``
    is ``True`` when any keyword is valid (``extra='allow'`` model or a
    ``**kwargs`` signature) — caller must also SKIP in that case.
    """
    # pydantic BaseModel
    if isinstance(obj, type) and issubclass(obj, pydantic.BaseModel):
        if obj.model_config.get("extra") == "allow":
            return set(), True
        valid: set[str] = set()
        for name, field in obj.model_fields.items():
            valid.add(name)
            if field.alias:
                valid.add(field.alias)
            va = field.validation_alias
            if isinstance(va, str):
                valid.add(va)
        return valid, False

    # dataclass
    if dataclasses.is_dataclass(obj) and isinstance(obj, type):
        return {f.name for f in dataclasses.fields(obj)}, False

    # plain class with an introspectable __init__
    if isinstance(obj, type):
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            return None, False
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return set(sig.parameters) - {"self"}, True
        return set(sig.parameters) - {"self"}, False

    return None, False


def _attr_chain(node: ast.Attribute) -> tuple[str | None, list[str]]:
    """Decompose ``a.b.c`` into (base_name, ['b', 'c']).

    Returns (None, []) if the chain doesn't bottom out in a bare Name.
    """
    parts: list[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        return cur.id, list(reversed(parts))
    return None, []


def _check_block(body: str, md: Path, start_line: int) -> list[Finding]:
    findings: list[Finding] = []
    try:
        tree = ast.parse(body)
    except SyntaxError:
        # Skill blocks are sometimes intentionally fragmentary. Don't fail on
        # parse errors — same posture as lint_skill_code.py.
        return findings

    def loc(node: ast.AST) -> int:
        return start_line + getattr(node, "lineno", 1) - 1

    # symtab: name -> resolved venice_ai object (from in-block imports)
    symtab: dict[str, object] = {}
    # client_vars: names assigned `VeniceClient(...)` in this block
    client_vars: set[str] = set()
    # model_vars: name -> resolvable pydantic/dataclass *type* (from ctor assign)
    model_vars: dict[str, type] = {}

    # --- pass 1: imports -----------------------------------------------------
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if not node.module or not node.module.startswith("venice_ai"):
            continue
        try:
            module = _try_import(node.module)
        except ModuleNotFoundError:
            # Missing venice_ai.* submodule — the import target itself is bogus.
            findings.append(
                Finding(
                    md,
                    loc(node),
                    f"no module named '{node.module}' in venice_ai "
                    "(module path does not exist)",
                )
            )
            continue
        if module is None:
            # Unresolvable due to a third-party optional dep — skip this import.
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            if hasattr(module, alias.name):
                symtab[alias.asname or alias.name] = getattr(module, alias.name)
                continue
            # Maybe it's a submodule rather than an attribute.
            if _is_importable_submodule(f"{node.module}.{alias.name}"):
                continue
            findings.append(
                Finding(
                    md,
                    loc(node),
                    f"cannot import name '{alias.name}' from '{node.module}' "
                    "(symbol does not exist)",
                )
            )

    # --- pass 2: collect constructor assignments -----------------------------
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        func = node.value.func
        if not isinstance(func, ast.Name):
            continue
        if func.id == "VeniceClient":
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    client_vars.add(tgt.id)
            continue
        cls = symtab.get(func.id)
        if not (isinstance(cls, type)):
            continue
        is_pyd = issubclass(cls, pydantic.BaseModel)
        is_dc = dataclasses.is_dataclass(cls)
        if not (is_pyd or is_dc):
            continue
        # extra='allow' pydantic models can hold arbitrary attributes — don't
        # track them for attribute-existence checks.
        if is_pyd and cls.model_config.get("extra") == "allow":
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                model_vars[tgt.id] = cls

    # --- pass 3: constructor & method keyword args ---------------------------
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and node.keywords):
            continue
        func = node.func

        # 3a. Constructor: Foo(bar=...) where Foo is an imported venice symbol.
        if isinstance(func, ast.Name):
            obj = symtab.get(func.id)
            if obj is None:
                continue
            valid, arbitrary = _resolvable_kwargs(obj)
            if valid is None or arbitrary:
                continue
            for kw in node.keywords:
                if kw.arg is None:  # **spread — can't reason about it
                    continue
                if kw.arg not in valid:
                    findings.append(
                        Finding(
                            md,
                            loc(node),
                            f"{func.id}(...) has no parameter '{kw.arg}' "
                            f"(valid: {', '.join(sorted(valid)) or '<none>'})",
                        )
                    )
            continue

        # 3b. Method: client.<chain>.<method>(bar=...) on a VeniceClient var.
        if isinstance(func, ast.Attribute) and _CLIENT_ORACLE is not None:
            base, chain = _attr_chain(func)
            if base is None or base not in client_vars:
                continue
            target: object = _CLIENT_ORACLE
            ok = True
            for part in chain:
                if hasattr(target, part):
                    target = getattr(target, part)
                else:
                    ok = False
                    break
            if not ok:
                continue
            try:
                sig = inspect.signature(target)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                continue  # method accepts arbitrary kwargs — skip
            valid_params = set(sig.parameters) - {"self"}
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                if kw.arg not in valid_params:
                    dotted = f"{base}.{'.'.join(chain)}"
                    findings.append(
                        Finding(
                            md,
                            loc(node),
                            f"{dotted}(...) has no parameter '{kw.arg}' "
                            f"(valid: {', '.join(sorted(valid_params)) or '<none>'})",
                        )
                    )

    # --- pass 4: attribute access on resolvable model instances --------------
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
            continue
        cls = model_vars.get(node.value.id)
        if cls is None:
            continue
        if node.attr in _DYNAMIC_ATTR_WHITELIST:
            continue
        if issubclass(cls, pydantic.BaseModel):
            declared = set(cls.model_fields)
            # include aliases (you can read by either)
            for f in cls.model_fields.values():
                if f.alias:
                    declared.add(f.alias)
        else:  # dataclass
            declared = {f.name for f in dataclasses.fields(cls)}
        # Union with everything on the class (methods, properties, descriptors,
        # @computed_field, inherited helpers). This is deliberately permissive —
        # we only want to catch a name that exists *nowhere* on the type.
        class_surface = {m for m in dir(cls) if not m.startswith("__")}
        if node.attr in declared or node.attr in class_surface:
            continue
        findings.append(
            Finding(
                md,
                loc(node),
                f"{cls.__name__} has no attribute '{node.attr}'",
            )
        )

    return findings


def main() -> int:
    if _CLIENT_ORACLE is None:
        print(
            "note: VeniceClient could not be instantiated offline; "
            "the method-kwarg sub-check is disabled for this run.",
            file=sys.stderr,
        )

    all_findings: list[Finding] = []
    blocks_checked = 0
    files_checked = 0

    for skill_dir in sorted(SKILLS_ROOT.glob("venice-py*")):
        if not skill_dir.is_dir():
            continue
        for md in sorted(skill_dir.rglob("*.md")):
            files_checked += 1
            text = md.read_text(encoding="utf-8")
            for start_line, body in _extract_python_blocks(text):
                blocks_checked += 1
                all_findings.extend(_check_block(body, md, start_line))

    print(
        f"Checked SDK symbols in {blocks_checked} python code block(s) "
        f"across {files_checked} markdown file(s)."
    )
    if not files_checked:
        print(f"error: no skill markdown found under {SKILLS_ROOT}", file=sys.stderr)
        return 1
    if all_findings:
        print()
        for f in sorted(all_findings, key=lambda x: (str(x.path), x.line)):
            print(f"{f.path.relative_to(REPO_ROOT)}:{f.line}: {f.message}")
        print()
        print(
            f"{len(all_findings)} finding(s); a skill references an SDK symbol / "
            "keyword / attribute that does not exist. Fix the SKILL.md to match "
            "the installed venice_ai surface."
        )
        return 1
    print("All resolvable SDK symbols, keywords, and attributes exist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
