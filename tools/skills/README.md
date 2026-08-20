# Venice Claude Code skills

Four [Claude Code skills](https://docs.claude.com/en/docs/claude-code/skills) that empower AI builders to write idiomatic Venice AI v2 code on the first try. They auto-load when their trigger contexts match — "Venice chat", "generate an image with Venice", "Venice rate limits", "Venice x402", etc.

| Skill | Purpose | Triggers on |
|---|---|---|
| **`venice-py`** | Setup, chat, streaming, tool calling / agent loops, structured output, model resolution, embeddings, characters, augment, response metadata, v1→v2 migration | `venice_ai` import, `VeniceClient`, "venice-py chat / stream / tool / agent" |
| **`venice-py-multimodal`** | Image, audio (TTS + STT), video, music generation; save patterns; async-job lifecycle | "venice-py image / TTS / STT / video / music" |
| **`venice-py-production`** | Retries with `RateLimitError.retry_after_seconds`, bounded concurrency via `client.gather()`, header-driven cost tracking, error taxonomy, observability, prompt caching | "production venice", "rate limit", "track cost", "retry / backoff venice" |
| **`venice-py-x402`** | `client.x402.*` (balance, transactions, top_up, top_up_with), SIWE auth via `X402Auth`, agent-framework wiring (Coinbase Agentkit, Eliza, x402-axios) | "x402 venice", "SIWE", "venice without API key", "venice from autonomous agent" |

## Install

```bash
# From the SDK repo root
make skills-install        # copy the skills into ~/.claude/skills/
make skills-symlink        # OR: symlink for live-edit during dev
make skills-uninstall      # remove the four Venice skills
```

The installer is also runnable directly:

```bash
tools/skills/install.sh                   # copy
tools/skills/install.sh --symlink         # symlink
tools/skills/install.sh --uninstall       # remove
tools/skills/install.sh --dry-run         # preview
SKILLS_DIR=~/elsewhere tools/skills/install.sh   # custom destination
```

After installing, open Claude Code in any project. The skills auto-trigger when their description's trigger contexts match — see each `SKILL.md`'s `description:` frontmatter for the phrasing.

## Anatomy

```
src/venice_ai/skills/<skill>/
├── SKILL.md           # the skill body — auto-loaded into Claude's context
├── references/*.md    # deeper docs the skill points at when needed
├── scripts/*.py       # helper scripts referenced by SKILL.md
└── evals/evals.json   # benchmark prompts
```

`SKILL.md` is the index/router — kept under the 500-line skill-creator guidance. References hold deep material. Scripts hold reusable utilities (e.g., `topup_eip3009.py` for x402 wallet top-up; `lint_v1_usage.py`, the standalone-script equivalent of `venice-py lint`).

The skill directories live inside the `venice_ai` Python package (`src/venice_ai/skills/`) and are shipped in the wheel. They are accessible at runtime via `importlib.resources.files("venice_ai") / "skills" / "<name>" / "SKILL.md"`.

## Validate

```bash
make skills-check          # size guard + example-path resolution + code-block lint
```

Four checks:
1. **`check_skill_size.py`** — every `SKILL.md` is ≤500 lines (skill-creator guidance).
2. **`check_skill_examples.py`** — every `examples/foo/bar.py` referenced in a SKILL.md or reference actually exists in `examples/`. Catches drift when example files get renamed.
3. **`lint_skill_code.py`** — extracts every \`\`\`python code block from skill markdown and pipes it through `venice-py lint`. Catches non-idiomatic patterns IN the skills (e.g., a skill teaching the wrong stream-iteration mode).
4. **`check_skill_symbols.py`** — verifies that SDK symbols (imports, constructor kwargs, attributes) referenced in skill code blocks actually exist in the installed package.

CI runs all four on every PR touching `src/venice_ai/skills/` or `tools/skills/`, every push to `main`, and weekly to catch slow drift.

## Develop

To iterate on a skill:

1. `make skills-symlink` (so edits in `src/venice_ai/skills/<skill>/` are live)
2. Edit + test in Claude Code
3. Validate: `make skills-check`
4. Commit
