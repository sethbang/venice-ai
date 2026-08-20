#!/usr/bin/env bash
# Install the Venice AI Claude Code skills into ~/.claude/skills/.
#
# By default the skills are COPIED so they're stable against future
# `git pull`s of this repo. Pass `--symlink` for a developer setup where
# edits in `src/venice_ai/skills/<skill>/` show up live in your installed skills.
#
# Usage:
#   tools/skills/install.sh           # copy
#   tools/skills/install.sh --symlink # symlink (live edits)
#   tools/skills/install.sh --uninstall  # remove the four Venice skills
#   tools/skills/install.sh --dry-run    # print what would happen
#   SKILLS_DIR=~/elsewhere tools/skills/install.sh   # custom destination
#
# Env:
#   SKILLS_DIR  destination skills root (default: ~/.claude/skills)
#   FORCE       if set, overwrite existing destinations without prompting

set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="${PWD}"
SKILLS_SRC_DIR="${REPO_ROOT}/src/venice_ai/skills"
SKILLS_DEST_DIR="${SKILLS_DIR:-${HOME}/.claude/skills}"

SKILLS=(venice-py venice-py-multimodal venice-py-production venice-py-x402)
# Directories these skills occupied before the rename. Both generations trigger
# in Claude Code if a pre-rename install is left behind, so installing and
# uninstalling both clear them. Symlinks count: `--symlink` setups from before
# the rename now dangle.
LEGACY_SKILLS=(venice-ai venice-ai-multimodal venice-ai-production venice-ai-x402)

mode="copy"
dry_run=0
uninstall=0

for arg in "$@"; do
  case "$arg" in
    --symlink|-s)  mode="symlink" ;;
    --copy|-c)     mode="copy" ;;
    --uninstall|--remove) uninstall=1 ;;
    --dry-run|-n)  dry_run=1 ;;
    --help|-h)
      sed -n '1,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "error: unknown argument: $arg" >&2
      echo "Run with --help to see usage." >&2
      exit 2
      ;;
  esac
done

run() {
  if [ "$dry_run" -eq 1 ]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

remove_superseded() {
  for skill in "${LEGACY_SKILLS[@]}"; do
    target="${SKILLS_DEST_DIR}/${skill}"
    if [ -e "$target" ] || [ -L "$target" ]; then
      run rm -rf "$target"
      echo "  removed superseded ${target}"
    fi
  done
}

if [ "$uninstall" -eq 1 ]; then
  echo "==> Uninstalling Venice skills from ${SKILLS_DEST_DIR}"
  for skill in "${SKILLS[@]}"; do
    target="${SKILLS_DEST_DIR}/${skill}"
    if [ -e "$target" ] || [ -L "$target" ]; then
      run rm -rf "$target"
      echo "  removed ${target}"
    fi
  done
  remove_superseded
  echo "==> Done."
  exit 0
fi

# Sanity: source dirs exist
for skill in "${SKILLS[@]}"; do
  src="${SKILLS_SRC_DIR}/${skill}"
  if [ ! -d "$src" ]; then
    echo "error: source skill missing: ${src}" >&2
    exit 1
  fi
done

run mkdir -p "${SKILLS_DEST_DIR}"

echo "==> Installing Venice skills (mode=${mode}) into ${SKILLS_DEST_DIR}"
remove_superseded
for skill in "${SKILLS[@]}"; do
  src="${SKILLS_SRC_DIR}/${skill}"
  dest="${SKILLS_DEST_DIR}/${skill}"

  if [ -e "$dest" ] || [ -L "$dest" ]; then
    if [ -z "${FORCE:-}" ]; then
      printf "  exists: %s\n" "$dest"
      printf "  Overwrite? [y/N] "
      read -r reply
      case "$reply" in
        y|Y|yes|YES) ;;
        *) echo "  skipped"; continue ;;
      esac
    fi
    run rm -rf "$dest"
  fi

  case "$mode" in
    symlink)
      run ln -s "$src" "$dest"
      echo "  symlinked ${dest} -> ${src}"
      ;;
    copy)
      run cp -R "$src" "$dest"
      echo "  copied ${dest}"
      ;;
  esac
done

echo "==> Done. Verify with: ls ${SKILLS_DEST_DIR}/venice-py*"
echo
echo "To use the skills: open Claude Code in any project. The skills"
echo "auto-load when their trigger contexts match (e.g., 'venice chat',"
echo "'venice image', 'venice x402'). See each SKILL.md for trigger phrasing."
