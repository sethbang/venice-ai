#!/usr/bin/env bash
# Build a wheel + sdist and smoke-install each extras combination in a
# clean venv. Verifies every extra resolves and the package imports.
#
# Usage:
#   tools/install_matrix.sh
#
# Env:
#   PYTHON       — Python interpreter to use for venvs (default python3.13)
#   TMPDIR       — base for ephemeral venvs (default /tmp)
#   KEEP_VENVS   — if set, don't rm -rf the venvs after run

set -euo pipefail

cd "$(dirname "$0")/.."
DIST_DIR="${PWD}/dist"
TMPDIR="${TMPDIR:-/tmp}"
PYTHON="${PYTHON:-python3.13}"

echo "==> Building wheel + sdist"
poetry build

WHEEL=$(ls -t "$DIST_DIR"/venice_py-*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
  echo "error: no venice_py-*.whl in $DIST_DIR" >&2
  exit 1
fi
echo "==> Wheel: $WHEEL"

EXTRAS=(
  ""           # base
  "[redis]"
  "[adaptive]"
  "[x402]"
  "[cli]"
  "[observability]"
  "[metrics]"
  "[enterprise]"
  "[all]"
)

PASS=()
FAIL=()

for extra in "${EXTRAS[@]}"; do
  label="${extra:-base}"
  venv="$TMPDIR/venice-im-${label//[\[\]]/}"
  echo ""
  echo "==> $extra"
  rm -rf "$venv"
  $PYTHON -m venv "$venv"
  "$venv/bin/pip" install --quiet --upgrade pip
  if "$venv/bin/pip" install --quiet "${WHEEL}${extra}" \
    && "$venv/bin/python" -c "from venice_ai import VeniceClient, __version__; print(f'  v{__version__} OK ${label}')"; then
    PASS+=("$label")
  else
    FAIL+=("$label")
    echo "  !! FAIL $label"
  fi
  if [ -z "${KEEP_VENVS:-}" ]; then
    rm -rf "$venv"
  fi
done

echo ""
echo "===== Summary ====="
echo "PASS: ${#PASS[@]} (${PASS[*]})"
echo "FAIL: ${#FAIL[@]} (${FAIL[*]})"

[ "${#FAIL[@]}" -eq 0 ]
