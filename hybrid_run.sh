#!/usr/bin/env bash
set -euo pipefail

# Serial hybrid runner: main-branch Car first, current expD non-Car second.
# Picks a healthy OpenPCDet python (or $PYTHON) so the launcher can import
# the post-processing pipeline.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${PYTHON:-}" ]; then
  PYTHON_BIN="$PYTHON"
else
  PYTHON_BIN=""
  for candidate in \
    "$HOME/miniconda3/envs/openpcdet/bin/python" \
    "$HOME/anaconda3/envs/openpcdet/bin/python" \
    "$HOME/miniconda3/envs/sustechpoints/bin/python" \
    "$HOME/anaconda3/envs/sustechpoints/bin/python"; do
    if [ -x "$candidate" ] && "$candidate" -c "import numpy, scipy, torch, pcdet, spconv" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
  if [ -z "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
  fi
fi

exec "$PYTHON_BIN" "${SCRIPT_DIR}/scripts/run_hybrid_prelabel.py" "$@"
