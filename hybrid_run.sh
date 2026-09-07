#!/usr/bin/env bash
set -euo pipefail

# Serial hybrid runner: main-branch Car first, current expD non-Car second.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

exec "$PYTHON_BIN" "${SCRIPT_DIR}/scripts/run_hybrid_prelabel.py" "$@"
