#!/usr/bin/env bash
set -euo pipefail

# 本机已有依赖时的一键入口：只探测/复用本地环境，不自动联网安装。
# 需要自动安装依赖时，改用 scripts/run_five_class.sh。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

exec "$PYTHON_BIN" \
  "${SCRIPT_DIR}/scripts/run_five_class.py" \
  --skip-install \
  "$@"
