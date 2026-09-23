"""`main_chain/` 那棵树的用例：把它放到 sys.path[0]（`geometry / region / ...` 都取这棵）。"""
from __future__ import annotations

import sys
from pathlib import Path

MAIN_CHAIN_ROOT = Path(__file__).resolve().parents[2] / "main_chain"

if str(MAIN_CHAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_CHAIN_ROOT))
