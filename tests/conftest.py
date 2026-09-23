"""测试根配置：把仓库根目录放进 sys.path（`tests/root/` 那批用例要 import
`geometry / filtering / tracking / classification / pipeline`）。

`tests/main_chain/` 那批用例针对 `main_chain/` 那棵已分叉的同名模块树，跑在自己的
子进程里（见 `tests/root/test_main_chain_suite.py`），由它自己的 conftest 设置 sys.path。
两棵树同名包内容不同，不能塞进同一个进程 —— 详见 `tests/README.md`。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
