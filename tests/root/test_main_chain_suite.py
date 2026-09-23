"""把 `tests/main_chain/` 那套用例挂进默认的 pytest 运行。

为什么要子进程：仓库里有两棵同名但已分叉的模块树（仓库根 / `main_chain/`），
一个进程里 `geometry / filtering / tracking / pipeline` 只能有一棵生效，而两套用例
分别需要不同的那棵（`main_chain` 的用例还用字符串打桩 `mock.patch("region.retrack...")`，
运行期按 sys.modules 解析）。所以这里用独立子进程跑，互不干扰。

  单独调试时也可以直接：  pytest tests/main_chain
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MAIN_CHAIN_TESTS = Path(__file__).resolve().parents[1] / "main_chain"


def test_main_chain_suite_passes():
    assert MAIN_CHAIN_TESTS.is_dir(), f"缺少测试目录：{MAIN_CHAIN_TESTS}"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", str(MAIN_CHAIN_TESTS), "-q"],
        text=True, capture_output=True)
    tail = "\n".join(
        line for line in completed.stdout.splitlines()
        if line.strip() and ("passed" in line or "failed" in line or "error" in line))
    assert completed.returncode == 0, (
        f"main_chain 测试套件失败（rc={completed.returncode}）：\n{tail}\n"
        f"单独复现：pytest {MAIN_CHAIN_TESTS}")
