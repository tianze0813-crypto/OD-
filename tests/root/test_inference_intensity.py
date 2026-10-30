"""推理侧强度缩放必须与训练对齐（训练 0~255，推理不能再 /255）。

2026-10-21 修：run_prelabel.py 的 --intensity_scale 默认曾是 1/255，
而 nuScenes/VOD/Police 训练 loader 都不归一化强度（只有 PandaSet 会 /255），
导致强度通道被压 255 倍、基本失效。这里用 AST 守住默认值，别再漂回去。
"""

import ast
import unittest
from pathlib import Path

RUN_PRELABEL = Path(__file__).resolve().parents[2] / "inference" / "run_prelabel.py"


def _intensity_scale_default():
    tree = ast.parse(RUN_PRELABEL.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and first.value == "--intensity_scale"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                return ast.literal_eval(keyword.value)
    raise AssertionError("run_prelabel.py 里找不到 --intensity_scale 的默认值")


class InferenceIntensityTest(unittest.TestCase):
    def test_default_scale_is_identity(self):
        """默认必须是 ×1.0（原样送入），否则又和训练差 255 倍。"""
        self.assertEqual(_intensity_scale_default(), 1.0)

    def test_no_hardcoded_255_in_inference(self):
        """推理源码里不应再出现写死的 /255 缩放。"""
        source = RUN_PRELABEL.read_text(encoding="utf-8")
        self.assertNotIn("/= 255", source)
        self.assertNotIn("/ 255.0", source)


if __name__ == "__main__":
    unittest.main()
