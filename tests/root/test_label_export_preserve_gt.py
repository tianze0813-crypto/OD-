"""`_write_labels` 不许删 ``label/`` 下的非预测子目录（人工 GT 等）。

回归用例：2026-10-17 在 P2 测试集上，旧实现 ``rmtree(<clip>/label)`` 把
``label/det_gt`` 一起删了。现在只清 ``label/det`` + 平铺的 ``label/*.json``。
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


_SCRIPT = Path(__file__).parents[2] / "scripts" / "run_hybrid_prelabel.py"
_SPEC = importlib.util.spec_from_file_location("hybrid_launcher_label", _SCRIPT)
launcher = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(launcher)


class WriteLabelsPreservesGtTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.clip = Path(self._tmp.name) / "clip_P2"
        self.clip.mkdir(parents=True)
        self.label = self.clip / "label"
        (self.label / "det").mkdir(parents=True)
        (self.label / "det" / "1.json").write_text("[]\n", encoding="utf-8")
        (self.label / "det" / "2.json").write_text("[]\n", encoding="utf-8")
        (self.label / "det_gt").mkdir()
        (self.label / "det_gt" / "1.json").write_text(
            json.dumps([{"obj_type": "Truck"}]), encoding="utf-8")
        (self.label / "det_010").mkdir()
        (self.label / "det_010" / "1.json").write_text("[]\n", encoding="utf-8")
        (self.label / "1.json").write_text("[]\n", encoding="utf-8")   # 平铺老产物

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_gt_and_other_subdirs_survive(self) -> None:
        frames = [{"frame_id": "9", "labels": [
            {"obj_id": "1000", "obj_type": "Truck"}]}]
        total = launcher._write_labels(frames, self.clip)

        self.assertEqual(total, 1)
        self.assertTrue((self.label / "det" / "9.json").is_file())
        # 上一轮预测被清掉
        self.assertFalse((self.label / "det" / "1.json").exists())
        self.assertFalse((self.label / "det" / "2.json").exists())
        # 平铺老产物被清掉
        self.assertFalse((self.label / "1.json").exists())
        # GT / 其它子目录必须原样保留
        gt = self.label / "det_gt" / "1.json"
        self.assertTrue(gt.is_file())
        self.assertEqual(json.loads(gt.read_text(encoding="utf-8")),
                         [{"obj_type": "Truck"}])
        self.assertTrue((self.label / "det_010" / "1.json").is_file())

    def test_flat_label_dir_without_det_subdir(self) -> None:
        """DET_SUBDIR 置空（回退到平铺 label/）时也不能误删同级目录。"""
        (self.clip / "label2").mkdir()
        original = launcher.DET_SUBDIR
        try:
            launcher.DET_SUBDIR = ""
            launcher._write_labels(
                [{"frame_id": "9", "labels": []}], self.clip)
        finally:
            launcher.DET_SUBDIR = original
        self.assertTrue((self.label / "9.json").is_file())
        self.assertFalse((self.label / "1.json").exists())
        self.assertTrue((self.label / "det_gt" / "1.json").is_file())


if __name__ == "__main__":
    unittest.main()
