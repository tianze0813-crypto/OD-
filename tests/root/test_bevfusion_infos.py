"""BEVFusion infos 的并发安全写（per-clip + 聚合合并）。

背景：聚合 infos 是全局单文件，之前每次整体重写 —— 两个进程/交错跑不同 clip 时互相覆盖，
推理端按 scene_token 过滤匹配不到就静默输出 0 帧，最后车链产出「只有 VRU 标签」的半成品。
"""
from __future__ import annotations

import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path

_PATH = (Path(__file__).parents[2] / "bevfusion" / "scripts" / "mmdet3d_prep.py")
_SPEC = importlib.util.spec_from_file_location("mmdet3d_prep", _PATH)
mmdet3d_prep = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(mmdet3d_prep)


def _info(clip, ts):
    return {"scene_token": clip, "timestamp": ts,
            "lidar_points": {"lidar_path": f"{clip}/lidar/lidar_top/{ts}.bin"}}


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)["data_list"]


class InfosWriteTest(unittest.TestCase):
    def test_per_clip_files_and_aggregate_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            mmdet3d_prep.write_infos(out, [_info("clipA", 1), _info("clipA", 2)])
            mmdet3d_prep.write_infos(out, [_info("clipB", 3)])
            # per-clip 各一份
            self.assertTrue((out / "clipA_infos.pkl").is_file())
            self.assertTrue((out / "clipB_infos.pkl").is_file())
            self.assertEqual(len(_load(out / "clipA_infos.pkl")), 2)
            self.assertEqual(len(_load(out / "clipB_infos.pkl")), 1)
            # 聚合：A 的条目不被 B 这次写冲掉
            agg = {x["scene_token"] for x in _load(out / "police_mmdet3d_infos.pkl")}
            self.assertEqual(agg, {"clipA", "clipB"})

    def test_rewrite_same_clip_replaces_only_that_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            mmdet3d_prep.write_infos(out, [_info("clipA", 1), _info("clipB", 2)])
            mmdet3d_prep.write_infos(out, [_info("clipB", 3), _info("clipB", 4)])
            data = _load(out / "police_mmdet3d_infos.pkl")
            per = {}
            for item in data:
                per.setdefault(item["scene_token"], []).append(item["timestamp"])
            self.assertEqual(per, {"clipA": [1], "clipB": [3, 4]})

    def test_broken_aggregate_is_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "police_mmdet3d_infos.pkl").write_bytes(b"not a pickle")
            stats = mmdet3d_prep.write_infos(out, [_info("clipA", 1)])
            self.assertEqual(stats["per_clip"], 1)
            self.assertEqual(len(_load(out / "police_mmdet3d_infos.pkl")), 1)


if __name__ == "__main__":
    unittest.main()
