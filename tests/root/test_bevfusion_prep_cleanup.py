"""step1 的 BEVFusion 预处理缓存必须用完即删（默认不保留过程数据）。

背景：prep_data.py 会把每个 clip 的 lidar_top 4 列 bin 复制成 5 列（LoadPointsFromFile
写死 load_dim=5），单帧 ~1.9MB、一个 80 帧的包 ~150MB；再加 transforms / infos / 去畸变图。
以前这份缓存只写不删，批量跑几百个包后 bevfusion/data/ 涨到 50G。

这里只测清理逻辑本身（不跑 mmdet3d）：清干净本 clip、不碰别的 clip、不碰共享软链。
"""
from __future__ import annotations

import argparse
import importlib.util
import unittest.mock
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.conftest import REPO_ROOT

_PATH = REPO_ROOT / "pipeline" / "step1_bevfusion_truck.py"
_SPEC = importlib.util.spec_from_file_location("step1_bevfusion_truck", _PATH)
step1 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(step1)


def _write_infos(path: Path, entries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"metainfo": {"dataset": "police_bevfusion"}, "data_list": entries}, f)


def _load_tokens(path: Path):
    with open(path, "rb") as f:
        return [x["scene_token"] for x in pickle.load(f)["data_list"]]


class PreprocessCleanupTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        root = Path(self._tmp.name) / "bevfusion"
        self.root = root
        # 一个 clip 的 prep 落盘产物（真实目录名/结构）
        for clip in ("clipA", "clipB"):
            (root / "data" / "police" / clip / "lidar" / "lidar_top").mkdir(parents=True)
            (root / "data" / "police" / clip / "lidar" / "lidar_top" / "1.bin").write_bytes(b"\0" * 64)
            (root / "data" / "police" / clip / "transforms").mkdir()
            (root / "data" / "police" / clip / "transforms" / "pose_data.txt").write_text("p")
            (root / "work" / "undist" / clip / "cam_front").mkdir(parents=True)
            (root / "work" / "undist" / clip / "cam_front" / "1.jpg").write_bytes(b"jpg")
            _write_infos(root / "work" / "infos" / f"{clip}_infos.pkl",
                         [{"scene_token": clip, "timestamp": 1}])
        _write_infos(root / "work" / "infos" / "police_mmdet3d_infos.pkl",
                     [{"scene_token": "clipA", "timestamp": 1},
                      {"scene_token": "clipB", "timestamp": 2}])
        _write_infos(root / "work" / "infos" / "police_val_infos.pkl",
                     [{"scene_token": "clipA", "timestamp": 1}])
        # prep_data 会在 data/police 下建一个指向 work/infos 的软链（共享，不能删）
        (root / "data" / "police" / "infos").symlink_to((root / "work" / "infos").resolve())
        self._orig_root = step1.BEVFUSION_ROOT
        step1.BEVFUSION_ROOT = root

    def tearDown(self):
        step1.BEVFUSION_ROOT = self._orig_root
        self._tmp.cleanup()

    def test_removes_all_process_data_of_that_clip_only(self):
        removed = step1.cleanup_prep(self.root / "data" / "police" / "clipA")

        # 本 clip 的数据/图/infos 全没了
        self.assertFalse((self.root / "data" / "police" / "clipA").exists())
        self.assertFalse((self.root / "work" / "undist" / "clipA").exists())
        self.assertFalse((self.root / "work" / "infos" / "clipA_infos.pkl").exists())
        self.assertGreaterEqual(len(removed), 3)

        # 别的 clip 一点没动
        self.assertTrue((self.root / "data" / "police" / "clipB" / "lidar" / "lidar_top" / "1.bin").is_file())
        self.assertTrue((self.root / "work" / "undist" / "clipB" / "cam_front" / "1.jpg").is_file())
        self.assertTrue((self.root / "work" / "infos" / "clipB_infos.pkl").is_file())

        # 聚合 infos 只摘掉本 clip（并发跑别的 clip 时不能被整份删掉）
        agg = self.root / "work" / "infos" / "police_mmdet3d_infos.pkl"
        self.assertEqual(_load_tokens(agg), ["clipB"])

        # prep_data 的冗余 val infos 也清掉；共享的 data/police/infos 软链要留着
        self.assertFalse((self.root / "work" / "infos" / "police_val_infos.pkl").exists())
        self.assertTrue((self.root / "data" / "police" / "infos").is_symlink())

    def test_aggregate_infos_removed_when_no_clip_left(self):
        step1.cleanup_prep(self.root / "data" / "police" / "clipA")
        step1.cleanup_prep(self.root / "data" / "police" / "clipB")
        self.assertFalse((self.root / "work" / "infos" / "police_mmdet3d_infos.pkl").exists())
        self.assertFalse((self.root / "data" / "police" / "clipB").exists())

    def test_broken_aggregate_is_dropped_not_fatal(self):
        agg = self.root / "work" / "infos" / "police_mmdet3d_infos.pkl"
        agg.write_bytes(b"not a pickle")
        step1.cleanup_prep(self.root / "data" / "police" / "clipA")   # 不该抛
        self.assertFalse(agg.exists())

    def test_cleanup_is_idempotent(self):
        step1.cleanup_prep(self.root / "data" / "police" / "clipA")
        self.assertEqual(step1.cleanup_prep(self.root / "data" / "police" / "clipA"), [])


class KeepPrepSwitchTest(unittest.TestCase):
    def test_flag_and_env(self):
        class Args:
            keep_prep = False

        self.assertFalse(step1.keep_prep_enabled(Args()))
        Args.keep_prep = True
        self.assertTrue(step1.keep_prep_enabled(Args()))

    def test_env_switch(self):
        import os

        class Args:
            keep_prep = False

        old = os.environ.get(step1.PREP_KEEP_ENV)
        try:
            os.environ[step1.PREP_KEEP_ENV] = "1"
            self.assertTrue(step1.keep_prep_enabled(Args()))
            os.environ[step1.PREP_KEEP_ENV] = "0"
            self.assertFalse(step1.keep_prep_enabled(Args()))
        finally:
            if old is None:
                os.environ.pop(step1.PREP_KEEP_ENV, None)
            else:
                os.environ[step1.PREP_KEEP_ENV] = old


class RunInferenceCleanupWiringTest(unittest.TestCase):
    """run_inference 调完（哪怕推理抛异常）必须把 prep 缓存清掉，除非 --keep-prep。"""

    def _args(self):
        return argparse.Namespace(
            work_root=self.work, mode="lidar", skip_prepare=False, jobs=1,
            cfg=step1.BEVFUSION_CFG, ckpt=step1.BEVFUSION_CKPT, score_thresh=0.1,
            z_convention="center", no_visibility_check=True, vis_occl_tol=0.3,
            keep_prep=False)

    def setUp(self):
        self._tmp = TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.root = tmp / "bevfusion"
        self.work = tmp / "work_out"
        self.work.mkdir()
        self.clip = tmp / "clipA"
        (self.clip / "lidar" / "lidar_top").mkdir(parents=True)
        (self.clip / "lidar" / "lidar_top" / "1.bin").write_bytes(b"\0" * 16)
        (self.clip / "transforms").mkdir()
        (self.clip / "transforms" / "calib.json").write_text("{}")
        self._orig_root = step1.BEVFUSION_ROOT
        step1.BEVFUSION_ROOT = self.root

    def tearDown(self):
        step1.BEVFUSION_ROOT = self._orig_root
        self._tmp.cleanup()

    def _fake_prepare(self, clip, jobs=6, images=True):
        """冒充 prep_data.py：造出真实的缓存落盘结构。"""
        cache = self.root / "data" / "police" / clip.name
        (cache / "lidar" / "lidar_top").mkdir(parents=True)
        (cache / "lidar" / "lidar_top" / "1.bin").write_bytes(b"\0" * 64)
        (cache / "transforms").mkdir()
        (cache / "transforms" / "calib.json").write_text("{}")
        _write_infos(self.root / "work" / "infos" / f"{clip.name}_infos.pkl",
                     [{"scene_token": clip.name, "timestamp": 1}])
        _write_infos(self.root / "work" / "infos" / "police_mmdet3d_infos.pkl",
                     [{"scene_token": clip.name, "timestamp": 1}])

    def _fake_infer(self, clip, raw_json, *a, **kw):
        Path(raw_json).parent.mkdir(parents=True, exist_ok=True)
        Path(raw_json).write_text("[]")

    def test_cache_removed_after_successful_run(self):
        with unittest.mock.patch.object(step1, "prepare", self._fake_prepare), \
             unittest.mock.patch.object(step1, "infer", self._fake_infer):
            raw = step1.run_inference(self.clip, self._args())
        self.assertTrue(raw.is_file())                                     # 产物在
        self.assertFalse((self.root / "data" / "police" / "clipA").exists())  # 缓存没了
        self.assertFalse((self.root / "work" / "infos" / "clipA_infos.pkl").exists())

    def test_cache_removed_even_when_infer_fails(self):
        def boom(*a, **kw):
            raise RuntimeError("CUDA out of memory")

        with unittest.mock.patch.object(step1, "prepare", self._fake_prepare), \
             unittest.mock.patch.object(step1, "infer", boom):
            with self.assertRaises(RuntimeError):
                step1.run_inference(self.clip, self._args())
        self.assertFalse((self.root / "data" / "police" / "clipA").exists())

    def test_keep_prep_flags_keep_the_cache(self):
        args = self._args()
        args.keep_prep = True
        with unittest.mock.patch.object(step1, "prepare", self._fake_prepare), \
             unittest.mock.patch.object(step1, "infer", self._fake_infer):
            step1.run_inference(self.clip, args)
        self.assertTrue((self.root / "data" / "police" / "clipA" / "lidar" / "lidar_top" / "1.bin").is_file())
        self.assertTrue((self.root / "work" / "infos" / "clipA_infos.pkl").is_file())


if __name__ == "__main__":
    unittest.main()
