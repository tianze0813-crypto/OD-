"""多雷达融合：静态外参 + 自车运动补偿（合成场景单测）。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.lidar_fusion import fuse_clip, fuse_frame


def _write_bin(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(rows, dtype=np.float32).reshape(-1, 4).tofile(str(path))


def _make_clip(root: Path, *, ego_speed: float = 5.0, dt_sec: float = 0.04,
               with_pose: bool = True):
    """top 帧在 T；front 帧在 T-dt；自车沿 +x 以 ego_speed 运动。

    base_from_lidar_top = I，base_from_lidar_front = +1m (x)；默认 dt=40ms、5m/s。
    front 点 (0,0,0) 静态融合后应落在 top 系 (1,0,0)；
    补偿后应落在 (1 - ego_speed*dt, 0, 0) = (0.8, 0, 0)。
    """
    T = 2_000_000_000
    t_front = int(T - dt_sec * 1e9)
    calib = {"tf2base_link": {
        "pose": np.eye(4).tolist(),
        "lidar_top": np.eye(4).tolist(),
        "lidar_front": [[1, 0, 0, 1.0], [0, 1, 0, 0],
                        [0, 0, 1, 0], [0, 0, 0, 1]],
    }}
    (root / "transforms").mkdir(parents=True, exist_ok=True)
    (root / "transforms" / "calib.json").write_text(json.dumps(calib), encoding="utf-8")
    if with_pose:
        rows = []
        for ts in (t_front, T):
            x = (ts - t_front) / 1e9 * ego_speed
            rows.append(f"{ts},{x},0,0,0,0,0,1")
        (root / "transforms" / "pose_data.txt").write_text(
            "\n".join(rows) + "\n", encoding="utf-8")
    _write_bin(root / "lidar" / "lidar_top" / f"{T}.bin", [[0.0, 0.0, 0.0, 1.0]])
    _write_bin(root / "lidar" / "lidar_front" / f"{t_front}.bin",
               [[0.0, 0.0, 0.0, 1.0]])
    return T, t_front


class LidarFusionTest(unittest.TestCase):
    def test_static_fusion_uses_only_extrinsics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            T, _ = _make_clip(root)
            fused, stats = fuse_frame(root, T, compensate_ego=False)
            xs = sorted(round(float(v), 4) for v in fused[:, 0])
            self.assertEqual(xs, [0.0, 1.0])
            self.assertFalse(stats["attached"]["front"]["compensated"])

    def test_ego_motion_compensation_shifts_aux_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            T, _ = _make_clip(root, ego_speed=5.0, dt_sec=0.04)
            fused, stats = fuse_frame(root, T, compensate_ego=True)
            xs = sorted(round(float(v), 4) for v in fused[:, 0])
            # ego moved +0.2m since the front frame -> point appears 0.2m closer
            self.assertEqual(xs, [0.0, 0.8])
            self.assertTrue(stats["attached"]["front"]["compensated"])

    def test_missing_pose_falls_back_to_static(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            T, _ = _make_clip(root, with_pose=False)
            fused, stats = fuse_frame(root, T, compensate_ego=True)
            xs = sorted(round(float(v), 4) for v in fused[:, 0])
            self.assertEqual(xs, [0.0, 1.0])
            self.assertFalse(stats["attached"]["front"]["compensated"])

    def test_fuse_clip_names_output_by_top_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            T, _ = _make_clip(root)
            out = Path(directory) / "fused"
            stats = fuse_clip(root, out, compensate_ego=True)
            self.assertEqual(stats["compensated_frames"], 1)
            self.assertTrue((out / f"{T}.bin").is_file())


if __name__ == "__main__":
    unittest.main()
