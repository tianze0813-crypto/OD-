import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from filtering.camera_visibility import load_base_link_points, load_clip_cameras
from tracking.tracker_conservative import CoordinateProvider


class CoordinateFrameContractTest(unittest.TestCase):
    def _write_clip(self, root: Path, include_top: bool = True) -> None:
        transforms = root / "transforms"
        transforms.mkdir()
        pose = np.eye(4)
        pose[0, 3] = 10.0
        top = np.eye(4)
        top[0, 3] = 2.0
        cam = np.eye(4)
        cam[1, 3] = 3.0
        tf = {"pose": pose.tolist(), "cam_front": cam.tolist()}
        if include_top:
            tf["lidar_top"] = top.tolist()
        calib = {
            "tf2base_link": tf,
            "cam_front": {
                "K": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "D": [0.0, 0.0, 0.0, 0.0],
                "imgw": 10,
                "imgh": 10,
            },
        }
        (transforms / "calib.json").write_text(
            json.dumps(calib), encoding="utf-8")
        (transforms / "pose_data.txt").write_text(
            "0,0,0,0,0,0,0,1\n", encoding="utf-8")

    def test_camera_projection_uses_base_link_contract(self):
        """【改动 2026-10-14】与 moga_data-main 的投影可视化同口径：检测系 = base_link。

        base_link 在 calib 变换图里是单位阵，所以 cam_from_base = inv(base_from_cam)，
        只由相机外参决定（不再受 pose 那一项影响）。
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_clip(root)
            cameras = load_clip_cameras(root)
        self.assertTrue(np.allclose(
            cameras["cam_front"]["T"][:3, 3], [0.0, -3.0, 0.0]))

    def test_base_link_contract_ignores_pose_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_clip(root, include_top=False)
            cameras = load_clip_cameras(root)
        self.assertTrue(np.allclose(
            cameras["cam_front"]["T"][:3, 3], [0.0, -3.0, 0.0]))

    def test_explicit_legacy_pose_detection_frame_still_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_clip(root)
            cameras = load_clip_cameras(root, detection_frame="pose")
        self.assertTrue(np.allclose(
            cameras["cam_front"]["T"][:3, 3], [10.0, -3.0, 0.0]))

    def test_points_are_loaded_in_base_link(self):
        """点云导出在 top 系；公共可见度阶段先按 base_from_top 变到 base_link。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_clip(root)
            lidar = root / "lidar" / "lidar_top"
            lidar.mkdir(parents=True)
            np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32).tofile(lidar / "7.bin")
            points = load_base_link_points(root, "7")
        self.assertTrue(np.allclose(points, [[3.0, 2.0, 3.0]]))

    def test_world_from_lidar_top_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_clip(root)
            coords = CoordinateProvider(root)
            transform = coords.world_from_lidar(0)
        self.assertTrue(np.allclose(transform[:3, 3], [-8.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
