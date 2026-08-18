import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from filtering.camera_visibility import load_clip_cameras
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

    def test_camera_projection_prefers_lidar_top(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_clip(root)
            cameras = load_clip_cameras(root)
        self.assertEqual(cameras["cam_front"]["source_frame"], "lidar_top")
        self.assertTrue(np.allclose(
            cameras["cam_front"]["T"][:3, 3], [2.0, -3.0, 0.0]))

    def test_legacy_pose_fallback_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_clip(root, include_top=False)
            cameras = load_clip_cameras(root)
        self.assertEqual(cameras["cam_front"]["source_frame"], "pose")

    def test_world_from_lidar_top_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_clip(root)
            coords = CoordinateProvider(root)
            transform = coords.world_from_lidar(0)
        self.assertTrue(np.allclose(transform[:3, 3], [-8.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
