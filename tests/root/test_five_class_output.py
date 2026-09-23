import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from filtering.five_class_output import apply_five_class_output
from tracking.tracker_conservative import CoordinateProvider


class FiveClassOutputTest(unittest.TestCase):
    def test_keeps_all_target_classes_and_only_converts_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transforms = root / "transforms"
            transforms.mkdir()
            transform = np.eye(4)
            transform[:3, 3] = [1.0, 2.0, 0.5]
            (transforms / "calib.json").write_text(json.dumps({
                "tf2base_link": {
                    "pose": np.eye(4).tolist(),
                    "lidar_top": transform.tolist(),
                }
            }), encoding="utf-8")
            (transforms / "pose_data.txt").write_text(
                "0,0,0,0,0,0,0,1\n", encoding="utf-8")
            coords = CoordinateProvider(root)
            frames = [{
                "frame_id": "0",
                "detections": [
                    {
                        "track_id": index + 1,
                        "class_name": name,
                        "score": 0.8,
                        "box_lidar": [0.0, 0.0, 0.0,
                                      4.0, 2.0, 1.5, 0.0],
                    }
                    for index, name in enumerate((
                        "Car", "Truck", "Bus", "Pedestrian",
                        "Nonmotorized_vehicle"))
                ],
            }]
            output, stats = apply_five_class_output(frames, coords)

        self.assertEqual(stats["boxes_converted"], 5)
        self.assertEqual(stats["removed_by_reason"], {})
        self.assertEqual(
            [d["class_name"] for d in output[0]["detections"]],
            ["Car", "Truck", "Bus", "Pedestrian", "Nonmotorized_vehicle"],
        )
        self.assertTrue(all(d["box_frame"] == "base_link"
                            for d in output[0]["detections"]))
        self.assertTrue(all(np.allclose(d["box_lidar"][:3], [1.0, 2.0, 0.5])
                            for d in output[0]["detections"]))

    def test_unknown_class_is_removed_but_no_point_or_lifecycle_filter_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transforms = root / "transforms"
            transforms.mkdir()
            (transforms / "calib.json").write_text(json.dumps({
                "tf2base_link": {
                    "pose": np.eye(4).tolist(),
                    "lidar_top": np.eye(4).tolist(),
                }
            }), encoding="utf-8")
            (transforms / "pose_data.txt").write_text(
                "0,0,0,0,0,0,0,1\n", encoding="utf-8")
            coords = CoordinateProvider(root)
            frames = [{"frame_id": "0", "detections": [
                {"track_id": 1, "class_name": "Cone",
                 "box_lidar": [0, 0, 0, 1, 1, 1, 0]},
                {"track_id": 2, "class_name": "Bus",
                 "box_lidar": [0, 0, 0, 1, 1, 1, 0]},
            ]}]
            output, stats = apply_five_class_output(frames, coords)
        self.assertEqual(len(output[0]["detections"]), 1)
        self.assertEqual(output[0]["detections"][0]["class_name"], "Bus")
        self.assertEqual(stats["removed_by_reason"], {"unknown_class": 1})


if __name__ == "__main__":
    unittest.main()
