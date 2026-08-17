import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from filtering.low_confidence_class_filter import (
    LowConfidenceClassFilterConfig,
    apply_low_confidence_class_filter,
)
from tracking.tracker_conservative import CoordinateProvider


def make_coords(root: Path) -> CoordinateProvider:
    transforms = root / "transforms"
    transforms.mkdir()
    (transforms / "calib.json").write_text(json.dumps({
        "tf2base_link": {
            "pose": np.eye(4).tolist(),
            "lidar_top": np.eye(4).tolist(),
        }
    }), encoding="utf-8")
    (transforms / "pose_data.txt").write_text("\n".join(
        f"{index * 1000000000},0,0,0,0,0,0,1" for index in range(30)
    ) + "\n", encoding="utf-8")
    return CoordinateProvider(root)


def det(class_name, x, y=0.0, track_id=None):
    value = {
        "class_name": class_name,
        "score": 0.8,
        "box_lidar": [float(x), float(y), 0.0, 1.8, 0.8, 1.6, 0.0],
    }
    if track_id is not None:
        value["track_id"] = track_id
    return value


def frames(rows):
    return [{
        "frame_id": str(index * 1000000000),
        "num_points": 0,
        "num_detections": len(items),
        "detections": items,
    } for index, items in enumerate(rows)]


class Step5FilterTest(unittest.TestCase):
    def test_truck_is_removed_by_default(self):
        source = frames([
            [det("Truck", 0.0, track_id=10), det("Car", 5.0, track_id=11)],
            [det("Truck", 0.2, track_id=10), det("Car", 5.1, track_id=11)],
        ])
        with TemporaryDirectory() as directory:
            output, stats = apply_low_confidence_class_filter(
                source, make_coords(Path(directory)))
        self.assertEqual(stats["truck_boxes_removed"], 2)
        self.assertTrue(all(
            d["class_name"] != "Truck"
            for f in output for d in f["detections"]))

    def test_static_cyclist_is_removed_and_moving_cyclist_is_kept(self):
        source = frames([
            [det("Cyclist", 0.0, track_id=20), det("Cyclist", 0.0, track_id=21)],
            [det("Cyclist", 0.05, track_id=20), det("Cyclist", 0.0, track_id=21)],
            [det("Cyclist", -0.04, track_id=20), det("Cyclist", 0.1, track_id=21)],
            [det("Cyclist", 0.06, track_id=20), det("Cyclist", 1.2, track_id=21)],
            [det("Cyclist", -0.02, track_id=20), det("Cyclist", 2.4, track_id=21)],
            [det("Cyclist", 0.03, track_id=20), det("Cyclist", 3.6, track_id=21)],
        ])
        with TemporaryDirectory() as directory:
            output, stats = apply_low_confidence_class_filter(
                source, make_coords(Path(directory)))
        ids = {d["track_id"] for f in output for d in f["detections"]}
        self.assertNotIn(20, ids)
        self.assertIn(21, ids)
        self.assertEqual(stats["static_nonmotorized_tracks_removed"], 1)

    def test_keep_flags_disable_filters(self):
        source = frames([
            [det("Truck", 0.0, track_id=10),
             det("Cyclist", 0.0, track_id=20)],
            [det("Truck", 0.1, track_id=10),
             det("Cyclist", 0.05, track_id=20)],
            [det("Truck", 0.2, track_id=10),
             det("Cyclist", -0.03, track_id=20)],
        ])
        config = LowConfidenceClassFilterConfig(
            drop_truck=False, drop_static_nonmotorized=False)
        with TemporaryDirectory() as directory:
            output, stats = apply_low_confidence_class_filter(
                source, make_coords(Path(directory)), config)
        self.assertEqual(stats["before_detections"], stats["after_detections"])
        self.assertEqual(
            {d["class_name"] for f in output for d in f["detections"]},
            {"Truck", "Cyclist"})


if __name__ == "__main__":
    unittest.main()
