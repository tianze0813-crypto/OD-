import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from filtering.low_confidence_class_filter import (
    LowConfidenceClassFilterConfig,
    apply_low_confidence_class_filter,
)
from filtering.final_filter import FinalFilterConfig, apply_final_filter
from tracking.tracker_conservative import CoordinateProvider, box_lidar_to_base_link


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


class Step5FinalFilterTest(unittest.TestCase):
    def _clip(self, root: Path) -> CoordinateProvider:
        transforms = root / "transforms"
        lidar = root / "lidar" / "lidar_top"
        transforms.mkdir(parents=True)
        lidar.mkdir(parents=True)
        top = np.eye(4)
        top[:2, 3] = [1.0, 2.0]
        (transforms / "calib.json").write_text(json.dumps({
            "tf2base_link": {
                "pose": np.eye(4).tolist(),
                "lidar_top": top.tolist(),
            }
        }), encoding="utf-8")
        (transforms / "pose_data.txt").write_text(
            "\n".join(f"{i},0,0,0,0,0,0,1" for i in range(4)) + "\n",
            encoding="utf-8")
        return CoordinateProvider(root)

    @staticmethod
    def _det(track_id: int, x: float) -> dict:
        return {
            "class_name": "Truck" if track_id == 2 else "Pedestrian",
            "score": 0.8,
            "track_id": track_id,
            "box_lidar": [x, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0],
        }

    def test_filters_sparse_and_short_tracks_then_converts_boxes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            coords = self._clip(root)
            frames = []
            for frame_index in range(4):
                frame_id = str(frame_index)
                # Eleven points are inside each box; the second track exists
                # for only three frames and must be removed by lifecycle.
                points = np.asarray(
                    [[0.1 * i, 0.0, 0.0, 1.0] for i in range(11)]
                    + [[10.0 + 0.1 * i, 0.0, 0.0, 1.0]
                       for i in range(11)], dtype=np.float32)
                points.tofile(root / "lidar" / "lidar_top" / f"{frame_id}.bin")
                detections = [self._det(1, 0.0)]
                if frame_index < 3:
                    detections.append(self._det(2, 10.0))
                frames.append({
                    "frame_id": frame_id,
                    "num_points": len(points),
                    "num_detections": len(detections),
                    "detections": detections,
                })

            output, stats = apply_final_filter(
                frames, root, coords,
                FinalFilterConfig(max_points_in_box=10,
                                  max_track_length=3))

        self.assertEqual(stats["point_filter_removed"], 0)
        self.assertEqual(stats["short_track_removed"], 3)
        self.assertEqual(stats["boxes_converted"], 4)
        self.assertEqual(
            [d["track_id"] for f in output for d in f["detections"]],
            [1, 1, 1, 1])
        for frame in output:
            box = frame["detections"][0]
            self.assertEqual(box["box_frame"], "base_link")
            self.assertTrue(np.allclose(box["box_lidar"][:3], [1.0, 2.0, 0.0]))

    def test_point_threshold_is_inclusive(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            coords = self._clip(root)
            points = np.asarray(
                [[0.1 * i, 0.0, 0.0, 1.0] for i in range(10)],
                dtype=np.float32)
            points.tofile(root / "lidar" / "lidar_top" / "0.bin")
            frames = [{
                "frame_id": "0", "num_points": len(points),
                "num_detections": 1, "detections": [self._det(1, 0.0)],
            }]
            output, stats = apply_final_filter(frames, root, coords)
        self.assertEqual(output[0]["detections"], [])
        self.assertEqual(stats["point_filter_removed"], 1)

    def test_box_conversion_rotates_heading_into_base_link(self):
        angle = np.pi / 2.0
        transform = np.eye(4)
        transform[:3, :3] = [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
        converted = box_lidar_to_base_link(
            [1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.0], transform)
        self.assertTrue(np.allclose(converted[:3], [-2.0, 1.0, 3.0]))
        self.assertAlmostEqual(converted[6], angle)


if __name__ == "__main__":
    unittest.main()
