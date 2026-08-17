import tempfile
import unittest
from pathlib import Path

import numpy as np

from geometry import box_geometry
from geometry.truck_box_fit import TruckBoxFitConfig, _estimate_track_size, _filter_truck_overlaps


def frame(frame_id, detections):
    return {"frame_id": frame_id, "num_points": 0,
            "num_detections": len(detections), "detections": detections}


def truck(track_id, x, y=0.0, dx=10.0, dy=2.8, yaw=0.0):
    return {
        "class_name": "Truck",
        "score": 0.8,
        "box_lidar": [float(x), float(y), 0.0, dx, dy, 3.5, yaw],
        "track_id": track_id,
    }


class TruckOverlapFilterTest(unittest.TestCase):
    def test_any_overlap_drops_less_stable_truck(self):
        frames = [
            frame("0", [truck(7, 0.0)]),
            frame("1", [truck(7, 0.0), truck(8, 9.9)]),
            frame("2", [truck(7, 0.0)]),
            frame("3", [truck(8, 9.9)]),
        ]
        stats = _filter_truck_overlaps(frames, TruckBoxFitConfig())
        self.assertIn(8, stats["dropped_track_ids"])
        self.assertEqual(stats["boxes_removed"], 2)
        self.assertGreater(stats["overlap_events"][0]["iou"], 0.0)

    def test_stable_short_track_beats_long_rotating_track(self):
        frames = [
            frame("0", [truck(7, 0.0, yaw=0.0)]),
            frame("1", [truck(7, 0.0, yaw=1.0471975511965976),
                        truck(8, 0.1, yaw=0.0)]),
            frame("2", [truck(7, 0.0, yaw=2.0943951023931953)]),
            frame("3", [truck(8, 0.1, yaw=0.0)]),
        ]
        stats = _filter_truck_overlaps(frames, TruckBoxFitConfig())
        self.assertEqual(stats["dropped_track_ids"], [7])
        self.assertGreater(
            stats["overlap_events"][0]["stability_b"],
            stats["overlap_events"][0]["stability_a"])

    def test_non_overlapping_trucks_are_kept(self):
        frames = [
            frame("0", [truck(7, 0.0), truck(8, 20.0)]),
            frame("1", [truck(7, 0.0), truck(8, 20.0)]),
        ]
        stats = _filter_truck_overlaps(frames, TruckBoxFitConfig())
        self.assertEqual(stats["trucks_removed"], 0)


class TruckSizeFitTest(unittest.TestCase):
    def test_consistent_point_evidence_shrinks_detector_size(self):
        config = TruckBoxFitConfig()
        frame_id = "1785719061020941019"
        points = np.column_stack([
            np.random.default_rng(3).uniform(-5.0, 5.0, 400),
            np.random.default_rng(4).uniform(-1.0, 1.0, 400),
            np.random.default_rng(5).uniform(0.0, 3.0, 400),
            np.zeros(400, dtype=np.float32),
        ]).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lidar_dir = root / "lidar" / "lidar_top"
            lidar_dir.mkdir(parents=True)
            points.tofile(lidar_dir / f"{frame_id}.bin")
            items = []
            for frame_index in range(6):
                items.append({
                    "frame_index": frame_index,
                    "frame_id": frame_id,
                    "det": {
                        "class_name": "Truck",
                        "score": 0.8,
                        "box_lidar": [0.0, 0.0, 1.5, 12.0, 3.2, 4.0, 0.0],
                    },
                    "evidence": None,
                })
            lidar = box_geometry._LidarCache(root)
            fixed, evidence_indices = _estimate_track_size(
                items, lidar, config)
        self.assertEqual(len(evidence_indices), 6)
        self.assertLess(fixed[0], 11.0)
        self.assertLess(fixed[1], 2.6)


if __name__ == "__main__":
    unittest.main()
