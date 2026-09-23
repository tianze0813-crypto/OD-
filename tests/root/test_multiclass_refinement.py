import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geometry.multiclass_refinement import (
    TruckOverlapConfig,
    merge_overlapping_truck_tracks,
    unify_nonmotorized_track_sizes,
    verify_multiclass_refinement,
)
from tracking.tracker_conservative import CoordinateProvider


def make_clip(root: Path) -> Path:
    (root / "transforms").mkdir(parents=True)
    identity = np.eye(4).tolist()
    (root / "transforms" / "calib.json").write_text(json.dumps({
        "tf2base_link": {"pose": identity, "lidar_top": identity},
    }), encoding="utf-8")
    (root / "transforms" / "pose_data.txt").write_text(
        "\n".join(f"{index * 300000000},0,0,0,0,0,0,1"
                   for index in range(8)) + "\n",
        encoding="utf-8")
    return root


def detection(class_name, track_id, box, score=0.8):
    return {
        "class_name": class_name,
        "track_id": track_id,
        "score": score,
        "box_lidar": list(box),
    }


class MulticlassRefinementTest(unittest.TestCase):
    def test_truck_high_overlap_merges_id_and_removes_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = make_clip(Path(directory))
            coords = CoordinateProvider(clip)
            frames = [{
                "frame_id": "0",
                "detections": [
                    detection("Truck", 8, [0.0, 0.0, 1.0, 4.0, 2.0, 1.8, 0.0], 0.9),
                    detection("Truck", 9, [0.3, 0.1, 1.0, 4.1, 2.1, 1.8, 0.0], 0.7),
                ],
                "num_detections": 2,
            }]
            diagnostics = merge_overlapping_truck_tracks(frames, coords)

        self.assertEqual(diagnostics["accepted_pairs"], 1)
        self.assertEqual(diagnostics["boxes_removed"], 1)
        self.assertEqual(len(frames[0]["detections"]), 1)
        self.assertEqual(frames[0]["detections"][0]["track_id"], 8)

    def test_truck_separate_boxes_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = make_clip(Path(directory))
            coords = CoordinateProvider(clip)
            frames = [{
                "frame_id": "0",
                "detections": [
                    detection("Truck", 8, [0.0, 0.0, 1.0, 4.0, 2.0, 1.8, 0.0]),
                    detection("Truck", 9, [8.0, 0.0, 1.0, 4.0, 2.0, 1.8, 0.0]),
                ],
                "num_detections": 2,
            }]
            diagnostics = merge_overlapping_truck_tracks(frames, coords)

        self.assertEqual(diagnostics["accepted_pairs"], 0)
        self.assertEqual(diagnostics["boxes_removed"], 0)
        self.assertEqual([det["track_id"] for det in frames[0]["detections"]], [8, 9])

    def test_truck_car_high_overlap_is_promoted_and_merged(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = make_clip(Path(directory))
            coords = CoordinateProvider(clip)
            frames = [{
                "frame_id": "0",
                "detections": [
                    detection("Truck", 8, [0.0, 0.0, 1.0, 4.0, 2.0, 1.8, 0.0], 0.5),
                    detection("Car", 9, [0.2, 0.1, 1.0, 3.8, 2.0, 1.8, 0.0], 0.9),
                ],
                "num_detections": 2,
            }]
            before = copy.deepcopy(frames)
            diagnostics = merge_overlapping_truck_tracks(frames, coords)

        self.assertEqual(diagnostics["accepted_pairs"], 1)
        self.assertEqual(len(diagnostics["cross_class_pairs"]), 1)
        self.assertEqual(diagnostics["class_converted_boxes"], 1)
        self.assertEqual(len(frames[0]["detections"]), 1)
        self.assertEqual(frames[0]["detections"][0]["class_name"], "Truck")
        self.assertEqual(frames[0]["detections"][0]["track_id"], 8)
        self.assertTrue(verify_multiclass_refinement(before, frames)["passed"])

    def test_truck_car_far_boxes_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = make_clip(Path(directory))
            coords = CoordinateProvider(clip)
            frames = [{
                "frame_id": "0",
                "detections": [
                    detection("Truck", 8, [0.0, 0.0, 1.0, 4.0, 2.0, 1.8, 0.0]),
                    detection("Car", 9, [8.0, 0.0, 1.0, 4.0, 2.0, 1.8, 0.0]),
                ],
                "num_detections": 2,
            }]
            diagnostics = merge_overlapping_truck_tracks(frames, coords)

        self.assertEqual(diagnostics["accepted_pairs"], 0)
        self.assertEqual(diagnostics["class_converted_boxes"], 0)
        self.assertEqual([det["class_name"] for det in frames[0]["detections"]],
                         ["Truck", "Car"])

    def test_two_trucks_near_for_multiple_frames_merge_without_iou(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = make_clip(Path(directory))
            coords = CoordinateProvider(clip)
            frames = []
            for index, timestamp in enumerate((0, 300000000)):
                frames.append({
                    "frame_id": str(timestamp),
                    "detections": [
                        detection("Truck", 8, [0.0, 0.0, 1.0, 10.0, 2.0, 1.8, 0.0]),
                        detection("Truck", 9, [0.6, 0.0, 1.0, 10.0, 2.0, 1.8,
                                                1.5707963267948966]),
                    ],
                    "num_detections": 2,
                })
            diagnostics = merge_overlapping_truck_tracks(
                frames, coords, TruckOverlapConfig(high_iou=0.95,
                                                   moderate_iou=0.90))

        self.assertEqual(diagnostics["accepted_pairs"], 1)
        self.assertEqual(len(diagnostics["near_truck_pairs"]), 1)
        self.assertEqual([det["track_id"] for det in frames[0]["detections"]], [8])
        self.assertEqual([det["track_id"] for det in frames[1]["detections"]], [8])

    def test_nonmotorized_refines_large_center_size_and_yaw(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = make_clip(Path(directory))
            coords = CoordinateProvider(clip)
            positions = [(0.0, 0.0), (1.0, 0.0), (5.0, 3.0),
                         (3.0, 3.0), (4.0, 4.0)]
            sizes = [(2.0, 0.8, 1.5), (0.7, 0.5, 1.0),
                     (5.0, 2.0, 2.0), (2.0, 0.8, 1.5),
                     (2.0, 0.8, 1.5)]
            frames = []
            for index, ((x, y), size) in enumerate(zip(positions, sizes)):
                frames.append({
                    "frame_id": str(index * 300000000),
                    "detections": [detection(
                        "Nonmotorized_vehicle", 41,
                        [x, y, 1.0, *size, 0.0], 0.9)],
                    "num_detections": 1,
                })
            before_large_center = frames[2]["detections"][0]["box_lidar"][:2]
            diagnostics = unify_nonmotorized_track_sizes(frames, coords)

        self.assertEqual(diagnostics["tracks_refined"], 1)
        self.assertGreater(diagnostics["centers_changed"], 0)
        self.assertGreater(diagnostics["yaw_boxes_updated"], 0)
        self.assertNotEqual(frames[2]["detections"][0]["box_lidar"][:2],
                            before_large_center)
        physical_sizes = [
            sorted(frame["detections"][0]["box_lidar"][3:5]) +
            [frame["detections"][0]["box_lidar"][5]]
            for frame in frames
        ]
        for value in physical_sizes[1:]:
            np.testing.assert_allclose(value, physical_sizes[0], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
