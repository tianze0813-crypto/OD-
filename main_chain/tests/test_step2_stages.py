import copy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from classification.class_refinement import (
    finalize_track_classes,
    preassociate_and_unify,
)
from filtering.hard_filters import deduplicate_same_center
from geometry.static_yaw import stabilize_static_yaw
from geometry.yaw_vehicle_dynamic import (
    YawVehicleDynamicConfig,
    _confirm_motion_onset,
    _raw_yaw_conflicts_with_pointcloud,
)
from geometry.yaw_integrated import PedestrianYawConfig, _pedestrian_targets
from tracking.tracker_conservative import CoordinateProvider
from tracking.tracker_static_first import StaticSlot


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
        f"{index * 400000000},0,0,0,0,0,0,1" for index in range(20)
    ) + "\n", encoding="utf-8")
    return CoordinateProvider(root)


def det(class_name, x, y=0.0, length=1.6, width=0.8, score=0.8,
        track_id=None):
    value = {
        "class_name": class_name,
        "score": score,
        "box_lidar": [float(x), float(y), 0.0, float(length), float(width),
                      1.6, 0.0],
    }
    if track_id is not None:
        value["track_id"] = track_id
    return value


def frames(rows):
    return [{
        "frame_id": str(index * 400000000),
        "num_points": 0,
        "num_detections": len(items),
        "detections": items,
    } for index, items in enumerate(rows)]


class ClassPreassociationTest(unittest.TestCase):
    def test_mixed_small_vehicle_chain_becomes_cyclist(self):
        rows = [[det("Pedestrian", 0.2 * index)] for index in range(3)]
        rows += [[det("Vehicle", 0.2 * index)] for index in range(3, 7)]
        source = frames(rows)
        geometry = copy.deepcopy([[d["box_lidar"] for d in f["detections"]]
                                  for f in source])
        with TemporaryDirectory() as directory:
            diagnostics = preassociate_and_unify(
                source, make_coords(Path(directory)))
        self.assertEqual({d["class_name"] for f in source
                          for d in f["detections"]}, {"Cyclist"})
        self.assertEqual(diagnostics["mixed_components"], 1)
        self.assertEqual(geometry, [[d["box_lidar"] for d in f["detections"]]
                                    for f in source])

    def test_pure_chain_keeps_original_class(self):
        source = frames([[det("Vehicle", 0.1 * index, length=1.8)]
                         for index in range(6)])
        with TemporaryDirectory() as directory:
            diagnostics = preassociate_and_unify(
                source, make_coords(Path(directory)))
        self.assertEqual({d["class_name"] for f in source
                          for d in f["detections"]}, {"Vehicle"})
        self.assertEqual(diagnostics["detections_changed"], 0)

    def test_ambiguous_cross_class_transition_is_not_unified(self):
        source = frames([
            [det("Pedestrian", 0.0)],
            [det("Pedestrian", 0.2)],
            [det("Vehicle", 0.4), det("Vehicle", 0.5)],
            [det("Vehicle", 0.6), det("Vehicle", 0.7)],
        ])
        with TemporaryDirectory() as directory:
            preassociate_and_unify(source, make_coords(Path(directory)))
        self.assertEqual(source[0]["detections"][0]["class_name"], "Pedestrian")

    def test_post_tracking_unifies_only_mixed_tracks(self):
        source = frames([
            [det("Vehicle", 0.0, track_id=7), det("Pedestrian", 10.0, track_id=8)],
            [det("Pedestrian", 0.2, track_id=7), det("Pedestrian", 10.2, track_id=8)],
        ])
        diagnostics = finalize_track_classes(source)
        self.assertEqual({d["class_name"] for f in source for d in f["detections"]
                          if d["track_id"] == 7}, {"Cyclist"})
        self.assertEqual({d["class_name"] for f in source for d in f["detections"]
                          if d["track_id"] == 8}, {"Pedestrian"})
        self.assertEqual(diagnostics["mixed_tracks_unified"], 1)

    def test_post_tracking_size_classifies_pure_vehicle_family_tracks(self):
        source = frames([
            [det("Vehicle", 0.0, length=6.4, width=2.4, track_id=10),
             det("Car", 10.0, length=2.2, width=1.0, track_id=11),
             det("Truck", 20.0, length=4.4, width=1.9, track_id=12),
             det("Pedestrian", 30.0, length=0.8, width=0.7, track_id=13)],
            [det("Vehicle", 0.1, length=6.6, width=2.5, track_id=10),
             det("Car", 10.1, length=2.4, width=1.1, track_id=11),
             det("Truck", 20.1, length=4.6, width=2.0, track_id=12),
             det("Pedestrian", 30.1, length=0.9, width=0.8, track_id=13)],
        ])
        diagnostics = finalize_track_classes(source)
        classes = {
            track_id: {d["class_name"] for frame in source
                       for d in frame["detections"]
                       if d["track_id"] == track_id}
            for track_id in range(10, 14)
        }
        self.assertEqual(classes[10], {"Truck"})
        self.assertEqual(classes[11], {"Car"})
        self.assertEqual(classes[12], {"Truck"})
        self.assertEqual(classes[13], {"Pedestrian"})
        self.assertEqual(diagnostics["vehicle_family_tracks_assessed"], 1)

    def test_step2_size_thresholds_resolve_generic_vehicle_tracks(self):
        source = frames([
            [det("Vehicle", 0.0, length=5.3, width=2.3, track_id=20),
             det("Vehicle", 10.0, length=3.3, width=1.45, track_id=21)],
            [det("Vehicle", 0.1, length=5.4, width=2.4, track_id=20),
             det("Vehicle", 10.1, length=3.4, width=1.46, track_id=21)],
        ])
        finalize_track_classes(source)
        classes = {
            tid: {d["class_name"] for frame in source
                  for d in frame["detections"] if d["track_id"] == tid}
            for tid in (20, 21)
        }
        self.assertEqual(classes[20], {"Car"})
        self.assertEqual(classes[21], {"Cyclist"})


class SameCenterDedupTest(unittest.TestCase):
    def test_prefers_longer_lifecycle_over_single_frame_score(self):
        source = frames([
            [det("Vehicle", 0.0, score=0.6, track_id=10)],
            [det("Vehicle", 0.0, score=0.6, track_id=10)],
            [det("Vehicle", 0.0, score=0.6, track_id=10),
             det("Vehicle", 0.1, score=0.99, track_id=11)],
            [det("Vehicle", 0.0, score=0.6, track_id=10)],
        ])
        diagnostics = deduplicate_same_center(source)
        self.assertEqual(diagnostics["boxes_removed"], 1)
        self.assertEqual([d["track_id"] for d in source[2]["detections"]], [10])


class StaticYawTest(unittest.TestCase):
    def test_changes_only_static_parking_yaw(self):
        source = frames([
            [det("Car", 0.0, track_id=4), det("Car", 10.0, track_id=20)],
            [det("Car", 0.1, track_id=4), det("Car", 10.2, track_id=20)],
            [det("Car", -0.1, track_id=4), det("Car", 10.4, track_id=20)],
        ])
        source[0]["detections"][0]["box_lidar"][6] = 0.10
        source[1]["detections"][0]["box_lidar"][6] = -0.08
        source[2]["detections"][0]["box_lidar"][6] = 0.05
        before = copy.deepcopy(source)
        slot = StaticSlot(
            slot_id=4, track_id=4, center=np.array([0.0, 0.0]),
            yaw=0.0, size=np.array([4.5, 2.0, 1.6]),
            class_name="Car", evidence_hits=10)
        with TemporaryDirectory() as directory:
            diagnostics = stabilize_static_yaw(
                source, make_coords(Path(directory)), [slot], {})
        parked_yaws = [frame["detections"][0]["box_lidar"][6]
                       for frame in source]
        self.assertTrue(np.allclose(parked_yaws, parked_yaws[0]))
        self.assertEqual(
            [frame["detections"][1] for frame in source],
            [frame["detections"][1] for frame in before])
        for left_frame, right_frame in zip(before, source):
            left = copy.deepcopy(left_frame["detections"][0])
            right = copy.deepcopy(right_frame["detections"][0])
            left["box_lidar"][6] = right["box_lidar"][6]
            self.assertEqual(left, right)
        self.assertEqual(diagnostics["parking_boxes_stabilized"], 3)

    def test_departure_segment_is_not_locked(self):
        source = frames([[det("Car", float(index), track_id=7)]
                         for index in range(6)])
        raw_yaws = [0.08, -0.05, 0.04, 0.65, 0.75, 0.85]
        for frame, yaw in zip(source, raw_yaws):
            frame["detections"][0]["box_lidar"][6] = yaw
        slot = StaticSlot(
            slot_id=7, track_id=7, center=np.array([0.0, 0.0]),
            yaw=0.0, size=np.array([4.5, 2.0, 1.6]),
            class_name="Car", evidence_hits=10)
        coordination = {"departures": [{
            "slot_id": 7, "inherited_track_id": 7,
            "start_timestamp": int(source[3]["frame_id"]),
        }]}
        with TemporaryDirectory() as directory:
            diagnostics = stabilize_static_yaw(
                source, make_coords(Path(directory)), [slot], coordination)
        output_yaws = [frame["detections"][0]["box_lidar"][6]
                       for frame in source]
        self.assertTrue(np.allclose(output_yaws[:3], output_yaws[0]))
        self.assertEqual(output_yaws[3:], raw_yaws[3:])
        self.assertEqual(diagnostics["parking_boxes_stabilized"], 3)


class YawVehicleDynamicTest(unittest.TestCase):
    @staticmethod
    def motion_items(x_values):
        return [{
            "frame_index": index,
            "timestamp": index * 200000000,
            "world": np.array([float(x), 0.0, 0.0]),
        } for index, x in enumerate(x_values)]

    def test_center_jitter_does_not_confirm_motion(self):
        items = self.motion_items([
            0.00, 0.08, -0.05, 0.06, -0.03, 0.04, 0.01, -0.02,
        ])
        self.assertIsNone(_confirm_motion_onset(items, YawVehicleDynamicConfig()))

    def test_sustained_motion_is_confirmed_after_stationary_prefix(self):
        items = self.motion_items([
            0.00, 0.05, -0.04, 0.03, 0.00,
            0.20, 0.55, 1.00, 1.55, 2.20, 2.90,
        ])
        result = _confirm_motion_onset(items, YawVehicleDynamicConfig())
        self.assertIsNotNone(result)
        onset, _end, details = result
        self.assertGreaterEqual(onset, 3)
        self.assertGreaterEqual(details["concentration"], 0.75)

    def test_stable_raw_yaw_rejects_contradictory_pointcloud_axis(self):
        evidence = [
            (math.radians(value), 10.0, 10)
            for value in (178.0, 179.0, -179.0, 177.0, 180.0)
        ]
        rejected, details = _raw_yaw_conflicts_with_pointcloud(
            evidence, math.radians(65.0), YawVehicleDynamicConfig())
        self.assertTrue(rejected)
        self.assertGreaterEqual(details["raw_directed_yaw_stability"], 0.98)

    def test_unstable_raw_yaw_allows_pointcloud_correction(self):
        evidence = [
            (math.radians(value), 10.0, 10)
            for value in (-143.0, -92.0, -159.0, -69.0, -39.0, -115.0)
        ]
        rejected, _details = _raw_yaw_conflicts_with_pointcloud(
            evidence, math.radians(166.0), YawVehicleDynamicConfig())
        self.assertFalse(rejected)


class PedestrianYawTest(unittest.TestCase):
    def test_two_frame_heading_uses_world_positions(self):
        source = frames([
            [det("Pedestrian", 0.0, track_id=30)],
            [det("Pedestrian", 0.3, track_id=30)],
            [det("Pedestrian", 0.6, track_id=30)],
        ])
        with TemporaryDirectory() as directory:
            targets, details = _pedestrian_targets(
                source, make_coords(Path(directory)), PedestrianYawConfig())
        self.assertEqual(len(targets), 3)
        self.assertTrue(all(abs(value) < 1e-9 for value in targets.values()))
        self.assertEqual(details[0]["boxes_updated"], 3)

    def test_stationary_jitter_keeps_original_yaw(self):
        source = frames([
            [det("Pedestrian", 0.00, track_id=31)],
            [det("Pedestrian", 0.04, track_id=31)],
            [det("Pedestrian", -0.02, track_id=31)],
        ])
        with TemporaryDirectory() as directory:
            targets, details = _pedestrian_targets(
                source, make_coords(Path(directory)), PedestrianYawConfig())
        self.assertEqual(targets, {})
        self.assertEqual(details[0]["rejected_displacement"], 3)


if __name__ == "__main__":
    unittest.main()
