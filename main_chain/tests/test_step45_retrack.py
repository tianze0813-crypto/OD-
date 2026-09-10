import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import numpy as np

from region.dynamic_region import DynamicRegionConfig, _extend_track
from region.region_mask import DynamicRegionMask
from region.retrack import (
    Step45Config,
    _combined_merge_lateral_ok,
    _lateral_jump_ok,
    _movement_compatible,
    align_dynamic_yaw,
    build_region,
    candidate_track_ids,
    collect_world_tracks,
    direction_filter,
    inherit_ids,
    is_moving_seed,
    mark_long_gap_isolated_frames,
    is_pure_static,
    is_weak_moving_seed,
    phase_stitch,
    queue_stitch,
    region_mask,
    retrack_dynamic,
    revert_dynamic_yaw,
    select_retrackable,
    track_motion_stats,
    verify_static_freeze,
    verify_unique_frame_ids,
)
from tracking.tracker_conservative import ConservativeTracker, CoordinateProvider


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
        f"{index * 400000000},0,0,0,0,0,0,1" for index in range(80)
    ) + "\n", encoding="utf-8")
    return CoordinateProvider(root)


def det(class_name, x, y, track_id, *, length=4.5, width=2.0):
    return {
        "class_name": class_name,
        "score": 0.9,
        "track_id": track_id,
        "box_lidar": [float(x), float(y), 0.0, float(length), float(width),
                      1.6, 0.0],
    }


def frames(rows):
    return [{
        "frame_id": str(index * 400000000),
        "num_points": 0,
        "num_detections": len(items),
        "detections": items,
    } for index, items in enumerate(rows)]


class Step45RetrackTest(unittest.TestCase):
    def test_static_freeze_and_dynamic_majority_id(self):
        source = frames([
            [det("Car", index * 4.0, 0.0, 1),
             det("Car", 100.0, 100.0, 2)]
            for index in range(6)
        ])
        with TemporaryDirectory() as directory:
            coords = make_coords(Path(directory))
            tracks, _ = collect_world_tracks(source, coords)
            config = Step45Config()
            candidates = candidate_track_ids(tracks, config.region)
            self.assertEqual(set(candidates), {1})
            region = build_region(tracks, [], config.region)
            mask = region_mask(region, config.region)
            retrackable, _selection = select_retrackable(
                source, coords, mask, candidates)
            self.assertEqual(len(retrackable), 6)
            retrack_dynamic(source, coords, retrackable, config)
            inherit_ids(source, tracks, retrackable, candidates, {}, config)
            freeze = verify_static_freeze(source, source, retrackable)
        self.assertTrue(freeze["passed"])
        dynamic_ids = {
            det["track_id"] for frame in source for det in frame["detections"]
            if det.get("_step45_retracked")
        }
        self.assertEqual(dynamic_ids, {1})
        self.assertEqual(source[0]["detections"][1]["track_id"], 2)

    def test_static_freeze_detects_modified_frozen_detection(self):
        source = frames([[det("Car", 0.0, 0.0, 1)] for _ in range(3)])
        before = json.loads(json.dumps(source))
        source[1]["detections"][0]["box_lidar"][0] = 5.0
        result = verify_static_freeze(before, source, set())
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["mismatches"]), 1)

    def test_slot_release_anchor_without_explicit_departure_event(self):
        source = frames(
            [[det("Car", 0.0, 0.0, 5)] for _ in range(5)]
            + [[det("Car", index * 3.0, 0.0, 6)] for index in range(6)]
        )
        step2_diagnostics = {
            "tracking": {
                "slot_details": [{
                    "track_id": 5, "slot_id": 5,
                    "center": [0.0, 0.0], "yaw": 0.0,
                    "size": [4.5, 2.0, 1.6],
                }],
                "slot_motion_coordination": {},
            }
        }
        with TemporaryDirectory() as directory:
            coords = make_coords(Path(directory))
            tracks, _ = collect_world_tracks(source, coords)
            config = Step45Config()
            candidates = candidate_track_ids(tracks, config.region)
            self.assertEqual(set(candidates), {6})
            region = build_region(
                tracks, step2_diagnostics["tracking"]["slot_details"],
                config.region)
            mask = region_mask(region, config.region)
            # This test targets the slot-release boundary anchor itself, so
            # feed only the dynamic fragment to pass 2 (region-only selection
            # is covered by test_select_retrackable_uses_region_only).
            retrackable = {
                (frame_index, detection_index)
                for frame_index, frame in enumerate(source)
                for detection_index, det in enumerate(
                    frame.get("detections", []))
                if det.get("track_id") == 6
            }
            retrack_dynamic(source, coords, retrackable, config)
            result = inherit_ids(
                source, tracks, retrackable, candidates,
                step2_diagnostics, config)
        dynamic_ids = {
            det["track_id"] for frame in source for det in frame["detections"]
            if det.get("_step45_retracked")
        }
        self.assertEqual(dynamic_ids, {5})
        self.assertTrue(any(
            item["reason"].startswith("static_departure")
            for item in result["assignments"]))

    def test_select_retrackable_uses_region_only(self):
        source = frames(
            [[det("Car", 0.0, 0.0, 5)] for _ in range(5)]
            + [[det("Car", index * 3.0, 0.0, 6)] for index in range(6)]
        )
        with TemporaryDirectory() as directory:
            coords = make_coords(Path(directory))
            mask = DynamicRegionMask.from_polygons(
                [[(-20.0, -20.0), (40.0, -20.0),
                  (40.0, 20.0), (-20.0, 20.0)]], resolution=1.0)
            keys, diagnostics = select_retrackable(source, coords, mask, {6})
        self.assertEqual(diagnostics["retrackable_detections"], 11)
        self.assertEqual(len(keys), 11)

    def test_phase_stitch_merges_dynamic_start_into_waiting_frozen_id(self):
        source = frames(
            [[det("Car", 0.0, 0.0, 1)] for _ in range(6)]
            + [[det("Car", (index - 10) * 3.0, 0.0, 2)]
               for index in range(10, 16)]
        )
        with TemporaryDirectory() as directory:
            coords = make_coords(Path(directory))
            tracks, _ = collect_world_tracks(source, coords)
            for item in tracks[2]:
                item["det"]["_step45_retracked"] = True
            config = Step45Config()
            fake_result = SimpleNamespace(track_traffic_states=[
                {
                    "track_id": 1, "movement": "straight",
                    "direction_id": 0,
                    "intervals": [{
                        "state": "waiting_red",
                        "start_timestamp": 0.0,
                        "end_timestamp": 2.4,
                    }],
                },
                {
                    "track_id": 2, "movement": "straight",
                    "direction_id": 0,
                    "intervals": [{
                        "state": "moving",
                        "start_timestamp": 4.0,
                        "end_timestamp": 6.0,
                    }],
                },
            ])
            with mock.patch(
                    "region.retrack.build_traffic_light_model",
                    return_value=fake_result):
                diagnostics = phase_stitch(
                    source, tracks, coords, config)
        self.assertEqual(len(diagnostics["applied"]), 1)
        dynamic_ids = {
            det["track_id"] for frame in source for det in frame["detections"]
            if det.get("_step45_retracked")
        }
        self.assertEqual(dynamic_ids, {1})
        frozen_ids = {
            det["track_id"] for frame in source for det in frame["detections"]
            if not det.get("_step45_retracked")
        }
        self.assertEqual(frozen_ids, {1})

    def test_moving_seed_and_pure_static_rules(self):
        config = Step45Config()
        moving = [
            {"timestamp": index * 0.6, "world": np.array([index * 4.0, 0.0])}
            for index in range(6)
        ]
        parked = [
            {"timestamp": index * 0.6,
             "world": np.array([0.1 * (index % 2), 0.0])}
            for index in range(6)
        ]
        moving_stats = track_motion_stats(moving)
        parked_stats = track_motion_stats(parked)
        self.assertTrue(is_moving_seed(moving_stats, config))
        self.assertFalse(is_moving_seed(parked_stats, config))
        self.assertTrue(is_pure_static(parked_stats, config))
        self.assertFalse(is_pure_static(moving_stats, config))

    def test_weak_moving_seed_accepts_short_start(self):
        config = Step45Config()
        start = [
            {"timestamp": index * 0.3,
             "world": np.array([index * 0.8, 0.0])}
            for index in range(5)
        ]
        stats = track_motion_stats(start)
        self.assertTrue(is_weak_moving_seed(stats, config))

    def test_queue_stitch_merges_sequential_fragments(self):
        source = frames(
            [[det("Car", 0.0, 0.0, 1)]]
            + [[det("Car", index * 2.0, 0.0, 1)] for index in range(1, 6)]
            + [[det("Car", 10.0, 0.0, 2)] for _ in range(4)]
        )
        for frame in source:
            for detection in frame["detections"]:
                detection["region"] = "dynamic"
        with TemporaryDirectory() as directory:
            coords = make_coords(Path(directory))
            tracks, _ = collect_world_tracks(source, coords)
            config = Step45Config()
            mask = DynamicRegionMask.from_polygons(
                [[(-20.0, -20.0), (40.0, -20.0),
                  (40.0, 20.0), (-20.0, 20.0)]], resolution=1.0)
            direction = {
                "direction_id": 0,
                "origin": [0.0, 0.0],
                "forward": [1.0, 0.0],
                "right": [0.0, 1.0],
            }
            assignments = {
                1: {"direction_id": 0, "movement": "straight"},
                2: {"direction_id": 0, "movement": None},
            }
            with mock.patch(
                    "region.retrack._direction_assignments",
                    return_value=([direction], assignments)):
                result = queue_stitch(
                    source, tracks, {}, mask, set(), config)
        self.assertTrue(result["merges"])
        ids = {det["track_id"] for frame in source
               for det in frame["detections"]}
        self.assertEqual(ids, {1})

    def test_queue_stitch_does_not_merge_time_overlap(self):
        source = frames([
            [det("Car", 0.0, 0.0, 1), det("Car", 0.0, 8.0, 2)]
            for _ in range(5)
        ])
        with TemporaryDirectory() as directory:
            coords = make_coords(Path(directory))
            tracks, _ = collect_world_tracks(source, coords)
            config = Step45Config()
            mask = DynamicRegionMask.from_polygons(
                [[(-20.0, -20.0), (20.0, -20.0),
                  (20.0, 20.0), (-20.0, 20.0)]], resolution=1.0)
            direction = {
                "direction_id": 0, "origin": [0.0, 0.0],
                "forward": [1.0, 0.0], "right": [0.0, 1.0],
            }
            assignments = {
                1: {"direction_id": 0, "movement": None},
                2: {"direction_id": 0, "movement": None},
            }
            with mock.patch(
                    "region.retrack._direction_assignments",
                    return_value=([direction], assignments)):
                result = queue_stitch(
                    source, tracks, {}, mask, {1}, config)
        self.assertFalse(result["merges"])

    def test_long_gap_isolated_tail_is_excluded_from_merge_and_target(self):
        rows = [[det("Car", index * 2.0, 0.0, 1)]
                for index in range(6)]
        rows.extend([[] for _ in range(31)])
        rows.append([det("Car", 10.0, 0.0, 1)])
        rows.extend([[det("Car", 10.0 + index * 2.0, 0.0, 2)]
                     for index in range(1, 7)])
        source = frames(rows)
        for frame in source:
            for detection in frame["detections"]:
                detection["region"] = "dynamic"
                detection["_step45_retracked"] = True
        with TemporaryDirectory() as directory:
            coords = make_coords(Path(directory))
            tracks, _ = collect_world_tracks(source, coords)
            config = Step45Config()
            isolated = mark_long_gap_isolated_frames(
                source, tracks, config)
            mask = DynamicRegionMask.from_polygons(
                [[(-20.0, -20.0), (80.0, -20.0),
                  (80.0, 40.0), (-20.0, 40.0)]], resolution=1.0)
            direction = {
                "direction_id": 0,
                "origin": [0.0, 0.0],
                "forward": [1.0, 0.0],
                "right": [0.0, 1.0],
            }
            assignments = {
                1: {"direction_id": 0, "movement": None},
                2: {"direction_id": 0, "movement": None},
            }
            with mock.patch(
                    "region.retrack._direction_assignments",
                    return_value=([direction], assignments)):
                result = queue_stitch(
                    source, tracks, {}, mask, set(), config)
        self.assertEqual(isolated["marked_detections"], 1)
        isolated_frame = next(
            frame for frame in source
            if any(det.get("_step45_isolated_after_gap")
                   for det in frame["detections"]))
        self.assertEqual(isolated_frame["detections"][0]["track_id"], 1)
        self.assertEqual(result["isolated_ids"], [1])
        self.assertTrue(result["merges"])
        self.assertEqual(result["merges"][0]["final_id"], 2)
        for frame in source:
            for detection in frame["detections"]:
                if detection.get("_step45_isolated_after_gap"):
                    self.assertEqual(detection["track_id"], 1)
                else:
                    self.assertEqual(detection["track_id"], 2)

    def test_position_jump_gate_rejects_stationary_flash(self):
        def make_frame(timestamp, x):
            return {
                "frame_id": str(timestamp),
                "num_points": 0,
                "num_detections": 1,
                "detections": [det("Car", x, 0.0, None)],
            }
        times = [index * 400000000 for index in range(11)]
        times += [index * 400000000 for index in range(16, 21)]
        positions = [0.0] * 11 + [5.7, 7.7, 9.7, 11.7, 13.7]
        frames_input = [make_frame(t, p)
                        for t, p in zip(times, positions)]
        with TemporaryDirectory() as directory:
            coords = make_coords(Path(directory))
            tracker = ConservativeTracker(
                coords, min_static_hits=10 ** 9,
                dynamic_max_gap=1.8, use_yaw=False,
                occlusion_enabled=True, occlusion_max_gap=2.6,
                physical_position_jump_enabled=True)
            output, _diag = tracker.process(frames_input)
        ids = {det["track_id"] for frame in output
               for det in frame["detections"]}
        self.assertGreater(len(ids), 1)

    def test_retrack_lateral_jump_gate_rejects_sideways_step(self):
        def make_frame(timestamp, x, y):
            return {
                "frame_id": str(timestamp),
                "num_points": 0,
                "num_detections": 1,
                "detections": [det("Car", x, y, None)],
            }
        positions = [(0.0, 0.0), (4.0, 0.0), (8.0, 0.0), (11.8, 2.6)]
        timestamps = [index * 400000000 for index in range(len(positions))]
        with TemporaryDirectory() as directory:
            coords = make_coords(Path(directory))
            disabled = ConservativeTracker(
                coords, min_static_hits=10 ** 9,
                dynamic_max_gap=1.8, use_yaw=False,
                physical_position_jump_enabled=False,
                lateral_jump_gate_enabled=False)
            output, _diagnostics = disabled.process(
                [make_frame(t, x, y)
                 for t, (x, y) in zip(timestamps, positions)],
                enable_stitching=False)
        disabled_ids = {
            detection["track_id"]
            for frame in output for detection in frame["detections"]
        }
        self.assertEqual(disabled_ids, {1})

        with TemporaryDirectory() as directory:
            coords = make_coords(Path(directory))
            enabled = ConservativeTracker(
                coords, min_static_hits=10 ** 9,
                dynamic_max_gap=1.8, use_yaw=False,
                physical_position_jump_enabled=False,
                lateral_jump_gate_enabled=True,
                lateral_jump_max_m=2.5)
            output, enabled_diagnostics = enabled.process(
                [make_frame(t, x, y)
                 for t, (x, y) in zip(timestamps, positions)],
                enable_stitching=False)
        enabled_ids = {
            detection["track_id"]
            for frame in output for detection in frame["detections"]
        }
        self.assertEqual(len(enabled_ids), 2)
        self.assertEqual(
            enabled_diagnostics["lateral_jump_triggered"], 1)

    def test_inherit_ids_rejects_lateral_jump_bridge(self):
        source = frames([
            [det("Car", x, y, 1)]
            for x, y in [(0.0, 0.0), (4.0, 0.0), (8.0, 0.0),
                         (11.8, 2.6), (15.8, 2.6), (19.8, 2.6)]
        ])
        with TemporaryDirectory() as directory:
            coords = make_coords(Path(directory))
            tracks, _ = collect_world_tracks(source, coords)
            retrackable = {
                (frame_index, 0) for frame_index in range(len(source))}
            config = Step45Config()
            retrack_dynamic(source, coords, retrackable, config)
            result = inherit_ids(
                source, tracks, retrackable, {}, {}, config)
        final_ids = [
            detection["track_id"]
            for frame in source for detection in frame["detections"]
        ]
        self.assertEqual(len(set(final_ids)), 2)
        self.assertEqual(final_ids, [1, 1, 1, 2, 2, 2])
        self.assertTrue(any(
            assignment["reason"] == "new_id"
            for assignment in result["assignments"]))

    def test_merge_lateral_gate_uses_incoming_axis_and_order(self):
        config = Step45Config()

        def point(timestamp, x, y):
            return {"timestamp": timestamp,
                    "world": np.asarray([x, y], dtype=np.float64)}

        incoming = [point(0.0, 0.0, 0.0), point(0.4, 4.0, 0.0)]
        lateral = [point(1.0, 8.0, 3.0)]
        # (4, 0) -> (8, 3) has 3m lateral offset from the incoming axis.
        self.assertFalse(_lateral_jump_ok(incoming, lateral, config))
        # Argument order must not hide the jump when fragments are passed
        # newest-first.
        self.assertFalse(_lateral_jump_ok(lateral, incoming, config))
        # Interleaved insertion is validated on the combined time order.
        existing = [point(0.0, 0.0, 0.0), point(0.4, 4.0, 0.0),
                    point(1.0, 8.0, 0.0)]
        inserted = [point(0.2, 3.5, 3.0)]
        self.assertFalse(_combined_merge_lateral_ok(
            existing, inserted, config))

    def test_final_trajectory_yaw_uses_nearest_pi_equivalent(self):
        def make_detection(x, yaw):
            detection = det("Car", x, 0.0, 1)
            detection["box_lidar"][6] = float(yaw)
            detection["region"] = "dynamic"
            detection["_step45_retracked"] = True
            return detection

        reversed_yaw = math.radians(120.0)
        frames_input = [{
            "frame_id": str(index * 400000000),
            "num_points": 0,
            "num_detections": 1,
            "detections": [
                make_detection(index * 4.0,
                               reversed_yaw if index < 2 else 0.0)
            ],
        } for index in range(5)]
        with TemporaryDirectory() as directory:
            coords = make_coords(Path(directory))
            tracks, _ = collect_world_tracks(frames_input, coords)
            result = revert_dynamic_yaw(
                frames_input, tracks, Step45Config())
        # A whole-track median vote would leave 120 deg untouched
        # (median deviation = 0).  The nearest-pi-equivalent rule must
        # rewrite only those two to -60 deg.
        self.assertEqual(result["reversed_detections"], 2)
        self.assertEqual(result["reversed_tracks"], 1)
        self.assertEqual(result["method"],
                         "final_trajectory_per_observation")
        for frame in frames_input[:2]:
            yaw = frame["detections"][0]["box_lidar"][6]
            self.assertAlmostEqual(yaw, -math.pi / 3.0, places=6)
            self.assertTrue(
                frame["detections"][0].get("_step45_yaw_reversed"))
        for frame in frames_input[2:]:
            yaw = frame["detections"][0]["box_lidar"][6]
            self.assertAlmostEqual(yaw, 0.0, places=6)

    def test_turn_extension_uses_swept_area_forward(self):
        config = DynamicRegionConfig()
        # A left-turning track: first half straight east, then turns north.
        points = []
        heading = 0.0
        current = np.array([0.0, 0.0])
        for index in range(55):
            if index == 30:
                heading = math.radians(60.0)
            current = current + 0.5 * np.array(
                [math.cos(heading), math.sin(heading)])
            points.append({
                "timestamp": index * 0.4,
                "world": current.copy(),
                "yaw": heading,
                "size": np.array([4.5, 2.0, 1.6]),
                "class_name": "Car",
            })
        straight = [
            {
                "timestamp": index * 0.4,
                "world": np.array([index * 1.0, 0.0]),
                "yaw": 0.0,
                "size": np.array([4.5, 2.0, 1.6]),
                "class_name": "Car",
            }
            for index in range(20)
        ]
        straight_extended, straight_details = _extend_track(straight, config)
        turn_extended, turn_details = _extend_track(points, config)
        self.assertTrue(any(
            item.get("synthetic_extension") for item in straight_extended))
        self.assertTrue(any(
            detail.get("kind") == "straight_extension"
            for detail in straight_details))
        turn_forward = [
            detail for detail in turn_details
            if detail.get("end") == "end"
            and detail.get("kind") == "straight_extension"
        ]
        self.assertFalse(turn_forward)
        self.assertTrue(any(
            detail.get("kind") == "left_turn_tail"
            for detail in turn_details))

    def test_direction_filter_removes_single_noise_frame(self):
        frames_input = []
        for index in range(6):
            moving = det("Car", index * 4.0, 0.0, 1)
            parked = det("Car", 100.0, 100.0, 2)
            moving["box_lidar"][6] = 0.0
            parked["box_lidar"][6] = math.pi / 2
            frames_input.append({
                "frame_id": str(index * 400000000), "num_points": 0,
                "num_detections": 2, "detections": [moving, parked]})
        # Noisy yaw on the moving track; static track is not a dynamic id.
        frames_input[2]["detections"][0]["box_lidar"][6] = math.pi / 2
        with TemporaryDirectory() as directory:
            coords = make_coords(Path(directory))
            tracks, _ = collect_world_tracks(frames_input, coords)
            cleaned, details, diag = direction_filter(
                frames_input, tracks, {1}, Step45Config())
        self.assertEqual(diag["noise_detections_removed"], 1)
        self.assertEqual(len(cleaned[2]["detections"]), 1)
        self.assertEqual(cleaned[2]["detections"][0]["track_id"], 2)
        self.assertEqual(len(cleaned[3]["detections"]), 2)

    def test_occlusion_recovery_keeps_same_id(self):
        def make_frame(timestamp, x):
            return {
                "frame_id": str(timestamp),
                "num_points": 0,
                "num_detections": 1,
                "detections": [det("Car", x, 0.0, None)]
                if x is not None else [],
            }
        times = [0, 400000000, 800000000, 1200000000,
                 3200000000, 3600000000, 4000000000]
        positions = [0.0, 2.0, 4.0, 6.0, 16.0, 18.0, 20.0]
        frames_input = [make_frame(t, p)
                        for t, p in zip(times, positions)]
        with TemporaryDirectory() as directory:
            coords = make_coords(Path(directory))
            tracker = ConservativeTracker(
                coords, min_static_hits=10 ** 9,
                dynamic_max_gap=1.8, use_yaw=False,
                occlusion_enabled=True, occlusion_max_gap=2.0)
            output, diag = tracker.process(frames_input)
        ids = {det["track_id"] for frame in output
               for det in frame["detections"]}
        self.assertEqual(len(ids), 1)
        self.assertGreaterEqual(diag["occlusion_recoveries"], 1)

    def test_dynamic_yaw_alignment_flips_180(self):
        frames_input = [
            {"frame_id": str(index * 400000000), "num_points": 0,
             "num_detections": 1,
             "detections": [det("Car", index * 4.0, 0.0, 1)]}
            for index in range(5)
        ]
        for frame in frames_input:
            detection = frame["detections"][0]
            detection["box_lidar"][6] = math.pi
            detection["_step45_retracked"] = True
            detection["region"] = "dynamic"
        with TemporaryDirectory() as directory:
            coords = make_coords(Path(directory))
            tracks, _ = collect_world_tracks(frames_input, coords)
            result = align_dynamic_yaw(
                frames_input, tracks, coords, {1}, Step45Config())
        self.assertGreater(result["dynamic_yaw_aligned"], 0)
        for frame in frames_input:
            yaw = frame["detections"][0]["box_lidar"][6]
            self.assertLess(abs(math.cos(yaw) - 1.0), 1e-6)

    def test_duplicate_frame_ids_are_rejected(self):
        duplicate = frames([
            [det("Car", 0.0, 0.0, 1), det("Car", 1.0, 0.0, 1)],
        ])
        with self.assertRaises(AssertionError):
            verify_unique_frame_ids(duplicate)
        clean = frames([[det("Car", 0.0, 0.0, 1)]])
        self.assertTrue(verify_unique_frame_ids(clean)["passed"])

    def test_movement_gate_and_lane_change_limit(self):
        config = Step45Config()
        # straight -> left is only allowed as a left-side lane change <=5m.
        self.assertTrue(_movement_compatible(
            "straight", "left", 0.0, -1.0, config))
        self.assertFalse(_movement_compatible(
            "straight", "left", 0.0, 5.5, config))
        # right turn is hard-gated to the right side only.
        self.assertTrue(_movement_compatible(
            "straight", "right", 0.0, 1.0, config))
        self.assertFalse(_movement_compatible(
            "straight", "right", 0.0, -1.0, config))


if __name__ == "__main__":
    unittest.main()
