import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tracking.tracker_conservative import CoordinateProvider, apply_post_filters
from tracking.tracker_static_first import StaticFirstTracker, StaticSlot, verify_identity_only


def coords(root: Path) -> CoordinateProvider:
    transforms = root / "transforms"
    transforms.mkdir()
    (transforms / "calib.json").write_text(json.dumps({
        "tf2base_link": {"pose": np.eye(4).tolist(),
                         "lidar_top": np.eye(4).tolist()}
    }))
    (transforms / "pose_data.txt").write_text("\n".join(
        f"{i * 1000000000},0,0,0,0,0,0,1" for i in range(30)) + "\n")
    return CoordinateProvider(root)


def detection(x, y, score=0.9):
    return {"class_name": "Vehicle", "score": score,
            "box_lidar": [float(x), float(y), 0.0, 4.5, 2.0, 1.6, 0.0]}


def frames(rows):
    return [{"frame_id": str(i * 200000000), "num_points": 0,
             "num_detections": len(items), "detections": items}
            for i, items in enumerate(rows)]


class StaticFirstTrackerTest(unittest.TestCase):
    def run_tracker(self, rows, **kwargs):
        with TemporaryDirectory() as directory:
            source = frames(rows)
            tracker = StaticFirstTracker(
                coords(Path(directory)), slot_min_hits=5,
                slot_min_duration=0.6, standalone_min_hits=5, **kwargs)
            output, diagnostics = tracker.process(source)
            return source, output, diagnostics

    def test_long_occlusion_reuses_same_static_identity(self):
        jitter = [0.0, 0.12, -0.10, 0.06, -0.04]
        rows = [[detection(jitter[i], 0.0)] for i in range(5)]
        rows += [[] for _ in range(10)]
        rows += [[detection(0.08, 0.03)], [detection(-0.05, -0.02)]]
        source, output, diagnostics = self.run_tracker(rows)
        ids = [d["track_id"] for f in output for d in f["detections"]]
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(diagnostics["static_slots"], 1)
        verify_identity_only(source, output)

    def test_dense_row_ids_do_not_cross_under_jitter(self):
        rows = []
        for i in range(12):
            jitter = 0.22 if i % 2 else -0.22
            rows.append([
                detection(0.0, 0.0 + jitter),
                detection(0.0, 2.5 - jitter),
                detection(0.0, 5.0 + jitter),
            ])
        _source, output, diagnostics = self.run_tracker(rows)
        per_position = [set(), set(), set()]
        for frame in output:
            for index, det in enumerate(sorted(frame["detections"],
                                               key=lambda d: d["box_lidar"][1])):
                per_position[index].add(det["track_id"])
        self.assertTrue(all(len(ids) == 1 for ids in per_position))
        self.assertEqual(len(set.union(*per_position)), 3)
        self.assertGreaterEqual(len(diagnostics["rows"]), 1)

    def test_topology_recovers_cross_row_offset_for_stable_slot(self):
        rows = []
        for index in range(12):
            left_cross = 1.55 if index >= 5 else 0.0
            rows.append([
                detection(left_cross, 0.0),
                detection(0.0, 2.5),
                detection(0.0, 5.0),
            ])
        source, output, diagnostics = self.run_tracker(rows)
        left_ids = {
            min(frame["detections"], key=lambda d: d["box_lidar"][1])["track_id"]
            for frame in output
        }
        self.assertEqual(len(left_ids), 1)
        topology = diagnostics["parking_row_topology"]
        self.assertGreaterEqual(topology["corrected_detections"], 5)
        verify_identity_only(source, output)

    def test_duplicate_noncoexisting_slot_is_merged_into_stronger_slot(self):
        rows = []
        for index in range(15):
            duplicate_x = 1.65 if index in {1, 4, 7, 10, 13} else 0.0
            rows.append([
                detection(duplicate_x, 0.0),
                detection(0.0, 2.5),
                detection(0.0, 5.0),
            ])
        _source, output, diagnostics = self.run_tracker(rows)
        self.assertEqual(diagnostics["duplicate_static_slots"]["merged_slots"], 1)
        self.assertEqual(diagnostics["static_slots"], 3)
        merged_slot = next(
            slot for slot in diagnostics["slot_details"]
            if slot["slot_id"] == diagnostics["duplicate_static_slots"]["merges"][0][
                "kept_slot_id"])
        self.assertEqual(merged_slot["alias_centers"], 1)
        first_car_ids = {
            min(frame["detections"], key=lambda d: d["box_lidar"][1])["track_id"]
            for frame in output
        }
        self.assertEqual(len(first_car_ids), 1)

    def test_short_bracketed_slot_gap_recovers_only_gap_fragment(self):
        with TemporaryDirectory() as directory:
            tracker = StaticFirstTracker(coords(Path(directory)))
            axis = np.array([0.0, 1.0], dtype=np.float64)
            tracker.slots = [StaticSlot(
                slot_id=41, track_id=41, center=np.array([0.0, 0.0]),
                yaw=0.0, size=np.array([4.5, 2.0, 1.6]),
                class_name="Vehicle", evidence_hits=10, row_id=1,
                row_axis=axis, row_coord=0.0)]
            source = frames([
                [dict(detection(0.0, 0.0), track_id=41)],
                [dict(detection(1.0, 0.1), track_id=155)],
                [dict(detection(1.0, -0.1), track_id=155)],
                [dict(detection(0.1, 0.0), track_id=41)],
            ])
            diagnostics = tracker._recover_slot_gap_fragments(source)
            self.assertEqual(diagnostics["corrected_detections"], 2)
            self.assertEqual(
                [frame["detections"][0]["track_id"] for frame in source],
                [41, 41, 41, 41])

            too_long = frames(
                [[dict(detection(0.0, 0.0), track_id=41)]]
                + [[dict(detection(1.0, 0.0), track_id=155)] for _ in range(4)]
                + [[dict(detection(0.0, 0.0), track_id=41)]])
            diagnostics = tracker._recover_slot_gap_fragments(too_long)
            self.assertEqual(diagnostics["corrected_detections"], 0)

    def test_topology_splits_impossible_motion_along_occupied_row(self):
        rows = []
        slot_y = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5]
        for frame_index in range(14):
            parked = [detection(0.0, y) for y in slot_y]
            if 6 <= frame_index <= 11:
                jumped_y = slot_y[frame_index - 6]
                parked = [d for d in parked
                          if abs(d["box_lidar"][1] - jumped_y) > 0.1]
                parked.append(detection(1.15, jumped_y))
            rows.append(parked)
        _source, output, diagnostics = self.run_tracker(rows)
        topology = diagnostics["parking_row_topology"]
        lateral = [x for x in topology["corrections"]
                   if x["reason"] == "lateral_row_jump"]
        self.assertTrue(lateral)
        jumped_ids = []
        for frame_index in range(6, 12):
            jumped_y = slot_y[frame_index - 6]
            det = next(d for d in output[frame_index]["detections"]
                       if abs(d["box_lidar"][1] - jumped_y) < 0.1)
            jumped_ids.append(det["track_id"])
        self.assertGreaterEqual(len(set(jumped_ids)), 4)

    def test_continued_slot_occupancy_rejects_false_departure(self):
        rows = []
        false_departure = (0.4, 1.2, 2.2, 3.4, 4.8, 6.2, 7.8, 9.4)
        for frame_index in range(14):
            items = [detection(0.0, 0.0), detection(0.0, 2.5),
                     detection(0.0, 5.0), detection(0.0, 7.5)]
            if frame_index >= 6:
                items.append(detection(false_departure[frame_index - 6], 0.0))
            rows.append(items)
        _source, _output, diagnostics = self.run_tracker(rows)
        coordination = diagnostics["slot_motion_coordination"]
        self.assertEqual(coordination["departure_count"], 0)
        self.assertGreater(coordination["rejected_departure_count"], 0)
        self.assertIn("slot_remains_occupied", {
            x["reason"] for x in coordination["rejected_departures"]})

    def test_static_pass_is_class_blind(self):
        rows = [[detection(0.0, 0.0), {
            "class_name": "Pedestrian", "score": 0.9,
            "box_lidar": [4.0, 0.0, 0.0, 0.8, 0.8, 1.7, 0.0],
        }] for _ in range(8)]
        _source, output, diagnostics = self.run_tracker(rows)
        self.assertEqual(diagnostics["static_slots"], 2)
        vehicle_ids = {f["detections"][0]["track_id"] for f in output}
        pedestrian_ids = {f["detections"][1]["track_id"] for f in output}
        self.assertEqual(len(vehicle_ids), 1)
        self.assertEqual(len(pedestrian_ids), 1)
        self.assertNotEqual(vehicle_ids, pedestrian_ids)

    def test_short_filter_still_runs_after_both_passes(self):
        rows = [[detection(0.0, 0.0)] for _ in range(8)]
        rows[1].append(detection(20.0, 20.0))
        rows[2].append(detection(20.2, 20.0))
        _source, output, _diagnostics = self.run_tracker(rows)
        stats = apply_post_filters(output, min_lifecycle=4)
        self.assertEqual(stats["tracks_dropped"], 1)
        self.assertEqual(sum(len(f["detections"]) for f in output), 8)

    def test_departing_vehicle_keeps_parked_identity(self):
        rows = [[detection(0.0, 0.0)] for _ in range(6)]
        rows += [[detection(x, 0.0)] for x in (0.4, 1.2, 2.2, 3.4, 4.8, 6.2)]
        _source, output, diagnostics = self.run_tracker(rows)
        ids = [d["track_id"] for f in output for d in f["detections"]]
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(
            diagnostics["slot_motion_coordination"]["departure_count"], 1)

    def test_vehicle_leaving_perpendicular_to_parking_row_keeps_id(self):
        rows = []
        for _ in range(6):
            rows.append([detection(0.0, 0.0), detection(0.0, 2.5),
                         detection(0.0, 5.0)])
        for cross in (0.4, 1.2, 2.2, 3.4, 4.8, 6.2):
            rows.append([detection(cross, 0.0), detection(0.0, 2.5),
                         detection(0.0, 5.0)])
        _source, output, diagnostics = self.run_tracker(rows)
        leaving_ids = [
            min(frame["detections"], key=lambda d: d["box_lidar"][1])["track_id"]
            for frame in output
        ]
        self.assertEqual(len(set(leaving_ids)), 1)
        self.assertEqual(
            diagnostics["slot_motion_coordination"]["departure_count"], 1)

    def test_red_light_follower_keeps_own_id_in_leader_position(self):
        rows = []
        for _ in range(6):
            rows.append([detection(0.0, 0.0), detection(-6.0, 0.0)])
        leader = (0.5, 1.5, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0)
        follower = (-6.0, -5.0, -4.0, -2.5, -1.0, 0.0, 0.0, 0.0)
        rows += [[detection(a, 0.0), detection(b, 0.0)]
                 for a, b in zip(leader, follower)]
        rows += [[detection(0.0, 0.0)] for _ in range(4)]
        _source, output, diagnostics = self.run_tracker(rows)
        leader_parked_id = output[0]["detections"][0]["track_id"]
        follower_parked_id = output[0]["detections"][1]["track_id"]
        follower_at_leader = [
            d["track_id"] for f in output[11:] for d in f["detections"]
            if abs(d["box_lidar"][0]) < 0.1
        ]
        self.assertNotEqual(leader_parked_id, follower_parked_id)
        self.assertTrue(follower_at_leader)
        self.assertEqual(set(follower_at_leader), {follower_parked_id})
        self.assertGreaterEqual(
            diagnostics["slot_motion_coordination"]["departure_count"], 2)
        self.assertGreaterEqual(
            diagnostics["slot_motion_coordination"]["ingress_count"], 1)

    def test_passing_vehicle_does_not_release_or_steal_parked_id(self):
        rows = []
        for index in range(12):
            parked = detection(0.0, 0.0)
            passer = detection(-8.0 + 1.6 * index, 2.8)
            rows.append([parked, passer])
        rows += [[] for _ in range(8)]
        rows += [[detection(0.05, -0.03)] for _ in range(3)]
        _source, output, diagnostics = self.run_tracker(rows)
        parked_ids = [
            d["track_id"] for f in output for d in f["detections"]
            if abs(d["box_lidar"][1]) < 0.2
        ]
        self.assertTrue(parked_ids)
        self.assertEqual(len(set(parked_ids)), 1)
        passer_ids = [
            d["track_id"] for f in output[:12] for d in f["detections"]
            if d["box_lidar"][1] > 2.0
        ]
        self.assertTrue(passer_ids)
        self.assertNotEqual(set(parked_ids), set(passer_ids))
        self.assertEqual(
            diagnostics["slot_motion_coordination"]["departure_count"], 0)


if __name__ == "__main__":
    unittest.main()
