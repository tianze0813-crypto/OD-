import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from region.traffic_light import (
    TrafficLightConfig,
    build_traffic_light_model,
    save_traffic_light_json,
)


def _track(points, *, start=0.0, dt=0.5, speed=None):
    items = []
    for index, point in enumerate(points):
        items.append({
            "timestamp": float(start + index * dt),
            "world": np.asarray(point, dtype=np.float64),
            "yaw": 0.0,
            "size": np.asarray([4.5, 2.0, 1.6]),
        })
    return items


def _straight_track(start=0.0):
    return _track([(0.0, -50.0 + index * 12.0) for index in range(10)],
                  start=start)


def _left_track(start=0.0):
    return _track(
        [(0.0, -50.0 + index * 12.0) for index in range(5)]
        + [(-12.0 * index, 10.0) for index in range(1, 5)],
        start=start)


def _right_track(start=0.0):
    return _track(
        [(0.0, -50.0 + index * 12.0) for index in range(5)]
        + [(12.0 * index, 10.0) for index in range(1, 5)],
        start=start)


def _enabled_config(**kwargs):
    values = dict(
        min_group_tracks=1,
        min_robust_tracks=1,
        min_straight_tracks=1,
        min_left_tracks=1,
        min_stop_events=0,
        max_straight_fraction=1.0,
    )
    values.update(kwargs)
    return TrafficLightConfig(**values)


class TrafficLightTest(unittest.TestCase):
    def test_movement_classification(self):
        result = build_traffic_light_model(
            {1: _straight_track(), 2: _left_track(), 3: _right_track()},
            config=_enabled_config())
        movements = {item["track_id"]: item["movement"]
                     for item in result.track_classification}
        self.assertEqual(movements[1], "straight")
        self.assertEqual(movements[2], "left")
        self.assertEqual(movements[3], "right")

    def test_movement_groups_and_right_always_green(self):
        result = build_traffic_light_model(
            {1: _straight_track(), 2: _left_track(), 3: _right_track()},
            config=_enabled_config())
        movements = {group["movement"] for group in result.groups}
        self.assertEqual(movements, {"straight", "left", "right"})
        # Right turn must be green in every inferred interval.
        for interval in result.phase_timeline:
            self.assertIn("right", interval["green"])

    def test_uturn_is_classified_as_left(self):
        # A physically plausible left U-turn: a half circle of radius 10 m
        # (entry heading north, exit heading south, ~20 m lateral offset).
        radius = 10.0
        points = []
        for index in range(16):
            angle = math.pi * index / 15
            points.append((-radius + radius * math.cos(angle),
                           radius * math.sin(angle)))
        uturn = _track(points)
        result = build_traffic_light_model(
            {1: uturn}, config=_enabled_config())
        self.assertEqual(result.track_classification[0]["movement"], "left")

    def test_turn_in_place_is_merged_to_straight(self):
        # A "turn in place" (multi-loop spin with no lateral displacement) is
        # not physically possible for a car; it must be treated as a yaw
        # artefact and merged into straight.
        radius = 1.5
        points = []
        for index in range(40):
            angle = 4.0 * math.pi * index / 39
            points.append((radius - radius * math.cos(angle),
                           radius * math.sin(angle) + 25.0 * index / 39))
        spin = _track(points)
        result = build_traffic_light_model(
            {1: spin}, config=_enabled_config())
        item = result.track_classification[0]
        self.assertTrue(item["turn_in_place"] or item["heading_inconsistent"])
        self.assertEqual(item["movement"], "straight")

    def _arc_track(self, degrees, radius=20.0, count=20, noise=0.0):
        points = []
        for index in range(count):
            angle = math.radians(degrees) * index / (count - 1)
            x = radius - radius * math.cos(angle)
            y = radius * math.sin(angle)
            if noise and 5 <= index <= 14:
                x += noise * math.sin(index * 2.3)
                y += noise * math.cos(index * 1.9)
            points.append((x, y))
        return _track(points)

    def test_heading_flip_is_not_uturn(self):
        flip = _track([(0.0, 0.0), (0.0, 10.0), (0.0, 20.0),
                       (0.0, 15.0), (0.0, 10.0), (0.0, 20.0),
                       (0.0, 30.0), (0.0, 40.0)])
        result = build_traffic_light_model(
            {1: flip}, config=_enabled_config())
        item = result.track_classification[0]
        self.assertEqual(item["movement"], "straight")
        self.assertTrue(item["heading_flip"])

    def test_weak_turn_in_straight_lane_is_merged(self):
        tracks = {index + 1: _straight_track(start=index * 0.1)
                  for index in range(5)}
        tracks[99] = self._arc_track(55.0, noise=0.19)
        result = build_traffic_light_model(
            tracks, config=_enabled_config())
        item = {entry["track_id"]: entry
                for entry in result.track_classification}[99]
        self.assertEqual(item["raw_movement"], "right")
        self.assertEqual(item["movement"], "straight")
        self.assertEqual(item["reclassify_reason"],
                         "straight_lane_minority")

    def test_lane_change_is_merged_to_straight(self):
        lane_change = _track([(0.0, 0.0), (0.0, 10.0), (1.0, 20.0),
                              (3.0, 30.0), (3.0, 40.0), (3.0, 50.0)])
        result = build_traffic_light_model(
            {1: lane_change}, config=_enabled_config())
        item = result.track_classification[0]
        self.assertEqual(item["movement"], "straight")
        self.assertTrue(item["lane_change"])

    def test_waiting_left_is_kept(self):
        def moving_north(x, start_y, count=12):
            return _track([(x, start_y + index * 6.0)
                           for index in range(count)])

        waiting = _track([(0.0, 0.0), (0.0, 10.0), (0.0, 20.0),
                          (0.0, 20.0), (0.0, 20.0), (0.0, 20.0)])
        result = build_traffic_light_model(
            {1: waiting, 2: moving_north(3.75, -30.0),
             3: moving_north(3.75, -60.0)},
            config=_enabled_config())
        item = {entry["track_id"]: entry
                for entry in result.track_classification}[1]
        self.assertTrue(item["waiting_left"])
        self.assertEqual(item["movement"], "left")

    def test_single_strong_turn_is_kept(self):
        result = build_traffic_light_model(
            {1: _left_track()}, config=_enabled_config())
        item = result.track_classification[0]
        self.assertEqual(item["movement"], "left")
        self.assertFalse(item["reclassified"])

    def test_movement_level_lanes_collapse_straight(self):
        # Four parallel straight lanes must produce one straight lane group,
        # not four laterally clustered lanes.
        tracks = {
            index + 1: _track([(index * 3.75, -50.0 + step * 12.0)
                                for step in range(10)])
            for index in range(4)
        }
        result = build_traffic_light_model(tracks, config=_enabled_config())
        lanes = result.diagnostics["lane_groups"]
        self.assertEqual(len(lanes), 1)
        self.assertEqual(lanes[0]["movement"], "straight")
        self.assertEqual(lanes[0]["track_count"], 4)

    def test_single_track_direction_has_no_lane(self):
        result = build_traffic_light_model(
            {1: _straight_track()}, config=_enabled_config())
        self.assertEqual(result.diagnostics["lane_count"], 0)
        self.assertEqual(result.diagnostics["unsupported_directions"], 1)
        self.assertIsNone(result.track_classification[0]["entry_lane_id"])

    def test_parallel_same_heading_roads_are_one_direction(self):
        # Two parallel roads 30 m apart with the same heading are the same
        # travel direction (reviewed rule: no lateral gate on directions).
        tracks = {
            1: _track([(0.0, -50.0 + step * 12.0) for step in range(10)]),
            2: _track([(30.0, -50.0 + step * 12.0) for step in range(10)]),
        }
        result = build_traffic_light_model(tracks, config=_enabled_config())
        self.assertEqual(result.diagnostics["direction_count"], 1)
        self.assertEqual(result.diagnostics["lane_count"], 1)

    def test_three_movement_lanes_per_direction(self):
        result = build_traffic_light_model(
            {1: _straight_track(), 2: _left_track(), 3: _right_track()},
            config=_enabled_config())
        movements = sorted(lane["movement"]
                           for lane in result.diagnostics["lane_groups"])
        self.assertEqual(movements, ["left", "right", "straight"])

    def test_normal_road_is_not_enabled_as_traffic_light(self):
        tracks = {index + 1: _straight_track(start=index * 0.1)
                  for index in range(5)}
        result = build_traffic_light_model(
            tracks, config=TrafficLightConfig(
                min_group_tracks=1, min_robust_tracks=25,
                min_straight_tracks=5, min_left_tracks=3,
                min_stop_events=5, max_straight_fraction=0.85,
                enable_gating=True))
        self.assertFalse(result.traffic_light_enabled)
        self.assertEqual(result.phase_timeline, [])

    def test_traffic_light_logic_is_enabled_by_default(self):
        tracks = {index + 1: _straight_track(start=index * 0.1)
                  for index in range(5)}
        result = build_traffic_light_model(tracks, config=TrafficLightConfig(
            min_group_tracks=1, min_robust_tracks=25,
            min_straight_tracks=5, min_left_tracks=3,
            min_stop_events=5, max_straight_fraction=0.85))
        self.assertTrue(result.traffic_light_enabled)

    def test_non_motor_vehicle_tracks_are_excluded_from_phase_model(self):
        car = _straight_track()
        pedestrian = _straight_track()
        for item in car:
            item["class_name"] = "Car"
        for item in pedestrian:
            item["class_name"] = "Pedestrian"
        result = build_traffic_light_model(
            {1: car, 2: pedestrian}, config=_enabled_config())
        self.assertEqual(result.diagnostics["motor_vehicle_tracks"], 1)
        self.assertEqual(result.diagnostics["non_motor_vehicle_tracks"], 1)
        self.assertEqual(
            {item["track_id"] for item in result.track_classification}, {1})

    def test_json_round_trip(self):
        result = build_traffic_light_model(
            {1: _straight_track()},
            config=_enabled_config())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "traffic_light.json"
            save_traffic_light_json(result, path)
            self.assertIn("phase_timeline", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
