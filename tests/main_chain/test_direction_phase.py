import math
import unittest

import numpy as np

from region.direction_phase import (
    _crossing_flags,
    _hold_unknowns,
    build_direction_phase_diagnostics,
    pair_directions_into_axes,
)
from region.traffic_light import (
    TrafficLightConfig,
    _build_direction_context,
)


def _stats(points, movement, stable_heading, stable_point, class_name="Car"):
    normalized = [
        {
            "timestamp": float(index),
            "world": np.asarray(point, dtype=np.float64),
            "yaw": 0.0,
            "size": np.asarray([4.5, 2.0, 1.6], dtype=np.float64),
            "class_name": class_name,
        }
        for index, point in enumerate(points)
    ]
    return {
        "normalized": normalized,
        "movement": movement,
        "raw_movement": movement,
        "final_movement": movement,
        "class_name": class_name,
        "stable_heading": float(stable_heading),
        "stable_point": np.asarray(stable_point, dtype=np.float64),
        "stable_points": np.asarray(points, dtype=np.float64),
    }


def _config(**kwargs):
    values = dict(
        min_direction_tracks=1,
        min_group_tracks=1,
        phase_bin_sec=0.2,
    )
    values.update(kwargs)
    return TrafficLightConfig(**values)


class DirectionPhaseTest(unittest.TestCase):
    def test_pair_opposite_directions_into_axes(self):
        directions = [
            {"direction_id": 0, "heading": 0.0, "track_ids": [1, 2],
             "track_count": 2},
            {"direction_id": 1, "heading": math.pi, "track_ids": [3],
             "track_count": 1},
            {"direction_id": 2, "heading": math.pi / 2.0, "track_ids": [4],
             "track_count": 1},
        ]
        records, axes = pair_directions_into_axes(directions, _config())
        self.assertEqual(len(axes), 2)
        axis_of = {record["direction_id"]: record["axis_id"]
                   for record in records}
        self.assertEqual(axis_of[0], axis_of[1])
        self.assertNotEqual(axis_of[0], axis_of[2])

    def test_slow_creep_across_stop_line_is_a_crossing(self):
        times = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        along = np.asarray([0.0, 0.5, 1.5, 2.5, 3.5])
        flags = _crossing_flags(
            times, along, line=2.0, tolerance=0.5, crossing_step=2.0,
            max_gap_sec=2.0)
        self.assertTrue(bool(flags[3]))

    def test_hold_unknowns_keeps_short_gaps(self):
        states = ["green", "unknown", "unknown", "red", "unknown"]
        held, flags = _hold_unknowns(states, hold=2)
        self.assertEqual(held[:3], ["green", "green", "green"])
        # A short trailing gap is also held: the signal does not disappear
        # just because no track of that group is visible for one bin.
        self.assertEqual(held[4], "red")
        self.assertEqual(flags[:3], [False, True, True])
        self.assertTrue(flags[4])

    def test_direction_phase_pipeline_on_synthetic_tracks(self):
        classified = [
            (1, _stats([(index * 5.0, 0.0) for index in range(10)],
                       "straight", 0.0, (20.0, 0.0))),
            (2, _stats(
                [(index * 5.0, 0.0) for index in range(5)]
                + [(20.0, index * 5.0) for index in range(1, 5)],
                "left", 0.0, (10.0, 0.0))),
            (3, _stats([(-index * 5.0, 0.0) for index in range(10)],
                       "straight", math.pi, (-20.0, 0.0))),
            (4, _stats([(0.0, index * 5.0) for index in range(10)],
                       "straight", math.pi / 2.0, (0.0, 20.0))),
        ]
        config = _config()
        directions = _build_direction_context(classified, config)
        self.assertGreaterEqual(len(directions), 3)
        result = build_direction_phase_diagnostics(
            classified, directions, {}, config)
        self.assertGreaterEqual(len(result["axes"]), 2)
        self.assertTrue(result["direction_signal_timeline"])
        self.assertTrue(result["axis_phase"]["timeline"])
        self.assertEqual(
            set(result["track_traffic_states"]), {1, 2, 3, 4})
        for state in result["track_traffic_states"].values():
            self.assertTrue(state["intervals"])


if __name__ == "__main__":
    unittest.main()
