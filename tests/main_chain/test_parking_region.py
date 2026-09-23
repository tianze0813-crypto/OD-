import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from region.parking_region import (
    ParkingRegionConfig,
    SlotRecord,
    build_parking_regions,
    load_regions_json,
    save_regions_json,
)


def _slot(track_id: int, x: float, y: float, row_id: int = 0) -> SlotRecord:
    return SlotRecord(
        track_id=track_id,
        center=np.array([float(x), float(y)]),
        size=np.array([4.5, 2.0, 1.6]),
        yaw=0.0,
        row_id=row_id,
    )


def _road_points(y: float, tracks: int = 3) -> tuple[np.ndarray, np.ndarray,
                                                     np.ndarray]:
    points = []
    ids = []
    timestamps = []
    for track_id in range(tracks):
        for index, x in enumerate(np.linspace(-2.0, 15.0, 40)):
            points.append([float(x), float(y) + 0.1 * track_id])
            ids.append(100 + track_id)
            timestamps.append(index * 0.2)
    return (np.asarray(points, dtype=np.float64),
            np.asarray(ids, dtype=np.int64),
            np.asarray(timestamps, dtype=np.float64))


class ParkingRegionTest(unittest.TestCase):
    def test_two_parking_rows_are_separated_by_dynamic_road(self):
        slots = [_slot(1 + index, index * 2.6, 0.0, row_id=0)
                 for index in range(6)]
        slots += [_slot(20 + index, index * 2.6, 12.0, row_id=1)
                  for index in range(6)]
        static = np.asarray([[slot.center[0], slot.center[1]]
                             for slot in slots], dtype=np.float64)
        dynamic, dynamic_ids, timestamps = _road_points(6.0)
        config = ParkingRegionConfig(
            min_slots_per_region=5,
            min_region_area_m2=20.0,
            road_dilate_radius=1.5,
            road_min_track_net_displacement=5.0,
            road_min_track_mean_speed=1.0,
        )
        result = build_parking_regions(
            slots, static, dynamic, dynamic_ids, timestamps, config)
        self.assertEqual(len(result.parking_polygons), 2)
        self.assertEqual(len(result.road_polygons), 1)
        self.assertEqual(result.diagnostics["parking_slot_count"], 12)
        # The synthetic road is strictly between the two rows and must not be
        # part of a parking polygon.
        self.assertEqual(
            result.diagnostics["dynamic_points_inside_parking"], 0)
        self.assertGreater(
            result.diagnostics["dynamic_points_inside_road"], 0)

    def test_small_cluster_is_not_a_parking_region(self):
        slots = [_slot(1 + index, index * 2.6, 0.0, row_id=0)
                 for index in range(2)]
        static = np.asarray([[slot.center[0], slot.center[1]]
                             for slot in slots], dtype=np.float64)
        dynamic = np.zeros((0, 2), dtype=np.float64)
        dynamic_ids = np.zeros(0, dtype=np.int64)
        result = build_parking_regions(
            slots, static, dynamic, dynamic_ids, None,
            ParkingRegionConfig(min_slots_per_region=5))
        self.assertEqual(result.parking_polygons, [])
        self.assertEqual(result.diagnostics["parking_slot_count"], 0)

    def test_oriented_slot_footprints_follow_vehicle_yaw(self):
        slots = [
            SlotRecord(
                track_id=1,
                center=np.array([0.0, 0.0]),
                size=np.array([4.5, 2.0, 1.6]),
                yaw=math.pi / 2.0,
            ),
        ]
        static = np.asarray([[0.0, 0.0]], dtype=np.float64)
        result = build_parking_regions(
            slots, static, np.zeros((0, 2)), np.zeros(0, dtype=np.int64),
            None, ParkingRegionConfig(min_slots_per_region=1,
                                      min_region_area_m2=1.0))
        self.assertEqual(len(result.parking_polygons), 1)
        # A yaw=pi/2 vehicle is long along world y and narrow along world x.
        xs = [point[0] for point in result.parking_polygons[0]["polygon"]]
        ys = [point[1] for point in result.parking_polygons[0]["polygon"]]
        self.assertLess(max(xs) - min(xs), max(ys) - min(ys))

    def test_four_slot_cluster_is_kept_by_default_threshold(self):
        slots = [_slot(1 + index, index * 2.6, 0.0, row_id=0)
                 for index in range(4)]
        static = np.asarray([[slot.center[0], slot.center[1]]
                             for slot in slots], dtype=np.float64)
        result = build_parking_regions(
            slots, static, np.zeros((0, 2)), np.zeros(0, dtype=np.int64),
            None, ParkingRegionConfig(min_region_area_m2=1.0))
        self.assertEqual(len(result.parking_polygons), 1)
        self.assertEqual(result.diagnostics["parking_slot_count"], 4)

    def test_three_slot_cluster_is_dropped_as_noise(self):
        slots = [_slot(1 + index, index * 2.6, 0.0, row_id=0)
                 for index in range(3)]
        static = np.asarray([[slot.center[0], slot.center[1]]
                             for slot in slots], dtype=np.float64)
        result = build_parking_regions(
            slots, static, np.zeros((0, 2)), np.zeros(0, dtype=np.int64),
            None, ParkingRegionConfig(min_region_area_m2=1.0))
        self.assertEqual(result.parking_polygons, [])

    def test_region_json_round_trip(self):
        slots = [_slot(1 + index, index * 2.6, 0.0, row_id=0)
                 for index in range(5)]
        static = np.asarray([[slot.center[0], slot.center[1]]
                             for slot in slots], dtype=np.float64)
        result = build_parking_regions(
            slots, static, np.zeros((0, 2)), np.zeros(0, dtype=np.int64),
            None, ParkingRegionConfig(min_slots_per_region=5))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "regions.json"
            save_regions_json(result, path)
            loaded = load_regions_json(path)
        self.assertEqual(loaded["frame"], "world")
        self.assertEqual(len(loaded["parking_polygons"]), 1)
        self.assertEqual(
            loaded["diagnostics"]["parking_slot_count"], 5)


if __name__ == "__main__":
    unittest.main()
