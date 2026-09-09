import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from region.dynamic_region import (
    DynamicRegionConfig,
    build_dynamic_regions,
    save_dynamic_regions_json,
)


def _high_speed_track(track_id: int = 1) -> list[tuple[float, float, float]]:
    return [
        (index * 0.5, float(index * 6.0), 0.0)
        for index in range(6)
    ]


def _low_speed_track(track_id: int = 2) -> list[tuple[float, float, float]]:
    return [
        (index * 0.5, 0.0, float(index * 2.0))
        for index in range(6)
    ]


class DynamicRegionTest(unittest.TestCase):
    def test_high_speed_track_becomes_dynamic_region(self):
        result = build_dynamic_regions(
            {1: _high_speed_track(), 2: _low_speed_track()},
            config=DynamicRegionConfig())
        self.assertEqual(len(result.dynamic_polygons), 1)
        self.assertEqual(result.diagnostics["high_speed_tracks"], 1)
        self.assertEqual(result.diagnostics["high_speed_track_ids"], [1])
        self.assertGreater(result.diagnostics["dynamic_area_m2"], 0)

    def test_default_region_has_no_buffer(self):
        result = build_dynamic_regions({1: _high_speed_track()})
        polygon = np.asarray(result.dynamic_polygons[0]["polygon"],
                             dtype=np.float32)
        # The swept 2 m wide car box reaches y = 1.0; there is no 1 m buffer.
        inside = cv2.pointPolygonTest(polygon, (15.0, 0.5), False)
        self.assertGreaterEqual(inside, 0.0)
        outside = cv2.pointPolygonTest(polygon, (15.0, 1.8), False)
        self.assertLess(outside, 0.0)
        self.assertEqual(result.config["buffer_radius"], 0.0)
        # Core and dynamic polygons are the same when no buffer is used.
        core = np.asarray(result.core_polygons[0]["polygon"],
                          dtype=np.float32)
        self.assertGreaterEqual(
            cv2.pointPolygonTest(core, (15.0, 0.5), False), 0.0)
        self.assertLess(
            cv2.pointPolygonTest(core, (15.0, 1.8), False), 0.0)

    def test_stable_heading_extension_covers_thirty_metres(self):
        result = build_dynamic_regions({1: _high_speed_track()})
        polygon = np.asarray(result.dynamic_polygons[0]["polygon"],
                             dtype=np.float32)
        # Track runs x = 0..30.  Both ends are extended by 30 m along +x.
        self.assertGreaterEqual(
            cv2.pointPolygonTest(polygon, (-20.0, 0.0), False), 0.0)
        self.assertGreaterEqual(
            cv2.pointPolygonTest(polygon, (55.0, 0.0), False), 0.0)
        self.assertLess(
            cv2.pointPolygonTest(polygon, (65.0, 0.0), False), 0.0)
        self.assertGreater(result.diagnostics["extended_tracks"], 0)
        self.assertEqual(
            result.diagnostics["extension_length_m"], 30.0)

    def test_track_hopping_across_static_slots_is_rejected(self):
        static_slots = [
            {"center": (float(index * 6.0), 0.0),
             "yaw": 0.0, "size": (4.5, 2.0, 1.6)}
            for index in range(6)
        ]
        result = build_dynamic_regions(
            {1: _high_speed_track()},
            static_slots=static_slots,
            config=DynamicRegionConfig())
        self.assertEqual(result.diagnostics["high_speed_tracks"], 0)
        self.assertEqual(result.dynamic_polygons, [])
        self.assertEqual(
            len(result.diagnostics["rejected_static_overlap_tracks"]), 1)

    def test_static_slot_footprint_is_removed_from_dynamic_mask(self):
        # Static slots are 1.2 m to the side: they do not trigger the hard
        # overlap rejection (radius 1.0 m), but their footprint is still
        # removed from the final dynamic mask.
        track = [
            (index * 0.5, float(index * 6.0), 0.0)
            for index in range(6)
        ]
        static_slots = [
            {"center": (0.0, 1.2), "yaw": 0.0,
             "size": (4.5, 2.0, 1.6)},
            {"center": (6.0, 1.2), "yaw": 0.0,
             "size": (4.5, 2.0, 1.6)},
        ]
        result = build_dynamic_regions(
            {1: track}, static_slots=static_slots,
            config=DynamicRegionConfig(mask_static_slot_footprints=True))
        self.assertEqual(result.diagnostics["high_speed_tracks"], 1)
        self.assertGreater(
            result.diagnostics["static_slot_masked_area_m2"], 0)
        polygon = np.asarray(result.dynamic_polygons[0]["polygon"],
                             dtype=np.float32)
        self.assertLess(
            cv2.pointPolygonTest(polygon, (0.0, 1.2), False), 0.0)

    def test_short_high_speed_track_is_rejected(self):
        short = [(index * 0.5, float(index * 2.5), 0.0)
                 for index in range(6)]
        result = build_dynamic_regions(
            {1: short}, config=DynamicRegionConfig(min_track_length=15.0))
        self.assertEqual(result.dynamic_polygons, [])
        self.assertEqual(result.diagnostics["high_speed_tracks"], 0)

    def test_json_round_trip(self):
        result = build_dynamic_regions({1: _high_speed_track()})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dynamic_regions.json"
            save_dynamic_regions_json(result, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["frame"], "world")
        self.assertEqual(len(loaded["dynamic_polygons"]), 1)
        self.assertEqual(
            loaded["diagnostics"]["high_speed_threshold"], 5.0)


if __name__ == "__main__":
    unittest.main()
