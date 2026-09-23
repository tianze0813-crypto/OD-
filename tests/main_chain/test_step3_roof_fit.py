import unittest

import numpy as np

from geometry.car_box_fit import (
    CarBoxFitConfig,
    _fit_z_boundaries,
    _repair_track_ground,
    _roof_evidence,
)


def roof_patch(z: float, x_offset: float = 0.0) -> np.ndarray:
    return np.asarray([
        [x + x_offset, y, z + dz]
        for x in (-0.70, -0.35, 0.0, 0.35, 0.70)
        for y in (-0.30, 0.0, 0.30)
        for dz in (-0.015, 0.0, 0.015)
    ], dtype=np.float64)


class BottomUpRoofEvidenceTest(unittest.TestCase):
    config = CarBoxFitConfig()
    bottom = 0.25
    box = [0.0, 0.0, 1.45, 4.4, 2.0, 2.4, 0.0]

    def test_continuous_roof_wins_over_high_narrow_branch(self):
        roof = roof_patch(1.75)
        branch = np.asarray([
            [x, y, 2.55 + dz]
            for x in (-0.15, 0.0, 0.15)
            for y in (-0.04, 0.04)
            for dz in (-0.02, 0.02)
        ], dtype=np.float64)

        roof_z, support, detail = _roof_evidence(
            np.vstack((roof, branch)), self.box, self.bottom, self.config)

        self.assertIsNotNone(roof_z)
        self.assertLess(abs(float(roof_z) - 1.765), 0.02)
        self.assertGreaterEqual(support, self.config.roof_min_points)
        self.assertEqual(detail["scan_direction"], "bottom_up")
        self.assertEqual(detail["scan_start"], self.bottom)
        self.assertTrue(detail["roof_gap_reached"])
        self.assertTrue(detail["shape"]["center_covered"])

    def test_patch_outside_footprint_center_is_rejected(self):
        roof_z, support, detail = _roof_evidence(
            roof_patch(1.75, x_offset=1.25), self.box, self.bottom,
            self.config)

        self.assertIsNone(roof_z)
        self.assertEqual(support, 0)
        self.assertEqual(detail["rejected_reason"],
                         "no_continuous_roof_section")

    def test_single_sparse_high_layer_is_not_a_roof(self):
        sparse = np.asarray([
            [x, y, 2.30 + dz]
            for x in (-0.9, 0.9)
            for y in (-0.45, 0.45)
            for dz in (-0.005, 0.005)
        ], dtype=np.float64)

        roof_z, support, detail = _roof_evidence(
            sparse, self.box, self.bottom, self.config)

        self.assertIsNone(roof_z)
        self.assertEqual(support, 0)
        self.assertIn(detail["rejected_reason"], {
            "no_continuous_roof_section",
            "roof_section_not_vertically_continuous",
        })

    def test_two_side_spans_with_empty_center_are_not_a_roof(self):
        # The two patches have enough total span and grid connectivity to pass
        # the old envelope check, but the box center contains no return.
        side_patches = np.asarray([
            [x, y, 2.30 + dz]
            for x in (-1.00, -0.75, 0.75, 1.00)
            for y in (-0.35, 0.35)
            for dz in (-0.005, 0.005)
        ], dtype=np.float64)

        roof_z, support, detail = _roof_evidence(
            side_patches, self.box, self.bottom, self.config)

        self.assertIsNone(roof_z)
        self.assertEqual(support, 0)
        self.assertEqual(detail["rejected_reason"],
                         "no_continuous_roof_section")


class ZBoundaryFallbackTest(unittest.TestCase):
    config = CarBoxFitConfig(z_min_center_change=0.0)
    track_height = 1.60

    @staticmethod
    def item(ground_z=None, ground_points=0, roof_z=None, roof_points=0):
        return {
            "det": {"box_lidar": [0.0, 0.0, 1.50, 4.4, 2.0, 2.4, 0.0]},
            "ground_z": ground_z,
            "ground_points": ground_points,
            "roof_z": roof_z,
            "roof_points": roof_points,
        }

    def test_both_reasonable_boundaries_are_used_directly(self):
        fit_z, height, mode = _fit_z_boundaries(
            self.item(0.20, 18, 1.70, 6), self.track_height, self.config)
        self.assertEqual(mode, "both")
        self.assertAlmostEqual(height, 1.50)
        self.assertAlmostEqual(fit_z, 0.99)

    def test_weak_roof_that_disagrees_with_track_height_uses_prior(self):
        item = self.item(0.20, 18, 1.45, 6)
        item["roof_detail"] = {"selected_run": [4, 5]}
        fit_z, height, mode = _fit_z_boundaries(
            item, self.track_height, self.config)
        self.assertEqual(mode, "ground_prior")
        self.assertAlmostEqual(height, self.track_height)
        self.assertAlmostEqual(fit_z, 1.04)

    def test_small_track_height_error_still_uses_prior(self):
        item = self.item(-1.685, 49, -0.125, 157)
        item["roof_detail"] = {"selected_run": [29, 30]}
        fit_z, height, mode = _fit_z_boundaries(
            item, 1.7456, self.config)
        self.assertEqual(mode, "ground_prior")
        self.assertAlmostEqual(height, 1.7456)
        self.assertAlmostEqual(fit_z, -0.7722, places=3)

    def test_strong_roof_run_keeps_reasonable_boundary_pair(self):
        item = self.item(0.20, 18, 1.70, 6)
        item["roof_detail"] = {"selected_run": [1, 2, 3, 4, 5]}
        fit_z, height, mode = _fit_z_boundaries(
            item, self.track_height, self.config)
        self.assertEqual(mode, "both")
        self.assertAlmostEqual(height, 1.50)
        self.assertAlmostEqual(fit_z, 0.99)

    def test_unreasonable_pair_keeps_ground_and_uses_track_height(self):
        fit_z, height, mode = _fit_z_boundaries(
            self.item(0.20, 18, 3.00, 6), self.track_height, self.config)
        self.assertEqual(mode, "ground_prior")
        self.assertAlmostEqual(height, self.track_height)
        self.assertAlmostEqual(fit_z, 1.04)

    def test_ground_only_keeps_ground_and_uses_track_height(self):
        fit_z, height, mode = _fit_z_boundaries(
            self.item(0.20, 18), self.track_height, self.config)
        self.assertEqual(mode, "ground")
        self.assertAlmostEqual(height, self.track_height)
        self.assertAlmostEqual(fit_z, 1.04)

    def test_roof_only_keeps_roof_and_uses_track_height(self):
        fit_z, height, mode = _fit_z_boundaries(
            self.item(roof_z=1.70, roof_points=6),
            self.track_height, self.config)
        self.assertEqual(mode, "roof_downward")
        self.assertAlmostEqual(height, self.track_height)
        self.assertAlmostEqual(fit_z, 0.94)

    def test_no_boundaries_keeps_center_and_uses_track_height(self):
        fit_z, height, mode = _fit_z_boundaries(
            self.item(), self.track_height, self.config)
        self.assertEqual(mode, "raw_fallback")
        self.assertAlmostEqual(height, self.track_height)
        self.assertAlmostEqual(fit_z, 1.50)


class GroundTemporalRepairTest(unittest.TestCase):
    config = CarBoxFitConfig()

    @staticmethod
    def item(timestamp: int, ground_z: float, points: int = 30):
        return {
            "timestamp": timestamp,
            "ground_z": ground_z,
            "ground_points": points,
        }

    def test_bimodal_jump_is_interpolated_between_lower_surface_samples(self):
        items = [
            self.item(0, -1.80),
            self.item(20_000_000, -1.82),
            self.item(40_000_000, -1.25),
            self.item(60_000_000, -1.23),
            self.item(80_000_000, -1.24),
            self.item(100_000_000, -1.81),
            self.item(120_000_000, -1.83),
        ]
        repaired = _repair_track_ground(items, self.config)
        self.assertEqual(repaired, 3)
        self.assertAlmostEqual(items[2]["ground_z"], -1.8175)
        self.assertAlmostEqual(items[3]["ground_z"], -1.815)
        self.assertAlmostEqual(items[4]["ground_z"], -1.8125)
        self.assertTrue(items[2]["ground_detail"]["repaired"])

    def test_single_jump_is_left_untouched(self):
        items = [
            self.item(0, -1.80),
            self.item(20_000_000, -1.25),
            self.item(40_000_000, -1.82),
        ]
        repaired = _repair_track_ground(items, self.config)
        self.assertEqual(repaired, 0)
        self.assertAlmostEqual(items[1]["ground_z"], -1.25)

    def test_long_prefix_uses_single_following_ground_anchor(self):
        items = [
            self.item(0, -1.00),
            self.item(20_000_000, -1.02),
            self.item(40_000_000, -1.01),
            self.item(60_000_000, -1.80),
            self.item(80_000_000, -1.82),
            self.item(100_000_000, -1.81),
            self.item(120_000_000, -1.00),
            self.item(140_000_000, -1.80),
            self.item(160_000_000, -1.82),
            self.item(180_000_000, -1.81),
        ]
        repaired = _repair_track_ground(items, self.config)
        self.assertGreaterEqual(repaired, 4)
        for item in items[:3]:
            self.assertAlmostEqual(item["ground_z"], -1.80, places=2)
            self.assertTrue(item["ground_detail"]["repaired"])


if __name__ == "__main__":
    unittest.main()
