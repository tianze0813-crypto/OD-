import unittest

import numpy as np

from geometry.car_box_fit import CarBoxFitConfig, _roof_only_boundary


def make_points():
    rng = np.random.default_rng(7)
    x = rng.uniform(-2.4, 2.4, 80)
    y = rng.uniform(-1.1, 1.1, 80)
    z = rng.uniform(-0.6, 0.2, 80)
    rx = rng.uniform(-1.6, 1.6, 24)
    ry = rng.uniform(-0.6, 0.6, 24)
    rz = 1.20 + rng.normal(0.0, 0.025, 24)
    return np.column_stack([
        np.r_[x, rx, 0.0],
        np.r_[y, ry, 0.0],
        np.r_[z, rz, 1.62],
    ]).astype(np.float64)


class RoofOnlyBoundaryTest(unittest.TestCase):
    def test_supported_roof_cluster_wins_over_single_high_outlier(self):
        box = [0.0, 0.0, 0.45, 4.5, 1.8, 1.5, 0.0]
        roof_z, detail = _roof_only_boundary(
            make_points(), box, CarBoxFitConfig())
        self.assertIsNotNone(roof_z)
        self.assertLess(abs(roof_z - 1.20), 0.06)
        self.assertLess(roof_z, 1.40)
        self.assertGreaterEqual(detail["roof_support"], 10)

    def test_too_few_points_are_rejected(self):
        box = [0.0, 0.0, 0.45, 4.5, 1.8, 1.5, 0.0]
        points = np.asarray([
            [0.0, 0.0, 1.20],
            [0.1, 0.0, 1.21],
            [0.2, 0.0, 1.19],
        ], dtype=np.float64)
        roof_z, detail = _roof_only_boundary(
            points, box, CarBoxFitConfig())
        self.assertIsNone(roof_z)
        self.assertEqual(detail["rejected_reason"], "too_few_crop_points")


if __name__ == "__main__":
    unittest.main()
