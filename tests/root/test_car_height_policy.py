"""Car 顶/底分档（最终版）回归测试：车顶主锚 + 地面/先验定框底 + 静止排共底。"""

import dataclasses
import unittest

import numpy as np

from geometry.car_box_fit import (
    CarBoxFitConfig,
    _car_height_ground_ok,
    _apply_car_height_policy,
    _car_height_longest_row,
    _car_height_ring_ground,
    _car_height_roof_frame,
)

GROUND = -1.70
BOX = [0.0, 0.0, -0.75, 4.6, 1.9, 1.90, 0.0]          # 底 -1.70 / 顶 +0.20
TEST_CONFIG = dataclasses.replace(CarBoxFitConfig(), car_height_ground_min_points=6,
                                  car_height_roof_min_points=4, car_height_clearance_m=0.0)


def cloud(roof_z=None, ground=GROUND, span=20.0, centre=(0.0, 0.0), body=True):
    """地面 + 可选车体/车顶小片。"""
    xs = np.arange(-span, span + 0.5, 1.0)
    pts = [[x, y, ground] for x in xs for y in xs]
    if roof_z is not None:
        pts += [[centre[0] + x, centre[1] + y, roof_z]
                for x in np.arange(-1.6, 1.61, 0.4) for y in np.arange(-0.6, 0.61, 0.4)]
        if body:
            pts += [[centre[0] + x, centre[1] + y, ground + 0.6]
                    for x in np.arange(-1.8, 1.81, 0.6) for y in (-0.8, 0.8)]
    return np.asarray(pts, dtype=np.float64)


class FakeLidar:
    def __init__(self, frames):
        self.frames = frames

    def get(self, frame_id):
        return self.frames.get(frame_id)


def item(frame_id, box, cls="Car", world=(0.0, 0.0), oid=1):
    matrix = np.eye(4)
    matrix[0, 3], matrix[1, 3] = world
    return {
        "frame_id": frame_id,
        "det": {"class_name": cls, "track_id": oid, "box_lidar": list(box)},
        "raw_world": np.asarray([world[0], world[1], 0.0]),
        "world_from_lidar": matrix,
        "lidar_from_world": np.linalg.inv(matrix),
    }


class RoofFrameTest(unittest.TestCase):
    config = TEST_CONFIG

    def test_roof_is_the_p90_of_the_box_points(self):
        points = cloud(GROUND + 1.60)
        roof = _car_height_roof_frame(points, BOX, self.config)
        self.assertIsNotNone(roof)
        self.assertGreater(roof, GROUND + 1.0)          # 在车顶附近（含车身点）
        self.assertLessEqual(roof, GROUND + 1.60 + 1e-6)

    def test_outlier_above_is_rejected(self):
        points = np.vstack([cloud(GROUND + 1.60),
                            np.asarray([[0.0, 0.0, GROUND + 12.0]])])
        roof = _car_height_roof_frame(points, BOX, self.config)
        self.assertLess(roof, GROUND + 2.0)             # 12 m 的离群点不能把顶拉上去

    def test_too_few_points_gives_none(self):
        points = np.asarray([[0.0, 0.0, GROUND + 1.5], [0.5, 0.0, GROUND + 1.5]])
        self.assertIsNone(_car_height_roof_frame(points, BOX, self.config))


class PolicyTest(unittest.TestCase):
    config = TEST_CONFIG

    def run_policy(self, dets, frames, static=()):
        tracks = {dets[0]["det"]["track_id"]: dets}
        stats = _apply_car_height_policy(tracks, FakeLidar(frames), self.config, static)
        return dets, stats

    def test_box_top_sits_on_the_roof_and_bottom_is_trimmed_to_the_ground(self):
        det = item("f0", BOX)
        dets, stats = self.run_policy([det], {"f0": cloud(GROUND + 1.60)}, static=[1])

        box = dets[0]["det"]["box_lidar"]
        self.assertAlmostEqual(box[2] + box[5] / 2.0, GROUND + 1.60, places=2)   # 顶钉在车顶
        self.assertGreaterEqual(box[5], 1.45 - 1e-6)       # 输出高度恒在 [1.45,1.70]
        self.assertLessEqual(box[5], 1.70 + 1e-6)
        self.assertAlmostEqual(box[5], 1.60, places=2)      # 裁到地面后 = 车顶−地面
        self.assertAlmostEqual(box[2] - box[5] / 2.0, GROUND, places=2)          # 底贴地
        self.assertEqual(stats["static_tracks"], 1)

    def test_missing_ground_falls_back_to_the_high_prior(self):
        det = item("f0", BOX)
        # 1.45~1.70 窗口里找不到地面 → 回退 1.70（输出高度恒在 [1.45,1.70]）
        dets, _stats = self.run_policy([det], {"f0": cloud(GROUND + 1.60, ground=GROUND + 0.60)}, static=[1])
        box = dets[0]["det"]["box_lidar"]
        self.assertAlmostEqual(box[5], 1.70, places=3)

    def test_dynamic_track_keeps_its_own_roof_and_one_id_height(self):
        # 车顶纯逐帧实测（滚动中位门控已去掉）；高度 = 本 ID 各帧最大深度 → 全 ID 统一
        distances = (0.0, 20.0, 40.0, 60.0)
        dets = [item("f%d" % i, [0.0, y, -0.75, 4.6, 1.9, 1.90, 0.0])
                for i, y in enumerate(distances)]
        roofs = (1.60, 1.60, 2.10, 1.60)          # 中间那帧被噪点抬高
        frames = {"f%d" % i: cloud(GROUND + roof, span=20.0, centre=(0.0, y))
                  for i, (roof, y) in enumerate(zip(roofs, distances))}
        dets, stats = self.run_policy(dets, frames)
        tops = [d["det"]["box_lidar"][2] + d["det"]["box_lidar"][5] / 2.0 for d in dets]
        heights = [d["det"]["box_lidar"][5] for d in dets]
        self.assertEqual(stats["moving_tracks"], 1)
        self.assertAlmostEqual(tops[2], GROUND + 2.10, places=2)   # 顶保留本帧实测（不再拉回）
        self.assertEqual(len({round(h, 3) for h in heights}), 1)   # 整个 ID 一个高度
        self.assertAlmostEqual(heights[0], 1.60, places=2)         # = 该 ID 各帧最大深度

    def test_static_row_shares_one_ground_when_a_frame_has_none(self):
        # 三辆并排的静止车（中心共线），其中第三辆自己那帧没有地面点
        first = item("f0", BOX, world=(0.0, 0.0), oid=1)
        second = item("f0", [6.0, 0.0, -0.75, 4.6, 1.9, 1.90, 0.0], world=(6.0, 0.0), oid=2)
        third = item("f1", [12.0, 0.0, -0.75, 4.6, 1.9, 1.90, 0.0], world=(12.0, 0.0), oid=3)
        roof_only = np.asarray([[12.0 + x, y, GROUND + 1.60] for x in np.arange(-1.6, 1.61, 0.4)
                                for y in np.arange(-0.6, 0.61, 0.4)])
        frames = {"f0": np.vstack([cloud(GROUND + 1.60, span=8.0, centre=(0.0, 0.0)),
                                   cloud(GROUND + 1.60, span=8.0, centre=(6.0, 0.0))]),
                  "f1": roof_only}
        tracks = {1: [first], 2: [second], 3: [third]}
        stats = _apply_car_height_policy(tracks, FakeLidar(frames), self.config, [1, 2, 3])
        self.assertGreaterEqual(stats.get("row_trimmed_boxes", 0), 1)
        for oid in (1, 2, 3):
            box = tracks[oid][0]["det"]["box_lidar"]
            self.assertAlmostEqual(box[2] - box[5] / 2.0, GROUND, places=2)   # 共底到真实地面

    def test_longest_row_picks_the_collinear_subset(self):
        points = np.asarray([[0.0, 0.0], [5.0, 0.1], [10.0, -0.1], [3.0, 6.0]])
        row = _car_height_longest_row(points, 0.5)
        self.assertEqual(sorted(row), [0, 1, 2])

    def test_only_car_is_touched(self):
        truck = item("f0", [0.0, 0.0, -0.6, 8.0, 2.4, 3.20, 0.0], cls="Truck")
        dets, stats = self.run_policy([truck], {"f0": cloud(GROUND + 1.60)})
        self.assertEqual(dets[0]["det"]["box_lidar"], [0.0, 0.0, -0.6, 8.0, 2.4, 3.20, 0.0])
        self.assertEqual(stats["tracks"], 0)

    def test_xy_and_yaw_are_untouched(self):
        box = [1.5, -2.5, -0.75, 4.6, 1.9, 1.90, 0.37]
        dets, _stats = self.run_policy([item("f0", box)], {"f0": cloud(GROUND + 1.62)}, static=[1])
        out = dets[0]["det"]["box_lidar"]
        self.assertEqual([out[0], out[1], out[3], out[4], out[6]], [1.5, -2.5, 4.6, 1.9, 0.37])

    def test_disabled_policy_leaves_boxes_alone(self):
        config = dataclasses.replace(self.config, car_height_policy_enabled=False)
        det = item("f0", BOX)
        tracks = {1: [det]}
        stats = _apply_car_height_policy(tracks, FakeLidar({"f0": cloud(GROUND + 1.60)}), config, [1])
        self.assertEqual(det["det"]["box_lidar"], list(BOX))
        self.assertFalse(stats["enabled"])

    def test_lowercase_class_name_is_recognised(self):
        det = item("f0", BOX, cls="car")
        dets, stats = self.run_policy([det], {"f0": cloud(GROUND + 1.60)}, static=[1])
        self.assertEqual(stats["tracks"], 1)
        self.assertLessEqual(dets[0]["det"]["box_lidar"][5], 1.70 + 1e-6)


class RingGroundTest(unittest.TestCase):
    config = TEST_CONFIG

    def test_ring_returns_the_local_ground(self):
        self.assertAlmostEqual(_car_height_ring_ground(cloud(), BOX, self.config), GROUND, places=3)

    def test_no_ring_points_yields_none(self):
        roof_only = np.asarray([[-1.0, 0.0, GROUND + 1.6], [1.0, 0.0, GROUND + 1.6]])
        self.assertIsNone(_car_height_ring_ground(roof_only, BOX, self.config))


class GroundPriorSlopeTest(unittest.TestCase):
    """地面先验：绝对差 ≤0.3 m 或 坡度 ≤ tan(5°) 都算合理。"""

    config = CarBoxFitConfig()

    def _ok(self, ground, ego_ground, distance):
        box = [distance, 0.0, 0.0, 4.6, 1.9, 1.6, 0.0]
        return _car_height_ground_ok(ground, ego_ground, box, None, self.config)

    def test_small_absolute_difference_is_accepted(self):
        self.assertTrue(self._ok(GROUND + 0.2, GROUND, 30.0))

    def test_far_target_on_a_mild_slope_is_accepted(self):
        # 20 m 外差 0.6 m → 坡度 1.7° < 5°
        self.assertTrue(self._ok(GROUND + 0.6, GROUND, 20.0))

    def test_far_target_with_a_large_jump_is_rejected(self):
        # 20 m 外差 3 m → 坡度 8.5° > 5°
        self.assertFalse(self._ok(GROUND + 3.0, GROUND, 20.0))

    def test_without_ego_reference_the_ground_is_accepted(self):
        self.assertTrue(self._ok(GROUND + 5.0, None, 20.0))

    def test_missing_ground_is_rejected(self):
        self.assertFalse(self._ok(None, GROUND, 20.0))

    def test_dynamic_car_does_not_get_the_curb_fallback(self):
        # 动态：近处 2 m 有 0.25 m 突变 → 坡度 7° > 5° → 不接受；同样的差给静态车可接受
        box = [2.0, 0.0, 0.0, 4.6, 1.9, 1.6, 0.0]
        self.assertFalse(_car_height_ground_ok(GROUND + 0.25, GROUND, box, None, self.config,
                                              is_static=False))
        self.assertTrue(_car_height_ground_ok(GROUND + 0.25, GROUND, box, None, self.config,
                                             is_static=True))

    def test_dynamic_car_on_a_slope_is_accepted(self):
        # 动态：30 m 外差 1.5 m → 坡度 2.9° < 5° → 接受
        box = [30.0, 0.0, 0.0, 4.6, 1.9, 1.6, 0.0]
        self.assertTrue(_car_height_ground_ok(GROUND + 1.5, GROUND, box, None, self.config,
                                             is_static=False))


if __name__ == "__main__":
    unittest.main()
