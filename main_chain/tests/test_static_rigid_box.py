"""Static rigid Car box: stacked frames -> one box fixed in the world frame."""
import copy
import math
import unittest

import numpy as np

from geometry.car_box_fit import (
    CarBoxFitConfig,
    _apply_static_rigid_boxes,
    _points_inside_box,
    _static_rigid_motion,
    _static_rigid_roof,
)


def car_body(centre, size=(4.4, 1.7, 1.5), spacing=0.12):
    """A filled box cloud, dense enough to look like a car body."""
    nx = int(size[0] / spacing) + 1
    ny = int(size[1] / spacing) + 1
    nz = int(size[2] / spacing) + 1
    xs = np.linspace(-size[0] / 2, size[0] / 2, max(nx, 3))
    ys = np.linspace(-size[1] / 2, size[1] / 2, max(ny, 3))
    zs = np.linspace(0.06, size[2], max(nz, 3))
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    offset = np.asarray([centre[0], centre[1], 0.0], dtype=np.float64)
    points = grid.reshape(-1, 3) + offset
    ground = np.asarray([[x, y, 0.0]
                         for x in np.linspace(-8, 8, 40)
                         for y in np.linspace(-6, 6, 40)], dtype=np.float64)
    return np.vstack((points, ground))


def item(frame_id, box, points, world_from_lidar, ground_z=0.0):
    world = np.asarray(world_from_lidar, dtype=np.float64)
    return {
        "frame_id": str(frame_id),
        "timestamp": int(frame_id),
        "det": {"class_name": "Car",
                "box_lidar": [float(value) for value in box]},
        "world_from_lidar": world,
        "lidar_from_world": np.linalg.inv(world),
        "raw_world": (world @ np.array([box[0], box[1], box[2], 1.0]))[:3],
        "points": points,
        "ground_z": ground_z,
    }


def rotating(degrees):
    angle = math.radians(degrees)
    matrix = np.eye(4)
    matrix[:2, :2] = [[math.cos(angle), -math.sin(angle)],
                      [math.sin(angle), math.cos(angle)]]
    return matrix


def translate(x, y):
    matrix = np.eye(4)
    matrix[0, 3], matrix[1, 3] = x, y
    return matrix


class StaticRigidMotionTest(unittest.TestCase):
    def test_parked_track_is_static_and_moving_track_is_not(self):
        box = [10.0, 0.0, 0.8, 4.6, 1.9, 1.6, 0.0]
        parked = [item(1 + index, box, car_body((10.0, 0.0)), np.eye(4))
                  for index in range(4)]
        moving = [
            item(1 + index,
                 [10.0 + 3.0 * index, 0.0, 0.8, 4.6, 1.9, 1.6, 0.0],
                 car_body((10.0 + 3.0 * index, 0.0)), np.eye(4))
            for index in range(4)]

        net, span, step = _static_rigid_motion(parked)
        self.assertLess(max(net, span, step), 1.0)
        net, span, step = _static_rigid_motion(moving)
        self.assertGreaterEqual(max(net, span, step), 1.0)


class StaticRigidRoofTest(unittest.TestCase):
    config = CarBoxFitConfig()

    def test_supported_roof_is_returned(self):
        stack = np.asarray([[x, y, z]
                            for x in np.linspace(-2.2, 2.2, 24)
                            for y in np.linspace(-0.85, 0.85, 10)
                            for z in (0.20, 0.90, 1.50)], dtype=np.float64)
        roof, detail = _static_rigid_roof(stack, self.config)
        self.assertIsNotNone(roof)
        self.assertAlmostEqual(float(roof), 1.50, places=3)
        self.assertGreaterEqual(detail["count"], detail["need"])

    def test_sparse_top_layer_is_rejected(self):
        stack = np.asarray([[x, y, z]
                            for x in np.linspace(-2.2, 2.2, 24)
                            for y in np.linspace(-0.85, 0.85, 10)
                            for z in (0.20, 0.90)], dtype=np.float64)
        stack = np.vstack((stack, np.asarray([[0.0, 0.0, 1.60],
                                              [0.5, 0.2, 1.61]])))
        roof, detail = _static_rigid_roof(stack, self.config)
        self.assertIsNone(roof)
        self.assertEqual(detail["reason"], "top_not_supported")


class StaticRigidApplyTest(unittest.TestCase):
    """A car parked at a fixed world position, seen from a moving ego."""

    world_centre = (10.0, 0.0)
    size = (4.6, 1.9, 1.6)
    yaw = 0.35

    def parked(self, worlds):
        items = []
        for index, world in enumerate(worlds):
            lidar_from_world = np.linalg.inv(world)
            point = lidar_from_world @ np.array([self.world_centre[0],
                                                 self.world_centre[1], 0.0, 1.0])
            box = [float(point[0]), float(point[1]), 0.8,
                   self.size[0], self.size[1], self.size[2], self.yaw]
            items.append(item(1000 + index, box,
                              car_body((float(point[0]), float(point[1]))),
                              world))
        return items

    def test_static_track_gets_one_rigid_box(self):
        worlds = [translate(0.0, 0.0), translate(1.5, 0.4),
                  translate(3.0, 0.9), translate(4.6, 1.1)]
        tracks = {7: self.parked(worlds)}
        result = _apply_static_rigid_boxes(
            tracks, CarBoxFitConfig(static_rigid_enabled=True))

        self.assertEqual(result["stats"]["tracks"], 1)
        self.assertEqual(result["stats"]["boxes"], 4)
        sizes = {tuple(round(float(value), 6)
                       for value in track["det"]["box_lidar"][3:6])
                 for track in tracks[7]}
        self.assertEqual(len(sizes), 1, "size must be identical in every frame")
        centre_world = []
        for track in tracks[7]:
            world = track["world_from_lidar"]
            box = track["det"]["box_lidar"]
            centre_world.append((world @ np.array(
                [box[0], box[1], box[2], 1.0]))[:2])
        centre_world = np.asarray(centre_world)
        self.assertLess(float(np.max(np.linalg.norm(
            centre_world - centre_world[0], axis=1))), 1e-6)
        self.assertLess(float(np.linalg.norm(
            centre_world[0] - np.asarray(self.world_centre))), 0.35)
        for track in tracks[7]:                      # yaw is a protected field
            self.assertEqual(float(track["det"]["box_lidar"][6]), self.yaw)
        for track in tracks[7]:                      # box sits on that frame's ground
            self.assertAlmostEqual(float(track["det"]["box_lidar"][2])
                                   - float(track["det"]["box_lidar"][5]) / 2.0,
                                   0.0, places=6)

    def test_moving_track_is_left_alone(self):
        items = []
        for index in range(4):
            box = [10.0 + 3.0 * index, 0.0, 0.8,
                   self.size[0], self.size[1], self.size[2], self.yaw]
            items.append(item(1000 + index, box,
                              car_body((10.0 + 3.0 * index, 0.0)), np.eye(4)))
        tracks = {7: items}
        before = copy.deepcopy([track["det"]["box_lidar"] for track in items])
        result = _apply_static_rigid_boxes(
            tracks, CarBoxFitConfig(static_rigid_enabled=True))

        self.assertEqual(result["stats"].get("tracks", 0), 0)
        self.assertEqual(result["stats"]["skipped_moving"], 1)
        self.assertEqual(before, [track["det"]["box_lidar"] for track in items])

    def test_short_track_is_skipped(self):
        tracks = {7: self.parked([np.eye(4)])}
        result = _apply_static_rigid_boxes(
            tracks, CarBoxFitConfig(static_rigid_enabled=True))
        self.assertEqual(result["stats"].get("tracks", 0), 0)
        self.assertEqual(result["stats"]["skipped_short"], 1)


class WorldFrameOffsetTest(unittest.TestCase):
    """Regression: a world frame far from the origin must round-trip exactly.

    The real clips have a world pose around z=18.5 m / y=-119 m with a ~1 deg
    tilt, so dropping the world z when projecting back into the lidar frame
    shifted whole tracks by several metres.
    """

    def world(self, x, y):
        """Roughly the real chain: 90 deg mount yaw + a slope, origin at z=18.5."""
        tilt = math.radians(8.0)
        pitch = np.eye(4)
        pitch[1:3, 1:3] = [[math.cos(tilt), -math.sin(tilt)],
                           [math.sin(tilt), math.cos(tilt)]]
        yaw = np.eye(4)
        yaw[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
        matrix = pitch @ yaw
        matrix[:3, 3] = [x, y, 18.5]
        return matrix

    def test_track_stays_where_it_was(self):
        worlds = [self.world(0.0, 0.0), self.world(1.4, 0.3),
                  self.world(2.9, 0.7), self.world(4.3, 1.0)]
        items = []
        for index, world in enumerate(worlds):
            lidar_from_world = np.linalg.inv(world)
            point = (lidar_from_world @ np.array([0.45, -119.17, 18.55, 1.0]))[:2]
            box = [float(point[0]), float(point[1]), 0.8, 4.6, 1.9, 1.6, 0.0]
            items.append(item(1000 + index, box,
                              car_body((float(point[0]), float(point[1]))),
                              world))
        before = [list(track["det"]["box_lidar"][:2]) for track in items]
        _apply_static_rigid_boxes({7: items},
                                  CarBoxFitConfig(static_rigid_enabled=True))
        for original, track in zip(before, items):
            box = track["det"]["box_lidar"]
            self.assertLess(float(np.linalg.norm(
                np.asarray(original) - np.asarray(box[:2]))), 0.30,
                "the rigid centre must stay on the object, not metres away")


class StaticRigidGuardTest(unittest.TestCase):
    """The rigid box must never leave a frame with fewer points than before."""

    def parked(self, frames=4):
        items = []
        for index in range(frames):
            box = [10.0 + index * 0.05, 0.0, 0.8, 4.6, 1.9, 1.6, 0.0]
            items.append(item(1000 + index, box,
                              car_body((10.0 + index * 0.05, 0.0)), np.eye(4)))
        return items

    def test_points_inside_box_matches_step5_rule(self):
        points = np.asarray([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0],
                             [0.0, 0.9, 0.0], [0.0, 0.0, 0.9],
                             [1.2, 0.0, 0.0]], dtype=np.float64)
        box = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]
        self.assertEqual(_points_inside_box(points, box), 4)

    def test_frame_restored_when_rigid_box_would_be_empty(self):
        items = self.parked()
        before = [list(track["det"]["box_lidar"][:6]) for track in items]
        result = _apply_static_rigid_boxes(
            {7: items},
            CarBoxFitConfig(static_rigid_enabled=True,
                            static_rigid_min_points_in_box=10 ** 6))
        self.assertEqual(result["stats"]["restored_frames"], len(items))
        self.assertEqual(before, [list(track["det"]["box_lidar"][:6])
                                  for track in items])

    def test_thin_stack_is_skipped(self):
        """A track whose stacked body is shorter than a car is left untouched."""
        items = []
        for index in range(4):
            box = [10.0, 0.0, 0.8, 4.6, 1.9, 1.6, 0.0]
            items.append(item(1000 + index, box, car_body((0.0, 0.0)),
                              np.eye(4)))
        for track in items:                      # only a 1 m patch of body points
            track["points"] = car_body((10.6, 0.0), size=(0.8, 1.7, 1.5))
        before = [list(track["det"]["box_lidar"][:6]) for track in items]
        result = _apply_static_rigid_boxes(
            {7: items}, CarBoxFitConfig(static_rigid_enabled=True))
        self.assertEqual(result["stats"].get("tracks", 0), 0)
        self.assertEqual(result["stats"]["skipped_thin_stack"], 1)
        self.assertEqual(before, [list(track["det"]["box_lidar"][:6])
                                  for track in items])

    def test_roof_height_is_capped_by_the_fitted_height(self):
        items = self.parked()
        for track in items:                      # a pole far above the car body
            track["points"] = np.vstack((track["points"],
                                         np.asarray([[10.0, 0.0, 2.55]] * 40)))
        result = _apply_static_rigid_boxes(
            {7: items}, CarBoxFitConfig(static_rigid_enabled=True))
        detail = result["details"][0]
        self.assertLessEqual(float(detail["height"]),
                             1.6 + CarBoxFitConfig().static_rigid_height_max_gain + 1e-9)


class CarBoxFitSwitchTest(unittest.TestCase):
    def test_defaults_keep_both_switches_off(self):
        config = CarBoxFitConfig()
        self.assertTrue(config.enabled)
        self.assertFalse(config.static_rigid_enabled)
        self.assertGreater(config.static_rigid_min_points_in_box, 5)


if __name__ == "__main__":
    unittest.main()
