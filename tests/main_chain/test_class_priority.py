"""车与车之间的类别优先级：冲突一律按 Car（用户 2026-09-21 决定）。"""
from __future__ import annotations

import unittest

import numpy as np

from classification.class_refinement import (
    ClassRefinementConfig,
    _Item,
    _mixed_target,
)
from filtering.hard_filters import deduplicate_same_center
from tracking import tracker_conservative as tracking
from tracking.tracker_static_first import _representative_slot_class


def _item(cls, timestamp=0, size=(4.6, 1.9, 1.5)):
    det = {"class_name": cls, "box_lidar": [0.0, 0.0, 0.0, *size, 0.0],
           "score": 0.9}
    return _Item(frame_index=0, timestamp=timestamp, detection=det,
                 original_class=cls, world=np.zeros(3, dtype=np.float64),
                 size=np.asarray(size, dtype=np.float64))


class ClassPriorityTest(unittest.TestCase):
    def test_priority_order(self):
        self.assertLess(tracking.class_priority("car"),
                        tracking.class_priority("truck"))
        self.assertLess(tracking.class_priority("Vehicle"),
                        tracking.class_priority("Truck"))
        # 挂车 / 工程车都并进 Truck，优先级与 Truck 相同（Bus 之后）
        self.assertEqual(tracking.class_priority("trailer"),
                         tracking.class_priority("truck"))
        self.assertEqual(tracking.class_priority("construction_vehicle"),
                         tracking.class_priority("truck"))
        # 未登记类别排最后
        self.assertEqual(tracking.class_priority("barrier"),
                         len(tracking.CLASS_PRIORITY) + 1)

    def test_class_map_folds_trailer_and_construction_into_truck(self):
        for raw in ("trailer", "Trailer", "construction_vehicle",
                    "Engineering_vehicle"):
            self.assertEqual(tracking.canonical_class_name(raw), "Truck", raw)
        self.assertEqual(tracking.canonical_class_name("car"), "Car")

    def test_mixed_vehicle_track_becomes_car(self):
        config = ClassRefinementConfig()
        # 一台车被 car 头与 truck 头同时框住（truck 观测更多也还是 Car）
        self.assertEqual(
            _mixed_target([_item("Truck"), _item("Truck", 1), _item("Car", 2)],
                          config), "Car")
        # 小写模型名同样识别
        self.assertEqual(
            _mixed_target([_item("truck"), _item("car", 1)], config), "Car")
        # "Vehicle" 在本项目里归一到 Car（Waymo 的通用车类），所以仍是 Car
        self.assertEqual(
            _mixed_target([_item("Truck"), _item("Vehicle", 1)], config),
            "Car")
        # 混合里没有车类证据（只剩 VRU）时才走尺寸判据
        self.assertEqual(
            _mixed_target([_item("Pedestrian", 0, (0.8, 0.6, 1.7)),
                           _item("Cyclist", 1, (1.8, 0.7, 1.7))], config),
            "Cyclist")

    def test_slot_class_prefers_car(self):
        self.assertEqual(_representative_slot_class(["Truck", "Truck", "Car"]),
                         "Car")
        self.assertEqual(_representative_slot_class(["Truck", "Truck"]),
                         "Truck")
        self.assertEqual(_representative_slot_class([]), "")

    def test_same_center_dedup_keeps_car(self):
        frames = [{
            "frame_id": "1",
            "detections": [
                {"class_name": "Truck", "track_id": 1, "score": 0.99,
                 "box_lidar": [0.0, 0.0, 0.0, 8.0, 2.5, 3.0, 0.0]},
                {"class_name": "Car", "track_id": 2, "score": 0.5,
                 "box_lidar": [0.1, 0.1, 0.0, 4.6, 1.9, 1.5, 0.0]},
            ],
        }]
        diag = deduplicate_same_center(frames)
        kept = [det["class_name"] for det in frames[0]["detections"]]
        self.assertEqual(kept, ["Car"])
        self.assertEqual(diag["boxes_removed"], 1)


if __name__ == "__main__":
    unittest.main()
