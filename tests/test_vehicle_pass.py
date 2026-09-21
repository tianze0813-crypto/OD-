"""车链合并后处理的纯函数单测（不需要检测器/GPU）。

覆盖：类别白名单过滤、单类别视图切分、Truck 几何还原（只借 id）。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.vehicle_pass import (_filter_classes, _select_class,
                                   restore_geometry)


def _det(cls, tid, box, score=0.9):
    return {"class_name": cls, "track_id": tid, "score": score,
            "box_lidar": list(box)}


class VehiclePassTest(unittest.TestCase):
    def test_class_filter_keeps_raw_names(self):
        frames = [{"frame_id": "1", "detections": [
            _det("car", 1, (0, 0, 0, 4.6, 1.9, 1.5, 0)),
            _det("truck", 2, (20, 0, 0, 9, 2.5, 3.2, 0)),
            _det("pedestrian", 3, (1, 1, 0, 0.8, 0.6, 1.7, 0)),
            _det("bicycle", 4, (2, 2, 0, 1.8, 0.7, 1.7, 0)),
        ]}]
        stats = _filter_classes(frames, ("Car", "Truck"))
        self.assertEqual(stats["detections_after"], 2)
        # 原始类名字符串保留（Truck 分支的挂车规则还要按原名处理）
        self.assertEqual([d["class_name"] for d in frames[0]["detections"]],
                         ["car", "truck"])

    def test_select_class_uses_canonical_names(self):
        frames = [{"frame_id": "1", "num_points": 3, "detections": [
            _det("car", 1, (0, 0, 0, 4, 2, 1.5, 0)),
            _det("truck", 2, (20, 0, 0, 9, 2.5, 3.2, 0)),
            _det("Trailer", 3, (40, 0, 0, 9, 2.5, 3.2, 0)),
        ]}]
        # Truck 视图必须同时收 Trailer（根目录 tracking 里 Trailer 归一后还是 Trailer）
        truck_view = _select_class(frames, ("Truck", "Trailer"))
        self.assertEqual([d["track_id"] for d in truck_view[0]["detections"]],
                         [2, 3])
        only_truck = _select_class(frames, "Truck")
        self.assertEqual([d["track_id"] for d in only_truck[0]["detections"]], [2])
        self.assertEqual(truck_view[0]["num_points"], 3)   # 其它字段保留
        car_view = _select_class(frames, "Car")
        self.assertEqual([d["track_id"] for d in car_view[0]["detections"]], [1])

    def test_restore_geometry_only_touches_non_car(self):
        raw = [{"frame_id": "1", "detections": [
            _det("car", None, (0.0, 0.0, 0.0, 4.6, 1.9, 1.5, 0.0)),
            _det("truck", None, (20.0, 0.0, 0.0, 9.0, 2.5, 3.2, 0.10)),
        ]}]
        tracked = [{"frame_id": "1", "detections": [
            # Car：main_chain 精修后的几何（不能被还原）
            _det("Car", 1, (0.3, 0.2, 0.05, 4.4, 1.8, 1.4, 0.30)),
            # Truck：step4.5 里被轿车 box fit 动过（要还原成原值）
            _det("Truck", 2, (20.4, 0.1, 0.0, 8.4, 2.4, 3.0, -0.20)),
        ]}]
        with tempfile.TemporaryDirectory() as tmp:
            raw_json = Path(tmp) / "raw.json"
            out_json = Path(tmp) / "tracked.json"
            raw_json.write_text(json.dumps(raw), encoding="utf-8")
            out_json.write_text(json.dumps(tracked), encoding="utf-8")
            report = restore_geometry(out_json, raw_json)
            result = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(report, {"restored": 1, "unmatched": 0})
        car_box = result[0]["detections"][0]["box_lidar"]
        truck_box = result[0]["detections"][1]["box_lidar"]
        self.assertAlmostEqual(car_box[0], 0.3)          # Car 保持精修结果
        self.assertAlmostEqual(truck_box[0], 20.0)       # Truck 还原成原值
        self.assertAlmostEqual(truck_box[6], 0.10)
        self.assertEqual(result[0]["detections"][1]["track_id"], 2)   # id 保留


if __name__ == "__main__":
    unittest.main()
