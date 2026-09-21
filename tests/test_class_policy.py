"""类别策略（2026-09-21 用户决定）：挂车/工程车并进 Truck、Car 优先、停用覆盖规则。"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from pipeline.hybrid_expD_truck import DEFAULTS as TRUCK_DEFAULTS
from tracking import tracker_conservative as tracking

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_hybrid_prelabel.py"
_SPEC = importlib.util.spec_from_file_location("hybrid_launcher", _SCRIPT)
hybrid_launcher = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(hybrid_launcher)


def _label(obj_id, obj_type, box):
    x, y, z, dx, dy, dz, yaw = box
    return {
        "obj_id": str(obj_id), "obj_type": obj_type, "score": 0.9,
        "psr": {
            "position": {"x": x, "y": y, "z": z},
            "rotation": {"x": 0.0, "y": 0.0, "z": yaw},
            "scale": {"x": dx, "y": dy, "z": dz},
        },
    }


class ClassPolicyTest(unittest.TestCase):
    def test_construction_vehicle_is_truck(self):
        self.assertEqual(tracking.canonical_class_name("construction_vehicle"),
                         "Truck")
        self.assertEqual(tracking.canonical_class_name("Engineering_vehicle"),
                         "Truck")

    def test_trailer_policy_defaults_to_to_truck(self):
        # 标注侧没有 Trailer 类别 -> 一律并成 Truck
        self.assertEqual(TRUCK_DEFAULTS["trailer_policy"], "to-truck")

    def test_car_covered_by_truck_is_not_dropped(self):
        car = _label(1, "Car", (0.0, 0.0, 0.0, 4.6, 1.9, 1.5, 0.0))
        truck = _label(1001, "Truck", (0.0, 0.0, 0.0, 12.0, 2.5, 3.2, 0.0))
        merged, diag = hybrid_launcher._merge_chain_labels(
            {"car": {"100": [car]}, "truck": {"100": [truck]}},
            ["car", "truck"], car_truck_cover_threshold=0.5)
        types = sorted(item["obj_type"] for item in merged[0]["labels"])
        self.assertEqual(types, ["Car", "Truck"])          # Car 不再被删
        self.assertFalse(diag["car_truck_overlap"]["enabled"])


if __name__ == "__main__":
    unittest.main()
