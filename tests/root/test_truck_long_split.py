"""超长卡车拆两段（2026-10-17 用户口径：单框上限 17 m，26 m 拆成 17 + 9）。"""
from __future__ import annotations

import math
import unittest

from geometry.truck_postprocess import TruckPostConfig, split_long_trucks


def _frames(length: float, *, yaw: float = 0.0, dx_long: bool = True,
            track_id: int = 7, cls: str = "Truck"):
    box = ([0.0, 0.0, 1.5, length, 2.5, 3.0, yaw] if dx_long
           else [0.0, 0.0, 1.5, 2.5, length, 3.0, yaw])
    return [{"frame_id": "1", "detections": [{
        "track_id": track_id, "class_name": cls, "score": 0.9,
        "box_lidar": box}]}]


class SplitLongTruckTest(unittest.TestCase):
    def test_26m_becomes_17_plus_9(self) -> None:
        frames = _frames(26.0)
        diag = split_long_trucks(frames)
        self.assertEqual(diag["boxes_split"], 1)
        dets = frames[0]["detections"]
        self.assertEqual(len(dets), 2)
        front, rear = dets
        self.assertEqual(front["track_id"], 7)
        self.assertNotEqual(rear["track_id"], 7)
        self.assertAlmostEqual(front["box_lidar"][3], 17.0)
        self.assertAlmostEqual(rear["box_lidar"][3], 9.0)
        # 首尾相接：前段覆盖 [-13, 4]，后段覆盖 [4, 13]
        self.assertAlmostEqual(front["box_lidar"][0], -4.5)
        self.assertAlmostEqual(rear["box_lidar"][0], 8.5)
        # 宽 / 高 / z / yaw / 类别 / 分数继承
        for key in ("class_name", "score"):
            self.assertEqual(front[key], rear[key])
        self.assertAlmostEqual(front["box_lidar"][4], rear["box_lidar"][4])
        self.assertAlmostEqual(front["box_lidar"][2], rear["box_lidar"][2])
        self.assertEqual(rear["_truck_split_from"], 7)

    def test_22m_becomes_17_plus_5(self) -> None:
        frames = _frames(22.0)
        split_long_trucks(frames)
        front, rear = frames[0]["detections"]
        self.assertAlmostEqual(front["box_lidar"][3], 17.0)
        self.assertAlmostEqual(rear["box_lidar"][3], 5.0)

    def test_17m_is_untouched(self) -> None:
        frames = _frames(17.0)
        diag = split_long_trucks(frames)
        self.assertEqual((diag["boxes_split"], diag["boxes_clipped_only"]), (0, 0))
        self.assertEqual(len(frames[0]["detections"]), 1)

    def test_small_remainder_only_clips(self) -> None:
        frames = _frames(17.5)
        diag = split_long_trucks(frames)
        self.assertEqual((diag["boxes_split"], diag["boxes_clipped_only"]), (0, 1))
        dets = frames[0]["detections"]
        self.assertEqual(len(dets), 1)
        self.assertAlmostEqual(dets[0]["box_lidar"][3], 17.0)

    def test_long_axis_on_y(self) -> None:
        """dy 是长边（yaw=0 时局部 +y 即世界 +y）：拆完两段中心仍沿长轴分开 13 m。"""
        frames = _frames(26.0, yaw=0.0, dx_long=False)
        split_long_trucks(frames)
        front, rear = frames[0]["detections"]
        self.assertAlmostEqual(front["box_lidar"][4], 17.0)   # dy 是长边
        self.assertAlmostEqual(rear["box_lidar"][4], 9.0)
        dx = rear["box_lidar"][0] - front["box_lidar"][0]
        dy = rear["box_lidar"][1] - front["box_lidar"][1]
        self.assertAlmostEqual(math.hypot(dx, dy), 13.0, places=6)
        self.assertAlmostEqual(dy, 13.0, places=6)
        self.assertAlmostEqual(dx, 0.0, places=6)

    def test_new_ids_are_unique_across_clip(self) -> None:
        frames = _frames(26.0) + _frames(26.0, track_id=31)
        frames[1]["frame_id"] = "2"
        split_long_trucks(frames)
        ids = [d["track_id"] for f in frames for d in f["detections"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_tail_id_is_stable_across_frames(self) -> None:
        """同一条长车轨迹拆出来的"后段"整段用同一个 id，不能每帧换一个。"""
        frames = []
        for index in range(4):
            frames.append({"frame_id": str(index), "detections": [{
                "track_id": 7, "class_name": "Truck", "score": 0.9,
                "box_lidar": [float(index), 0.0, 1.5, 26.0, 2.5, 3.0, 0.0]}]})
        diag = split_long_trucks(frames)
        self.assertEqual(diag["boxes_split"], 4)
        self.assertEqual(diag["split_pairs"], {"7": 8})
        for frame in frames:
            ids = sorted(d["track_id"] for d in frame["detections"])
            self.assertEqual(ids, [7, 8])
            self.assertAlmostEqual(frame["detections"][1]["box_lidar"][3], 9.0)

    def test_non_vehicle_class_untouched(self) -> None:
        frames = _frames(26.0, cls="Car")
        diag = split_long_trucks(frames)
        self.assertEqual(diag["boxes_split"], 0)
        self.assertEqual(len(frames[0]["detections"]), 1)

    def test_disabled_switch(self) -> None:
        frames = _frames(26.0)
        diag = split_long_trucks(frames, TruckPostConfig(long_split_enabled=False))
        self.assertFalse(diag["enabled"])
        self.assertEqual(len(frames[0]["detections"]), 1)


if __name__ == "__main__":
    unittest.main()
