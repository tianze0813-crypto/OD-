"""超长卡车拆两段（2026-10-17 用户口径）：>20 m 才拆，前段 17 m 按行进方向定车头。"""
from __future__ import annotations

import math
import unittest

import numpy as np

from geometry.truck_postprocess import TruckPostConfig, split_long_trucks


def _frames(length: float, *, yaw: float = 0.0, dx_long: bool = True,
            track_id: int = 7, cls: str = "Truck", shift: float = 0.0):
    box = ([shift, 0.0, 1.5, length, 2.5, 3.0, yaw] if dx_long
           else [shift, 0.0, 1.5, 2.5, length, 3.0, yaw])
    return {"frame_id": "1", "detections": [{
        "track_id": track_id, "class_name": cls, "score": 0.9,
        "box_lidar": box}]}


class _StubCoords:
    """世界系 = lidar 系（单位阵），只用于给"行进方向"提供外参。"""

    base_from_lidar_top = np.eye(4)

    def world_from_lidar(self, ts):
        return np.eye(4)

    def lidar_from_world(self, ts):
        return np.eye(4)


class SplitLongTruckTest(unittest.TestCase):
    def test_26m_becomes_17_plus_9(self) -> None:
        frame = _frames(26.0)
        diag = split_long_trucks([frame])
        self.assertEqual(diag["boxes_split"], 1)
        front, rear = frame["detections"]
        self.assertEqual(front["track_id"], 7)
        self.assertNotEqual(rear["track_id"], 7)
        self.assertAlmostEqual(front["box_lidar"][3], 17.0)
        self.assertAlmostEqual(rear["box_lidar"][3], 9.0)
        # 没有运动信息 -> 退回"框长轴正方向那头算车头"：前段占 [+4, +13]
        self.assertAlmostEqual(front["box_lidar"][0], 4.5)
        self.assertAlmostEqual(rear["box_lidar"][0], -8.5)
        for key in ("class_name", "score"):
            self.assertEqual(front[key], rear[key])
        self.assertAlmostEqual(front["box_lidar"][4], rear["box_lidar"][4])
        self.assertEqual(rear["_truck_split_from"], 7)

    def test_21m_becomes_17_plus_4(self) -> None:
        frame = _frames(21.0)
        split_long_trucks([frame])
        front, rear = frame["detections"]
        self.assertAlmostEqual(front["box_lidar"][3], 17.0)
        self.assertAlmostEqual(rear["box_lidar"][3], 4.0)

    def test_20m_and_below_untouched(self) -> None:
        for length in (12.0, 17.0, 17.004, 18.5, 20.0):
            frame = _frames(length)
            diag = split_long_trucks([frame])
            self.assertEqual(diag["boxes_split"], 0, length)
            self.assertEqual(diag["boxes_skipped_short_remainder"], 0, length)
            self.assertEqual(len(frame["detections"]), 1, length)
            # 长度原样保留（17.004 也不裁）
            self.assertAlmostEqual(frame["detections"][0]["box_lidar"][3], length)

    def test_long_axis_on_y(self) -> None:
        frame = _frames(26.0, yaw=0.0, dx_long=False)
        split_long_trucks([frame])
        front, rear = frame["detections"]
        self.assertAlmostEqual(front["box_lidar"][4], 17.0)
        self.assertAlmostEqual(rear["box_lidar"][4], 9.0)
        self.assertAlmostEqual(front["box_lidar"][1] - rear["box_lidar"][1], 13.0)

    def test_head_is_chosen_by_travel_direction(self) -> None:
        """框 yaw 指向 -x（长轴 mod 180° 分不出车头），但车实际在往 +x 开：
        17 m 前段必须落在 +x 那一端。"""
        frames = []
        for index in range(4):                     # 车沿 +x 走 1 m/帧
            frames.append(_frames(22.0, yaw=math.pi, shift=float(index)))
            frames[-1]["frame_id"] = str(index)
        diag = split_long_trucks(frames, _StubCoords())
        self.assertEqual(diag["boxes_split"], 4)
        self.assertEqual(diag["examples"][0]["head_source"], "travel")
        front, rear = frames[0]["detections"]
        self.assertAlmostEqual(front["box_lidar"][3], 17.0)
        # 车头在 +x：前段覆盖 [-6, +11]（中心 +2.5），后段覆盖 [-11, -6]（中心 -8.5）
        self.assertAlmostEqual(front["box_lidar"][0], 2.5)
        self.assertAlmostEqual(rear["box_lidar"][0], -8.5)
        self.assertGreater(front["box_lidar"][0], rear["box_lidar"][0])

    def test_head_follows_reverse_motion(self) -> None:
        """车往 -x 开（框 yaw=0）：前段应落在 -x 那一端。"""
        frames = []
        for index in range(4):
            frames.append(_frames(22.0, yaw=0.0, shift=float(-index)))
            frames[-1]["frame_id"] = str(index)
        split_long_trucks(frames, _StubCoords())
        front, rear = frames[0]["detections"]
        self.assertAlmostEqual(front["box_lidar"][3], 17.0)
        self.assertLess(front["box_lidar"][0], rear["box_lidar"][0])

    def test_tail_id_is_stable_across_frames(self) -> None:
        frames = []
        for index in range(4):
            frames.append(_frames(26.0, shift=float(index)))
            frames[-1]["frame_id"] = str(index)
        diag = split_long_trucks(frames)
        self.assertEqual(diag["boxes_split"], 4)
        for frame in frames:
            self.assertEqual(sorted(d["track_id"] for d in frame["detections"]), [7, 8])

    def test_new_ids_are_unique_across_clip(self) -> None:
        frames = [_frames(26.0), _frames(26.0, track_id=31)]
        frames[1]["frame_id"] = "2"
        split_long_trucks(frames)
        ids = [d["track_id"] for f in frames for d in f["detections"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_non_vehicle_class_untouched(self) -> None:
        frame = _frames(26.0, cls="Car")
        self.assertEqual(split_long_trucks([frame])["boxes_split"], 0)
        self.assertEqual(len(frame["detections"]), 1)

    def test_disabled_switch(self) -> None:
        frame = _frames(26.0)
        diag = split_long_trucks([frame], None, TruckPostConfig(long_split_enabled=False))
        self.assertFalse(diag["enabled"])
        self.assertEqual(len(frame["detections"]), 1)


if __name__ == "__main__":
    unittest.main()
