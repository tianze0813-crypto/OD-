"""Step 5A（静态 slot Car 内部空洞补帧）用例。

覆盖：正常补帧 / >=6 与 ==5 的边界 / 非静态区域与类别排除 / 洞两侧邻居不一致 /
同帧已有车框压住 / 缺 lidar bin / 不补端点外 / 单洞长度旋钮 / 关闭开关。
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from pipeline.step5a_slot_gap_fill import Step5aConfig, run


def _det(track_id, x, y=0.0, *, class_name="Car", region="static",
         retracked=False, length=4.5, width=2.0, height=1.6, yaw=0.0,
         visibility=None):
    detection = {
        "class_name": class_name,
        "score": 0.9,
        "track_id": track_id,
        "box_lidar": [float(x), float(y), 0.0, float(length), float(width),
                      float(height), float(yaw)],
        "region": region,
    }
    if retracked:
        detection["_step45_retracked"] = True
    if visibility is not None:
        detection["visibility"] = dict(visibility)
    return detection


def _frames(rows):
    return [{
        "frame_id": str(index),
        "num_points": 0,
        "num_detections": len(items),
        "detections": items,
    } for index, items in enumerate(rows)]


def _points_inside(count=12, x=0.0, y=0.0, z=0.0):
    """框内点（默认 box 在原点、4.5x2.0x1.6）。"""
    return [[x + 0.05 * index, y, z, 1.0] for index in range(count)]


class Step5aSlotGapFillTest(unittest.TestCase):
    def _setup(self, root: Path, rows, points_by_frame, *,
               slots=None, yaw_slots=None):
        transforms = root / "transforms"
        lidar = root / "lidar" / "lidar_top"
        transforms.mkdir(parents=True)
        lidar.mkdir(parents=True)
        (transforms / "calib.json").write_text(json.dumps({
            "tf2base_link": {
                "pose": np.eye(4).tolist(),
                "lidar_top": np.eye(4).tolist(),
            }
        }), encoding="utf-8")
        (transforms / "pose_data.txt").write_text(
            "\n".join(f"{index},0,0,0,0,0,0,1"
                      for index in range(len(rows))) + "\n",
            encoding="utf-8")
        frames = _frames(rows)
        for frame in frames:
            values = points_by_frame.get(frame["frame_id"])
            if values is None:
                continue
            np.asarray(values, dtype=np.float32).tofile(
                lidar / f"{frame['frame_id']}.bin")
        step45_json = root / "clip_step45.json"
        step45_json.write_text(json.dumps(frames), encoding="utf-8")
        step2_diagnostics = root / "clip_step2_diagnostics.json"
        step2_diagnostics.write_text(json.dumps({
            "tracking": {"slot_details": list(slots or [])},
            "static_yaw_stabilization": {"slots": list(yaw_slots or [])},
        }), encoding="utf-8")
        return step45_json, step2_diagnostics, frames

    def _run(self, root: Path, rows, points_by_frame, *, slots, yaw_slots,
             config=None):
        step45_json, step2_diagnostics, source = self._setup(
            root, rows, points_by_frame, slots=slots, yaw_slots=yaw_slots)
        out_json = root / "clip_step5a.json"
        out_diag = root / "clip_step5a_diagnostics.json"
        diagnostics = run(step45_json, root, step2_diagnostics,
                          out_json, out_diag, config or Step5aConfig())
        return diagnostics, json.loads(out_json.read_text(encoding="utf-8")), source

    # ------------------------------------------------------------------ #
    def test_fills_interior_hole_with_stacked_box(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            visibility = {"tag": 1, "ratio": 0.5}
            rows = [
                [_det(1, 0.0, visibility=visibility)],
                [_det(1, 0.0, visibility=visibility)],
                [],                                   # 洞（第 2 帧）
                [_det(1, 0.0, visibility=visibility)],
                [_det(1, 0.0, visibility=visibility)],
            ]
            points = {str(index): _points_inside() for index in range(5)}
            diagnostics, output, source = self._run(
                root, rows, points,
                slots=[{"track_id": 1, "class_name": "Car"}],
                yaw_slots=[{"track_id": 1, "target_world_yaw": 0.0,
                            "direction_flip": False}])

        self.assertEqual(diagnostics["inserted_detections"], 1)
        self.assertEqual(diagnostics["append_only_check"]["passed"], True)
        self.assertEqual(diagnostics["before_detections"], 4)
        self.assertEqual(diagnostics["after_detections"], 5)
        filled = [det for det in output[2]["detections"]
                  if det.get("_step5a_filled")]
        self.assertEqual(len(filled), 1)
        detection = filled[0]
        self.assertEqual(detection["track_id"], 1)
        self.assertEqual(detection["class_name"], "Car")
        self.assertEqual(detection["region"], "static")
        self.assertEqual(detection["score"], 0.0)
        self.assertEqual(detection["visibility"], visibility)
        box = detection["box_lidar"]
        self.assertAlmostEqual(box[0], 0.0, places=6)
        self.assertAlmostEqual(box[3], 4.5, places=6)
        self.assertAlmostEqual(box[5], 1.6, places=6)
        self.assertEqual(output[2]["num_detections"], 1)
        # 已有检测一字不改
        for index, raw_frame in enumerate(source):
            self.assertEqual(output[index]["detections"][:len(
                raw_frame["detections"])], raw_frame["detections"])

    def test_point_threshold_boundary_is_six(self):
        """>=6 点才补；==5 点会被跳过（与 step5 的 count<=5 删除口径对齐）。"""
        for count, expected in ((6, 1), (5, 0)):
            with self.subTest(points=count):
                with TemporaryDirectory() as directory:
                    root = Path(directory)
                    rows = [
                        [_det(1, 0.0)],
                        [_det(1, 0.0)],
                        [],
                        [_det(1, 0.0)],
                    ]
                    points = {str(index): _points_inside() for index in range(4)}
                    points["2"] = _points_inside(count=count)
                    diagnostics, output, _source = self._run(
                        root, rows, points,
                        slots=[{"track_id": 1, "class_name": "Car"}],
                        yaw_slots=[{"track_id": 1, "target_world_yaw": 0.0}])
                self.assertEqual(diagnostics["inserted_detections"], expected)
                self.assertEqual(
                    diagnostics["skip_reason_counts"].get("too_few_points", 0),
                    1 - expected)
                self.assertEqual(len(output[2]["detections"]), expected)

    def test_skips_dynamic_region_retracked_and_wrong_class(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                [_det(1, 0.0, region="dynamic"), _det(2, 0.0),
                 _det(3, 0.0, retracked=True), _det(4, 0.0, class_name="Truck")],
                [_det(1, 0.0, region="dynamic"), _det(2, 0.0),
                 _det(3, 0.0, retracked=True), _det(4, 0.0, class_name="Truck")],
                [],
                [_det(1, 0.0, region="dynamic"), _det(2, 0.0),
                 _det(3, 0.0, retracked=True), _det(4, 0.0, class_name="Truck")],
            ]
            points = {str(index): _points_inside() for index in range(4)}
            diagnostics, _output, _source = self._run(
                root, rows, points,
                slots=[{"track_id": index, "class_name": "Car"}
                       for index in (1, 2, 3, 4)],
                yaw_slots=[{"track_id": index, "target_world_yaw": 0.0}
                           for index in (1, 2, 3, 4)])

        self.assertEqual(diagnostics["inserted_detections"], 1)
        reasons = {entry["track_id"]: entry["reason"]
                   for entry in diagnostics["skipped_tracks"]}
        self.assertEqual(reasons[1], "outside_static_region")
        self.assertEqual(reasons[3], "outside_static_region")
        self.assertEqual(reasons[4], "not_pure_car")
        # track 2（干净的静态 Car）会补
        self.assertEqual(
            len([fill for fill in diagnostics["fills"]
                 if fill["track_id"] == 2]), 1)

    def test_truck_slot_is_not_a_candidate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [[_det(1, 0.0)], [_det(1, 0.0)], [], [_det(1, 0.0)]]
            points = {str(index): _points_inside() for index in range(4)}
            diagnostics, _output, _source = self._run(
                root, rows, points,
                slots=[{"track_id": 1, "class_name": "Truck"}],
                yaw_slots=[{"track_id": 1, "target_world_yaw": 0.0}])
        self.assertEqual(diagnostics["candidate_slots"], 0)
        self.assertEqual(diagnostics["inserted_detections"], 0)

    def test_skips_when_bracketing_neighbours_moved(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                [_det(1, 0.0)],
                [_det(1, 0.0)],
                [],
                [_det(1, 6.0)],          # 右侧邻居在世界系里跑了 6 m
            ]
            points = {str(index): _points_inside() for index in range(4)}
            points["3"] = _points_inside(x=6.0)
            diagnostics, output, _source = self._run(
                root, rows, points,
                slots=[{"track_id": 1, "class_name": "Car"}],
                yaw_slots=[{"track_id": 1, "target_world_yaw": 0.0}])
        self.assertEqual(diagnostics["inserted_detections"], 0)
        self.assertEqual(
            diagnostics["skipped_frames"][0]["reason"], "neighbour_moved")
        self.assertEqual(len(output[2]["detections"]), 0)

    def test_skips_when_same_frame_vehicle_overlaps_patch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                [_det(1, 0.0)],
                [_det(1, 0.0)],
                [_det(9, 0.1)],          # 洞帧里已经有另一辆车压在同一位置
                [_det(1, 0.0)],
            ]
            points = {str(index): _points_inside() for index in range(4)}
            diagnostics, _output, _source = self._run(
                root, rows, points,
                slots=[{"track_id": 1, "class_name": "Car"}],
                yaw_slots=[{"track_id": 1, "target_world_yaw": 0.0}])
        self.assertEqual(diagnostics["inserted_detections"], 0)
        self.assertEqual(diagnostics["skipped_frames"][0]["reason"],
                         "overlap_with_existing_vehicle")
        self.assertEqual(diagnostics["skipped_frames"][0]["overlap_track_id"], 9)

    def test_missing_lidar_bin_is_skipped(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [[_det(1, 0.0)], [_det(1, 0.0)], [], [_det(1, 0.0)]]
            points = {str(index): _points_inside() for index in range(4)}
            points.pop("2")              # 洞帧没有点云文件
            diagnostics, _output, _source = self._run(
                root, rows, points,
                slots=[{"track_id": 1, "class_name": "Car"}],
                yaw_slots=[{"track_id": 1, "target_world_yaw": 0.0}])
        self.assertEqual(diagnostics["inserted_detections"], 0)
        self.assertEqual(diagnostics["skipped_frames"][0]["reason"],
                         "no_lidar_frame")

    def test_does_not_extend_beyond_the_observed_span(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                [],                      # 端点外：不补
                [_det(1, 0.0)],
                [_det(1, 0.0)],
                [_det(1, 0.0)],
                [],                      # 端点外：不补
            ]
            points = {str(index): _points_inside() for index in range(5)}
            diagnostics, output, _source = self._run(
                root, rows, points,
                slots=[{"track_id": 1, "class_name": "Car"}],
                yaw_slots=[{"track_id": 1, "target_world_yaw": 0.0}])
        self.assertEqual(diagnostics["inserted_detections"], 0)
        self.assertEqual(diagnostics["tracks"][0]["hole_frames"], [])
        self.assertEqual(len(output[0]["detections"]), 0)
        self.assertEqual(len(output[4]["detections"]), 0)

    def test_max_hole_frames_knob_skips_long_holes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                [_det(1, 0.0)],
                [_det(1, 0.0)],
                [],
                [],
                [],
                [_det(1, 0.0)],
            ]
            points = {str(index): _points_inside() for index in range(6)}
            diagnostics, _output, _source = self._run(
                root, rows, points,
                slots=[{"track_id": 1, "class_name": "Car"}],
                yaw_slots=[{"track_id": 1, "target_world_yaw": 0.0}],
                config=Step5aConfig(max_hole_frames=1))
        self.assertEqual(diagnostics["inserted_detections"], 0)
        self.assertEqual(diagnostics["tracks"][0]["hole_frames"], [2, 3, 4])
        self.assertEqual(diagnostics["tracks"][0]["hole_runs"], [[2, 4]])

    def test_uses_slot_axis_and_direction_flip(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [[_det(1, 0.0)], [_det(1, 0.0)], [], [_det(1, 0.0)]]
            points = {str(index): _points_inside() for index in range(4)}
            diagnostics, output, _source = self._run(
                root, rows, points,
                slots=[{"track_id": 1, "class_name": "Car"}],
                yaw_slots=[{"track_id": 1, "target_world_yaw": 0.4,
                            "direction_flip": True}])
        self.assertEqual(diagnostics["inserted_detections"], 1)
        entry = diagnostics["tracks"][0]
        self.assertEqual(entry["yaw_source"], "slot_target_world_yaw")
        self.assertTrue(entry["yaw_flipped"])
        expected = (0.4 + np.pi + np.pi) % (2.0 * np.pi) - np.pi
        self.assertAlmostEqual(entry["world_yaw"], expected, places=6)
        filled = [det for det in output[2]["detections"]
                  if det.get("_step5a_filled")][0]
        # 世界系 yaw -> lidar 系（这里是恒等变换）后同样只差 wrap
        self.assertAlmostEqual(filled["box_lidar"][6], expected, places=5)

    def test_falls_back_to_track_circular_median_without_slot_axis(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [[_det(1, 0.0, yaw=0.2)], [_det(1, 0.0, yaw=0.2)], [],
                    [_det(1, 0.0, yaw=0.2)]]
            points = {str(index): _points_inside() for index in range(4)}
            diagnostics, _output, _source = self._run(
                root, rows, points,
                slots=[{"track_id": 1, "class_name": "Car"}],
                yaw_slots=[])
        self.assertEqual(diagnostics["inserted_detections"], 1)
        self.assertEqual(diagnostics["tracks"][0]["yaw_source"],
                         "track_circular_median")

    def test_disabled_config_is_a_noop(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [[_det(1, 0.0)], [_det(1, 0.0)], [], [_det(1, 0.0)]]
            points = {str(index): _points_inside() for index in range(4)}
            diagnostics, output, _source = self._run(
                root, rows, points,
                slots=[{"track_id": 1, "class_name": "Car"}],
                yaw_slots=[{"track_id": 1, "target_world_yaw": 0.0}],
                config=Step5aConfig(enabled=False))
        self.assertEqual(diagnostics["enabled"], False)
        self.assertEqual(diagnostics["inserted_detections"], 0)
        self.assertEqual(len(output[2]["detections"]), 0)


if __name__ == "__main__":
    unittest.main()
