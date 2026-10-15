"""公共收尾阶段（合并后统一算遮挡属性 + 可见度过滤）的用例。"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from filtering.camera_visibility import load_base_link_points
from pipeline.final_visibility import apply_final_visibility, label_to_box


_SCRIPT = Path(__file__).parents[2] / "scripts" / "run_hybrid_prelabel.py"
_SPEC = importlib.util.spec_from_file_location("hybrid_launcher_vis", _SCRIPT)
hybrid_launcher = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(hybrid_launcher)


def _write_clip(root: Path, frame_id: str = "7") -> Path:
    """最小 clip：一台相机（单位外参、无畸变）+ 一帧 top 点云。"""
    transforms = root / "transforms"
    transforms.mkdir(parents=True)
    calib = {
        "tf2base_link": {
            "lidar_top": np.eye(4).tolist(),
            "cam_front": np.eye(4).tolist(),
        },
        "cam_front": {
            "K": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
            "D": [0.0, 0.0, 0.0, 0.0],
            "imgw": 100,
            "imgh": 100,
        },
    }
    (transforms / "calib.json").write_text(json.dumps(calib), encoding="utf-8")
    (transforms / "pose_data.txt").write_text("0,0,0,0,0,0,0,1\n", encoding="utf-8")
    lidar = root / "lidar" / "lidar_top"
    lidar.mkdir(parents=True)
    # 让"车框(中心 z=5, dz=1.5)"和"行人框(中心 z=10, dz=1.8)"里的点都跨满框高，
    # 这样点云高度比判为完全可见（否则会被当成被遮挡）。
    points = np.array([
        [0.0, 0.0, 4.3, 0.5], [0.0, 0.0, 5.7, 0.5],
        [0.0, 0.0, 9.15, 0.5], [0.0, 0.0, 10.85, 0.5],
    ], dtype=np.float32)
    points.tofile(lidar / f"{frame_id}.bin")
    return root


def _write_points(clip: Path, frame_id: str, points) -> None:
    np.asarray(points, dtype=np.float32).tofile(
        clip / "lidar" / "lidar_top" / f"{frame_id}.bin")


def _label(obj_id: str, obj_type: str, z: float, dx: float, dy: float, dz: float):
    return {
        "obj_id": obj_id, "obj_type": obj_type, "score": 0.8,
        "psr": {
            "position": {"x": 0.0, "y": 0.0, "z": float(z)},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "scale": {"x": float(dx), "y": float(dy), "z": float(dz)},
        },
    }


class FinalVisibilityTest(unittest.TestCase):
    def test_label_to_box_rejects_broken_labels(self):
        self.assertIsNone(label_to_box({"obj_type": "Car"}))
        self.assertIsNone(label_to_box(_label("1", "Car", 0.0, 0.0, 1.0, 1.0)))

    def test_cross_class_occlusion_happens_after_merge(self):
        """行人被更近的车挡住 -> 公共阶段才能判出来（旧实现里 VRU 链帧内没有车）。"""
        with tempfile.TemporaryDirectory() as directory:
            clip = _write_clip(Path(directory))
            merged = [{"frame_id": "7", "labels": [
                _label("1000", "Car", 5.0, 2.0, 2.0, 1.5),          # 近，挡住行人
                _label("2001", "Pedestrian", 10.0, 0.6, 0.6, 1.8),  # 远，被完全遮挡
            ]}]
            stats = apply_final_visibility(merged, clip, min_ratio=0.05,
                                           use_point_occlusion=False)
        self.assertEqual(stats["boxes_removed"], 1)
        self.assertEqual(stats["removed_by_class"], {"Pedestrian": 1})
        self.assertEqual([item["obj_type"] for item in merged[0]["labels"]], ["Car"])
        self.assertEqual(merged[0]["labels"][0]["visibility"]["tag"], 1)

    def test_alone_pedestrian_is_fully_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = _write_clip(Path(directory))
            merged = [{"frame_id": "7", "labels": [
                _label("2001", "Pedestrian", 10.0, 0.6, 0.6, 1.8),
            ]}]
            stats = apply_final_visibility(merged, clip, use_point_occlusion=False)
        self.assertEqual(stats["boxes_removed"], 0)
        self.assertEqual(stats["tag1"], 1)
        self.assertAlmostEqual(
            merged[0]["labels"][0]["visibility"]["ratio"], 1.0, places=3)

    def test_visibility_below_threshold_is_configurable(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = _write_clip(Path(directory))
            merged = [{"frame_id": "7", "labels": [
                _label("1000", "Car", 5.0, 2.0, 2.0, 1.5),
                _label("2001", "Pedestrian", 10.0, 0.6, 0.6, 1.8),
            ]}]
            stats = apply_final_visibility(
                merged, clip, min_ratio=0.0, use_point_occlusion=False)
        self.assertEqual(stats["boxes_removed"], 0)     # 阈值 <= 0 = 只算不删
        self.assertFalse(stats["policy"]["drop_enabled"])
        self.assertEqual(len(merged[0]["labels"]), 2)

    def test_point_cloud_height_ratio_uses_fused_dir_when_present(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = _write_clip(Path(directory))
            fused = Path(directory) / "fused"
            fused.mkdir()
            np.array([[0.0, 0.0, 9.15, 0.5], [0.0, 0.0, 10.85, 0.5]],
                     dtype=np.float32).tofile(fused / "7.bin")
            merged = [{"frame_id": "7", "labels": [
                _label("2001", "Pedestrian", 10.0, 0.6, 0.6, 1.8)]}]
            stats = apply_final_visibility(merged, clip, points_dir=fused)
        self.assertEqual(stats["policy"]["points_dir"], str(fused))
        self.assertEqual(stats["boxes_removed"], 0)

    def test_default_call_computes_attribute_without_dropping(self):
        """默认（min_ratio=0）= 只算遮挡属性，不删任何框。"""
        with tempfile.TemporaryDirectory() as directory:
            clip = _write_clip(Path(directory))
            merged = [{"frame_id": "7", "labels": [
                _label("1000", "Car", 5.0, 2.0, 2.0, 1.5),
                _label("2001", "Pedestrian", 10.0, 0.6, 0.6, 1.8),
            ]}]
            stats = apply_final_visibility(merged, clip, use_point_occlusion=False)
        self.assertFalse(stats["policy"]["drop_enabled"])
        self.assertEqual(stats["boxes_removed"], 0)
        self.assertEqual(len(merged[0]["labels"]), 2)
        self.assertTrue(all("visibility" in item for item in merged[0]["labels"]))
        # 属性照样算出来了：行人在相机凸包下被车完全挡住 -> tag2
        ped = [item for item in merged[0]["labels"] if item["obj_type"] == "Pedestrian"][0]
        self.assertEqual(ped["visibility"]["tag"], 2)

    def test_missing_calibration_fails_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / "scene"
            clip.mkdir()
            with self.assertRaises(FileNotFoundError):
                apply_final_visibility([{"frame_id": "7", "labels": []}], clip)


class PointHeightOcclusionTest(unittest.TestCase):
    """点云高度比（LiDAR 补判）要真的起作用：光靠相机凸包看不到"被非框物体挡住"。"""

    def _run(self, points):
        with tempfile.TemporaryDirectory() as directory:
            clip = _write_clip(Path(directory))
            _write_points(clip, "7", points)
            merged = [{"frame_id": "7", "labels": [
                _label("2001", "Pedestrian", 10.0, 0.6, 0.6, 1.8)]}]
            stats = apply_final_visibility(merged, clip, min_ratio=0.05)
            return stats, merged

    def test_low_point_span_is_treated_as_occluded_and_dropped(self):
        # 框 z ∈ [9.1, 10.9]，点只落在框底 5cm（跨度/框高 ≈ 0.03）-> 可见度 ≈ 0.03
        stats, merged = self._run([[0.0, 0.0, 9.20, 0.5], [0.0, 0.0, 9.25, 0.5]])
        self.assertEqual(stats["boxes_removed"], 1)
        self.assertEqual(merged[0]["labels"], [])

    def test_full_point_span_is_fully_visible(self):
        stats, _ = self._run([[0.0, 0.0, 9.15, 0.5], [0.0, 0.0, 10.85, 0.5]])
        self.assertEqual(stats["boxes_removed"], 0)
        self.assertEqual(stats["tag1"], 1)


class PointLoaderTest(unittest.TestCase):
    def test_points_dir_wins_over_clip_lidar(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = _write_clip(Path(directory))
            fused = Path(directory) / "fused"
            fused.mkdir()
            np.array([[1.0, 2.0, 3.0, 0.0]], dtype=np.float32).tofile(fused / "7.bin")
            points = load_base_link_points(clip, "7", fused)
        self.assertTrue(np.allclose(points, [[1.0, 2.0, 3.0]]))

    def test_falls_back_to_clip_lidar_top(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = _write_clip(Path(directory))
            fused = Path(directory) / "fused-empty"
            fused.mkdir()
            points = load_base_link_points(clip, "7", fused)
        self.assertEqual(points.shape[0], 4)


class InPlaceChainReuseTest(unittest.TestCase):
    """就地重跑部分链：盘上不属于本次重算类别的标签要被捡回来。"""

    def _clip_with_existing_labels(self, root: Path) -> Path:
        label_dir = root / "label" / "det"
        label_dir.mkdir(parents=True)
        labels = [
            _label("1", "Car", 0.0, 4.0, 2.0, 1.5),
            _label("1001", "Truck", 0.0, 6.0, 2.5, 3.0),
            _label("2001", "Pedestrian", 0.0, 0.6, 0.6, 1.8),
        ]
        (label_dir / "7.json").write_text(json.dumps(labels), encoding="utf-8")
        return root

    def test_vru_rerun_keeps_car_and_truck(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = self._clip_with_existing_labels(Path(directory))
            kept, dropped = hybrid_launcher._reusable_existing_labels(clip, ["vru"])
        self.assertEqual(dropped, 1)                      # 只剔掉 Pedestrian
        self.assertEqual(sorted(item["obj_type"] for item in kept["7"]),
                         ["Car", "Truck"])

    def test_vehicle_rerun_keeps_vru(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = self._clip_with_existing_labels(Path(directory))
            kept, dropped = hybrid_launcher._reusable_existing_labels(
                clip, ["car", "truck"])
        self.assertEqual(dropped, 2)                      # Car + Truck
        self.assertEqual([item["obj_type"] for item in kept["7"]], ["Pedestrian"])

    def test_no_existing_labels_returns_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory)
            clip.mkdir(parents=True, exist_ok=True)
            kept, dropped = hybrid_launcher._reusable_existing_labels(clip, ["vru"])
        self.assertEqual(kept, {})
        self.assertEqual(dropped, 0)


if __name__ == "__main__":
    unittest.main()


class RunnerWiringTest(unittest.TestCase):
    """端到端接线：三链标签合并 -> 公共阶段 -> 写 label/det。"""

    def test_runner_applies_final_visibility_to_merged_labels(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_clip(root / "scene")
            output_root = root / "out"

            def fake_vehicle(_raw, _clip, _work_root, _python, **_kwargs):
                return {
                    "car_frames": [{"frame_id": "7", "detections": [{
                        "track_id": 1, "class_name": "Car", "score": 0.9,
                        "box_lidar": [0.0, 0.0, 5.0, 2.0, 2.0, 1.5, 0.0]}]}],
                    "truck_frames": [],
                    "diagnostics": {"car_detections": 1, "truck_detections": 0}}

            def fake_bev(_python, _clip, work_root, _threshold, _mode,
                         _cfg=None, _ckpt=None, fuse_lidars=False):
                work_root.mkdir(parents=True, exist_ok=True)
                raw = work_root / "bev_raw.json"
                raw.write_text("[]", encoding="utf-8")
                return raw

            def fake_raw(_python, _clip, _cfg, _ckpt, work_root, name,
                         _threshold, fuse_lidars=False):
                work_root.mkdir(parents=True, exist_ok=True)
                raw = work_root / f"{name}_raw.json"
                raw.write_text("[]", encoding="utf-8")
                return raw

            def fake_vru(_raw, _clip, out_json, _diag, **_kwargs):
                frames = [{"frame_id": "7", "detections": [{
                    "track_id": 1, "class_name": "Pedestrian", "score": 0.8,
                    "box_lidar": [0.0, 0.0, 10.0, 0.6, 0.6, 1.8, 0.0]}]}]
                out_json.write_text(json.dumps(frames), encoding="utf-8")
                return {"final_detections": 1}

            with patch.object(hybrid_launcher, "run_vehicle_pass", fake_vehicle), \
                    patch.object(hybrid_launcher, "_run_raw_bevfusion", fake_bev), \
                    patch.object(hybrid_launcher, "_run_raw", fake_raw), \
                    patch.object(hybrid_launcher, "run_expd_vru", fake_vru):
                result = hybrid_launcher.run_clip(
                    Path("python"), source, output_root, overwrite=False,
                    drop_vis_below=0.05, score_threshold=None,
                    short_track_max_frames=4,
                    chains=("car", "truck", "vru"), truck_detector="bevfusion")

            labels = json.loads(
                (output_root / "scene_pre" / "label" / "det" / "7.json")
                .read_text(encoding="utf-8"))
            self.assertEqual([item["obj_type"] for item in labels], ["Car"])
            self.assertEqual(labels[0]["visibility"]["tag"], 1)
            self.assertEqual(result["merge"]["final_visibility"]["boxes_removed"], 1)
            # VRU 默认融合点云：跑完公共阶段后不留融合目录
            self.assertFalse(
                any(p.name == "lidar_fused"
                    for p in (output_root / "scene_pre").rglob("lidar_fused")))


class FusedCloudLifetimeTest(unittest.TestCase):
    """融合点云要活到公共阶段，用完由编排删掉。"""

    def test_fused_dir_is_found_and_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "work"
            clip = Path(directory) / "scene"
            clip.mkdir(parents=True)
            self.assertIsNone(hybrid_launcher._fused_lidar_dir(work, clip))

            fused = work / "vru_raw" / "lidar_fused" / "scene"
            fused.mkdir(parents=True)
            (fused / "7.bin").write_bytes(b"\0" * 16)
            self.assertEqual(hybrid_launcher._fused_lidar_dir(work, clip), fused)

            hybrid_launcher._cleanup_fused_lidar(fused)
            self.assertFalse(fused.exists())
            # 空的 lidar_fused 父目录一并收掉，但 vru_raw 留着
            self.assertFalse((work / "vru_raw" / "lidar_fused").exists())
            self.assertTrue((work / "vru_raw").is_dir())


class InPlaceVruRerunTest(unittest.TestCase):
    """只重跑 VRU 链并就地重写：盘上已有的 Car 不能被抹掉，可见度要整体重算。"""

    def test_vru_only_rerun_keeps_existing_car_labels(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = _write_clip(root / "scene_pre")
            # Car 在 (0,0,5)、行人（本轮新算）在 (3,0,10)：两边的框都要有跨满框高的点
            _write_points(clip, "7", [[0.0, 0.0, 4.3, 0.5], [0.0, 0.0, 5.7, 0.5],
                                      [3.0, 0.0, 9.15, 0.5], [3.0, 0.0, 10.85, 0.5]])
            label_dir = clip / "label" / "det"
            label_dir.mkdir(parents=True)
            car = _label("1", "Car", 5.0, 2.0, 2.0, 1.5)
            (label_dir / "7.json").write_text(json.dumps([car]), encoding="utf-8")

            def fake_raw(_python, _clip, _cfg, _ckpt, work_root, name,
                         _threshold, fuse_lidars=False):
                work_root.mkdir(parents=True, exist_ok=True)
                raw = work_root / f"{name}_raw.json"
                raw.write_text("[]", encoding="utf-8")
                return raw

            def fake_vru(_raw, _clip, out_json, _diag, **_kwargs):
                # x=3 偏离车体投影，行人自身可见
                frames = [{"frame_id": "7", "detections": [{
                    "track_id": 1, "class_name": "Pedestrian", "score": 0.8,
                    "box_lidar": [3.0, 0.0, 10.0, 0.6, 0.6, 1.8, 0.0]}]}]
                out_json.write_text(json.dumps(frames), encoding="utf-8")
                return {"final_detections": 1}

            def fake_bev(_python, _clip, work_root, _threshold, _mode,
                         _cfg=None, _ckpt=None, fuse_lidars=False):
                work_root.mkdir(parents=True, exist_ok=True)
                raw = work_root / "bev_raw.json"
                raw.write_text("[]", encoding="utf-8")
                return raw

            with patch.object(hybrid_launcher, "_run_raw", fake_raw), \
                    patch.object(hybrid_launcher, "_run_raw_bevfusion", fake_bev), \
                    patch.object(hybrid_launcher, "run_expd_vru", fake_vru):
                result = hybrid_launcher.run_clip(
                    Path("python"), clip, root / "out", overwrite=True,
                    in_place=True, drop_vis_below=0.05, score_threshold=None,
                    short_track_max_frames=4, chains=("vru",))

            labels = json.loads((label_dir / "7.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(item["obj_type"] for item in labels),
                             ["Car", "Pedestrian"])
            self.assertTrue(all("visibility" in item for item in labels))
            reuse = result["merge"]["reused_existing_labels"]
            self.assertEqual(reuse["labels"], 1)
            self.assertEqual(reuse["dropped_from_disk"], 0)
