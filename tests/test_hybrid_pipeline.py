import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline.hybrid_expD_noncar import (
    _noncar_filter,
    drop_long_stationary_nonmotorized,
    drop_spinning_vehicle,
)
from pipeline.hybrid_merge import merge_frames, merge_label_frames


_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_hybrid_prelabel.py"
_SPEC = importlib.util.spec_from_file_location("hybrid_launcher", _SCRIPT)
hybrid_launcher = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(hybrid_launcher)


class _IdentityCoords:
    """Minimal CoordinateProvider stub: lidar frame == world frame."""

    def world_from_lidar(self, timestamp):
        return np.eye(4, dtype=np.float64)


def _moving_frames(class_name, centers, track_id=7):
    frames = []
    for index, x in enumerate(centers):
        frames.append({
            "frame_id": str((index + 1) * 1_000_000_000),
            "detections": [{
                "track_id": track_id,
                "class_name": class_name,
                "score": 0.5,
                "box_lidar": [float(x), 0.0, 0.0, 1.5, 0.8, 1.5, 0.0],
            }],
        })
    return frames


def _det(track_id, name):
    return {
        "track_id": track_id,
        "class_name": name,
        "score": 0.8,
        "box_lidar": [1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.0],
    }


class HybridPipelineTest(unittest.TestCase):
    def test_early_non_car_filter_removes_car_and_canonicalizes_aliases(self):
        frames = [{
            "frame_id": "1",
            "detections": [
                _det(1, "Car"), _det(2, "Truck"), _det(3, "Cyclist"),
            ],
        }]

        stats = _noncar_filter(frames)

        self.assertEqual(stats["detections_removed"], 1)
        self.assertEqual(
            [det["class_name"] for det in frames[0]["detections"]],
            ["Truck", "Nonmotorized_vehicle"],
        )

    def test_merge_is_frame_aligned_and_remaps_exp_d_ids(self):
        main = [{"frame_id": "1", "num_points": 7,
                 "detections": [_det(1, "Car")]}]
        expd = [{"frame_id": "1", "num_points": 9,
                 "detections": [_det(1, "Truck"), _det(2, "Bus")]}]

        output, stats = merge_frames(main, expd)

        self.assertEqual(stats["main_car_detections"], 1)
        self.assertEqual(stats["expd_non_car_detections"], 2)
        self.assertEqual(
            [det["class_name"] for det in output[0]["detections"]],
            ["Car", "Truck", "Bus"],
        )
        self.assertEqual(
            [det["track_id"] for det in output[0]["detections"]],
            [1, 2, 3],
        )
        self.assertEqual(output[0]["num_points"], 7)

    def test_merge_rejects_frame_mismatch(self):
        with self.assertRaisesRegex(ValueError, "frame IDs differ"):
            merge_frames([{"frame_id": "1", "detections": []}],
                         [{"frame_id": "2", "detections": []}])

    def test_merge_label_frames_preserves_main_labels_and_remaps_exp_d_ids(self):
        main = {"1": [{"obj_id": "1", "obj_type": "Car", "score": 0.9}]}
        expd = [{"frame_id": "1", "detections": [_det(1, "Truck")]}]

        output, stats = merge_label_frames(main, expd)

        self.assertEqual([label["obj_type"] for label in output[0]["labels"]],
                         ["Car", "Truck"])
        self.assertEqual([label["obj_id"] for label in output[0]["labels"]],
                         ["1", "2"])
        self.assertEqual(stats["merged_detections"], 2)

    def test_drop_spinning_vehicle_removes_erratic_tracks_only(self):
        def frames_for(entries):
            frames = []
            for track_id, yaws in entries.items():
                for index, yaw in enumerate(yaws):
                    frames.append({
                        "frame_id": str(index),
                        "detections": [{
                            "track_id": track_id, "class_name": "Bus",
                            "score": 0.8,
                            "box_lidar": [0, 0, 1, 10, 2, 3, yaw],
                        }],
                        "num_detections": 1,
                    })
            return frames

        spinning = {50: [1.33, -2.87, 2.2, 2.98, 2.65, 1.2, 2.68, 2.63, 0.4],
                    33: [-0.52, -0.3, -0.54, -0.62, -1.58, -0.47]}
        stable = {24: [-0.62, -0.57, -0.55, -0.63, -0.6, -0.61, -0.62],
                  43: [-0.65, -0.64, -0.78, -0.69, -0.78, -0.63, -0.78]}
        frames = frames_for(spinning) + frames_for(stable)

        dropped, stats = drop_spinning_vehicle(frames)

        self.assertEqual(stats["dropped_track_ids"], [33, 50])
        self.assertEqual(stats["tracks_dropped"], 2)
        survivors = {
            det["track_id"]
            for frame in frames for det in frame.get("detections", [])
        }
        self.assertEqual(survivors, set(stable))

    def test_hybrid_defaults_do_not_reference_moga_paths(self):
        # Defaults must be derived from this checkout root (ROOT), not from a
        # hardcoded personal home folder.  The checkout itself may live under a
        # directory named "moga", so only the launcher source is checked.
        root = Path(hybrid_launcher.__file__).resolve().parents[1]
        self.assertEqual(
            hybrid_launcher.NONCAR_CFG,
            root / "models" / "voxelnext_fiveclass_nuscenes_infer.yaml")
        self.assertEqual(
            hybrid_launcher.NONCAR_CKPT,
            root / "models" / "vod_2cls_ft_e12.pth")
        source = Path(hybrid_launcher.__file__).read_text(encoding="utf-8")
        self.assertNotIn("/home/moga", source)
        self.assertNotIn("moga/", source)

    def test_runner_is_serial_and_writes_one_merged_clip(self):
        events = []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scene"
            (source / "lidar" / "lidar_top").mkdir(parents=True)
            (source / "lidar" / "lidar_top" / "1.bin").write_bytes(b"")
            output_root = root / "out"

            def fake_main(_python, _clip, _work_root, **_kwargs):
                events.append("main")
                return ({"1": [{
                    "obj_id": "1", "obj_type": "Car", "score": 0.9,
                }]}, {"final_detections": 1})

            def fake_raw(_python, _clip, _cfg, _ckpt, work_root, name, _threshold):
                events.append("expd_inference")
                work_root.mkdir(parents=True, exist_ok=True)
                raw = work_root / f"{name}_raw.json"
                raw.write_text("[]", encoding="utf-8")
                return raw

            def fake_expd(_raw, _clip, out_json, _diag, **_kwargs):
                events.append("expd_postprocess")
                frames = [{"frame_id": "1", "detections": [{
                    "track_id": 1, "class_name": "Truck", "score": 0.8,
                    "box_lidar": [1, 2, 0, 4, 2, 1.5, 0],
                }]}]
                out_json.write_text(json.dumps(frames), encoding="utf-8")
                return {"final_detections": 1}

            with patch.object(hybrid_launcher, "run_main_car", fake_main), \
                    patch.object(hybrid_launcher, "_run_raw", fake_raw), \
                    patch.object(hybrid_launcher, "run_expd_noncar", fake_expd):
                result = hybrid_launcher.run_clip(
                    Path("python"), source, output_root, overwrite=False,
                    drop_vis_below=0.05, score_threshold=None,
                    short_track_max_frames=4)

            self.assertEqual(events, ["main", "expd_inference", "expd_postprocess"])
            labels = json.loads(
                (output_root / "scene_pre" / "label" / "1.json").read_text())
            self.assertEqual([label["obj_type"] for label in labels],
                             ["Car", "Truck"])
            self.assertFalse((source / "label").exists())
            self.assertEqual(result["labels"], 2)

    def test_runner_without_export_does_not_write_sust(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scene"
            (source / "lidar" / "lidar_top").mkdir(parents=True)
            (source / "lidar" / "lidar_top" / "1.bin").write_bytes(b"")
            output_root = root / "out"

            def fake_main(_python, _clip, _work_root, **_kwargs):
                return ({"1": [{
                    "obj_id": "1", "obj_type": "Car", "score": 0.9,
                }]}, {"final_detections": 1})

            def fake_raw(_python, _clip, _cfg, _ckpt, work_root, name, _threshold):
                work_root.mkdir(parents=True, exist_ok=True)
                raw = work_root / f"{name}_raw.json"
                raw.write_text("[]", encoding="utf-8")
                return raw

            def fake_expd(_raw, _clip, out_json, _diag, **_kwargs):
                frames = [{"frame_id": "1", "detections": [{
                    "track_id": 1, "class_name": "Truck", "score": 0.8,
                    "box_lidar": [1, 2, 0, 4, 2, 1.5, 0],
                }]}]
                out_json.write_text(json.dumps(frames), encoding="utf-8")
                return {"final_detections": 1}

            with patch.object(hybrid_launcher, "run_main_car", fake_main), \
                    patch.object(hybrid_launcher, "_run_raw", fake_raw), \
                    patch.object(hybrid_launcher, "run_expd_noncar", fake_expd):
                result = hybrid_launcher.run_clip(
                    Path("python"), source, output_root, overwrite=False,
                    export_sust=False, drop_vis_below=0.05,
                    score_threshold=None, short_track_max_frames=4)

            self.assertIsNone(result["final_clip"])
            self.assertFalse((output_root / "scene_pre").exists())
            self.assertEqual(result["labels"], 2)


class LongStationaryNonmotorizedTest(unittest.TestCase):
    def test_long_static_nmv_track_is_dropped(self):
        frames = _moving_frames("Nonmotorized_vehicle", [0.0] * 10)
        dropped, stats = drop_long_stationary_nonmotorized(
            frames, _IdentityCoords(), min_frames=8,
            max_world_displacement=1.0)
        self.assertEqual(dropped, {7})
        self.assertEqual(stats["tracks_dropped"], 1)
        self.assertEqual(stats["boxes_removed"], 10)
        self.assertTrue(all(not frame["detections"] for frame in frames))

    def test_moving_nmv_track_is_kept(self):
        frames = _moving_frames(
            "Nonmotorized_vehicle", [0.2 * index for index in range(10)])
        dropped, stats = drop_long_stationary_nonmotorized(
            frames, _IdentityCoords(), min_frames=8,
            max_world_displacement=1.0)
        self.assertEqual(dropped, set())
        self.assertEqual(stats["tracks_checked"], 1)
        self.assertTrue(all(len(frame["detections"]) == 1 for frame in frames))

    def test_track_with_cumulative_movement_is_kept(self):
        # span stays <= 1m, but cumulative world path exceeds 1m: the track
        # must be kept so waiting/creeping objects are not dropped.
        frames = _moving_frames(
            "Nonmotorized_vehicle",
            [0.6 * (index % 2) for index in range(10)])
        dropped, stats = drop_long_stationary_nonmotorized(
            frames, _IdentityCoords(), min_frames=8,
            max_world_displacement=1.0)
        self.assertEqual(dropped, set())
        self.assertEqual(stats["tracks_dropped"], 0)

    def test_short_static_nmv_track_is_kept(self):
        frames = _moving_frames("Nonmotorized_vehicle", [0.0] * 6)
        dropped, _stats = drop_long_stationary_nonmotorized(
            frames, _IdentityCoords(), min_frames=8,
            max_world_displacement=1.0)
        self.assertEqual(dropped, set())

    def test_static_track_of_other_class_is_kept(self):
        frames = _moving_frames("Truck", [0.0] * 10)
        dropped, _stats = drop_long_stationary_nonmotorized(
            frames, _IdentityCoords(), min_frames=8,
            max_world_displacement=1.0)
        self.assertEqual(dropped, set())


if __name__ == "__main__":
    unittest.main()
