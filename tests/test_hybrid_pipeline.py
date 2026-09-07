import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.hybrid_expD_noncar import _noncar_filter
from pipeline.hybrid_merge import merge_frames, merge_label_frames


_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_hybrid_prelabel.py"
_SPEC = importlib.util.spec_from_file_location("hybrid_launcher", _SCRIPT)
hybrid_launcher = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(hybrid_launcher)


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
                         ["1", "expd_1"])
        self.assertEqual(stats["merged_detections"], 2)

    def test_hybrid_defaults_do_not_reference_moga_paths(self):
        self.assertNotIn("moga", str(hybrid_launcher.EXPD_CFG))

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


if __name__ == "__main__":
    unittest.main()
