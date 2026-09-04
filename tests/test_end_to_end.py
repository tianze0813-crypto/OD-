import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_end_to_end


_LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "five_class_launcher", Path(__file__).parents[1] / "scripts" /
    "run_five_class.py")
run_five_class = importlib.util.module_from_spec(_LAUNCHER_SPEC)
assert _LAUNCHER_SPEC.loader is not None
_LAUNCHER_SPEC.loader.exec_module(run_five_class)


class EndToEndLabelWriterTest(unittest.TestCase):
    def test_writes_only_label_dir_into_existing_clip(self):
        frames = [{
            "frame_id": "123",
            "num_points": 0,
            "num_detections": 1,
            "detections": [{
                "class_name": "Car",
                "score": 0.9,
                "box_lidar": [1.0, 2.0, 0.0, 4.0, 1.8, 1.5, 0.1],
                "track_id": 7,
            }],
        }]
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "scene_clip_pre"
            clip.mkdir()
            (clip / "lidar").mkdir()
            labels = run_end_to_end.write_labels_only(frames, clip)
            self.assertEqual(labels, 1)
            self.assertTrue((clip / "label" / "123.json").is_file())
            payload = json.loads(
                (clip / "label" / "123.json").read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["obj_id"], "7")

    def test_preserve_input_materializes_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "scene_clip"
            destination = root / "scene_clip_pre"
            source.mkdir()
            (source / "marker.txt").write_text("raw", encoding="utf-8")

            run_end_to_end.materialize_final_clip(
                source, destination, preserve_input=True)

            self.assertTrue(source.is_dir())
            self.assertEqual(
                (destination / "marker.txt").read_text(encoding="utf-8"),
                "raw")

    def test_launcher_ignores_existing_pre_output_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("scene_clip", "scene_clip_pre"):
                lidar = root / name / "lidar" / "lidar_top"
                lidar.mkdir(parents=True)
                (lidar / "1.bin").write_bytes(b"")

            clips = run_five_class._collect_clips(root)

            self.assertEqual([clip.name for clip in clips], ["scene_clip"])

    def test_raw_launcher_copies_only_raw_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "scene_clip"
            clip.mkdir()
            raw_output = root / "raw"

            def fake_step1(command, **_kwargs):
                work_root = Path(command[command.index("--work-root") + 1])
                work_root.mkdir(parents=True, exist_ok=True)
                (work_root / "scene_clip_raw.json").write_text(
                    "[]\n", encoding="utf-8")
                return mock.Mock(returncode=0)

            with mock.patch.object(run_five_class, "_run", side_effect=fake_step1):
                run_five_class._run_raw_inference(
                    Path("python"), [clip], Path("config.yaml"),
                    Path("checkpoint.pth"), raw_output,
                    score_thresh=0.1, overwrite=False)

            self.assertEqual(
                sorted(path.name for path in raw_output.iterdir()),
                ["scene_clip_raw.json"])
            self.assertEqual(
                (raw_output / "scene_clip_raw.json").read_text(encoding="utf-8"),
                "[]\n")


if __name__ == "__main__":
    unittest.main()
