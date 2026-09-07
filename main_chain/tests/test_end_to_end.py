import json
import tempfile
import unittest
from pathlib import Path

import run_end_to_end


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


if __name__ == "__main__":
    unittest.main()
