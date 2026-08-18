import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pipeline.step6_car_only_filter import run
from filtering.car_only_filter import apply_car_only_filter
from tracking.tracker_conservative import box_to_label


def frames():
    return [{
        "frame_id": "1",
        "num_points": 0,
        "num_detections": 4,
        "detections": [
            {"class_name": "Car", "track_id": 1,
             "box_lidar": [1.0, 0.0, 0.0, 4.0, 1.8, 1.5, 0.0]},
            {"class_name": "Vehicle", "track_id": 2,
             "box_lidar": [2.0, 0.0, 0.0, 4.0, 1.8, 1.5, 0.0]},
            {"class_name": "Truck", "track_id": 3,
             "box_lidar": [3.0, 0.0, 0.0, 7.0, 2.5, 3.0, 0.0]},
            {"class_name": "Pedestrian", "track_id": 4,
             "box_lidar": [4.0, 0.0, 0.0, 0.8, 0.8, 1.7, 0.0]},
        ],
    }]


class Step6CarOnlyFilterTest(unittest.TestCase):
    def test_keeps_only_detections_exported_as_car(self):
        output, stats = apply_car_only_filter(frames())
        detections = output[0]["detections"]
        self.assertEqual([d["track_id"] for d in detections], [1, 2])
        self.assertEqual(
            {box_to_label(d)["obj_type"] for d in detections}, {"Car"})
        self.assertEqual(stats["before_detections"], 4)
        self.assertEqual(stats["after_detections"], 2)
        self.assertEqual(stats["detections_removed"], 2)
        self.assertEqual(stats["classes_removed"], {
            "Pedestrian": 1, "Truck": 1})

    def test_does_not_mutate_input(self):
        source = frames()
        apply_car_only_filter(source)
        self.assertEqual(source[0]["num_detections"], 4)
        self.assertEqual(len(source[0]["detections"]), 4)

    def test_pipeline_run_writes_car_only_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / "clip_step5"
            clip.mkdir()
            step5_json = root / "step5.json"
            out_json = root / "step6.json"
            out_clip = root / "clip_step6"
            diagnostics = root / "step6_diagnostics.json"
            step5_json.write_text(json.dumps(frames()), encoding="utf-8")

            result = run(step5_json, clip, out_json, out_clip, diagnostics)
            labels = json.loads(
                (out_clip / "label" / "1.json").read_text(encoding="utf-8"))

        self.assertEqual(result["after_detections"], 2)
        self.assertEqual({label["obj_type"] for label in labels}, {"Car"})
        self.assertEqual(len(labels), 2)


if __name__ == "__main__":
    unittest.main()
