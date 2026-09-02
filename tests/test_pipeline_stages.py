import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline import step2_5_class_correction as step25
from pipeline import step2_identity as step2
from pipeline import step3_refinement as step3


def make_clip(root: Path) -> Path:
    (root / "transforms").mkdir(parents=True)
    (root / "lidar" / "lidar_top").mkdir(parents=True)
    identity = np.eye(4).tolist()
    (root / "transforms" / "calib.json").write_text(json.dumps({
        "tf2base_link": {
            "pose": identity,
            "lidar_top": identity,
        }
    }), encoding="utf-8")
    (root / "transforms" / "pose_data.txt").write_text(
        "0,0,0,0,0,0,0,1\n1000000000,0,0,0,0,0,0,1\n",
        encoding="utf-8")
    return root


def frame(class_name="Car", track_id=None):
    det = {
        "class_name": class_name,
        "score": 0.9,
        "box_lidar": [1.0, 0.0, 1.0, 4.0, 2.0, 1.6, 0.0],
    }
    if track_id is not None:
        det["track_id"] = track_id
    return {"frame_id": "0", "detections": [det], "num_detections": 1}


class PipelineStageContractTest(unittest.TestCase):
    def test_step2_filters_after_identity_assignment(self):
        source = [frame()]
        seen = {}

        class FakeTracker:
            slots = []

            def __init__(self, _coords):
                pass

            def process(self, frames):
                frames[0]["detections"][0]["track_id"] = 17
                return frames, {"tracks_total": 1}

        def fake_filter(frames, _clip, _config):
            seen["track_id"] = frames[0]["detections"][0].get("track_id")
            return {"detections_removed": 0}

        with tempfile.TemporaryDirectory() as directory:
            clip = make_clip(Path(directory))
            input_json = clip / "input.json"
            input_json.write_text(json.dumps(source), encoding="utf-8")
            with patch.object(step2.static_first, "StaticFirstTracker", FakeTracker), \
                    patch.object(step2, "apply_hard_filters", fake_filter), \
                    patch.object(step2, "deduplicate_same_center",
                                 return_value={"boxes_removed": 0}):
                result = step2.run(input_json, clip, clip / "out.json",
                                   diagnostics_path=clip / "diag.json")
        self.assertEqual(seen["track_id"], 17)
        self.assertEqual(result["stage_order"][0], "class_blind_identity_tracking")

    def test_step25_votes_class_then_runs_second_filter(self):
        source = [frame("Bus", 3), frame("Bus", 3), frame("Truck", 3)]
        seen = {}

        def fake_filter(frames, _clip, _config):
            seen["class"] = frames[0]["detections"][0]["class_name"]
            seen["track_id"] = frames[0]["detections"][0]["track_id"]
            return {"detections_removed": 0}

        with tempfile.TemporaryDirectory() as directory:
            clip = make_clip(Path(directory))
            input_json = clip / "step2.json"
            diag_json = clip / "step2_diag.json"
            input_json.write_text(json.dumps(source), encoding="utf-8")
            diag_json.write_text(json.dumps({"tracking": {}}), encoding="utf-8")
            with patch.object(step25, "apply_hard_filters", fake_filter):
                result = step25.run(input_json, diag_json, clip,
                                    clip / "out.json", diagnostics_path=clip / "diag25.json",
                                    min_lifecycle=0)
        self.assertEqual(seen, {"class": "Bus", "track_id": 3})
        self.assertEqual(result["class_only_check"]["passed"], True)
        self.assertEqual(result["stage_order"][1], "hard_filters_pass_2")

    def test_step3_non_car_route_is_passthrough(self):
        before = [frame("Truck", 4)]
        after = copy.deepcopy(before)
        self.assertEqual(step3._verify_non_car_geometry(before, after)["passed"], True)


if __name__ == "__main__":
    unittest.main()
