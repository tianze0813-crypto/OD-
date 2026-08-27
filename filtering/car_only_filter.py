"""Step 6: keep only detections that export as the SUST ``Car`` class."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence

from tracking import tracker_conservative as tracking


def _label_class(det: Mapping[str, Any]) -> str:
    """Return the canonical class written by ``tracking.box_to_label``."""
    raw = det.get("class_name", "")
    mapped = tracking.CLASS_MAP.get(raw)
    if mapped is not None:
        return mapped
    return tracking.CLASS_MAP.get(str(raw).lower(), str(raw))


def apply_car_only_filter(
        frames: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter a step-5 result without changing any kept detection fields."""
    before = copy.deepcopy(list(frames))
    output = copy.deepcopy(list(frames))
    before_classes = Counter(
        _label_class(det)
        for frame in before
        for det in frame.get("detections", [])
    )
    removed_classes = Counter()

    for frame in output:
        kept = []
        for det in frame.get("detections", []):
            label_class = _label_class(det)
            if label_class == "Car":
                kept.append(det)
            else:
                removed_classes[label_class] += 1
        frame["detections"] = kept
        frame["num_detections"] = len(kept)

    _verify_removal_only(before, output)
    before_detections = sum(len(f.get("detections", [])) for f in before)
    after_detections = sum(len(f.get("detections", [])) for f in output)
    return output, {
        "policy": {
            "pipeline_position": "after_step5_final_filter",
            "keep_label_class": "Car",
            "criterion": "canonical class emitted by box_to_label",
            "mutated_fields": "remove non-Car detections only",
        },
        "classes_before": dict(sorted(before_classes.items())),
        "classes_removed": dict(sorted(removed_classes.items())),
        "before_detections": before_detections,
        "after_detections": after_detections,
        "detections_removed": before_detections - after_detections,
    }


def _verify_removal_only(
        before: Sequence[Dict[str, Any]],
        after: Sequence[Dict[str, Any]],
) -> None:
    if len(before) != len(after):
        raise AssertionError("step6 filter changed frame count")
    for left_frame, right_frame in zip(before, after):
        if left_frame.get("frame_id") != right_frame.get("frame_id"):
            raise AssertionError("step6 filter changed frame order")
        left_dets = left_frame.get("detections", [])
        right_dets = right_frame.get("detections", [])
        cursor = 0
        for det in right_dets:
            while cursor < len(left_dets) and left_dets[cursor] != det:
                cursor += 1
            if cursor >= len(left_dets):
                raise AssertionError("step6 filter changed a kept detection")
            cursor += 1
