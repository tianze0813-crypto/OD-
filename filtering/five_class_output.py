"""Final adapter for the five-class model.

The legacy Step5 combined sparse-point filtering, lifecycle filtering and a
Car-only gate.  The five-class pipeline has already performed the agreed
annotation filters in Step2, so its final adapter only normalizes the model
class names and converts every kept box to ``base_link`` for SUST export.
"""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from tracking import tracker_conservative as tracking


TARGET_CLASSES = tracking.TARGET_CLASSES


def apply_five_class_output(
        frames: Sequence[Mapping[str, Any]],
        coords: tracking.CoordinateProvider,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Normalize and convert detections without class- or point-based drops."""
    output = copy.deepcopy(list(frames))
    before = sum(len(frame.get("detections", [])) for frame in output)
    removed: Counter[str] = Counter()
    converted = 0
    for frame in output:
        kept = []
        for det in frame.get("detections", []):
            canonical = tracking.canonical_class_name(
                det.get("class_name", ""))
            if canonical is None:
                removed["unknown_class"] += 1
                continue
            if not tracking.finite_box(det):
                removed["invalid_box"] += 1
                continue
            det["class_name"] = canonical
            det["box_lidar"] = tracking.box_lidar_to_base_link(
                det["box_lidar"], coords.base_from_lidar_top)
            det["box_frame"] = "base_link"
            converted += 1
            kept.append(det)
        frame["detections"] = kept
        frame["num_detections"] = len(kept)

    after = sum(len(frame.get("detections", [])) for frame in output)
    return output, {
        "target_classes": list(TARGET_CLASSES),
        "before_detections": before,
        "after_detections": after,
        "boxes_converted": converted,
        "removed_by_reason": dict(sorted(removed.items())),
        "policy": {
            "class_filter": "five target classes",
            "car_only": False,
            "drop_truck": False,
            "drop_static_nonmotorized": False,
            "point_filter": False,
            "lifecycle_filter": False,
            "output_box_frame": "base_link",
        },
    }
