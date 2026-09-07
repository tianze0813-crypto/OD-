"""Step 4 class gate for Car detections with truck-sized boxes."""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class LargeCarFilterConfig:
    """A track is truck-sized when its robust length reaches this threshold."""

    truck_length_min: float = 6.0


def _track_key(det: Dict[str, Any], fallback: int) -> tuple[str, int]:
    track_id = det.get("track_id")
    return ("track", int(track_id)) if track_id is not None else ("det", fallback)


def _box_length(det: Dict[str, Any]) -> float | None:
    box = det.get("box_lidar")
    if not isinstance(box, list) or len(box) < 5:
        return None
    values = np.asarray(box[3:5], dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        return None
    return float(max(values[0], values[1]))


def apply_large_car_to_truck(
        frames: Sequence[Dict[str, Any]],
        config: LargeCarFilterConfig = LargeCarFilterConfig(),
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Relabel only truck-sized Car tracks; leave geometry untouched."""
    output = copy.deepcopy(list(frames))
    groups: Dict[tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    fallback = 0
    for frame in output:
        for det in frame.get("detections", []):
            if str(det.get("class_name", "")) != "Car":
                continue
            groups[_track_key(det, fallback)].append(det)
            fallback += 1

    relabeled_tracks = 0
    relabeled_detections = 0
    examples: List[Dict[str, Any]] = []
    for key, detections in groups.items():
        lengths = [value for det in detections
                   if (value := _box_length(det)) is not None]
        if not lengths:
            continue
        robust_length = float(np.median(lengths))
        if robust_length < float(config.truck_length_min):
            continue
        changed = 0
        for det in detections:
            if det.get("class_name") == "Car":
                det["class_name"] = "Truck"
                changed += 1
        if changed:
            relabeled_tracks += 1
            relabeled_detections += changed
            if len(examples) < 30:
                examples.append({
                    "track_id": (key[1] if key[0] == "track" else None),
                    "detections": changed,
                    "median_length": round(robust_length, 4),
                })

    for frame in output:
        frame["num_detections"] = len(frame.get("detections", []))
    return output, {
        "truck_length_min": float(config.truck_length_min),
        "tracks_checked": len(groups),
        "large_car_tracks_relabelled": relabeled_tracks,
        "large_car_detections_relabelled": relabeled_detections,
        "examples": examples,
        "mutated_fields": ["class_name"],
    }
