"""Final detection filtering and box-frame conversion for Step 5.

Point counts are measured against the original ``lidar_top`` point cloud and
the pre-conversion boxes.  Only detections with more than the configured point
threshold and tracks longer than the configured lifecycle threshold are kept.
Kept boxes are then converted to ``base_link`` using the clip calibration.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from filtering.hard_filters import count_points_in_boxes
from tracking import tracker_conservative as tracking


@dataclass(frozen=True)
class FinalFilterConfig:
    """Thresholds use an inclusive drop rule: ``<=`` is removed."""

    max_points_in_box: int = 10
    max_track_length: int = 3


def _load_lidar_xyz(clip: Path, frame_id: str) -> np.ndarray:
    path = Path(clip) / "lidar" / "lidar_top" / f"{frame_id}.bin"
    if not path.is_file():
        raise FileNotFoundError(f"lidar frame not found: {path}")
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4 != 0:
        raise ValueError(f"lidar frame is not xyzi float32: {path}")
    return values.reshape(-1, 4)[:, :3]


def _valid_box(det: Dict[str, Any]) -> bool:
    if not tracking.finite_box(det):
        return False
    return all(float(value) > 0.0 for value in det["box_lidar"][3:6])


def _point_counts(
        frames: Sequence[Dict[str, Any]],
        clip: Path,
) -> Dict[tuple[int, int], int]:
    counts: Dict[tuple[int, int], int] = {}
    for frame_index, frame in enumerate(frames):
        detections = frame.get("detections", [])
        if not detections:
            continue
        points = _load_lidar_xyz(clip, str(frame["frame_id"]))
        valid_indices = [
            index for index, det in enumerate(detections) if _valid_box(det)
        ]
        boxes = [detections[index]["box_lidar"] for index in valid_indices]
        values = count_points_in_boxes(points, boxes)
        for index, value in zip(valid_indices, values):
            counts[(frame_index, index)] = int(value)
    return counts


def _track_lifecycles(frames: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    by_track: Dict[int, set[int]] = defaultdict(set)
    for frame_index, frame in enumerate(frames):
        for det in frame.get("detections", []):
            track_id = det.get("track_id")
            if track_id is not None:
                by_track[int(track_id)].add(frame_index)
    return {track_id: len(indices) for track_id, indices in by_track.items()}


def apply_final_filter(
        frames: Sequence[Dict[str, Any]],
        clip: Path,
        coords: tracking.CoordinateProvider,
        config: FinalFilterConfig = FinalFilterConfig(),
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter final detections and convert every kept box to base_link.

    The input and output frame lists are not mutated.  Point filtering happens
    before lifecycle filtering; a track shortened by sparse observations is
    therefore evaluated using the observations that remain.
    """
    source = copy.deepcopy(list(frames))
    output = copy.deepcopy(list(frames))
    point_counts = _point_counts(source, Path(clip))

    point_removed = 0
    invalid_removed = 0
    point_drop_examples: List[Dict[str, Any]] = []
    for frame_index, frame in enumerate(output):
        kept = []
        for detection_index, det in enumerate(frame.get("detections", [])):
            if not _valid_box(det):
                invalid_removed += 1
                continue
            count = point_counts.get((frame_index, detection_index), 0)
            if count <= int(config.max_points_in_box):
                point_removed += 1
                if len(point_drop_examples) < 30:
                    point_drop_examples.append({
                        "frame_id": str(frame["frame_id"]),
                        "detection_index": detection_index,
                        "track_id": det.get("track_id"),
                        "points_in_box": count,
                    })
                continue
            kept.append(det)
        frame["detections"] = kept
        frame["num_detections"] = len(kept)

    lifecycles = _track_lifecycles(output)
    short_track_ids = {
        track_id for track_id, length in lifecycles.items()
        if length <= int(config.max_track_length)
    }
    short_track_removed = 0
    for frame in output:
        old = frame.get("detections", [])
        kept = [
            det for det in old
            if det.get("track_id") is None
            or int(det["track_id"]) not in short_track_ids
        ]
        short_track_removed += len(old) - len(kept)
        frame["detections"] = kept
        frame["num_detections"] = len(kept)

    converted = 0
    for frame in output:
        for det in frame.get("detections", []):
            det["box_lidar"] = tracking.box_lidar_to_base_link(
                det["box_lidar"], coords.base_from_lidar_top)
            det["box_frame"] = "base_link"
            converted += 1

    before_detections = sum(len(f.get("detections", [])) for f in source)
    after_detections = sum(len(f.get("detections", [])) for f in output)
    before_classes = Counter(
        str(det.get("class_name", ""))
        for frame in source for det in frame.get("detections", []))
    after_classes = Counter(
        str(det.get("class_name", ""))
        for frame in output for det in frame.get("detections", []))
    classes_removed = before_classes - after_classes
    return output, {
        "policy": {
            "pipeline_position": "after_truck_size_interpolation_overlap",
            "point_count_frame": "lidar_top",
            "input_box_frame": "lidar_top",
            "output_box_frame": "base_link",
            "drop_points_in_box_at_or_below": int(config.max_points_in_box),
            "drop_track_length_at_or_below": int(config.max_track_length),
            "class_filtering": "none",
            "point_cloud_conversion": "none",
            "mutated_fields": ["detections", "box_lidar", "box_frame"],
        },
        "before_detections": before_detections,
        "point_filter_removed": point_removed,
        "invalid_box_removed": invalid_removed,
        "short_track_ids": sorted(short_track_ids),
        "short_track_removed": short_track_removed,
        "after_detections": after_detections,
        "detections_removed": before_detections - after_detections,
        "boxes_converted": converted,
        "point_drop_examples": point_drop_examples,
        "track_lifecycles_after_point_filter": {
            str(track_id): length
            for track_id, length in sorted(lifecycles.items())
        },
        "classes_removed": dict(sorted(classes_removed.items())),
    }
