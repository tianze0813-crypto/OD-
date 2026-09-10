"""Agreed annotation filters and conservative same-center deduplication."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

from filtering import camera_visibility
from tracking import tracker_conservative as tracking


@dataclass(frozen=True)
class HardFilterConfig:
    score_threshold: float = 0.3
    range_front: float = 80.0
    range_rear: float = 20.0
    range_side: float = 40.0
    sparsity_max_points: int = 10
    visibility_min_ratio: float = 0.05
    visibility_occlusion_tolerance: float = 0.3
    pedestrian_max_distance: float = 20.0
    keep_classes: Tuple[str, ...] = (
        "Vehicle", "Car", "Truck", "Pedestrian", "Cyclist"
    )
    diagnostic_examples_per_reason: int = 30


def _load_lidar_xyz(clip: Path, frame_id: str) -> np.ndarray:
    path = clip / "lidar" / "lidar_top" / f"{frame_id}.bin"
    if not path.is_file():
        raise FileNotFoundError(f"lidar frame not found: {path}")
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4 != 0:
        raise ValueError(f"lidar frame is not xyzi float32: {path}")
    return values.reshape(-1, 4)[:, :3]


def count_points_in_boxes(points: np.ndarray,
                          boxes: Sequence[Sequence[float]]) -> np.ndarray:
    counts = np.zeros(len(boxes), dtype=np.int64)
    for index, box in enumerate(boxes):
        x, y, z, dx, dy, dz, yaw = (float(v) for v in box[:7])
        c, s = math.cos(-yaw), math.sin(-yaw)
        px = (points[:, 0] - x) * c - (points[:, 1] - y) * s
        py = (points[:, 0] - x) * s + (points[:, 1] - y) * c
        counts[index] = int(np.count_nonzero(
            (np.abs(px) <= dx / 2.0)
            & (np.abs(py) <= dy / 2.0)
            & (np.abs(points[:, 2] - z) <= dz / 2.0)
        ))
    return counts


def _valid_box(det: Dict[str, Any]) -> bool:
    if not tracking.finite_box(det):
        return False
    return all(float(value) > 0.0 for value in det["box_lidar"][3:6])


def _in_annotation_range(box: Sequence[float], config: HardFilterConfig) -> bool:
    # Detector local frame: x is lateral and forward is -y.
    x, y = float(box[0]), float(box[1])
    return (abs(x) <= config.range_side
            and -config.range_front <= y <= config.range_rear)


def apply_hard_filters(frames: List[Dict[str, Any]], clip: Path,
                       config: HardFilterConfig = HardFilterConfig()) -> Dict[str, Any]:
    """Apply the annotation contract before class pre-association.

    Every failing rule is recorded even when a detection fails more than one
    rule. ``primary_reason_counts`` counts actual removals without double
    counting. Kept detections retain their visibility tag for SUST export.
    """
    clip = Path(clip)
    cameras = camera_visibility.load_clip_cameras(clip)
    if not cameras:
        raise ValueError(f"no configured cameras found in {clip / 'transforms/calib.json'}")

    rule_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    before = sum(len(frame.get("detections", [])) for frame in frames)

    for frame in frames:
        frame_id = str(frame["frame_id"])
        detections = frame.get("detections", [])
        valid_indices = [i for i, det in enumerate(detections) if _valid_box(det)]
        valid_detections = [detections[i] for i in valid_indices]

        visibility_stats = camera_visibility.compute_frame_visibility(
            valid_detections, cameras, config.visibility_occlusion_tolerance)
        del visibility_stats
        points = _load_lidar_xyz(clip, frame_id)
        point_counts = count_points_in_boxes(
            points, [det["box_lidar"] for det in valid_detections])
        point_count_by_index = {
            source_index: int(point_count)
            for source_index, point_count in zip(valid_indices, point_counts)
        }

        kept = []
        for detection_index, det in enumerate(detections):
            reasons: List[str] = []
            if not _valid_box(det):
                reasons.append("invalid_box")
            else:
                box = det["box_lidar"]
                class_name = str(det.get("class_name", ""))
                if float(det.get("score", 0.0)) < config.score_threshold:
                    reasons.append("score")
                if class_name not in config.keep_classes:
                    reasons.append("class_whitelist")
                if not _in_annotation_range(box, config):
                    reasons.append("annotation_range")
                if (class_name == "Pedestrian"
                        and math.hypot(float(box[0]), float(box[1]))
                        > config.pedestrian_max_distance):
                    reasons.append("distant_pedestrian")
                if point_count_by_index[detection_index] <= config.sparsity_max_points:
                    reasons.append("sparse_points")
                if (float(det.get("visibility", {}).get("ratio", 0.0))
                        <= config.visibility_min_ratio):
                    reasons.append("low_visibility")

            if not reasons:
                kept.append(det)
                continue

            for reason in reasons:
                rule_counts[reason] += 1
                if len(examples[reason]) < config.diagnostic_examples_per_reason:
                    examples[reason].append({
                        "frame_id": frame_id,
                        "detection_index": detection_index,
                        "class_name": det.get("class_name"),
                        "score": round(float(det.get("score", 0.0)), 4),
                        "box_lidar": [round(float(x), 4)
                                      for x in det.get("box_lidar", [])[:7]],
                        "points_in_box": point_count_by_index.get(detection_index),
                        "visibility_ratio": det.get("visibility", {}).get("ratio"),
                        "all_reasons": reasons,
                    })
            primary_counts[reasons[0]] += 1
        frame["detections"] = kept
        frame["num_detections"] = len(kept)

    after = sum(len(frame.get("detections", [])) for frame in frames)
    return {
        "policy": {
            "score_threshold": config.score_threshold,
            "range_front": config.range_front,
            "range_rear": config.range_rear,
            "range_side": config.range_side,
            "sparsity_drop_at_or_below": config.sparsity_max_points,
            "visibility_drop_at_or_below": config.visibility_min_ratio,
            "pedestrian_max_distance": config.pedestrian_max_distance,
            "keep_classes": list(config.keep_classes),
        },
        "detections_before": before,
        "detections_after": after,
        "detections_removed": before - after,
        "rule_failure_counts": dict(sorted(rule_counts.items())),
        "primary_reason_counts": dict(sorted(primary_counts.items())),
        "removed_examples": dict(sorted(examples.items())),
    }


def deduplicate_same_center(
        frames: List[Dict[str, Any]], *, static_track_ids: Set[int] | None = None,
        center_gate: float = 0.35) -> Dict[str, Any]:
    """Keep one stable identity when multiple boxes occupy the same center.

    This intentionally does not implement broad IoU deduplication. Dense
    parking boxes may overlap legitimately; the agreed target is duplicate
    predictions centered on the same physical object.
    """
    static_track_ids = set(static_track_ids or ())
    lifecycle: Counter[int] = Counter()
    score_by_track: Dict[int, List[float]] = defaultdict(list)
    for frame in frames:
        seen = set()
        for det in frame.get("detections", []):
            tid = det.get("track_id")
            if tid is None:
                continue
            tid = int(tid)
            seen.add(tid)
            score_by_track[tid].append(float(det.get("score", 0.0)))
        lifecycle.update(seen)
    median_score = {
        tid: float(np.median(values)) for tid, values in score_by_track.items()
    }

    removals = []
    frames_affected = 0
    for frame in frames:
        detections = frame.get("detections", [])
        parent = list(range(len(detections)))

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            a, b = root(left), root(right)
            if a != b:
                parent[b] = a

        for left in range(len(detections)):
            a = detections[left]
            if not _valid_box(a):
                continue
            for right in range(left):
                b = detections[right]
                if not _valid_box(b):
                    continue
                distance = math.hypot(
                    float(a["box_lidar"][0]) - float(b["box_lidar"][0]),
                    float(a["box_lidar"][1]) - float(b["box_lidar"][1]))
                if distance <= center_gate:
                    union(left, right)

        groups: Dict[int, List[int]] = defaultdict(list)
        for index in range(len(detections)):
            groups[root(index)].append(index)
        remove_indices = set()
        for indices in groups.values():
            if len(indices) <= 1:
                continue

            def rank(index: int) -> Tuple[int, int, float, float]:
                det = detections[index]
                tid = int(det.get("track_id", -1))
                return (lifecycle[tid], int(tid in static_track_ids),
                        median_score.get(tid, 0.0), float(det.get("score", 0.0)))

            keep_index = max(indices, key=rank)
            for index in indices:
                if index == keep_index:
                    continue
                remove_indices.add(index)
                kept = detections[keep_index]
                removed = detections[index]
                removals.append({
                    "frame_id": str(frame["frame_id"]),
                    "kept_track_id": kept.get("track_id"),
                    "removed_track_id": removed.get("track_id"),
                    "kept_lifecycle": lifecycle[int(kept.get("track_id", -1))],
                    "removed_lifecycle": lifecycle[int(removed.get("track_id", -1))],
                    "center_distance": round(math.hypot(
                        float(kept["box_lidar"][0]) - float(removed["box_lidar"][0]),
                        float(kept["box_lidar"][1]) - float(removed["box_lidar"][1])), 4),
                })
        if remove_indices:
            frames_affected += 1
            frame["detections"] = [det for index, det in enumerate(detections)
                                   if index not in remove_indices]
            frame["num_detections"] = len(frame["detections"])
    return {
        "center_gate": center_gate,
        "frames_affected": frames_affected,
        "boxes_removed": len(removals),
        "removals": removals,
    }
