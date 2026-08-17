"""Conservative class pre-association and track-level class finalization."""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from tracking import tracker_conservative as tracking


VEHICLE_FAMILY = frozenset({"Vehicle", "Car", "Truck"})


@dataclass(frozen=True)
class ClassRefinementConfig:
    max_cross_class_gap_sec: float = 0.8
    max_cross_class_distance: float = 4.0
    max_cross_class_speed: float = 12.0
    max_relative_size_delta: float = 0.55
    uniqueness_margin: float = 0.35
    truck_length_min: float = 6.0
    pedestrian_length_max: float = 1.25
    pedestrian_width_max: float = 1.10
    cyclist_length_max: float = 3.5
    cyclist_width_max: float = 1.50


@dataclass
class _Item:
    frame_index: int
    timestamp: int
    detection: Dict[str, Any]
    original_class: str
    world: np.ndarray
    size: np.ndarray


def _relative_size_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b)) / max(
        float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0)


def _robust_dimensions(items: Sequence[_Item]) -> Tuple[float, float, float]:
    dimensions = np.asarray([
        [max(float(item.size[0]), float(item.size[1])),
         min(float(item.size[0]), float(item.size[1])),
         float(item.size[2])]
        for item in items
    ], dtype=np.float64)
    length, width, height = np.median(dimensions, axis=0)
    return float(length), float(width), float(height)


def _mixed_target(items: Sequence[_Item], config: ClassRefinementConfig) -> str:
    """Mixed chains start as Car, then size evidence applies agreed renames."""
    length, width, _height = _robust_dimensions(items)
    if length >= config.truck_length_min:
        return "Truck"
    if (length <= config.pedestrian_length_max
            and width <= config.pedestrian_width_max):
        return "Pedestrian"
    if (length < config.cyclist_length_max
            and width < config.cyclist_width_max):
        return "Cyclist"
    return "Car"


def _vehicle_target(items: Sequence[_Item], config: ClassRefinementConfig) -> str:
    """Classify a stable vehicle-family track from robust track dimensions."""
    length, width, _height = _robust_dimensions(items)
    if length >= config.truck_length_min:
        return "Truck"
    if (length < config.cyclist_length_max
            and width < config.cyclist_width_max):
        return "Cyclist"
    return "Car"


def _unique_nearest(candidate: _Item, reference: _Item,
                    frame_items: Sequence[_Item], margin: float) -> Tuple[bool, float, float | None]:
    distances = sorted(
        (float(np.linalg.norm((item.world - reference.world)[:2])), id(item.detection))
        for item in frame_items
    )
    if not distances:
        return False, math.inf, None
    candidate_distance = float(np.linalg.norm((candidate.world - reference.world)[:2]))
    if distances[0][1] != id(candidate.detection):
        return False, candidate_distance, distances[0][0]
    second = distances[1][0] if len(distances) > 1 else None
    if second is not None and second - candidate_distance < margin:
        return False, candidate_distance, second
    return True, candidate_distance, second


def _valid_cross_class_edge(left: _Item, right: _Item,
                            by_frame: Sequence[Sequence[_Item]],
                            config: ClassRefinementConfig) -> Tuple[bool, Dict[str, Any]]:
    dt = (right.timestamp - left.timestamp) / 1e9
    distance = float(np.linalg.norm((right.world - left.world)[:2]))
    distance_gate = min(
        config.max_cross_class_distance,
        0.45 + config.max_cross_class_speed * max(dt, 0.0))
    size_delta = _relative_size_delta(left.size, right.size)
    forward_unique, _forward, forward_second = _unique_nearest(
        right, left, by_frame[right.frame_index], config.uniqueness_margin)
    backward_unique, _backward, backward_second = _unique_nearest(
        left, right, by_frame[left.frame_index], config.uniqueness_margin)
    accepted = (
        0.0 < dt <= config.max_cross_class_gap_sec
        and distance <= distance_gate
        and size_delta <= config.max_relative_size_delta
        and forward_unique
        and backward_unique
    )
    return accepted, {
        "left_frame": left.frame_index,
        "right_frame": right.frame_index,
        "left_class": left.original_class,
        "right_class": right.original_class,
        "gap_sec": round(dt, 3),
        "distance": round(distance, 3),
        "distance_gate": round(distance_gate, 3),
        "relative_size_delta": round(size_delta, 3),
        "forward_second_distance": (None if forward_second is None
                                    else round(forward_second, 3)),
        "backward_second_distance": (None if backward_second is None
                                     else round(backward_second, 3)),
        "accepted": accepted,
    }


def preassociate_and_unify(
        frames: List[Dict[str, Any]], coords: tracking.CoordinateProvider,
        config: ClassRefinementConfig = ClassRefinementConfig()) -> Dict[str, Any]:
    """Run class-blind shadow association, then relabel only proven mixed parts.

    The shadow IDs are never copied to the real detections. Same-class portions
    remain connected; a cross-class boundary connects only when it is mutually
    unique and passes strict time, distance, and size gates.
    """
    shadow = copy.deepcopy(frames)
    for frame in shadow:
        for det in frame.get("detections", []):
            det.pop("track_id", None)
            det["class_name"] = ""
    tracker = tracking.ConservativeTracker(
        coords, min_static_hits=10 ** 9, dynamic_max_gap=1.8)
    shadow_output, shadow_diagnostics = tracker.process(
        shadow, enable_stitching=False)

    by_frame: List[List[_Item]] = []
    by_shadow_id: Dict[int, List[_Item]] = defaultdict(list)
    for frame_index, (real_frame, shadow_frame) in enumerate(zip(frames, shadow_output)):
        timestamp = int(real_frame["frame_id"])
        world_from_lidar = coords.world_from_lidar(timestamp)
        if world_from_lidar is None:
            raise ValueError(f"pose unavailable for class pre-association: {timestamp}")
        frame_items = []
        real_detections = real_frame.get("detections", [])
        shadow_detections = shadow_frame.get("detections", [])
        if len(real_detections) != len(shadow_detections):
            raise AssertionError("shadow association changed detection count")
        for real_det, shadow_det in zip(real_detections, shadow_detections):
            box = real_det["box_lidar"]
            item = _Item(
                frame_index=frame_index,
                timestamp=timestamp,
                detection=real_det,
                original_class=str(real_det.get("class_name", "")),
                world=tracking.center_world(box, world_from_lidar),
                size=np.asarray(box[3:6], dtype=np.float64),
            )
            frame_items.append(item)
            by_shadow_id[int(shadow_det["track_id"])].append(item)
        by_frame.append(frame_items)

    changes: Counter[Tuple[str, str]] = Counter()
    components_diag = []
    transitions_diag = []
    detections_changed = set()
    for shadow_id, track_items in by_shadow_id.items():
        items = sorted(track_items, key=lambda item: (item.timestamp, item.frame_index))
        parent = list(range(len(items)))

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            a, b = root(left), root(right)
            if a != b:
                parent[b] = a

        for index, (left, right) in enumerate(zip(items, items[1:])):
            if left.original_class == right.original_class:
                union(index, index + 1)
                continue
            accepted, detail = _valid_cross_class_edge(
                left, right, by_frame, config)
            detail["shadow_track_id"] = shadow_id
            transitions_diag.append(detail)
            if accepted:
                union(index, index + 1)

        components: Dict[int, List[_Item]] = defaultdict(list)
        for index, item in enumerate(items):
            components[root(index)].append(item)
        for component in components.values():
            class_counts = Counter(item.original_class for item in component)
            if len(class_counts) <= 1:
                continue
            target = _mixed_target(component, config)
            length, width, height = _robust_dimensions(component)
            changed = 0
            for item in component:
                if item.detection.get("class_name") != target:
                    changes[(str(item.detection.get("class_name", "")), target)] += 1
                    item.detection["class_name"] = target
                    detections_changed.add(id(item.detection))
                    changed += 1
            components_diag.append({
                "shadow_track_id": shadow_id,
                "frames": len({item.frame_index for item in component}),
                "detections": len(component),
                "original_classes": dict(sorted(class_counts.items())),
                "target_class": target,
                "detections_changed": changed,
                "median_dimensions": {
                    "length": round(length, 3),
                    "width": round(width, 3),
                    "height": round(height, 3),
                },
                "first_frame": min(item.frame_index for item in component),
                "last_frame": max(item.frame_index for item in component),
            })

    return {
        "policy": {
            "pure_class_chains": "unchanged",
            "mixed_chain_base": "Car",
            "truck_length_min": config.truck_length_min,
            "pedestrian_max": [config.pedestrian_length_max,
                               config.pedestrian_width_max],
            "cyclist_max": [config.cyclist_length_max,
                            config.cyclist_width_max],
            "cross_class_gap_sec": config.max_cross_class_gap_sec,
            "cross_class_distance_max": config.max_cross_class_distance,
            "cross_class_size_delta_max": config.max_relative_size_delta,
            "uniqueness_margin": config.uniqueness_margin,
        },
        "shadow_tracks": len(by_shadow_id),
        "mixed_components": len(components_diag),
        "detections_changed": len(detections_changed),
        "class_changes": {
            f"{source}->{target}": count
            for (source, target), count in sorted(changes.items())
        },
        "components": components_diag,
        "cross_class_transitions": transitions_diag,
        "shadow_tracker": {
            key: shadow_diagnostics.get(key)
            for key in ("tracks_total", "births", "matches", "rejections")
        },
    }


def finalize_track_classes(
        frames: List[Dict[str, Any]],
        config: ClassRefinementConfig = ClassRefinementConfig()) -> Dict[str, Any]:
    """Finalize mixed tracks and size-classify stable vehicle-family IDs."""
    tracks: Dict[int, List[_Item]] = defaultdict(list)
    for frame_index, frame in enumerate(frames):
        for det in frame.get("detections", []):
            tid = det.get("track_id")
            if tid is None:
                continue
            box = det["box_lidar"]
            tracks[int(tid)].append(_Item(
                frame_index=frame_index,
                timestamp=int(frame["frame_id"]), detection=det,
                original_class=str(det.get(
                    "_step2_source_class", det.get("class_name", ""))),
                world=np.zeros(3, dtype=np.float64),
                size=np.asarray(box[3:6], dtype=np.float64)))

    finalized = []
    changed = 0
    mixed_tracks = 0
    vehicle_family_tracks = 0
    for track_id, items in tracks.items():
        class_counts = Counter(item.original_class for item in items)
        original_classes = set(class_counts)
        if len(class_counts) == 1 and original_classes <= {
                "Car", "Truck", "Cyclist", "Pedestrian"}:
            # An explicit, track-consistent model class is stronger evidence
            # than a coarse size boundary. Size resolves only generic Vehicle
            # and mixed-class tracks.
            continue
        if len(class_counts) > 1:
            target = (_vehicle_target(items, config)
                      if original_classes <= VEHICLE_FAMILY
                      else _mixed_target(items, config))
            reason = "mixed_class_consistency"
            mixed_tracks += 1
        elif original_classes <= VEHICLE_FAMILY:
            target = _vehicle_target(items, config)
            reason = "vehicle_family_median_dimensions"
            vehicle_family_tracks += 1
        else:
            continue

        track_changed = 0
        for item in items:
            if item.detection.get("class_name") != target:
                item.detection["class_name"] = target
                changed += 1
                track_changed += 1
        length, width, height = _robust_dimensions(items)
        finalized.append({
            "track_id": track_id,
            "original_classes": dict(sorted(class_counts.items())),
            "target_class": target,
            "reason": reason,
            "detections": len(items),
            "detections_changed": track_changed,
            "median_dimensions": [round(length, 3), round(width, 3),
                                  round(height, 3)],
        })
    return {
        "tracks_checked": len(tracks),
        "mixed_tracks_unified": mixed_tracks,
        "vehicle_family_tracks_assessed": vehicle_family_tracks,
        "tracks_finalized": len(finalized),
        "detections_changed": changed,
        "policy": {
            "vehicle_family": sorted(VEHICLE_FAMILY),
            "truck_length_min": config.truck_length_min,
            "cyclist_max": [config.cyclist_length_max,
                             config.cyclist_width_max],
            "protected_pure_classes": [
                "Car", "Truck", "Pedestrian", "Cyclist",
            ],
        },
        "tracks": finalized,
    }
