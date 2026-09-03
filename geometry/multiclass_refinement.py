"""Class-specific refinements that run after public yaw in Step 3.

The identity tracker remains class-blind.  This module only consumes the
track IDs and canonical classes assigned by Step 2.5:

* Truck detections with strong same-frame BEV overlap, near-distance evidence,
  or high-overlap Car counterparts are treated as duplicate observations of
  one physical truck. IDs are remapped and duplicate boxes in a frame are
  removed; swallowed Car observations are promoted to Truck.
* Nonmotorized tracks receive one robust physical size for the whole track;
  large enclosing boxes are pulled toward the local trajectory and yaw is
  refreshed from the corrected centers.
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from geometry.yaw_static_direction import _world_yaw_to_local
from tracking import tracker_conservative as tracking


@dataclass(frozen=True)
class TruckOverlapConfig:
    high_iou: float = 0.50
    moderate_iou: float = 0.25
    max_center_distance: float = 1.10
    max_relative_size_delta: float = 0.45
    moderate_overlap_frames: int = 2
    cross_class_high_iou: float = 0.60
    cross_class_max_center_distance: float = 1.25
    cross_class_max_relative_size_delta: float = 0.55
    near_truck_max_distance: float = 0.80
    near_truck_max_relative_size_delta: float = 0.55
    near_truck_frames: int = 2


@dataclass(frozen=True)
class NonmotorizedSizeConfig:
    min_observations: int = 3
    max_relative_outlier: float = 0.45
    length_bounds: Tuple[float, float] = (0.75, 4.00)
    width_bounds: Tuple[float, float] = (0.45, 1.80)
    height_bounds: Tuple[float, float] = (0.90, 2.50)
    center_correction_max_distance: float = 1.50
    center_correction_max_gap: float = 1.20
    yaw_half_window: int = 2
    yaw_min_speed: float = 0.35
    yaw_min_displacement: float = 0.10


def _physical_size(box: Sequence[float]) -> np.ndarray:
    dx, dy, dz = (float(box[index]) for index in (3, 4, 5))
    return np.asarray([max(dx, dy), min(dx, dy), dz], dtype=np.float64)


def _relative_size_delta(left: Sequence[float], right: Sequence[float]) -> float:
    a = _physical_size(left)
    b = _physical_size(right)
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(a)),
                                               float(np.linalg.norm(b)), 1.0))


def _track_rank(values: Sequence[Mapping[str, Any]]) -> Tuple[int, float, int]:
    scores = [float(value.get("score", 0.0)) for value in values]
    track_id = int(values[0].get("track_id", 0))
    return (len(values), float(np.median(scores)) if scores else 0.0, -track_id)


def merge_overlapping_truck_tracks(
        frames: List[Dict[str, Any]], coords: tracking.CoordinateProvider,
        config: TruckOverlapConfig = TruckOverlapConfig()) -> Dict[str, Any]:
    """Merge duplicate vehicle tracklets using same-frame spatial evidence.

    Truck is the canonical representative for both supported merge routes:
    same-class Truck duplicates and high-overlap Truck/Car duplicates.  A
    Car track that is swallowed is relabelled as Truck and keeps observations
    from frames where the Truck track was absent; same-frame duplicates are
    then reduced to one box.  This preserves temporal coverage while keeping
    the annotation output free of overlapping duplicate boxes.
    """
    by_track: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for frame_index, frame in enumerate(frames):
        timestamp = int(frame["frame_id"])
        world_from_lidar = coords.world_from_lidar(timestamp)
        if world_from_lidar is None:
            continue
        for detection_index, det in enumerate(frame.get("detections", [])):
            if (det.get("track_id") is None
                    or det.get("class_name") not in {"Truck", "Car"}
                    or not tracking.finite_box(det)):
                continue
            box = det["box_lidar"]
            item = {
                "frame_index": frame_index,
                "detection_index": detection_index,
                "track_id": int(det["track_id"]),
                "det": det,
                "center": tracking.center_world(box, world_from_lidar),
                "yaw": tracking.yaw_world(float(box[6]), world_from_lidar),
                "size": np.asarray(box[3:6], dtype=np.float64),
                "class_name": str(det.get("class_name")),
            }
            by_track[item["track_id"]].append(item)
            by_frame[frame_index].append(item)

    track_values = {track_id: [item["det"] for item in items]
                    for track_id, items in by_track.items()}
    ranks = {track_id: _track_rank(values)
             for track_id, values in track_values.items()}
    track_classes = {
        track_id: Counter(item["class_name"] for item in items).most_common(1)[0][0]
        for track_id, items in by_track.items()
    }

    parent = {track_id: track_id for track_id in by_track}

    def root(track_id: int) -> int:
        value = int(track_id)
        path = []
        while parent.get(value, value) != value:
            path.append(value)
            value = parent[value]
        for node in path:
            parent[node] = value
        return value

    def union(left: int, right: int) -> None:
        a, b = root(left), root(right)
        if a == b:
            return
        if ranks[a] < ranks[b]:
            a, b = b, a
        parent[b] = a

    pair_evidence: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    pair_rules: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    for frame_index, items in by_frame.items():
        for index, left in enumerate(items):
            for right in items[:index]:
                if left["track_id"] == right["track_id"]:
                    continue
                classes = {left["class_name"], right["class_name"]}
                if classes not in ({"Truck"}, {"Truck", "Car"}):
                    continue
                center_distance = float(np.linalg.norm(
                    left["center"][:2] - right["center"][:2]))
                iou = tracking.bev_iou(
                    left["center"], left["size"], left["yaw"],
                    right["center"], right["size"], right["yaw"])
                size_delta = _relative_size_delta(
                    left["det"]["box_lidar"], right["det"]["box_lidar"])
                pair = tuple(sorted((left["track_id"], right["track_id"])))
                evidence = {
                    "frame_index": frame_index,
                    "track_ids": list(pair),
                    "classes": [left["class_name"], right["class_name"]],
                    "bev_iou": round(float(iou), 4),
                    "center_distance": round(center_distance, 4),
                    "relative_size_delta": round(size_delta, 4),
                }
                if classes == {"Truck", "Car"}:
                    if (center_distance <= config.cross_class_max_center_distance
                            and size_delta <= config.cross_class_max_relative_size_delta
                            and iou >= config.cross_class_high_iou):
                        pair_evidence[pair].append(evidence)
                        pair_rules[pair].add("truck_car_high_iou")
                    continue
                overlap = (center_distance <= config.max_center_distance
                            and size_delta <= config.max_relative_size_delta
                            and iou >= config.moderate_iou)
                near = (center_distance <= config.near_truck_max_distance
                        and size_delta <= config.near_truck_max_relative_size_delta)
                if overlap or near:
                    pair_evidence[pair].append(evidence)
                    if overlap:
                        if iou >= config.high_iou:
                            pair_rules[pair].add("truck_overlap_high_iou")
                        else:
                            pair_rules[pair].add("truck_overlap_multiframe")
                    if near:
                        pair_rules[pair].add("truck_near_distance")

    accepted: List[Tuple[Tuple[int, int], List[Dict[str, Any]]]] = []
    for pair, evidence in sorted(pair_evidence.items()):
        classes = {track_classes.get(track_id) for track_id in pair}
        if classes == {"Truck", "Car"}:
            accepted_by_rule = "truck_car_high_iou" in pair_rules[pair]
        else:
            strong = [item for item in evidence
                      if item["bev_iou"] >= config.high_iou]
            moderate = [item for item in evidence
                        if item["bev_iou"] >= config.moderate_iou
                        and item["center_distance"] <= config.max_center_distance
                        and item["relative_size_delta"] <= config.max_relative_size_delta]
            near = [item for item in evidence
                    if item["center_distance"] <= config.near_truck_max_distance
                    and item["relative_size_delta"] <= config.near_truck_max_relative_size_delta]
            accepted_by_rule = (
                bool(strong)
                or len(moderate) >= int(config.moderate_overlap_frames)
                or len(near) >= int(config.near_truck_frames))
        if accepted_by_rule:
            accepted.append((pair, evidence))
            union(*pair)

    components: Dict[int, List[int]] = defaultdict(list)
    for track_id in by_track:
        components[root(track_id)].append(track_id)
    representative: Dict[int, int] = {}
    for members in components.values():
        trucks = [track_id for track_id in members
                  if track_classes.get(track_id) == "Truck"]
        if not trucks:
            continue
        keep = max(trucks, key=lambda track_id: ranks[track_id])
        for track_id in members:
            representative[track_id] = keep

    remap = {
        track_id: keep for track_id, keep in sorted(representative.items())
        if track_id != keep
    }
    class_converted = 0
    original_class_by_detection = {
        id(det): str(det.get("class_name"))
        for frame in frames for det in frame.get("detections", [])
    }
    for frame in frames:
        for det in frame.get("detections", []):
            track_id = det.get("track_id")
            if track_id is None or int(track_id) not in representative:
                continue
            keep = representative[int(track_id)]
            if keep == int(track_id) and det.get("class_name") != "Car":
                continue
            det["track_id"] = keep
            if det.get("class_name") == "Car":
                det["class_name"] = "Truck"
                class_converted += 1

    representative_size_values: Dict[int, List[np.ndarray]] = defaultdict(list)
    for track_id, items in by_track.items():
        mapped = representative.get(track_id, track_id)
        representative_size_values[mapped].extend(
            item["size"] for item in items)
    representative_size = {
        track_id: np.median(np.asarray(values), axis=0)
        for track_id, values in representative_size_values.items()
    }

    removed: List[Dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        groups: Dict[int, List[int]] = defaultdict(list)
        for index, det in enumerate(frame.get("detections", [])):
            if det.get("class_name") == "Truck" and det.get("track_id") is not None:
                groups[int(det["track_id"])].append(index)
        remove_indices = set()
        for track_id, indices in groups.items():
            if len(indices) <= 1:
                continue

            def rank(index: int) -> Tuple[float, float, float, float]:
                det = frame["detections"][index]
                size_error = float(np.linalg.norm(
                    _physical_size(det["box_lidar"])
                    - representative_size.get(track_id,
                                              _physical_size(det["box_lidar"]))))
                source_class = original_class_by_detection.get(id(det), "Truck")
                class_priority = 1.0 if source_class == "Truck" else 0.0
                return (class_priority, float(det.get("score", 0.0)), -size_error,
                        -float(index))

            keep_index = max(indices, key=rank)
            for index in indices:
                if index == keep_index:
                    continue
                remove_indices.add(index)
                removed_det = frame["detections"][index]
                kept_det = frame["detections"][keep_index]
                removed.append({
                    "frame_index": frame_index,
                    "track_id": track_id,
                    "kept_score": round(float(kept_det.get("score", 0.0)), 4),
                    "removed_score": round(float(removed_det.get("score", 0.0)), 4),
                })
        if remove_indices:
            frame["detections"] = [
                det for index, det in enumerate(frame["detections"])
                if index not in remove_indices]
            frame["num_detections"] = len(frame["detections"])

    return {
        "policy": {
            "class": "Truck/Car",
            "high_iou": config.high_iou,
            "moderate_iou": config.moderate_iou,
            "max_center_distance": config.max_center_distance,
            "max_relative_size_delta": config.max_relative_size_delta,
            "moderate_overlap_frames": config.moderate_overlap_frames,
            "cross_class_high_iou": config.cross_class_high_iou,
            "cross_class_max_center_distance": config.cross_class_max_center_distance,
            "cross_class_max_relative_size_delta": config.cross_class_max_relative_size_delta,
            "near_truck_max_distance": config.near_truck_max_distance,
            "near_truck_max_relative_size_delta": config.near_truck_max_relative_size_delta,
            "near_truck_frames": config.near_truck_frames,
            "mutated_fields": ["track_id", "class_name", "box_presence"],
        },
        "tracks_before": len(by_track),
        "candidate_pairs": len(pair_evidence),
        "accepted_pairs": len(accepted),
        "id_remaps": [
            {"from_track_id": source, "to_track_id": target}
            for source, target in sorted(remap.items())],
        "class_id_remaps": [
            {"from_track_id": source, "to_track_id": target,
             "from_class": track_classes.get(source), "to_class": "Truck"}
            for source, target in sorted(remap.items())],
        "overlap_pairs": [
            {"track_ids": list(pair), "rules": sorted(pair_rules[pair]),
             "evidence": evidence}
            for pair, evidence in accepted],
        "cross_class_pairs": [
            {"track_ids": list(pair), "rules": sorted(pair_rules[pair]),
             "evidence": evidence}
            for pair, evidence in accepted
            if {track_classes.get(track_id) for track_id in pair} == {"Truck", "Car"}],
        "near_truck_pairs": [
            {"track_ids": list(pair), "rules": sorted(pair_rules[pair]),
             "evidence": evidence}
            for pair, evidence in accepted
            if ("truck_near_distance" in pair_rules[pair]
                and {track_classes.get(track_id) for track_id in pair} == {"Truck"})],
        "class_converted_boxes": class_converted,
        "boxes_removed": len(removed),
        "removed_duplicates": removed,
    }


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = np.maximum(weights[order], 1e-6)
    cutoff = float(np.sum(sorted_weights)) * 0.5
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def unify_nonmotorized_track_sizes(
        frames: List[Dict[str, Any]], coords: tracking.CoordinateProvider,
        config: NonmotorizedSizeConfig = NonmotorizedSizeConfig()) -> Dict[str, Any]:
    """Stabilize NMV dimensions, large-box centers, and trajectory yaw."""
    tracks: Dict[int, List[MutableMapping[str, Any]]] = defaultdict(list)
    for frame_index, frame in enumerate(frames):
        timestamp = int(frame["frame_id"])
        world_from_lidar = coords.world_from_lidar(timestamp)
        if world_from_lidar is None:
            continue
        for detection_index, det in enumerate(frame.get("detections", [])):
            if (det.get("track_id") is None
                    or det.get("class_name") != "Nonmotorized_vehicle"
                    or not tracking.finite_box(det)):
                continue
            tracks[int(det["track_id"])].append({
                "frame_index": frame_index,
                "detection_index": detection_index,
                "frame_id": timestamp,
                "det": det,
                "world": tracking.center_world(det["box_lidar"], world_from_lidar),
                "size": _physical_size(det["box_lidar"]),
            })

    details: List[Dict[str, Any]] = []
    changed = 0
    center_changed = 0
    yaw_changed = 0
    skipped = 0
    for track_id, items in sorted(tracks.items()):
        items.sort(key=lambda item: int(item["frame_id"]))
        if len(items) < int(config.min_observations):
            skipped += 1
            continue
        values = np.asarray([item["size"] for item in items], dtype=np.float64)
        weights = np.asarray([
            max(float(item["det"].get("score", 0.0)), 0.05)
            for item in items], dtype=np.float64)
        original_median = np.median(values, axis=0)
        relative = np.abs(values - original_median) / np.maximum(original_median, 0.50)
        inlier_mask = np.all(relative <= config.max_relative_outlier, axis=1)
        min_inliers = max(2, int(math.ceil(len(items) * 0.40)))
        if int(np.count_nonzero(inlier_mask)) < min_inliers:
            inlier_mask = np.ones(len(items), dtype=bool)
        fixed = np.asarray([
            _weighted_median(values[inlier_mask, axis], weights[inlier_mask])
            for axis in range(3)], dtype=np.float64)
        bounds = (config.length_bounds, config.width_bounds,
                  config.height_bounds)
        fixed = np.asarray([
            np.clip(fixed[axis], bounds[axis][0], bounds[axis][1])
            for axis in range(3)], dtype=np.float64)

        # Large footprint boxes are enclosing detections. Small boxes retain
        # their measured centers and are completed with the track consensus.
        relative_to_fixed = np.abs(values - fixed) / np.maximum(fixed, 0.50)
        large_mask = np.any(relative_to_fixed > config.max_relative_outlier,
                            axis=1)
        large_mask &= (
            np.prod(values[:, :2], axis=1)
            >= np.prod(fixed[:2]) * (1.0 + config.max_relative_outlier))
        support_indices = [index for index, is_large in enumerate(large_mask)
                           if not is_large]
        if len(support_indices) < 2:
            support_indices = list(range(len(items)))

        def trajectory_target(index: int) -> np.ndarray | None:
            if not large_mask[index] or len(support_indices) < 2:
                return None
            timestamp = int(items[index]["frame_id"])
            # Use nearby centers from the complete track as support. A run of
            # large boxes can still have valid centers, and excluding that run
            # would leave endpoint gaps with no correction evidence.
            nearby_support = [candidate for candidate in range(len(items))
                              if candidate != index]
            before = [candidate for candidate in nearby_support
                      if int(items[candidate]["frame_id"]) < timestamp]
            after = [candidate for candidate in nearby_support
                     if int(items[candidate]["frame_id"]) > timestamp]
            left = max(before, key=lambda candidate: int(items[candidate]["frame_id"]),
                       default=None)
            right = min(after, key=lambda candidate: int(items[candidate]["frame_id"]),
                        default=None)
            if left is not None and right is not None:
                t_left = int(items[left]["frame_id"])
                t_right = int(items[right]["frame_id"])
                gap = (t_right - t_left) / 1e9
                if 1e-6 < gap <= config.center_correction_max_gap:
                    alpha = (timestamp - t_left) / float(t_right - t_left)
                    return ((1.0 - alpha) * items[left]["world"]
                            + alpha * items[right]["world"])
            # Without observations on both sides, a nearest-neighbor pull is
            # ambiguous at a track endpoint. Keep the measured center there.
            return None

        track_changed = 0
        track_center_changed = 0
        track_yaw_changed = 0
        axis_swaps = 0
        for index, item in enumerate(items):
            box = item["det"]["box_lidar"]
            old = np.asarray(box[3:6], dtype=np.float64)
            if old[0] >= old[1]:
                new = fixed
            else:
                new = fixed[[1, 0, 2]]
                axis_swaps += 1
            if not np.allclose(old, new, atol=1e-9, rtol=0.0):
                track_changed += 1
                changed += 1
                box[3:6] = [float(value) for value in new]

            target = trajectory_target(index)
            if target is None:
                continue
            delta = target - np.asarray(item["world"], dtype=np.float64)
            distance = float(np.linalg.norm(delta))
            if distance > config.center_correction_max_distance:
                delta *= config.center_correction_max_distance / distance
            if float(np.linalg.norm(delta)) <= 1e-6:
                continue
            target_world = np.asarray(item["world"], dtype=np.float64) + delta
            lidar_from_world = coords.lidar_from_world(int(item["frame_id"]))
            if lidar_from_world is None:
                continue
            target_local = (lidar_from_world @ np.r_[target_world, 1.0])[:3]
            box[:3] = [float(value) for value in target_local]
            item["world"] = target_world
            track_center_changed += 1
            center_changed += 1

        # Recompute headings only after all center corrections are applied.
        yaw_updates = _nonmotorized_yaw_updates(items, config)
        for index, target_world_yaw in yaw_updates.items():
            item = items[index]
            world_from_lidar = coords.world_from_lidar(int(item["frame_id"]))
            if world_from_lidar is None:
                continue
            box = item["det"]["box_lidar"]
            target_local_yaw = _world_yaw_to_local(target_world_yaw,
                                                   world_from_lidar)
            if abs(float(box[6]) - target_local_yaw) <= 1e-9:
                continue
            box[6] = float(target_local_yaw)
            track_yaw_changed += 1
            yaw_changed += 1

        details.append({
            "track_id": track_id,
            "observations": len(items),
            "inlier_observations": int(np.count_nonzero(inlier_mask)),
            "outlier_observations": int(len(items) - np.count_nonzero(inlier_mask)),
            "large_boxes": int(np.count_nonzero(large_mask)),
            "small_or_inlier_boxes": int(len(items) - np.count_nonzero(large_mask)),
            "original_median_physical_size": [round(float(value), 4)
                                                for value in original_median],
            "fixed_physical_size": [round(float(value), 4)
                                     for value in fixed],
            "boxes_changed": track_changed,
            "centers_changed": track_center_changed,
            "yaw_boxes_updated": track_yaw_changed,
            "axis_swaps": axis_swaps,
        })

    return {
        "policy": {
            "class": "Nonmotorized_vehicle",
            "min_observations": config.min_observations,
            "max_relative_outlier": config.max_relative_outlier,
            "bounds_physical_lwh": [list(config.length_bounds),
                                     list(config.width_bounds),
                                     list(config.height_bounds)],
            "center_correction_max_distance": config.center_correction_max_distance,
            "center_correction_max_gap": config.center_correction_max_gap,
            "yaw_half_window": config.yaw_half_window,
            "yaw_min_speed": config.yaw_min_speed,
            "yaw_min_displacement": config.yaw_min_displacement,
            "mutated_fields": ["box_lidar[:7]"],
        },
        "tracks": len(tracks),
        "tracks_refined": len(details),
        "tracks_skipped": skipped,
        "boxes_changed": changed,
        "centers_changed": center_changed,
        "yaw_boxes_updated": yaw_changed,
        "details": details,
    }


def _nonmotorized_yaw_updates(
        items: Sequence[Mapping[str, Any]],
        config: NonmotorizedSizeConfig) -> Dict[int, float]:
    """Return local-window world headings for sufficiently moving NMV boxes."""
    if len(items) < 2:
        return {}
    updates: Dict[int, float] = {}
    for index, item in enumerate(items):
        lo = max(0, index - int(config.yaw_half_window))
        hi = min(len(items), index + int(config.yaw_half_window) + 1)
        window = list(items[lo:hi])
        if len(window) < 2:
            continue
        times = np.asarray([
            (int(candidate["frame_id"]) - int(item["frame_id"])) / 1e9
            for candidate in window
        ], dtype=np.float64)
        if np.ptp(times) <= 1e-3:
            continue
        if np.any(np.diff(times) > config.center_correction_max_gap):
            continue
        times -= float(np.mean(times))
        points = np.asarray([candidate["world"][:2] for candidate in window],
                            dtype=np.float64)
        centered = points - np.mean(points, axis=0)
        denominator = float(np.dot(times, times))
        if denominator <= 1e-9:
            continue
        velocity = times @ centered / denominator
        speed = float(np.linalg.norm(velocity))
        displacement = float(np.linalg.norm(points[-1] - points[0]))
        if speed < config.yaw_min_speed or displacement < config.yaw_min_displacement:
            continue
        heading = math.atan2(float(velocity[1]), float(velocity[0]))
        box = item["det"]["box_lidar"]
        if float(box[3]) < float(box[4]):
            heading -= math.pi / 2.0
        updates[index] = heading
    return updates


def verify_multiclass_refinement(
        before: Sequence[Mapping[str, Any]],
        after: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Verify the class-specific mutation contract.

    Truck refinement may remap IDs, promote swallowed Car observations to
    Truck, and remove duplicate boxes, but every surviving vehicle detection
    keeps its score, complete box geometry, and non-track metadata.
    Nonmotorized refinement may replace ``box_lidar[:7]`` while preserving its
    class, ID, and all non-geometry metadata. Bus and Pedestrian are strict
    pass-throughs; Car is unchanged unless swallowed by a Truck merge.
    """
    if len(before) != len(after):
        raise AssertionError("multiclass refinement changed frame count")

    def without_track(det: Mapping[str, Any]) -> Dict[str, Any]:
        value = copy.deepcopy(dict(det))
        value.pop("track_id", None)
        return value

    checked = 0
    truck_removed = 0
    vehicle_converted = 0
    nmv_boxes = 0
    for left_frame, right_frame in zip(before, after):
        if left_frame.get("frame_id") != right_frame.get("frame_id"):
            raise AssertionError("multiclass refinement changed frame order")
        left_dets = list(left_frame.get("detections", []))
        right_dets = list(right_frame.get("detections", []))

        # Bus/Pedestrian remain strict pass-throughs. Car may be swallowed by
        # a Truck component, so it is checked below with the vehicle pool.
        left_protected = [
            det for det in left_dets
            if det.get("class_name") in {"Bus", "Pedestrian"}
        ]
        right_protected = [
            det for det in right_dets
            if det.get("class_name") in {"Bus", "Pedestrian"}
        ]
        if left_protected != right_protected:
            raise AssertionError(
                "Bus/Pedestrian refinement changed detection")

        left_nmv = [det for det in left_dets
                    if det.get("class_name") == "Nonmotorized_vehicle"]
        right_nmv = [det for det in right_dets
                     if det.get("class_name") == "Nonmotorized_vehicle"]
        if len(left_nmv) != len(right_nmv):
            raise AssertionError("Nonmotorized refinement changed box presence")
        for left, right in zip(left_nmv, right_nmv):
            if left.get("track_id") != right.get("track_id"):
                raise AssertionError("Nonmotorized refinement changed track_id")
            if left.get("class_name") != right.get("class_name"):
                raise AssertionError("Nonmotorized refinement changed class_name")
            left_copy = without_track(left)
            right_copy = without_track(right)
            left_box = list(left_copy.pop("box_lidar"))
            right_box = list(right_copy.pop("box_lidar"))
            if left_copy != right_copy or len(left_box) != len(right_box):
                raise AssertionError(
                    "Nonmotorized refinement changed a protected field")
            if left_box[7:] != right_box[7:]:
                raise AssertionError(
                    "Nonmotorized refinement changed extra box fields")
            nmv_boxes += 1
            checked += 1

        def vehicle_signature(det: Mapping[str, Any]) -> Dict[str, Any]:
            value = without_track(det)
            value.pop("class_name", None)
            return value

        left_vehicle = [det for det in left_dets
                        if det.get("class_name") in {"Car", "Truck"}]
        remaining_vehicle = [
            {"class_name": str(det.get("class_name")),
             "signature": vehicle_signature(det)}
            for det in left_vehicle
        ]
        for right in (det for det in right_dets
                      if det.get("class_name") in {"Car", "Truck"}):
            candidate = vehicle_signature(right)
            allowed = [index for index, item in enumerate(remaining_vehicle)
                       if item["signature"] == candidate
                       and (right.get("class_name") == item["class_name"]
                            or (right.get("class_name") == "Truck"
                                and item["class_name"] == "Car"))]
            if not allowed:
                raise AssertionError(
                    "Truck/Car refinement changed a protected detection field")
            # Prefer a native Truck source when both classes have the same box.
            keep_index = next((index for index in allowed
                               if remaining_vehicle[index]["class_name"] == "Truck"),
                              allowed[0])
            source_class = remaining_vehicle.pop(keep_index)["class_name"]
            if source_class == "Car" and right.get("class_name") == "Truck":
                vehicle_converted += 1
            checked += 1
        truck_removed += sum(item["class_name"] == "Truck"
                             for item in remaining_vehicle)

    return {
        "passed": True,
        "detections_checked": checked,
        "truck_boxes_removed": truck_removed,
        "vehicle_boxes_converted_car_to_truck": vehicle_converted,
        "nonmotorized_boxes_checked": nmv_boxes,
        "policy": "Truck ID/duplicate merge plus Nonmotorized size/center/yaw",
    }
