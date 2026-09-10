#!/usr/bin/env python3
"""Merge main-branch Car detections with expD non-Car detections."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Sequence

from tracking import tracker_conservative as tracking


NON_CAR_CLASSES = frozenset({
    "Truck", "Bus", "Pedestrian", "Nonmotorized_vehicle",
})


def _label_to_box(label: Mapping[str, Any]) -> List[float]:
    """Rebuild box_lidar [x,y,z,dx,dy,dz,yaw] from a SUST label (base_link)."""
    p = label["psr"]["position"]
    s = label["psr"]["scale"]
    rz = label["psr"]["rotation"]["z"]
    return [float(p["x"]), float(p["y"]), float(p["z"]),
            float(s["x"]), float(s["y"]), float(s["z"]), float(rz)]


def _absorb_high_iou_cross_class(
        main_labels: Mapping[str, Sequence[Mapping[str, Any]]],
        expd_frames: Sequence[Mapping[str, Any]],
        high_iou: float = 0.8,
) -> tuple[Dict[str, List[Mapping[str, Any]]], List[Mapping[str, Any]],
           Dict[str, Any]]:
    """Absorb same-frame Car vs Truck/Bus with BEV IoU >= high_iou.

    Both inputs are already in base_link.  For every frame a Car label and a
    Truck/Bus detection are merged into one object; the longer-lifecycle side
    (more observed frames) is kept and the other is dropped across all frames.
    """
    expd_by_frame = _frame_index(expd_frames)
    car_life: Counter[str] = Counter()
    expd_life: Counter[str] = Counter()
    for labels in main_labels.values():
        for label in labels:
            if str(label.get("obj_type")) == "Car":
                car_life[str(label.get("obj_id"))] += 1
    for frame in expd_frames:
        for det in frame.get("detections", []):
            if (tracking.canonical_class_name(det.get("class_name", ""))
                    in {"Truck", "Bus"} and det.get("track_id") is not None):
                expd_life[str(det.get("track_id"))] += 1

    pair_frames: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for frame_id, expd_frame in expd_by_frame.items():
        car_boxes: List[Tuple[str, List[float]]] = []
        for label in main_labels.get(frame_id, []):
            if str(label.get("obj_type")) != "Car":
                continue
            try:
                car_boxes.append((str(label.get("obj_id")), _label_to_box(label)))
            except (KeyError, TypeError, ValueError):
                continue
        for det in expd_frame.get("detections", []):
            canonical = tracking.canonical_class_name(det.get("class_name", ""))
            if canonical not in {"Truck", "Bus"} or det.get("track_id") is None:
                continue
            expd_box = det["box_lidar"]
            for car_id, car_box in car_boxes:
                iou = tracking.bev_iou(
                    car_box[:3], car_box[3:6], car_box[6],
                    expd_box[:3], expd_box[3:6], expd_box[6])
                if iou >= high_iou:
                    pair_frames[(car_id, str(det["track_id"]))].append(frame_id)

    parent: Dict[str, str] = {}

    def root(node: str) -> str:
        value = node
        while parent.get(value, value) != value:
            value = parent[value]
        while parent.get(node, node) != node:
            parent[node], node = value, parent[node]
        return value

    def union(left: str, right: str) -> None:
        a, b = root(left), root(right)
        if a != b:
            parent[b] = a

    for car_id, track_id in pair_frames:
        a, b = "car:" + car_id, "expd:" + track_id
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    components: Dict[str, List[str]] = defaultdict(list)
    for node in list(parent):
        components[root(node)].append(node)

    life = {"car:" + key: value for key, value in car_life.items()}
    life.update({"expd:" + key: value for key, value in expd_life.items()})
    absorbed_car: set[str] = set()
    absorbed_expd: set[str] = set()
    absorbed_pairs: List[Dict[str, str]] = []
    for members in components.values():
        if len(members) <= 1:
            continue
        best = max(members, key=lambda node: (life.get(node, 0), node))
        for member in members:
            if member == best:
                continue
            kind, _, identifier = member.partition(":")
            if kind == "car":
                absorbed_car.add(identifier)
            else:
                absorbed_expd.add(identifier)
            absorbed_pairs.append({"absorbed": member, "kept": best})

    filtered_main = {
        frame_id: [
            label for label in labels
            if str(label.get("obj_id")) not in absorbed_car
        ]
        for frame_id, labels in main_labels.items()
    }
    filtered_expd: List[Mapping[str, Any]] = []
    for frame in expd_frames:
        kept = [
            det for det in frame.get("detections", [])
            if not (tracking.canonical_class_name(det.get("class_name", ""))
                    in {"Truck", "Bus"}
                    and str(det.get("track_id")) in absorbed_expd)
        ]
        out = dict(frame)
        out["detections"] = kept
        out["num_detections"] = len(kept)
        filtered_expd.append(out)

    return filtered_main, filtered_expd, {
        "pipeline": "hybrid_high_iou_absorb",
        "threshold": high_iou,
        "candidate_pairs": len(pair_frames),
        "absorbed_car_obj_ids": sorted(absorbed_car),
        "absorbed_expd_track_ids": sorted(absorbed_expd),
        "car_absorbed": len(absorbed_car),
        "expd_absorbed": len(absorbed_expd),
        "absorbed_pairs": absorbed_pairs,
    }


def _remove_frame_level_car_overlap(
        frames: List[Dict[str, Any]],
        *,
        threshold: float = 0.5,
) -> Dict[str, Any]:
    """Remove a Car label only in a frame where Truck/Bus covers it.

    The overlap metric is BEV intersection area divided by the Car box area.
    Only the Car label in that single frame is removed; Truck/Bus labels and
    the same Car track in all other frames are kept.  This runs after the
    existing track-level high-IoU absorption policy and does not change it.
    """
    removed: List[Dict[str, Any]] = []
    pairs_tested = 0
    for frame in frames:
        labels = frame.get("labels", [])
        if not labels:
            continue
        blockers: List[tuple[Mapping[str, Any], List[float]]] = []
        for label in labels:
            if str(label.get("obj_type")) not in ("Truck", "Bus"):
                continue
            try:
                blockers.append((label, _label_to_box(label)))
            except (KeyError, TypeError, ValueError):
                continue
        if not blockers:
            continue

        kept: List[Any] = []
        for label in labels:
            if str(label.get("obj_type")) != "Car":
                kept.append(label)
                continue
            try:
                car_box = _label_to_box(label)
                car_poly = tracking.rectangle_corners(
                    car_box[:2], car_box[3:5], car_box[6])
                car_area = tracking.polygon_area(car_poly)
            except (KeyError, TypeError, ValueError):
                kept.append(label)
                continue

            hit: tuple[Mapping[str, Any], float] | None = None
            if car_area > 1e-9:
                for blocker, blocker_box in blockers:
                    blocker_poly = tracking.rectangle_corners(
                        blocker_box[:2], blocker_box[3:5], blocker_box[6])
                    intersection = tracking.polygon_area(
                        tracking.convex_intersection(car_poly, blocker_poly))
                    pairs_tested += 1
                    ratio = float(intersection) / float(car_area)
                    if ratio > float(threshold):
                        hit = (blocker, ratio)
                        break
            if hit is None:
                kept.append(label)
                continue
            blocker, ratio = hit
            removed.append({
                "frame_id": str(frame.get("frame_id", "")),
                "car_obj_id": str(label.get("obj_id")),
                "blocker_obj_id": str(blocker.get("obj_id")),
                "blocker_type": str(blocker.get("obj_type")),
                "intersection_over_car": round(float(ratio), 4),
            })
        frame["labels"] = kept

    return {
        "policy": {
            "metric": "bev_intersection_area_over_car_area",
            "threshold": float(threshold),
            "scope": "single frame only",
            "kept": ["Car in other frames", "Truck", "Bus"],
        },
        "pairs_tested": pairs_tested,
        "car_labels_removed": len(removed),
        "removed": removed,
    }


def merge_label_frames(
        main_labels: Mapping[str, Sequence[Mapping[str, Any]]],
        expd_frames: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Merge already-exported main Car labels with expD frame detections."""
    filtered_main, filtered_expd, absorb_stats = _absorb_high_iou_cross_class(
        main_labels, expd_frames)
    expd_by_frame = _frame_index(filtered_expd)
    if set(filtered_main) != set(expd_by_frame):
        raise ValueError(
            "hybrid frame IDs differ: "
            f"missing_main={sorted(set(expd_by_frame) - set(filtered_main))}, "
            f"missing_expD={sorted(set(filtered_main) - set(expd_by_frame))}")

    output: List[Dict[str, Any]] = []
    used_ids = {
        str(label.get("obj_id"))
        for labels in filtered_main.values() for label in labels
        if label.get("obj_id") is not None
    }
    used_ints: set[int] = set()
    for value in used_ids:
        try:
            used_ints.add(int(value))
        except ValueError:
            pass

    source_ids = sorted({
        int(det.get("track_id"))
        for frame in filtered_expd for det in frame.get("detections", [])
        if det.get("track_id") is not None
    })
    id_map: Dict[str, str] = {}
    next_id = max(used_ints, default=0) + 1
    for source_id in source_ids:
        candidate = str(source_id)
        if candidate in used_ids:
            # Colliding expD tracks get a fresh plain id (no expd_ prefix).
            while next_id in used_ints:
                next_id += 1
            candidate = str(next_id)
            used_ints.add(next_id)
            used_ids.add(candidate)
            next_id += 1
        else:
            used_ints.add(source_id)
            used_ids.add(candidate)
        id_map[str(source_id)] = candidate

    class_counts: Counter[str] = Counter()
    expd_count = 0
    for frame_id, car_labels in filtered_main.items():
        expd_frame = expd_by_frame[frame_id]
        labels = [copy.deepcopy(dict(label)) for label in car_labels]
        for label in labels:
            if label.get("obj_type") != "Car":
                raise AssertionError("main label set contains a non-Car label")
            class_counts["Car"] += 1
        for det in expd_frame.get("detections", []):
            canonical = tracking.canonical_class_name(det.get("class_name", ""))
            if canonical not in NON_CAR_CLASSES:
                raise AssertionError(
                    f"expD label set contains invalid class: {canonical}")
            label = tracking.box_to_label(copy.deepcopy(det))
            source_id = str(label["obj_id"])
            label["obj_id"] = id_map[source_id]
            labels.append(label)
            class_counts[canonical] += 1
            expd_count += 1
        output.append({"frame_id": frame_id, "labels": labels})

    frame_overlap_stats = _remove_frame_level_car_overlap(output)
    final_class_counts: Counter[str] = Counter(
        str(label.get("obj_type", ""))
        for frame in output for label in frame.get("labels", []))
    main_count = sum(len(labels) for labels in filtered_main.values())
    return output, {
        "pipeline": "hybrid_merge",
        "frames": len(output),
        "main_car_detections": main_count,
        "expd_non_car_detections": expd_count,
        "merged_detections": sum(len(frame["labels"]) for frame in output),
        "class_counts": dict(sorted(final_class_counts.items())),
        "class_counts_before_frame_overlap": dict(sorted(class_counts.items())),
        "frame_car_overlap": frame_overlap_stats,
        "track_id_remap": {
            "source_track_ids": sorted(source_ids),
            "remapped_track_ids": id_map,
        },
        "high_iou_absorb": absorb_stats,
        "policy": {
            "main_classes": ["Car"],
            "expd_classes": sorted(NON_CAR_CLASSES),
            "merge_key": "frame_id",
            "output_order": "main Car labels followed by expD non-Car labels",
        },
    }


def _frame_index(frames: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    indexed: Dict[str, Mapping[str, Any]] = {}
    for frame in frames:
        frame_id = str(frame.get("frame_id", ""))
        if frame_id in indexed:
            raise ValueError(f"duplicate frame_id: {frame_id}")
        indexed[frame_id] = frame
    return indexed


def _remap_track_ids(
        frames: List[Dict[str, Any]],
        used_ids: set[int],
) -> Dict[str, Any]:
    """Make expD IDs disjoint from main IDs while preserving track identity."""
    source_ids = sorted({
        int(det["track_id"])
        for frame in frames for det in frame.get("detections", [])
        if det.get("track_id") is not None
    })
    next_id = max(used_ids, default=0) + 1
    mapping: Dict[int, int] = {}
    for source_id in source_ids:
        while next_id in used_ids:
            next_id += 1
        mapping[source_id] = next_id
        used_ids.add(next_id)
        next_id += 1
    for frame in frames:
        for det in frame.get("detections", []):
            if det.get("track_id") is not None:
                det["track_id"] = mapping[int(det["track_id"])]
    return {
        "source_track_ids": source_ids,
        "remapped_track_ids": mapping,
        "used_main_track_ids": sorted(used_ids - set(mapping.values())),
    }


def merge_frames(
        main_frames: Sequence[Mapping[str, Any]],
        expd_frames: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Merge Car-only main frames with non-Car expD frames by frame ID."""
    main_by_frame = _frame_index(main_frames)
    expd_by_frame = _frame_index(expd_frames)
    if set(main_by_frame) != set(expd_by_frame):
        missing_main = sorted(set(expd_by_frame) - set(main_by_frame))
        missing_expd = sorted(set(main_by_frame) - set(expd_by_frame))
        raise ValueError(
            "hybrid frame IDs differ: "
            f"missing_main={missing_main}, missing_expD={missing_expd}")

    output: List[Dict[str, Any]] = []
    used_ids = {
        int(det["track_id"])
        for frame in main_frames for det in frame.get("detections", [])
        if det.get("track_id") is not None
    }
    expd = copy.deepcopy([dict(frame) for frame in expd_frames])
    remap = _remap_track_ids(expd, used_ids)

    main_count = expd_count = 0
    class_counts: Counter[str] = Counter()
    for main_frame in main_frames:
        frame_id = str(main_frame["frame_id"])
        expd_frame = next(frame for frame in expd
                          if str(frame["frame_id"]) == frame_id)
        main_detections = copy.deepcopy(list(main_frame.get("detections", [])))
        expd_detections = []
        for det in expd_frame.get("detections", []):
            canonical = tracking.canonical_class_name(det.get("class_name", ""))
            if canonical not in NON_CAR_CLASSES:
                raise AssertionError(
                    f"hybrid expD input contains non-target class: {canonical}")
            det = copy.deepcopy(det)
            det["class_name"] = canonical
            expd_detections.append(det)
        detections = main_detections + expd_detections
        for det in detections:
            canonical = tracking.canonical_class_name(det.get("class_name", ""))
            if canonical is None:
                raise AssertionError(
                    f"hybrid main input contains unknown class: {det.get('class_name')}")
            if canonical == "Car" and det not in main_detections:
                raise AssertionError("Car detection leaked from expD route")
            det["class_name"] = canonical
            class_counts[canonical] += 1
        output.append({
            "frame_id": frame_id,
            "num_points": main_frame.get("num_points", expd_frame.get("num_points", 0)),
            "num_detections": len(detections),
            "detections": detections,
        })
        main_count += len(main_detections)
        expd_count += len(expd_detections)

    diagnostics = {
        "pipeline": "hybrid_merge",
        "frames": len(output),
        "main_car_detections": main_count,
        "expd_non_car_detections": expd_count,
        "merged_detections": sum(len(f["detections"]) for f in output),
        "class_counts": dict(sorted(class_counts.items())),
        "track_id_remap": remap,
        "policy": {
            "main_classes": ["Car"],
            "expd_classes": sorted(NON_CAR_CLASSES),
            "merge_key": "frame_id",
            "output_order": "main Car detections followed by expD non-Car detections",
        },
    }
    return output, diagnostics
