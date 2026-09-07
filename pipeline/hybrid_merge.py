#!/usr/bin/env python3
"""Merge main-branch Car detections with expD non-Car detections."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence

from tracking import tracker_conservative as tracking


NON_CAR_CLASSES = frozenset({
    "Truck", "Bus", "Pedestrian", "Nonmotorized_vehicle",
})


def merge_label_frames(
        main_labels: Mapping[str, Sequence[Mapping[str, Any]]],
        expd_frames: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Merge already-exported main Car labels with expD frame detections."""
    expd_by_frame = _frame_index(expd_frames)
    if set(main_labels) != set(expd_by_frame):
        raise ValueError(
            "hybrid frame IDs differ: "
            f"missing_main={sorted(set(expd_by_frame) - set(main_labels))}, "
            f"missing_expD={sorted(set(main_labels) - set(expd_by_frame))}")

    output: List[Dict[str, Any]] = []
    used_ids = {
        str(label.get("obj_id"))
        for labels in main_labels.values() for label in labels
        if label.get("obj_id") is not None
    }
    source_ids = {
        str(det.get("track_id"))
        for frame in expd_frames for det in frame.get("detections", [])
        if det.get("track_id") is not None
    }
    id_map: Dict[str, str] = {}
    for source_id in sorted(source_ids):
        candidate = source_id
        if candidate in used_ids:
            candidate = f"expd_{source_id}"
            suffix = 2
            while candidate in used_ids:
                candidate = f"expd_{source_id}_{suffix}"
                suffix += 1
        id_map[source_id] = candidate
        used_ids.add(candidate)

    class_counts: Counter[str] = Counter()
    expd_count = 0
    for frame_id, car_labels in main_labels.items():
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

    main_count = sum(len(labels) for labels in main_labels.values())
    return output, {
        "pipeline": "hybrid_merge",
        "frames": len(output),
        "main_car_detections": main_count,
        "expd_non_car_detections": expd_count,
        "merged_detections": sum(len(frame["labels"]) for frame in output),
        "class_counts": dict(sorted(class_counts.items())),
        "track_id_remap": {
            "source_track_ids": sorted(source_ids),
            "remapped_track_ids": id_map,
        },
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
