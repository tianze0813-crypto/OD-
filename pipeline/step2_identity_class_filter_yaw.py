#!/usr/bin/env python3
"""Step 2: identity tracking, hard filtering, and integrated yaw for one clip."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification.class_refinement import (
    ClassRefinementConfig,
    finalize_track_classes,
    preassociate_and_unify,
)
from filtering.hard_filters import (
    HardFilterConfig,
    apply_hard_filters,
    deduplicate_same_center,
)
from geometry.static_yaw import stabilize_static_yaw
from geometry.yaw_integrated import apply_yaw_integrated
from tracking import tracker_conservative as tracking
from tracking import tracker_static_first as static_first


def _count(frames: Sequence[Dict[str, Any]]) -> int:
    return sum(len(frame.get("detections", [])) for frame in frames)


def _assert_class_only(before: Sequence[Dict[str, Any]],
                       after: Sequence[Dict[str, Any]]) -> Dict[str, int | bool]:
    if len(before) != len(after):
        raise AssertionError("class refinement changed frame count")
    checked = 0
    changed = 0
    for left_frame, right_frame in zip(before, after):
        if left_frame.get("frame_id") != right_frame.get("frame_id"):
            raise AssertionError("class refinement changed frame order")
        left_detections = left_frame.get("detections", [])
        right_detections = right_frame.get("detections", [])
        if len(left_detections) != len(right_detections):
            raise AssertionError("class refinement changed detection count")
        for left, right in zip(left_detections, right_detections):
            comparable = copy.deepcopy(right)
            comparable["class_name"] = left.get("class_name")
            if comparable != left:
                raise AssertionError("class refinement changed a non-class field")
            if left.get("class_name") != right.get("class_name"):
                changed += 1
            checked += 1
    return {"passed": True, "detections_checked": checked,
            "classes_changed": changed}


def run(
        in_json: Path, clip: Path, out_json: Path,
        out_clip: Optional[Path], diagnostics_path: Path, *,
        hard_filter_config: HardFilterConfig = HardFilterConfig(),
        class_config: ClassRefinementConfig = ClassRefinementConfig(),
        min_lifecycle: int = 4,
        same_center_gate: float = 0.35) -> Dict[str, Any]:
    source = json.loads(Path(in_json).read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError(f"input must be a list of frames: {in_json}")
    frames: List[Dict[str, Any]] = copy.deepcopy(source)
    diagnostics: Dict[str, Any] = {
        "pipeline": "step2_identity_class_filter_yaw",
        "clip": str(Path(clip).resolve()),
        "input_json": str(Path(in_json).resolve()),
        "input_frames": len(frames),
        "input_detections": _count(frames),
        "stage_order": [
            "class_preassociation_and_unification",
            "static_first_identity_tracking",
            "hard_annotation_filters",
            "same_center_deduplication",
            "short_track_filter",
            "confirmed_static_parking_yaw_stabilization",
            "integrated_vehicle_and_pedestrian_yaw",
            "final_track_class_consistency_and_size_classification",
            "sust_export",
        ],
    }

    before_classes = copy.deepcopy(frames)
    coords = tracking.CoordinateProvider(Path(clip))
    # Preserve the reviewed pre-tracking class behavior so final size
    # thresholds cannot perturb identity tracking. Final size decisions use
    # complete tracks after static-first identity assignment.
    tracking_class_config = ClassRefinementConfig(
        max_cross_class_gap_sec=class_config.max_cross_class_gap_sec,
        max_cross_class_distance=class_config.max_cross_class_distance,
        max_cross_class_speed=class_config.max_cross_class_speed,
        max_relative_size_delta=class_config.max_relative_size_delta,
        uniqueness_margin=class_config.uniqueness_margin,
        truck_length_min=5.0,
        pedestrian_length_max=class_config.pedestrian_length_max,
        pedestrian_width_max=class_config.pedestrian_width_max,
        cyclist_length_max=3.0,
        cyclist_width_max=1.35,
    )
    diagnostics["class_preassociation"] = preassociate_and_unify(
        frames, coords, tracking_class_config)
    diagnostics["class_only_check"] = _assert_class_only(before_classes, frames)

    pre_tracking = copy.deepcopy(frames)
    tracker = static_first.StaticFirstTracker(coords)
    tracked, tracking_diagnostics = tracker.process(frames)
    diagnostics["tracking"] = tracking_diagnostics
    diagnostics["identity_only_check"] = static_first.verify_identity_only(
        pre_tracking, tracked)

    for source_frame, tracked_frame in zip(source, tracked):
        source_detections = source_frame.get("detections", [])
        tracked_detections = tracked_frame.get("detections", [])
        if len(source_detections) != len(tracked_detections):
            raise AssertionError("cannot align source classes after tracking")
        for source_det, tracked_det in zip(
                source_detections, tracked_detections):
            tracked_det["_step2_source_class"] = str(
                source_det.get("class_name", ""))
    diagnostics["hard_filters"] = apply_hard_filters(
        tracked, Path(clip), hard_filter_config)
    static_track_ids = {int(slot.track_id) for slot in tracker.slots}
    diagnostics["same_center_deduplication"] = deduplicate_same_center(
        tracked, static_track_ids=static_track_ids,
        center_gate=same_center_gate)
    diagnostics["short_track_filter"] = tracking.apply_post_filters(
        tracked, min_lifecycle=min_lifecycle)
    pre_static_yaw = copy.deepcopy(tracked)
    diagnostics["static_yaw_stabilization"] = stabilize_static_yaw(
        tracked, coords, tracker.slots,
        tracking_diagnostics.get("slot_motion_coordination", {}))
    tracked, diagnostics["yaw_integrated"] = apply_yaw_integrated(
        tracked, pre_static_yaw, coords, Path(clip),
        tracking_diagnostics, diagnostics["static_yaw_stabilization"])
    before_final_class = copy.deepcopy(tracked)
    diagnostics["class_finalization"] = finalize_track_classes(
        tracked, class_config)
    for before_frame, after_frame in zip(before_final_class, tracked):
        for before_det, after_det in zip(
                before_frame.get("detections", []),
                after_frame.get("detections", [])):
            before_det.pop("_step2_source_class", None)
            after_det.pop("_step2_source_class", None)
    diagnostics["final_class_only_check"] = _assert_class_only(
        before_final_class, tracked)
    diagnostics["final_detections"] = _count(tracked)

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(
        json.dumps(tracked, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    if out_clip is not None:
        diagnostics["sust_labels"] = tracking.export_clip(
            tracked, Path(clip), Path(out_clip))
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: class refinement, static-first identity tracking, and integrated yaw")
    parser.add_argument("--in-json", "--in_json", dest="in_json",
                        required=True, type=Path)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--out-json", "--out_json", dest="out_json",
                        required=True, type=Path)
    parser.add_argument("--out-clip", "--out_clip", dest="out_clip", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--min-lifecycle", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--range-front", type=float, default=80.0)
    parser.add_argument("--range-rear", type=float, default=20.0)
    parser.add_argument("--range-side", type=float, default=40.0)
    parser.add_argument("--sparsity-max-points", type=int, default=10)
    parser.add_argument("--visibility-min-ratio", type=float, default=0.05)
    parser.add_argument("--pedestrian-max-distance", type=float, default=20.0)
    parser.add_argument("--keep-classes",
                        default="Vehicle,Car,Truck,Pedestrian,Cyclist")
    parser.add_argument("--same-center-gate", type=float, default=0.35)
    parser.add_argument("--truck-length-min", type=float, default=6.0)
    parser.add_argument("--pedestrian-length-max", type=float, default=1.25)
    parser.add_argument("--pedestrian-width-max", type=float, default=1.10)
    parser.add_argument("--cyclist-length-max", type=float, default=3.5)
    parser.add_argument("--cyclist-width-max", type=float, default=1.50)
    args = parser.parse_args()

    hard_config = HardFilterConfig(
        score_threshold=args.score_threshold,
        range_front=args.range_front,
        range_rear=args.range_rear,
        range_side=args.range_side,
        sparsity_max_points=args.sparsity_max_points,
        visibility_min_ratio=args.visibility_min_ratio,
        pedestrian_max_distance=args.pedestrian_max_distance,
        keep_classes=tuple(x.strip() for x in args.keep_classes.split(",")
                           if x.strip()),
    )
    class_config = ClassRefinementConfig(
        truck_length_min=args.truck_length_min,
        pedestrian_length_max=args.pedestrian_length_max,
        pedestrian_width_max=args.pedestrian_width_max,
        cyclist_length_max=args.cyclist_length_max,
        cyclist_width_max=args.cyclist_width_max,
    )
    diagnostics_path = args.diagnostics or args.out_json.with_name(
        args.out_json.stem + "_diagnostics.json")
    diagnostics = run(
        args.in_json, args.clip, args.out_json, args.out_clip,
        diagnostics_path, hard_filter_config=hard_config,
        class_config=class_config, min_lifecycle=args.min_lifecycle,
        same_center_gate=args.same_center_gate)
    print(json.dumps({
        "input_detections": diagnostics["input_detections"],
        "hard_filter_removed": diagnostics["hard_filters"]["detections_removed"],
        "class_pre_changed": diagnostics["class_preassociation"]["detections_changed"],
        "tracks": diagnostics["tracking"].get("tracks_total"),
        "same_center_removed": diagnostics["same_center_deduplication"]["boxes_removed"],
        "short_tracks_removed": diagnostics["short_track_filter"]["tracks_dropped"],
        "yaw_boxes_by_mode": diagnostics["yaw_integrated"]["boxes_by_mode"],
        "final_detections": diagnostics["final_detections"],
        "sust_labels": diagnostics.get("sust_labels"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
