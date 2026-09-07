#!/usr/bin/env python3
"""Step 2.5: track-level class correction and the second hard-filter pass.

Only ``class_name`` is changed by the correction itself.  Track IDs and box
geometry are treated as immutable, then the same annotation filters are run a
second time against the corrected semantic labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification.class_refinement import ClassRefinementConfig, finalize_model_track_classes
from filtering.hard_filters import HardFilterConfig, apply_hard_filters
from tracking import tracker_conservative as tracking


def _count(frames: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(frame.get("detections", [])) for frame in frames)


def _assert_class_only(before: Sequence[Mapping[str, Any]],
                       after: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(before) != len(after):
        raise AssertionError("class correction changed frame count")
    checked = changed = 0
    for left_frame, right_frame in zip(before, after):
        if left_frame.get("frame_id") != right_frame.get("frame_id"):
            raise AssertionError("class correction changed frame order")
        left_dets = left_frame.get("detections", [])
        right_dets = right_frame.get("detections", [])
        if len(left_dets) != len(right_dets):
            raise AssertionError("class correction changed detection count")
        for left, right in zip(left_dets, right_dets):
            comparable = copy.deepcopy(right)
            comparable["class_name"] = left.get("class_name")
            if comparable != left:
                raise AssertionError("class correction changed a non-class field")
            changed += left.get("class_name") != right.get("class_name")
            checked += 1
    return {"passed": True, "detections_checked": checked,
            "classes_changed": int(changed),
            "protected_fields": ["track_id", "box_lidar", "box_presence"]}


def static_rotating_car_track_ids(
        frames: Sequence[Mapping[str, Any]],
        coords: tracking.CoordinateProvider,
        *,
        classes: Sequence[str] = ("Car",),
        center_gate: float = 3.0,
        overlap_gate: float = 0.35,
        yaw_gate: float = 1.30,
        step_gate: float = 0.80,
        min_flips: int = 2,
        min_frames: int = 5,
) -> tuple[set[int], Dict[str, Any]]:
    """Find stationary vehicle tracks whose heading spins in place (false positive).

    A moving turn also shows a large yaw spread, so the center alone is not
    enough.  Wrap every observed box of a track into a world frame and require
    BOTH a tight spatial footprint (a small center span AND boxes that keep
    overlapping, i.e. the footprint never leaves the spot) AND a large yaw
    swing (the heading covers most of the pi-periodic range, with several big
    steps).  Such an ID is removed as a whole.
    """
    by_id: dict[int, list[tuple[int, np.ndarray, float, np.ndarray]]] = {}
    for frame in frames:
        ts = int(frame["frame_id"])
        wf = coords.world_from_lidar(ts)
        if wf is None:
            continue
        for det in frame.get("detections", []):
            if det.get("track_id") is None or det.get("class_name") not in classes:
                continue
            box = det.get("box_lidar")
            if not tracking.finite_box(det):
                continue
            center = tracking.center_world(box, wf)
            yaw = tracking.yaw_world(float(box[6]), wf)
            size = np.asarray(box[3:6], dtype=np.float64)
            by_id.setdefault(int(det["track_id"]), []).append(
                (int(ts), center, yaw, size))

    dropped: set[int] = set()
    details: List[Dict[str, Any]] = []
    for track_id, items in by_id.items():
        items = sorted(items, key=lambda value: value[0])
        centers = np.asarray([center[:2] for _, center, _, _ in items],
                             dtype=np.float64)
        yaws = [yaw for _, _, yaw, _ in items]
        if len(centers) < min_frames:
            continue
        centroid = np.median(centers, axis=0)
        span = float(np.max(np.linalg.norm(centers - centroid, axis=1)))
        if span > center_gate:
            continue
        ious = [
            tracking.bev_iou(
                np.r_[items[index - 1][1][:2], 0.0],
                items[index - 1][3], items[index - 1][2],
                np.r_[items[index][1][:2], 0.0],
                items[index][3], items[index][2])
            for index in range(1, len(items))
        ]
        mean_iou = float(np.mean(ious)) if ious else 1.0
        if mean_iou < overlap_gate:
            continue
        yaw_span = 0.0
        for left in range(len(yaws)):
            for right in range(left + 1, len(yaws)):
                yaw_span = max(
                    yaw_span, tracking.angle_distance(
                        yaws[left], yaws[right], modulo_pi=True))
        if yaw_span < yaw_gate:
            continue
        deltas = [tracking.angle_distance(yaws[index], yaws[index - 1],
                                          modulo_pi=True)
                  for index in range(1, len(yaws))]
        flips = int(sum(1 for delta in deltas if delta >= step_gate))
        if flips < min_flips:
            continue
        dropped.add(track_id)
        details.append({
            "track_id": track_id,
            "observations": len(centers),
            "center_span": round(span, 4),
            "mean_consec_iou": round(mean_iou, 4),
            "yaw_span": round(yaw_span, 4),
            "big_yaw_steps": flips,
        })
    return dropped, {
        "enabled": True,
        "dropped_track_ids": sorted(dropped),
        "tracks_dropped": len(dropped),
        "details": details,
        "config": {
            "center_gate": center_gate,
            "overlap_gate": overlap_gate,
            "yaw_gate": yaw_gate,
            "step_gate": step_gate,
            "min_flips": min_flips,
            "min_frames": min_frames,
        },
    }


def _drop_track_ids(frames: Sequence[Mapping[str, Any]],
                    ids: set[int]) -> int:
    ids = set(ids)
    removed = 0
    for frame in frames:
        old = frame.get("detections", [])
        frame["detections"] = [d for d in old if d.get("track_id") not in ids]
        removed += len(old) - len(frame["detections"])
        frame["num_detections"] = len(frame["detections"])
    return removed


def run(
        step2_json: Path,
        step2_diagnostics: Path,
        clip: Path,
        out_json: Path,
        out_clip: Optional[Path] = None,
        diagnostics_path: Optional[Path] = None,
        *,
        hard_filter_config: HardFilterConfig = HardFilterConfig(),
        class_config: ClassRefinementConfig = ClassRefinementConfig(),
        min_lifecycle: int = 4,
        static_rotation_enabled: bool = True,
        static_rotation_classes: Tuple[str, ...] = ("Car",),
        rot_center_gate: float = 3.0,
        rot_overlap_gate: float = 0.35,
        rot_yaw_gate: float = 1.30,
        rot_step_gate: float = 0.80,
        rot_min_flips: int = 2,
        rot_min_frames: int = 5,
) -> Dict[str, Any]:
    source = json.loads(Path(step2_json).read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError(f"input must be a list of frames: {step2_json}")
    previous = json.loads(Path(step2_diagnostics).read_text(encoding="utf-8"))
    frames: List[Dict[str, Any]] = copy.deepcopy(source)

    before_class = copy.deepcopy(frames)
    class_correction = finalize_model_track_classes(
        frames, tracking.TARGET_CLASSES)
    class_only_check = _assert_class_only(before_class, frames)

    static_rotation = {"enabled": False}
    if static_rotation_enabled:
        coords = tracking.CoordinateProvider(Path(clip))
        rotating_ids, static_rotation = static_rotating_car_track_ids(
            frames, coords,
            classes=tuple(static_rotation_classes),
            center_gate=float(rot_center_gate),
            overlap_gate=float(rot_overlap_gate),
            yaw_gate=float(rot_yaw_gate),
            step_gate=float(rot_step_gate),
            min_flips=int(rot_min_flips),
            min_frames=int(rot_min_frames))
        _drop_track_ids(frames, rotating_ids)

    # The second filter sees canonical classes and is therefore the final
    # authority on which detections enter annotation export.
    second_filter = apply_hard_filters(frames, Path(clip), hard_filter_config)
    short_track_filter = tracking.apply_post_filters(
        frames, min_lifecycle=int(min_lifecycle))

    diagnostics: Dict[str, Any] = {
        "pipeline": "step2_5_class_correction",
        "clip": str(Path(clip).resolve()),
        "source_step2_json": str(Path(step2_json).resolve()),
        "source_step2_diagnostics": str(Path(step2_diagnostics).resolve()),
        "input_frames": len(source),
        "input_detections": _count(source),
        "stage_order": [
            "track_class_canonicalization_and_majority_vote",
            "hard_filters_pass_2",
            "static_car_rotating_filter",
            "short_track_filter",
        ],
        "execution_order": [
            "track_class_canonicalization_and_majority_vote",
            "static_car_rotating_filter",
            "hard_filters_pass_2",
            "short_track_filter",
        ],
        "tracking": previous.get("tracking", {}),
        "tracking_diagnostics": previous.get("tracking", {}),
        "hard_filters_pass_1": previous.get("hard_filters_pass_1",
                                             previous.get("hard_filters", {})),
        "class_correction": class_correction,
        "class_only_check": class_only_check,
        "static_car_rotating_filter": static_rotation,
        "hard_filters": second_filter,
        "hard_filters_pass_2": second_filter,
        "short_track_filter": short_track_filter,
        "final_detections": _count(frames),
        "yaw_pending": True,
    }

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(frames, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    if out_clip is not None:
        diagnostics["sust_labels"] = tracking.export_clip(
            frames, Path(clip), Path(out_clip))
    target = Path(diagnostics_path or out_json.with_name(
        out_json.stem + "_diagnostics.json"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return diagnostics


def _hard_config(args: argparse.Namespace) -> HardFilterConfig:
    class_score_thresholds = (
        ("Car", args.car_score_threshold),
        ("Truck", args.truck_score_threshold),
        ("Bus", args.bus_score_threshold),
        ("Pedestrian", args.pedestrian_score_threshold),
        ("Nonmotorized_vehicle", args.nonmotorized_score_threshold),
    )
    return HardFilterConfig(
        score_threshold=args.score_threshold,
        class_score_thresholds=class_score_thresholds,
        range_front=args.range_front,
        range_rear=args.range_rear,
        range_side=args.range_side,
        sparsity_max_points=args.sparsity_max_points,
        visibility_min_ratio=args.visibility_min_ratio,
        pedestrian_max_distance=args.pedestrian_max_distance,
        keep_classes=tuple(x.strip() for x in args.keep_classes.split(",")
                            if x.strip()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step2-json", type=Path, required=True)
    parser.add_argument("--step2-diagnostics", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-clip", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--min-lifecycle", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--car-score-threshold", type=float, default=0.25)
    parser.add_argument("--truck-score-threshold", type=float, default=0.4)
    parser.add_argument("--bus-score-threshold", type=float, default=0.4)
    parser.add_argument("--pedestrian-score-threshold", type=float, default=0.3)
    parser.add_argument("--nonmotorized-score-threshold", type=float, default=0.3)
    parser.add_argument("--range-front", type=float, default=80.0)
    parser.add_argument("--range-rear", type=float, default=20.0)
    parser.add_argument("--range-side", type=float, default=40.0)
    parser.add_argument("--sparsity-max-points", type=int, default=10)
    parser.add_argument("--visibility-min-ratio", type=float, default=0.05)
    parser.add_argument("--pedestrian-max-distance", type=float, default=20.0)
    parser.add_argument("--disable-static-rotation-filter", action="store_true",
                        help="turn off the static Car rotating-yaw filter")
    parser.add_argument("--rot-center-gate", type=float, default=3.0)
    parser.add_argument("--rot-overlap-gate", type=float, default=0.35)
    parser.add_argument("--rot-yaw-gate", type=float, default=1.30)
    parser.add_argument("--rot-step-gate", type=float, default=0.80)
    parser.add_argument("--rot-min-flips", type=int, default=2)
    parser.add_argument("--rot-min-frames", type=int, default=5)
    parser.add_argument("--keep-classes",
                        default=",".join(tracking.TARGET_CLASSES))
    args = parser.parse_args()
    diagnostics = run(
        args.step2_json, args.step2_diagnostics, args.clip, args.out_json,
        args.out_clip, args.diagnostics,
        hard_filter_config=_hard_config(args), min_lifecycle=args.min_lifecycle,
        static_rotation_enabled=not args.disable_static_rotation_filter,
        rot_center_gate=args.rot_center_gate,
        rot_overlap_gate=args.rot_overlap_gate,
        rot_yaw_gate=args.rot_yaw_gate,
        rot_step_gate=args.rot_step_gate,
        rot_min_flips=args.rot_min_flips,
        rot_min_frames=args.rot_min_frames)
    print(json.dumps({
        "class_changed": diagnostics["class_correction"]["detections_changed"],
        "hard_filter_removed": diagnostics["hard_filters_pass_2"]["detections_removed"],
        "short_tracks_removed": diagnostics["short_track_filter"]["tracks_dropped"],
        "static_rotating_removed": diagnostics["static_car_rotating_filter"].get("tracks_dropped", 0),
        "final_detections": diagnostics["final_detections"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
