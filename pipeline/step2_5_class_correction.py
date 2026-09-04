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
        center_gate: float = 0.6,
        min_frames: int = 6,
        min_total_rotation: float = 1.0,
        min_steps: int = 3,
        max_reversal_fraction: float = 0.30,
        step_gate: float = 0.12,
) -> tuple[set[int], Dict[str, Any]]:
    """Find stationary Car tracks whose heading keeps rotating.

    A track qualifies when all of its observations are ``Car``, the world XY
    center span stays within ``center_gate`` (so the position is fixed), and
    the world heading rotates continuously (enough accumulated signed
    rotation and few direction reversals).  Such an ID is treated as a
    detection artifact and is removed as a whole.
    """
    by_id: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for frame in frames:
        ts = int(frame["frame_id"])
        wf = coords.world_from_lidar(ts)
        if wf is None:
            continue
        for det in frame.get("detections", []):
            if det.get("track_id") is None or det.get("class_name") != "Car":
                continue
            box = det.get("box_lidar")
            if not tracking.finite_box(det):
                continue
            center = tracking.center_world(box, wf)
            yaw = tracking.yaw_world(float(box[6]), wf)
            by_id.setdefault(int(det["track_id"]), []).append(
                (int(ts), center, yaw))

    dropped: set[int] = set()
    details: List[Dict[str, Any]] = []
    for track_id, items in by_id.items():
        items = sorted(items, key=lambda value: value[0])
        centers = np.asarray([center[:2] for _, center, _ in items],
                             dtype=np.float64)
        yaws = [yaw for _, _, yaw in items]
        if len(centers) < min_frames:
            continue
        centroid = np.median(centers, axis=0)
        span = float(np.max(np.linalg.norm(centers - centroid, axis=1)))
        if span > center_gate:
            continue
        deltas = [tracking.wrap_angle(yaws[index] - yaws[index - 1])
                  for index in range(1, len(yaws))]
        total = float(sum(abs(delta) for delta in deltas))
        abs_steps = int(sum(1 for delta in deltas if abs(delta) >= step_gate))
        signs = [1 if delta >= 0.0 else -1 for delta in deltas
                 if abs(delta) >= step_gate]
        reversals = int(sum(1 for index in range(1, len(signs))
                            if signs[index] != signs[index - 1]))
        reversal_fraction = reversals / max(len(signs) - 1, 1)
        if (total >= min_total_rotation and abs_steps >= min_steps
                and reversal_fraction <= max_reversal_fraction):
            dropped.add(track_id)
            details.append({
                "track_id": track_id,
                "observations": len(centers),
                "center_span": round(span, 4),
                "total_rotation": round(total, 4),
                "abs_steps": abs_steps,
                "reversal_fraction": round(reversal_fraction, 4),
            })
    return dropped, {
        "enabled": True,
        "dropped_track_ids": sorted(dropped),
        "tracks_dropped": len(dropped),
        "details": details,
        "config": {
            "center_gate": center_gate,
            "min_frames": min_frames,
            "min_total_rotation": min_total_rotation,
            "min_steps": min_steps,
            "max_reversal_fraction": max_reversal_fraction,
            "step_gate": step_gate,
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
        rot_center_gate: float = 0.6,
        rot_min_frames: int = 6,
        rot_min_total: float = 1.0,
        rot_min_steps: int = 3,
        rot_max_reversal: float = 0.30,
        rot_step_gate: float = 0.12,
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
            center_gate=float(rot_center_gate),
            min_frames=int(rot_min_frames),
            min_total_rotation=float(rot_min_total),
            min_steps=int(rot_min_steps),
            max_reversal_fraction=float(rot_max_reversal),
            step_gate=float(rot_step_gate))
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
    parser.add_argument("--rot-center-gate", type=float, default=0.6)
    parser.add_argument("--rot-min-frames", type=int, default=6)
    parser.add_argument("--rot-min-total", type=float, default=1.0)
    parser.add_argument("--rot-min-steps", type=int, default=3)
    parser.add_argument("--rot-max-reversal", type=float, default=0.30)
    parser.add_argument("--rot-step-gate", type=float, default=0.12)
    parser.add_argument("--keep-classes",
                        default=",".join(tracking.TARGET_CLASSES))
    args = parser.parse_args()
    diagnostics = run(
        args.step2_json, args.step2_diagnostics, args.clip, args.out_json,
        args.out_clip, args.diagnostics,
        hard_filter_config=_hard_config(args), min_lifecycle=args.min_lifecycle,
        static_rotation_enabled=not args.disable_static_rotation_filter,
        rot_center_gate=args.rot_center_gate,
        rot_min_frames=args.rot_min_frames,
        rot_min_total=args.rot_min_total,
        rot_min_steps=args.rot_min_steps,
        rot_max_reversal=args.rot_max_reversal,
        rot_step_gate=args.rot_step_gate)
    print(json.dumps({
        "class_changed": diagnostics["class_correction"]["detections_changed"],
        "hard_filter_removed": diagnostics["hard_filters_pass_2"]["detections_removed"],
        "short_tracks_removed": diagnostics["short_track_filter"]["tracks_dropped"],
        "static_rotating_removed": diagnostics["static_car_rotating_filter"].get("tracks_dropped", 0),
        "final_detections": diagnostics["final_detections"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
