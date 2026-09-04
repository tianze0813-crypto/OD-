#!/usr/bin/env python3
"""Step 2: class-blind identity tracking and the first hard-filter pass.

This stage protects the annotation identity contract.  It assigns IDs before
any score, visibility, range, sparsity, or class correction can remove a
detection.  It does not change class names, yaw, dimensions, or box centers.
"""

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

from filtering.hard_filters import HardFilterConfig, apply_hard_filters, deduplicate_same_center
from tracking import tracker_conservative as tracking
from tracking import tracker_static_first as static_first


def _count(frames: Sequence[Dict[str, Any]]) -> int:
    return sum(len(frame.get("detections", [])) for frame in frames)


def run(
        in_json: Path,
        clip: Path,
        out_json: Path,
        out_clip: Optional[Path] = None,
        diagnostics_path: Optional[Path] = None,
        *,
        hard_filter_config: HardFilterConfig = HardFilterConfig(),
        same_center_gate: float = 0.35,
) -> Dict[str, Any]:
    source = json.loads(Path(in_json).read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError(f"input must be a list of frames: {in_json}")

    frames: List[Dict[str, Any]] = copy.deepcopy(source)
    coords = tracking.CoordinateProvider(Path(clip))
    tracker = static_first.StaticFirstTracker(coords)
    tracked, tracking_diagnostics = tracker.process(frames)
    identity_check = static_first.verify_identity_only(source, tracked)

    # The first filter is intentionally after association.  Static tracks are
    # allowed to keep their IDs even when one filtered observation disappears.
    first_filter = apply_hard_filters(tracked, Path(clip), hard_filter_config)
    static_track_ids = {
        int(slot.track_id) for slot in tracker.slots
        if slot.track_id is not None
    }
    dedup = deduplicate_same_center(
        tracked, static_track_ids=static_track_ids,
        center_gate=float(same_center_gate))

    diagnostics: Dict[str, Any] = {
        "pipeline": "step2_identity",
        "clip": str(Path(clip).resolve()),
        "input_json": str(Path(in_json).resolve()),
        "input_frames": len(source),
        "input_detections": _count(source),
        "stage_order": [
            "class_blind_identity_tracking",
            "hard_filters_pass_1",
            "same_center_deduplication",
        ],
        "tracking": tracking_diagnostics,
        "identity_only_check": identity_check,
        "hard_filters": first_filter,
        "hard_filters_pass_1": first_filter,
        "same_center_deduplication": dedup,
        "final_detections": _count(tracked),
        "class_correction_pending": True,
        "yaw_pending": True,
        "short_track_filter_pending": True,
    }

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(tracked, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    if out_clip is not None:
        diagnostics["sust_labels"] = tracking.export_clip(
            tracked, Path(clip), Path(out_clip))
    target = Path(diagnostics_path or out_json.with_name(
        out_json.stem + "_diagnostics.json"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
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
    parser.add_argument("--in-json", "--in_json", dest="in_json", required=True, type=Path)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--out-json", "--out_json", dest="out_json", required=True, type=Path)
    parser.add_argument("--out-clip", "--out_clip", dest="out_clip", type=Path)
    parser.add_argument("--diagnostics", type=Path)
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
    parser.add_argument(
        "--keep-classes",
        default="Car,Truck,Bus,Pedestrian,Nonmotorized_vehicle",
    )
    parser.add_argument("--same-center-gate", type=float, default=0.35)
    args = parser.parse_args()
    diagnostics = run(
        args.in_json, args.clip, args.out_json, args.out_clip,
        args.diagnostics, hard_filter_config=_hard_config(args),
        same_center_gate=args.same_center_gate)
    print(json.dumps({
        "input_detections": diagnostics["input_detections"],
        "hard_filter_removed": diagnostics["hard_filters_pass_1"]["detections_removed"],
        "tracks": diagnostics["tracking"].get("tracks_total"),
        "same_center_removed": diagnostics["same_center_deduplication"]["boxes_removed"],
        "final_detections": diagnostics["final_detections"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
