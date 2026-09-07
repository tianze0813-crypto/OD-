#!/usr/bin/env python3
"""Run the current five-class route while keeping only non-Car classes."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification.class_refinement import ClassRefinementConfig
from filtering.five_class_output import apply_five_class_output
from filtering.hard_filters import HardFilterConfig, apply_category_score_filter
from geometry.box_geometry import GeometryConfig
from geometry.multiclass_refinement import NonmotorizedSizeConfig, TruckOverlapConfig
from pipeline import step2_5_class_correction as step2_5
from pipeline import step2_identity
from pipeline import step3_refinement
from tracking import tracker_conservative as tracking


NON_CAR_CLASSES = (
    "Truck", "Bus", "Pedestrian", "Nonmotorized_vehicle",
)


def _count(frames: List[Dict[str, Any]]) -> int:
    return sum(len(frame.get("detections", [])) for frame in frames)


def _noncar_filter(frames: List[Dict[str, Any]]) -> Dict[str, int]:
    before = _count(frames)
    removed = 0
    removed_by_class: Dict[str, int] = {}
    for frame in frames:
        kept = []
        for det in frame.get("detections", []):
            canonical = tracking.canonical_class_name(det.get("class_name", ""))
            if canonical in NON_CAR_CLASSES:
                det["class_name"] = canonical
                kept.append(det)
            else:
                removed += 1
                key = canonical or str(det.get("class_name", ""))
                removed_by_class[key] = removed_by_class.get(key, 0) + 1
        frame["detections"] = kept
        frame["num_detections"] = len(kept)
    return {
        "detections_before": before,
        "detections_after": _count(frames),
        "detections_removed": removed,
        "removed_by_class": dict(sorted(removed_by_class.items())),
    }


def _hard_config(*, sparsity_max_points: int,
                 visibility_min_ratio: float,
                 score_threshold: float | None) -> HardFilterConfig:
    fallback = 0.3 if score_threshold is None else float(score_threshold)
    thresholds = fallback if score_threshold is not None else None
    per_class = tuple(
        (name, fallback if thresholds is not None else value)
        for name, value in (
            ("Truck", 0.4), ("Bus", 0.4),
            ("Pedestrian", 0.3), ("Nonmotorized_vehicle", 0.3),
        )
    )
    return HardFilterConfig(
        score_threshold=fallback,
        class_score_thresholds=per_class,
        sparsity_max_points=int(sparsity_max_points),
        visibility_min_ratio=float(visibility_min_ratio),
        keep_classes=NON_CAR_CLASSES,
    )


def run(raw_json: Path, clip: Path, out_json: Path,
        diagnostics_path: Path | None = None,
        *, sparsity_max_points: int = 10,
        visibility_min_ratio: float = 0.05,
        short_track_max_frames: int = 4,
        score_threshold: float | None = None) -> Dict[str, Any]:
    source = json.loads(Path(raw_json).read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError(f"input must be a list of frames: {raw_json}")
    frames: List[Dict[str, Any]] = copy.deepcopy(source)
    diagnostics: Dict[str, Any] = {
        "pipeline": "hybrid_expD_noncar",
        "source_raw_json": str(Path(raw_json).resolve()),
        "clip": str(Path(clip).resolve()),
        "input_frames": len(frames),
        "input_detections": _count(frames),
        "stage_order": [
            "early_non_car_class_filter",
            "category_score_filter_non_car_only",
            "current_identity_tracking",
            "current_class_correction_and_filters_without_static_car_pass",
            "current_non_car_geometry_refinement",
            "base_link_conversion",
        ],
    }
    diagnostics["early_non_car_filter"] = _noncar_filter(frames)

    hard_config = _hard_config(
        sparsity_max_points=sparsity_max_points,
        visibility_min_ratio=visibility_min_ratio,
        score_threshold=score_threshold,
    )
    work_root = Path(out_json).parent
    diagnostics["pre_step2_score_filter"] = apply_category_score_filter(
        frames, hard_config)
    filtered_input = work_root / (Path(out_json).stem + "_filtered_raw.json")
    filtered_input.write_text(
        json.dumps(frames, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    step2_json = work_root / (Path(out_json).stem + "_step2.json")
    step2_diag = work_root / (Path(out_json).stem + "_step2_diagnostics.json")
    step2_identity.run(
        filtered_input, Path(clip), step2_json,
        diagnostics_path=step2_diag,
        hard_filter_config=hard_config,
    )

    step2_5_json = work_root / (Path(out_json).stem + "_step2_5.json")
    step2_5_diag = work_root / (Path(out_json).stem + "_step2_5_diagnostics.json")
    step2_5.run(
        step2_json, step2_diag, Path(clip), step2_5_json,
        diagnostics_path=step2_5_diag,
        hard_filter_config=hard_config,
        class_config=ClassRefinementConfig(),
        min_lifecycle=int(short_track_max_frames),
        static_rotation_enabled=True,
        static_rotation_classes=("Truck", "Bus"),
    )

    step3_json = work_root / (Path(out_json).stem + "_step3.json")
    step3_diag = work_root / (Path(out_json).stem + "_step3_diagnostics.json")
    step3_refinement.run(
        step2_5_json, step2_5_diag, Path(clip), step3_json,
        diagnostics_path=step3_diag,
        geometry_config=GeometryConfig(),
        truck_config=TruckOverlapConfig(),
        nonmotorized_config=NonmotorizedSizeConfig(),
        car_refinement_enabled=False,
    )

    processed = json.loads(step3_json.read_text(encoding="utf-8"))
    output, final_diag = apply_five_class_output(
        processed, tracking.CoordinateProvider(Path(clip)))
    leaked = [
        det.get("class_name")
        for frame in output for det in frame.get("detections", [])
        if tracking.canonical_class_name(det.get("class_name", "")) == "Car"
    ]
    if leaked:
        raise AssertionError("expD non-Car route leaked Car detections")
    diagnostics["step2"] = json.loads(step2_diag.read_text(encoding="utf-8"))
    diagnostics["step2_5"] = json.loads(step2_5_diag.read_text(encoding="utf-8"))
    diagnostics["step3"] = json.loads(step3_diag.read_text(encoding="utf-8"))
    diagnostics["final_output"] = final_diag
    diagnostics["final_detections"] = _count(output)
    diagnostics["output_classes"] = sorted({
        tracking.canonical_class_name(det.get("class_name", ""))
        for frame in output for det in frame.get("detections", [])
        if tracking.canonical_class_name(det.get("class_name", "")) is not None
    })

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    target = Path(diagnostics_path or out_json.with_name(
        out_json.stem + "_diagnostics.json"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", required=True, type=Path)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--sparsity-max-points", type=int, default=10)
    parser.add_argument("--visibility-min-ratio", type=float, default=0.05)
    parser.add_argument("--short-track-max-frames", type=int, default=4)
    parser.add_argument("--score-threshold", type=float)
    args = parser.parse_args()
    result = run(
        args.raw_json, args.clip, args.out_json, args.diagnostics,
        sparsity_max_points=args.sparsity_max_points,
        visibility_min_ratio=args.visibility_min_ratio,
        short_track_max_frames=args.short_track_max_frames,
        score_threshold=args.score_threshold,
    )
    print(json.dumps({
        "pipeline": result["pipeline"],
        "input_detections": result["input_detections"],
        "final_detections": result["final_detections"],
        "output_classes": result["output_classes"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
