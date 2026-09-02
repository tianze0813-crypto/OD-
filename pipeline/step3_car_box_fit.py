#!/usr/bin/env python3
"""Step 3: class-aware box fitting on one completed step-2 clip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry.box_geometry import GeometryConfig, apply_geometry_legacy
from geometry.car_box_fit import CarBoxFitConfig, apply_car_box_fit
from tracking import tracker_conservative as tracking


def run(step2_json: Path, step2_diagnostics: Path, clip: Path,
        out_json: Path, out_clip: Path | None, diagnostics_path: Path) -> dict:
    frames = json.loads(step2_json.read_text(encoding="utf-8"))
    diagnostics = json.loads(step2_diagnostics.read_text(encoding="utf-8"))
    coords = tracking.CoordinateProvider(Path(clip))
    # The generic geometry pass fits all five classes with their class-aware
    # physical priors.  The reviewed bottom-up roof/face fitter then makes its
    # additional Car-only refinement, preserving the old high-quality path.
    generic_output, generic_result = apply_geometry_legacy(
        frames, coords, Path(clip), diagnostics["tracking"],
        diagnostics["static_yaw_stabilization"], GeometryConfig())
    output, result = apply_car_box_fit(
        generic_output, coords, Path(clip), diagnostics["tracking"],
        diagnostics["static_yaw_stabilization"], CarBoxFitConfig())
    result["multiclass_geometry"] = generic_result
    result["multiclass_tracks"] = generic_result.get("tracks", 0)
    result["multiclass_boxes"] = generic_result.get("boxes", 0)
    result.update({
        "source_step2_json": str(step2_json.resolve()),
        "source_step2_diagnostics": str(step2_diagnostics.resolve()),
        "source_clip": str(Path(clip).resolve()),
    })
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    if out_clip is not None:
        result["labels"] = tracking.export_clip(output, Path(clip), Path(out_clip))
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step2-json", type=Path, required=True)
    parser.add_argument("--step2-diagnostics", type=Path)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-clip", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    step2_diagnostics = args.step2_diagnostics or args.step2_json.with_name(
        args.step2_json.stem + "_diagnostics.json")
    diagnostics_path = args.diagnostics or args.out_json.with_name(
        args.out_json.stem + "_diagnostics.json")
    result = run(args.step2_json, step2_diagnostics, args.clip, args.out_json,
                 args.out_clip, diagnostics_path)
    print(json.dumps({k: result.get(k) for k in (
        "tracks", "car_tracks", "car_boxes", "static_boxes", "dynamic_boxes",
        "both_side_boxes", "single_side_boxes", "unchanged_xy_boxes",
        "size_smoothed_boxes",
        "z_both_boxes", "z_ground_boxes", "z_roof_boxes", "z_fallback_boxes",
        "roof_evidence_boxes", "roof_rejections",
        "ground_adjusted_boxes", "roof_adjusted_boxes", "final_detections",
        "invariant_check", "labels")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
