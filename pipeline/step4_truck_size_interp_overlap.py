#!/usr/bin/env python3
"""Step 4: Truck size fitting, interpolation, and overlap filtering."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry.truck_box_fit import TruckBoxFitConfig, apply_truck_box_fit
from tracking import tracker_conservative as tracking


def run(step3_json: Path, clip: Path, out_json: Path, out_clip: Path | None,
        diagnostics_path: Path, sust_data_root: Path | None = None) -> dict:
    frames = json.loads(step3_json.read_text(encoding="utf-8"))
    coords = tracking.CoordinateProvider(Path(clip))
    output, result = apply_truck_box_fit(frames, coords, Path(clip), TruckBoxFitConfig())
    result.update({
        "source_step3_json": str(step3_json.resolve()),
        "source_clip": str(Path(clip).resolve()),
    })
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    if out_clip is not None:
        result["labels"] = tracking.export_clip(output, Path(clip), Path(out_clip))
        if sust_data_root is not None:
            try:
                dest = Path(sust_data_root) / out_clip.name
                shutil.copytree(out_clip, dest, dirs_exist_ok=True)
                result["sust_copy"] = {"status": "ok", "dest": str(dest)}
            except Exception as exc:
                result["sust_copy"] = {
                    "status": "error", "dest": str(dest),
                    "error": str(exc)[:300]}
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step3-json", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True,
                        help="completed step-3 clip copied by the step-3 stage")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-clip", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--sust-data-root", type=Path,
                        default=None)
    args = parser.parse_args()
    diagnostics_path = args.diagnostics or args.out_json.with_name(
        args.out_json.stem + "_diagnostics.json")
    result = run(args.step3_json, args.clip, args.out_json, args.out_clip,
                 diagnostics_path, args.sust_data_root)
    print(json.dumps({
        "truck_tracks": result.get("truck_tracks"),
        "before_detections": result.get("before_detections"),
        "after_detections": result.get("after_detections"),
        "interpolated_truck_boxes": result.get("interpolated_truck_boxes"),
        "truck_overlap_filter": result.get("truck_overlap_filter"),
        "visibility": result.get("visibility"),
        "labels": result.get("labels"),
        "sust_copy": result.get("sust_copy"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
