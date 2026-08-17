#!/usr/bin/env python3
"""Step 5: optional low-confidence class filters.

By default all Truck boxes are removed.  Non-motorized (Cyclist) tracks that
never move are removed as well; cyclists that wait and later move are kept.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filtering.low_confidence_class_filter import (
    LowConfidenceClassFilterConfig,
    apply_low_confidence_class_filter,
)
from tracking import tracker_conservative as tracking


def run(step4_json: Path, clip: Path, out_json: Path,
        out_clip: Path | None, diagnostics_path: Path,
        config: LowConfidenceClassFilterConfig = LowConfidenceClassFilterConfig()
        ) -> dict:
    frames = json.loads(step4_json.read_text(encoding="utf-8"))
    coords = tracking.CoordinateProvider(Path(clip))
    output, result = apply_low_confidence_class_filter(frames, coords, config)
    result.update({
        "source_step4_json": str(step4_json.resolve()),
        "source_clip": str(Path(clip).resolve()),
    })
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    if out_clip is not None:
        result["labels"] = tracking.export_clip(
            output, Path(clip), Path(out_clip))
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step4-json", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-clip", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--keep-truck", action="store_true",
                        help="do not remove Truck detections")
    parser.add_argument("--keep-static-nonmotorized", action="store_true",
                        help="do not remove static non-motorized tracks")
    args = parser.parse_args()
    config = LowConfidenceClassFilterConfig(
        drop_truck=not args.keep_truck,
        drop_static_nonmotorized=not args.keep_static_nonmotorized,
    )
    diagnostics_path = args.diagnostics or args.out_json.with_name(
        args.out_json.stem + "_diagnostics.json")
    result = run(args.step4_json, args.clip, args.out_json, args.out_clip,
                 diagnostics_path, config)
    print(json.dumps({
        "before_detections": result["before_detections"],
        "after_detections": result["after_detections"],
        "truck_boxes_removed": result["truck_boxes_removed"],
        "static_nonmotorized_tracks_removed": result[
            "static_nonmotorized_tracks_removed"],
        "static_nonmotorized_boxes_removed": result[
            "static_nonmotorized_boxes_removed"],
        "labels": result.get("labels"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
