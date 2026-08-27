#!/usr/bin/env python3
"""Step 5: final sparse/short-track filtering and box-frame conversion.

Point counts are evaluated in the original ``lidar_top`` cloud.  Kept boxes
are converted to ``base_link`` after filtering; the point-cloud files are not
rewritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filtering.final_filter import FinalFilterConfig, apply_final_filter
from tracking import tracker_conservative as tracking


def run(step4_json: Path, clip: Path, out_json: Path,
        out_clip: Path | None, diagnostics_path: Path,
        config: FinalFilterConfig = FinalFilterConfig()
        ) -> dict:
    frames = json.loads(step4_json.read_text(encoding="utf-8"))
    coords = tracking.CoordinateProvider(Path(clip))
    output, result = apply_final_filter(frames, Path(clip), coords, config)
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
    parser.add_argument("--sparsity-max-points", type=int, default=5,
                        help="remove boxes containing this many points or fewer")
    parser.add_argument("--short-track-max-frames", type=int, default=3,
                        help="remove tracks observed in this many frames or fewer")
    args = parser.parse_args()
    config = FinalFilterConfig(
        max_points_in_box=args.sparsity_max_points,
        max_track_length=args.short_track_max_frames,
    )
    diagnostics_path = args.diagnostics or args.out_json.with_name(
        args.out_json.stem + "_diagnostics.json")
    result = run(args.step4_json, args.clip, args.out_json, args.out_clip,
                 diagnostics_path, config)
    print(json.dumps({
        "before_detections": result["before_detections"],
        "after_detections": result["after_detections"],
        "point_filter_removed": result["point_filter_removed"],
        "short_track_removed": result["short_track_removed"],
        "boxes_converted": result["boxes_converted"],
        "labels": result.get("labels"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
