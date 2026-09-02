#!/usr/bin/env python3
"""Legacy Step 6: keep only Car labels after a step-5 result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filtering.car_only_filter import apply_car_only_filter
from tracking import tracker_conservative as tracking


def run(step5_json: Path, clip: Path, out_json: Path,
        out_clip: Path | None, diagnostics_path: Path) -> dict:
    frames = json.loads(step5_json.read_text(encoding="utf-8"))
    output, result = apply_car_only_filter(frames)
    result.update({
        "source_step5_json": str(step5_json.resolve()),
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
    parser.add_argument("--step5-json", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-clip", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    diagnostics_path = args.diagnostics or args.out_json.with_name(
        args.out_json.stem + "_diagnostics.json")
    result = run(args.step5_json, args.clip, args.out_json, args.out_clip,
                 diagnostics_path)
    print(json.dumps({
        "before_detections": result["before_detections"],
        "after_detections": result["after_detections"],
        "detections_removed": result["detections_removed"],
        "classes_removed": result["classes_removed"],
        "labels": result.get("labels"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
