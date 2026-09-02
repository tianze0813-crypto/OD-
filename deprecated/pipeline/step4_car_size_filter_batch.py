#!/usr/bin/env python3
"""Step 4 batch: relabel truck-sized Car tracks as Truck."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filtering.car_size_filter import LargeCarFilterConfig
from deprecated.pipeline.step4_car_size_filter import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step3-work-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step3_car_box_fit")
    parser.add_argument("--work-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step4_car_size_filter")
    parser.add_argument("--truck-length-min", type=float, default=6.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    step3_jsons = sorted(args.step3_work_root.glob("*_step3.json"))
    if not step3_jsons:
        raise SystemExit(f"no *_step3.json found under {args.step3_work_root}")
    args.work_root.mkdir(parents=True, exist_ok=True)
    config = LargeCarFilterConfig(truck_length_min=args.truck_length_min)
    summaries = []
    for index, step3_json in enumerate(step3_jsons, start=1):
        clip_name = step3_json.name[:-len("_step3.json")]
        out_json = args.work_root / f"{clip_name}_step4.json"
        diagnostics = args.work_root / f"{clip_name}_step4_diagnostics.json"
        if (out_json.exists() or diagnostics.exists()) and not args.overwrite:
            raise SystemExit(f"output exists, pass --overwrite: {out_json}")
        print(f"[{index}/{len(step3_jsons)}] {clip_name}", flush=True)
        result = run(step3_json, out_json, diagnostics, config)
        summaries.append({
            "clip": clip_name,
            "out_json": str(out_json),
            "diagnostics": str(diagnostics),
            "large_car_tracks_relabelled": result[
                "large_car_tracks_relabelled"],
            "large_car_detections_relabelled": result[
                "large_car_detections_relabelled"],
        })

    summary_path = args.work_root / "batch_summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"clips": summaries, "summary": str(summary_path)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
