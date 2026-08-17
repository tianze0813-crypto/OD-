#!/usr/bin/env python3
"""Step 4 batch: Truck size fitting, interpolation, and overlap filtering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.step4_truck_size_interp_overlap import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step3-work-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step3_car_box_fit")
    parser.add_argument("--clip-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step3_car_box_fit" / "data")
    parser.add_argument("--out-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step4_truck_size_interp_overlap" / "data")
    parser.add_argument("--work-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step4_truck_size_interp_overlap")
    parser.add_argument("--suffix", type=str, default="_step4")
    parser.add_argument("--sust-data-root", type=Path,
                        default=PROJECT_ROOT.parent / "SUSTechPOINTS" / "data")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    step3_jsons = sorted(args.step3_work_root.glob("*_step3.json"))
    if not step3_jsons:
        raise SystemExit(f"no *_step3.json found under {args.step3_work_root}")
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index, step3_json in enumerate(step3_jsons, start=1):
        clip_name = step3_json.name[:-len("_step3.json")]
        clip = args.clip_root / f"{clip_name}_step3"
        if not clip.is_dir():
            raise SystemExit(f"missing step-3 clip for {clip_name}: {clip}")
        out_json = args.work_root / f"{clip_name}_step4.json"
        diagnostics = args.work_root / f"{clip_name}_step4_diagnostics.json"
        out_clip = args.out_root / f"{clip_name}{args.suffix}"
        if (out_clip.exists() or out_json.exists()) and not args.overwrite:
            raise SystemExit(f"output exists, pass --overwrite: {out_clip}")
        print(f"[{index}/{len(step3_jsons)}] {clip_name}", flush=True)
        result = run(step3_json, clip, out_json, out_clip, diagnostics,
                     args.sust_data_root)
        summaries.append({
            "clip": clip_name,
            "out_clip": str(out_clip),
            "out_json": str(out_json),
            "diagnostics": str(diagnostics),
            "truck_tracks": result.get("truck_tracks"),
            "before_detections": result.get("before_detections"),
            "after_detections": result.get("after_detections"),
            "interpolated_truck_boxes": result.get("interpolated_truck_boxes"),
            "truck_overlap_filter": result.get("truck_overlap_filter"),
            "visibility": result.get("visibility"),
            "sust_copy": result.get("sust_copy"),
        })

    summary_path = args.work_root / "batch_summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"clips": summaries, "summary": str(summary_path)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
