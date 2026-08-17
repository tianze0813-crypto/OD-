#!/usr/bin/env python3
"""Step 3 batch: Car box fitting over all completed step-2 clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.step3_car_box_fit import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step2-work-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step2_identity")
    parser.add_argument("--clip-root", type=Path,
                        default=Path("/media/moga/police/交警OD预标注/outputs"))
    parser.add_argument("--out-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step3_car_box_fit" / "data")
    parser.add_argument("--work-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step3_car_box_fit")
    parser.add_argument("--suffix", type=str, default="_step3")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    step2_jsons = sorted(args.step2_work_root.glob("*_step2.json"))
    if not step2_jsons:
        raise SystemExit(f"no *_step2.json found under {args.step2_work_root}")
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index, step2_json in enumerate(step2_jsons, start=1):
        clip_name = step2_json.name[:-len("_step2.json")]
        clip = args.clip_root / clip_name
        step2_diagnostics = step2_json.with_name(step2_json.stem + "_diagnostics.json")
        if not clip.is_dir() or not step2_diagnostics.is_file():
            raise SystemExit(f"missing clip or step-2 diagnostics for {clip_name}")
        out_json = args.work_root / f"{clip_name}_step3.json"
        diagnostics = args.work_root / f"{clip_name}_step3_diagnostics.json"
        out_clip = args.out_root / f"{clip_name}{args.suffix}"
        if (out_clip.exists() or out_json.exists()) and not args.overwrite:
            raise SystemExit(f"output exists, pass --overwrite: {out_clip}")
        print(f"[{index}/{len(step2_jsons)}] {clip_name}", flush=True)
        result = run(step2_json, step2_diagnostics, clip, out_json, out_clip,
                     diagnostics)
        summaries.append({
            "clip": clip_name,
            "out_clip": str(out_clip),
            "out_json": str(out_json),
            "diagnostics": str(diagnostics),
            "tracks": result.get("tracks"),
            "car_tracks": result.get("car_tracks"),
            "car_boxes": result.get("car_boxes"),
            "static_boxes": result["static_boxes"],
            "dynamic_boxes": result["dynamic_boxes"],
            "both_side_boxes": result.get("both_side_boxes"),
            "single_side_boxes": result.get("single_side_boxes"),
            "unchanged_xy_boxes": result.get("unchanged_xy_boxes"),
            "size_smoothed_boxes": result.get("size_smoothed_boxes"),
            "z_both_boxes": result.get("z_both_boxes"),
            "z_ground_boxes": result.get("z_ground_boxes"),
            "z_roof_boxes": result.get("z_roof_boxes"),
            "z_fallback_boxes": result.get("z_fallback_boxes"),
            "ground_adjusted_boxes": result["ground_adjusted_boxes"],
            "roof_adjusted_boxes": result["roof_adjusted_boxes"],
            "invariant_check": result["invariant_check"],
        })

    summary_path = args.work_root / "batch_summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"clips": summaries, "summary": str(summary_path)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
