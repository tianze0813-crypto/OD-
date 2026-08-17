#!/usr/bin/env python3
"""Step 5 batch: optional low-confidence class filters over step-4 clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filtering.low_confidence_class_filter import LowConfidenceClassFilterConfig
from pipeline.step5_class_motion_filter import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step4-work-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step4_truck_size_interp_overlap")
    parser.add_argument("--clip-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step4_truck_size_interp_overlap" / "data")
    parser.add_argument("--out-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step5_class_motion_filter" / "data")
    parser.add_argument("--work-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step5_class_motion_filter")
    parser.add_argument("--suffix", type=str, default="_step5")
    parser.add_argument("--keep-truck", action="store_true")
    parser.add_argument("--keep-static-nonmotorized", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    step4_jsons = sorted(args.step4_work_root.glob("*_step4.json"))
    if not step4_jsons:
        raise SystemExit(f"no *_step4.json found under {args.step4_work_root}")
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    config = LowConfidenceClassFilterConfig(
        drop_truck=not args.keep_truck,
        drop_static_nonmotorized=not args.keep_static_nonmotorized)
    summaries = []
    for index, step4_json in enumerate(step4_jsons, start=1):
        clip_name = step4_json.name[:-len("_step4.json")]
        clip = args.clip_root / f"{clip_name}_step4"
        if not clip.is_dir():
            raise SystemExit(f"missing step-4 clip for {clip_name}: {clip}")
        out_json = args.work_root / f"{clip_name}_step5.json"
        diagnostics = args.work_root / f"{clip_name}_step5_diagnostics.json"
        out_clip = args.out_root / f"{clip_name}{args.suffix}"
        if (out_clip.exists() or out_json.exists()) and not args.overwrite:
            raise SystemExit(f"output exists, pass --overwrite: {out_clip}")
        print(f"[{index}/{len(step4_jsons)}] {clip_name}", flush=True)
        result = run(step4_json, clip, out_json, out_clip, diagnostics, config)
        summaries.append({
            "clip": clip_name,
            "out_clip": str(out_clip),
            "out_json": str(out_json),
            "diagnostics": str(diagnostics),
            "before_detections": result["before_detections"],
            "after_detections": result["after_detections"],
            "truck_boxes_removed": result["truck_boxes_removed"],
            "static_nonmotorized_tracks_removed": result[
                "static_nonmotorized_tracks_removed"],
            "static_nonmotorized_boxes_removed": result[
                "static_nonmotorized_boxes_removed"],
        })

    summary_path = args.work_root / "batch_summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"clips": summaries, "summary": str(summary_path)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
