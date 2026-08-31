#!/usr/bin/env python3
"""Legacy Step 6 batch: keep only Car labels over step-5 clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.step6_car_only_filter import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step5-work-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step5_class_motion_filter")
    parser.add_argument("--clip-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step5_class_motion_filter" / "data")
    parser.add_argument("--out-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step6_car_only_filter" / "data")
    parser.add_argument("--work-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step6_car_only_filter")
    parser.add_argument("--suffix", type=str, default="_step6")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    step5_jsons = sorted(args.step5_work_root.glob("*_step5.json"))
    if not step5_jsons:
        raise SystemExit(f"no *_step5.json found under {args.step5_work_root}")
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index, step5_json in enumerate(step5_jsons, start=1):
        clip_name = step5_json.name[:-len("_step5.json")]
        clip = args.clip_root / f"{clip_name}_step5"
        if not clip.is_dir():
            raise SystemExit(f"missing step-5 clip for {clip_name}: {clip}")
        out_json = args.work_root / f"{clip_name}_step6.json"
        diagnostics = args.work_root / f"{clip_name}_step6_diagnostics.json"
        out_clip = args.out_root / f"{clip_name}{args.suffix}"
        if (out_clip.exists() or out_json.exists()) and not args.overwrite:
            raise SystemExit(f"output exists, pass --overwrite: {out_clip}")
        print(f"[{index}/{len(step5_jsons)}] {clip_name}", flush=True)
        result = run(step5_json, clip, out_json, out_clip, diagnostics)
        summaries.append({
            "clip": clip_name,
            "out_clip": str(out_clip),
            "out_json": str(out_json),
            "diagnostics": str(diagnostics),
            "before_detections": result["before_detections"],
            "after_detections": result["after_detections"],
            "detections_removed": result["detections_removed"],
            "classes_removed": result["classes_removed"],
        })

    summary_path = args.work_root / "batch_summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"clips": summaries, "summary": str(summary_path)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
