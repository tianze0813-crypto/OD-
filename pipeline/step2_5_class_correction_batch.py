#!/usr/bin/env python3
"""Batch runner for :mod:`pipeline.step2_5_class_correction`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.step2_5_class_correction import _hard_config, run
from tracking import tracker_conservative as tracking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step2-root", type=Path, default=PROJECT_ROOT / "work" / "step2_identity")
    parser.add_argument("--step2-json", action="append", dest="step2_jsons")
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=PROJECT_ROOT / "work" / "step2_5_class_correction")
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-lifecycle", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--car-score-threshold", type=float, default=0.25)
    parser.add_argument("--truck-score-threshold", type=float, default=0.4)
    parser.add_argument("--bus-score-threshold", type=float, default=0.4)
    parser.add_argument("--pedestrian-score-threshold", type=float, default=0.3)
    parser.add_argument("--nonmotorized-score-threshold", type=float, default=0.3)
    parser.add_argument("--range-front", type=float, default=80.0)
    parser.add_argument("--range-rear", type=float, default=20.0)
    parser.add_argument("--range-side", type=float, default=40.0)
    parser.add_argument("--sparsity-max-points", type=int, default=10)
    parser.add_argument("--visibility-min-ratio", type=float, default=0.05)
    parser.add_argument("--pedestrian-max-distance", type=float, default=20.0)
    parser.add_argument("--keep-classes", default=",".join(tracking.TARGET_CLASSES))
    args = parser.parse_args()
    files = ([Path(path).resolve() for path in args.step2_jsons]
             if args.step2_jsons else sorted(args.step2_root.glob("*_step2.json")))
    if not files:
        raise SystemExit(f"no *_step2.json found under {args.step2_root}")
    args.work_root.mkdir(parents=True, exist_ok=True)
    if args.out_root is not None:
        args.out_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, step2_json in enumerate(files, 1):
        name = step2_json.name[:-len("_step2.json")] if step2_json.name.endswith("_step2.json") else step2_json.stem
        clip = args.clip_root / name
        step2_diag = step2_json.with_name(step2_json.stem + "_diagnostics.json")
        if not clip.is_dir() or not step2_diag.is_file():
            raise SystemExit(f"missing clip or Step2 diagnostics for {name}")
        out_json = args.work_root / f"{name}_step2_5.json"
        diag_path = args.work_root / f"{name}_step2_5_diagnostics.json"
        out_clip = None if args.out_root is None else args.out_root / f"{name}_step2_5"
        if not args.overwrite and (out_json.exists() or (out_clip and out_clip.exists())):
            raise SystemExit(f"output exists, pass --overwrite: {out_json}")
        print(f"[{index}/{len(files)}] {name}", flush=True)
        diagnostics = run(step2_json, step2_diag, clip, out_json, out_clip,
                          diag_path, hard_filter_config=_hard_config(args),
                          min_lifecycle=args.min_lifecycle)
        summaries.append({
            "clip": name,
            "out_json": str(out_json),
            "diagnostics": str(diag_path),
            "out_clip": str(out_clip) if out_clip else None,
            "class_changed": diagnostics["class_correction"]["detections_changed"],
            "hard_filter_removed": diagnostics["hard_filters_pass_2"]["detections_removed"],
            "short_tracks_removed": diagnostics["short_track_filter"]["tracks_dropped"],
            "final_detections": diagnostics["final_detections"],
        })
    summary_path = args.work_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"clips": summaries, "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
