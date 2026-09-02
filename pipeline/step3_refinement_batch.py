#!/usr/bin/env python3
"""Batch runner for :mod:`pipeline.step3_refinement`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.step3_refinement import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step2-5-root", type=Path, default=PROJECT_ROOT / "work" / "step2_5_class_correction")
    parser.add_argument("--step2-5-json", action="append", dest="step2_5_jsons")
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=PROJECT_ROOT / "work" / "step3_refinement")
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    files = ([Path(path).resolve() for path in args.step2_5_jsons]
             if args.step2_5_jsons else sorted(args.step2_5_root.glob("*_step2_5.json")))
    if not files:
        raise SystemExit(f"no *_step2_5.json found under {args.step2_5_root}")
    args.work_root.mkdir(parents=True, exist_ok=True)
    if args.out_root is not None:
        args.out_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, source_json in enumerate(files, 1):
        name = source_json.name[:-len("_step2_5.json")] if source_json.name.endswith("_step2_5.json") else source_json.stem
        clip = args.clip_root / name
        source_diag = source_json.with_name(source_json.stem + "_diagnostics.json")
        if not clip.is_dir() or not source_diag.is_file():
            raise SystemExit(f"missing clip or Step2.5 diagnostics for {name}")
        out_json = args.work_root / f"{name}_step3.json"
        diag_path = args.work_root / f"{name}_step3_diagnostics.json"
        out_clip = None if args.out_root is None else args.out_root / f"{name}_step3"
        if not args.overwrite and (out_json.exists() or (out_clip and out_clip.exists())):
            raise SystemExit(f"output exists, pass --overwrite: {out_json}")
        print(f"[{index}/{len(files)}] {name}", flush=True)
        diagnostics = run(source_json, source_diag, clip, out_json, out_clip, diag_path)
        summaries.append({
            "clip": name,
            "out_json": str(out_json),
            "diagnostics": str(diag_path),
            "out_clip": str(out_clip) if out_clip else None,
            "tracks": diagnostics["tracks"],
            "car_tracks": diagnostics["car_tracks"],
            "car_boxes": diagnostics["car_boxes"],
            "final_detections": diagnostics["final_detections"],
        })
    summary_path = args.work_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"clips": summaries, "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
