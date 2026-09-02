#!/usr/bin/env python3
"""Step 2 batch: identity tracking, hard filtering, and integrated yaw."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deprecated.pipeline.step2_identity_class_filter_yaw import run


def clip_name_from_raw(path: Path) -> str:
    suffix = "_raw.json"
    return path.name[:-len(suffix)] if path.name.endswith(suffix) else path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step1_inference")
    parser.add_argument("--raw-json", action="append", dest="raw_jsons")
    parser.add_argument("--clip-root", type=Path,
                        default=Path("/media/moga/police/交警OD预标注/outputs"))
    parser.add_argument("--out-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step2_identity" / "data")
    parser.add_argument("--work-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step2_identity")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_files = ([Path(path).resolve() for path in args.raw_jsons]
                 if args.raw_jsons else sorted(args.raw_root.glob("*_raw.json")))
    if not raw_files:
        raise SystemExit(f"no *_raw.json found under {args.raw_root}")
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.out_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, raw_path in enumerate(raw_files, start=1):
        clip_name = clip_name_from_raw(raw_path)
        clip = args.clip_root / clip_name
        if not clip.is_dir():
            raise SystemExit(f"matching clip not found: {clip}")
        out_clip = args.out_root / f"{clip_name}_step2"
        if out_clip.exists() and not args.overwrite:
            raise SystemExit(f"output exists, pass --overwrite: {out_clip}")
        out_json = args.work_root / f"{clip_name}_step2.json"
        diagnostics_path = args.work_root / f"{clip_name}_step2_diagnostics.json"
        print(f"[{index}/{len(raw_files)}] {clip_name}", flush=True)
        diagnostics = run(
            raw_path, clip, out_json, out_clip, diagnostics_path)
        summaries.append({
            "clip": clip_name,
            "raw_json": str(raw_path),
            "out_clip": str(out_clip),
            "out_json": str(out_json),
            "diagnostics": str(diagnostics_path),
            "input_detections": diagnostics["input_detections"],
            "hard_filter_removed": diagnostics["hard_filters"]["detections_removed"],
            "tracks": diagnostics["tracking"].get("tracks_total"),
            "class_changes": diagnostics["class_finalization"]["detections_changed"],
            "yaw_boxes_by_mode": diagnostics["yaw_integrated"]["boxes_by_mode"],
            "final_detections": diagnostics["final_detections"],
        })
    summary_path = args.work_root / "batch_summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"clips": summaries, "summary": str(summary_path)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
