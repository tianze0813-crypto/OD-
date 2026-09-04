#!/usr/bin/env python3
"""Batch runner for :mod:`pipeline.step2_identity`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.step2_identity import _hard_config, run


def _clip_name(path: Path) -> str:
    return path.name[:-len("_raw.json")] if path.name.endswith("_raw.json") else path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "work" / "step1_inference")
    parser.add_argument("--raw-json", action="append", dest="raw_jsons")
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=PROJECT_ROOT / "work" / "step2_identity")
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--car-score-threshold", type=float, default=0.25)
    parser.add_argument("--truck-score-threshold", type=float, default=0.25)
    parser.add_argument("--bus-score-threshold", type=float, default=0.25)
    parser.add_argument("--pedestrian-score-threshold", type=float, default=0.3)
    parser.add_argument("--nonmotorized-score-threshold", type=float, default=0.3)
    parser.add_argument("--range-front", type=float, default=80.0)
    parser.add_argument("--range-rear", type=float, default=20.0)
    parser.add_argument("--range-side", type=float, default=40.0)
    parser.add_argument("--sparsity-max-points", type=int, default=10)
    parser.add_argument("--visibility-min-ratio", type=float, default=0.05)
    parser.add_argument("--pedestrian-max-distance", type=float, default=20.0)
    parser.add_argument("--keep-classes", default="Car,Truck,Bus,Pedestrian,Nonmotorized_vehicle")
    parser.add_argument("--same-center-gate", type=float, default=0.35)
    args = parser.parse_args()
    raw_files = ([Path(path).resolve() for path in args.raw_jsons]
                 if args.raw_jsons else sorted(args.raw_root.glob("*_raw.json")))
    if not raw_files:
        raise SystemExit(f"no *_raw.json found under {args.raw_root}")
    args.work_root.mkdir(parents=True, exist_ok=True)
    out_root = args.out_root
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, raw_path in enumerate(raw_files, 1):
        name = _clip_name(raw_path)
        clip = args.clip_root / name
        if not clip.is_dir():
            raise SystemExit(f"matching clip not found: {clip}")
        out_json = args.work_root / f"{name}_step2.json"
        diagnostics_path = args.work_root / f"{name}_step2_diagnostics.json"
        out_clip = None if out_root is None else out_root / f"{name}_step2"
        if not args.overwrite and (out_json.exists() or (out_clip and out_clip.exists())):
            raise SystemExit(f"output exists, pass --overwrite: {out_json}")
        print(f"[{index}/{len(raw_files)}] {name}", flush=True)
        diagnostics = run(
            raw_path, clip, out_json, out_clip, diagnostics_path,
            hard_filter_config=_hard_config(args),
            same_center_gate=args.same_center_gate)
        summaries.append({
            "clip": name,
            "out_json": str(out_json),
            "diagnostics": str(diagnostics_path),
            "out_clip": str(out_clip) if out_clip else None,
            "tracks": diagnostics["tracking"].get("tracks_total"),
            "hard_filter_removed": diagnostics["hard_filters_pass_1"]["detections_removed"],
            "final_detections": diagnostics["final_detections"],
        })
    summary_path = args.work_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"clips": summaries, "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
