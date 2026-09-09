#!/usr/bin/env python3
"""Step 4.5 batch: dynamic-region re-tracking / ID inheritance / phase stitch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from region.dynamic_region import DynamicRegionConfig
from region.retrack import Step45Config
from pipeline.step4_5_region_phase_retrack import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step4-work-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step4")
    parser.add_argument("--step2-work-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step2")
    parser.add_argument("--clip-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step3_car_box_fit" / "data")
    parser.add_argument("--work-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step4_5")
    parser.add_argument("--extension-length-m", type=float, default=30.0)
    parser.add_argument("--phase-merge-max-gap-sec", type=float, default=30.0)
    parser.add_argument("--yielding-max-gap-sec", type=float, default=6.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_jsons = sorted(args.step4_work_root.glob("*_step4.json"))
    if not input_jsons:
        raise SystemExit(
            f"no *_step4.json found under {args.step4_work_root}")
    args.work_root.mkdir(parents=True, exist_ok=True)
    config = Step45Config(
        region=DynamicRegionConfig(
            extension_length_m=args.extension_length_m),
        phase_merge_max_gap_sec=args.phase_merge_max_gap_sec,
        yielding_max_gap_sec=args.yielding_max_gap_sec,
    )
    summaries = []
    for index, step4_json in enumerate(input_jsons, start=1):
        clip_name = step4_json.name[:-len("_step4.json")]
        clip = args.clip_root / f"{clip_name}_step3"
        if not clip.is_dir():
            raise SystemExit(f"missing clip for {clip_name}: {clip}")
        step2_diagnostics = (
            args.step2_work_root / f"{clip_name}_step2_diagnostics.json")
        if not step2_diagnostics.is_file():
            raise SystemExit(
                f"missing step2 diagnostics for {clip_name}: "
                f"{step2_diagnostics}")
        out_json = args.work_root / f"{clip_name}_step45.json"
        diagnostics = args.work_root / f"{clip_name}_step45_diagnostics.json"
        if (out_json.exists() or diagnostics.exists()) and not args.overwrite:
            raise SystemExit(f"output exists, pass --overwrite: {out_json}")
        print(f"[{index}/{len(input_jsons)}] {clip_name}", flush=True)
        result = run(step4_json, clip, step2_diagnostics,
                     out_json, diagnostics, config)
        summaries.append({
            "clip": clip_name,
            "out_json": str(out_json),
            "diagnostics": str(diagnostics),
            "candidate_tracks": result["candidate_tracks"],
            "retrackable_detections": result["selection"][
                "retrackable_detections"],
            "phase_merges": len(result["phase_stitching"]["applied"]),
            "static_freeze_passed": result["static_freeze"]["passed"],
        })
    summary_path = args.work_root / "batch_summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({"clips": summaries, "summary": str(summary_path)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
