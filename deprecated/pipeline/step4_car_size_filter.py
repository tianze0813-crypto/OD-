#!/usr/bin/env python3
"""Step 4: relabel truck-sized Car tracks as Truck for the final Car gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filtering.car_size_filter import LargeCarFilterConfig, apply_large_car_to_truck


def run(step3_json: Path, out_json: Path, diagnostics_path: Path,
        config: LargeCarFilterConfig = LargeCarFilterConfig()) -> dict:
    frames = json.loads(Path(step3_json).read_text(encoding="utf-8"))
    output, result = apply_large_car_to_truck(frames, config)
    result.update({
        "pipeline": "step4_car_size_filter",
        "source_step3_json": str(Path(step3_json).resolve()),
    })
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step3-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--truck-length-min", type=float, default=6.0)
    args = parser.parse_args()
    diagnostics_path = args.diagnostics or args.out_json.with_name(
        args.out_json.stem + "_diagnostics.json")
    result = run(
        args.step3_json, args.out_json, diagnostics_path,
        LargeCarFilterConfig(truck_length_min=args.truck_length_min))
    print(json.dumps({key: result[key] for key in (
        "truck_length_min", "tracks_checked", "large_car_tracks_relabelled",
        "large_car_detections_relabelled")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
