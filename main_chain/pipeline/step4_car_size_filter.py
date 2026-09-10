#!/usr/bin/env python3
"""Step 4: Car -> Truck size gate, then keep only Car for step 4.5.

The reviewed order is fixed:

1. relabel truck-sized ``Car`` tracks as ``Truck``;
2. remove every detection whose canonical export class is not ``Car``.

Only canonical ``Car`` tracks reach step 4.5.  Large cars that were relabelled
to ``Truck`` in step 1 are therefore removed as well.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filtering.car_only_filter import apply_car_only_filter
from filtering.car_size_filter import LargeCarFilterConfig, apply_large_car_to_truck


def run(step3_json: Path, out_json: Path, diagnostics_path: Path,
        config: LargeCarFilterConfig = LargeCarFilterConfig()) -> dict:
    frames = json.loads(Path(step3_json).read_text(encoding="utf-8"))
    relabelled, result = apply_large_car_to_truck(frames, config)
    output, car_only = apply_car_only_filter(relabelled)
    result.update({
        "pipeline": "step4_car_size_filter_then_car_only",
        "source_step3_json": str(Path(step3_json).resolve()),
        "car_only": car_only,
        "before_detections": car_only["before_detections"],
        "after_detections": car_only["after_detections"],
        "car_only_removed": car_only["detections_removed"],
        "classes_removed": car_only["classes_removed"],
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
        "large_car_detections_relabelled", "before_detections",
        "after_detections", "car_only_removed", "classes_removed")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
