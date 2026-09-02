#!/usr/bin/env python3
"""Pipeline entry point for final five-class normalization and box conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from filtering.five_class_output import apply_five_class_output
from tracking import tracker_conservative as tracking


def run(input_json: Path, clip: Path, output_json: Path,
        diagnostics_path: Path | None = None) -> dict:
    frames = json.loads(Path(input_json).read_text(encoding="utf-8"))
    output, diagnostics = apply_five_class_output(
        frames, tracking.CoordinateProvider(Path(clip)))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    target = diagnostics_path or output_json.with_name(
        output_json.stem + "_diagnostics.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", "--step3-json", dest="input_json",
                        type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output-json", "--out-json", dest="output_json",
                        type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.input_json, args.clip, args.output_json,
                         args.diagnostics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
