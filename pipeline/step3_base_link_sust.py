#!/usr/bin/env python3
"""Run Step2 and Step3 on SUST labels whose boxes are already in ``base_link``.

The geometry implementation fits against ``lidar_top`` points.  This adapter
temporarily converts input SUST boxes from ``base_link`` to ``lidar_top``, runs
the normal Step2 identity/class/yaw stage, invokes Step3, converts the fitted
boxes back, and writes SUST labels.  It does not run Step5 or any final filter.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry.car_box_fit import CarBoxFitConfig, apply_car_box_fit
from pipeline import step2_identity_class_filter_yaw as step2
from tracking import tracker_conservative as tracking


def _label_box(label: Mapping[str, Any]) -> List[float]:
    psr = label.get("psr", {})
    position = psr.get("position", {})
    rotation = psr.get("rotation", {})
    scale = psr.get("scale", {})
    values = [
        position.get("x"), position.get("y"), position.get("z"),
        scale.get("x"), scale.get("y"), scale.get("z"),
        rotation.get("z"),
    ]
    if any(value is None for value in values):
        raise ValueError("SUST label is missing a complete psr box")
    box = [float(value) for value in values]
    if not bool(np.all(np.isfinite(np.asarray(box, dtype=np.float64)))):
        raise ValueError("SUST label contains a non-finite psr box")
    return box


def _track_id(label: Mapping[str, Any]) -> int:
    try:
        return int(label["obj_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"SUST obj_id must be an integer: {label.get('obj_id')}") from exc


def _base_to_lidar(box: Sequence[float], inverse: np.ndarray) -> List[float]:
    return tracking.box_lidar_to_base_link(box, inverse)


def load_sust_frames(clip: Path, coords: tracking.CoordinateProvider
                     ) -> List[Dict[str, Any]]:
    label_dir = clip / "label"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"clip lacks label/: {clip}")
    label_files = sorted(label_dir.glob("*.json"), key=lambda path: int(path.stem))
    if not label_files:
        raise FileNotFoundError(f"clip has no label/*.json: {clip}")

    lidar_top_from_base = np.linalg.inv(coords.base_from_lidar_top)
    frames: List[Dict[str, Any]] = []
    for label_path in label_files:
        text = label_path.read_text(encoding="utf-8")
        labels = json.loads(text) if text.strip() else []
        if not isinstance(labels, list):
            raise ValueError(f"SUST label must be a list: {label_path}")
        detections: List[Dict[str, Any]] = []
        for label in labels:
            if not isinstance(label, Mapping):
                raise ValueError(f"SUST object must be a mapping: {label_path}")
            det: Dict[str, Any] = {
                "track_id": _track_id(label),
                "class_name": str(label.get("obj_type", "")),
                "score": float(label.get("score", 0.0)),
                "box_lidar": _base_to_lidar(
                    _label_box(label), lidar_top_from_base),
            }
            if "visibility" in label:
                det["visibility"] = copy.deepcopy(label["visibility"])
            detections.append(det)
        frames.append({
            "frame_id": label_path.stem,
            "detections": detections,
            "num_detections": len(detections),
        })
    return frames


def _write_sust_labels(frames: Sequence[Mapping[str, Any]], destination: Path,
                       coords: tracking.CoordinateProvider) -> int:
    label_dir = destination / "label"
    if label_dir.exists():
        shutil.rmtree(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    base_from_lidar_top = coords.base_from_lidar_top
    count = 0
    for frame in frames:
        labels: List[Dict[str, Any]] = []
        for source_det in frame.get("detections", []):
            det = copy.deepcopy(source_det)
            det["box_lidar"] = tracking.box_lidar_to_base_link(
                det["box_lidar"], base_from_lidar_top)
            labels.append(tracking.box_to_label(det))
        count += len(labels)
        (label_dir / f"{frame['frame_id']}.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    return count


def run_clip(clip: Path, output_root: Path, overwrite: bool = False) -> Dict[str, Any]:
    coords = tracking.CoordinateProvider(clip)
    source_frames = load_sust_frames(clip, coords)
    with tempfile.TemporaryDirectory(prefix="step3_base_link_sust_") as temp:
        temp_root = Path(temp)
        input_json = temp_root / "input.json"
        step2_json = temp_root / "step2.json"
        step2_diagnostics_path = temp_root / "step2_diagnostics.json"
        input_json.write_text(
            json.dumps(source_frames, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        step2_diagnostics = step2.run(
            input_json, clip, step2_json, None, step2_diagnostics_path)
        frames = json.loads(step2_json.read_text(encoding="utf-8"))
        fitted, diagnostics = apply_car_box_fit(
            frames, coords, clip,
            step2_diagnostics["tracking"],
            step2_diagnostics["static_yaw_stabilization"],
            CarBoxFitConfig())

    destination = output_root / clip.name
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"output exists, pass --overwrite: {destination}")
        shutil.rmtree(destination)
    shutil.copytree(clip, destination, ignore=shutil.ignore_patterns("label"))
    labels = _write_sust_labels(fitted, destination, coords)
    return {
        "input_clip": str(clip.resolve()),
        "output_clip": str(destination.resolve()),
        "input_frames": len(source_frames),
        "input_detections": sum(len(f["detections"]) for f in source_frames),
        "step2_detections": sum(len(f.get("detections", [])) for f in frames),
        "output_labels": labels,
        "car_boxes": diagnostics["car_boxes"],
        "roof_evidence_boxes": diagnostics["roof_evidence_boxes"],
        "roof_rejections": diagnostics["roof_rejections"],
        "z_modes": {
            key: diagnostics[key]
            for key in ("z_both_boxes", "z_ground_boxes", "z_roof_boxes",
                        "z_fallback_boxes")
        },
        "invariant_check": diagnostics["invariant_check"],
    }


def _valid_clip(path: Path) -> bool:
    bins = path / "lidar" / "lidar_top"
    return (path.is_dir() and (path / "label").is_dir()
            and (path / "transforms" / "pose_data.txt").is_file()
            and (path / "transforms" / "calib.json").is_file()
            and any(p.stat().st_size > 0 for p in bins.glob("*.bin")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", action="append", type=Path, default=[])
    parser.add_argument("--clip-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    clips = [Path(path).resolve() for path in args.clip]
    if args.clip_dir:
        clips.extend(sorted(
            path.resolve() for path in args.clip_dir.iterdir()
            if _valid_clip(path)))
    if not clips:
        raise SystemExit("provide --clip or --clip-dir")
    invalid = [path for path in clips if not _valid_clip(path)]
    if invalid:
        raise SystemExit("invalid or empty SUST clips:\n" +
                         "\n".join(str(path) for path in invalid))

    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, clip in enumerate(clips, 1):
        print(f"[{index}/{len(clips)}] {clip.name}", flush=True)
        summaries.append(run_clip(clip, args.output_root, args.overwrite))
    print(json.dumps({"clips": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
