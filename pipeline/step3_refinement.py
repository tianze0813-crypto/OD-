#!/usr/bin/env python3
"""Step 3: public yaw followed by class-specific geometry refinement.

Yaw is shared by all five classes. Car keeps the reviewed legacy geometry and
Car box fitting path. Truck receives duplicate-track cleanup, which may absorb
a duplicate Car track. Nonmotorized_vehicle receives track-size, center, and
trajectory-yaw refinement; Bus and Pedestrian remain yaw-only pass-throughs.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry.box_geometry import GeometryConfig, apply_geometry_legacy
from geometry.car_box_fit import CarBoxFitConfig, apply_car_box_fit
from geometry.multiclass_refinement import (
    NonmotorizedSizeConfig,
    TruckOverlapConfig,
    merge_overlapping_truck_tracks,
    unify_nonmotorized_track_sizes,
    verify_multiclass_refinement,
)
from geometry.static_yaw import stabilize_static_yaw
from geometry.yaw_integrated import apply_yaw_integrated
from tracking import tracker_conservative as tracking
from tracking.tracker_static_first import StaticSlot


def slots_from_tracking_diagnostics(
        tracking_diagnostics: Mapping[str, Any]) -> List[StaticSlot]:
    """Rehydrate the small static-slot contract needed by the yaw stage."""
    slots: List[StaticSlot] = []
    for detail in tracking_diagnostics.get("slot_details", []):
        track_id = detail.get("track_id")
        if track_id is None:
            continue
        center = np.asarray(detail.get("center", [0.0, 0.0]), dtype=np.float64)
        size = np.asarray(detail.get("size", [4.0, 2.0, 1.5]), dtype=np.float64)
        if center.shape != (2,) or size.shape != (3,):
            continue
        slots.append(StaticSlot(
            slot_id=int(detail.get("slot_id", track_id)),
            track_id=int(track_id),
            center=center,
            yaw=float(detail.get("yaw", 0.0)),
            size=size,
            class_name=str(detail.get("class_name", "")),
            evidence_hits=int(detail.get("evidence_hits", 0)),
            row_id=(None if detail.get("row_id") is None
                    else int(detail["row_id"])),
        ))
    return slots


def _count(frames: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(frame.get("detections", [])) for frame in frames)


def _verify_non_car_geometry(before: Sequence[Mapping[str, Any]],
                             after: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return verify_multiclass_refinement(before, after)


def run(
        step2_5_json: Path,
        step2_5_diagnostics: Path,
        clip: Path,
        out_json: Path,
        out_clip: Optional[Path] = None,
        diagnostics_path: Optional[Path] = None,
        *, geometry_config: GeometryConfig = GeometryConfig(),
        car_config: CarBoxFitConfig = CarBoxFitConfig(),
        truck_config: TruckOverlapConfig = TruckOverlapConfig(),
        nonmotorized_config: NonmotorizedSizeConfig = NonmotorizedSizeConfig(),
) -> Dict[str, Any]:
    source = json.loads(Path(step2_5_json).read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError(f"input must be a list of frames: {step2_5_json}")
    previous = json.loads(Path(step2_5_diagnostics).read_text(encoding="utf-8"))
    coords = tracking.CoordinateProvider(Path(clip))
    tracking_diagnostics = previous.get("tracking", {})
    slots = slots_from_tracking_diagnostics(tracking_diagnostics)

    frames: List[Dict[str, Any]] = copy.deepcopy(source)
    before_yaw = copy.deepcopy(frames)
    static_yaw_diagnostics = stabilize_static_yaw(
        frames, coords, slots,
        tracking_diagnostics.get("slot_motion_coordination", {}))
    yawed, yaw_diagnostics = apply_yaw_integrated(
        frames, before_yaw, coords, Path(clip), tracking_diagnostics,
        static_yaw_diagnostics)

    # Keep the reviewed Car geometry path unchanged. It runs after the public
    # yaw stage and before the new Truck/NMV routes; the latter only add their
    # class-specific behavior and do not replace this Car fitting.
    car_generic, generic_diagnostics = apply_geometry_legacy(
        yawed, coords, Path(clip), tracking_diagnostics,
        static_yaw_diagnostics, geometry_config, classes=("Car",))
    car_output, car_diagnostics = apply_car_box_fit(
        car_generic, coords, Path(clip), tracking_diagnostics,
        static_yaw_diagnostics, car_config)

    # Keeping the original IDs through yaw and Car fitting preserves
    # static-slot diagnostics. Truck IDs are then merged and NMV
    # centers/sizes/yaw are refined for annotation.
    before_multiclass = copy.deepcopy(car_output)
    yawed = car_output
    truck_diagnostics = merge_overlapping_truck_tracks(
        yawed, coords, truck_config)
    nonmotorized_diagnostics = unify_nonmotorized_track_sizes(
        yawed, coords, nonmotorized_config)
    output = yawed
    non_car_check = _verify_non_car_geometry(before_multiclass, output)

    output_track_ids = {
        int(det["track_id"])
        for frame in output
        for det in frame.get("detections", [])
        if det.get("track_id") is not None
    }
    class_counts = Counter(
        str(det.get("class_name", ""))
        for frame in output for det in frame.get("detections", []))
    diagnostics: Dict[str, Any] = {
        "pipeline": "step3_refinement",
        "clip": str(Path(clip).resolve()),
        "source_step2_5_json": str(Path(step2_5_json).resolve()),
        "source_step2_5_diagnostics": str(Path(step2_5_diagnostics).resolve()),
        "input_frames": len(source),
        "input_detections": _count(source),
        "stage_order": [
            "public_yaw_static_and_dynamic",
            "class_route_car_geometry",
            "class_route_truck_overlap_id_merge",
            "class_route_nonmotorized_size_center_yaw",
            "class_route_bus_passthrough",
            "class_route_pedestrian_passthrough",
        ],
        "tracking": tracking_diagnostics,
        "static_yaw_stabilization": static_yaw_diagnostics,
        "yaw_integrated": yaw_diagnostics,
        "class_routes": {
            "Car": "apply_geometry_legacy_then_apply_car_box_fit",
            "Truck": "merge_truck_overlaps_near_tracks_and_truck_car_duplicates",
            "Bus": "passthrough",
            "Pedestrian": "passthrough",
            "Nonmotorized_vehicle": "robust_track_size_then_large_box_center_and_yaw_refinement",
        },
        "truck_overlap_merge": truck_diagnostics,
        "nonmotorized_size_refinement": nonmotorized_diagnostics,
        "geometry": generic_diagnostics,
        "car_refinement": car_diagnostics,
        "multiclass_geometry": {
            "truck": truck_diagnostics,
            "nonmotorized_vehicle": nonmotorized_diagnostics,
        },
        "car_tracks": car_diagnostics.get("car_tracks", 0),
        "car_boxes": car_diagnostics.get("car_boxes", 0),
        "tracks": len(output_track_ids),
        "boxes": _count(output),
        "class_counts": dict(sorted(class_counts.items())),
        "non_car_geometry_check": non_car_check,
        "final_detections": _count(output),
    }

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    if out_clip is not None:
        diagnostics["sust_labels"] = tracking.export_clip(
            output, Path(clip), Path(out_clip))
    target = Path(diagnostics_path or out_json.with_name(
        out_json.stem + "_diagnostics.json"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step2-5-json", "--step2_5-json", dest="step2_5_json",
                        type=Path, required=True)
    parser.add_argument("--step2-5-diagnostics", "--step2_5-diagnostics",
                        dest="step2_5_diagnostics", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-clip", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    diagnostics = run(
        args.step2_5_json, args.step2_5_diagnostics, args.clip,
        args.out_json, args.out_clip, args.diagnostics)
    print(json.dumps({
        "yaw_boxes_by_mode": diagnostics["yaw_integrated"].get("boxes_by_mode", {}),
        "car_refinement": diagnostics["car_refinement"],
        "truck_boxes_removed": diagnostics["truck_overlap_merge"].get(
            "boxes_removed", 0),
        "truck_class_converted_boxes": diagnostics["truck_overlap_merge"].get(
            "class_converted_boxes", 0),
        "nonmotorized_size_boxes": diagnostics[
            "nonmotorized_size_refinement"].get("boxes_changed", 0),
        "nonmotorized_center_boxes": diagnostics[
            "nonmotorized_size_refinement"].get("centers_changed", 0),
        "nonmotorized_yaw_boxes": diagnostics[
            "nonmotorized_size_refinement"].get("yaw_boxes_updated", 0),
        "non_car_geometry_check": diagnostics["non_car_geometry_check"],
        "final_detections": diagnostics["final_detections"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
