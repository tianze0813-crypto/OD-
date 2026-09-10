#!/usr/bin/env python3
"""Step 4.5: dynamic-region re-tracking, ID inheritance and phase stitching.

Input is the step-4 Car-only JSON.  Static detections outside the high-speed
dynamic region are frozen; only dynamic-region candidates are re-tracked with
motion-only association.  The output keeps the frozen detections unchanged and
writes final IDs / boxes for the dynamic part.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from region.dynamic_region import DynamicRegionConfig
from region.retrack import (
    Step45Config,
    align_dynamic_yaw,
    build_region,
    collect_world_tracks,
    direction_filter,
    dynamic_box_fit,
    inherit_ids,
    mark_long_gap_isolated_frames,
    phase_stitch,
    queue_stitch,
    region_mask,
    retrack_dynamic,
    revert_dynamic_yaw,
    seed_track_ids,
    select_retrackable,
    single_frame_overlap_filter,
    verify_static_freeze,
    verify_unique_frame_ids,
)
from tracking import tracker_conservative as tracking


def run(step4_json: Path, clip: Path, step2_diagnostics: Path,
        out_json: Path, diagnostics_path: Path,
        config: Step45Config = Step45Config()) -> dict:
    frames = json.loads(Path(step4_json).read_text(encoding="utf-8"))
    if not isinstance(frames, list):
        raise ValueError(f"step4 input must be a list of frames: {step4_json}")
    step2 = json.loads(Path(step2_diagnostics).read_text(encoding="utf-8"))
    coords = tracking.CoordinateProvider(Path(clip))

    # Pass-1 single-frame overlap noise filter.
    tracks_pass1, _by_key = collect_world_tracks(frames, coords)
    if config.overlap_filter_enabled:
        frames, overlap_details, overlap_filter_diag = (
            single_frame_overlap_filter(
                frames, tracks_pass1, Path(clip), config))
    else:
        overlap_details = []
        overlap_filter_diag = {
            "enabled": False,
            "noise_detections_removed": 0,
            "removed_details": [],
        }

    # Optional pass-1 driving-direction noise filter (disabled by default:
    # it can lock yaw on some vehicles).
    direction_filter_details = []
    if config.direction_filter_enabled:
        tracks_pass1, _by_key = collect_world_tracks(frames, coords)
        seeds_pass1 = seed_track_ids(
            tracks_pass1, config.region, config)
        frames, direction_filter_details, direction_filter_diag = (
            direction_filter(
                frames, tracks_pass1, set(seeds_pass1), config))
    else:
        direction_filter_diag = {
            "enabled": False,
            "noise_detections_removed": 0,
            "noise_details": [],
        }

    # Rebuild pass-1 structures on the cleaned detections.
    tracks, _by_key = collect_world_tracks(frames, coords)
    static_slots = list(
        step2.get("tracking", {}).get("slot_details", []))
    seeds = seed_track_ids(tracks, config.region, config)
    region = build_region(
        tracks, static_slots, config.region,
        accepted_track_ids=set(seeds))
    mask = region_mask(region, config.region)
    retrackable, selection = select_retrackable(
        frames, coords, mask, seeds)
    for frame_index, frame in enumerate(frames):
        for detection_index, det in enumerate(frame.get("detections", [])):
            det["region"] = (
                "dynamic" if (frame_index, detection_index) in retrackable
                else "static")
    before_step45 = copy.deepcopy(frames)

    # Pass 2: occlusion-aware motion-only re-tracking.
    retrack_diagnostics = retrack_dynamic(
        frames, coords, retrackable, config)
    inheritance = inherit_ids(
        frames, tracks, retrackable, seeds, step2, config)
    isolated_gap = mark_long_gap_isolated_frames(frames, tracks, config)
    queue = queue_stitch(
        frames, tracks, step2, mask, set(seeds), config)
    exempt_keys = set(retrackable) | set(queue.get("modified_keys", []))
    phase = phase_stitch(frames, tracks, coords, config)
    unique_ids = verify_unique_frame_ids(frames)

    tracking_diagnostics = step2.get("tracking", {})
    static_yaw_diagnostics = step2.get("static_yaw_stabilization", {})
    _fitted_frames, box_fit_diagnostics = dynamic_box_fit(
        frames, Path(clip), coords, tracking_diagnostics,
        static_yaw_diagnostics)

    # Final dynamic yaw: remove the 180-degree flip ambiguity and align the
    # box axis to the local driving heading.
    if config.yaw_align_enabled:
        dynamic_final_ids = {
            int(det["track_id"]) for frame in frames
            for det in frame.get("detections", [])
            if det.get("track_id") is not None
            and det.get("_step45_retracked")}
        yaw_diagnostics = align_dynamic_yaw(
            frames, tracks, coords, dynamic_final_ids, config)
    else:
        yaw_diagnostics = {
            "enabled": False,
            "dynamic_yaw_aligned": 0,
            "tracks": 0,
            "details": [],
        }
    if config.yaw_reversal_enabled:
        yaw_reversal_diag = revert_dynamic_yaw(frames, tracks, config)
    else:
        yaw_reversal_diag = {
            "enabled": False,
            "reversed_detections": 0,
            "reversed_tracks": 0,
            "details": [],
        }
    static_freeze = verify_static_freeze(
        before_step45, frames, exempt_keys)

    for frame in frames:
        frame["num_detections"] = len(frame.get("detections", []))

    diagnostics = {
        "pipeline": "step4_5_region_phase_retrack",
        "source_step4_json": str(Path(step4_json).resolve()),
        "source_clip": str(Path(clip).resolve()),
        "source_step2_diagnostics": str(
            Path(step2_diagnostics).resolve()),
        "config": {
            "region": config.region.to_dict(),
            "traffic_light": config.traffic_light.to_dict(),
            "dynamic_max_gap_sec": config.dynamic_max_gap_sec,
            "phase_merge_max_gap_sec": config.phase_merge_max_gap_sec,
            "yielding_max_gap_sec": config.yielding_max_gap_sec,
        },
        "dynamic_region": region.to_dict(),
        "dynamic_region_mask": mask.to_dict(),
        "candidate_tracks": len(seeds),
        "candidate_track_ids": sorted(seeds),
        "direction_filter": direction_filter_diag,
        "direction_filter_details": direction_filter_details,
        "overlap_filter": overlap_filter_diag,
        "overlap_filter_details": overlap_details,
        "selection": selection,
        "retracking": retrack_diagnostics,
        "id_inheritance": inheritance,
        "isolated_gap_frames": isolated_gap,
        "queue_stitching": {
            "queues": queue.get("queues", 0),
            "edges": queue.get("edges", 0),
            "merges": queue.get("merges", []),
            "modified_detections": len(queue.get("modified_keys", [])),
            "direction_assignments": queue.get(
                "direction_assignments", {}),
            "isolated_ids": queue.get("isolated_ids", []),
            "edges_detail": queue.get("edges_detail", []),
        },
        "phase_stitching": phase,
        "unique_frame_ids": unique_ids,
        "box_fit": box_fit_diagnostics,
        "dynamic_yaw_alignment": yaw_diagnostics,
        "yaw_reversal": yaw_reversal_diag,
        "static_freeze": static_freeze,
        "final_detections": sum(
            len(frame.get("detections", [])) for frame in frames),
    }
    if not static_freeze["passed"]:
        raise AssertionError(
            f"step4.5 static freeze violated: "
            f"{static_freeze['mismatches'][:3]}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(frames, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step4-json", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--step2-diagnostics", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--extension-length-m", type=float, default=30.0)
    parser.add_argument("--phase-merge-max-gap-sec", type=float, default=30.0)
    parser.add_argument("--yielding-max-gap-sec", type=float, default=6.0)
    args = parser.parse_args()
    diagnostics_path = args.diagnostics or args.out_json.with_name(
        args.out_json.stem + "_diagnostics.json")
    region_config = DynamicRegionConfig(
        extension_length_m=args.extension_length_m)
    config = Step45Config(
        region=region_config,
        phase_merge_max_gap_sec=args.phase_merge_max_gap_sec,
        yielding_max_gap_sec=args.yielding_max_gap_sec,
    )
    diagnostics = run(
        args.step4_json, args.clip, args.step2_diagnostics,
        args.out_json, diagnostics_path, config)
    print(json.dumps({
        "candidate_tracks": diagnostics["candidate_tracks"],
        "retrackable_detections": diagnostics["selection"][
            "retrackable_detections"],
        "direction_noise_removed": diagnostics[
            "direction_filter"]["noise_detections_removed"],
        "overlap_noise_removed": diagnostics[
            "overlap_filter"]["noise_detections_removed"],
        "yaw_reversed_detections": diagnostics[
            "yaw_reversal"]["reversed_detections"],
        "id_assignments": len(diagnostics["id_inheritance"]["assignments"]),
        "queue_merges": len(diagnostics["queue_stitching"]["merges"]),
        "isolated_gap_frames": diagnostics[
            "isolated_gap_frames"]["marked_detections"],
        "phase_merges": len(diagnostics["phase_stitching"]["applied"]),
        "occlusion_recoveries": diagnostics["retracking"].get(
            "occlusion_recoveries", 0),
        "dynamic_yaw_aligned": diagnostics[
            "dynamic_yaw_alignment"]["dynamic_yaw_aligned"],
        "final_detections": diagnostics["final_detections"],
        "static_freeze_passed": diagnostics["static_freeze"]["passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
