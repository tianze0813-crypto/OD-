"""Integrated yaw stage built on the reviewed vehicle policy."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from geometry.yaw_static_direction import _world_yaw_to_local
from geometry.yaw_vehicle_dynamic import apply_yaw_vehicle_dynamic
from tracking import tracker_conservative as tracking


@dataclass(frozen=True)
class PedestrianYawConfig:
    max_observation_gap: float = 1.2
    min_displacement: float = 0.10


def _pedestrian_targets(
        frames: Sequence[Dict[str, Any]],
        coords: tracking.CoordinateProvider,
        config: PedestrianYawConfig,
) -> Tuple[Dict[Tuple[int, int], float], List[Dict[str, Any]]]:
    tracks: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for frame_index, frame in enumerate(frames):
        timestamp = int(frame["frame_id"])
        world_from_lidar = coords.world_from_lidar(timestamp)
        if world_from_lidar is None:
            continue
        for detection_index, det in enumerate(frame.get("detections", [])):
            if (det.get("track_id") is None
                    or det.get("class_name") != "Pedestrian"
                    or not tracking.finite_box(det)):
                continue
            tracks[int(det["track_id"])].append({
                "frame_index": frame_index,
                "detection_index": detection_index,
                "timestamp": timestamp,
                "world": tracking.center_world(det["box_lidar"], world_from_lidar),
                "det": det,
            })

    targets: Dict[Tuple[int, int], float] = {}
    details = []
    for track_id, items in tracks.items():
        items.sort(key=lambda item: item["timestamp"])
        if len(items) < 2:
            continue
        accepted = 0
        rejected_gap = 0
        rejected_displacement = 0
        for index, item in enumerate(items):
            left, right = ((items[0], items[1]) if index == 0
                           else (items[index - 1], item))
            dt = (right["timestamp"] - left["timestamp"]) / 1e9
            if dt <= 1e-3 or dt > config.max_observation_gap:
                rejected_gap += 1
                continue
            vector = right["world"][:2] - left["world"][:2]
            distance = float(np.linalg.norm(vector))
            if distance < config.min_displacement:
                rejected_displacement += 1
                continue
            heading = math.atan2(float(vector[1]), float(vector[0]))
            box = item["det"]["box_lidar"]
            if float(box[3]) < float(box[4]):
                heading -= math.pi / 2.0
            targets[(item["frame_index"], item["detection_index"])] = heading
            accepted += 1
        details.append({
            "track_id": track_id,
            "observations": len(items),
            "yaw_mode": "pedestrian_two_frame_heading",
            "boxes_updated": accepted,
            "rejected_gap": rejected_gap,
            "rejected_displacement": rejected_displacement,
        })
    return targets, details


def apply_yaw_integrated(
        final_frames: Sequence[Dict[str, Any]],
        pre_yaw_frames: Sequence[Dict[str, Any]],
        coords: tracking.CoordinateProvider,
        clip: Path,
        tracking_diagnostics: Mapping[str, Any],
        static_yaw_diagnostics: Mapping[str, Any],
        pedestrian_config: PedestrianYawConfig = PedestrianYawConfig(),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output, diagnostics = apply_yaw_vehicle_dynamic(
        final_frames, pre_yaw_frames, coords, clip,
        tracking_diagnostics, static_yaw_diagnostics)
    pedestrian_targets, pedestrian_details = _pedestrian_targets(
        output, coords, pedestrian_config)

    pedestrian_boxes = 0
    for frame_index, frame in enumerate(output):
        world_from_lidar = coords.world_from_lidar(int(frame["frame_id"]))
        if world_from_lidar is None:
            continue
        for detection_index, det in enumerate(frame.get("detections", [])):
            target = pedestrian_targets.get((frame_index, detection_index))
            if target is None:
                continue
            det["box_lidar"][6] = _world_yaw_to_local(
                target, world_from_lidar)
            pedestrian_boxes += 1

    _verify_yaw_only(final_frames, output)
    diagnostics["policy"]["pipeline_position"] = (
        "after_identity_class_filters_and_short_tracks")
    diagnostics["policy"]["priority"] = [
        "static_direction_vote",
        "confirmed_motion_heading",
        "stationary_multiframe_pointcloud_axis",
        "pedestrian_two_frame_heading",
        "keep_original",
    ]
    diagnostics["boxes_by_mode"][
        "pedestrian_two_frame_heading"] = pedestrian_boxes
    diagnostics["pedestrian"] = {
        "tracks": len(pedestrian_details),
        "boxes": pedestrian_boxes,
        "config": {
            "max_observation_gap": pedestrian_config.max_observation_gap,
            "min_displacement": pedestrian_config.min_displacement,
        },
        "details": pedestrian_details,
    }
    return output, diagnostics


def _verify_yaw_only(before: Sequence[Dict[str, Any]],
                     after: Sequence[Dict[str, Any]]) -> None:
    if len(before) != len(after):
        raise AssertionError("integrated yaw changed frame count")
    for left_frame, right_frame in zip(before, after):
        if left_frame.get("frame_id") != right_frame.get("frame_id"):
            raise AssertionError("integrated yaw changed frame order")
        left = left_frame.get("detections", [])
        right = right_frame.get("detections", [])
        if len(left) != len(right):
            raise AssertionError("integrated yaw changed detection count")
        for left_det, right_det in zip(left, right):
            comparable = copy.deepcopy(right_det)
            comparable["box_lidar"][6] = left_det["box_lidar"][6]
            if comparable != left_det:
                raise AssertionError("integrated yaw changed a non-yaw field")
