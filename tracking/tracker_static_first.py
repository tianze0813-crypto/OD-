#!/usr/bin/env python3
"""Static-first, clip-local BEV identity association.

Tracking is deliberately split into two independent passes. Persistent
Vehicle/Car/Truck positions are discovered from the full clip in world
coordinates before any ID exists. Observations assigned to those positions
receive one immutable slot identity. Every remaining observation is handled
by the ordinary motion tracker from :mod:`tracker_conservative`.

The identity stage only adds ``track_id``. It never changes a raw box, class,
score, or frame count, and never creates a box for a missed detection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from tracking import tracker_conservative as tracking


STATIC_CLASSES = tracking.STATIC_CLASSES


def circular_median_pi(values: Sequence[float]) -> float:
    """Robust representative of a box heading, whose period is pi."""
    if not values:
        return 0.0
    doubled = 2.0 * np.asarray(values, dtype=np.float64)
    center = 0.5 * math.atan2(float(np.median(np.sin(doubled))),
                              float(np.median(np.cos(doubled))))
    candidates = [center, center + math.pi / 2.0]
    return min(candidates, key=lambda x: sum(
        tracking.angle_distance(v, x, modulo_pi=True) for v in values))


def size_compatible(a: np.ndarray, b: np.ndarray, max_delta: float = 0.65) -> bool:
    return (float(np.linalg.norm(a - b))
            / max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0)
            <= float(max_delta))


@dataclass
class StaticSlot:
    slot_id: int
    center: np.ndarray
    yaw: float
    size: np.ndarray
    class_name: str
    evidence_hits: int
    evidence_frames: set[int] = field(default_factory=set)
    spread90: float = 0.0
    row_id: Optional[int] = None
    row_axis: Optional[np.ndarray] = None
    row_coord: float = 0.0
    cross_coord: float = 0.0
    track_id: Optional[int] = None
    matched: List[tracking.Observation] = field(default_factory=list)
    first_matched_frame: Optional[int] = None
    last_matched_frame: Optional[int] = None
    alias_centers: List[np.ndarray] = field(default_factory=list)


@dataclass
class SlotCandidate:
    center: np.ndarray
    yaw: float
    size: np.ndarray
    class_name: str
    member_ids: set[int]
    evidence_frames: set[int]
    hits: int
    spread90: float
    score: float


class StaticFirstTracker:
    """Discover persistent parking positions, then run motion association."""

    def __init__(
        self,
        coords: tracking.CoordinateProvider,
        *,
        slot_evidence_radius: float = 0.8,
        slot_min_hits: int = 8,
        slot_min_duration: float = 1.2,
        slot_nms_distance: float = 1.55,
        slot_match_radius: float = 1.05,
        slot_ambiguous_radius: float = 0.72,
        slot_ambiguous_margin: float = 0.20,
        slot_yaw_gate: float = 0.55,
        row_neighbor_min: float = 1.55,
        row_neighbor_max: float = 3.6,
        row_parallel_gate: float = 0.50,
        row_cross_gate: float = 0.60,
        standalone_min_hits: int = 20,
        dynamic_max_gap: float = 1.8,
        topology_along_gate: float = 0.95,
        topology_cross_gate: float = 2.0,
        topology_min_stable_hits: int = 5,
        topology_occupancy_window: int = 8,
        duplicate_slot_along_gate: float = 0.45,
        duplicate_slot_cross_gate: float = 2.2,
        duplicate_slot_iou_gate: float = 0.35,
        duplicate_slot_max_weak_fraction: float = 0.5,
        topology_gap_max_frames: int = 3,
    ):
        self.coords = coords
        self.slot_evidence_radius = float(slot_evidence_radius)
        self.slot_min_hits = int(slot_min_hits)
        self.slot_min_duration = float(slot_min_duration)
        self.slot_nms_distance = float(slot_nms_distance)
        self.slot_match_radius = float(slot_match_radius)
        self.slot_ambiguous_radius = float(slot_ambiguous_radius)
        self.slot_ambiguous_margin = float(slot_ambiguous_margin)
        self.slot_yaw_gate = float(slot_yaw_gate)
        self.row_neighbor_min = float(row_neighbor_min)
        self.row_neighbor_max = float(row_neighbor_max)
        self.row_parallel_gate = float(row_parallel_gate)
        self.row_cross_gate = float(row_cross_gate)
        self.standalone_min_hits = int(standalone_min_hits)
        self.dynamic_max_gap = float(dynamic_max_gap)
        self.topology_along_gate = float(topology_along_gate)
        self.topology_cross_gate = float(topology_cross_gate)
        self.topology_min_stable_hits = int(topology_min_stable_hits)
        self.topology_occupancy_window = int(topology_occupancy_window)
        self.duplicate_slot_along_gate = float(duplicate_slot_along_gate)
        self.duplicate_slot_cross_gate = float(duplicate_slot_cross_gate)
        self.duplicate_slot_iou_gate = float(duplicate_slot_iou_gate)
        self.duplicate_slot_max_weak_fraction = float(
            duplicate_slot_max_weak_fraction)
        self.topology_gap_max_frames = int(topology_gap_max_frames)
        self.slots: List[StaticSlot] = []
        self.motion_tracker: Optional[tracking.ConservativeTracker] = None
        self.diagnostics: Dict[str, Any] = {
            "coordinate_transform": coords.description,
            "mode": "offline_static_first_then_dynamic",
            "geometry_policy": "track_id_only_no_box_mutation_no_box_generation",
            "frames": 0,
            "detections": 0,
            "static_observations": 0,
            "static_slots": 0,
            "static_matches": 0,
            "static_ambiguous_unassigned": 0,
            "static_outside_gate": 0,
            "rows": [],
        }

    def _world_observations(self, frames: Sequence[Dict[str, Any]]) -> List[List[tracking.Observation]]:
        by_frame: List[List[tracking.Observation]] = [[] for _ in frames]
        for frame_index, frame in enumerate(frames):
            ts = int(frame["frame_id"])
            wf = self.coords.world_from_lidar(ts)
            for det in frame.get("detections", []):
                if wf is None or not tracking.finite_box(det):
                    continue
                box = det["box_lidar"]
                by_frame[frame_index].append(tracking.Observation(
                    frame_index=frame_index,
                    timestamp=ts,
                    detection=det,
                    world=tracking.center_world(box, wf),
                    yaw=tracking.yaw_world(float(box[6]), wf),
                    size=np.asarray(box[3:6], dtype=np.float64),
                ))
        self.diagnostics["frames"] = len(frames)
        self.diagnostics["detections"] = sum(len(x) for x in by_frame)
        self.diagnostics["static_observations"] = sum(
            1 for items in by_frame for o in items
            if o.detection.get("class_name") in STATIC_CLASSES)
        return by_frame

    @staticmethod
    def _track_is_clear_motion(observations: Sequence[tracking.Observation]) -> Tuple[bool, Dict[str, Any]]:
        """Require several coherent steps, not one adjacent-slot ID jump."""
        items = sorted(observations, key=lambda o: o.timestamp)
        if len(items) < 5:
            return False, {"reason": "too_short", "hits": len(items)}
        xy = np.asarray([o.world[:2] for o in items], dtype=np.float64)
        centered = xy - np.median(xy, axis=0)
        covariance = centered.T @ centered
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        if float(np.dot(xy[-1] - xy[0], axis)) < 0.0:
            axis = -axis
        progress = xy @ axis
        steps = np.diff(progress)
        times = np.asarray([o.timestamp for o in items], dtype=np.float64) / 1e9
        dt = np.diff(times)
        valid = (dt > 1e-3) & (dt <= 1.9)
        significant = steps[valid] > 0.35
        backward = -np.minimum(steps[valid], 0.0)
        forward = np.maximum(steps[valid], 0.0)
        span = float(np.ptp(progress))
        net = float(progress[-1] - progress[0])
        duration = max(float(times[-1] - times[0]), 1e-3)
        clear = (
            span >= 3.0
            and net >= 2.5
            and int(np.count_nonzero(significant)) >= 3
            and float(np.sum(backward)) <= max(0.8, 0.25 * float(np.sum(forward)))
            and net / duration >= 0.35
        )
        return clear, {
            "hits": len(items), "span": round(span, 3), "net": round(net, 3),
            "duration": round(duration, 3),
            "significant_forward_steps": int(np.count_nonzero(significant)),
            "backward_distance": round(float(np.sum(backward)), 3),
        }

    @staticmethod
    def _motion_onset_index(observations: Sequence[tracking.Observation]) -> int:
        """First observation of a sustained motion run along the track axis."""
        items = sorted(observations, key=lambda o: o.timestamp)
        if len(items) < 4:
            return 0
        xy = np.asarray([o.world[:2] for o in items], dtype=np.float64)
        centered = xy - np.median(xy, axis=0)
        values, vectors = np.linalg.eigh(centered.T @ centered)
        axis = vectors[:, int(np.argmax(values))]
        if float(np.dot(xy[-1] - xy[0], axis)) < 0.0:
            axis = -axis
        progress = xy @ axis
        timestamps = np.asarray([o.timestamp for o in items], dtype=np.float64) / 1e9
        steps = np.diff(progress)
        dt = np.diff(timestamps)
        moving = (dt > 1e-3) & (dt <= 1.9) & (steps > 0.30)
        for start in range(len(moving)):
            window = moving[start:min(len(moving), start + 4)]
            if int(np.count_nonzero(window)) >= 3:
                return start
        return 0

    def _motion_probe(self, frames: Sequence[Dict[str, Any]]) -> Tuple[Dict[Tuple[int, int], int], Dict[str, Any]]:
        """Find only unambiguous continuous trajectories before slot assignment.

        The probe sees every raw box but its IDs are accepted only for tracks
        with several coherent motion steps. Parked jitter and a single
        neighboring-slot jump cannot claim dynamic priority.
        """
        probe_frames = copy.deepcopy(list(frames))
        probe = tracking.ConservativeTracker(
            self.coords, min_static_hits=10 ** 9,
            dynamic_max_gap=self.dynamic_max_gap)
        probe_output, probe_diag = probe.process(
            probe_frames, enable_stitching=False)
        grouped: Dict[int, List[tracking.Observation]] = {}
        keyed: Dict[int, List[Tuple[int, int]]] = {}
        for frame_index, frame in enumerate(probe_output):
            wf = self.coords.world_from_lidar(int(frame["frame_id"]))
            if wf is None:
                continue
            for detection_index, det in enumerate(frame.get("detections", [])):
                tid = int(det["track_id"])
                box = det["box_lidar"]
                grouped.setdefault(tid, []).append(tracking.Observation(
                    frame_index=frame_index,
                    timestamp=int(frame["frame_id"]),
                    detection=det,
                    world=tracking.center_world(box, wf),
                    yaw=tracking.yaw_world(float(box[6]), wf),
                    size=np.asarray(box[3:6], dtype=np.float64),
                ))
                keyed.setdefault(tid, []).append((frame_index, detection_index))
        confirmed = {}
        onset_by_track = {}
        details = []
        for tid, observations in grouped.items():
            accepted, metrics = self._track_is_clear_motion(observations)
            if accepted:
                confirmed[tid] = observations
                onset_by_track[tid] = self._motion_onset_index(observations)
                details.append({"probe_track_id": tid,
                                "motion_onset_index": onset_by_track[tid],
                                **metrics})
        protected = {
            key: tid for tid, keys in keyed.items() if tid in confirmed
            for key in keys[onset_by_track[tid]:]
        }
        return protected, {
            "probe_tracks": len(grouped),
            "confirmed_motion_tracks": len(confirmed),
            "protected_detections": len(protected),
            "confirmed_details": details,
            "online_tracker": probe_diag,
        }

    def _discover_candidates(self, frames: Sequence[Dict[str, Any]],
                             by_frame: Sequence[Sequence[tracking.Observation]]) -> List[SlotCandidate]:
        observations = [
            o for items in by_frame for o in items
            if o.detection.get("class_name") in STATIC_CLASSES
        ]
        if not observations:
            return []
        xy = np.asarray([o.world[:2] for o in observations], dtype=np.float64)
        tree = cKDTree(xy)
        candidates: List[SlotCandidate] = []
        for seed_index, seed in enumerate(observations):
            neighbor_ids = tree.query_ball_point(xy[seed_index], self.slot_evidence_radius)
            by_seen_frame: Dict[int, int] = {}
            for member_index in neighbor_ids:
                obs = observations[member_index]
                if not tracking.class_compatible(
                        obs.detection.get("class_name", ""),
                        seed.detection.get("class_name", "")):
                    continue
                if not size_compatible(obs.size, seed.size):
                    continue
                if tracking.angle_distance(obs.yaw, seed.yaw, modulo_pi=True) > self.slot_yaw_gate:
                    continue
                old_index = by_seen_frame.get(obs.frame_index)
                if old_index is None or float(obs.detection.get("score", 0.0)) > float(
                        observations[old_index].detection.get("score", 0.0)):
                    by_seen_frame[obs.frame_index] = member_index
            if len(by_seen_frame) < self.slot_min_hits:
                continue
            member_ids = list(by_seen_frame.values())
            center = np.median(xy[member_ids], axis=0)

            refined_ids = tree.query_ball_point(center, self.slot_evidence_radius)
            by_seen_frame = {}
            for member_index in refined_ids:
                obs = observations[member_index]
                if not tracking.class_compatible(
                        obs.detection.get("class_name", ""),
                        seed.detection.get("class_name", "")):
                    continue
                if not size_compatible(obs.size, seed.size):
                    continue
                if tracking.angle_distance(obs.yaw, seed.yaw, modulo_pi=True) > self.slot_yaw_gate:
                    continue
                old_index = by_seen_frame.get(obs.frame_index)
                if old_index is None or float(obs.detection.get("score", 0.0)) > float(
                        observations[old_index].detection.get("score", 0.0)):
                    by_seen_frame[obs.frame_index] = member_index
            member_ids = list(by_seen_frame.values())
            if len(member_ids) < self.slot_min_hits:
                continue
            timestamps = [observations[i].timestamp for i in member_ids]
            if (max(timestamps) - min(timestamps)) / 1e9 < self.slot_min_duration:
                continue
            center = np.median(xy[member_ids], axis=0)
            distances = np.linalg.norm(xy[member_ids] - center, axis=1)
            spread90 = float(np.percentile(distances, 90))
            if spread90 > self.slot_evidence_radius:
                continue
            member_obs = [observations[i] for i in member_ids]
            candidates.append(SlotCandidate(
                center=center,
                yaw=circular_median_pi([o.yaw for o in member_obs]),
                size=np.median(np.asarray([o.size for o in member_obs]), axis=0),
                class_name=max(
                    (o.detection.get("class_name", "") for o in member_obs),
                    key=lambda name: sum(o.detection.get("class_name", "") == name
                                         for o in member_obs),
                ),
                member_ids=set(member_ids),
                evidence_frames={o.frame_index for o in member_obs},
                hits=len(member_ids),
                spread90=spread90,
                score=float(np.median([
                    float(o.detection.get("score", 0.0)) for o in member_obs
                ])),
            ))

        candidates.sort(key=lambda c: (-c.hits, c.spread90, -c.score))
        kept: List[SlotCandidate] = []
        for candidate in candidates:
            duplicate = False
            for old in kept:
                distance = float(np.linalg.norm(candidate.center - old.center))
                shared = len(candidate.member_ids & old.member_ids)
                shared_fraction = shared / max(
                    1, min(len(candidate.member_ids), len(old.member_ids)))
                if distance <= self.slot_nms_distance or shared_fraction >= 0.65:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        self.diagnostics["candidate_peaks"] = len(candidates)
        self.diagnostics["candidate_after_nms"] = len(kept)
        return kept

    @staticmethod
    def _union_find_components(size: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
        parent = list(range(size))

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for left, right in edges:
            a, b = root(left), root(right)
            if a != b:
                parent[b] = a
        groups: Dict[int, List[int]] = {}
        for index in range(size):
            groups.setdefault(root(index), []).append(index)
        return list(groups.values())

    def _build_slots(self, candidates: Sequence[SlotCandidate]) -> List[StaticSlot]:
        edges: List[Tuple[int, int]] = []
        for left in range(len(candidates)):
            a = candidates[left]
            for right in range(left):
                b = candidates[right]
                delta = a.center - b.center
                distance = float(np.linalg.norm(delta))
                if not self.row_neighbor_min < distance < self.row_neighbor_max:
                    continue
                if tracking.angle_distance(a.yaw, b.yaw, modulo_pi=True) > self.row_parallel_gate:
                    continue
                heading = np.array([math.cos(a.yaw), math.sin(a.yaw)], dtype=np.float64)
                if abs(float(np.dot(delta / distance, heading))) > self.row_cross_gate:
                    continue
                edges.append((left, right))
        components = self._union_find_components(len(candidates), edges)
        slots: List[StaticSlot] = []
        rows = []
        next_slot = 1
        next_row = 1
        for component in components:
            use_component = len(component) >= 2
            selected = [i for i in component if (
                use_component or candidates[i].hits >= self.standalone_min_hits
            )]
            if not selected:
                continue
            row_id: Optional[int] = None
            row_axis: Optional[np.ndarray] = None
            ordered = list(selected)
            if len(component) >= 2:
                row_id = next_row
                next_row += 1
                # Parking centers advance across a row, perpendicular to the
                # vehicles' longitudinal heading. The sign is normalized for
                # deterministic slot order.
                headings = np.asarray([
                    [math.cos(candidates[i].yaw), math.sin(candidates[i].yaw)]
                    for i in component
                ], dtype=np.float64)
                heading = np.median(headings, axis=0)
                heading /= max(float(np.linalg.norm(heading)), 1e-9)
                row_axis = np.array([-heading[1], heading[0]], dtype=np.float64)
                if row_axis[0] < 0.0 or (abs(row_axis[0]) < 1e-9 and row_axis[1] < 0.0):
                    row_axis = -row_axis
                ordered.sort(key=lambda i: float(np.dot(candidates[i].center, row_axis)))
            row_slot_ids = []
            for candidate_index in ordered:
                c = candidates[candidate_index]
                slot = StaticSlot(
                    slot_id=next_slot,
                    center=c.center.copy(),
                    yaw=float(c.yaw),
                    size=c.size.copy(),
                    class_name=c.class_name,
                    evidence_hits=c.hits,
                    evidence_frames=set(c.evidence_frames),
                    spread90=c.spread90,
                    row_id=row_id,
                    row_axis=None if row_axis is None else row_axis.copy(),
                    row_coord=(0.0 if row_axis is None
                               else float(np.dot(c.center, row_axis))),
                    cross_coord=(0.0 if row_axis is None
                                 else float(np.dot(c.center,
                                                   np.array([-row_axis[1], row_axis[0]])))),
                )
                slots.append(slot)
                row_slot_ids.append(next_slot)
                next_slot += 1
            if row_id is not None:
                rows.append({
                    "row_id": row_id,
                    "slot_ids": row_slot_ids,
                    "axis": [round(float(x), 6) for x in row_axis],
                })
        slots.sort(key=lambda s: s.slot_id)
        self.diagnostics["rows"] = rows
        self.diagnostics["static_slots"] = len(slots)
        self.diagnostics["standalone_slots"] = sum(s.row_id is None for s in slots)
        return slots

    def _slot_cost(self, obs: tracking.Observation, slot: StaticSlot) -> float:
        class_name = obs.detection.get("class_name", "")
        if class_name not in STATIC_CLASSES or not tracking.class_compatible(class_name, slot.class_name):
            return 1e9
        distance = float(np.linalg.norm(obs.world[:2] - slot.center))
        if distance > self.slot_match_radius:
            return 1e9
        if not size_compatible(obs.size, slot.size, max_delta=0.8):
            return 1e9
        yaw_delta = tracking.angle_distance(obs.yaw, slot.yaw, modulo_pi=True)
        if yaw_delta > self.slot_yaw_gate:
            return 1e9
        scale_delta = float(np.linalg.norm(obs.size - slot.size)) / max(
            float(np.linalg.norm(slot.size)), 1.0)
        iou = tracking.bev_iou(obs.world, obs.size, obs.yaw,
                         np.r_[slot.center, obs.world[2]], slot.size, slot.yaw)
        return (distance / max(self.slot_match_radius, 1e-6)
                + 0.30 * scale_delta
                + 0.15 * yaw_delta / max(self.slot_yaw_gate, 1e-6)
                + 0.10 * (1.0 - iou))

    def _assign_static(self, by_frame: Sequence[Sequence[tracking.Observation]]) -> set[int]:
        assigned_detection_ids: set[int] = set()
        for frame_index, observations in enumerate(by_frame):
            eligible = [o for o in observations
                        if (o.detection.get("class_name") in STATIC_CLASSES
                            and o.detection.get("track_id") is None)]
            if not eligible or not self.slots:
                continue
            costs = np.full((len(eligible), len(self.slots)), 1e9, dtype=np.float64)
            for oi, obs in enumerate(eligible):
                for si, slot in enumerate(self.slots):
                    costs[oi, si] = self._slot_cost(obs, slot)

            # A distant box close to two adjacent slots is exactly the failure
            # mode that caused parked IDs to cross in the legacy tracker. It does not get a
            # stable identity unless one position is clearly closer.
            ambiguous = set()
            for oi, obs in enumerate(eligible):
                feasible = [
                    (float(np.linalg.norm(obs.world[:2] - slot.center)), si)
                    for si, slot in enumerate(self.slots) if costs[oi, si] < 1e8
                ]
                feasible.sort()
                if (len(feasible) >= 2
                        and feasible[0][0] > self.slot_ambiguous_radius
                        and feasible[1][0] - feasible[0][0] < self.slot_ambiguous_margin):
                    ambiguous.add(oi)
                    costs[oi, :] = 1e9

            rows, cols = linear_sum_assignment(costs)
            frame_matches: List[Tuple[int, int]] = []
            for oi, si in zip(rows, cols):
                if costs[oi, si] < 1e8:
                    frame_matches.append((int(oi), int(si)))

            # Enforce monotonic order inside a parking row. If two observations
            # would cross neighboring slot identities, keep the lower-cost
            # association and send the weaker box to the dynamic/short-chain
            # path. Stable slot IDs are never exchanged to retain a detection.
            rejected = set()
            by_row: Dict[int, List[Tuple[int, int]]] = {}
            for oi, si in frame_matches:
                row_id = self.slots[si].row_id
                if row_id is not None:
                    by_row.setdefault(row_id, []).append((oi, si))
            for row_matches in by_row.values():
                row_matches.sort(key=lambda pair: self.slots[pair[1]].row_coord)
                changed = True
                while changed:
                    changed = False
                    for left, right in zip(row_matches, row_matches[1:]):
                        loi, lsi = left
                        roi, rsi = right
                        axis = self.slots[lsi].row_axis
                        assert axis is not None
                        left_coord = float(np.dot(eligible[loi].world[:2], axis))
                        right_coord = float(np.dot(eligible[roi].world[:2], axis))
                        if left_coord <= right_coord or loi in rejected or roi in rejected:
                            continue
                        drop = left if costs[loi, lsi] >= costs[roi, rsi] else right
                        rejected.add(drop[0])
                        changed = True

            for oi, si in frame_matches:
                if oi in rejected:
                    continue
                obs, slot = eligible[oi], self.slots[si]
                obs.detection["track_id"] = int(slot.track_id)
                slot.matched.append(obs)
                slot.first_matched_frame = (frame_index if slot.first_matched_frame is None
                                            else min(slot.first_matched_frame, frame_index))
                slot.last_matched_frame = (frame_index if slot.last_matched_frame is None
                                           else max(slot.last_matched_frame, frame_index))
                assigned_detection_ids.add(id(obs.detection))
            self.diagnostics["static_ambiguous_unassigned"] += len(ambiguous) + len(rejected)
        self.diagnostics["static_matches"] = len(assigned_detection_ids)
        return assigned_detection_ids

    def _merge_duplicate_slots(self) -> Dict[str, Any]:
        """Merge non-coexisting row anchors that describe one physical car."""
        candidates = []
        for left_index, left in enumerate(self.slots):
            if left.row_id is None or left.row_axis is None:
                continue
            for right in self.slots[:left_index]:
                if right.row_id != left.row_id:
                    continue
                delta = left.center - right.center
                cross_axis = np.array([-left.row_axis[1], left.row_axis[0]])
                along = abs(float(np.dot(delta, left.row_axis)))
                cross = abs(float(np.dot(delta, cross_axis)))
                if (along > self.duplicate_slot_along_gate
                        or cross > self.duplicate_slot_cross_gate):
                    continue
                if not tracking.class_compatible(left.class_name, right.class_name):
                    continue
                if (tracking.angle_distance(left.yaw, right.yaw, modulo_pi=True)
                        > self.slot_yaw_gate):
                    continue
                overlap = tracking.bev_iou(
                    np.r_[left.center, 0.0], left.size, left.yaw,
                    np.r_[right.center, 0.0], right.size, right.yaw)
                if overlap < self.duplicate_slot_iou_gate:
                    continue
                matched_overlap = {o.frame_index for o in left.matched} & {
                    o.frame_index for o in right.matched}
                evidence_overlap = left.evidence_frames & right.evidence_frames
                if matched_overlap or evidence_overlap:
                    continue
                strength_left = (len(left.matched), left.evidence_hits, -left.spread90)
                strength_right = (len(right.matched), right.evidence_hits, -right.spread90)
                keep, drop = ((left, right) if strength_left > strength_right
                              else (right, left))
                if (len(drop.matched)
                        > self.duplicate_slot_max_weak_fraction
                        * max(len(keep.matched), 1)):
                    continue
                candidates.append((-(len(drop.matched) + len(keep.matched)),
                                   keep.slot_id, drop.slot_id, keep, drop,
                                   along, cross, overlap))

        used = set()
        merges = []
        for (_negative_hits, _keep_id, _drop_id, keep, drop,
             along, cross, overlap) in sorted(candidates):
            if keep.slot_id in used or drop.slot_id in used:
                continue
            if any(o.frame_index in {x.frame_index for x in keep.matched}
                   for o in drop.matched):
                continue
            used.update((keep.slot_id, drop.slot_id))
            for obs in drop.matched:
                obs.detection["track_id"] = int(keep.track_id)
                keep.matched.append(obs)
            keep.alias_centers.append(drop.center.copy())
            keep.alias_centers.extend(center.copy() for center in drop.alias_centers)
            keep.evidence_frames.update(drop.evidence_frames)
            keep.evidence_hits += drop.evidence_hits
            first_frames = [x for x in (
                keep.first_matched_frame, drop.first_matched_frame) if x is not None]
            last_frames = [x for x in (
                keep.last_matched_frame, drop.last_matched_frame) if x is not None]
            keep.first_matched_frame = min(first_frames) if first_frames else None
            keep.last_matched_frame = max(last_frames) if last_frames else None
            merges.append({
                "kept_slot_id": keep.slot_id, "dropped_slot_id": drop.slot_id,
                "reassigned_detections": len(drop.matched),
                "along_offset": round(along, 3),
                "cross_offset": round(cross, 3), "bev_iou": round(overlap, 4),
            })

        dropped_ids = {x["dropped_slot_id"] for x in merges}
        if dropped_ids:
            self.slots = [s for s in self.slots if s.slot_id not in dropped_ids]
            for row in self.diagnostics["rows"]:
                row["slot_ids"] = [sid for sid in row["slot_ids"]
                                   if sid not in dropped_ids]
            self.diagnostics["static_slots"] = len(self.slots)
            self.diagnostics["static_matches"] = sum(len(s.matched) for s in self.slots)
        return {"merged_slots": len(merges), "merges": merges,
                "along_gate": self.duplicate_slot_along_gate,
                "cross_gate": self.duplicate_slot_cross_gate,
                "iou_gate": self.duplicate_slot_iou_gate,
                "max_weak_fraction": self.duplicate_slot_max_weak_fraction}

    def _run_dynamic(self, frames: Sequence[Dict[str, Any]],
                     assigned_detection_ids: set[int],
                     id_offset: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        dynamic_frames = []
        source_refs: List[List[Dict[str, Any]]] = []
        for frame in frames:
            refs = [d for d in frame.get("detections", [])
                    if (id(d) not in assigned_detection_ids
                        and d.get("track_id") is None)]
            dynamic_frames.append({
                "frame_id": frame["frame_id"],
                "num_points": frame.get("num_points", 0),
                "num_detections": len(refs),
                "detections": [copy.deepcopy(d) for d in refs],
            })
            source_refs.append(refs)
        self.motion_tracker = tracking.ConservativeTracker(
            self.coords,
            # Static evidence is authoritative in pass 1. Disable the legacy tracker's
            # track-first promotion so parking fragments cannot create anchors.
            min_static_hits=10 ** 9,
            dynamic_max_gap=self.dynamic_max_gap,
        )
        dynamic_output, dynamic_diag = self.motion_tracker.process(
            dynamic_frames, enable_stitching=False)
        for refs, frame in zip(source_refs, dynamic_output):
            for source, tracked in zip(refs, frame.get("detections", [])):
                if tracked.get("track_id") is not None:
                    source["track_id"] = int(tracked["track_id"]) + id_offset
        return dynamic_output, dynamic_diag

    def _all_tracks(self, frames: Sequence[Dict[str, Any]]) -> Dict[int, List[tracking.Observation]]:
        grouped: Dict[int, List[tracking.Observation]] = {}
        for frame_index, frame in enumerate(frames):
            wf = self.coords.world_from_lidar(int(frame["frame_id"]))
            if wf is None:
                continue
            for det in frame.get("detections", []):
                if det.get("track_id") is None:
                    continue
                box = det["box_lidar"]
                grouped.setdefault(int(det["track_id"]), []).append(tracking.Observation(
                    frame_index=frame_index, timestamp=int(frame["frame_id"]),
                    detection=det, world=tracking.center_world(box, wf),
                    yaw=tracking.yaw_world(float(box[6]), wf),
                    size=np.asarray(box[3:6], dtype=np.float64),
                ))
        for observations in grouped.values():
            observations.sort(key=lambda o: o.timestamp)
        return grouped

    @staticmethod
    def _endpoint_velocity(items: Sequence[tracking.Observation], at_end: bool) -> np.ndarray:
        selected = list(items[-4:] if at_end else items[:4])
        if len(selected) < 2:
            return np.zeros(2, dtype=np.float64)
        t0 = selected[0].timestamp
        times = np.asarray([(o.timestamp - t0) / 1e9 for o in selected])
        if float(np.ptp(times)) < 1e-3:
            return np.zeros(2, dtype=np.float64)
        design = np.column_stack([times, np.ones(len(times))])
        return np.asarray([
            np.linalg.lstsq(design, np.asarray([o.world[k] for o in selected]),
                            rcond=None)[0][0] for k in (0, 1)
        ], dtype=np.float64)

    def _near_parking_region(self, point: np.ndarray, radius: float = 2.5) -> bool:
        return any(float(np.linalg.norm(point[:2] - slot.center)) <= radius
                   for slot in self.slots)

    def _stitch_dynamic(self, frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge only mutual, short-gap motion continuations outside slots."""
        grouped = self._all_tracks(frames)
        static_ids = {int(slot.track_id) for slot in self.slots}
        candidates = []
        for end_id, end_items in grouped.items():
            if end_id in static_ids:
                continue
            end = end_items[-1]
            if self._near_parking_region(end.world):
                continue
            end_velocity = self._endpoint_velocity(end_items, at_end=True)
            for start_id, start_items in grouped.items():
                if start_id == end_id or start_id in static_ids:
                    continue
                start = start_items[0]
                gap = (start.timestamp - end.timestamp) / 1e9
                if gap <= 0.0 or gap > 1.2 or self._near_parking_region(start.world):
                    continue
                if not tracking.class_compatible(
                        end.detection.get("class_name", ""),
                        start.detection.get("class_name", "")):
                    continue
                size_delta = float(np.linalg.norm(end.size - start.size)) / max(
                    float(np.linalg.norm(end.size)), 1.0)
                if size_delta > 0.45:
                    continue
                start_velocity = self._endpoint_velocity(start_items, at_end=False)
                forward_error = float(np.linalg.norm(
                    start.world[:2] - (end.world[:2] + end_velocity * gap)))
                backward_error = float(np.linalg.norm(
                    end.world[:2] - (start.world[:2] - start_velocity * gap)))
                speed = max(float(np.linalg.norm(end_velocity)),
                            float(np.linalg.norm(start_velocity)))
                gate = min(4.5, 0.8 + 0.8 * gap + 0.15 * speed * gap)
                if forward_error > gate or backward_error > gate:
                    continue
                if np.linalg.norm(end_velocity) > 1.0 and np.linalg.norm(start_velocity) > 1.0:
                    cosine = float(np.dot(end_velocity, start_velocity) /
                                   (np.linalg.norm(end_velocity)
                                    * np.linalg.norm(start_velocity)))
                    if cosine < 0.5:
                        continue
                score = ((forward_error + backward_error) / max(2.0 * gate, 1e-6)
                         + 0.45 * size_delta + 0.06 * gap)
                candidates.append((score, end_id, start_id, gap,
                                   forward_error, backward_error))
        by_end: Dict[int, List[Tuple[Any, ...]]] = {}
        by_start: Dict[int, List[Tuple[Any, ...]]] = {}
        for candidate in candidates:
            by_end.setdefault(candidate[1], []).append(candidate)
            by_start.setdefault(candidate[2], []).append(candidate)
        accepted = []
        remap: Dict[int, int] = {}
        for candidate in sorted(candidates):
            score, end_id, start_id, gap, forward_error, backward_error = candidate
            if min(by_end[end_id]) != candidate or min(by_start[start_id]) != candidate:
                continue
            end_scores = sorted(x[0] for x in by_end[end_id])
            start_scores = sorted(x[0] for x in by_start[start_id])
            if len(end_scores) > 1 and end_scores[1] - score < 0.18:
                continue
            if len(start_scores) > 1 and start_scores[1] - score < 0.18:
                continue
            if end_id in remap.values() or start_id in remap:
                continue
            remap[start_id] = end_id
            accepted.append({
                "from_track_id": start_id, "to_track_id": end_id,
                "gap_sec": round(gap, 3), "score": round(score, 4),
                "forward_error": round(forward_error, 3),
                "backward_error": round(backward_error, 3),
            })

        def root(track_id: int) -> int:
            seen = set()
            while track_id in remap and track_id not in seen:
                seen.add(track_id)
                track_id = remap[track_id]
            return track_id

        for frame in frames:
            for det in frame.get("detections", []):
                if det.get("track_id") is not None:
                    det["track_id"] = root(int(det["track_id"]))
        return {"candidate_pairs": len(candidates),
                "stitched_pairs": len(accepted), "pairs": accepted,
                "parking_exclusion_radius": 2.5, "max_gap_sec": 1.2}

    def _topology_cost(self, obs: tracking.Observation,
                       slot: StaticSlot) -> Optional[Tuple[float, float, float]]:
        """Anisotropic row gate: strict between spaces, tolerant across them."""
        if slot.row_axis is None:
            return None
        class_name = obs.detection.get("class_name", "")
        if (class_name not in STATIC_CLASSES
                or not tracking.class_compatible(class_name, slot.class_name)
                or not size_compatible(obs.size, slot.size, max_delta=0.8)):
            return None
        yaw_delta = tracking.angle_distance(obs.yaw, slot.yaw, modulo_pi=True)
        if yaw_delta > 0.70:
            return None
        cross_axis = np.array([-slot.row_axis[1], slot.row_axis[0]])
        offsets = []
        for center in [slot.center, *slot.alias_centers]:
            delta = obs.world[:2] - center
            along = abs(float(np.dot(delta, slot.row_axis)))
            cross = abs(float(np.dot(delta, cross_axis)))
            if along <= self.topology_along_gate and cross <= self.topology_cross_gate:
                offsets.append((along / max(self.topology_along_gate, 1e-6)
                                + 0.35 * cross / max(self.topology_cross_gate, 1e-6),
                                along, cross))
        if not offsets:
            return None
        positional_cost, along, cross = min(offsets)
        size_delta = float(np.linalg.norm(obs.size - slot.size)) / max(
            float(np.linalg.norm(slot.size)), 1.0)
        cost = (positional_cost
                + 0.20 * yaw_delta / 0.70 + 0.15 * size_delta)
        return cost, along, cross

    def _ordered_row_matches(
            self, observations: Sequence[tracking.Observation],
            slots: Sequence[StaticSlot]) -> List[Tuple[tracking.Observation, StaticSlot,
                                                       float, float, float]]:
        """Maximum-cardinality monotonic one-to-one alignment for one frame."""
        ordered_obs = sorted(observations, key=lambda o: float(
            np.dot(o.world[:2], slots[0].row_axis)))
        ordered_slots = sorted(slots, key=lambda s: s.row_coord)
        n_obs, n_slots = len(ordered_obs), len(ordered_slots)
        best: List[List[Tuple[int, float]]] = [
            [(0, 0.0) for _ in range(n_slots + 1)] for _ in range(n_obs + 1)]
        choice = [["" for _ in range(n_slots + 1)] for _ in range(n_obs + 1)]

        def prefer(left: Tuple[int, float], right: Tuple[int, float]) -> bool:
            return left[0] > right[0] or (left[0] == right[0] and left[1] < right[1])

        for oi in range(1, n_obs + 1):
            choice[oi][0] = "obs"
        for si in range(1, n_slots + 1):
            choice[0][si] = "slot"
        for oi in range(1, n_obs + 1):
            for si in range(1, n_slots + 1):
                winner = best[oi - 1][si]
                action = "obs"
                if prefer(best[oi][si - 1], winner):
                    winner, action = best[oi][si - 1], "slot"
                metrics = self._topology_cost(
                    ordered_obs[oi - 1], ordered_slots[si - 1])
                if metrics is not None:
                    matched = (best[oi - 1][si - 1][0] + 1,
                               best[oi - 1][si - 1][1] + metrics[0])
                    if prefer(matched, winner):
                        winner, action = matched, "match"
                best[oi][si], choice[oi][si] = winner, action

        matches = []
        oi, si = n_obs, n_slots
        while oi > 0 and si > 0:
            action = choice[oi][si]
            if action == "match":
                obs, slot = ordered_obs[oi - 1], ordered_slots[si - 1]
                metrics = self._topology_cost(obs, slot)
                assert metrics is not None
                matches.append((obs, slot, *metrics))
                oi -= 1
                si -= 1
            elif action == "slot":
                si -= 1
            else:
                oi -= 1
        matches.reverse()
        return matches

    def _validate_row_topology(self, frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Recover parking identities rejected or fragmented by the first pass.

        This stage never invents an observation. It only relabels stationary
        fragments tied to one row position, or physically impossible tracks
        that move laterally through several occupied parking positions.
        """
        rows: Dict[int, List[StaticSlot]] = {}
        slot_by_track = {}
        for slot in self.slots:
            slot_by_track[int(slot.track_id)] = slot
            if slot.row_id is not None:
                rows.setdefault(slot.row_id, []).append(slot)
        static_ids = set(slot_by_track)
        proposed: Dict[int, Tuple[tracking.Observation, StaticSlot, float, float, float]] = {}
        proposed_count = 0

        for frame_index, frame in enumerate(frames):
            wf = self.coords.world_from_lidar(int(frame["frame_id"]))
            if wf is None:
                continue
            dynamic_obs = []
            claimed = set()
            for det in frame.get("detections", []):
                tid = int(det["track_id"])
                if tid in static_ids:
                    claimed.add(tid)
                    continue
                if det.get("class_name") not in STATIC_CLASSES or not tracking.finite_box(det):
                    continue
                box = det["box_lidar"]
                dynamic_obs.append(tracking.Observation(
                    frame_index=frame_index, timestamp=int(frame["frame_id"]),
                    detection=det, world=tracking.center_world(box, wf),
                    yaw=tracking.yaw_world(float(box[6]), wf),
                    size=np.asarray(box[3:6], dtype=np.float64)))

            by_row: Dict[int, List[tracking.Observation]] = {}
            for obs in dynamic_obs:
                row_choices = []
                for row_id, row_slots in rows.items():
                    feasible = [self._topology_cost(obs, slot) for slot in row_slots
                                if int(slot.track_id) not in claimed]
                    feasible = [x for x in feasible if x is not None]
                    if feasible:
                        row_choices.append((min(x[0] for x in feasible), row_id))
                if row_choices:
                    by_row.setdefault(min(row_choices)[1], []).append(obs)

            for row_id, observations in by_row.items():
                available = [slot for slot in rows[row_id]
                             if int(slot.track_id) not in claimed]
                if not available:
                    continue
                for match in self._ordered_row_matches(observations, available):
                    obs = match[0]
                    proposed[id(obs.detection)] = match
                    proposed_count += 1

        by_track: Dict[int, List[Tuple[tracking.Observation, StaticSlot,
                                       float, float, float]]] = {}
        for match in proposed.values():
            by_track.setdefault(int(match[0].detection["track_id"]), []).append(match)

        corrections = []
        corrected_detections = set()
        all_tracks = self._all_tracks(frames)
        for track_id, matches in by_track.items():
            matches.sort(key=lambda x: x[0].timestamp)
            all_track_items = all_tracks.get(track_id, [])
            clear_motion, motion_metrics = self._track_is_clear_motion(all_track_items)
            slot_counts: Dict[int, int] = {}
            for _obs, slot, _cost, _along, _cross in matches:
                slot_counts[slot.slot_id] = slot_counts.get(slot.slot_id, 0) + 1
            dominant_slot, dominant_hits = max(
                slot_counts.items(), key=lambda item: (item[1], -item[0]))
            stable = (not clear_motion
                      and dominant_hits >= self.topology_min_stable_hits
                      and dominant_hits >= math.ceil(0.8 * len(matches)))

            dominant_row = max(
                (slot.row_id for _obs, slot, *_rest in matches),
                key=lambda row_id: sum(slot.row_id == row_id
                                       for _obs, slot, *_rest in matches))
            row_matches = [x for x in matches if x[1].row_id == dominant_row]
            row_coords = np.asarray([float(np.dot(x[0].world[:2], x[1].row_axis))
                                     for x in row_matches])
            cross_coords = np.asarray([float(np.dot(
                x[0].world[:2], np.array([-x[1].row_axis[1], x[1].row_axis[0]])))
                                       for x in row_matches])
            distinct_slots = len({x[1].slot_id for x in row_matches})
            lateral_jump = (
                clear_motion and len(row_matches) >= 5 and distinct_slots >= 4
                and float(np.ptp(row_coords)) >= 5.0
                and float(np.ptp(row_coords)) >= 2.5 * max(float(np.ptp(cross_coords)), 0.5))

            selected = ([x for x in matches if x[1].slot_id == dominant_slot]
                        if stable else row_matches if lateral_jump else [])
            if not selected:
                continue
            changed = 0
            targets = set()
            for obs, slot, _cost, _along, _cross in selected:
                if id(obs.detection) in corrected_detections:
                    continue
                old_id = int(obs.detection["track_id"])
                new_id = int(slot.track_id)
                if old_id != new_id:
                    obs.detection["track_id"] = new_id
                    corrected_detections.add(id(obs.detection))
                    changed += 1
                    targets.add(new_id)
            if changed:
                corrections.append({
                    "source_track_id": track_id,
                    "reason": "stable_slot_fragment" if stable else "lateral_row_jump",
                    "detections": changed, "target_track_ids": sorted(targets),
                    "candidate_hits": len(matches), "distinct_slots": distinct_slots,
                    "motion": motion_metrics,
                })

        return {
            "candidate_matches": proposed_count,
            "corrected_detections": len(corrected_detections),
            "corrected_tracks": len(corrections),
            "corrections": corrections,
            "along_gate": self.topology_along_gate,
            "cross_gate": self.topology_cross_gate,
        }

    def _recover_slot_gap_fragments(
            self, frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Restore a slot ID only inside a short, bracketed observation gap."""
        by_frame: List[Dict[int, List[tracking.Observation]]] = []
        for frame_index, frame in enumerate(frames):
            wf = self.coords.world_from_lidar(int(frame["frame_id"]))
            grouped: Dict[int, List[tracking.Observation]] = {}
            if wf is not None:
                for det in frame.get("detections", []):
                    if det.get("track_id") is None or not tracking.finite_box(det):
                        continue
                    box = det["box_lidar"]
                    obs = tracking.Observation(
                        frame_index=frame_index, timestamp=int(frame["frame_id"]),
                        detection=det, world=tracking.center_world(box, wf),
                        yaw=tracking.yaw_world(float(box[6]), wf),
                        size=np.asarray(box[3:6], dtype=np.float64))
                    grouped.setdefault(int(det["track_id"]), []).append(obs)
            by_frame.append(grouped)

        recoveries = []
        corrected = set()
        for slot in self.slots:
            slot_id = int(slot.track_id)
            present = [index for index, grouped in enumerate(by_frame)
                       if slot_id in grouped]
            for left, right in zip(present, present[1:]):
                gap = right - left - 1
                if gap <= 0 or gap > self.topology_gap_max_frames:
                    continue
                candidates = []
                valid = True
                for frame_index in range(left + 1, right):
                    feasible = []
                    for tid, observations in by_frame[frame_index].items():
                        if tid == slot_id:
                            continue
                        for obs in observations:
                            metrics = self._topology_cost(obs, slot)
                            if metrics is not None:
                                feasible.append((metrics[0], tid, obs, metrics))
                    if len(feasible) != 1:
                        valid = False
                        break
                    if id(feasible[0][2].detection) in corrected:
                        valid = False
                        break
                    candidates.append(feasible[0])
                if not valid or not candidates:
                    continue
                source_ids = {item[1] for item in candidates}
                if len(source_ids) != 1:
                    continue
                source_id = next(iter(source_ids))
                # A moving object that enters a released position must keep its
                # own ID. Bracketing by the old slot plus a short gap prevents
                # this recovery from becoming an ingress identity transfer.
                for _cost, _tid, obs, _metrics in candidates:
                    obs.detection["track_id"] = slot_id
                    corrected.add(id(obs.detection))
                recoveries.append({
                    "slot_id": slot.slot_id, "source_track_id": source_id,
                    "left_frame": left, "right_frame": right,
                    "detections": len(candidates),
                    "max_cost": round(max(item[0] for item in candidates), 4),
                })
        return {"recoveries": recoveries, "recovery_count": len(recoveries),
                "corrected_detections": len(corrected),
                "max_gap_frames": self.topology_gap_max_frames}

    @staticmethod
    def _radial_departure(items: Sequence[tracking.Observation], center: np.ndarray) -> Tuple[bool, Optional[int]]:
        if len(items) < 4:
            return False, None
        radial = np.asarray([np.linalg.norm(o.world[:2] - center) for o in items])
        if radial[0] > 1.45 or float(np.max(radial)) < 3.0:
            return False, None
        limit = min(len(radial), 8)
        changes = np.diff(radial[:limit])
        times = np.asarray([o.timestamp for o in items[:limit]], dtype=np.float64) / 1e9
        intervals = np.diff(times)
        positive = ((changes > 0.30) & (intervals > 1e-3)
                    & (intervals <= 0.8))
        if int(np.count_nonzero(positive)) < 3 or float(np.sum(np.maximum(changes, 0.0))) < 2.0:
            return False, None
        indices = np.flatnonzero(radial >= 2.5)
        return (len(indices) > 0,
                None if len(indices) == 0 else int(items[int(indices[0])].timestamp))

    @staticmethod
    def _radial_ingress(items: Sequence[tracking.Observation], center: np.ndarray) -> bool:
        if len(items) < 4:
            return False
        radial = np.asarray([np.linalg.norm(o.world[:2] - center) for o in items])
        if radial[-1] > 1.45 or float(np.max(radial)) < 3.0:
            return False
        # Stop-and-go traffic may remain in the destination for many frames,
        # so inspect the approach up to the first entry rather than only the
        # last fixed-size window.
        entry_candidates = np.flatnonzero(radial <= 1.45)
        if len(entry_candidates) == 0:
            return False
        entry_index = int(entry_candidates[-1])
        for index in entry_candidates:
            if float(np.max(radial[:int(index) + 1])) >= 3.0:
                entry_index = int(index)
                break
        start_index = int(np.argmax(radial[:entry_index + 1]))
        changes = -np.diff(radial[start_index:entry_index + 1])
        return (int(np.count_nonzero(changes > 0.30)) >= 3
                and float(np.sum(np.maximum(changes, 0.0))) >= 2.0)

    def _coordinate_slot_motion(self, frames: Sequence[Dict[str, Any]],
                                next_id: int) -> Tuple[int, Dict[str, Any]]:
        """Apply explicit departure/ingress state transitions.

        Missing observations never release a slot. A dynamic trajectory can
        inherit a parked ID only after sustained outward motion. Conversely,
        a continuous trajectory entering a position that was empty from the
        start keeps its motion ID.
        """
        grouped = self._all_tracks(frames)
        static_ids = {int(slot.track_id) for slot in self.slots}
        dynamic = {tid: items for tid, items in grouped.items() if tid not in static_ids}
        departures = []
        ingresses = []
        rejected_departures = []
        used_dynamic = set()
        for slot in self.slots:
            slot_items = grouped.get(int(slot.track_id), [])
            if not slot_items:
                continue
            departure_candidates = []
            for tid, items in dynamic.items():
                if tid in used_dynamic:
                    continue
                clear, _metrics = self._track_is_clear_motion(items)
                explicit, confirmation_ts = self._radial_departure(items, slot.center)
                if not clear or not explicit:
                    continue
                if slot.row_axis is not None:
                    cross_axis = np.array([-slot.row_axis[1], slot.row_axis[0]])
                    deltas = np.asarray([o.world[:2] - slot.center for o in items])
                    along = np.abs(deltas @ slot.row_axis)
                    cross = np.abs(deltas @ cross_axis)
                    outside = np.flatnonzero(cross >= 1.35)
                    if (len(outside) == 0
                            or (float(np.max(along[:int(outside[0]) + 1]))
                                > 1.35)):
                        rejected_departures.append({
                            "slot_id": slot.slot_id, "dynamic_track_id": tid,
                            "reason": "no_initial_cross_row_departure",
                        })
                        continue
                start_distance = float(np.linalg.norm(items[0].world[:2] - slot.center))
                prior = [o for o in slot_items if o.timestamp < items[0].timestamp]
                if not prior:
                    continue
                prior_gap = (items[0].timestamp - prior[-1].timestamp) / 1e9
                bridge_distance = float(np.linalg.norm(
                    items[0].world[:2] - prior[-1].world[:2]))
                bridge_gate = 0.65 + 5.0 * max(prior_gap, 0.0)
                bridge_size_delta = float(np.linalg.norm(
                    items[0].size - prior[-1].size)) / max(
                        float(np.linalg.norm(prior[-1].size)), 1.0)
                bridge_yaw_delta = tracking.angle_distance(
                    items[0].yaw, prior[-1].yaw, modulo_pi=True)
                if (prior_gap > 0.8
                        or bridge_distance > bridge_gate
                        or bridge_size_delta > 0.35
                        or bridge_yaw_delta > 0.65):
                    continue
                confirmation_frame = next(
                    (o.frame_index for o in items
                     if o.timestamp >= int(confirmation_ts)), items[-1].frame_index)
                occupied_during_departure = [
                    o for o in slot_items
                    if (items[0].timestamp <= o.timestamp <= int(confirmation_ts)
                        and float(np.linalg.norm(o.world[:2] - slot.center)) <= 1.15)]
                occupied_after_confirmation = [
                    o for o in slot_items
                    if (int(confirmation_ts) < o.timestamp
                        and o.frame_index <= (confirmation_frame
                                              + self.topology_occupancy_window)
                        and float(np.linalg.norm(o.world[:2] - slot.center)) <= 1.15)]
                if occupied_during_departure and occupied_after_confirmation:
                    rejected_departures.append({
                        "slot_id": slot.slot_id, "dynamic_track_id": tid,
                        "reason": "slot_remains_occupied",
                        "first_overlapping_timestamp": int(
                            occupied_during_departure[0].timestamp),
                        "first_later_timestamp": int(
                            occupied_after_confirmation[0].timestamp),
                    })
                    continue
                departure_candidates.append((items[0].timestamp, start_distance, tid,
                                             confirmation_ts, prior_gap,
                                             bridge_distance))
            if departure_candidates:
                (start_ts, start_distance, tid, confirmation_ts, prior_gap,
                 bridge_distance) = min(departure_candidates)
                used_dynamic.add(tid)
                for frame in frames:
                    for det in frame.get("detections", []):
                        if det.get("track_id") == tid:
                            det["track_id"] = int(slot.track_id)
                departures.append({
                    "slot_id": slot.slot_id, "dynamic_track_id": tid,
                    "inherited_track_id": int(slot.track_id),
                    "start_timestamp": int(start_ts),
                    "confirmation_timestamp": int(confirmation_ts),
                    "start_distance": round(start_distance, 3),
                    "prior_static_gap_sec": round(prior_gap, 3),
                    "bridge_distance": round(bridge_distance, 3),
                    "later_occupancy_track_id": None,
                })

        # Every slot is immutable until its explicit departure event. After
        # release, a coherent incoming motion track owns the new occupancy;
        # otherwise the first later static observation starts a new identity.
        departure_by_slot = {item["slot_id"]: item for item in departures}
        for slot in self.slots:
            departure = departure_by_slot.get(slot.slot_id)
            if departure is None:
                continue
            confirmation_ts = int(departure["confirmation_timestamp"])
            departing_tid = int(departure["dynamic_track_id"])
            candidates = []
            for tid, items in dynamic.items():
                if tid == departing_tid:
                    continue
                if items[-1].timestamp < confirmation_ts:
                    continue
                clear, _metrics = self._track_is_clear_motion(items)
                if not clear or not self._radial_ingress(items, slot.center):
                    continue
                arrival = next((o for o in items
                                if (o.timestamp >= confirmation_ts
                                    and float(np.linalg.norm(
                                        o.world[:2] - slot.center)) <= 1.45)), None)
                if arrival is None:
                    continue
                candidates.append((arrival.timestamp,
                                   float(np.linalg.norm(arrival.world[:2] - slot.center)),
                                   tid, items[-1].timestamp))
            slot_items = grouped.get(int(slot.track_id), [])
            later_static = [o for o in slot_items if o.timestamp > confirmation_ts]
            if candidates:
                arrival_ts, arrival_distance, tid, end_ts = min(candidates)
                for obs in later_static:
                    if obs.timestamp >= arrival_ts:
                        obs.detection["track_id"] = tid
                ingresses.append({
                    "slot_id": slot.slot_id, "kept_dynamic_track_id": tid,
                    "arrival_timestamp": int(arrival_ts),
                    "arrival_distance": round(arrival_distance, 3),
                    "last_dynamic_timestamp": int(end_ts),
                })
                departure["later_occupancy_track_id"] = tid
                # Any detections between confirmed release and the incoming
                # trajectory's arrival are not allowed to retain the old ID.
                orphan = [o for o in later_static if o.timestamp < arrival_ts]
            else:
                orphan = later_static
            if orphan:
                replacement_id = next_id
                next_id += 1
                for obs in orphan:
                    obs.detection["track_id"] = replacement_id
                if departure["later_occupancy_track_id"] is None:
                    departure["later_occupancy_track_id"] = replacement_id
        return next_id, {"departures": departures, "ingresses": ingresses,
                         "rejected_departures": rejected_departures,
                         "departure_count": len(departures),
                         "ingress_count": len(ingresses),
                         "rejected_departure_count": len(rejected_departures)}

    def process(self, frames: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        working = copy.deepcopy(list(frames))
        for frame in working:
            for det in frame.get("detections", []):
                det.pop("track_id", None)
        by_frame = self._world_observations(working)
        protected_keys, motion_probe_diag = self._motion_probe(working)
        self.diagnostics["motion_probe"] = motion_probe_diag
        candidates = self._discover_candidates(working, by_frame)
        self.slots = self._build_slots(candidates)
        for slot in self.slots:
            slot.track_id = slot.slot_id
        protected_track_ids = sorted(set(protected_keys.values()))
        protected_remap = {
            probe_id: len(self.slots) + index + 1
            for index, probe_id in enumerate(protected_track_ids)
        }
        for (frame_index, detection_index), probe_id in protected_keys.items():
            working[frame_index]["detections"][detection_index]["track_id"] = (
                protected_remap[probe_id])
        static_assigned = self._assign_static(by_frame)
        duplicate_diag = self._merge_duplicate_slots()
        self.diagnostics["duplicate_static_slots"] = duplicate_diag
        dynamic_offset = max(
            [int(s.track_id) for s in self.slots]
            + list(protected_remap.values()) + [0])
        _dynamic_output, dynamic_diag = self._run_dynamic(
            working, static_assigned, dynamic_offset)
        stitch_diag = self._stitch_dynamic(working)
        self.diagnostics["conservative_dynamic_stitching"] = stitch_diag
        topology_diag = self._validate_row_topology(working)
        self.diagnostics["parking_row_topology"] = topology_diag
        gap_diag = self._recover_slot_gap_fragments(working)
        self.diagnostics["slot_gap_recovery"] = gap_diag
        next_id = max((int(d["track_id"])
                       for frame in working for d in frame.get("detections", [])
                       if d.get("track_id") is not None), default=0) + 1
        _next_id, coordination = self._coordinate_slot_motion(working, next_id)
        self.diagnostics["slot_motion_coordination"] = coordination
        self.diagnostics["dynamic"] = dynamic_diag
        all_ids = {int(d["track_id"]) for frame in working
                   for d in frame.get("detections", [])
                   if d.get("track_id") is not None}
        self.diagnostics["tracks_total"] = len(all_ids)
        self.diagnostics["births"] = (
            len(protected_track_ids) + int(dynamic_diag.get("births", 0)))
        self.diagnostics["matches"] = (
            int(self.diagnostics["static_matches"]) + int(dynamic_diag.get("matches", 0)))
        self.diagnostics["slot_details"] = [{
            "slot_id": s.slot_id,
            "track_id": s.track_id,
            "center": [round(float(x), 4) for x in s.center],
            "yaw": round(float(s.yaw), 5),
            "size": [round(float(x), 4) for x in s.size],
            "evidence_hits": s.evidence_hits,
            "matched_hits": len(s.matched),
            "spread90": round(float(s.spread90), 4),
            "row_id": s.row_id,
            "alias_centers": len(s.alias_centers),
        } for s in self.slots]
        return working, self.diagnostics


def verify_identity_only(source: Sequence[Dict[str, Any]],
                         tracked: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Prove that association did not mutate or synthesize raw detections."""
    if len(source) != len(tracked):
        raise AssertionError("tracking changed the number of frames")
    checked = 0
    for raw_frame, out_frame in zip(source, tracked):
        if raw_frame.get("frame_id") != out_frame.get("frame_id"):
            raise AssertionError("tracking changed frame order")
        raw_dets = raw_frame.get("detections", [])
        out_dets = out_frame.get("detections", [])
        if len(raw_dets) != len(out_dets):
            raise AssertionError("tracking generated or removed a detection")
        for raw, out in zip(raw_dets, out_dets):
            comparable = copy.deepcopy(out)
            comparable.pop("track_id", None)
            raw_comparable = copy.deepcopy(raw)
            raw_comparable.pop("track_id", None)
            if comparable != raw_comparable:
                raise AssertionError("tracking mutated a detection field other than track_id")
            checked += 1
    return {"frames": len(source), "detections": checked, "passed": True}


def run(in_json: Path, clip: Path, out_json: Path, out_clip: Optional[Path],
        diagnostics_path: Path, min_lifecycle: int = 4) -> Dict[str, Any]:
    source = json.loads(in_json.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError(f"input must be a list of frames: {in_json}")
    tracker = StaticFirstTracker(tracking.CoordinateProvider(clip))
    tracked, diagnostics = tracker.process(source)
    diagnostics["identity_only_check"] = verify_identity_only(source, tracked)
    diagnostics["post_filter"] = tracking.apply_post_filters(
        tracked, min_lifecycle=min_lifecycle)
    diagnostics["final_detections"] = sum(
        len(frame.get("detections", [])) for frame in tracked)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(tracked, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    if out_clip is not None:
        diagnostics["sust_labels"] = tracking.export_clip(tracked, clip, out_clip)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Static-first clip-local BEV tracker")
    parser.add_argument("--in-json", "--in_json", dest="in_json", required=True, type=Path)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--out-json", "--out_json", dest="out_json", required=True, type=Path)
    parser.add_argument("--out-clip", "--out_clip", dest="out_clip", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--min-lifecycle", type=int, default=4,
                        help="drop tracks observed in <= this many frames")
    args = parser.parse_args()
    diagnostics_path = args.diagnostics or args.out_json.with_name(
        args.out_json.stem + "_diagnostics.json")
    diagnostics = run(args.in_json, args.clip, args.out_json, args.out_clip,
                      diagnostics_path, args.min_lifecycle)
    print(json.dumps({
        key: diagnostics.get(key) for key in (
            "frames", "detections", "static_slots", "static_matches",
            "static_ambiguous_unassigned", "tracks_total", "births",
            "matches", "final_detections", "sust_labels")
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
