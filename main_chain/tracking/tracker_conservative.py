#!/usr/bin/env python3
"""Conservative, clip-local BEV identity tracking.

This module deliberately separates identity association from later annotation
cleanup.  Input detections are copied byte-for-byte as far as their geometry
is concerned; tracking only adds ``track_id``.  A parking anchor is learned
from an already established vehicle track and is never used to create or
recycle an identity.  Once an anchored vehicle is parked, missing detections
do not release the anchor.  It is released only after observations provide
clear outward motion evidence.

The coordinate contract is explicit:

    world_from_lidar_top(t) = world_from_pose(t)
                              @ inv(base_from_pose)
                              @ base_from_lidar_top

``pose_data.txt`` supplies ``world_from_pose`` and ``calib.json`` supplies the
two static extrinsics.  This is the same transform direction used by the
project's ``SensorCalib.get_transform`` and ``visualize_map.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


STATIC_CLASSES = {"Vehicle", "Car", "Truck"}
VEHICLE_CLASSES = STATIC_CLASSES | {"Bus", "Other Vehicle"}
CLASS_MAP = {
    "car": "Car", "truck": "Truck", "bus": "Bus",
    "construction_vehicle": "Engineering_vehicle",
    "pedestrian": "Pedestrian", "bicycle": "Nonmotorized_vehicle",
    "motorcycle": "Nonmotorized_vehicle", "Cyclist": "Nonmotorized_vehicle",
    "Car": "Car", "Truck": "Truck", "Vehicle": "Car",
    "Pedestrian": "Pedestrian", "Cyclist": "Nonmotorized_vehicle",
}


def wrap_angle(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def angle_distance(a: float, b: float, modulo_pi: bool = False) -> float:
    d = abs(wrap_angle(float(a) - float(b)))
    return min(d, math.pi - d) if modulo_pi else d


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        q = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        q /= n
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def slerp(q0: Sequence[float], q1: Sequence[float], alpha: float) -> np.ndarray:
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    a /= max(float(np.linalg.norm(a)), 1e-12)
    b /= max(float(np.linalg.norm(b)), 1e-12)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        out = a + float(alpha) * (b - a)
        return out / max(float(np.linalg.norm(out)), 1e-12)
    theta = math.acos(max(-1.0, min(1.0, dot)))
    st = math.sin(theta)
    return (math.sin((1 - alpha) * theta) / st) * a + (math.sin(alpha * theta) / st) * b


class CoordinateProvider:
    """Interpolate pose data and expose named transform directions."""

    def __init__(self, clip: Path, max_pose_gap_sec: float = 0.6):
        transforms = clip / "transforms"
        self.pose_path = transforms / "pose_data.txt"
        self.calib_path = transforms / "calib.json"
        if not self.pose_path.exists() or not self.calib_path.exists():
            raise FileNotFoundError(f"clip lacks transforms/pose_data.txt or calib.json: {clip}")
        calib = json.loads(self.calib_path.read_text(encoding="utf-8"))
        tf = calib.get("tf2base_link", {})
        self.base_from_pose = np.asarray(tf["pose"], dtype=np.float64)
        self.base_from_lidar_top = np.asarray(tf["lidar_top"], dtype=np.float64)
        for name, mat in (("base_from_pose", self.base_from_pose),
                          ("base_from_lidar_top", self.base_from_lidar_top)):
            if mat.shape != (4, 4) or not np.all(np.isfinite(mat)):
                raise ValueError(f"invalid {name} in {self.calib_path}")
        self.lidar_top_from_pose = np.linalg.inv(self.base_from_pose) @ self.base_from_lidar_top
        self.max_gap_ns = int(max_pose_gap_sec * 1e9)
        self.rows: List[Tuple[int, float, float, float, float, float, float, float]] = []
        for line in self.pose_path.read_text(encoding="utf-8").splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < 8:
                continue
            try:
                row = tuple(float(x) for x in p[:8])
            except ValueError:
                continue
            self.rows.append((int(row[0]), *row[1:]))
        unique = {r[0]: r for r in self.rows}
        self.rows = [unique[k] for k in sorted(unique)]
        self.timestamps = [r[0] for r in self.rows]
        self._cache: Dict[int, np.ndarray] = {}

    @property
    def description(self) -> str:
        return "pose_data(world_from_pose) @ inv(tf2base_link.pose) @ tf2base_link.lidar_top"

    def world_from_lidar(self, timestamp: int) -> Optional[np.ndarray]:
        ts = int(timestamp)
        if ts in self._cache:
            return self._cache[ts]
        if not self.rows:
            return None
        import bisect
        i = bisect.bisect_left(self.timestamps, ts)
        if i == 0:
            r0 = r1 = self.rows[0]
            alpha = 0.0
        elif i >= len(self.rows):
            r0 = r1 = self.rows[-1]
            alpha = 0.0
        else:
            r0, r1 = self.rows[i - 1], self.rows[i]
            span = r1[0] - r0[0]
            if span <= 0 or (self.max_gap_ns > 0 and span > self.max_gap_ns):
                r0 = r1 = min((r0, r1), key=lambda r: abs(r[0] - ts))
                alpha = 0.0
            else:
                alpha = (ts - r0[0]) / span
        q = slerp(r0[4:8], r1[4:8], alpha)
        t = np.array([r0[j] + alpha * (r1[j] - r0[j]) for j in (1, 2, 3)])
        world_from_pose = np.eye(4, dtype=np.float64)
        world_from_pose[:3, :3] = quat_to_matrix(*q[:3], q[3])
        world_from_pose[:3, 3] = t
        out = world_from_pose @ self.lidar_top_from_pose
        self._cache[ts] = out
        return out

    def lidar_from_world(self, timestamp: int) -> Optional[np.ndarray]:
        w = self.world_from_lidar(timestamp)
        return None if w is None else np.linalg.inv(w)


def class_compatible(a: str, b: str) -> bool:
    if not a or not b:
        return True
    if a == b:
        return True
    return a in VEHICLE_CLASSES and b in VEHICLE_CLASSES


def center_world(box: Sequence[float], world_from_lidar: np.ndarray) -> np.ndarray:
    h = np.array([float(box[0]), float(box[1]), float(box[2]), 1.0])
    return (world_from_lidar @ h)[:3]


def yaw_world(local_yaw: float, world_from_lidar: np.ndarray) -> float:
    v = world_from_lidar[:3, :3] @ np.array([math.cos(local_yaw), math.sin(local_yaw), 0.0])
    return math.atan2(float(v[1]), float(v[0]))


def finite_box(det: Dict[str, Any]) -> bool:
    box = det.get("box_lidar")
    return isinstance(box, list) and len(box) >= 7 and bool(np.all(np.isfinite(np.asarray(box[:7], dtype=float))))


def box_lidar_to_base_link(
        box: Sequence[float],
        base_from_lidar_top: np.ndarray,
) -> List[float]:
    """Transform a lidar-top box to the base_link box representation.

    The box format stores a center, three extents, and one planar yaw.  A
    rigid sensor extrinsic preserves the extents; the center is transformed
    homogeneously and the heading vector is rotated before its XY yaw is
    recovered.  If the extrinsic contains roll/pitch, the returned yaw is the
    XY projection because this seven-value format cannot encode a fully tilted
    3-D box.
    """
    values = np.asarray(box[:7], dtype=np.float64)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError("box must contain seven finite values")
    transform = np.asarray(base_from_lidar_top, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("base_from_lidar_top must be a finite 4x4 matrix")
    rotation = transform[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if abs(determinant) < 1e-9:
        raise ValueError("base_from_lidar_top rotation is singular")

    center = (transform @ np.r_[values[:3], 1.0])[:3]
    heading = rotation @ np.array([
        math.cos(float(values[6])), math.sin(float(values[6])), 0.0,
    ], dtype=np.float64)
    horizontal_norm = float(np.linalg.norm(heading[:2]))
    if horizontal_norm < 1e-9:
        raise ValueError("box heading has no base_link XY projection")
    yaw = math.atan2(float(heading[1]), float(heading[0]))
    return [
        float(center[0]), float(center[1]), float(center[2]),
        float(values[3]), float(values[4]), float(values[5]), float(yaw),
    ]


def rectangle_corners(center: Sequence[float], size: Sequence[float], yaw: float) -> np.ndarray:
    hx, hy = max(float(size[0]), 1e-3) / 2.0, max(float(size[1]), 1e-3) / 2.0
    local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]], dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rot.T + np.asarray(center[:2], dtype=np.float64)


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    return abs(float(np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
                     - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))) * 0.5


def _inside(p: np.ndarray, a: np.ndarray, b: np.ndarray, orientation: float) -> bool:
    u, v = b - a, p - a
    cross = float(u[0] * v[1] - u[1] * v[0])
    return cross * orientation >= -1e-9


def _intersection(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r, s = p2 - p1, b - a
    den = float(r[0] * s[1] - r[1] * s[0])
    if abs(den) < 1e-12:
        return p2.copy()
    ap = a - p1
    t = float(ap[0] * s[1] - ap[1] * s[0]) / den
    return p1 + t * r


def convex_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = [p.copy() for p in subject]
    signed = float(np.dot(clip[:, 0], np.roll(clip[:, 1], -1))
                   - np.dot(clip[:, 1], np.roll(clip[:, 0], -1)))
    orientation = 1.0 if signed >= 0 else -1.0
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        if not output:
            break
        source = output
        output = []
        prev = source[-1]
        prev_inside = _inside(prev, a, b, orientation)
        for cur in source:
            cur_inside = _inside(cur, a, b, orientation)
            if cur_inside:
                if not prev_inside:
                    output.append(_intersection(prev, cur, a, b))
                output.append(cur)
            elif prev_inside:
                output.append(_intersection(prev, cur, a, b))
            prev, prev_inside = cur, cur_inside
    return np.asarray(output, dtype=np.float64)


def bev_iou(center_a: Sequence[float], size_a: Sequence[float], yaw_a: float,
            center_b: Sequence[float], size_b: Sequence[float], yaw_b: float) -> float:
    pa = rectangle_corners(center_a, size_a, yaw_a)
    pb = rectangle_corners(center_b, size_b, yaw_b)
    inter = polygon_area(convex_intersection(pa, pb))
    union = polygon_area(pa) + polygon_area(pb) - inter
    return 0.0 if union <= 1e-9 else max(0.0, min(1.0, inter / union))


@dataclass
class Observation:
    frame_index: int
    timestamp: int
    detection: Dict[str, Any]
    world: np.ndarray
    yaw: float
    size: np.ndarray


@dataclass
class Track:
    track_id: int
    class_name: str
    first_ts: int
    last_ts: int
    last_world: np.ndarray
    velocity: np.ndarray
    state: np.ndarray
    covariance: np.ndarray
    size: np.ndarray
    last_yaw: float
    observations: List[Observation] = field(default_factory=list)
    missed_sec: float = 0.0
    is_static: bool = False
    slot_id: Optional[int] = None
    slot_anchor: Optional[np.ndarray] = None
    departure_count: int = 0
    outward_count: int = 0
    last_match_reason: str = "birth"

    def predict(self, timestamp: int, process_noise: float = 3.0) -> Tuple[np.ndarray, np.ndarray, float]:
        dt = max(0.0, (int(timestamp) - self.last_ts) / 1e9)
        f = np.array([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt],
                      [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        q = float(process_noise) ** 2
        qmat = q * np.array([
            [dt ** 4 / 4, 0.0, dt ** 3 / 2, 0.0],
            [0.0, dt ** 4 / 4, 0.0, dt ** 3 / 2],
            [dt ** 3 / 2, 0.0, dt ** 2, 0.0],
            [0.0, dt ** 3 / 2, 0.0, dt ** 2],
        ])
        state = f @ self.state
        covariance = f @ self.covariance @ f.T + qmat
        predicted = self.last_world.copy()
        predicted[:2] = state[:2]
        return predicted, covariance, dt


class ConservativeTracker:
    """Two-stage association with persistent parking anchors."""

    def __init__(self, coords: CoordinateProvider, *, min_static_hits: int = 6,
                 min_static_duration: float = 1.0, static_span: float = 1.25,
                 static_speed: float = 0.8, static_reacquire_radius: float = 2.25,
                 departure_radius: float = 2.5, departure_frames: int = 3,
                 dynamic_max_gap: float = 1.8, dynamic_base_gate: float = 1.6,
                 dynamic_max_velocity: float = 28.0, dynamic_max_gate: float = 9.0):
        self.coords = coords
        self.min_static_hits = int(min_static_hits)
        self.min_static_duration = float(min_static_duration)
        self.static_span = float(static_span)
        self.static_speed = float(static_speed)
        self.static_reacquire_radius = float(static_reacquire_radius)
        self.departure_radius = float(departure_radius)
        self.departure_frames = int(departure_frames)
        self.dynamic_max_gap = float(dynamic_max_gap)
        self.dynamic_base_gate = float(dynamic_base_gate)
        self.dynamic_max_velocity = float(dynamic_max_velocity)
        self.dynamic_max_gate = float(dynamic_max_gate)
        self.next_id = 1
        self.next_slot = 1
        self.tracks: Dict[int, Track] = {}
        self.diagnostics: Dict[str, Any] = {
            "coordinate_transform": coords.description,
            "frames": 0, "detections": 0, "matches": 0, "births": 0,
            "static_locks": 0, "static_recoveries": 0, "departures": 0,
            "rejections": {}, "ambiguous_recoveries": 0, "events": [],
        }

    def _event(self, event: str, tr: Track, timestamp: int, **fields: Any) -> None:
        row = {"event": event, "track_id": tr.track_id, "timestamp": int(timestamp)}
        row.update(fields)
        self.diagnostics["events"].append(row)

    def _reject(self, reason: str) -> None:
        self.diagnostics["rejections"][reason] = self.diagnostics["rejections"].get(reason, 0) + 1

    def _world_detections(self, frame: Dict[str, Any], frame_index: int) -> List[Observation]:
        ts = int(frame["frame_id"])
        wf = self.coords.world_from_lidar(ts)
        out: List[Observation] = []
        for det in frame.get("detections", []):
            if not finite_box(det) or wf is None:
                self._reject("invalid_box_or_pose")
                continue
            box = det["box_lidar"]
            out.append(Observation(
                frame_index=frame_index, timestamp=ts, detection=det,
                world=center_world(box, wf), yaw=yaw_world(float(box[6]), wf),
                size=np.asarray(box[3:6], dtype=np.float64),
            ))
        return out

    def _cost(self, tr: Track, obs: Observation, predicted: np.ndarray,
              covariance: np.ndarray, dt: float,
              static_mode: bool = False) -> Tuple[float, str]:
        dxy = float(np.linalg.norm((obs.world - predicted)[:2]))
        if static_mode:
            if tr.slot_anchor is None:
                return 1e9, "static_anchor_gate"
            anchor_dist = float(np.linalg.norm(obs.world[:2] - tr.slot_anchor[:2]))
            # Immediately after a valid observation, permit a wider corridor
            # so an actually departing car keeps its identity long enough to
            # produce explicit release evidence. After a long occlusion only
            # the original parking anchor may recover the identity.
            recent_departure_gate = dt <= 0.8 and anchor_dist <= self.departure_radius + 3.0
            if anchor_dist > self.static_reacquire_radius and not recent_departure_gate:
                return 1e9, "static_anchor_gate"
            gate = self.departure_radius + 3.0 if recent_departure_gate else self.static_reacquire_radius
            # A parked identity has no valid velocity extrapolation across a
            # long occlusion. Its prior is the persistent parking anchor.
            if not recent_departure_gate:
                predicted = predicted.copy()
                predicted[:2] = tr.slot_anchor[:2]
                covariance = covariance.copy()
                covariance[:2, :2] = np.eye(2) * max(0.45, self.static_reacquire_radius / 2.0) ** 2
                dxy = anchor_dist
        else:
            gate = min(self.dynamic_max_gate,
                       self.dynamic_base_gate + self.dynamic_max_velocity * max(dt - 0.1, 0.0))
            if dxy > gate:
                return 1e9, "distance_gate"
        if not class_compatible(obs.detection.get("class_name", ""), tr.class_name):
            return 1e9, "class_gate"
        scale_delta = float(np.linalg.norm(obs.size - tr.size) / max(float(np.linalg.norm(tr.size)), 1.0))
        if scale_delta > 1.35 and dxy > 0.75:
            return 1e9, "size_gate"
        yaw_delta = angle_distance(obs.yaw, tr.last_yaw, modulo_pi=True)
        innovation = obs.world[:2] - predicted[:2]
        innovation_cov = covariance[:2, :2] + np.eye(2) * 0.35 ** 2
        try:
            mahalanobis2 = float(innovation @ np.linalg.solve(innovation_cov, innovation))
        except np.linalg.LinAlgError:
            return 1e9, "covariance_gate"
        if mahalanobis2 > 13.82:  # chi-square(2), 99.9%; hard probabilistic gate
            return 1e9, "mahalanobis_gate"
        iou = bev_iou(predicted, tr.size, tr.last_yaw, obs.world, obs.size, obs.yaw)
        if dt <= 0.5 and dxy > 0.8 and iou < 0.01 and mahalanobis2 > 6.0:
            return 1e9, "iou_gate"
        # A box heading is symmetric modulo pi. Position uncertainty and IoU
        # dominate; size and yaw only resolve close, dense-scene alternatives.
        cost = math.sqrt(max(0.0, mahalanobis2)) / math.sqrt(13.82)
        cost += 0.55 * min(scale_delta, 2.0) + 0.25 * (1.0 - iou)
        cost += 0.10 * (yaw_delta / math.pi)
        cost += 0.02 * max(0.0, dt)
        cost -= min(0.04, max(0.0, float(obs.detection.get("score", 0.0))) * 0.02)
        return cost, "static_anchor" if static_mode else "motion"

    def _associate(self, tracks: List[Track], observations: List[Observation], *, static_mode: bool = False) -> Tuple[Dict[int, int], set]:
        if not tracks or not observations:
            return {}, set()
        n_obs, n_tracks = len(observations), len(tracks)
        real_cost = np.full((n_obs, n_tracks), 1e9, dtype=np.float64)
        reasons: Dict[Tuple[int, int], str] = {}
        for oi, obs in enumerate(observations):
            for ti, tr in enumerate(tracks):
                pred, covariance, dt = tr.predict(obs.timestamp)
                c, reason = self._cost(tr, obs, pred, covariance, dt, static_mode=static_mode)
                real_cost[oi, ti] = c
                reasons[(oi, ti)] = reason
        # Square augmented assignment:
        #   top-left     real detection -> real track
        #   top-right    detection -> its own birth dummy
        #   bottom-left  track -> its own unmatched/death dummy
        #   bottom-right dummy-to-dummy completion at zero cost
        # A real match competes against one birth plus one missed-track edge;
        # two half-cost edges preserve the intended 1.25 acceptance threshold.
        unmatched_cost = 0.625
        size = n_obs + n_tracks
        cost = np.full((size, size), 1e6, dtype=np.float64)
        cost[:n_obs, :n_tracks] = real_cost
        for oi in range(n_obs):
            cost[oi, n_tracks + oi] = unmatched_cost
        for ti in range(n_tracks):
            cost[n_obs + ti, ti] = unmatched_cost
        cost[n_obs:, n_tracks:] = 0.0
        rows, cols = linear_sum_assignment(cost)
        matches: Dict[int, int] = {}
        used_obs = set()
        for oi, ti in zip(rows, cols):
            if oi >= n_obs or ti >= n_tracks:
                continue
            if real_cost[oi, ti] >= 1e8:
                self._reject(reasons[(oi, ti)])
                continue
            matches[ti] = oi
            used_obs.add(oi)
        return matches, used_obs

    def _maybe_lock_static(self, tr: Track) -> None:
        if tr.is_static or tr.class_name not in STATIC_CLASSES or len(tr.observations) < self.min_static_hits:
            return
        times = [o.timestamp for o in tr.observations]
        duration = (max(times) - min(times)) / 1e9
        if duration < self.min_static_duration:
            return
        xy = np.asarray([o.world[:2] for o in tr.observations], dtype=np.float64)
        anchor = np.median(xy, axis=0)
        span = float(np.percentile(np.linalg.norm(xy - anchor, axis=1), 90))
        speeds = []
        for a, b in zip(tr.observations, tr.observations[1:]):
            dt = max((b.timestamp - a.timestamp) / 1e9, 1e-3)
            speeds.append(float(np.linalg.norm((b.world - a.world)[:2]) / dt))
        if span <= self.static_span and (not speeds or float(np.median(speeds)) <= self.static_speed):
            tr.is_static = True
            tr.slot_id = self.next_slot
            self.next_slot += 1
            tr.slot_anchor = anchor.copy()
            self.diagnostics["static_locks"] += 1
            self._event("static_lock", tr, tr.last_ts, slot_id=tr.slot_id,
                        anchor=[round(float(x), 4) for x in anchor])

    def _update(self, tr: Track, obs: Observation, reason: str) -> None:
        dt = max((obs.timestamp - tr.last_ts) / 1e9, 1e-3)
        _predicted, predicted_covariance, _ = tr.predict(obs.timestamp)
        f = np.array([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt],
                      [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        predicted_state = f @ tr.state
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        r = np.eye(2) * 0.35 ** 2
        innovation = obs.world[:2] - h @ predicted_state
        s = h @ predicted_covariance @ h.T + r
        gain = predicted_covariance @ h.T @ np.linalg.inv(s)
        tr.state = predicted_state + gain @ innovation
        tr.covariance = (np.eye(4) - gain @ h) @ predicted_covariance
        tr.velocity[:2] = tr.state[2:4]
        speed = float(np.linalg.norm(tr.velocity[:2]))
        if speed > self.dynamic_max_velocity:
            tr.velocity[:2] *= self.dynamic_max_velocity / max(speed, 1e-9)
        tr.last_world = obs.world.copy()
        tr.last_ts = obs.timestamp
        tr.last_yaw = obs.yaw
        tr.size = 0.7 * tr.size + 0.3 * obs.size
        tr.missed_sec = 0.0
        tr.last_match_reason = reason
        tr.observations.append(obs)
        if tr.is_static and tr.slot_anchor is not None:
            radial_vector = obs.world[:2] - tr.slot_anchor[:2]
            radial = float(np.linalg.norm(radial_vector))
            outward = float(np.dot(radial_vector, tr.velocity[:2])) > 0.0
            # Release requires a clear outward excursion plus sustained
            # motion. Use a floor below the parked-speed threshold so a
            # slowly departing car is not mistaken for detector jitter.
            departure_speed = max(0.55, self.static_speed * 0.75)
            if radial > self.departure_radius and outward and speed > departure_speed:
                tr.departure_count += 1
            else:
                tr.departure_count = max(0, tr.departure_count - 1)
            if tr.departure_count >= self.departure_frames:
                tr.is_static = False
                tr.slot_anchor = None
                tr.slot_id = None
                tr.diagnostics_departure = True
                self.diagnostics["departures"] += 1
                self._event("confirmed_departure", tr, obs.timestamp)
        self._maybe_lock_static(tr)

    def _birth(self, obs: Observation) -> Track:
        tid = self.next_id
        self.next_id += 1
        state = np.array([obs.world[0], obs.world[1], 0.0, 0.0], dtype=np.float64)
        covariance = np.diag([0.35 ** 2, 0.35 ** 2, 8.0 ** 2, 8.0 ** 2])
        tr = Track(tid, obs.detection.get("class_name", ""), obs.timestamp, obs.timestamp,
                   obs.world.copy(), np.zeros(3, dtype=np.float64), state, covariance,
                   obs.size.copy(), obs.yaw)
        tr.observations.append(obs)
        tr.last_match_reason = "birth"
        self.tracks[tid] = tr
        self.diagnostics["births"] += 1
        self._event("birth", tr, obs.timestamp,
                    class_name=tr.class_name,
                    world_xy=[round(float(x), 4) for x in obs.world[:2]])
        return tr

    @staticmethod
    def _endpoint_velocity(items: List[Observation], at_end: bool) -> np.ndarray:
        ordered = items[-4:] if at_end else items[:4]
        if len(ordered) < 2:
            return np.zeros(2, dtype=np.float64)
        t0 = ordered[0].timestamp
        times = np.asarray([(o.timestamp - t0) / 1e9 for o in ordered], dtype=np.float64)
        if float(np.ptp(times)) < 1e-3:
            return np.zeros(2, dtype=np.float64)
        a = np.column_stack([times, np.ones(len(times))])
        return np.asarray([np.linalg.lstsq(a, np.asarray([o.world[k] for o in ordered]), rcond=None)[0][0]
                           for k in (0, 1)], dtype=np.float64)

    def stitch_tracklets(self, frames: List[Dict[str, Any]], max_gap_sec: float = 3.0) -> Dict[str, Any]:
        """Conservatively merge dynamic fragments without generating boxes."""
        candidates = []
        values = list(self.tracks.values())
        for end in values:
            # A track still bound to a parking anchor has no finite death; its
            # identity recovery is handled by the anchor state machine.
            if end.is_static or not end.observations:
                continue
            end_obs = end.observations[-1]
            ve = self._endpoint_velocity(end.observations, at_end=True)
            for start in values:
                if start.track_id == end.track_id or not start.observations:
                    continue
                start_obs = start.observations[0]
                gap = (start_obs.timestamp - end_obs.timestamp) / 1e9
                if gap <= 0.0 or gap > max_gap_sec:
                    continue
                if not class_compatible(end.class_name, start.class_name):
                    continue
                vs = self._endpoint_velocity(start.observations, at_end=False)
                pred_fwd = end_obs.world[:2] + ve * gap
                pred_back = start_obs.world[:2] - vs * gap
                forward_error = float(np.linalg.norm(start_obs.world[:2] - pred_fwd))
                backward_error = float(np.linalg.norm(end_obs.world[:2] - pred_back))
                speed = max(float(np.linalg.norm(ve)), float(np.linalg.norm(vs)))
                gate = min(10.0, 1.2 + 1.2 * gap + 0.12 * speed * gap)
                if min(forward_error, backward_error) > gate or max(forward_error, backward_error) > gate * 1.8:
                    continue
                size_delta = float(np.linalg.norm(end.size - start.size)
                                   / max(float(np.linalg.norm(end.size)), 1.0))
                if size_delta > 0.65:
                    continue
                if np.linalg.norm(ve) > 1.0 and np.linalg.norm(vs) > 1.0:
                    cosine = float(np.dot(ve, vs) / (np.linalg.norm(ve) * np.linalg.norm(vs)))
                    if cosine < 0.25:
                        continue
                score = 0.5 * (forward_error + backward_error) / max(gate, 0.1)
                score += 0.5 * size_delta + 0.04 * gap
                candidates.append((score, end.track_id, start.track_id, gap,
                                   forward_error, backward_error))
        used_ends, used_starts = set(), set()
        remap: Dict[int, int] = {}

        def root(tid: int) -> int:
            while tid in remap:
                tid = remap[tid]
            return tid

        stitched = []
        for score, end_id, start_id, gap, ferr, berr in sorted(candidates):
            if end_id in used_ends or start_id in used_starts:
                continue
            a, b = root(end_id), root(start_id)
            if a == b:
                continue
            remap[b] = a
            used_ends.add(end_id)
            used_starts.add(start_id)
            stitched.append({"from_track_id": start_id, "to_track_id": a,
                             "gap_sec": round(gap, 3), "score": round(score, 4),
                             "forward_error": round(ferr, 3),
                             "backward_error": round(berr, 3)})
        if remap:
            for frame in frames:
                for det in frame.get("detections", []):
                    tid = det.get("track_id")
                    if tid is not None:
                        det["track_id"] = root(int(tid))
        stats = {"candidate_pairs": len(candidates), "stitched_pairs": len(stitched),
                 "pairs": stitched}
        self.diagnostics["tracklet_stitching"] = stats
        return stats

    def process(self, frames: Sequence[Dict[str, Any]], *,
                enable_stitching: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        ordered = sorted(enumerate(frames), key=lambda x: int(x[1]["frame_id"]))
        output = [copy.deepcopy(f) for f in frames]
        for f in output:
            for det in f.get("detections", []):
                det.pop("track_id", None)
        for frame_index, frame in ordered:
            observations = self._world_detections(frame, frame_index)
            self.diagnostics["frames"] += 1
            self.diagnostics["detections"] += len(observations)
            ts = int(frame["frame_id"])
            static_tracks = [t for t in self.tracks.values() if t.is_static]
            # A parked track that was observed recently participates in the
            # motion pass as well. This is what lets the original ID survive
            # the first departure frames; only a long disappearance falls back
            # to anchor-based static recovery.
            moving_tracks = [t for t in self.tracks.values()
                             if (ts - t.last_ts) / 1e9 <= self.dynamic_max_gap]
            # A departing anchored vehicle is handled by the motion pass; a
            # parked anchor is deliberately considered only after moving tracks
            # have had first claim on the detection.
            moving_matches, used = self._associate(moving_tracks, observations)
            assigned: Dict[int, Observation] = {}
            for ti, oi in moving_matches.items():
                tr = moving_tracks[ti]
                assigned[tr.track_id] = observations[oi]
            remaining = [o for i, o in enumerate(observations) if i not in used]
            static_candidates = [t for t in static_tracks if t.track_id not in assigned]
            static_matches, static_used = self._associate(static_candidates, remaining, static_mode=True)
            for ti, oi in static_matches.items():
                tr = static_candidates[ti]
                assigned[tr.track_id] = remaining[oi]
                self.diagnostics["static_recoveries"] += 1
            used_ids = set(assigned)
            used_obs = set(id(o) for o in assigned.values())
            for tid, obs in assigned.items():
                tr = self.tracks[tid]
                self._update(tr, obs, "static_recover" if tr.is_static else "motion")
                obs.detection["track_id"] = tid
                self.diagnostics["matches"] += 1
                self._event("match", tr, obs.timestamp, reason=tr.last_match_reason)
            # Every remaining detection becomes a new identity.  No boxes are
            # synthesized for unmatched/lost tracks.
            for obs in observations:
                if id(obs) in used_obs:
                    continue
                tr = self._birth(obs)
                obs.detection["track_id"] = tr.track_id
            # Static tracks are intentionally retained through arbitrary
            # missing intervals. Dynamic tracks expire only by time.
            for tr in self.tracks.values():
                if tr.last_ts != ts:
                    tr.missed_sec = max(0.0, (ts - tr.last_ts) / 1e9)
        # Copy assigned IDs back to the original frame order by detection
        # object position.  Raw geometry and all other fields remain untouched.
        for out_frame, src_frame in zip(output, frames):
            for out_det, src_det in zip(out_frame.get("detections", []), src_frame.get("detections", [])):
                if "track_id" in src_det:
                    out_det["track_id"] = src_det["track_id"]
        self.diagnostics["tracks_total"] = len(self.tracks)
        self.diagnostics["static_tracks"] = sum(1 for t in self.tracks.values() if t.is_static)
        if enable_stitching:
            self.stitch_tracklets(output)
        else:
            self.diagnostics["tracklet_stitching"] = {
                "disabled": "caller_uses_conservative_scene_aware_stitching"
            }
        return output, self.diagnostics


def apply_post_filters(frames: List[Dict[str, Any]], min_lifecycle: int = 4) -> Dict[str, Any]:
    """Apply only the agreed hard filter after association.

    ``<= min_lifecycle`` means number of distinct observed frames, not number
    of detections. No interpolation, box smoothing, or geometry filtering is
    performed here; those are intentionally separate future policies.
    """
    by_id: Dict[int, set] = {}
    for frame in frames:
        for det in frame.get("detections", []):
            tid = det.get("track_id")
            if tid is not None:
                by_id.setdefault(int(tid), set()).add(int(frame["frame_id"]))
    dropped = {tid for tid, ids in by_id.items() if len(ids) <= int(min_lifecycle)}
    removed = 0
    for frame in frames:
        old = frame.get("detections", [])
        frame["detections"] = [d for d in old if d.get("track_id") not in dropped]
        removed += len(old) - len(frame["detections"])
        frame["num_detections"] = len(frame["detections"])
    return {"min_lifecycle": int(min_lifecycle), "tracks_before": len(by_id),
            "tracks_dropped": len(dropped), "boxes_removed": removed,
            "dropped_track_ids": sorted(dropped)}


def box_to_label(det: Dict[str, Any]) -> Dict[str, Any]:
    box = det["box_lidar"]
    cls = CLASS_MAP.get(det.get("class_name", ""), det.get("class_name", ""))
    label = {
        "obj_id": str(det["track_id"]), "obj_type": cls,
        "score": round(float(det.get("score", 0.0)), 4),
        "psr": {"position": {"x": round(float(box[0]), 4), "y": round(float(box[1]), 4), "z": round(float(box[2]), 4)},
                "rotation": {"x": 0.0, "y": 0.0, "z": round(float(box[6]), 4)},
                "scale": {"x": round(float(box[3]), 4), "y": round(float(box[4]), 4), "z": round(float(box[5]), 4)}},
    }
    if "visibility" in det:
        label["visibility"] = copy.deepcopy(det["visibility"])
    return label


def export_clip(frames: List[Dict[str, Any]], source_clip: Path, out_clip: Path) -> int:
    if out_clip.exists():
        shutil.rmtree(out_clip)
    shutil.copytree(source_clip, out_clip, ignore=shutil.ignore_patterns("label"))
    label_dir = out_clip / "label"
    label_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for frame in frames:
        labels = [box_to_label(d) for d in frame.get("detections", []) if d.get("track_id") is not None]
        n += len(labels)
        (label_dir / f"{frame['frame_id']}.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return n


def run(in_json: Path, clip: Path, out_json: Path, out_clip: Optional[Path], diagnostics_path: Path,
        min_lifecycle: int = 4) -> Dict[str, Any]:
    frames = json.loads(in_json.read_text(encoding="utf-8"))
    if not isinstance(frames, list):
        raise ValueError(f"input must be a list of frames: {in_json}")
    coords = CoordinateProvider(clip)
    tracker = ConservativeTracker(coords)
    tracked, diag = tracker.process(frames)
    filter_stats = apply_post_filters(tracked, min_lifecycle=min_lifecycle)
    diag["post_filter"] = filter_stats
    diag["final_detections"] = sum(len(f.get("detections", [])) for f in tracked)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(tracked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    labels = None
    if out_clip is not None:
        labels = export_clip(tracked, clip, out_clip)
        diag["sust_labels"] = labels
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diag, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return diag


def main() -> None:
    ap = argparse.ArgumentParser(description="Conservative clip-local BEV tracker")
    ap.add_argument("--in-json", "--in_json", dest="in_json", required=True, type=Path)
    ap.add_argument("--clip", required=True, type=Path)
    ap.add_argument("--out-json", "--out_json", dest="out_json", required=True, type=Path)
    ap.add_argument("--out-clip", "--out_clip", dest="out_clip", type=Path)
    ap.add_argument("--diagnostics", type=Path)
    ap.add_argument("--min-lifecycle", type=int, default=4,
                    help="drop tracks observed in <= this many frames")
    ap.add_argument("--score-thresh", "--score_thresh", type=float, default=None,
                    help="compatibility option; input is already inference-thresholded")
    args = ap.parse_args()
    diag_path = args.diagnostics or args.out_json.with_name(args.out_json.stem + "_diagnostics.json")
    diag = run(args.in_json, args.clip, args.out_json, args.out_clip, diag_path, args.min_lifecycle)
    print(json.dumps({k: diag[k] for k in ("frames", "detections", "tracks_total", "matches", "births", "static_locks", "static_recoveries", "departures", "final_detections") if k in diag}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
