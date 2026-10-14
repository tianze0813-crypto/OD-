#!/usr/bin/env python3
"""多雷达点云融合：lidar_front / lidar_rear -> lidar_top 系（2026-10-10）。

背景：VoxelNeXt / VRU 链原来只用 lidar_top 作为输入，而 top 雷达装在车顶，
近处（车头/车尾附近）是盲区。前/后雷达能看到这些近场区域。

做法：按 transforms/calib.json 的 tf2base_link 静态外参，把前/后雷达点云变换到
top 雷达坐标系后与 top 点云拼接：

    p_top = inv(base_from_lidar_top) @ base_from_lidar_src @ p_src

每帧以 **top 的时间戳** 为基准，前/后雷达取时间最近的一帧（默认容差 50ms）。
输出的融合 bin 用 top 时间戳命名，因此下游 frame_id / 位姿查询完全不变。

作为公共预处理放在推理最前面。默认做**自车运动补偿**：前/后雷达帧时间与 top 帧
差几毫秒到几十毫秒，用 pose_data 把点云统一到 top 帧时刻的车体系，避免运动错位：

    M = inv(W(T)) @ W(t_X) @ inv(base_from_lidar_top) @ base_from_lidar_src
    W(t) = world_from_lidar_top(t)

（无 pose_data 时自动退回静态外参。）
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

REFERENCE = "top"
DEFAULT_INCLUDE = ("front", "rear")
DEFAULT_MAX_GAP_MS = 50.0
DEFAULT_COMPENSATE_EGO = True


def _load_extrinsics(clip: Path) -> dict:
    calib = json.loads((Path(clip) / "transforms" / "calib.json").read_text(encoding="utf-8"))
    tf = calib.get("tf2base_link")
    if not tf:
        raise ValueError(f"calib.json 缺少 tf2base_link: {clip}")
    out = {}
    for name in ("front", "rear", "top"):
        key = f"lidar_{name}"
        if key in tf:
            out[name] = np.asarray(tf[key], dtype=np.float64)
    if REFERENCE not in out:
        raise ValueError(f"calib.json 缺少 tf2base_link.{REFERENCE}: {clip}")
    return out


def _read_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    columns = 4 if raw.size % 4 == 0 else 5
    points = raw.reshape(-1, columns)
    if columns > 4:
        points = points[:, :4]
    return points.astype(np.float32)


def _frame_timestamps(clip: Path, sensor: str) -> List[int]:
    directory = Path(clip) / "lidar" / f"lidar_{sensor}"
    if not directory.is_dir():
        return []
    stamps = []
    for entry in directory.glob("*.bin"):
        try:
            stamps.append(int(entry.stem))
        except ValueError:
            continue
    return sorted(stamps)


def _nearest(stamps: Sequence[int], target: int, tol_ns: int) -> int | None:
    if not stamps:
        return None
    index = bisect.bisect_left(stamps, target)
    best = None
    best_delta = None
    for candidate in (index - 1, index):
        if 0 <= candidate < len(stamps):
            delta = abs(stamps[candidate] - target)
            if best_delta is None or delta < best_delta:
                best, best_delta = stamps[candidate], delta
    if best is None or best_delta is None or best_delta > tol_ns:
        return None
    return best


def _coordinate_provider(clip: Path):
    """Lazy import so the module stays usable without the tracking stack."""
    from tracking.tracker_conservative import CoordinateProvider
    return CoordinateProvider(Path(clip))


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    xyz = points[:, :3].astype(np.float64)
    homo = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    moved = (matrix @ homo.T).T[:, :3]
    out = points.astype(np.float32).copy()
    out[:, :3] = moved.astype(np.float32)
    return out


def fuse_frame(clip: Path, timestamp: int, *,
               include: Iterable[str] = DEFAULT_INCLUDE,
               max_gap_ms: float = DEFAULT_MAX_GAP_MS,
               extrinsics: dict | None = None,
               timestamps: dict | None = None,
               compensate_ego: bool = DEFAULT_COMPENSATE_EGO,
               coords=None) -> Tuple[np.ndarray, dict]:
    """Fuse one reference(top) frame; returns (points Nx4, stats)."""
    clip = Path(clip)
    if extrinsics is None:
        extrinsics = _load_extrinsics(clip)
    if timestamps is None:
        timestamps = {name: _frame_timestamps(clip, name)
                      for name in ("front", "rear", "top")}
    tol_ns = int(max_gap_ms * 1e6)
    base_ref = extrinsics[REFERENCE]
    ref_points = _read_bin(clip / "lidar" / "lidar_top" / f"{timestamp}.bin")
    pieces = [ref_points]
    stats = {"reference_points": int(len(ref_points)), "attached": {},
             "compensate_ego": bool(compensate_ego)}
    world_ref = None
    if compensate_ego:
        try:
            if coords is None:
                coords = _coordinate_provider(clip)
            world_ref = coords.world_from_lidar(int(timestamp))
        except Exception as error:      # 缺少 pose/标定时退回静态外参
            stats["compensate_error"] = str(error)
            world_ref = None
    for sensor in include:
        if sensor == REFERENCE or sensor not in extrinsics:
            continue
        stamp = _nearest(timestamps.get(sensor, []), timestamp, tol_ns)
        if stamp is None:
            stats["attached"][sensor] = {"frame": None, "points": 0,
                                         "delta_ms": None, "compensated": False}
            continue
        points = _read_bin(clip / "lidar" / f"lidar_{sensor}" / f"{stamp}.bin")
        matrix = np.linalg.inv(base_ref) @ extrinsics[sensor]
        compensated = False
        if world_ref is not None:
            world_x = coords.world_from_lidar(int(stamp))
            if world_x is not None:
                matrix = np.linalg.inv(world_ref) @ world_x @ matrix
                compensated = True
        pieces.append(_transform(points, matrix))
        stats["attached"][sensor] = {
            "frame": int(stamp),
            "points": int(len(points)),
            "delta_ms": round((stamp - int(timestamp)) / 1e6, 2),
            "compensated": compensated,
        }
    fused = np.concatenate(pieces, axis=0) if pieces else np.zeros((0, 4), np.float32)
    stats["fused_points"] = int(len(fused))
    return fused, stats


def fuse_clip(clip: Path, out_dir: Path, *,
              include: Iterable[str] = DEFAULT_INCLUDE,
              max_gap_ms: float = DEFAULT_MAX_GAP_MS,
              reference_sensor: str = REFERENCE,
              compensate_ego: bool = DEFAULT_COMPENSATE_EGO) -> dict:
    """Materialise fused bins (named by top timestamps) into ``out_dir``."""
    clip = Path(clip)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extrinsics = _load_extrinsics(clip)
    timestamps = {name: _frame_timestamps(clip, name)
                  for name in ("front", "rear", "top")}
    ref_stamps = timestamps.get(reference_sensor, [])
    if not ref_stamps:
        raise ValueError(f"clip 没有 lidar_{reference_sensor} 帧: {clip}")
    coords = None
    if compensate_ego:
        try:
            coords = _coordinate_provider(clip)
        except Exception:
            coords = None
    total = 0
    attached_total = 0
    compensated_frames = 0
    for stamp in ref_stamps:
        fused, stats = fuse_frame(clip, stamp, include=include,
                                  max_gap_ms=max_gap_ms,
                                  extrinsics=extrinsics, timestamps=timestamps,
                                  compensate_ego=compensate_ego, coords=coords)
        fused.tofile(str(out_dir / f"{stamp}.bin"))
        total += len(fused)
        attached_total += sum(v["points"] for v in stats["attached"].values())
        if any(v.get("compensated") for v in stats["attached"].values()):
            compensated_frames += 1
    return {
        "pipeline": "lidar_fusion",
        "clip": str(clip),
        "out_dir": str(out_dir),
        "reference_sensor": reference_sensor,
        "include": list(include),
        "max_gap_ms": float(max_gap_ms),
        "compensate_ego": bool(compensate_ego),
        "compensated_frames": compensated_frames,
        "frames": len(ref_stamps),
        "fused_points": total,
        "attached_points": attached_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--include", default="front,rear")
    parser.add_argument("--max-gap-ms", type=float, default=DEFAULT_MAX_GAP_MS)
    parser.add_argument("--no-compensate-ego", action="store_true",
                        help="关闭自车运动补偿（退回静态外参）")
    args = parser.parse_args()
    include = tuple(name.strip() for name in args.include.split(",") if name.strip())
    stats = fuse_clip(args.clip, args.out_dir, include=include,
                      max_gap_ms=args.max_gap_ms,
                      compensate_ego=not args.no_compensate_ego)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
