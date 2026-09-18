"""世界系下的"一排行人"过滤（【改动】新增）。

动机：行人检测在近距离相似特征上会频繁误检，表现为同一帧内一排几乎共线的行人。
规则：世界坐标系下，同一帧内若存在 >= min_row 个行人共线（到直线的垂距 <= tolerance），
      则这一排全部删除。逐帧迭代，直到该帧不再有满足条件的排。

实测（8 条 SUST clip / 1,355 个行人）：min_row=6, tolerance=1.0m 时
  只删除 clip3(69个/18.5%) 与 clip5(24个/6.8%)，其余 6 条 clip 删除数为 0。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np

from tracking import tracker_conservative as tracking


def _largest_row(points: np.ndarray, min_row: int, tolerance: float,
                 max_span: float) -> List[int]:
    """返回最大共线子集的索引；不足 min_row 返回空。"""
    count = len(points)
    if count < min_row:
        return []
    best: List[int] = []
    for i in range(count):
        for j in range(i + 1, count):
            direction = points[j] - points[i]
            length = float(np.linalg.norm(direction))
            if length < 1e-6 or length > max_span:
                continue
            unit = direction / length
            relative = points - points[i]
            along = relative @ unit
            perpendicular = np.abs(relative[:, 0] * unit[1] - relative[:, 1] * unit[0])
            on_line = ((perpendicular <= tolerance)
                       & (along >= -1.0) & (along <= length + 1.0))
            index = np.where(on_line)[0]
            if len(index) >= min_row and len(index) > len(best):
                best = [int(k) for k in index]
    return best


def drop_pedestrian_rows(frames: List[Dict[str, Any]],
                         coords: tracking.CoordinateProvider, *,
                         min_row: int = 6, tolerance: float = 1.0,
                         max_span: float = 60.0,
                         pedestrian_class: str = "Pedestrian",
                         box_frame: str = "base_link") -> Dict[str, Any]:
    # 【改动】box_frame="base_link"：框是 base_link（早期阶段用不到）；
    #          box_frame="lidar_top" ：框还在检测器局部系（链路后段，base_link 转换之前）
    if box_frame == "lidar_top" or getattr(coords, "base_from_lidar_top", None) is None:
        base_to_lidar = np.eye(4)
    else:
        base_to_lidar = np.linalg.inv(coords.base_from_lidar_top)
    before = sum(len(f.get("detections", [])) for f in frames)
    removed = 0
    frames_with_rows = 0
    row_sizes: List[int] = []
    for frame in frames:
        timestamp = int(frame["frame_id"])
        world_from_lidar = coords.world_from_lidar(timestamp)
        if world_from_lidar is None:
            continue
        world_from_base = world_from_lidar @ base_to_lidar
        detections = frame.get("detections", [])
        rows = []
        for index, det in enumerate(detections):
            if tracking.canonical_class_name(det.get("class_name", "")) != pedestrian_class:
                continue
            box = det.get("box_lidar")
            if not box:
                continue
            point = world_from_base @ np.array([float(box[0]), float(box[1]),
                                                float(box[2]), 1.0], dtype=np.float64)
            rows.append((index, point[:2]))
        if len(rows) < min_row:
            continue
        kept = [i for i, _ in rows]
        points = np.asarray([p for _, p in rows], dtype=np.float64)
        drop: set = set()
        while True:
            index = _largest_row(points, min_row, tolerance, max_span)
            if not index:
                break
            row_sizes.append(len(index))
            for k in index:
                drop.add(rows[k][0])
            alive = [k for k in range(len(rows)) if k not in index]
            if len(alive) < min_row:
                break
            rows = [rows[k] for k in alive]
            points = np.asarray([p for _, p in rows], dtype=np.float64)
        if drop:
            frames_with_rows += 1
            removed += len(drop)
            frame["detections"] = [d for i, d in enumerate(detections) if i not in drop]
            frame["num_detections"] = len(frame["detections"])
    return {
        "detections_before": before,
        "detections_after": before - removed,
        "detections_removed": removed,
        "frames_with_rows": frames_with_rows,
        "row_sizes": row_sizes,
        "min_row": int(min_row),
        "tolerance_m": float(tolerance),
    }
