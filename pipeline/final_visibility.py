"""公共收尾阶段：合并标签之后的遮挡属性计算 + 可见度过滤（【改动】2026-10-14）。

背景：遮挡属性原来在每条链各自的 hard filter 里算，帧内只有本链的框，
所以 VRU 链永远看不到"行人被车挡住"（帧内没有 Car/Truck 框），
可见度几乎恒为 1.0。现在把这一步搬到 car/truck/VRU 标签**合并之后**、
写 ``label/det/`` 之前统一算一次：

  * 参与判定的框 = 该帧最终合并标签里的**全部类别**（跨类遮挡才成立）；
  * 遮挡属性 ``visibility`` 对所有框重算并覆盖；
  * 可见度过滤**默认关闭**（``min_ratio=0`` = 只算属性不删框）；
    传正数阈值（例 0.05）才会把 ``ratio <= 该值`` 的框删掉；
  * 点云高度比补判（``use_point_occlusion``）保留 —— 点云优先用 VRU 推理时的
    front/rear 融合点云（补 top 近场盲区），没有则退回 ``lidar/lidar_top``。

坐标系与 moga_data-main/steps/visualize.py 的投影可视化完全一致（权威口径）：
公共参考系是 **base_link**；框取自合并后 label 的 ``psr``（本来就是 base_link），
点云按 ``tf2base_link["lidar_top"]`` 变到 base_link，相机用 ``inv(base_from_cam)``。
详见 ``filtering/camera_visibility.py`` 的模块 docstring。
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from filtering import camera_visibility


def label_to_box(label: Mapping[str, Any]) -> Optional[List[float]]:
    """SUST label（base_link）-> [x, y, z, dx, dy, dz, yaw]；字段缺失返回 None。"""
    try:
        psr = label["psr"]
        position = psr["position"]
        scale = psr["scale"]
        rotation = psr.get("rotation", {})
        box = [
            float(position["x"]), float(position["y"]), float(position["z"]),
            float(scale["x"]), float(scale["y"]), float(scale["z"]),
            float(rotation.get("z", 0.0)),
        ]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in box):
        return None
    if any(value <= 0.0 for value in box[3:6]):
        return None
    return box


def apply_final_visibility(
        merged: Sequence[Mapping[str, Any]],
        clip: Path,
        *,
        points_dir: Optional[Path] = None,
        min_ratio: float = 0.0,
        occl_tol: float = 0.3,
        use_point_occlusion: bool = True,
        point_occlusion_trigger: float = 0.01,
        point_occlusion_full_vis_ratio: float = 0.9,
) -> Dict[str, Any]:
    """就地重算合并标签的遮挡属性，并按可见度删除框。

    ``merged``：``[{"frame_id": ..., "labels": [...]}, ...]``，label 已在 base_link。
    ``points_dir``：融合点云目录（可选）；缺省用 ``<clip>/lidar/lidar_top``。
    ``min_ratio <= 0``（默认 0）= **只算属性、不删框**；给正数才过滤
    （沿用旧的 drop_below <= 0 = 不过滤语义）。
    返回统计（只用于日志/测试，不落盘）。
    """
    clip = Path(clip)
    cams = camera_visibility.load_clip_cameras(clip)
    if not cams:
        raise ValueError(f"no configured cameras found in {clip / 'transforms/calib.json'}")

    drop_enabled = float(min_ratio) > 0.0
    stats: Dict[str, Any] = {
        "policy": {
            "frame": camera_visibility.DETECTION_FRAME,
            "visibility_drop_at_or_below": float(min_ratio),
            "drop_enabled": drop_enabled,
            "occlusion_tolerance": float(occl_tol),
            "use_point_occlusion": bool(use_point_occlusion),
            "points_dir": str(points_dir) if points_dir is not None else None,
        },
        "cameras": len(cams),
        "frames": 0,
        "boxes_before": 0,
        "boxes_after": 0,
        "boxes_removed": 0,
        "tag1": 0,
        "tag2": 0,
        "removed_by_class": {},
    }
    for frame in merged:
        labels = frame.get("labels") or []
        stats["boxes_before"] += len(labels)
        if not labels:
            continue
        frame_id = str(frame.get("frame_id", ""))
        dets: List[Dict[str, Any]] = []
        passthrough: List[Mapping[str, Any]] = []
        for label in labels:
            box = label_to_box(label)
            if box is None:
                passthrough.append(label)      # 字段不全：不动它，也不参与遮挡
                continue
            dets.append({"box_lidar": box, "label": label})
        if dets:
            points = camera_visibility.load_base_link_points(
                clip, frame_id, points_dir) if use_point_occlusion else None
            camera_visibility.compute_frame_visibility(
                dets, cams, occl_tol,
                points=points,
                use_point_occlusion=use_point_occlusion,
                point_occlusion_trigger=point_occlusion_trigger,
                full_vis_ratio=point_occlusion_full_vis_ratio)
        kept: List[Any] = []
        for det in dets:
            visibility = det.get("visibility") or {}
            ratio = float(visibility.get("ratio", 0.0))
            stats["tag1" if visibility.get("tag") == 1 else "tag2"] += 1
            if drop_enabled and ratio <= float(min_ratio):
                stats["boxes_removed"] += 1
                class_name = str(det["label"].get("obj_type", ""))
                removed = stats["removed_by_class"]
                removed[class_name] = removed.get(class_name, 0) + 1
                continue
            label = copy.deepcopy(dict(det["label"]))
            label["visibility"] = visibility
            kept.append(label)
        kept.extend(copy.deepcopy(dict(label)) for label in passthrough)
        frame["labels"] = kept
        stats["boxes_after"] += len(kept)
        stats["frames"] += 1
    stats["removed_by_class"] = dict(sorted(stats["removed_by_class"].items()))
    return stats
