#!/usr/bin/env python3
"""相机可见度计算（融合标注用）。

把每个 3D box 投影到全部相机，用深度排序判断遮挡，得到可见度比例
（同时扣截断和遮挡），并映射为 tag：
    tag 1 = 遮挡 < 50%（可见度 > 50%），正常可见；
    tag 2 = 遮挡 >= 50%（可见度 <= 50%），重度遮挡/基本不可见。

原理：lidar 只在扫得到的面上有点，不能用“目标自己的点数”衡量遮挡；
改为把 box 的 8 个角点按鱼眼模型（KANNALA_BRANDT）投影到每路相机，
取投影凸包的像素面积，扣掉被“深度更近的其他 box”盖住的面积和出界
面积，剩余比例即该相机下的可见度；多相机取最大。

注意：tag（1/2，50% 遮挡为界）只是标注侧的描述字段，不是过滤条件；
过滤条件是**被遮挡 >= 95%**（可见度 <= 5%）的 box 直接删除（drop_vis_below
默认 0.05）。本仓库里可见度只在**最终公共阶段**（pipeline/final_visibility.py）
算一次，作用在 car/truck/VRU 合并之后的标签上。

坐标系：公共参考系 = **base_link**，与 moga_data-main/steps/visualize.py 的
投影可视化保持同一口径（该脚本 _load_merged_lidar_points 把每个雷达按静态
外参变换到 base_link 再合并，_draw_lidar_on_image 用 base2imgs = base_link->cam
投影）。calib.json 的 tf2base_link[X] 是 base_from_X，且 base_link 在变换图里
是单位阵，因此：

    cam_from_base = inv(base_from_cam)
    base_from_top  = tf2base_link["lidar_top"]

输入 box（base_link 系）与点云（先按 base_from_top 变到 base_link）必须同系，
逐帧静态，不需要 pose_data.txt。
"""

import json
import math
from pathlib import Path

import numpy as np
import cv2


CAMERA_NAMES = ["cam_front", "cam_rear", "cam_left", "cam_right", "cam_x8d"]


# 公共参考系。viewer 把点云/检测框统一到 base_link 再投影，见模块 docstring。
DETECTION_FRAME = "base_link"


def load_clip_cameras(clip_root, detection_frame: str = DETECTION_FRAME):
    """返回 {cam: {K, D, w, h, T}}，``T`` = cam_from_detection_frame。

    ``detection_frame`` 默认 ``"base_link"``（= 变换图里的单位阵，也就是
    tf2base_link 的基系）。历史实现用的是 ``"pose"``，那套 calib 里
    ``pose`` 与前雷达外参完全相同，只适用于标在 lidar_front 系的标签。
    """
    clip_root = Path(clip_root)
    calib = json.loads((clip_root / "transforms" / "calib.json").read_text(encoding="utf-8"))
    tf = calib["tf2base_link"]
    if detection_frame == "base_link":
        base_from_detection = np.eye(4, dtype=np.float64)
    elif detection_frame in tf:
        base_from_detection = np.asarray(tf[detection_frame], dtype=np.float64)
    else:
        raise ValueError(
            f"calib.json 的 tf2base_link 缺少检测系 {detection_frame!r}"
            f"（可用：base_link / {sorted(tf)}）")
    cams = {}
    for name in CAMERA_NAMES:
        if name not in calib or name not in tf:
            continue
        c = calib[name]
        cams[name] = {
            "K": np.asarray(c["K"], dtype=np.float64),
            "D": np.asarray(c["D"], dtype=np.float64),
            "w": int(c["imgw"]),
            "h": int(c["imgh"]),
            # cam_from_detection_frame
            "T": np.linalg.inv(np.asarray(tf[name], dtype=np.float64)) @ base_from_detection,
        }
    return cams


def project_lidar_points(points, cam):
    """lidar 系 Nx3 点 -> 相机像素 (u, v, valid)。KANNALA_BRANDT 鱼眼。"""
    p = points @ cam["T"][:3, :3].T + cam["T"][:3, 3]
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    valid = z > 0.1
    r = np.hypot(x, y)
    th = np.arctan2(r, np.maximum(z, 1e-6))
    D = cam["D"]
    thd = th * (1.0 + D[0] * th ** 2 + D[1] * th ** 4 + D[2] * th ** 6 + D[3] * th ** 8)
    xr = np.divide(x, r, out=np.zeros_like(r), where=r > 1e-6)
    yr = np.divide(y, r, out=np.zeros_like(r), where=r > 1e-6)
    K = cam["K"]
    u = K[0, 0] * thd * xr + K[0, 2]
    v = K[1, 1] * thd * yr + K[1, 2]
    return u, v, valid


def box_corners_lidar(box):
    """box_lidar [x,y,z,dx,dy,dz,yaw] -> 8 角点 (lidar 系)。"""
    x, y, z, dx, dy, dz, yaw = box
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    local = np.array([
        [hx, hy, hz], [hx, hy, -hz], [hx, -hy, hz], [hx, -hy, -hz],
        [-hx, hy, hz], [-hx, hy, -hz], [-hx, -hy, hz], [-hx, -hy, -hz],
    ], dtype=np.float64)
    corners = np.empty_like(local)
    corners[:, 0] = x + local[:, 0] * c - local[:, 1] * s
    corners[:, 1] = y + local[:, 0] * s + local[:, 1] * c
    corners[:, 2] = z + local[:, 2]
    return corners


def _ratio_in_camera(hull, idx, hulls, depths, cam, occl_tol):
    """单相机内可见比例：总投影面积 - 出界 - 被更近 box 遮挡。"""
    W, H = cam["w"], cam["h"]
    hull = hull.astype(np.int32)
    u_min, v_min = int(hull[:, 0].min()), int(hull[:, 1].min())
    u_max, v_max = int(hull[:, 0].max()), int(hull[:, 1].max())
    cw = max(u_max - u_min + 1, 1)
    ch = max(v_max - v_min + 1, 1)
    off = np.array([u_min, v_min], dtype=np.int32)
    total_mask = np.zeros((ch, cw), dtype=np.uint8)
    cv2.fillPoly(total_mask, [hull - off], 1)
    total = int(total_mask.sum())
    if total == 0:
        return 0.0, 0.0, 1.0

    clip = hull.copy()
    clip[:, 0] = np.clip(clip[:, 0], 0, W - 1)
    clip[:, 1] = np.clip(clip[:, 1], 0, H - 1)
    inimg = np.zeros((ch, cw), dtype=np.uint8)
    cv2.fillPoly(inimg, [clip - off], 1)

    occ = np.zeros((ch, cw), dtype=np.uint8)
    d_self = depths[idx]
    for j, (hj, dj) in enumerate(zip(hulls, depths)):
        if j == idx or hj is None or dj > d_self - occl_tol:
            continue
        poly = hj.astype(np.int32)
        pu_min, pv_min = int(poly[:, 0].min()), int(poly[:, 1].min())
        pu_max, pv_max = int(poly[:, 0].max()), int(poly[:, 1].max())
        if pu_max < u_min or pu_min > u_max or pv_max < v_min or pv_min > v_max:
            continue  # bbox 不相交，快速跳过
        cv2.fillPoly(occ, [poly - off], 1)

    visible = int(((inimg > 0) & (occ == 0)).sum())
    occluded = int(((inimg > 0) & (occ > 0)).sum())
    truncated = total - int(inimg.sum())
    return visible / total, occluded / total, truncated / total


def base_from_top(clip_root) -> np.ndarray:
    """``base_from_lidar_top`` from calib.json (points are exported in the top frame)."""
    calib = json.loads((Path(clip_root) / "transforms" / "calib.json")
                       .read_text(encoding="utf-8"))
    tf = calib.get("tf2base_link") or {}
    if "lidar_top" not in tf:
        raise ValueError(
            f"calib.json 的 tf2base_link 缺少 lidar_top：{Path(clip_root) / 'transforms' / 'calib.json'}")
    return np.asarray(tf["lidar_top"], dtype=np.float64)


def load_base_link_points(clip_root, frame_id, points_dir=None) -> np.ndarray:
    """读一帧点云（top 系）并变到 base_link 系，返回 Nx3。

    ``points_dir`` 给融合点云目录（<work>/lidar_fused/<clip>，同样是 top 系、
    按 top 时间戳命名）时优先用它；否则退回 ``<clip>/lidar/lidar_top``。
    与 viewer 一致：点云先按静态外参变到 base_link，再参与投影/遮挡判断。
    """
    path = None
    if points_dir is not None:
        candidate = Path(points_dir) / f"{frame_id}.bin"
        if candidate.is_file():
            path = candidate
    if path is None:
        path = Path(clip_root) / "lidar" / "lidar_top" / f"{frame_id}.bin"
    if not path.is_file():
        raise FileNotFoundError(f"lidar frame not found: {path}")
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4 != 0:
        raise ValueError(f"lidar frame is not xyzi float32: {path}")
    points = values.reshape(-1, 4)[:, :3].astype(np.float64)
    transform = base_from_top(clip_root)
    return points @ transform[:3, :3].T + transform[:3, 3]


def _points_in_box(points, box) -> np.ndarray:
    """Return the XYZ points inside the box volume (box_lidar local frame)."""
    x, y, z, dx, dy, dz, yaw = (float(v) for v in box[:7])
    c, s = math.cos(-yaw), math.sin(-yaw)
    px = (points[:, 0] - x) * c - (points[:, 1] - y) * s
    py = (points[:, 0] - x) * s + (points[:, 1] - y) * c
    mask = (
        (np.abs(px) <= dx / 2.0)
        & (np.abs(py) <= dy / 2.0)
        & (np.abs(points[:, 2] - z) <= dz / 2.0)
    )
    return points[mask]


def point_occlusion_from_pc(points_in_box, box) -> float:
    """Occlusion degree (0..1) from the point-cloud height ratio.

    For the nearest layer (no nearer box in the image) the camera box-overlap
    method reports ``occluded=0`` even when the target is partly hidden by
    non-box geometry.  This uses the LiDAR points inside the box: the vertical
    span of the measured points over the box height (dz).  A fully visible
    surface spans nearly the whole height; a partly hidden one leaves a gap.
    """
    dz = float(box[5])
    if points_in_box is None or points_in_box.shape[0] == 0:
        return 1.0
    if dz <= 0.0:
        return 0.0
    zs = points_in_box[:, 2]
    span = float(np.max(zs) - np.min(zs))
    height_ratio = float(np.clip(span / dz, 0.0, 1.0))
    return 1.0 - height_ratio


def compute_frame_visibility(dets, cams, occl_tol=0.3,
                             points=None, use_point_occlusion=True,
                             point_occlusion_trigger=0.01,
                             full_vis_ratio=0.9):
    """为帧内每个 det 写回 det['visibility']，返回统计。

    ``dets`` 的 ``box_lidar`` 与 ``points`` 必须都在 ``cams`` 的检测系里
    （默认 base_link）。帧内**所有类别**要一起传进来，跨类遮挡才判得出来。
    """
    boxes = [d["box_lidar"] for d in dets]
    n = len(boxes)
    cam_data = {}
    for name, cam in cams.items():
        hulls = [None] * n
        depths = np.full(n, 1e18)
        for i, b in enumerate(boxes):
            u, v, ok = project_lidar_points(box_corners_lidar(b), cam)
            if not bool(ok.all()):
                continue
            pts = np.column_stack([u[ok], v[ok]]).astype(np.float32)
            if len(pts) < 3:
                continue
            hulls[i] = cv2.convexHull(pts).reshape(-1, 2)
            depths[i] = float(
                (np.array([b[0], b[1], b[2]]) @ cam["T"][:3, :3].T + cam["T"][:3, 3])[2]
            )
        cam_data[name] = (hulls, depths, cam)

    stats = {"checked": 0, "tag1": 0, "tag2": 0}
    for i, d in enumerate(dets):
        best = None
        for name in cams:
            hulls, depths, cam = cam_data[name]
            hull = hulls[i]
            if hull is None:
                continue
            ratio, occ, trunc = _ratio_in_camera(hull, i, hulls, depths, cam, occl_tol)
            if best is None or ratio > best[0]:
                best = (ratio, occ, trunc, name)
        # Nearest layer: no nearer box occludes it in the image, so the camera
        # method reports a fully visible target.  Complement it with the LiDAR
        # point-cloud height ratio to catch occlusion by non-box objects.
        if (best is not None and use_point_occlusion
                and points is not None
                and float(best[1]) <= point_occlusion_trigger):
            box = d["box_lidar"]
            ratio, occ, trunc, name = best
            visible_ratio = 1.0 - point_occlusion_from_pc(
                _points_in_box(points, box), box)
            if visible_ratio > full_vis_ratio:
                # 可见度高于阈值视为完全可见（顶部/底部小空隙不算遮挡）。
                occ = 0.0
                ratio = max(0.0, 1.0 - float(trunc))
            else:
                occ = max(float(occ), 1.0 - visible_ratio)
                ratio = max(0.0, 1.0 - float(trunc) - occ)
            # 【修·2026-10-14】原实现把上面算好的值又用 `= best` 覆盖掉了，
            # 点云高度比补判（use_point_occlusion）从来没生效过。写回 best 才用得上。
            best = (ratio, occ, trunc, name)
        if best is None:
            vis = {"tag": 2, "ratio": 0.0, "occluded": 0.0, "truncated": 1.0, "best_cam": None}
        else:
            ratio, occ, trunc, name = best
            vis = {
                "tag": 1 if ratio >= 0.5 else 2,
                "ratio": round(float(ratio), 4),
                "occluded": round(float(occ), 4),
                "truncated": round(float(trunc), 4),
                "best_cam": name,
            }
        d["visibility"] = vis
        stats["checked"] += 1
        stats["tag1" if vis["tag"] == 1 else "tag2"] += 1
    return stats
