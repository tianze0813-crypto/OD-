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
默认 0.05）。过滤建议在模型推理后、跟踪/后处理前执行，避免不可见目标
参与跟踪制造 ID 碎片。

坐标系：输入点云和 box 使用统一的 pose 局部帧。calib.json 的 tf2base_link 给的是
base_from_sensor，因此 cam_from_local = inv(base_from_cam) @ base_from_pose，
逐帧静态，不需要 pose_data.txt。
"""

import json
import math
from pathlib import Path

import numpy as np
import cv2


CAMERA_NAMES = ["cam_front", "cam_rear", "cam_left", "cam_right", "cam_x8d"]


def load_clip_cameras(clip_root):
    clip_root = Path(clip_root)
    calib = json.loads((clip_root / "transforms" / "calib.json").read_text(encoding="utf-8"))
    tf = calib["tf2base_link"]
    # Exported points and detector boxes use the standardized pose frame.
    B_P = np.asarray(tf["pose"], dtype=np.float64)  # base_from_detection_frame
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
            "T": np.linalg.inv(np.asarray(tf[name], dtype=np.float64)) @ B_P,
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


def compute_frame_visibility(dets, cams, occl_tol=0.3):
    """为帧内每个 det 写回 det['visibility']，返回统计。"""
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


def compute_clip_visibility(out_frames, clip_root, args):
    """对整个 clip 计算可见度；可见度低于 drop_vis_below 的 box 删除。"""
    cams = load_clip_cameras(clip_root)
    stats = {"frames": 0, "checked": 0, "tag1": 0, "tag2": 0,
             "dropped": 0, "static_slots_protected": 0,
             "cameras": len(cams)}
    for frame in out_frames:
        if not frame["detections"]:
            continue
        fstats = compute_frame_visibility(frame["detections"], cams, args.vis_occl_tol)
        for k in ("checked", "tag1", "tag2"):
            stats[k] += fstats[k]
        stats["frames"] += 1
        drop_below = getattr(args, "drop_vis_below", 0.0)
        if drop_below > 0:
            stats["static_slots_protected"] += sum(
                1 for d in frame["detections"]
                if (d.get("slot_static")
                    and d.get("visibility", {}).get("ratio", 1.0) <= drop_below)
            )
            kept = [d for d in frame["detections"]
                    if (d.get("slot_static")
                        or d.get("visibility", {}).get("ratio", 1.0) > drop_below)]
            stats["dropped"] += len(frame["detections"]) - len(kept)
            frame["detections"] = kept
    return stats


def filter_raw_frames(frames, clip_root, drop_below, occl_tol=0.3):
    """模型推理后的原始检测 JSON 帧列表：计算可见度并过滤被遮挡>=95% 的 box。
    原地修改 frames（附加 visibility 字段、删除被过滤的检测），返回统计。"""
    cams = load_clip_cameras(clip_root)
    stats = {"frames": 0, "checked": 0, "tag1": 0, "tag2": 0,
             "dropped": 0, "cameras": len(cams)}
    for frame in frames:
        dets = frame["detections"]
        if not dets:
            continue
        fstats = compute_frame_visibility(dets, cams, occl_tol)
        for k in ("checked", "tag1", "tag2"):
            stats[k] += fstats[k]
        stats["frames"] += 1
        if drop_below > 0:
            kept = [d for d in dets if d.get("visibility", {}).get("ratio", 1.0) > drop_below]
            stats["dropped"] += len(dets) - len(kept)
            frame["detections"] = kept
    return stats
