#!/usr/bin/env python3
"""交警 clip -> BEVFusion(OpenPCDet 版) 可吃的输入。

做三件事：
  1) lidar_top 的 4 列 xyzi .bin  -> 5 列 (x,y,z,intensity,0)，NuScenesDataset 硬要求 5 列
  2) 4 路环视鱼眼（KANNALA_BRANDT）-> 针孔图，K 用 calib 自己的 K
     （BEVFusion 的 LSS 视角变换是针孔模型，直接喂鱼眼图会算错相机射线）
  3) 生成 nuScenes 风格 infos（cams 内含内参/外参），供 NuScenesDataset 读取

产出：
  work/undist/<clip>/<cam>/<ts>.jpg         去畸变针孔图
  data/police/<clip>/lidar/lidar_top/<ts>.bin   5 列 bin
  data/police/<clip>/image/<cam>            -> work/undist/<clip>/<cam>
  data/police/<clip>/transforms             -> 原始 calib/pose
  work/infos/police_val_infos.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

CAMS_ALL = ["cam_front", "cam_left", "cam_right", "cam_rear", "cam_x8d"]
CAMS_DEFAULT = ["cam_front", "cam_left", "cam_right", "cam_rear"]  # 4 路环视广角

# ---------- 几何 ----------


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def cam2lidar(calib: dict, cam: str) -> np.ndarray:
    """相机 -> lidar_top 的 4x4 变换矩阵（X_lidar = R X_cam + t）。"""
    T_l2b = np.asarray(calib["tf2base_link"]["lidar_top"], dtype=np.float64)
    T_c2b = np.asarray(calib["tf2base_link"][cam], dtype=np.float64)
    return np.linalg.inv(T_l2b) @ T_c2b


def scale_K(K: np.ndarray, s: float) -> np.ndarray:
    K = K.copy().astype(np.float64)
    K[0, :] *= s
    K[1, :] *= s
    return K


# ---------- 去畸变 ----------


def undistort_image(img: np.ndarray, K: np.ndarray, D: np.ndarray,
                    out_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """鱼眼 -> 针孔。返回 (图, 针孔 K)。图统一成 out_size（不够的部分补黑）。"""
    h, w = img.shape[:2]
    ow, oh = out_size
    s = min(ow / w, oh / h)
    if abs(s - 1.0) > 1e-6:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        K = scale_K(K, s)
    h, w = img.shape[:2]
    k_new = K.copy()
    und = cv2.fisheye.undistortImage(img, K, np.asarray(D, np.float64),
                                     Knew=k_new, new_size=(w, h))
    if (w, h) != (ow, oh):
        canvas = np.zeros((oh, ow, 3), np.uint8)
        ox, oy = (ow - w) // 2, (oh - h) // 2
        canvas[oy:oy + h, ox:ox + w] = und
        und = canvas
        k_new = k_new.copy()
        k_new[0, 2] += ox
        k_new[1, 2] += oy
    return und, k_new


def undistort_job(job):
    src, dst, K, D, out_size = job
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return str(dst)
    img = cv2.imread(str(src))
    if img is None:
        return f"ERR read {src}"
    und, _ = undistort_image(img, np.asarray(K, np.float64), np.asarray(D, np.float64), out_size)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), und, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(dst)


# ---------- 主流程 ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="存放 clip 的目录（每个 clip 一个子目录，含 image/lidar/transforms）")
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parent.parent),
                    help="缓存/产物根（默认 = <project>/bevfusion）")
    ap.add_argument("--cams", default=",".join(CAMS_DEFAULT))
    ap.add_argument("--out-size", default="1920x1536", help="去畸变后图像尺寸 WxH（所有相机必须一致）")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--clips", default="", help="只处理这些 clip（逗号分隔），默认全部")
    ap.add_argument("--limit-frames", type=int, default=0, help=">0 时每个 clip 只取前 N 帧（冒烟用）")
    ap.add_argument("--no-images", action="store_true",
                    help="纯雷达模式：只写 5 列 bin + infos，不去畸变（省掉 90% 预处理时间）")
    args = ap.parse_args()

    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    for c in cams:
        assert c in CAMS_ALL, f"未知相机 {c}"
    ow, oh = (int(x) for x in args.out_size.split("x"))
    out_size = (ow, oh)

    root = Path(args.out_root)
    data_root = Path(args.data_root)
    clips = sorted([d for d in os.listdir(data_root) if d.startswith("scene_")])
    if args.clips:
        want = {c.strip() for c in args.clips.split(",") if c.strip()}
        clips = [c for c in clips if c in want or Path(c).name in want]

    infos = []
    report = []
    jobs = []
    for clip in clips:
        cdir = data_root / clip
        calib = load_json(cdir / "transforms" / "calib.json")
        tss = sorted(p.stem for p in (cdir / "lidar" / "lidar_top").glob("*.bin"))
        if args.limit_frames:
            tss = tss[: args.limit_frames]
        if not tss:
            report.append(f"!! {clip}: 没有 lidar_top/*.bin，跳过")
            continue

        # 图像与雷达时间戳不同名（差几 ms）-> 按最近时间戳配对
        its_by_cam = {}
        for cam in cams:
            its_by_cam[cam] = sorted(p.stem for p in (cdir / "image" / cam).glob("*.jpg"))
        base_its = its_by_cam[cams[0]]
        for cam in cams:
            assert its_by_cam[cam] == base_its, f"{clip}/{cam} 图像帧集合与 {cams[0]} 不一致"
        img_of_ts, dts = {}, []
        for ts in tss:
            it = min(base_its, key=lambda t: abs(int(t) - int(ts)))
            img_of_ts[ts] = it
            dts.append((int(it) - int(ts)) / 1e6)      # ms
        dts_arr = np.asarray(dts)

        # --- 目录 & 软链 ---
        (root / "data" / "police" / clip / "lidar" / "lidar_top").mkdir(parents=True, exist_ok=True)
        tdir = root / "data" / "police" / clip / "transforms"
        if not (tdir.exists() or tdir.is_symlink()):
            tdir.symlink_to(cdir / "transforms")
        for cam in cams:
            p = root / "data" / "police" / clip / "image" / cam
            p.parent.mkdir(parents=True, exist_ok=True)
            if not (p.exists() or p.is_symlink()):
                p.symlink_to(root / "work" / "undist" / clip / cam)
            if args.no_images:
                continue
            K = np.asarray(calib[cam]["K"], np.float64)
            D = np.asarray(calib[cam]["D"], np.float64)
            for ts in tss:
                it = img_of_ts[ts]
                jobs.append((cdir / "image" / cam / f"{it}.jpg",
                             root / "work" / "undist" / clip / cam / f"{it}.jpg",
                             K.tolist(), D.tolist(), out_size))

        # --- lidar 5 列 bin ---
        npts = []
        for ts in tss:
            src = cdir / "lidar" / "lidar_top" / f"{ts}.bin"
            dst = root / "data" / "police" / clip / "lidar" / "lidar_top" / f"{ts}.bin"
            if not dst.exists() or dst.stat().st_size == 0:
                p = np.fromfile(str(src), np.float32).reshape(-1, 4)
                five = np.concatenate([p, np.zeros((len(p), 1), np.float32)], axis=1)
                five.tofile(str(dst))
            else:
                p = np.fromfile(str(src), np.float32).reshape(-1, 4)
            npts.append(len(p))

        # --- infos ---
        for ts, np_ in zip(tss, npts):
            cams_info = {}
            for cam in cams:
                M = cam2lidar(calib, cam)
                K = np.asarray(calib[cam]["K"], np.float64)
                # 去畸变后 K：与 undistort_image 完全一致的缩放/平移
                w, h = int(calib[cam]["imgw"]), int(calib[cam]["imgh"])
                s = min(ow / w, oh / h)
                Kp = scale_K(K, s)
                nw, nh = int(round(w * s)), int(round(h * s))
                Kp[0, 2] += (ow - nw) // 2
                Kp[1, 2] += (oh - nh) // 2
                cams_info[cam] = {
                    "data_path": f"{clip}/image/{cam}/{img_of_ts[ts]}.jpg",
                    "cam_intrinsic": Kp.astype(np.float32),
                    "camera_intrinsics": Kp.astype(np.float32),
                    "sensor2lidar_rotation": M[:3, :3].astype(np.float32),
                    "sensor2lidar_translation": M[:3, 3].astype(np.float32),
                    "sensor2ego_rotation": Rotation.from_matrix(
                        np.asarray(calib["tf2base_link"][cam], np.float64)[:3, :3]).as_quat().astype(np.float32),
                    "sensor2ego_translation": np.asarray(
                        calib["tf2base_link"][cam], np.float64)[:3, 3].astype(np.float32),
                    "ego2global_rotation": np.array([0, 0, 0, 1], np.float32),
                    "ego2global_translation": np.zeros(3, np.float32),
                    "timestamp": int(img_of_ts[ts]),
                }
            infos.append({
                "token": f"{clip}_{ts}",
                "scene_token": clip,
                "timestamp": int(ts),
                "frame_id": ts,
                "lidar_path": f"{clip}/lidar/lidar_top/{ts}.bin",
                "sweeps": [],
                "num_lidar_pts": int(np_),
                "cams": cams_info,
            })
        report.append(f"== {clip}: {len(tss)} 帧, 点云 P50 {int(np.median(npts))}, "
                      f"相机 {cams}, 图-雷达时差 P50 {np.median(dts_arr):.1f}ms "
                      f"[{dts_arr.min():.0f}, {dts_arr.max():.0f}]")

    # 去畸变（多进程，已存在会跳过）
    if jobs:
        print(f"[undistort] {len(jobs)} 张图，{args.jobs} 进程 ...", flush=True)
        done = 0
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for i, r in enumerate(ex.map(undistort_job, jobs, chunksize=8), 1):
                if str(r).startswith("ERR"):
                    print("  ", r, flush=True)
                done += 1
                if done % 200 == 0:
                    print(f"    {done}/{len(jobs)}", flush=True)

    out = root / "work" / "infos" / "police_val_infos.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    # INFO_PATH 是相对 dataset root（data/police）解析的
    link = root / "data" / "police" / "infos"
    if not (link.exists() or link.is_symlink()):
        link.symlink_to(root / "work" / "infos")
    with open(out, "wb") as f:
        pickle.dump(infos, f)
    print("\n".join(report))
    print(f"\n[infos] {len(infos)} 帧 -> {out}")
    if infos:
        print(f"[cams] 帧0 相机顺序: {list(infos[0]['cams'].keys())}")
        c0 = infos[0]["cams"][list(infos[0]["cams"])[0]]
        print(f"[示例] {c0['data_path']}\n        K=\n{np.round(c0['camera_intrinsics'],2)}")


if __name__ == "__main__":
    sys.exit(main())
