#!/usr/bin/env python3
"""Truck 链 Step1（BEVFusion 版）——替换原来的 VoxelNeXt truck 检测器。

三步：
  ① 预处理：4 路环视鱼眼去畸变成针孔图 + lidar_top 写成 5 列 bin + 生成 nuScenes 风格 infos
     （缓存复用，已存在就跳过）
  ② 推理：mmdet3d 1.x 版 BEVFusion（官方 20 epoch 权重，nv val NDS 71.4 / mAP 68.6）出 10 类框，
     输出前已把 z 从「框底面」统一成「框中心」（见 bevfusion_truck/scripts/infer_mmdet3d.py）
  ③ 挂相机可见性元数据（与旧 step1_lidar_inference.py 完全一致，供后面 hard filter 用）

输出：``<work-root>/<clip.name>_raw.json``，结构与 ``inference/run_prelabel.py`` 的一致
（frames: [{frame_id, detections:[{class_name, score, box_lidar}]}]，lidar 系），
可被 step2 及之后的所有阶段直接消费。

【改动·2026-10-08】过程数据不落盘：①的缓存（bevfusion/data/police/<clip> 每包约 150MB、
work/infos/*、去畸变图）在推理结束时就地删掉，只留上面的 raw json。批量跑包不会再让
``bevfusion/data/`` 无限增长。要留缓存调试用 ``--keep-prep``（或 ``BEVFUSION_KEEP_PREP=1``）。

本脚本自身跑在 **openpcdet** 环境（需要 filtering.camera_visibility）；
①② 通过外部 BEVFusion 工程用 **mmdet3d** 环境的 python 以子进程执行，
路径可用环境变量覆盖（云端部署时必改）：
    BEVFUSION_ROOT    默认 <project>/bevfusion（配置/脚本/缓存都随项目走）
    BEVFUSION_PYTHON  默认 ~/miniconda3/envs/mmdet3d/bin/python（mmdet3d 环境）
    BEVFUSION_TRUCK_CFG / BEVFUSION_TRUCK_CKPT         C+L 配置/权重（默认 models/bevfusion_mmdet3d_lidarcam.pth）
    BEVFUSION_TRUCK_LIDAR_CFG / ..._LIDAR_CKPT         纯雷达配置/权重（mode=lidar，默认）：
        默认 = 交警域微调 ct2-A —— configs/police_bevfusion_mmdet3d_lidaronly_ct2roi.py +
        models/ft_ct2_A_fulllr_epoch8.pth（可用环境变量指回官方 lidaronly 那套）
    MMDET3D_ROOT      mmdetection3d 源码位置（默认 ~/MMDetection/mmdetection3d 等常见位置）
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filtering import camera_visibility  # noqa: E402

BEVFUSION_ROOT = Path(os.environ.get(
    "BEVFUSION_ROOT", str(PROJECT_ROOT / "bevfusion")))      # 配置/脚本/缓存都随项目走
BEVFUSION_PYTHON = Path(os.environ.get(
    "BEVFUSION_PYTHON", str(Path.home() / "miniconda3" / "envs" / "mmdet3d" / "bin" / "python")))
BEVFUSION_CFG = os.environ.get("BEVFUSION_TRUCK_CFG",
                               "configs/police_bevfusion_mmdet3d.py")
BEVFUSION_CKPT = os.environ.get(
    "BEVFUSION_TRUCK_CKPT", str(PROJECT_ROOT / "models" / "bevfusion_mmdet3d_lidarcam.pth"))
BEVFUSION_Z = os.environ.get("BEVFUSION_TRUCK_Z", "center")
# 【改动·2026-09-29】纯雷达（lidar-only）默认换成交警域微调权重 ct2-A（epoch 8）：
#   权重 models/ft_ct2_A_fulllr_epoch8.pth（car/truck 两类微调，5.7 万帧自动+人工数据）；
#   配置 configs/police_bevfusion_mmdet3d_lidaronly_ct2roi.py（训练几何：侧向 ±50.4 /
#   前进 80.4 / 后退 20.4，1344×1344 正方网格；与权重必须成对使用）。
#   旧官方权重（bevfusion_mmdet3d_lidaronly.pth + 对称 ±54 配置）仍可用环境变量或
#   --bev-cfg/--bev-ckpt 显式指回。
BEVFUSION_LIDAR_CFG = os.environ.get(
    "BEVFUSION_TRUCK_LIDAR_CFG", "configs/police_bevfusion_mmdet3d_lidaronly_ct2roi.py")
BEVFUSION_LIDAR_CKPT = os.environ.get(
    "BEVFUSION_TRUCK_LIDAR_CKPT", str(PROJECT_ROOT / "models" / "ft_ct2_A_fulllr_epoch8.pth"))


def validate_clip(clip: Path) -> None:
    lidar_top = clip / "lidar" / "lidar_top"
    if not lidar_top.is_dir() or not list(lidar_top.glob("*.bin")):
        raise ValueError(f"clip 缺少 lidar/lidar_top/*.bin：{clip}")
    for name in ("transforms", "image"):
        if not (clip / name).is_dir():
            raise ValueError(f"clip 缺少 {name}/：{clip}（BEVFusion 相机分支需要）")
    if not (clip / "transforms" / "calib.json").is_file():
        raise ValueError(f"clip 缺少 transforms/calib.json：{clip}")


def _run(cmd) -> None:
    print(">> " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def prepare(clip: Path, jobs: int = 6, images: bool = True) -> None:
    """去畸变（可选）+ 5 列 bin + infos（bevfusion_truck/scripts/prep_data.py，幂等）。"""
    cmd = [BEVFUSION_PYTHON, BEVFUSION_ROOT / "scripts" / "prep_data.py",
           "--data-root", clip.parent, "--out-root", BEVFUSION_ROOT,
           "--clips", clip.name, "--jobs", str(jobs)]
    if not images:          # 【改动】纯雷达模式：跳过鱼眼去畸变（省掉 ~90% 预处理时间）
        cmd.append("--no-images")
    _run(cmd)
    # 【改动】再生成 mmdet3d 需要的 middle-format infos（纯雷达模式下图片时间戳回退到原始 image/）
    _run([BEVFUSION_PYTHON, BEVFUSION_ROOT / "scripts" / "mmdet3d_prep.py",
          "--data-root", clip.parent, "--out-root", BEVFUSION_ROOT,
          "--clips", clip.name])


def infer(clip: Path, raw_json: Path, score_thresh: float, cfg: str, ckpt: str,
          z_convention: str) -> None:
    _run([BEVFUSION_PYTHON, BEVFUSION_ROOT / "scripts" / "infer_mmdet3d.py",
          "--cfg_file", str(BEVFUSION_ROOT / cfg if not Path(cfg).is_absolute() else cfg),
          "--ckpt", str(BEVFUSION_ROOT / ckpt if not Path(ckpt).is_absolute() else ckpt),
          "--out_json", str(raw_json), "--clips", clip.name,
          "--score_thresh", str(score_thresh),
          "--z-convention", z_convention])


# 【改动·2026-10-08】预处理缓存默认跑完即删。
#   背景：prep_data.py 会把每个 clip 的 lidar_top 4 列 bin 复制成 5 列（LoadPointsFromFile
#   写死 load_dim=5，不补列喂不进去）——单帧 ~1.9MB、比源数据还大 25%，一个 80 帧的包
#   ~150MB；再加 transforms 复制、infos pkl、去畸变图。以前这份缓存只写不删，批量跑几百个
#   包后 <project>/bevfusion/data/police/ 涨到 50G（见 README「过程数据」）。
#   现在 step1 推理结束（成功或失败）就清掉本次 clip 的全部过程数据，只留
#   <work-root>/<clip>_raw.json。要留着缓存调试 -> --keep-prep 或 BEVFUSION_KEEP_PREP=1。
PREP_KEEP_ENV = "BEVFUSION_KEEP_PREP"


def keep_prep_enabled(args) -> bool:
    """是否保留 BEVFusion 预处理缓存。"""
    if bool(getattr(args, "keep_prep", False)):
        return True
    return os.environ.get(PREP_KEEP_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _drop_clip_from_aggregate_infos(infos_dir: Path, clip_name: str) -> Path | None:
    """从聚合 infos 里摘掉该 clip 的条目，保留其它 clip（坏文件直接删）。

    聚合文件是所有 clip 共用的一份，所以只摘本次 clip，不能整份删 —— 否则并发跑别的
    clip 的进程会读到空 infos（正是 tests/root/test_bevfusion_infos.py 记录的那个坑）。
    """
    agg = infos_dir / "police_mmdet3d_infos.pkl"
    if not agg.is_file():
        return None
    try:
        with open(agg, "rb") as f:
            payload = pickle.load(f)
        data = payload.get("data_list", []) if isinstance(payload, dict) else []
        kept = [x for x in data if x.get("scene_token") != clip_name]
    except Exception:                       # 读不了 / 结构不对：留着也没用
        agg.unlink(missing_ok=True)
        return agg
    if len(kept) == len(data):              # 本来就没有本 clip 的条目：不动它
        return None
    if not kept:                            # 没有别的 clip 了 -> 直接删
        agg.unlink(missing_ok=True)
        return agg
    if isinstance(payload, dict):
        payload["data_list"] = kept
    else:
        payload = {"metainfo": {"dataset": "police_bevfusion"}, "data_list": kept}
    tmp = agg.with_name(agg.name + ".tmp")
    with open(tmp, "wb") as f:              # 先写临时文件再改名：并发下不会读到半个 pkl
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, agg)
    return agg


def cleanup_prep(clip: Path) -> list[Path]:
    """删掉本次 prep 产生的过程数据，返回实际删掉的路径。

    覆盖 prep_data.py / mmdet3d_prep.py / 去畸变的全部落盘点：
      data/police/<clip>/                5 列 bin + transforms 复制 + image 软链（大头）
      work/undist/<clip>/                去畸变针孔图（仅 fusion 模式有）
      work/infos/<clip>_infos.pkl        per-clip infos
      work/infos/police_mmdet3d_infos.pkl 聚合 infos（只摘本 clip 条目）
      work/infos/police_val_infos.pkl     prep_data 的冗余 infos（只留最后一次跑的，删）
    """
    name = clip.name
    removed: list[Path] = []
    for target in (BEVFUSION_ROOT / "data" / "police" / name,
                   BEVFUSION_ROOT / "work" / "undist" / name):
        if target.is_symlink() or target.exists():
            if target.is_symlink() or target.is_file():
                target.unlink(missing_ok=True)
            else:
                shutil.rmtree(target, ignore_errors=True)
            removed.append(target)
    infos_dir = BEVFUSION_ROOT / "work" / "infos"
    for name_only in (f"{name}_infos.pkl", "police_val_infos.pkl"):
        p = infos_dir / name_only
        if p.is_file():
            p.unlink()
            removed.append(p)
    if (pruned := _drop_clip_from_aggregate_infos(infos_dir, name)) is not None:
        removed.append(pruned)
    return removed


def run_inference(clip: Path, args) -> Path:
    raw_json = Path(args.work_root) / f"{clip.name}_raw.json"
    raw_json.parent.mkdir(parents=True, exist_ok=True)
    mode = str(getattr(args, "mode", "fusion"))
    if mode == "lidar":
        # 纯雷达：不需要 image/ 目录，也不需要去畸变
        lidar_top = clip / "lidar" / "lidar_top"
        if not lidar_top.is_dir() or not list(lidar_top.glob("*.bin")):
            raise ValueError(f"clip 缺少 lidar/lidar_top/*.bin：{clip}")
        if not (clip / "transforms" / "calib.json").is_file():
            raise ValueError(f"clip 缺少 transforms/calib.json：{clip}")
    else:
        validate_clip(clip)
    try:
        if not args.skip_prepare:
            prepare(clip, jobs=args.jobs, images=(mode != "lidar"))
        cfg = args.cfg if mode != "lidar" else os.environ.get(
            "BEVFUSION_TRUCK_LIDAR_CFG", BEVFUSION_LIDAR_CFG)
        ckpt = args.ckpt if mode != "lidar" else os.environ.get(
            "BEVFUSION_TRUCK_LIDAR_CKPT", BEVFUSION_LIDAR_CKPT)
        if mode == "lidar" and (args.cfg != BEVFUSION_CFG or args.ckpt != BEVFUSION_CKPT):  # 显式传了就用显式值
            cfg, ckpt = args.cfg, args.ckpt    # 显式传了就用显式值
        infer(clip, raw_json, args.score_thresh, cfg, ckpt, args.z_convention)
        if not raw_json.is_file():
            raise RuntimeError(f"BEVFusion 推理没有产出 {raw_json}")

        if not args.no_visibility_check:
            frames = json.loads(raw_json.read_text(encoding="utf-8"))
            stats = camera_visibility.filter_raw_frames(frames, clip, 0.0, args.vis_occl_tol)
            raw_json.write_text(json.dumps(frames, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
            print(f"visibility metadata: checked={stats['checked']} "
                  f"dropped={stats['dropped']}", flush=True)
        return raw_json
    finally:
        # 【改动·2026-10-08】放在 finally：推理失败也不会把 ~150MB/包 的过程数据留在盘上。
        if keep_prep_enabled(args):
            print(f"[prep] 保留预处理缓存（--keep-prep / {PREP_KEEP_ENV}=1）："
                  f"{BEVFUSION_ROOT / 'data' / 'police' / clip.name}", flush=True)
        else:
            try:
                removed = cleanup_prep(clip)
            except Exception as exc:        # 清理失败不能把整批跑挂掉，但必须吼出来
                print(f"[prep][warn] 过程数据清理失败（{exc}），请手动删 "
                      f"{BEVFUSION_ROOT / 'data' / 'police' / clip.name}",
                      file=sys.stderr, flush=True)
            else:
                for p in removed:
                    print(f"[prep] 已清理过程数据 {p}", flush=True)


def collect_clips(args):
    clips = [Path(c).resolve() for c in args.clip]
    for root in args.clip_dir:
        root = Path(root).resolve()
        for cand in (sorted(root.iterdir()) if root.is_dir() else []):
            try:      # 【改动】跳过 lost+found 等权限不足/无关条目
                if cand.name in {"lost+found", ".Trash-1000"} or cand.name.startswith("."):
                    continue
                if (cand / "lidar" / "lidar_top").is_dir():
                    clips.append(cand)
            except OSError:
                continue
    if not clips:
        raise SystemExit("没有输入 clip（--clip 或 --clip-dir）")
    return clips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", action="append", default=[])
    parser.add_argument("--clip-dir", action="append", default=[])
    parser.add_argument("--work-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step1_inference")
    parser.add_argument("--mode", choices=["fusion", "lidar"], default="fusion",
                        help="fusion=C+L（默认，读 4 路图）; lidar=纯雷达权重（不读图/不去畸变，快 6 倍）")
    parser.add_argument("--cfg", default=BEVFUSION_CFG)
    parser.add_argument("--ckpt", default=BEVFUSION_CKPT)
    parser.add_argument("--score-thresh", type=float, default=0.1,
                        help="检测器原始阈值（链内类别阈值在 step2 把关）")
    parser.add_argument("--z-convention", default=BEVFUSION_Z,
                        choices=["center", "bottom"])
    parser.add_argument("--skip-prepare", action="store_true",
                        help="复用已有去畸变图/infos（调试用；默认跑完就删缓存，"
                             "所以要和 --keep-prep 一起用）")
    parser.add_argument("--keep-prep", action="store_true",
                        help="【默认关】保留本次 clip 的过程数据"
                             "（bevfusion/data/police/<clip>、work/infos/*、去畸变图）。"
                             f"默认推理一结束就删；也可用 {PREP_KEEP_ENV}=1 打开")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--vis-occl-tol", type=float, default=0.3)
    parser.add_argument("--no-visibility-check", action="store_true")
    args = parser.parse_args()

    # 【改动·2026-10-08】--skip-prepare 依赖上一次跑留下的缓存，而缓存默认跑完就删
    # -> 组合不成立，直接在入口拦掉，免得跑到一半报「找不到 bin/calib」。
    if args.skip_prepare and not keep_prep_enabled(args):
        parser.error("--skip-prepare 需要缓存已在（bevfusion/data/police/<clip>）；"
                     "缓存默认跑完即删，请加 --keep-prep 复用（或先跑一次 --keep-prep）")

    if not BEVFUSION_PYTHON.exists():
        raise SystemExit(f"BEVFUSION_PYTHON 不存在：{BEVFUSION_PYTHON}")
    if not (BEVFUSION_ROOT / "scripts" / "infer_mmdet3d.py").is_file():
        raise SystemExit(f"BEVFUSION_ROOT 下找不到 scripts/infer_mmdet3d.py：{BEVFUSION_ROOT}")

    summaries = []
    for index, clip in enumerate(collect_clips(args), 1):
        print(f"[{index}] {clip.name}", flush=True)
        raw = run_inference(clip, args)
        summaries.append({"clip": str(clip), "raw_json": str(raw)})
    print(json.dumps({"clips": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
