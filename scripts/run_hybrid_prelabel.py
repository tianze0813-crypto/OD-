#!/usr/bin/env python3
"""串行运行三条链并合并为一个 SUST 预标注结果（【改动】2026-09-18）。

接入顺序（用户指定）：**先 Car -> 再 Truck -> 最后 Pedestrian/Nonmotorized_vehicle**

  1. Car   : 合并车链（BEVFusion Car+Truck 一次推理 + 共享跟踪 + Car 专属 step3/4/4.5/5）
             —— 旧单 Car 链（Waymo Car 头 / hybrid_expD_car）已于 2026-09-22 删除
  2. Truck : pipeline/hybrid_expD_truck.py，权重 models/voxelnext_truckB_epoch15.pth（obj_id +1000）
  3. VRU   : pipeline/hybrid_expD_vru.py，权重 models/voxelnext_vru_1head2cls_epoch20.pth（obj_id +2000）

三条链各自独立推理，最后按 frame_id 合成一份 label/。
用 --chains 可选子集，例如 --chains car,truck；旧五类单链用 --chains car,noncar。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.hybrid_expD_noncar import run as run_expd_noncar
from pipeline.hybrid_expD_truck import run as run_expd_truck
from pipeline.hybrid_expD_vru import run as run_expd_vru
from pipeline.vehicle_pass import run as run_vehicle_pass        # 【改动】车链合并后处理
from pipeline.hybrid_merge import merge_label_frames


DEFAULT_OUTPUT_ROOT = Path.home() / "SUSTechPOINTS" / "data"
NONCAR_CFG = ROOT / "models" / "voxelnext_fiveclass_nuscenes_infer.yaml"
# Final production non-Car weight: VOD 2-class fine-tune, epoch 12
# (checkpoint stores epoch 6).  The older expD_e8.pth is no longer the default
# and was archived to bak/models-unused/ (2026-09-23 cleanup); pass an explicit
# path via --noncar-ckpt if you need it back.
NONCAR_CKPT = ROOT / "models" / "vod_2cls_ft_e12.pth"
# 【改动】三条链各自的权重/配置
TRUCK_CFG = ROOT / "models" / "voxelnext_truck_infer.yaml"
TRUCK_CKPT = ROOT / "models" / "voxelnext_truckB_epoch15.pth"
VRU_CFG = ROOT / "models" / "voxelnext_vru_infer.yaml"
VRU_CKPT = ROOT / "models" / "voxelnext_vru_1head2cls_epoch20.pth"
TRUCK_ID_OFFSET = 1000
VRU_ID_OFFSET = 2000
DEFAULT_CHAINS = ("car", "truck", "vru")
REQUIRED_MODULES = ("numpy", "scipy", "cv2", "PIL", "pandas", "av2",
                    "kornia", "yaml", "torch", "spconv", "pcdet")


def _print(message: str) -> None:
    print(f"[hybrid] {message}", flush=True)


def _run(command: List[Any], *, check: bool = True,
         capture: bool = False) -> subprocess.CompletedProcess[str]:
    _print("$ " + " ".join(str(value) for value in command))
    return subprocess.run([str(value) for value in command], check=check,
                          text=True, capture_output=capture)


# 【改动】跳过滤镜目录/权限不足的条目：移动盘上常有 root 权限的 lost+found，
# 旧代码对它 stat() 会直接 PermissionError 崩掉整批收集。
_SKIP_DIR_NAMES = {"lost+found", ".Trash-1000", ".Trash", "$RECYCLE.BIN",
                   "System Volume Information"}


PRE_SUFFIX = "_pre"


def _pre_clip_name(base: str, tag: str = "",
                   output_suffix: str = "") -> str:
    """目标输出目录名：绝不叠加第二个 ``_pre``。

    - ``<clip>``              -> ``<clip>_pre``（有 tag 时 ``<clip>_<tag>_pre``）
    - ``<clip>_pre``          -> 原样返回：重跑就地覆盖 ``label/``，不生成
      ``<clip>_pre_pre``（输入本身就是预标产物，只重写标签）
    - 显式 ``output_suffix``  -> ``<clip><后缀>``，按调用方给的名字来
    """
    if output_suffix:                        # 【改动】<clip名><后缀>，如 ..._clip4_pre_bev
        return f"{base}{output_suffix}"
    if base.endswith(PRE_SUFFIX):            # 【改动】已是 *_pre：重跑就地覆盖，不再套一层
        return base
    return f"{base}_{tag}_pre" if tag else f"{base}_pre"


def _skip_entry(path: Path, include_pre: bool = False) -> bool:
    name = path.name
    if name in _SKIP_DIR_NAMES or name.startswith("."):
        return True
    return name.endswith(PRE_SUFFIX) and not include_pre  # 【改动】include_pre 时也收 _pre


def _is_clip(path: Path, include_pre: bool = False) -> bool:
    try:
        if _skip_entry(path, include_pre) or not path.is_dir():
            return False
        lidar = path / "lidar" / "lidar_top"
        return lidar.is_dir() and any(lidar.glob("*.bin"))
    except OSError:      # 权限不足（如移动盘的 lost+found）-> 视为不是 clip
        return False


def _collect_clips(input_root: Path, include_pre: bool = False) -> List[Path]:
    if not input_root.is_dir():
        raise RuntimeError(f"input directory does not exist: {input_root}")
    # Accept a single clip directory directly, or a parent holding many clips.
    # 【改动】直接点名的那一个目录就算是 *_pre 也照收（用户明确指定 = 想在它上面
    # 就地重跑 label/）；include_pre=False 的过滤只用于自动遍历父目录时跳过已标注的。
    if _is_clip(input_root, include_pre=True):
        return [input_root.resolve()]
    clips = []
    for path in sorted(input_root.iterdir()):
        try:
            if _skip_entry(path, include_pre) or not _is_clip(path, include_pre):
                continue
            clips.append(path.resolve())
        except OSError:      # 【改动】权限不足的条目直接跳过
            continue
    if not clips:
        raise RuntimeError(
            f"no raw clips found under {input_root}; expected lidar/lidar_top/*.bin")
    # 【改动】--include-pre 时如果同时存在 X 与 X_pre，只跑 X_pre（重复跑 X 会先把 X_pre 删掉）
    if include_pre:
        pre_names = {path.name for path in clips
                     if path.name.endswith(PRE_SUFFIX)}
        dropped = [path for path in clips
                   if not path.name.endswith(PRE_SUFFIX)
                   and f"{path.name}{PRE_SUFFIX}" in pre_names]
        if dropped:
            _print("--include-pre：以下 clip 已有 *_pre 版本，跳过 "
                   + ", ".join(path.name for path in dropped))
            clips = [path for path in clips if path not in dropped]
    names = [path.name for path in clips]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate clip names in input batch")
    return clips


def _probe(python: Path) -> Dict[str, Any]:
    modules = repr(REQUIRED_MODULES)
    code = (
        "import importlib.util, json, sys; "
        f"modules = {modules}; "
        "result={'python':sys.executable,'modules':{},'cuda':False}; "
        "result['modules']={n:bool(importlib.util.find_spec(n)) for n in modules}; "
        "\ntry:\n import torch; result['cuda']=bool(torch.cuda.is_available())\n"
        "except Exception as exc: result['torch_error']=str(exc)\n"
        "print(json.dumps(result))"
    )
    result = subprocess.run([str(python), "-c", code], text=True,
                            capture_output=True)
    if result.returncode:
        return {"error": result.stderr.strip()}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": result.stdout.strip()}


def _healthy(probe: Dict[str, Any]) -> bool:
    return (not probe.get("error")
            and bool(probe.get("cuda"))
            and all(probe.get("modules", {}).get(name, False)
                    for name in REQUIRED_MODULES))


def _validate_weight(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"checkpoint not found: {path}")
    with path.open("rb") as stream:
        header = stream.read(256)
    if b"git-lfs.github.com/spec" in header:
        raise RuntimeError(
            f"checkpoint is a Git LFS pointer, not model data: {path}; "
            "run git lfs pull before starting hybrid inference")


def _python_candidates(explicit: Path | None) -> List[Path]:
    values: List[Path] = []
    if explicit:
        values.append(explicit.expanduser().resolve())
    env_python = os.environ.get("OPENPCDET_PYTHON")
    if env_python:
        values.append(Path(env_python).expanduser().resolve())
    values.append(Path(sys.executable).resolve())
    home = Path.home()
    values.extend([
        home / "miniconda3" / "envs" / "openpcdet" / "bin" / "python",
        home / "anaconda3" / "envs" / "openpcdet" / "bin" / "python",
        home / "miniconda3" / "envs" / "sustechpoints" / "bin" / "python",
        home / "anaconda3" / "envs" / "sustechpoints" / "bin" / "python",
    ])
    result: List[Path] = []
    seen = set()
    for candidate in values:
        if str(candidate) in seen or not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        seen.add(str(candidate))
        result.append(candidate)
    return result


def _write_labels(frames: List[Dict[str, Any]], clip: Path) -> int:
    label_dir = clip / "label"
    if label_dir.exists():
        shutil.rmtree(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for frame in frames:
        labels = [copy.deepcopy(label) for label in frame.get("labels", [])]
        (label_dir / f"{frame['frame_id']}.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        count += len(labels)
    return count


def _run_raw(python: Path, clip: Path, cfg: Path, ckpt: Path,
             work_root: Path, name: str, score_thresh: float) -> Path:
    # step1_lidar_inference.py writes <clip.name>_raw.json under work_root.
    output = work_root / f"{clip.name}_raw.json"
    _run([
        python, ROOT / "pipeline" / "step1_lidar_inference.py",
        "--clip", clip, "--work-root", work_root,
        "--cfg", cfg, "--ckpt", ckpt,
        "--score-thresh", score_thresh, "--drop-vis-below", "0.0",
    ])
    if not output.is_file():
        raise RuntimeError(f"inference did not create raw JSON: {output}")
    return output


def _run_raw_bevfusion(python: Path, clip: Path, work_root: Path,
                       score_thresh: float, mode: str = "lidar") -> Path:
    """【改动】2026-09-20：Truck 链改用 BEVFusion 检测器（pipeline/step1_bevfusion_truck.py）。

    该脚本内部会：去畸变+5列bin+infos（幂等缓存）-> mmdet3d BEVFusion 推理（z 已统一到框中心）
    -> 挂相机可见性；产物仍是 <clip>_raw.json，结构与旧链路一致。
    """
    output = work_root / f"{clip.name}_raw.json"
    _run([
        python, ROOT / "pipeline" / "step1_bevfusion_truck.py",
        "--clip", clip, "--work-root", work_root,
        "--score-thresh", score_thresh, "--mode", str(mode),
    ])
    if not output.is_file():
        raise RuntimeError(f"BEVFusion 推理没有创建 raw JSON: {output}")
    return output


def _frames_to_labels(frames: List[Dict[str, Any]],
                      id_offset: int) -> Dict[str, List[Dict[str, Any]]]:
    """把某条链的 frames(detections) 转成 {frame_id: [SUST label]}，obj_id 统一加偏移。"""
    from tracking import tracker_conservative as tracking
    out: Dict[str, List[Dict[str, Any]]] = {}
    for frame in frames:
        items = []
        for det in frame.get("detections", []):
            if det.get("track_id") is None:
                continue
            item = tracking.box_to_label(det)
            try:
                item["obj_id"] = str(int(item["obj_id"]) + int(id_offset))
            except (TypeError, ValueError):
                item["obj_id"] = "x" + str(item["obj_id"])
            items.append(item)
        out[str(frame["frame_id"])] = items
    return out


def _label_box7(label: Dict[str, Any]) -> tuple:
    """SUST label -> box7 (x, y, z, dx, dy, dz, yaw)，用于算 BEV IoU。"""
    psr = label["psr"]
    return (float(psr["position"]["x"]), float(psr["position"]["y"]),
            float(psr["position"]["z"]), float(psr["scale"]["x"]),
            float(psr["scale"]["y"]), float(psr["scale"]["z"]),
            float(psr["rotation"]["z"]))


def _car_cover_ratio(car_box: tuple, truck_box: tuple) -> tuple:
    """【改动】Car 被 Truck 覆盖的面积比 = 交面积 / Car 面积；同时给出 BEV IoU 作参考。"""
    from tracking import tracker_conservative as tracking
    car_poly = tracking.rectangle_corners(car_box[:2], car_box[3:5], car_box[6])
    truck_poly = tracking.rectangle_corners(truck_box[:2], truck_box[3:5],
                                            truck_box[6])
    inter = tracking.polygon_area(
        tracking.convex_intersection(car_poly, truck_poly))
    car_area = tracking.polygon_area(car_poly)
    truck_area = tracking.polygon_area(truck_poly)
    union = car_area + truck_area - inter
    cover = 0.0 if car_area <= 1e-9 else inter / car_area
    iou = 0.0 if union <= 1e-9 else inter / union
    return cover, iou


def _drop_cars_covered_by_trucks(
        frames: List[Dict[str, Any]], threshold: float = 0.8
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """【改动】Car 被 Truck 覆盖的面积 >= Car 面积的阈值 → 删掉【该 Car id 的全部帧】。

    规则来自用户 2026-09-18：不是用 IoU（Truck 比 Car 大得多，IoU 天然偏小），
    而是「交面积 / Car 面积」>= 80% 时以 Truck 为准；且不是只删重叠那一帧，
    而是整条 Car 轨迹都删。
    """
    from tracking import tracker_conservative as tracking
    if threshold is None or float(threshold) <= 0:
        return frames, {"enabled": False}
    dropped_ids: set = set()
    matched_pairs = 0
    worst: List[Dict[str, Any]] = []
    for frame in frames:
        cars = [item for item in frame.get("labels", [])
                if item.get("obj_type") == "Car"]
        trucks = [item for item in frame.get("labels", [])
                  if item.get("obj_type") == "Truck"]
        if not cars or not trucks:
            continue
        truck_boxes = [_label_box7(item) for item in trucks]
        for car in cars:
            car_id = str(car.get("obj_id"))
            if car_id in dropped_ids:
                continue
            car_box = _label_box7(car)
            best_cover = 0.0
            best_iou = 0.0
            best_truck = None
            for truck, truck_box in zip(trucks, truck_boxes):
                cover, iou = _car_cover_ratio(car_box, truck_box)
                if cover > best_cover:
                    best_cover, best_iou = cover, iou
                    best_truck = str(truck.get("obj_id"))
            if best_cover >= float(threshold):
                dropped_ids.add(car_id)
                matched_pairs += 1
                worst.append({"frame_id": frame.get("frame_id"),
                              "car_id": car_id, "truck_id": best_truck,
                              "cover": round(best_cover, 3),
                              "iou": round(best_iou, 3)})
    if not dropped_ids:
        return frames, {"enabled": True, "threshold": float(threshold),
                        "metric": "intersection / car_area",
                        "dropped_car_ids": 0, "dropped_boxes": 0,
                        "matched_pairs": 0}
    kept_frames: List[Dict[str, Any]] = []
    dropped_boxes = 0
    for frame in frames:
        labels = []
        for item in frame.get("labels", []):
            if (item.get("obj_type") == "Car"
                    and str(item.get("obj_id")) in dropped_ids):
                dropped_boxes += 1
                continue
            labels.append(item)
        kept_frames.append({**frame, "labels": labels})
    return kept_frames, {"enabled": True, "threshold": float(threshold),
                         "metric": "intersection / car_area",
                         "dropped_car_ids": len(dropped_ids),
                         "dropped_boxes": dropped_boxes,
                         "matched_pairs": matched_pairs,
                         "dropped_ids": sorted(dropped_ids)[:40],
                         "examples": worst[:10]}


def _merge_chain_labels(chain_labels: Dict[str, Dict[str, List[Dict[str, Any]]]],
                        order: List[str],
                        car_truck_cover_threshold: float = 0.5
                        ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """【改动】按 order 顺序把多条链的 label 合成一份（帧内 obj_id 去重）。

    合成后执行 Car/Truck 重叠规则：Car 被 Truck 覆盖的面积 >=
    car_truck_cover_threshold 时，删掉该 Car id 的所有帧。
    """
    frame_ids: List[str] = []
    for name in order:
        for frame_id in chain_labels.get(name, {}):
            if frame_id not in frame_ids:
                frame_ids.append(frame_id)
    frame_ids.sort()
    output: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    collisions = 0
    for frame_id in frame_ids:
        labels: List[Dict[str, Any]] = []
        for name in order:
            labels.extend(copy.deepcopy(chain_labels.get(name, {}).get(frame_id, [])))
        seen: set[str] = set()
        kept: List[Dict[str, Any]] = []
        for item in labels:
            key = str(item.get("obj_id"))
            if key in seen:                      # 同帧 id 冲突：换一个空闲 id
                collisions += 1
                spare = 900000
                while str(spare) in seen:
                    spare += 1
                item["obj_id"] = str(spare)
                key = str(spare)
            seen.add(key)
            kept.append(item)
            counts[str(item.get("obj_type"))] += 1
        output.append({"frame_id": frame_id, "labels": kept})
    # 【改动】2026-09-21 用户决定：Car/Truck 冲突一律按 Car 算。车链合并成一条后处理
    # 之后，冲突在链内就用类别优先级仲裁掉了，所以旧的「Car 被 Truck 覆盖 >= 阈值
    # 就删掉整条 Car 轨迹」规则停用（函数体保留在下面，需要时把下面两行恢复即可）。
    # overlapping, overlap_stats = _drop_cars_covered_by_trucks(
    #     output, car_truck_cover_threshold)
    # if overlap_stats.get("enabled"):
    #     counts = Counter()
    #     for frame in overlapping:
    #         for item in frame.get("labels", []):
    #             counts[str(item.get("obj_type"))] += 1
    #     output = overlapping
    overlap_stats = {"enabled": False,
                     "reason": "disabled_2026-09-21_car_priority"}
    missing = {name: sorted(set(frame_ids) - set(chain_labels.get(name, {})))
               for name in order}
    return output, {"frames": len(output), "labels": dict(counts),
                    "total": int(sum(counts.values())),
                    "id_collisions": collisions,
                    "car_truck_overlap": overlap_stats,
                    "chains": {name: len(chain_labels.get(name, {})) for name in order},
                    "frames_missing_per_chain": {k: len(v) for k, v in missing.items()}}


def run_clip(python: Path, clip: Path, output_root: Path, *, overwrite: bool,
             export_sust: bool = True, in_place: bool = False,
             drop_vis_below: float,
             score_threshold: float | None,
             short_track_max_frames: int,
             noncar_cfg: Path = NONCAR_CFG,
             noncar_ckpt: Path = NONCAR_CKPT,
             output_tag: str = "",
             raw_score_threshold: float = 0.2,   # 【改动】0.3 -> 0.2
             class_score_thresholds: Dict[str, float] | None = None,
             pedestrian_max_distance: float = 15.0,   # 【改动】20 -> 15
             nonmotorized_max_distance: float = 60.0,
             sparsity_max_points: int = 10,
             nonmotorized_min_net_displacement: float = 15.0,
             chains: tuple = DEFAULT_CHAINS,          # 【改动】car,truck,vru
             truck_cfg: Path = TRUCK_CFG,
             truck_ckpt: Path = TRUCK_CKPT,
             truck_raw_threshold: float = 0.4,
             # 【改动】2026-09-20 Truck 链检测器与货车/挂车规则
             truck_detector: str = "bevfusion",
             truck_detector_mode: str = "lidar",     # 【改动】fusion | lidar
             # 【改动】Car / VRU 链也可换成 BEVFusion 检测（复用同一份 raw json，只推理一次）
             vru_detector: str = "voxelnext",        # voxelnext | bevfusion
             bev_raw_threshold: float = 0.1,         # 共享 BEVFusion raw json 的分数门槛
             bev_raw_dir: Path | None = None,        # 【改动】复用已生成好的 raw json 目录（不再重新推理）
             output_suffix: str = "",                # 非空则输出名 = <clip名><后缀>（不套 _pre）
             link_only: bool = False,                # 输出目录只放软链 + label（不 copytree）
             trailer_rules: bool = True,
             trailer_score_threshold: float = 0.25,
             trailer_dup_iom: float = 0.70,
             trailer_dup_iou: float = 0.50,
             trailer_merge_iou: float = 0.05,
             # 【改动】2026-09-21：标注侧没有 Trailer 类别 -> 默认并成 Truck
             trailer_policy: str = "to-truck",
             vru_cfg: Path = VRU_CFG,
             vru_ckpt: Path = VRU_CKPT,
             vru_raw_threshold: float = 0.3,
             keep_chain_labels: bool = False,
             # 已停用（见 _merge_chain_labels：Car 优先接管 Car/Truck 冲突）
             car_truck_cover_threshold: float = 0.5,
             # 【改动】2026-09-21 Car+Truck 合并成一条后处理（共享动静态区域）；
             # --no-car-truck-merged 可回退到原来的两条链，方便 A/B。
             # 【改动】车链过程诊断（槽位/动态区域/…）默认不落盘；调试时才写进输出目录
             keep_vehicle_diagnostics: bool = False,
             static_rigid: bool = False) -> Dict[str, Any]:
    base = clip.name
    tag = output_tag.strip("_-")
    # 【改动】已经是 <clip>_pre 的输入 -> output_name == base：重跑就地覆盖，
    # 不改名、不生成 <clip>_pre_pre（见 _pre_clip_name）。
    output_name = _pre_clip_name(base, tag, output_suffix)
    destination: Path | None = None
    # 【改动】重跑已标注的 clip（名字本身就是目标输出名，例如 *_pre）：就地覆盖，不改名、
    # 更不能 rmtree 输入目录（否则把输入删了）。`--include-pre` 收进来的就是这种。
    rerun_in_place = bool(in_place and output_name == base)
    if in_place:
        destination = clip if rerun_in_place else clip.parent / output_name
        if not rerun_in_place and destination.exists():
            if not overwrite:
                raise RuntimeError(
                    f"output exists, pass --overwrite: {destination}")
            shutil.rmtree(destination)
        elif rerun_in_place and not overwrite:
            raise RuntimeError(
                f"重跑已标注的 clip 要带 --overwrite（会覆盖 {clip}/label）: {clip}")
        if rerun_in_place:
            _print(f"{base}: 就地重跑（已标注，覆盖 label/ 与分链标签）")
    elif export_sust:
        destination = output_root / output_name
        if destination.exists():
            if not overwrite:
                raise RuntimeError(
                    f"output exists, pass --overwrite: {destination}")
            shutil.rmtree(destination)

    selected = [name for name in ("car", "truck", "vru", "noncar") if name in set(chains)]
    if not selected:
        raise RuntimeError("--chains 至少选一条")
    # 【改动】车链是否可用：chains 同时含 car+truck、且 truck 检测器是 BEVFusion。
    # --car-detector auto（默认）在这里解析：能用就用 BEVFusion（= 车链），否则用 Waymo。
    # 【改动】2026-09-22 旧单 Car 链（Waymo Car 头 / hybrid_expD_car）已删除：
    # Car 一律走合并车链（BEVFusion Car+Truck 一次推理 + 共享跟踪），所以 car 必须带 truck。
    if "car" in selected and "truck" not in selected:
        selected.append("truck")
        selected.sort(key=("car", "truck", "vru", "noncar").index)
        _print(f"{base}: Car 只走合并车链（BEVFusion），自动补上 truck")
    merged_vehicle = bool("car" in selected and "truck" in selected
                          and str(truck_detector) == "bevfusion")
    if "car" in selected and not merged_vehicle:
        raise RuntimeError(
            "Car 只能走合并车链（BEVFusion Car+Truck）：请用 --truck-detector bevfusion")
    chain_labels: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    chain_stats: Dict[str, Any] = {}
    merged: List[Dict[str, Any]] | None = None
    merge_diag: Dict[str, Any] | None = None
    timings: Dict[str, float] = {}
    # 车链诊断（在 work 临时目录里，跑完要落到输出目录；直接留 dict）
    vehicle_diag_data: Dict[str, Any] | None = None
    clip_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"hybrid_{base}_") as temp:
        work = Path(temp)
        total = len(selected)
        step = 0
        bev_raw: Path | None = None
        if "bevfusion" in (str(truck_detector), str(vru_detector)):
            # 【改动】三条链共用一份 BEVFusion 原始检测（只推理一次）；
            # 给了 bev_raw_dir 就直接用那份（保证与留档的原始检测是同一份）
            if bev_raw_dir is not None:
                bev_raw = Path(bev_raw_dir) / f"{clip.name}_raw.json"
                if not bev_raw.is_file():
                    raise RuntimeError(f"bev_raw_dir 里缺少 {bev_raw.name}: {bev_raw_dir}")
                _print(f"{base}: 复用 BEVFusion raw json -> {bev_raw}")
            else:
                bev_raw = _run_raw_bevfusion(python, clip, work / "bev_raw",
                                             bev_raw_threshold, truck_detector_mode)
                _print(f"{base}: BEVFusion raw json（三链共用）-> {bev_raw.name}")
        if merged_vehicle:
            # 【改动】2026-09-21 车链：Car 与 Truck 共用一份 BEVFusion 检测 + 一次后处理
            # （静态槽位 / 动态区域 / 重跟踪 / ID 继承只算一次，类别冲突按 Car 优先）
            step += 1
            _print(f"{base}: {step}/{total} 车链（Car+Truck 合并后处理：一次跟踪 + 一次动态区域）")
            _t = time.monotonic()
            if bev_raw is None:
                bev_raw = _run_raw_bevfusion(python, clip, work / "bev_raw",
                                             bev_raw_threshold, truck_detector_mode)
            vehicle = run_vehicle_pass(
                bev_raw, clip, work / "vehicle", python,
                diagnostics_path=(work / "vehicle_diagnostics.json"
                                  if keep_vehicle_diagnostics else None),
                trailer_rules=bool(trailer_rules),
                trailer_dup_iom=float(trailer_dup_iom),
                trailer_dup_iou=float(trailer_dup_iou),
                trailer_merge_iou=float(trailer_merge_iou),
                trailer_policy=str(trailer_policy),
                static_rigid=bool(static_rigid),
                class_score_thresholds={
                    "Car": 0.2,
                    "Truck": 0.2,
                    "Trailer": float(trailer_score_threshold)},
                truck_short_track_max_frames=int(short_track_max_frames) or 4,
            )
            elapsed = round(time.monotonic() - _t, 1)
            timings["vehicle"] = elapsed
            timings["car"] = elapsed
            timings["truck"] = 0.0
            chain_labels["car"] = _frames_to_labels(vehicle["car_frames"], 0)
            chain_labels["truck"] = _frames_to_labels(vehicle["truck_frames"],
                                                      TRUCK_ID_OFFSET)
            vd = vehicle["diagnostics"]
            vehicle_diag_data = vd
            chain_stats["car"] = {
                "pipeline": "vehicle_pass(car)",
                "detector": "bevfusion",
                "final_detections": vd.get("car_detections"),
                "frames": len(chain_labels["car"]),
                "shared_region": "一次（Car+Truck 并集）"}
            chain_stats["truck"] = {
                "pipeline": "vehicle_pass(truck)",
                "detector": str(truck_detector),
                "trailer_policy": str(trailer_policy),
                "final_detections": vd.get("truck_detections"),
                "frames": len(chain_labels["truck"])}
        if "truck" in selected and not merged_vehicle:
            step += 1
            if truck_detector == "bevfusion":
                _print(f"{base}: {step}/{total} Truck 链（BEVFusion "
                       f"{'纯雷达' if truck_detector_mode == 'lidar' else 'C+L'} 官方20ep + 货车/挂车规则）")
                _t = time.monotonic()
                raw = bev_raw if bev_raw is not None else _run_raw_bevfusion(
                    python, clip, work / "truck_raw", truck_raw_threshold, truck_detector_mode)
            else:
                _print(f"{base}: {step}/{total} Truck 链（{truck_ckpt.name}）")
                _t = time.monotonic()
                raw = _run_raw(python, clip, truck_cfg, truck_ckpt,
                               work / "truck_raw", "truck", truck_raw_threshold)
            out = work / "truck.json"
            diag_path = work / "truck_diagnostics.json"
            result = run_expd_truck(
                raw, clip, out, diag_path,
                class_score_thresholds={"Truck": 0.2,   # 【改动】0.4 -> 0.2
                                        "Trailer": float(trailer_score_threshold)},
                trailer_rules=bool(trailer_rules),
                trailer_dup_iom=float(trailer_dup_iom),
                trailer_dup_iou=float(trailer_dup_iou),
                trailer_merge_iou=float(trailer_merge_iou),
                trailer_policy=str(trailer_policy))
            frames = json.loads(out.read_text(encoding="utf-8"))
            timings["truck"] = round(time.monotonic() - _t, 1)
            chain_labels["truck"] = _frames_to_labels(frames, TRUCK_ID_OFFSET)
            chain_stats["truck"] = {
                "detector": str(truck_detector),
                "detector_mode": str(truck_detector_mode),
                "checkpoint": str(truck_ckpt), "config": str(truck_cfg),
                "raw_score_threshold": float(truck_raw_threshold),
                "trailer_rules": bool(trailer_rules),
                "trailer_score_threshold": float(trailer_score_threshold),
                "final_detections": (result or {}).get("final_detections"),
                "frames": len(chain_labels["truck"])}
        if "vru" in selected:
            step += 1
            vru_src = ("BEVFusion ped/bicycle/motorcycle 头" if vru_detector == "bevfusion"
                       else vru_ckpt.name)
            _print(f"{base}: {step}/{total} VRU 链（{vru_src}: Pedestrian + Nonmotorized_vehicle）")
            _t = time.monotonic()
            raw = bev_raw if vru_detector == "bevfusion" else _run_raw(
                python, clip, vru_cfg, vru_ckpt, work / "vru_raw", "vru", vru_raw_threshold)
            out = work / "vru.json"
            diag_path = work / "vru_diagnostics.json"
            result = run_expd_vru(raw, clip, out, diag_path)
            frames = json.loads(out.read_text(encoding="utf-8"))
            timings["vru"] = round(time.monotonic() - _t, 1)
            chain_labels["vru"] = _frames_to_labels(frames, VRU_ID_OFFSET)
            chain_stats["vru"] = {
                "detector": str(vru_detector),
                "checkpoint": str(vru_ckpt), "config": str(vru_cfg),
                "raw_score_threshold": float(vru_raw_threshold),
                "final_detections": (result or {}).get("final_detections"),
                "frames": len(chain_labels["vru"])}
        if "noncar" in selected:
            step += 1
            _t = time.monotonic()
            _print(f"{base}: {step}/{total} 旧五类非车链（{noncar_ckpt.name}）")
            expd_raw = _run_raw(python, clip, noncar_cfg, noncar_ckpt,
                                work / "expd", "expd", raw_score_threshold)
            expd_json = work / "expd_noncar.json"
            expd_diag = work / "expd_noncar_diagnostics.json"
            expd_result = run_expd_noncar(
                expd_raw, clip, expd_json, expd_diag,
                visibility_min_ratio=drop_vis_below,
                short_track_max_frames=short_track_max_frames,
                score_threshold=score_threshold,
                class_score_thresholds=class_score_thresholds,
                pedestrian_max_distance=pedestrian_max_distance,
                nonmotorized_max_distance=nonmotorized_max_distance,
                sparsity_max_points=sparsity_max_points,
                nonmotorized_min_net_displacement=(
                    nonmotorized_min_net_displacement),
            )
            expd_frames = json.loads(expd_json.read_text(encoding="utf-8"))
            timings["noncar"] = round(time.monotonic() - _t, 1)
            chain_stats["noncar"] = {
                "checkpoint": str(noncar_ckpt), "config": str(noncar_cfg),
                "final_detections": (expd_result or {}).get("final_detections")}
            chain_labels["noncar"] = _frames_to_labels(expd_frames, 0)
            if "car" in chain_labels and len(selected) == 2:
                # 旧的 Car + 五类非车 两链模式：沿用原有 Car/非车 互斥吸收逻辑
                merged, merge_diag = merge_label_frames(
                    chain_labels["car"], expd_frames)
        _t = time.monotonic()
        if merged is None:
            merged, merge_diag = _merge_chain_labels(
                chain_labels, selected,
                car_truck_cover_threshold=car_truck_cover_threshold)
        merged_label_count = sum(len(frame["labels"]) for frame in merged)
        timings["merge"] = round(time.monotonic() - _t, 1)
        merge_diag = dict(merge_diag or {})
        merge_diag["chain_stats"] = chain_stats

    _t = time.monotonic()
    if link_only and destination is not None:
        # 【改动】输出目录只放软链 + label：输入 clip（可能已经是 <clip>_pre）保持不动，
        # 也不复制 image/lidar（SUST 能直接识别该目录）。
        if destination.exists():
            if not overwrite:
                raise RuntimeError(f"output exists, pass --overwrite: {destination}")
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        for sub in ("image", "lidar", "transforms", "readme.json"):
            src = clip / sub
            if src.exists():
                (destination / sub).symlink_to(src.resolve())
        labels = _write_labels(merged, destination)
    elif in_place and rerun_in_place:
        # 重跑已标注的 clip：目录名不变，只把 label/ 整个重写（_write_labels 会先 rmtree）
        labels = _write_labels(merged, destination)
    elif in_place:
        # 端到端原地模式：把输入 clip 改名为 <clip>_pre，再把标签写进去，
        # 不额外保留一份 raw，也不往 SUST 拷贝。
        try:
            clip.rename(destination)
            labels = _write_labels(merged, destination)
        except Exception:
            if destination is not None and destination.exists() and not clip.exists():
                destination.rename(clip)
            raise
    elif export_sust:
        shutil.copytree(clip, destination)
        try:
            labels = _write_labels(merged, destination)
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise
    else:
        # 只跑链路、不落盘：临时结果随上面的 TemporaryDirectory 清理。
        labels = merged_label_count
        destination = None
    timings["export"] = round(time.monotonic() - _t, 1)
    timings["total_without_export"] = round(
        timings.get("car", 0.0) + timings.get("truck", 0.0)
        + timings.get("vru", 0.0) + timings.get("noncar", 0.0)
        + timings.get("merge", 0.0), 1)
    timings["total"] = round(time.monotonic() - clip_start, 1)
    _print(f"{base}: 计时 " + ", ".join(
        f"{key}={value}s" for key, value in timings.items()))
    # 【改动】车链诊断写到输出目录（临时 work 目录结束后会被删，批跑时就看不到了）
    # 【改动】过程数据默认不写进产出目录；--keep-vehicle-diagnostics 才留（调试用）
    if (keep_vehicle_diagnostics and vehicle_diag_data is not None
            and destination is not None and destination.exists()):
        (destination / "vehicle_pass_diagnostics.json").write_text(
            json.dumps(vehicle_diag_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    if keep_chain_labels and destination is not None and destination.exists():
        for name, subdir in (("car", "label_car"), ("truck", "label_truck"),
                             ("vru", "label_vru")):      # 【改动】车链也留 Car 单支
            items = chain_labels.get(name) or {}
            if not items:
                continue
            target = destination / subdir
            if target.exists():      # 【改动】重跑时先清掉上一轮的分链标签
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            for frame_id, frame_labels in items.items():
                (target / f"{frame_id}.json").write_text(
                    json.dumps(frame_labels, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
        _print(f"{base}: 已另存 label_truck/ label_vru/ 供对比")
    return {
        "input_clip": str(clip),
        "final_clip": str(destination) if destination is not None else None,
        "labels": labels,
        "chains": chain_stats,
        "timings": timings,
        "merge": merge_diag,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", nargs="?", type=Path,
                        default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-install", action="store_true",
                        help="require an existing CUDA/OpenPCDet environment")
    parser.add_argument("--drop-vis-below", type=float, default=0.05)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--short-track-max-frames", type=int, default=4)
    parser.add_argument("--noncar-raw-threshold", type=float,
                        help="non-Car raw inference score cutoff "
                             "(default: min of class thresholds)")
    parser.add_argument("--truck-score-threshold", type=float,
                        help="default 0.4 when --score-threshold is unset")
    parser.add_argument("--bus-score-threshold", type=float,
                        help="default 0.4 when --score-threshold is unset")
    parser.add_argument("--pedestrian-score-threshold", type=float,
                        help="default 0.2 when --score-threshold is unset")  # 【改动】0.15 -> 0.2
    parser.add_argument("--nonmotorized-score-threshold", type=float,
                        help="default 0.2 when --score-threshold is unset")
    parser.add_argument("--pedestrian-max-distance", type=float, default=15.0)  # 【改动】20 -> 15
    parser.add_argument("--nonmotorized-max-distance", type=float, default=60.0)
    parser.add_argument("--sparsity-max-points", type=int, default=10)
    parser.add_argument("--nonmotorized-min-net-displacement",
                        type=float, default=15.0,
                        help="drop NMV tracks whose world-frame XY net "
                             "displacement (first to last usable center) is "
                             "not greater than this")
    parser.add_argument("--noncar-cfg", type=Path, default=NONCAR_CFG,
                        help="non-Car inference config (default: expD config)")
    parser.add_argument("--noncar-ckpt", type=Path, default=NONCAR_CKPT,
                        help="non-Car checkpoint (default: models/vod_2cls_ft_e12.pth)")
    parser.add_argument("--chains", type=str, default=",".join(DEFAULT_CHAINS),
                        help="要跑的链，逗号分隔；默认 car+truck 合成一条车链跑完再跑 vru；"
                             "可选 car,truck,vru,noncar（noncar=旧五类单链）。"
                             "只选 car 或只选 truck 时走各自的单链（调试用）")
    parser.add_argument("--truck-cfg", type=Path, default=TRUCK_CFG)
    parser.add_argument("--truck-ckpt", type=Path, default=TRUCK_CKPT)
    parser.add_argument("--truck-raw-threshold", type=float, default=0.1)
    # 【改动】2026-09-20 Truck 链检测器（BEVFusion / 旧 VoxelNeXt）与货车/挂车规则
    parser.add_argument("--include-pre", action="store_true",
                        help="批量遍历父目录时把已经预标过的 <clip>_pre 也收进来：配 --in-place 时"
                             "**就地覆盖重跑**（目录名不变、只重写 label/，不会生成 <clip>_pre_pre）；"
                             "不加则跳过所有 *_pre。直接点名一个 <clip>_pre 目录时不用这个开关也收。"
                             "注意 X 与 X_pre 同时存在时只跑 X_pre")
    parser.add_argument("--vru-detector", choices=["voxelnext", "bevfusion"], default="voxelnext",
                        help="VRU 链检测器：voxelnext（默认）或 bevfusion（ped/bicycle/motorcycle 头）")
    parser.add_argument("--bev-raw-dir", type=Path, default=None,
                        help="复用该目录下 <clip名>_raw.json（不重新跑 BEVFusion 推理）")
    parser.add_argument("--bev-raw-threshold", type=float, default=0.1,
                        help="三链共享的 BEVFusion raw json 分数门槛（链内阈值另外把关）")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="输出名 = <输入clip名><后缀>（如 _bev）；默认仍按 <clip>_pre / <clip>_<tag>_pre")
    parser.add_argument("--link-only", action="store_true",
                        help="输出目录只放 image/lidar/transforms 软链 + label（不复制数据、不改名输入）")
    parser.add_argument("--truck-detector-mode", choices=["lidar", "fusion"], default="lidar",
                        help="lidar=纯雷达 BEVFusion（默认，快）；fusion=C+L（读相机图）")
    parser.add_argument("--truck-detector", choices=["bevfusion", "voxelnext"],
                        default="bevfusion",
                        help="bevfusion: pipeline/step1_bevfusion_truck.py（默认）；"
                             "voxelnext: 旧 VoxelNeXt truckB 权重")
    parser.add_argument("--no-trailer-rules", action="store_true",
                        help="关掉挂车去重/并集与轨迹级 Truck 统一")
    parser.add_argument("--trailer-score-threshold", type=float, default=0.25)
    parser.add_argument("--trailer-dup-iom", type=float, default=0.70)
    parser.add_argument("--trailer-dup-iou", type=float, default=0.50)
    parser.add_argument("--trailer-merge-iou", type=float, default=0.05)
    parser.add_argument("--static-rigid", action="store_true",
                        help="【改动】静态 Car 轨迹叠帧拟一个刚性 box 固定在世界系（默认关）")
    parser.add_argument("--keep-vehicle-diagnostics", action="store_true",
                        help="【改动】把车链过程诊断 vehicle_pass_diagnostics.json 写进输出 clip"
                             "（默认不写，只调试用）")
    parser.add_argument("--trailer-policy", choices=["keep", "to-truck"],
                        default="to-truck",
                        help="keep: 纯挂车轨迹保留 Trailer；to-truck（默认）: 一律并成 Truck")
    parser.add_argument("--vru-cfg", type=Path, default=VRU_CFG)
    parser.add_argument("--vru-ckpt", type=Path, default=VRU_CKPT)
    parser.add_argument("--vru-raw-threshold", type=float, default=0.3)
    parser.add_argument("--car-truck-cover-threshold", type=float, default=0.5,
                        help="Car 被 Truck 覆盖的面积 / Car 面积 达到该值就删掉这条 "
                             "Car 轨迹的全部帧（0 关闭）")
    parser.add_argument("--keep-chain-labels", action="store_true",
                        help="额外把 label_truck/ label_vru/ 写进输出 clip")
    parser.add_argument("--output-tag", type=str, default="",
                        help="insert a tag before _pre in the exported clip "
                             "name, e.g. vod_e12 -> <clip>_vod_e12_pre")
    export_group = parser.add_mutually_exclusive_group()
    export_group.add_argument(
        "--export-sust", dest="export_mode", action="store_const",
        const="sust",
        help="write the merged <clip>_pre into output_root")
    export_group.add_argument(
        "--no-export-sust", dest="export_mode", action="store_const",
        const="none",
        help="run the chain without writing the merged clip")
    export_group.add_argument(
        "--in-place", dest="export_mode", action="store_const",
        const="in_place",
        help="rename each input clip to <clip>_pre in place; "
             "no SUST copy and no extra raw copy. An input already named "
             "<clip>_pre stays as-is and only label/ is rewritten "
             "(never <clip>_pre_pre)")
    parser.set_defaults(export_mode="sust")
    args = parser.parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    noncar_cfg = args.noncar_cfg.expanduser().resolve()
    noncar_ckpt = args.noncar_ckpt.expanduser().resolve()
    output_tag = args.output_tag.strip("_-")
    export_sust = args.export_mode == "sust"
    in_place = args.export_mode == "in_place"
    if export_sust and input_root == output_root:
        raise RuntimeError("input_root and output_root must differ")

    class_names = ("Truck", "Bus", "Pedestrian", "Nonmotorized_vehicle")
    explicit = {
        "Truck": args.truck_score_threshold,
        "Bus": args.bus_score_threshold,
        "Pedestrian": args.pedestrian_score_threshold,
        "Nonmotorized_vehicle": args.nonmotorized_score_threshold,
    }
    if args.score_threshold is not None:
        base = {name: float(args.score_threshold) for name in class_names}
    else:
        base = {
            "Truck": 0.4,
            "Bus": 0.4,
            # Higher-recall defaults for the VOD-finetuned non-Car heads.
            "Pedestrian": 0.2,   # 【改动】0.15 -> 0.2
            "Nonmotorized_vehicle": 0.2,
        }
    class_thresholds: Dict[str, float] = {
        name: float(explicit[name]) if explicit[name] is not None else base[name]
        for name in class_names
    }
    if args.noncar_raw_threshold is not None:
        noncar_raw_threshold = float(args.noncar_raw_threshold)
    elif any(value is not None for value in explicit.values()):
        noncar_raw_threshold = min(class_thresholds.values())
    elif args.score_threshold is not None:
        noncar_raw_threshold = (0.1 if args.score_threshold < 0.3 else 0.3)
    else:
        noncar_raw_threshold = 0.3
    noncar_raw_threshold = min(noncar_raw_threshold,
                              min(class_thresholds.values()))
    _print(f"non-Car raw threshold={noncar_raw_threshold:.3f}, "
           f"class thresholds={class_thresholds}, "
           f"pedestrian_max_distance={args.pedestrian_max_distance}, "
           f"nonmotorized_max_distance={args.nonmotorized_max_distance}, "
           f"sparsity_max_points={args.sparsity_max_points}, "
           f"nonmotorized_min_net_displacement="
           f"{args.nonmotorized_min_net_displacement}")
    chains = tuple(name.strip() for name in args.chains.split(",") if name.strip())
    unknown = [name for name in chains if name not in ("car", "truck", "vru", "noncar")]
    if unknown:
        raise RuntimeError(f"--chains 里有未知链: {unknown}")
    truck_cfg = args.truck_cfg.expanduser().resolve()
    truck_ckpt = args.truck_ckpt.expanduser().resolve()
    vru_cfg = args.vru_cfg.expanduser().resolve()
    vru_ckpt = args.vru_ckpt.expanduser().resolve()
    if "truck" in chains:
        _validate_weight(truck_ckpt)
        if not truck_cfg.is_file():
            raise RuntimeError(f"config not found: {truck_cfg}")
    if "vru" in chains:
        _validate_weight(vru_ckpt)
        if not vru_cfg.is_file():
            raise RuntimeError(f"config not found: {vru_cfg}")
    # 【改动】把「实际跑的是哪两个检测器」一次说清楚：只看 ckpt 名字容易误解
    # （车链下 car/truck 都用 BEVFusion，truck_ckpt 只在 --no-car-truck-merged 时才用）
    chains_set = set(str(c).strip() for c in chains) if not isinstance(chains, str) \
        else set(str(c).strip() for c in chains.split(",") if c.strip())
    vehicle_chain = ({"car", "truck"} <= chains_set
                     and str(args.truck_detector) == "bevfusion")
    if vehicle_chain:
        weights = ("models/bevfusion_mmdet3d_lidarcam.pth"
                   if args.truck_detector_mode == "fusion"
                   else "models/bevfusion_mmdet3d_lidaronly.pth")
        _print(f"chains={chains} | 车链 Car+Truck: detector=BEVFusion "
               f"mode={args.truck_detector_mode} weights={weights} "
               f"(trailer_rules={not args.no_trailer_rules}, policy={args.trailer_policy}) "
               f"| VRU: detector={args.vru_detector} weights={vru_ckpt.name}")
    else:
        _print(f"chains={chains}  truck={truck_ckpt.name}  vru={vru_ckpt.name}")
        if "truck" in chains_set:
            weights = (truck_ckpt.name if args.truck_detector == "voxelnext"
                       else "models/bevfusion_mmdet3d_lidaronly.pth")
            _print(f"truck chain(回退): detector={args.truck_detector} mode={args.truck_detector_mode} "
                   f"thresholds={{'Truck': 0.2, 'Trailer': {args.trailer_score_threshold}}} "
                   f"trailer_rules={not args.no_trailer_rules} weights={weights}")
    if "noncar" in chains:
        _validate_weight(noncar_ckpt)
        if not noncar_cfg.is_file():
            raise RuntimeError(f"config not found: {noncar_cfg}")

    python = None
    for candidate in _python_candidates(args.python):
        probe = _probe(candidate)
        _print(f"probe {candidate}: " + ("ready" if _healthy(probe) else "needs setup"))
        if _healthy(probe):
            python = candidate
            break
    if python is None:
        if args.skip_install:
            raise RuntimeError("no CUDA/OpenPCDet Python environment found")
        raise RuntimeError(
            "automatic installation is not enabled by hybrid runner; "
            "prepare the OpenPCDet environment and pass --python")

    if export_sust:
        output_root.mkdir(parents=True, exist_ok=True)
    clips = _collect_clips(input_root, include_pre=bool(args.include_pre))
    summaries = []
    for index, clip in enumerate(clips, 1):
        _print(f"clip [{index}/{len(clips)}]: {clip.name}")
        summaries.append(run_clip(
            python, clip, output_root, overwrite=args.overwrite,
            export_sust=export_sust, in_place=in_place,
            drop_vis_below=args.drop_vis_below,
            score_threshold=args.score_threshold,
            short_track_max_frames=args.short_track_max_frames,
            noncar_cfg=noncar_cfg,
            noncar_ckpt=noncar_ckpt,
            output_tag=output_tag,
            raw_score_threshold=noncar_raw_threshold,
            class_score_thresholds=class_thresholds,
            pedestrian_max_distance=args.pedestrian_max_distance,
            nonmotorized_max_distance=args.nonmotorized_max_distance,
            sparsity_max_points=args.sparsity_max_points,
            nonmotorized_min_net_displacement=(
                args.nonmotorized_min_net_displacement),
            chains=chains,
            truck_cfg=truck_cfg,
            truck_ckpt=truck_ckpt,
            truck_raw_threshold=args.truck_raw_threshold,
            truck_detector=args.truck_detector,
            truck_detector_mode=args.truck_detector_mode,
            vru_detector=args.vru_detector,
            bev_raw_threshold=args.bev_raw_threshold,
            bev_raw_dir=args.bev_raw_dir,
            output_suffix=args.output_suffix,
            link_only=args.link_only,
            trailer_rules=not args.no_trailer_rules,
            trailer_score_threshold=args.trailer_score_threshold,
            trailer_dup_iom=args.trailer_dup_iom,
            trailer_dup_iou=args.trailer_dup_iou,
            trailer_merge_iou=args.trailer_merge_iou,
            trailer_policy=args.trailer_policy,
            static_rigid=bool(args.static_rigid),
            keep_vehicle_diagnostics=bool(args.keep_vehicle_diagnostics),
            vru_cfg=vru_cfg,
            vru_ckpt=vru_ckpt,
            vru_raw_threshold=args.vru_raw_threshold,
            keep_chain_labels=args.keep_chain_labels,
            car_truck_cover_threshold=args.car_truck_cover_threshold,
        ))
    print(json.dumps({"clips": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        _print(f"ERROR: {exc}")
        raise SystemExit(1)
