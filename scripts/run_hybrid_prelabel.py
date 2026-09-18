#!/usr/bin/env python3
"""串行运行三条链并合并为一个 SUST 预标注结果（【改动】2026-09-18）。

接入顺序（用户指定）：**先 Car -> 再 Truck -> 最后 Pedestrian/Nonmotorized_vehicle**

  1. Car   : main_chain（Waymo Car + Step4.5），权重 main_chain/models/vn_waymo_v2_4gpu_full_epoch10.pth
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
from pipeline.hybrid_main_car import run as run_main_car
from pipeline.hybrid_merge import merge_label_frames


DEFAULT_OUTPUT_ROOT = Path.home() / "SUSTechPOINTS" / "data"
NONCAR_CFG = ROOT / "models" / "voxelnext_fiveclass_nuscenes_infer.yaml"
# Final production non-Car weight: VOD 2-class fine-tune, epoch 12
# (checkpoint stores epoch 6).  The older expD_e8.pth remains available via
# --noncar-ckpt but is no longer the default.
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


def _is_clip(path: Path) -> bool:
    lidar = path / "lidar" / "lidar_top"
    return path.is_dir() and lidar.is_dir() and any(lidar.glob("*.bin"))


def _collect_clips(input_root: Path) -> List[Path]:
    if not input_root.is_dir():
        raise RuntimeError(f"input directory does not exist: {input_root}")
    # Accept a single clip directory directly, or a parent holding many clips.
    if _is_clip(input_root):
        return [input_root.resolve()]
    clips = [path.resolve() for path in sorted(input_root.iterdir())
             if _is_clip(path) and not path.name.endswith("_pre")]
    if not clips:
        raise RuntimeError(
            f"no raw clips found under {input_root}; expected lidar/lidar_top/*.bin")
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
    overlapping, overlap_stats = _drop_cars_covered_by_trucks(
        output, car_truck_cover_threshold)
    if overlap_stats.get("enabled"):
        counts = Counter()
        for frame in overlapping:
            for item in frame.get("labels", []):
                counts[str(item.get("obj_type"))] += 1
        output = overlapping
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
             vru_cfg: Path = VRU_CFG,
             vru_ckpt: Path = VRU_CKPT,
             vru_raw_threshold: float = 0.3,
             keep_chain_labels: bool = False,
             car_truck_cover_threshold: float = 0.5) -> Dict[str, Any]:
    base = clip.name
    tag = output_tag.strip("_-")
    output_name = f"{base}_{tag}_pre" if tag else f"{base}_pre"
    destination: Path | None = None
    if in_place:
        destination = clip.parent / output_name
        if destination.exists():
            if not overwrite:
                raise RuntimeError(
                    f"output exists, pass --overwrite: {destination}")
            shutil.rmtree(destination)
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
    chain_labels: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    chain_stats: Dict[str, Any] = {}
    merged: List[Dict[str, Any]] | None = None
    merge_diag: Dict[str, Any] | None = None
    timings: Dict[str, float] = {}
    clip_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"hybrid_{base}_") as temp:
        work = Path(temp)
        total = len(selected)
        step = 0
        if "car" in selected:
            step += 1
            _print(f"{base}: {step}/{total} Car 链（main_chain: Waymo Car + Step4.5）")
            _t = time.monotonic()
            main_labels, main_result = run_main_car(
                python, clip, work / "main", overwrite=True)
            timings["car"] = round(time.monotonic() - _t, 1)
            chain_labels["car"] = {str(key): list(value)
                                   for key, value in main_labels.items()}
            chain_stats["car"] = {
                "checkpoint": "main_chain/models/vn_waymo_v2_4gpu_full_epoch10.pth",
                "final_detections": main_result.get("final_detections"),
                "frames": len(chain_labels["car"])}
        if "truck" in selected:
            step += 1
            _print(f"{base}: {step}/{total} Truck 链（{truck_ckpt.name}）")
            _t = time.monotonic()
            raw = _run_raw(python, clip, truck_cfg, truck_ckpt,
                           work / "truck_raw", "truck", truck_raw_threshold)
            out = work / "truck.json"
            diag_path = work / "truck_diagnostics.json"
            result = run_expd_truck(raw, clip, out, diag_path)
            frames = json.loads(out.read_text(encoding="utf-8"))
            timings["truck"] = round(time.monotonic() - _t, 1)
            chain_labels["truck"] = _frames_to_labels(frames, TRUCK_ID_OFFSET)
            chain_stats["truck"] = {
                "checkpoint": str(truck_ckpt), "config": str(truck_cfg),
                "raw_score_threshold": float(truck_raw_threshold),
                "final_detections": (result or {}).get("final_detections"),
                "frames": len(chain_labels["truck"])}
        if "vru" in selected:
            step += 1
            _print(f"{base}: {step}/{total} VRU 链（{vru_ckpt.name}: Pedestrian + Nonmotorized_vehicle）")
            _t = time.monotonic()
            raw = _run_raw(python, clip, vru_cfg, vru_ckpt,
                           work / "vru_raw", "vru", vru_raw_threshold)
            out = work / "vru.json"
            diag_path = work / "vru_diagnostics.json"
            result = run_expd_vru(raw, clip, out, diag_path)
            frames = json.loads(out.read_text(encoding="utf-8"))
            timings["vru"] = round(time.monotonic() - _t, 1)
            chain_labels["vru"] = _frames_to_labels(frames, VRU_ID_OFFSET)
            chain_stats["vru"] = {
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
    if in_place:
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
    if keep_chain_labels and destination is not None and destination.exists():
        for name, subdir in (("truck", "label_truck"), ("vru", "label_vru")):
            items = chain_labels.get(name) or {}
            if not items:
                continue
            target = destination / subdir
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
                        help="non-Car checkpoint (default: expD_e8.pth)")
    parser.add_argument("--chains", type=str, default=",".join(DEFAULT_CHAINS),
                        help="要跑的链，逗号分隔，按 car -> truck -> vru 顺序执行；"
                             "可选 car,truck,vru,noncar（noncar=旧五类单链）")
    parser.add_argument("--truck-cfg", type=Path, default=TRUCK_CFG)
    parser.add_argument("--truck-ckpt", type=Path, default=TRUCK_CKPT)
    parser.add_argument("--truck-raw-threshold", type=float, default=0.4)
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
             "no SUST copy and no extra raw copy")
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
    _print(f"chains={chains}  truck={truck_ckpt.name}  vru={vru_ckpt.name}")
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
    clips = _collect_clips(input_root)
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
