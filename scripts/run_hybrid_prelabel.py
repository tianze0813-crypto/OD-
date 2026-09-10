#!/usr/bin/env python3
"""串行运行 main-Car + VOD-non-Car 并合并为一个 SUST 预标注结果。"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.hybrid_expD_noncar import run as run_expd_noncar
from pipeline.hybrid_main_car import run as run_main_car
from pipeline.hybrid_merge import merge_label_frames


DEFAULT_OUTPUT_ROOT = Path.home() / "SUSTechPOINTS" / "data"
NONCAR_CFG = ROOT / "models" / "voxelnext_fiveclass_nuscenes_infer.yaml"
# Final production non-Car weight: VOD 2-class fine-tune, epoch 12
# (checkpoint stores epoch 6).  The older expD_e8.pth remains available via
# --noncar-ckpt but is no longer the default.
NONCAR_CKPT = ROOT / "models" / "vod_2cls_ft_e12.pth"
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


def run_clip(python: Path, clip: Path, output_root: Path, *, overwrite: bool,
             export_sust: bool = True, drop_vis_below: float,
             score_threshold: float | None,
             short_track_max_frames: int,
             noncar_cfg: Path = NONCAR_CFG,
             noncar_ckpt: Path = NONCAR_CKPT,
             output_tag: str = "",
             raw_score_threshold: float = 0.3,
             class_score_thresholds: Dict[str, float] | None = None,
             pedestrian_max_distance: float = 20.0,
             nonmotorized_max_distance: float = 60.0,
             sparsity_max_points: int = 10,
             nonmotorized_min_net_displacement: float = 15.0) -> Dict[str, Any]:
    base = clip.name
    tag = output_tag.strip("_-")
    output_name = f"{base}_{tag}_pre" if tag else f"{base}_pre"
    destination: Path | None = None
    if export_sust:
        destination = output_root / output_name
        if destination.exists():
            if not overwrite:
                raise RuntimeError(
                    f"output exists, pass --overwrite: {destination}")
            shutil.rmtree(destination)

    with tempfile.TemporaryDirectory(prefix=f"hybrid_{base}_") as temp:
        work = Path(temp)
        _print(f"{base}: 1/2 main inference + Car chain")
        main_labels, main_result = run_main_car(
            python, clip, work / "main", overwrite=True)

        _print(f"{base}: 2/2 non-Car inference + chain "
               f"({noncar_ckpt.name})")
        _print(f"{base}: non-Car raw threshold={raw_score_threshold:.3f}, "
               f"class thresholds={class_score_thresholds or 'default'}")
        expd_raw = _run_raw(
            python, clip, noncar_cfg, noncar_ckpt, work / "expd", "expd",
            raw_score_threshold)
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
        merged, merge_diag = merge_label_frames(main_labels, expd_frames)
        merged_label_count = sum(len(frame["labels"]) for frame in merged)

    if export_sust:
        shutil.copytree(clip, destination)
        try:
            labels = _write_labels(merged, destination)
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise
    else:
        # 只跑链路、不落盘到 SUST：临时结果随上面的 TemporaryDirectory 清理。
        labels = merged_label_count
        destination = None
    return {
        "input_clip": str(clip),
        "final_clip": str(destination) if destination is not None else None,
        "labels": labels,
        "main": {
            "final_detections": main_result["final_detections"],
        },
        "expD": {
            "checkpoint": str(noncar_ckpt),
            "config": str(noncar_cfg),
            "raw_score_threshold": float(raw_score_threshold),
            "class_score_thresholds": dict(class_score_thresholds or {}),
            "pedestrian_max_distance": float(pedestrian_max_distance),
            "nonmotorized_max_distance": float(nonmotorized_max_distance),
            "sparsity_max_points": int(sparsity_max_points),
            "nonmotorized_min_net_displacement": float(
                nonmotorized_min_net_displacement),
            "raw_json": "temporary (cleaned after merge)",
            "final_detections": expd_result["final_detections"],
        },
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
                        help="default 0.15 when --score-threshold is unset")
    parser.add_argument("--nonmotorized-score-threshold", type=float,
                        help="default 0.2 when --score-threshold is unset")
    parser.add_argument("--pedestrian-max-distance", type=float, default=20.0)
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
    parser.add_argument("--output-tag", type=str, default="",
                        help="insert a tag before _pre in the exported clip "
                             "name, e.g. vod_e12 -> <clip>_vod_e12_pre")
    export_group = parser.add_mutually_exclusive_group()
    export_group.add_argument("--export-sust", dest="export_sust",
                              action="store_true",
                              help="write the merged <clip>_pre into output_root")
    export_group.add_argument("--no-export-sust", dest="export_sust",
                              action="store_false",
                              help="run the chain without writing to SUST")
    parser.set_defaults(export_sust=True)
    args = parser.parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    noncar_cfg = args.noncar_cfg.expanduser().resolve()
    noncar_ckpt = args.noncar_ckpt.expanduser().resolve()
    output_tag = args.output_tag.strip("_-")
    if input_root == output_root:
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
            "Pedestrian": 0.15,
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

    if args.export_sust:
        output_root.mkdir(parents=True, exist_ok=True)
    clips = _collect_clips(input_root)
    summaries = []
    for index, clip in enumerate(clips, 1):
        _print(f"clip [{index}/{len(clips)}]: {clip.name}")
        summaries.append(run_clip(
            python, clip, output_root, overwrite=args.overwrite,
            export_sust=args.export_sust,
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
        ))
    print(json.dumps({"clips": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        _print(f"ERROR: {exc}")
        raise SystemExit(1)
