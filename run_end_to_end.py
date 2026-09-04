#!/usr/bin/env python3
"""End-to-end pre-annotation pipeline.

Input:
    /path/to/<clip>          unlabelled SUST clip

Output:
    /path/to/<clip>_pre      the input clip renamed, with label/<frame>.json

Intermediate JSON files are written to a system temporary directory and are
removed automatically.  ``--export-sust`` only copies the final ``<clip>_pre``
into ``SUSTechPOINTS/data``.

Examples:
    python run_end_to_end.py --clip /path/to/scene_clip --export-sust
    python run_end_to_end.py --clip-dir /path/to/clips --export-sust
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking import tracker_conservative as tracking
from filtering.five_class_output import apply_five_class_output
from filtering.hard_filters import HardFilterConfig, apply_category_score_filter


def _first_existing(*candidates: Path) -> str:
    """Choose an explicit env override, then a known local conda install."""
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return sys.executable


_HOME = Path.home()
DEFAULT_INFERENCE_PYTHON = _first_existing(
    Path(os.environ["OPENPCDET_PYTHON"])
    if os.environ.get("OPENPCDET_PYTHON") else Path(""),
    _HOME / "miniconda3" / "envs" / "openpcdet" / "bin" / "python",
    _HOME / "anaconda3" / "envs" / "openpcdet" / "bin" / "python",
)
DEFAULT_POST_PYTHON = _first_existing(
    Path(os.environ["POSTPROCESS_PYTHON"])
    if os.environ.get("POSTPROCESS_PYTHON") else Path(""),
    Path(os.environ["OPENPCDET_PYTHON"])
    if os.environ.get("OPENPCDET_PYTHON") else Path(""),
    _HOME / "miniconda3" / "envs" / "openpcdet" / "bin" / "python",
    _HOME / "miniconda3" / "envs" / "sustechpoints" / "bin" / "python",
    _HOME / "anaconda3" / "envs" / "openpcdet" / "bin" / "python",
)
DEFAULT_SUST_ROOT = ROOT.parent / "SUSTechPOINTS" / "data"


def run(cmd):
    print(">> " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def is_clip_dir(path: Path) -> bool:
    lidar_top = path / "lidar" / "lidar_top"
    return path.is_dir() and lidar_top.is_dir() and any(lidar_top.glob("*.bin"))


def collect_clips(args):
    clips = [Path(c).resolve() for c in args.clip]
    for root in args.clip_dir:
        root = Path(root).resolve()
        if not root.is_dir():
            raise SystemExit(f"--clip-dir 不是目录：{root}")
        clips.extend(sorted(p for p in root.iterdir() if is_clip_dir(p)))
    if not clips:
        raise SystemExit("请至少给一个 --clip 或 --clip-dir")
    return clips


def replace_clip_copy(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_labels_only(frames, clip_dir: Path) -> int:
    """Write per-frame label JSON into an existing clip directory."""
    label_dir = clip_dir / "label"
    if label_dir.exists():
        shutil.rmtree(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for frame in frames:
        labels = [
            tracking.box_to_label(det)
            for det in frame.get("detections", [])
            if det.get("track_id") is not None
        ]
        count += len(labels)
        (label_dir / f"{frame['frame_id']}.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", action="append", default=[])
    parser.add_argument("--clip-dir", action="append", default=[])
    parser.add_argument("--raw-json", type=Path,
                        help="已有 step1 推理 JSON（仅支持单个 --clip）")
    parser.add_argument("--inference-python", type=Path,
                        default=Path(DEFAULT_INFERENCE_PYTHON))
    parser.add_argument("--post-python", type=Path,
                        default=Path(DEFAULT_POST_PYTHON))
    parser.add_argument("--cfg", type=Path,
                        default=ROOT / "models" / "voxelnext_fiveclass_nuscenes_infer.yaml",
                        help="Step1 OpenPCDet inference config")
    parser.add_argument("--ckpt", type=Path,
                        default=ROOT / "models" / "vn5_nuscenes_checkpoint_epoch_12.pth",
                        help="Step1 model checkpoint")
    parser.add_argument("--export-sust", action="store_true",
                        help="把最终 pre clip 复制到 SUSTechPOINTS/data")
    parser.add_argument("--sust-root", type=Path, default=DEFAULT_SUST_ROOT)
    parser.add_argument("--final-suffix", type=str, default="_pre")
    parser.add_argument("--overwrite", action="store_true",
                        help="如果 <clip>_pre 已存在，先删除再生成")
    # Retained as hidden compatibility flags. Five-class output does not run
    # the legacy Step5 point/lifecycle or Car-only filters.
    parser.add_argument("--sparsity-max-points", type=int, default=5,
                        help=argparse.SUPPRESS)
    parser.add_argument("--short-track-max-frames", type=int, default=3,
                        help=argparse.SUPPRESS)
    parser.add_argument("--car-only", "--step6-car-only", dest="car_only",
                        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--score-thresh", type=float, default=None,
        help="兼容参数：显式设置后覆盖全部类别阈值")
    parser.add_argument("--car-score-thresh", type=float, default=0.25)
    parser.add_argument("--truck-score-thresh", type=float, default=0.4)
    parser.add_argument("--bus-score-thresh", type=float, default=0.4)
    parser.add_argument("--pedestrian-score-thresh", type=float, default=0.3)
    parser.add_argument("--nonmotorized-score-thresh", type=float, default=0.3)
    parser.add_argument("--drop-vis-below", type=float, default=0.05)
    args = parser.parse_args()

    if args.score_thresh is not None:
        # Preserve the old all-class override for callers that still pass it.
        args.car_score_thresh = args.truck_score_thresh = args.bus_score_thresh = args.score_thresh
        args.pedestrian_score_thresh = args.nonmotorized_score_thresh = args.score_thresh
    step1_score_thresh = min(
        args.car_score_thresh, args.truck_score_thresh,
        args.bus_score_thresh, args.pedestrian_score_thresh,
        args.nonmotorized_score_thresh)
    fallback_score_thresh = (args.score_thresh
                             if args.score_thresh is not None else 0.3)
    score_config = HardFilterConfig(
        score_threshold=fallback_score_thresh,
        class_score_thresholds=(
            ("Car", args.car_score_thresh),
            ("Truck", args.truck_score_thresh),
            ("Bus", args.bus_score_thresh),
            ("Pedestrian", args.pedestrian_score_thresh),
            ("Nonmotorized_vehicle", args.nonmotorized_score_thresh),
        ),
    )

    clips = collect_clips(args)
    if args.raw_json and len(clips) != 1:
        raise SystemExit("--raw-json 仅支持单个 --clip")

    summaries = []
    with tempfile.TemporaryDirectory(prefix="fullchain_") as tmp:
        tmp_root = Path(tmp)
        step1_root = tmp_root / "step1"
        step2_root = tmp_root / "step2"
        step2_5_root = tmp_root / "step2_5"
        step3_root = tmp_root / "step3"
        final_root = tmp_root / "final"
        for path in (step1_root, step2_root, step2_5_root, step3_root, final_root):
            path.mkdir(parents=True, exist_ok=True)

        for index, clip in enumerate(clips, 1):
            base = clip.name
            input_clip_path = str(clip)
            final_clip = clip.with_name(base + args.final_suffix)
            print(f"\n===== [{index}/{len(clips)}] {base} =====", flush=True)

            if final_clip.exists():
                if args.overwrite:
                    shutil.rmtree(final_clip)
                else:
                    raise SystemExit(
                        f"输出已存在，加 --overwrite 覆盖：{final_clip}")

            raw_json = (Path(args.raw_json).resolve() if args.raw_json
                        else step1_root / f"{base}_raw.json")
            if not args.raw_json:
                run([args.inference_python,
                     ROOT / "pipeline" / "step1_lidar_inference.py",
                     "--clip", clip, "--work-root", step1_root,
                     "--cfg", args.cfg,
                     "--ckpt", args.ckpt,
                     "--score-thresh", step1_score_thresh,
                     "--drop-vis-below", args.drop_vis_below])

            # Step1 must run at the lowest category threshold.  Remove only
            # category-score noise here; all geometry/visibility filters stay
            # after ID assignment in Step2 and Step2.5.
            raw_frames = json.loads(raw_json.read_text(encoding="utf-8"))
            pre_step2_score_filter = apply_category_score_filter(
                raw_frames, score_config)
            raw_json.write_text(
                json.dumps(raw_frames, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            print(
                "pre-Step2 category score filter: "
                f"removed={pre_step2_score_filter['detections_removed']}",
                flush=True)

            step2_json = step2_root / f"{base}_step2.json"
            step2_diag = step2_root / f"{base}_step2_diagnostics.json"
            run([args.post_python,
                 ROOT / "pipeline" / "step2_identity.py",
                 "--in-json", raw_json, "--clip", clip,
                 "--out-json", step2_json,
                 "--diagnostics", step2_diag,
                 "--score-threshold", fallback_score_thresh,
                 "--car-score-threshold", args.car_score_thresh,
                 "--truck-score-threshold", args.truck_score_thresh,
                 "--bus-score-threshold", args.bus_score_thresh,
                 "--pedestrian-score-threshold", args.pedestrian_score_thresh,
                 "--nonmotorized-score-threshold", args.nonmotorized_score_thresh,
                 "--visibility-min-ratio", args.drop_vis_below,
                 "--sparsity-max-points", args.sparsity_max_points])

            step2_5_json = step2_5_root / f"{base}_step2_5.json"
            step2_5_diag = step2_5_root / f"{base}_step2_5_diagnostics.json"
            run([args.post_python,
                 ROOT / "pipeline" / "step2_5_class_correction.py",
                 "--step2-json", step2_json,
                 "--step2-diagnostics", step2_diag,
                 "--clip", clip, "--out-json", step2_5_json,
                 "--diagnostics", step2_5_diag,
                 "--score-threshold", fallback_score_thresh,
                 "--car-score-threshold", args.car_score_thresh,
                 "--truck-score-threshold", args.truck_score_thresh,
                 "--bus-score-threshold", args.bus_score_thresh,
                 "--pedestrian-score-threshold", args.pedestrian_score_thresh,
                 "--nonmotorized-score-threshold", args.nonmotorized_score_thresh,
                 "--visibility-min-ratio", args.drop_vis_below,
                 "--sparsity-max-points", args.sparsity_max_points,
                 "--min-lifecycle", args.short_track_max_frames])

            step3_json = step3_root / f"{base}_step3.json"
            step3_diag = step3_root / f"{base}_step3_diagnostics.json"
            run([args.post_python,
                 ROOT / "pipeline" / "step3_refinement.py",
                 "--step2-5-json", step2_5_json,
                 "--step2-5-diagnostics", step2_5_diag,
                 "--clip", clip, "--out-json", step3_json,
                 "--diagnostics", step3_diag])

            final_json = final_root / f"{base}_final.json"
            final_diag = final_root / f"{base}_final_diagnostics.json"
            step3_frames = json.loads(step3_json.read_text(encoding="utf-8"))
            coords = tracking.CoordinateProvider(clip)
            final_frames, final_result = apply_five_class_output(
                step3_frames, coords)
            final_json.write_text(
                json.dumps(final_frames, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            final_diag.write_text(
                json.dumps(final_result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")

            # No intermediate clip is stored.  The input clip itself is
            # renamed to <clip>_pre and only label/ is added.
            clip.rename(final_clip)
            try:
                labels = write_labels_only(
                    json.loads(final_json.read_text(encoding="utf-8")),
                    final_clip)
            except Exception:
                final_clip.rename(clip)
                raise

            sust_dest = None
            if args.export_sust:
                sust_dest = args.sust_root / final_clip.name
                replace_clip_copy(final_clip, sust_dest)
                print(f"SUST export: {sust_dest}", flush=True)

            summaries.append({
                "input_clip": input_clip_path,
                "final_clip": str(final_clip),
                "labels": labels,
                "target_classes": list(tracking.TARGET_CLASSES),
                "boxes_converted": final_result["boxes_converted"],
                "removed_by_reason": final_result["removed_by_reason"],
                "pre_step2_score_filter": pre_step2_score_filter,
                "sust_export": str(sust_dest) if sust_dest else None,
            })

    print("\n===== 端到端完成 =====")
    print(json.dumps({"clips": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
