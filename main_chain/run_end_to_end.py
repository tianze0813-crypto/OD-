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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking import tracker_conservative as tracking


DEFAULT_INFERENCE_PYTHON = "/home/moga/miniconda3/envs/openpcdet/bin/python"
DEFAULT_POST_PYTHON = "/home/moga/miniconda3/envs/sustechpoints/bin/python"
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
    parser.add_argument("--export-sust", action="store_true",
                        help="把最终 pre clip 复制到 SUSTechPOINTS/data")
    parser.add_argument("--sust-root", type=Path, default=DEFAULT_SUST_ROOT)
    parser.add_argument("--final-suffix", type=str, default="_pre")
    parser.add_argument("--overwrite", action="store_true",
                        help="如果 <clip>_pre 已存在，先删除再生成")
    parser.add_argument("--sparsity-max-points", type=int, default=5,
                        help="step5: remove boxes containing this many points or fewer")
    parser.add_argument("--short-track-max-frames", type=int, default=3,
                        help="step5: remove tracks observed in this many frames or fewer")
    # Kept as a hidden compatibility flag; Step5 now always applies Car-only.
    parser.add_argument("--car-only", "--step6-car-only", dest="car_only",
                        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--drop-vis-below", type=float, default=0.05)
    args = parser.parse_args()

    clips = collect_clips(args)
    if args.raw_json and len(clips) != 1:
        raise SystemExit("--raw-json 仅支持单个 --clip")

    summaries = []
    with tempfile.TemporaryDirectory(prefix="fullchain_") as tmp:
        tmp_root = Path(tmp)
        step1_root = tmp_root / "step1"
        step2_root = tmp_root / "step2"
        step3_root = tmp_root / "step3"
        step4_root = tmp_root / "step4"
        step5_root = tmp_root / "step5"
        for path in (step1_root, step2_root, step3_root, step4_root, step5_root):
            path.mkdir(parents=True, exist_ok=True)

        for index, clip in enumerate(clips, 1):
            base = clip.name
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
                     "--score-thresh", args.score_thresh,
                     "--drop-vis-below", args.drop_vis_below])

            step2_json = step2_root / f"{base}_step2.json"
            step2_diag = step2_root / f"{base}_step2_diagnostics.json"
            run([args.post_python,
                 ROOT / "pipeline" / "step2_identity_class_filter_yaw.py",
                 "--in-json", raw_json, "--clip", clip,
                 "--out-json", step2_json,
                 "--diagnostics", step2_diag])

            step3_json = step3_root / f"{base}_step3.json"
            step3_diag = step3_root / f"{base}_step3_diagnostics.json"
            run([args.post_python,
                 ROOT / "pipeline" / "step3_car_box_fit.py",
                 "--step2-json", step2_json,
                 "--step2-diagnostics", step2_diag,
                 "--clip", clip, "--out-json", step3_json,
                 "--diagnostics", step3_diag])

            step4_json = step4_root / f"{base}_step4.json"
            step4_diag = step4_root / f"{base}_step4_diagnostics.json"
            run([args.post_python,
                 ROOT / "pipeline" / "step4_car_size_filter.py",
                 "--step3-json", step3_json,
                 "--out-json", step4_json,
                 "--diagnostics", step4_diag])

            step5_json = step5_root / f"{base}_step5.json"
            step5_diag = step5_root / f"{base}_step5_diagnostics.json"
            step5_cmd = [
                args.post_python,
                ROOT / "pipeline" / "step5_class_motion_filter.py",
                "--step4-json", step4_json, "--clip", clip,
                "--out-json", step5_json, "--diagnostics", step5_diag,
                "--sparsity-max-points", args.sparsity_max_points,
                "--short-track-max-frames", args.short_track_max_frames,
            ]
            run(step5_cmd)

            # Step5 already enforces the final Car-only policy.
            final_json = step5_json

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
                "input_clip": str(clip),
                "final_clip": str(final_clip),
                "labels": labels,
                "car_only": True,
                "large_car_tracks_relabelled": json.loads(
                    step4_diag.read_text(encoding="utf-8")
                )["large_car_tracks_relabelled"],
                "large_car_detections_relabelled": json.loads(
                    step4_diag.read_text(encoding="utf-8")
                )["large_car_detections_relabelled"],
                "sust_export": str(sust_dest) if sust_dest else None,
            })

    print("\n===== 端到端完成 =====")
    print(json.dumps({"clips": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
