#!/usr/bin/env python3
"""Execute the exact ``main`` branch chain and return its Car labels."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_CHAIN_ROOT = PROJECT_ROOT / "main_chain"


def _run(command: List[Any]) -> None:
    print("[hybrid-main] $ " + " ".join(str(value) for value in command),
          flush=True)
    subprocess.run([str(value) for value in command], check=True)


def _main_source() -> Path:
    """Return the vendored main snapshot committed in this hybrid branch."""
    required = (
        MAIN_CHAIN_ROOT / "run_end_to_end.py",
        MAIN_CHAIN_ROOT / "pipeline" / "step1_lidar_inference.py",
        MAIN_CHAIN_ROOT / "pipeline" / "step4_5_region_phase_retrack.py",
        MAIN_CHAIN_ROOT / "region" / "__init__.py",
        MAIN_CHAIN_ROOT / "models" / "vn_waymo_v2_4gpu_full_epoch10.pth",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "main_chain is incomplete; missing: " + ", ".join(missing))
    return MAIN_CHAIN_ROOT


def _read_main_labels(final_clip: Path) -> Dict[str, List[Dict[str, Any]]]:
    label_dir = final_clip / "label"
    if not label_dir.is_dir():
        raise RuntimeError(f"main chain did not produce label/: {final_clip}")
    labels: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(label_dir.glob("*.json"), key=lambda item: item.stem):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"main label file must contain a list: {path}")
        for label in value:
            if not isinstance(label, dict) or label.get("obj_type") != "Car":
                raise AssertionError(
                    f"main chain produced a non-Car label: {path}")
        labels[path.stem] = value
    return labels


def run(
        python: Path, clip: Path, work_root: Path, *, overwrite: bool = True,
        raw_json: Path | None = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Run ``main`` on an isolated copy of one clip and keep only its labels.

    【改动】raw_json 可选：给定后跳过 main_chain 自带的 Waymo step1，直接把这份
    raw json（lidar 系，schema 与本工程一致）喂给 main_chain 的 step2~step5，
    用于「换检测器重跑」的实验（main_chain 的 run_end_to_end.py 本来就支持 --raw-json）。
    """
    main_source = _main_source()
    main_input = work_root / "main_input" / clip.name
    shutil.copytree(clip, main_input)
    command = [
        python, main_source / "run_end_to_end.py",
        "--clip", main_input,
        "--inference-python", python,
        "--post-python", python,
    ]
    if overwrite:
        command.append("--overwrite")
    if raw_json is not None:      # 【改动】外部检测结果（跳过 Waymo step1）
        command += ["--raw-json", str(Path(raw_json).resolve())]
    _run(command)
    final_clip = main_input.with_name(clip.name + "_pre")
    labels = _read_main_labels(final_clip)
    return labels, {
        "pipeline": "main",
        "detector_raw_json": (str(Path(raw_json).resolve()) if raw_json is not None else None),
        "source_ref": "main",
        "source_clip_copy": str(main_input),
        "frames": len(labels),
        "car_detections": sum(len(value) for value in labels.values()),
        "final_detections": sum(len(value) for value in labels.values()),
        "output_classes": ["Car"],
    }
