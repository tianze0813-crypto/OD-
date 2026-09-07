#!/usr/bin/env python3
"""Execute the exact ``main`` branch chain and return its Car labels."""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(command: List[Any]) -> None:
    print("[hybrid-main] $ " + " ".join(str(value) for value in command),
          flush=True)
    subprocess.run([str(value) for value in command], check=True)


def _export_main_source(destination: Path) -> Path:
    """Materialize the repository's local ``main`` ref without changing HEAD."""
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / "main-source.tar"
    with archive.open("wb") as stream:
        subprocess.run(
            ["git", "archive", "--format=tar", "main"],
            cwd=PROJECT_ROOT, stdout=stream, check=True)
    with tarfile.open(archive, "r:") as tar:
        tar.extractall(destination)
    archive.unlink()
    return destination


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
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Run ``main`` on an isolated copy of one clip and keep only its labels."""
    main_source = _export_main_source(work_root / "main_source")
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
    _run(command)
    final_clip = main_input.with_name(clip.name + "_pre")
    labels = _read_main_labels(final_clip)
    return labels, {
        "pipeline": "main",
        "source_ref": "main",
        "source_clip_copy": str(main_input),
        "frames": len(labels),
        "car_detections": sum(len(value) for value in labels.values()),
        "output_classes": ["Car"],
    }
