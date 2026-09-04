#!/usr/bin/env python3
"""Install/check the runtime and run the five-class pre-label pipeline.

This launcher intentionally uses only the Python standard library so it can
run before the OpenPCDet environment exists.  It reuses a healthy existing
environment when possible; otherwise it installs the pinned runtime packages
and OpenPCDet into the selected Python environment.
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
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = Path.home() / "sust" / "data"
DEFAULT_OUTPUT_ROOT = Path.home() / "SUSTechPOINTS" / "data"
DEFAULT_RAW_ROOT = Path.home() / "sust" / "raw_inference"
DEFAULT_CFG = ROOT / "models" / "voxelnext_fiveclass_nuscenes_infer.yaml"
DEFAULT_CKPT = ROOT / "models" / "vn5_nuscenes_checkpoint_epoch_12.pth"
DEFAULT_ENV_NAME = "fiveclass-prelabel"
REQUIRED_MODULES = ("numpy", "scipy", "cv2", "PIL", "pandas", "av2",
                    "kornia", "yaml", "torch", "spconv", "pcdet")

# Curated aliases for the local checkpoints.  ``waymo`` has a three-class
# head layout and can only be run in raw inference mode; the other presets are
# compatible with the five-class full chain.
WEIGHT_PRESETS = {
    "default": (
        DEFAULT_CKPT,
        DEFAULT_CFG,
        "full",
    ),
    "epoch12": (
        DEFAULT_CKPT,
        DEFAULT_CFG,
        "full",
    ),
    "epoch15": (
        ROOT / "models" / "nusc_frozen20_epoch15.pth",
        DEFAULT_CFG,
        "full",
    ),
    "epoch17": (
        ROOT / "models" / "nusc_frozen20_epoch17.pth",
        DEFAULT_CFG,
        "full",
    ),
    "epoch20": (
        ROOT / "models" / "nusc_frozen20_epoch20.pth",
        DEFAULT_CFG,
        "full",
    ),
    "argo2": (
        ROOT / "models" / "argo2_protected_epoch6.pth",
        DEFAULT_CFG,
        "full",
    ),
    "waymo": (
        ROOT / "models" / "vn_waymo_v2_4gpu_full_epoch10.pth",
        ROOT / "models" / "voxelnext_v2_waymo_infer.yaml",
        "inference",
    ),
}


def _print(message: str) -> None:
    print(f"[five-class] {message}", flush=True)


def _run(command: list[str], *, check: bool = True,
         capture: bool = False) -> subprocess.CompletedProcess[str]:
    _print("$ " + " ".join(str(value) for value in command))
    return subprocess.run(
        [str(value) for value in command], check=check,
        text=True, capture_output=capture,
    )


def _print_weight_table() -> None:
    print("可用权重别名：")
    print(f"{'别名':<10} {'checkpoint':<40} {'适用模式':<10}")
    for alias, (ckpt, _cfg, mode) in WEIGHT_PRESETS.items():
        print(f"{alias:<10} {ckpt.name:<40} {mode:<10}")


def _resolve_weight(
    weight: str | None,
    ckpt: Path | None,
    cfg: Path | None,
) -> tuple[Path, Path, str]:
    """Resolve a curated weight alias or an explicit checkpoint.

    Returns ``(checkpoint, config, supported_mode)``.
    """
    if weight is None:
        selected_ckpt = (ckpt or DEFAULT_CKPT).expanduser().resolve()
        selected_cfg = (cfg or DEFAULT_CFG).expanduser().resolve()
        for _alias, (preset_ckpt, _preset_cfg, mode) in WEIGHT_PRESETS.items():
            if selected_ckpt.name == preset_ckpt.name and mode == "inference":
                return selected_ckpt, selected_cfg, mode
        return selected_ckpt, selected_cfg, "full"
    if ckpt is not None:
        raise RuntimeError("--weight and --ckpt are mutually exclusive")

    key = weight.lower()
    if Path(weight).suffix.lower() == ".pth":
        key = Path(weight).name.lower()
    for alias, (preset_ckpt, preset_cfg, mode) in WEIGHT_PRESETS.items():
        candidates = {
            alias.lower(),
            preset_ckpt.name.lower(),
            preset_ckpt.stem.lower(),
        }
        if key in candidates:
            return (
                preset_ckpt.expanduser().resolve(),
                (cfg or preset_cfg).expanduser().resolve(),
                mode,
            )
    available = ", ".join(WEIGHT_PRESETS)
    raise RuntimeError(f"unknown weight alias: {weight} (available: {available})")


def _probe(python: Path) -> dict[str, Any]:
    code = r'''
import importlib.util
import json
import sys
modules = %(modules)r
result = {"python": sys.executable, "version": list(sys.version_info[:3]),
          "modules": {}, "cuda": False}
for name in modules:
    result["modules"][name] = bool(importlib.util.find_spec(name))
try:
    import torch
    result["cuda"] = bool(torch.cuda.is_available())
    result["torch_cuda"] = torch.version.cuda
except Exception as exc:
    result["torch_error"] = str(exc)
print(json.dumps(result))
''' % {"modules": REQUIRED_MODULES}
    result = subprocess.run(
        [str(python), "-c", code], text=True, capture_output=True,
    )
    if result.returncode != 0:
        return {"python": str(python), "error": result.stderr.strip()}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"python": str(python), "error": result.stdout.strip()}


def _healthy(probe: dict[str, Any]) -> bool:
    return (not probe.get("error")
            and all(probe.get("modules", {}).get(name, False)
                    for name in REQUIRED_MODULES)
            and bool(probe.get("cuda", False)))


def _python_candidates(explicit: Path | None) -> list[Path]:
    values: list[Path] = []
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise RuntimeError(f"Python executable not found: {candidate}")
        values.append(candidate)
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
    unique: list[Path] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key in seen or not value.is_file() or not os.access(value, os.X_OK):
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _conda_executable() -> str | None:
    value = shutil.which("conda")
    if value:
        return value
    for candidate in (Path.home() / "miniconda3" / "bin" / "conda",
                      Path.home() / "anaconda3" / "bin" / "conda"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _conda_python(conda: str, env_name: str) -> Path | None:
    result = _run([conda, "run", "-n", env_name, "python", "-c",
                   "import sys; print(sys.executable)"],
                  check=False, capture=True)
    if result.returncode:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    candidate = Path(lines[-1])
    return candidate if candidate.is_file() else None


def _create_conda_environment(conda: str, env_name: str) -> Path:
    _print(f"creating conda environment: {env_name} (Python 3.10)")
    _run([conda, "create", "-y", "-n", env_name, "python=3.10", "pip"])
    python = _conda_python(conda, env_name)
    if python is None:
        raise RuntimeError(f"cannot locate Python in conda environment {env_name}")
    return python


def _find_openpcdet_root(explicit: Path | None) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not ((candidate / "pcdet").is_dir()
                and (candidate / "setup.py").is_file()):
            raise RuntimeError(
                f"OpenPCDet root must contain pcdet/ and setup.py: {candidate}")
        candidates.append(candidate)
    env_root = os.environ.get("OPENPCDET_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    home = Path.home()
    candidates.extend([
        home / "OpenPCDet",
        home / "openpcdet",
        home / "桌面" / "OpenPCDet",
        home / "桌面" / "OpenPcdet" / "OD预标注" / "OpenPCDet",
        home / "桌面" / "OD预标注" / "OpenPCDet",
        ROOT.parent / "OpenPCDet",
        ROOT.parent / "OpenPcdet" / "OD预标注" / "OpenPCDet",
    ])
    for candidate in candidates:
        if (candidate / "pcdet").is_dir() and (candidate / "setup.py").is_file():
            return candidate
    return None


def _clone_openpcdet() -> Path:
    cache = Path(os.environ.get(
        "FIVE_CLASS_CACHE", Path.home() / ".cache" / "fiveclass-prelabel"))
    target = cache / "OpenPCDet"
    if (target / "pcdet").is_dir() and (target / "setup.py").is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    repo = os.environ.get(
        "OPENPCDET_REPO", "https://github.com/open-mmlab/OpenPCDet.git")
    _run(["git", "clone", "--depth", "1", repo, str(target)])
    return target


def _install_runtime(python: Path, openpcdet_root: Path | None) -> Path | None:
    probe = _probe(python)
    if probe.get("error"):
        raise RuntimeError(f"cannot execute {python}: {probe['error']}")
    missing = [name for name in REQUIRED_MODULES
               if not probe.get("modules", {}).get(name, False)]
    if not missing and probe.get("cuda"):
        _print("existing runtime is ready")
        return openpcdet_root

    _print("missing runtime components: "
           + (", ".join(missing) if missing else "CUDA runtime"))
    _run([python, "-m", "pip", "install", "--upgrade", "pip"])
    # A CPU-only torch import is present but still unusable for Step1.  Treat
    # it like a missing runtime so the configured CUDA wheel can replace it.
    if "torch" in missing or not probe.get("cuda", False):
        torch_index = os.environ.get(
            "TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu124")
        torch_command = [
            python, "-m", "pip", "install", "torch==2.5.1",
            "torchvision==0.20.1", "--index-url", torch_index,
        ]
        if not probe.get("cuda", False):
            torch_command.append("--force-reinstall")
        _run(torch_command)
    if "spconv" in missing:
        _run([python, "-m", "pip", "install", "spconv-cu124==2.3.8"])
    _run([python, "-m", "pip", "install", "-r",
          str(ROOT / "requirements-step1.txt")])

    if "pcdet" in missing:
        openpcdet_root = openpcdet_root or _find_openpcdet_root(None)
        if openpcdet_root is None:
            _print("OpenPCDet source not found; cloning the configured repository")
            openpcdet_root = _clone_openpcdet()
        _run([python, "-m", "pip", "install", "-e", str(openpcdet_root)])
    return openpcdet_root


def _check_environment(
    python: Path, cfg: Path, ckpt: Path, *, mode: str
) -> None:
    result = _run([python, str(ROOT / "scripts" / "check_step1_env.py"),
                   "--cfg", str(cfg), "--ckpt", str(ckpt), "--mode", mode],
                  check=False)
    if result.returncode:
        raise RuntimeError("Step1 environment check failed; see the diagnostics above")


def _is_clip(path: Path) -> bool:
    lidar = path / "lidar" / "lidar_top"
    return path.is_dir() and lidar.is_dir() and any(lidar.glob("*.bin"))


def _collect_clips(input_root: Path) -> list[Path]:
    """Scan the clip parent directory for all raw clip subdirectories."""
    if not input_root.is_dir():
        raise RuntimeError(f"input directory does not exist: {input_root}")
    clips = [path.resolve() for path in sorted(input_root.iterdir())
             if _is_clip(path) and not path.name.endswith("_pre")]
    if not clips:
        raise RuntimeError(
            f"no raw clips found under {input_root}; expected child directories "
            "containing lidar/lidar_top/*.bin")
    seen: set[str] = set()
    for clip in clips:
        if clip.name in seen:
            raise RuntimeError(f"duplicate clip name in batch: {clip.name}")
        seen.add(clip.name)
    return clips


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check/install the runtime and run the five-class pipeline")
    parser.add_argument("input_root", nargs="?", type=Path,
                        default=DEFAULT_INPUT_ROOT,
                        help=f"raw SUST data directory (default: {DEFAULT_INPUT_ROOT})")
    parser.add_argument("output_root", nargs="?", type=Path,
                        default=DEFAULT_OUTPUT_ROOT,
                        help=f"SUST output directory (default: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--mode", choices=("full", "inference"), default="full",
                        help="full post-processing chain or raw Step1 inference")
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_ROOT,
                        help=f"raw JSON output directory (default: {DEFAULT_RAW_ROOT})")
    parser.add_argument("--python", type=Path,
                        help="use this Python instead of environment discovery")
    parser.add_argument("--openpcdet-root", type=Path,
                        help="OpenPCDet checkout containing pcdet/ and setup.py")
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME,
                        help="conda environment created when no ready env exists")
    parser.add_argument("--cfg", type=Path)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--weight", default=None,
                        help="权重别名：default/epoch12, epoch15, epoch17, "
                             "epoch20, argo2, waymo；可用 --list-weights 查看")
    parser.add_argument("--list-weights", action="store_true",
                        help="列出可用权重别名后退出")
    parser.add_argument("--score-thresh", type=float,
                        help="override all category thresholds")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing output clips")
    export_group = parser.add_mutually_exclusive_group()
    export_group.add_argument("--export-sust", dest="export_sust",
                              action="store_true",
                              help="write full-chain clips to output_root")
    export_group.add_argument("--no-export-sust", dest="export_sust",
                              action="store_false",
                              help="run full chain in a temporary output and discard it")
    parser.add_argument("--skip-install", "--local-only", "--no-install",
                        dest="skip_install", action="store_true",
                        help="fail instead of installing missing dependencies")
    parser.add_argument("--check-only", action="store_true",
                        help="check/install the runtime without running inference")
    parser.set_defaults(export_sust=True)
    return parser.parse_args()


def _run_raw_inference(python: Path, clips: list[Path], cfg: Path,
                       ckpt: Path, raw_output: Path, *, score_thresh: float,
                       overwrite: bool) -> None:
    """Keep only raw model JSON; all Step1 work files stay temporary."""
    raw_output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="raw_inference_") as temp:
        work_root = Path(temp)
        for index, clip in enumerate(clips, 1):
            destination = raw_output / f"{clip.name}_raw.json"
            if destination.exists() and not overwrite:
                raise RuntimeError(
                    f"raw output exists, pass --overwrite: {destination}")
            _print(f"raw inference [{index}/{len(clips)}]: {clip.name}")
            _run([
                python, str(ROOT / "pipeline" / "step1_lidar_inference.py"),
                "--clip", clip, "--work-root", work_root,
                "--cfg", cfg, "--ckpt", ckpt,
                "--score-thresh", str(score_thresh),
            ])
            source = work_root / f"{clip.name}_raw.json"
            if destination.exists():
                destination.unlink()
            shutil.copy2(source, destination)
            _print(f"raw JSON: {destination}")


def _run_full_pipeline(python: Path, clips: list[Path], cfg: Path,
                       ckpt: Path, output_root: Path, *, export_sust: bool,
                       score_thresh: float | None, overwrite: bool) -> None:
    """Run the complete chain with temporary intermediate JSON files."""
    output_root = output_root.expanduser().resolve()
    if export_sust:
        output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="full_pipeline_output_") as temp:
        final_root = output_root if export_sust else Path(temp)
        command = [
            python, str(ROOT / "run_end_to_end.py"),
            "--inference-python", python, "--post-python", python,
            "--cfg", cfg, "--ckpt", ckpt,
            "--final-root", final_root, "--preserve-input",
        ]
        if export_sust:
            command.extend(["--sust-root", output_root, "--export-sust"])
        if overwrite:
            command.append("--overwrite")
        if score_thresh is not None:
            command.extend(["--score-thresh", str(score_thresh)])
        for clip in clips:
            command.extend(["--clip", clip])
        _run(command)
        if export_sust:
            _print(f"SUST output: {output_root}")
        else:
            _print("full-chain output discarded after validation")


def main() -> int:
    args = _parse_args()
    if args.list_weights:
        _print_weight_table()
        return 0

    ckpt, cfg, weight_mode = _resolve_weight(args.weight, args.ckpt, args.cfg)
    if args.mode == "full" and weight_mode == "inference":
        raise RuntimeError(
            "waymo weight uses a three-class head and only supports "
            "--mode inference; use a five-class weight for the full chain"
        )

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if input_root == output_root:
        raise RuntimeError("input_root and output_root must be different directories")
    if not cfg.is_file():
        raise RuntimeError(f"config not found: {cfg}")
    if not ckpt.is_file():
        raise RuntimeError(f"checkpoint not found: {ckpt}")

    candidates = _python_candidates(args.python)
    python: Path | None = None
    for candidate in candidates:
        probe = _probe(candidate)
        _print(f"probe {candidate}: "
               + ("ready" if _healthy(probe)
                  else "needs setup"))
        if _healthy(probe):
            python = candidate
            break
    if python is None:
        if args.skip_install:
            raise RuntimeError(
                "no CUDA/OpenPCDet Python environment found; remove --skip-install "
                "to install one automatically")
        conda = _conda_executable()
        if conda and args.python is None and not os.environ.get("OPENPCDET_PYTHON"):
            python = _conda_python(conda, args.env_name)
            if python is None:
                python = _create_conda_environment(conda, args.env_name)
        else:
            python = (candidates[0] if candidates else Path(sys.executable))
            version = _probe(python).get("version", [])
            if version and not (3, 10) <= tuple(version[:2]) <= (3, 12):
                raise RuntimeError(
                    "auto-install supports Python 3.10-3.12; install conda "
                    "or pass --python to a supported environment")

    openpcdet_root = _find_openpcdet_root(args.openpcdet_root)
    if args.skip_install:
        probe = _probe(python)
        if not _healthy(probe):
            raise RuntimeError("selected Python environment is incomplete; see probes above")
    else:
        openpcdet_root = _install_runtime(python, openpcdet_root)
    _check_environment(python, cfg, ckpt, mode=args.mode)
    if args.check_only:
        _print("environment check complete")
        return 0

    clips = _collect_clips(input_root)
    _print(f"input: {input_root}")
    _print(f"clips: {len(clips)}")
    if args.mode == "inference":
        raw_output = args.raw_output.expanduser().resolve()
        raw_score = 0.1 if args.score_thresh is None else args.score_thresh
        _run_raw_inference(
            python, clips, cfg, ckpt, raw_output,
            score_thresh=raw_score, overwrite=args.overwrite)
    else:
        _run_full_pipeline(
            python, clips, cfg, ckpt, output_root,
            export_sust=args.export_sust,
            score_thresh=args.score_thresh,
            overwrite=args.overwrite)
    _print("completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        _print(f"ERROR: {exc}")
        raise SystemExit(1)
