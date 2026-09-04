#!/usr/bin/env python3
"""Validate the Python/CUDA/checkpoint prerequisites for Step1."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "models" / "voxelnext_fiveclass_nuscenes_infer.yaml"
DEFAULT_CKPT = ROOT / "models" / "vn5_nuscenes_checkpoint_epoch_12.pth"
REQUIRED_MODULES = {
    "numpy": ("numpy",),
    "scipy": ("scipy",),
    "cv2": ("opencv-python",),
    "PIL": ("Pillow",),
    "pandas": ("pandas",),
    "av2": ("av2",),
    "kornia": ("kornia",),
    "torch": ("torch",),
    "spconv": ("spconv-cu124", "spconv-cu120", "spconv-cu118"),
    "pcdet": ("pcdet",),
    "yaml": ("PyYAML",),
}
EXPECTED_CLASSES = (
    "Car", "Truck", "Bus", "Pedestrian", "Nonmotorized_vehicle",
)


def _version(distributions: tuple[str, ...]) -> str:
    for distribution in distributions:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "installed (distribution name not detected)"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    errors: list[str] = []

    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    for module_name, distributions in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            errors.append(f"{module_name} ({'/'.join(distributions)}): {exc}")
            print(f"{module_name}: ERROR ({exc})")
        else:
            print(f"{module_name}: ok ({_version(distributions)})")

    try:
        import torch

        print(
            "torch.cuda: "
            f"available={torch.cuda.is_available()} "
            f"version={torch.version.cuda or 'none'} "
            f"devices={torch.cuda.device_count()}"
        )
        if not torch.cuda.is_available():
            errors.append(
                "CUDA is unavailable; inference/run_prelabel.py calls model.cuda()"
            )
    except Exception:
        pass

    if not args.cfg.is_file():
        errors.append(f"config not found: {args.cfg}")
    if not args.ckpt.is_file():
        errors.append(f"checkpoint not found: {args.ckpt}")

    if args.cfg.is_file() and args.ckpt.is_file():
        try:
            import torch
            from pcdet.config import cfg, cfg_from_yaml_file

            cfg_from_yaml_file(str(args.cfg), cfg)
            classes = tuple(str(name) for name in cfg.CLASS_NAMES)
            if classes != EXPECTED_CLASSES:
                errors.append(
                    "config classes do not match the five-class checkpoint: "
                    f"{classes}"
                )

            checkpoint = torch.load(
                args.ckpt, map_location="cpu", weights_only=False
            )
            state = checkpoint.get("model_state", checkpoint.get("state_dict"))
            if not isinstance(state, dict):
                errors.append("checkpoint has no model_state/state_dict mapping")
            else:
                heads = {
                    key.split("dense_head.heads_list.", 1)[1].split(".", 1)[0]
                    for key in state
                    if key.startswith("dense_head.heads_list.")
                }
                if heads != {str(i) for i in range(len(EXPECTED_CLASSES))}:
                    errors.append(
                        "checkpoint head count is incompatible with five classes: "
                        f"{sorted(heads)}"
                    )
                print(
                    "checkpoint: ok "
                    f"epoch={checkpoint.get('epoch', '?')} "
                    f"heads={len(heads)}"
                )
        except Exception as exc:  # pragma: no cover - environment dependent
            errors.append(f"config/checkpoint validation failed: {exc}")

    if errors:
        print("\nStep1 environment check: FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("\nStep1 environment check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
