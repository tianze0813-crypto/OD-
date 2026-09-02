#!/usr/bin/env python3
"""Production pre-annotation inference for xyzi .bin point clouds.

Run from /home/moga/桌面/OpenPcdet/OD预标注 (where the `cfgs` symlink lives):
    python /home/moga/桌面/pandarset/run_prelabel.py \
        --cfg_file cfgs/pandaset_models/voxelnext_pandaset.yaml \
        --ckpt /path/to/checkpoint_epoch_N.pth \
        --lidar_dir /path/to/clip/lidar/lidar_top \
        --out_json /path/to/results.json

Output JSON has the same structure as run_openpcdet_demo20.py, so it can be
consumed by export_sustech_prelabels.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class PcdBinDataset(DatasetTemplate):
    """Reads <frame_id>.bin files with float32 x,y,z,intensity columns."""

    def __init__(self, dataset_cfg, class_names, root_path, logger):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names,
            training=False, root_path=root_path, logger=logger,
        )
        self.sample_file_list = sorted(Path(root_path).glob("*.bin"))

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        point_path = self.sample_file_list[index]
        raw = np.fromfile(point_path, dtype=np.float32)
        if raw.size % 4 != 0:
            raise ValueError(f"unexpected value count in {point_path}: {raw.size}")
        points = raw.reshape(-1, 4).astype(np.float32)
        # PandaSet training normalizes intensity to [0,1]; keep it consistent.
        points[:, 3] /= 255.0
        # Some nuScenes training configs declare a fifth timestamp channel
        # although the deployed SUST frames are single-sweep XYZI.  Supply a
        # neutral timestamp only when the selected config requires it.
        src_features = list(
            self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)
        if len(src_features) > points.shape[1]:
            if src_features[:4] != ["x", "y", "z", "intensity"]:
                raise ValueError(
                    "unsupported point feature layout: " + str(src_features))
            points = np.pad(points,
                            ((0, 0), (0, len(src_features) - points.shape[1])),
                            mode="constant")
        elif len(src_features) < points.shape[1]:
            points = points[:, :len(src_features)]
        return self.prepare_data({
            "points": points,
            "frame_id": point_path.stem,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--lidar_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--score_thresh", type=float, default=0.3)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    dataset = PcdBinDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=Path(args.lidar_dir),
        logger=logger,
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    results = []
    with torch.no_grad():
        for data_dict in dataset:
            frame_id = data_dict["frame_id"]
            batch_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model.forward(batch_dict)
            pred = pred_dicts[0]
            scores = pred["pred_scores"].detach().cpu().numpy()
            keep = scores >= args.score_thresh
            boxes = pred["pred_boxes"].detach().cpu().numpy()[keep]
            kept_scores = scores[keep]
            labels = pred["pred_labels"].detach().cpu().numpy()[keep]
            detections = []
            for box, score, label in zip(boxes, kept_scores, labels):
                detections.append({
                    "class_name": cfg.CLASS_NAMES[int(label) - 1],
                    "score": float(score),
                    "box_lidar": box.astype(float).tolist(),
                })
            results.append({
                "frame_id": frame_id,
                "num_points": int(batch_dict["points"].shape[0]),
                "num_detections": len(detections),
                "detections": detections,
            })

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps({
        "frames": len(results),
        "detections": sum(r["num_detections"] for r in results),
        "out_json": str(out_path),
    }, indent=2))


if __name__ == "__main__":
    main()
