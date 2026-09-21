"""Car 链（2026-09-21 新增）：只用「追踪 + 硬过滤 + base_link 输出」这套通用后处理。

**用途**：换检测器时（例如用 BEVFusion 的 car 头）想快速看效果，跳过 main_chain/Waymo
那一整套 Car 专属精修（step3_car_box_fit / step4 size filter / step4.5 region retrack / step5），
只跑后续真正要用的：类别范围过滤 → 硬过滤 → 静态优先+匈牙利跟踪 → 短轨迹/硬过滤第二遍
→ 通用几何精修 → 五类出口转 base_link。

**类别映射以原始 raw json 为准**：raw json 里是 `car`/`truck`/... 还是 `Car` 都行 ——
链内统一走 `tracking.canonical_class_name()` 归一（hybrid 的硬过滤 `_class_allowed` /
`score_threshold_for` 都接受原始名），**不需要预先重映射**。

参数（可用 run(...) 的 overrides 覆盖，或用命令行开关）：
  * 类别      keep_classes=("Car",)，分数阈值 Car 0.2
  * 范围      前 80 / 后 20 / 左右 40 m
  * 稀疏度    ≤5 点   可见度 0.05   短轨迹 3 帧
  * yaw       v2（直线段运动方向 / 静止点云主轴），Car 允许静态 slot 与静态 yaw 稳定
  * obj_id    从 1 起（单独看用；合并到三链时由 orchestrator 加偏移）

**与 main_chain Car 链的分工**：main_chain 那条是 Waymo 权重 + Car 专属精修
（step3_car_box_fit / step4 size filter / step4.5 region retrack / step5，
分数/稀疏度门限与"原始类名字符串白名单"都是给 Waymo 头调的）；本链只保留通用后处理，
换检测器（尤其 BEVFusion 的 car 头）时数更符合直觉。两条链的输出都是 SUST 可直接读的
base_link 标签，可以并排对比。

**整批集成**：`scripts/run_hybrid_prelabel.py --car-pipeline hybrid --car-detector bevfusion`
即用本链（与 truck/vru 共用同一份 BEVFusion 原始检测）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.hybrid_expD_noncar import run as _noncar_run  # noqa: E402
from tracking import tracker_conservative as tracking        # noqa: E402

KEEP_CLASSES = ("Car",)
ID_OFFSET = 0
LABEL_SUBDIR = "label_car"

DEFAULTS: Dict[str, Any] = dict(
    keep_classes=KEEP_CLASSES,
    class_score_thresholds={"Car": 0.2},
    range_front=80.0,
    range_rear=20.0,
    range_side=40.0,
    sparsity_max_points=5,
    visibility_min_ratio=0.05,
    short_track_max_frames=3,
    nonmotorized_min_net_displacement=0.0,
    yaw_impl="v2",
    static_rotation_classes=("Car",),
    disable_slot_binding=False,
    static_yaw_enabled=True,
    yaw_vehicle_flags=None,
    truck_postprocess=False,
    truck_merge_enabled=True,
    pre_tracking_filters=True,
)


def run(raw_json: Path, clip: Path, out_json: Path,
        diagnostics_path: Optional[Path] = None,
        **overrides: Any) -> Dict[str, Any]:
    """raw json（lidar 系）-> 通用后处理 -> out json（base_link，含 track_id）。"""
    params = dict(DEFAULTS)
    params.update(overrides)
    return _noncar_run(raw_json, clip, out_json, diagnostics_path, **params)


def export_labels(frames, clip: Path, subdir: str = LABEL_SUBDIR,
                  id_offset: int = ID_OFFSET) -> int:
    """把链路输出写成 SUST label（base_link 系）。"""
    label_dir = Path(clip) / subdir
    if label_dir.is_dir():
        shutil.rmtree(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for frame in frames:
        labels = []
        for det in frame.get("detections", []):
            if det.get("track_id") is None:
                continue
            item = tracking.box_to_label(det)
            try:
                item["obj_id"] = str(int(item["obj_id"]) + int(id_offset))
            except (TypeError, ValueError):
                item["obj_id"] = "c" + str(item["obj_id"])
            labels.append(item)
        total += len(labels)
        (label_dir / f"{frame['frame_id']}.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    return total


def link_output(clip: Path, label_dir: Path, target: Path) -> None:
    """把 label 放到 <target>/label，并把 image/lidar/transforms/readme.json 软链过去
    （SUST 可直接识别该目录；不复制数据、不改动输入 clip）。"""
    target = Path(target)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for sub in ("image", "lidar", "transforms", "readme.json"):
        src = clip / sub
        if src.exists():
            (target / sub).symlink_to(src.resolve())
    out = target / "label"
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(label_dir.glob("*.json")):
        shutil.copy2(path, out / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", type=Path, default=None,
                        help="--detector voxelnext/已有结果：直接给 raw json；"
                             "bevfusion 模式留空则本脚本自己跑 step1")
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--detector", choices=["bevfusion", "raw"], default="raw",
                        help="bevfusion: 自己跑 BEVFusion step1（与 Truck 链同一套）；raw: 用 --raw-json")
    parser.add_argument("--detector-mode", choices=["lidar", "fusion"], default="lidar")
    parser.add_argument("--raw-score-threshold", type=float, default=0.1)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--car-score-threshold", type=float, default=0.2)
    parser.add_argument("--range-front", type=float, default=DEFAULTS["range_front"])
    parser.add_argument("--range-rear", type=float, default=DEFAULTS["range_rear"])
    parser.add_argument("--range-side", type=float, default=DEFAULTS["range_side"])
    parser.add_argument("--sparsity-max-points", type=int, default=DEFAULTS["sparsity_max_points"])
    parser.add_argument("--visibility-min-ratio", type=float, default=DEFAULTS["visibility_min_ratio"])
    parser.add_argument("--short-track-max-frames", type=int, default=DEFAULTS["short_track_max_frames"])
    parser.add_argument("--label-subdir", default=LABEL_SUBDIR)
    parser.add_argument("--id-offset", type=int, default=ID_OFFSET)
    parser.add_argument("--link-dir", type=Path, default=None,
                        help="额外生成 SUST 可打开的目录（软链 + label）")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    raw_json = args.raw_json
    if args.detector == "bevfusion":       # 复用 Truck 链那套 BEVFusion step1（同一份权重/配置）
        from pipeline import step1_bevfusion_truck as bf
        work_root = args.work_root or (PROJECT_ROOT / "work" / "step1_bevfusion")
        bf_args = argparse.Namespace(
            work_root=work_root,
            cfg=bf.BEVFUSION_CFG, ckpt=bf.BEVFUSION_CKPT,
            mode=str(args.detector_mode),
            score_thresh=float(args.raw_score_threshold),
            z_convention="center", skip_prepare=False, jobs=6,
            vis_occl_tol=0.3, no_visibility_check=False)
        raw_json = bf.run_inference(Path(args.clip).resolve(), bf_args)
        print(">> BEVFusion raw json: %s" % raw_json)
    if raw_json is None:
        parser.error("--detector raw 时必须给 --raw-json")

    diag = run(raw_json, args.clip, args.out_json, args.diagnostics,
               class_score_thresholds={"Car": float(args.car_score_threshold)},
               range_front=float(args.range_front),
               range_rear=float(args.range_rear),
               range_side=float(args.range_side),
               sparsity_max_points=int(args.sparsity_max_points),
               visibility_min_ratio=float(args.visibility_min_ratio),
               short_track_max_frames=int(args.short_track_max_frames))
    if not args.no_export:
        frames = json.loads(Path(args.out_json).read_text(encoding="utf-8"))
        n = export_labels(frames, Path(args.clip), args.label_subdir, args.id_offset)
        diag = dict(diag or {})
        diag["exported_labels"] = n
        print("exported %d labels -> %s" % (n, Path(args.clip) / args.label_subdir))
        if args.link_dir:
            link_output(Path(args.clip), Path(args.clip) / args.label_subdir, args.link_dir)
            print("SUST 目录 -> %s" % args.link_dir)
    print("car_chain final_detections=%s" % (diag or {}).get("final_detections"))


if __name__ == "__main__":
    main()
