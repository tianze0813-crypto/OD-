"""Truck 链（大型车）：独立的推理后处理。

与 VRU 链完全分离的参数（【改动】按用户 2026-09-18 需求）：
  * 范围      前 80 / 后 20 / 左右 40
  * 分数阈值  Truck 0.4
  * 短轨迹    4（保留短轨迹过滤，阈值暂定 4）
  * 静止过滤  不做（卡车停车等灯是常态；链内静止过滤只作用于 Nonmotorized_vehicle，
              本链已把 NMV 过滤掉，因此天然不生效）
  * yaw       新版 v2（apply_motion_yaw=False：动态段/拐弯保留 detector yaw）
  * obj_id    从 1000 开始（与 VRU 链的 2000 起互不冲突）

【改动】2026-09-20：检测器换成 BEVFusion（mmdet3d 1.x 官方 20ep 权重），并引入 Trailer（挂车）：
  * Step1 用 pipeline/step1_bevfusion_truck.py（C+L / 纯雷达可切）
  * 盒级规则：挂车被货车罩住 → 丢挂车（以 Truck 为准）；
              挂车与货车有交集 → 并集合并成一个大长 Truck
              （见 geometry/truck_trailer_rules.py）
  * 轨迹级规则：同一个 obj_id 里只要出现过 Truck，整条轨迹都改成 Truck
  * 类别阈值：Truck **0.2**（2026-09-20 由 0.4 下调，让分数偏低的货车也能和挂车并集）；Trailer 0.25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.hybrid_expD_noncar import run as _noncar_run  # noqa: E402
from tracking import tracker_conservative as tracking        # noqa: E402


def _enable_trailer_vocabulary() -> None:
    """【改动】2026-09-20：只在本链（truck）里放开 Trailer 词表。

    共享文件（tracking/tracker_conservative.py）保持原样：这里在 **Truck 链被导入时**把
    Trailer 加进 tracking 的类别映射与目标类别。代价是同一进程内 TARGET_CLASSES 会多一个
    Trailer —— Car / VRU 链不会产生这个类，逻辑与阈值都不受影响（只多一行诊断列表）。
    """
    if "Trailer" not in tracking.TARGET_CLASSES:
        tracking.TARGET_CLASSES = tuple(tracking.TARGET_CLASSES) + ("Trailer",)
    tracking.CLASS_MAP.setdefault("trailer", "Trailer")
    tracking.CLASS_MAP.setdefault("Trailer", "Trailer")


_enable_trailer_vocabulary()

KEEP_CLASSES = ("Truck", "Trailer")   # 【改动】2026-09-20 加 Trailer
TRAILER_SCORE_THRESHOLD = 0.25        # 【改动】挂车单独阈值（比 Truck 低，便于把头+挂并起来）
ID_OFFSET = 1000
LABEL_SUBDIR = "label_truck"

# 本链的默认参数（可被 run(...) 的 overrides 覆盖）
DEFAULTS: Dict[str, Any] = dict(
    keep_classes=KEEP_CLASSES,
    class_score_thresholds={"Truck": 0.2, "Trailer": TRAILER_SCORE_THRESHOLD},  # 【改动】Truck 0.4 -> 0.2
    range_front=80.0,          # 【改动】前 80
    range_rear=20.0,
    range_side=40.0,
    sparsity_max_points=10,
    visibility_min_ratio=0.05,
    short_track_max_frames=4,  # 【改动】短轨迹过滤保留，阈值 4
    nonmotorized_min_net_displacement=0.0,   # 本链无 NMV，不生效
    yaw_impl="v2",             # 【改动】新版 yaw：动态段保留 detector yaw
    static_rotation_classes=("Truck", "Bus"),
    # 【改动】2026-09-20 关掉 step2_5 的「静止旋转轨迹整条删除」：
    # 判据是相邻帧 IoU >= 0.35（= 车没动）且 yaw 抖动大 -> 整条轨迹删掉，
    # 交警域里「停着等灯的卡车 + yaw 估计抖动」会被误删（用户 2026-09-20 要求去掉）。
    static_rotation_enabled=False,
    # 【改动】2026-09-20 用户确认：范围/分数过滤保持【早期】（跟踪前），
    # 只有生命周期（短轨迹）过滤在跟踪之后 —— 那条本来就在 step2_5 里，无需改。
    pre_tracking_filters=True,
    # 【改动】不把动态轨迹钉到停车位 id（保留跟踪器其余逻辑）
    disable_slot_binding=True,
    # 【改动】2026-09-20 方案A：动态跟踪加固（只搬 Car 链 tracker 的能力，不搬 region 那套）
    #  ① 遮挡复活：动态轨迹活到 3.0 s（原来 1.8 s 就死且不会再接回）；超过 1.8 s 的重现
    #     用「可达速度 × 间隔」的走廊 + 方向一致性判定，而不是 10 Hz 的固定小门。
    #     实测依据：5 个 clip 里卡车被切开的 26 处，间隔中位 1.9 s / P90 4.3 s / 最大 5.1 s，
    #     其中 50% 超过 1.8 s，很多切开处端点只差 0.2~0.8 m（同车同位置丢 id）。
    #  ② 横向跳变门：不许横向瞬移到隔壁车（卡车车队邻距小，防串 id）。
    #  ③ 静态锚点：min_static_hits 由 10**9 改为 6 —— 停着的卡车能够升级成静态锚点，
    #     长期保留 id 并且能在丢失后用锚点复活（这正是"排队/等灯卡车"要的行为）。
    dynamic_occlusion_max_gap=3.0,
    lateral_jump_gate=True,
    static_anchor_min_hits=6,
    # 【改动】2026-09-21 方案B（用户要求「照搬 Car 的跟踪逻辑」）：跟踪这一遍用
    # main_chain 的 step2 + step4.5，插在本链 step2_5（类别修正）/ step3（卡车几何 +
    # yaw v2）之前 —— main_chain 只负责 id，几何与 yaw 仍由本链精修定稿。
    # 实测 5 clip：轨迹 118→56、多 id 物理对象 17→6（方案A 只有 116/17）。
    step2_impl="car",
    car_step2_keep_classes="Truck",
    car_step2_score_threshold=0.2,
    # 【改动】跳过静态 yaw 稳定（不把静止段 yaw 锁到停车方向）
    static_yaw_enabled=False,
    # 【改动】yaw v2 开关：关静态方向投票；直线行驶的轨迹用运动方向作 yaw
    yaw_vehicle_flags={"apply_static_direction_vote": False,
                       "apply_straight_motion_yaw": True,
                       "apply_motion_yaw": False},
    # 【改动】Truck 专用后处理：①yaw旋转帧 ②IoU并集合并(0.1) ③xy贴合 ④yaw翻转(>90°)
    truck_postprocess=True,
    truck_merge_enabled=False,   # 【改动】关掉 step3 自带的 Truck 合并，改用并集长框合并
    # 【改动】2026-09-20 货车/挂车规则（geometry/truck_trailer_rules.py）
    trailer_rules=True,
    trailer_dup_iom=0.70,        # 挂车被货车罩住 >= 70% -> 判重复，丢挂车
    trailer_dup_iou=0.50,        # 或 BEV IoU >= 0.5 -> 判重复
    trailer_merge_iou=0.05,      # 有交集 -> 并集合并成一个大长 Truck
    # 【改动】2026-09-21：标注侧没有 Trailer 类别 -> 默认一律并成 Truck
    trailer_policy="to-truck",   # keep=纯挂车轨迹保留 Trailer | to-truck=一律并成 Truck
)


def run(raw_json: Path, clip: Path, out_json: Path,
        diagnostics_path: Optional[Path] = None,
        **overrides: Any) -> Dict[str, Any]:
    """Truck 链：raw json -> 类别合并（跟踪前）-> 跟踪/过滤/精修 -> 轨迹级类别统一 -> out json。

    链路顺序（用户 2026-09-20 明确）：
        检测 → 范围/分数过滤(链内早期) → **类别合并(merge_classes_pre)** → ID 跟踪
             → 短轨迹/硬过滤等其他过滤(step2/step2_5) → 精修(step3 + truck_postprocess)
             → **轨迹级类别统一(unify_track_classes)** → 导出 SUST label
    """
    from geometry import truck_trailer_rules as rules

    params = dict(DEFAULTS)
    params.update(overrides)
    enabled = bool(params.pop("trailer_rules", True))
    dup_iom = float(params.pop("trailer_dup_iom", DEFAULTS["trailer_dup_iom"]))
    dup_iou = float(params.pop("trailer_dup_iou", DEFAULTS["trailer_dup_iou"]))
    merge_iou = float(params.pop("trailer_merge_iou", DEFAULTS["trailer_merge_iou"]))
    policy = str(params.pop("trailer_policy", DEFAULTS["trailer_policy"]))

    staged: Path = Path(raw_json)
    extra_diag: Dict[str, Any] = {}
    if enabled:   # 类别合并：类名归一 + 挂车去重 + 有交集并集成大长 Truck（跟踪之前）
        frames = json.loads(Path(raw_json).read_text(encoding="utf-8"))
        merged_frames, report = rules.merge_classes_pre(
            frames, dup_iom=dup_iom, dup_iou=dup_iou, merge_iou=merge_iou)
        staged = Path(raw_json).with_name(Path(raw_json).stem + "_classmerge.json")
        staged.write_text(json.dumps(merged_frames, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        extra_diag["truck_trailer_class_merge"] = report

    diag = _noncar_run(staged, clip, out_json, diagnostics_path, **params)

    if enabled:   # 轨迹级类别统一：同一 id 里出现过 Truck -> 整条都算 Truck
        frames = json.loads(Path(out_json).read_text(encoding="utf-8"))
        frames, unify = rules.unify_track_classes(frames)
        if policy == "to-truck":
            frames, pol = rules.trailer_to_truck_all(frames)
            unify["trailer_policy"] = pol
        Path(out_json).write_text(
            json.dumps(frames, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        diag = dict(diag or {})
        diag.update(extra_diag)
        diag["track_class_unify"] = unify
        target = Path(diagnostics_path or Path(out_json).with_name(
            Path(out_json).stem + "_diagnostics.json"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(diag, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    return diag


def export_labels(frames, clip: Path, subdir: str = LABEL_SUBDIR,
                  id_offset: int = ID_OFFSET) -> int:
    """把链路输出写成 SUST label（base_link 系），obj_id 加 id_offset。"""
    import shutil
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
                item["obj_id"] = "t" + str(item["obj_id"])
            labels.append(item)
        total += len(labels)
        (label_dir / f"{frame['frame_id']}.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", type=Path, default=None,
                        help="--detector voxelnext 时必填；bevfusion 模式下由本脚本生成")
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--label-subdir", default=LABEL_SUBDIR)
    parser.add_argument("--id-offset", type=int, default=ID_OFFSET)
    parser.add_argument("--no-export", action="store_true")
    # 【改动】2026-09-20 BEVFusion 检测器 + 货车/挂车规则参数
    parser.add_argument("--detector", choices=["bevfusion", "voxelnext", "raw"],
                        default="bevfusion",
                        help="bevfusion: 先用 pipeline/step1_bevfusion_truck.py 出 raw json；"
                             "raw: 直接用外部传入的 --raw-json（检测器已由别处跑好）；"
                             "voxelnext: 旧 VoxelNeXt truckB 权重（保留兼容）")
    parser.add_argument("--truck-cfg", type=Path, default=None,
                        help="BEVFusion 配置（默认取 BEVFUSION_TRUCK_CFG 或工程内置默认）")
    parser.add_argument("--truck-ckpt", type=Path, default=None)
    parser.add_argument("--detector-mode", choices=["fusion", "lidar"], default="lidar",
                        help="lidar=纯雷达权重（默认；不读图/不去畸变，单帧 696ms->108ms）；fusion=C+L（读 4 路相机图）"
                             "（不读图、不去畸变，单帧 696ms -> 110ms，truck 指标基本不变）")
    parser.add_argument("--work-root", type=Path, default=None,
                        help="step1 产物目录（默认 <project>/work/step1_inference）")
    parser.add_argument("--trailer-score-threshold", type=float, default=TRAILER_SCORE_THRESHOLD)
    parser.add_argument("--raw-score-threshold", type=float, default=0.1,
                        help="BEVFusion 检测器的原始阈值（链内阈值由 --truck/trailer-score-threshold 把关）")
    parser.add_argument("--no-trailer-rules", action="store_true",
                        help="关掉货车/挂车去重与并集（回退到只有 Truck 的旧行为）")
    parser.add_argument("--trailer-dup-iom", type=float, default=DEFAULTS["trailer_dup_iom"])
    parser.add_argument("--trailer-dup-iou", type=float, default=DEFAULTS["trailer_dup_iou"])
    parser.add_argument("--trailer-merge-iou", type=float, default=DEFAULTS["trailer_merge_iou"])
    parser.add_argument("--trailer-policy", choices=["keep", "to-truck"],
                        default=DEFAULTS["trailer_policy"],
                        help="keep: 纯挂车轨迹保留 Trailer；to-truck（默认）: 所有挂车也并成 Truck，"
                             "因为标注侧没有 Trailer 类别")
    parser.add_argument("--range-front", type=float, default=DEFAULTS["range_front"])
    parser.add_argument("--range-rear", type=float, default=DEFAULTS["range_rear"])
    parser.add_argument("--range-side", type=float, default=DEFAULTS["range_side"])
    parser.add_argument("--truck-score-threshold", type=float, default=0.2)  # 【改动】0.4 -> 0.2
    parser.add_argument("--short-track-max-frames", type=int, default=4)
    parser.add_argument("--sparsity-max-points", type=int, default=10)
    parser.add_argument("--yaw-impl", default="v2")
    args = parser.parse_args()

    raw_json = args.raw_json
    if args.detector == "bevfusion":   # 【改动】BEVFusion 检测器直接在本步产出 raw json
        from pipeline import step1_bevfusion_truck as bf
        work_root = args.work_root or (PROJECT_ROOT / "work" / "step1_inference")
        bf_args = argparse.Namespace(
            work_root=work_root,
            cfg=str(args.truck_cfg) if args.truck_cfg else bf.BEVFUSION_CFG,
            ckpt=str(args.truck_ckpt) if args.truck_ckpt else bf.BEVFUSION_CKPT,
            mode=str(args.detector_mode),
            score_thresh=float(args.raw_score_threshold),
            z_convention="center", skip_prepare=False, jobs=6,
            vis_occl_tol=0.3, no_visibility_check=False,
        )
        raw_json = bf.run_inference(Path(args.clip).resolve(), bf_args)
        print(">> BEVFusion raw json: %s" % raw_json)
    elif raw_json is None:
        parser.error("--detector raw/voxelnext 时必须给 --raw-json")

    diag = run(raw_json, args.clip, args.out_json, args.diagnostics,
               class_score_thresholds={"Truck": float(args.truck_score_threshold),
                                       "Trailer": float(args.trailer_score_threshold)},
               trailer_rules=not args.no_trailer_rules,
               trailer_dup_iom=args.trailer_dup_iom,
               trailer_dup_iou=args.trailer_dup_iou,
               trailer_merge_iou=args.trailer_merge_iou,
               trailer_policy=args.trailer_policy,
               range_front=args.range_front, range_rear=args.range_rear,
               range_side=args.range_side,
               short_track_max_frames=int(args.short_track_max_frames),
               sparsity_max_points=int(args.sparsity_max_points),
               yaw_impl=str(args.yaw_impl))
    if not args.no_export:
        frames = json.loads(Path(args.out_json).read_text(encoding="utf-8"))
        n = export_labels(frames, Path(args.clip), args.label_subdir, args.id_offset)
        diag = dict(diag or {})
        diag["exported_labels"] = n
        print("exported %d labels -> %s" % (n, Path(args.clip) / args.label_subdir))
    print("truck_chain final_detections=%s yaw_impl=%s" % (
        (diag or {}).get("final_detections"), (diag or {}).get("yaw_impl")))


if __name__ == "__main__":
    main()
