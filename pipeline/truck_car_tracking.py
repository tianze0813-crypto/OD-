#!/usr/bin/env python3
"""【改动】2026-09-21 Truck 链「照搬 Car 链跟踪逻辑」的接线（用户要求）。

Car 链的跟踪栈是 ``main_chain`` 里的：
  step2 = 类过滤/统一 + StaticFirstTracker（槽位绑定开启）+ 硬过滤 + 静态 yaw + yaw integrated
  step4.5 = 动态区域掩膜 → 区域重跟踪（occlusion 2.6 s / motion-only）→ ID 继承 →
            队列拼接 → 相位拼接（≤30 s）

Car 链的 step3（轿车框拟合）与 step4（Car→Truck 尺寸门 + 只留 Car）**对卡车不适用**
（step4 会把卡车整条删掉），所以这里只复用 step2 + step4.5，插在 truck 链自己的
step2_5（类别修正）/ step3（卡车几何 + yaw v2）之前 —— 几何与 yaw 仍由 truck 链自己
的精修定稿，main_chain 这一步只负责「id」。

输入：truck 链 step1 的 raw json（已做类别合并，全部是 Truck）
输出：``<work-root>/step45/<clip.name>_step45.json`` + 同名 ``_step45_diagnostics.json``

【重要】main_chain 只负责 **id**：它的 step2 还会跑 Car 的 yaw（里面同样带"静止多帧
点云主轴"那条，会把静止卡车拉歪 ~16°），step4.5 还会做轿车式 box fit。所以跑完后把
每个检测的 ``box_lidar`` 从输入 raw json 按「同帧 + 同类 + 最近中心」还原回来，只保留
id；几何与 yaw 仍由本链自己的 step2_5 / step3 定稿。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_CHAIN_ROOT = PROJECT_ROOT / "main_chain"

STEP2_SCRIPT = MAIN_CHAIN_ROOT / "pipeline" / "step2_identity_class_filter_yaw.py"
STEP45_SCRIPT = (MAIN_CHAIN_ROOT / "pipeline"
                 / "step4_5_region_phase_retrack_batch.py")


def _run(command) -> None:
    print("[car-style-track] $ " + " ".join(str(value) for value in command),
          flush=True)
    subprocess.run([str(value) for value in command], check=True)


def _restore_geometry(out_json: Path, raw_json: Path,
                      max_distance: float = 1.0) -> Dict[str, Any]:
    """把 step45 输出的 box_lidar 还原成输入 raw 的原始值（只保留 id）。"""
    import math
    frames = json.loads(Path(out_json).read_text(encoding="utf-8"))
    source = json.loads(Path(raw_json).read_text(encoding="utf-8"))
    raw_by_frame: Dict[int, list] = {}
    for frame in source:
        raw_by_frame[int(frame["frame_id"])] = list(frame.get("detections", []))
    restored = 0
    missing = 0
    shifted = []
    for frame in frames:
        ts = int(frame["frame_id"])
        candidates = raw_by_frame.get(ts, [])
        used = set()
        for det in frame.get("detections", []):
            box = det.get("box_lidar")
            if not isinstance(box, list) or len(box) < 7 or not candidates:
                continue
            best = None
            for index, raw_det in enumerate(candidates):
                if index in used:
                    continue
                if str(raw_det.get("class_name")) != str(det.get("class_name")):
                    continue
                raw_box = raw_det.get("box_lidar")
                if not isinstance(raw_box, list) or len(raw_box) < 7:
                    continue
                distance = math.hypot(float(raw_box[0]) - float(box[0]),
                                      float(raw_box[1]) - float(box[1]))
                if best is None or distance < best[0]:
                    best = (distance, index, raw_box)
            if best is None or best[0] > float(max_distance):
                missing += 1
                continue
            used.add(best[1])
            raw_box = best[2]
            if abs(float(raw_box[6]) - float(box[6])) > 1e-6:
                shifted.append({
                    "frame_id": ts,
                    "track_id": det.get("track_id"),
                    "car_stack_yaw_deg": round(math.degrees(float(box[6])), 2),
                    "detector_yaw_deg": round(math.degrees(float(raw_box[6])), 2),
                })
            det["box_lidar"] = list(raw_box)
            det["_car_stack_geometry_restored"] = True
            restored += 1
    Path(out_json).write_text(
        json.dumps(frames, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return {"restored": restored, "unmatched": missing,
            "yaw_restored": len(shifted), "examples": shifted[:10]}


def run(raw_json: Path, clip: Path, work_root: Path, python: Path,
        *, keep_classes: str = "Truck", score_threshold: float = 0.2,
        min_lifecycle: int = 4, overwrite: bool = True) -> Path:
    """跑 main_chain 的 step2 + step4.5，返回 step45 的 JSON 路径。"""
    for script in (STEP2_SCRIPT, STEP45_SCRIPT):
        if not script.is_file():
            raise RuntimeError(f"main_chain 缺少脚本: {script}")
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    # ---- main_chain step2（类过滤 + 跟踪 + 静态 yaw + yaw integrated）----
    step2_json = work_root / f"{clip.name}_step2.json"
    step2_diag = work_root / f"{clip.name}_step2_diagnostics.json"
    _run([python, STEP2_SCRIPT,
          "--in-json", Path(raw_json), "--clip", Path(clip),
          "--out-json", step2_json, "--diagnostics", step2_diag,
          "--keep-classes", str(keep_classes),
          # Truck 链的阈值（main_chain step2 默认 score 0.3 / lifecycle 4 是给轿车的）
          "--score-threshold", str(float(score_threshold)),
          "--min-lifecycle", str(int(min_lifecycle))])

    # ---- main_chain step4.5 ----
    # 批量脚本按 Car 的目录约定找输入：<step4-work-root>/<clip>_step4.json
    # 与 <clip-root>/<clip>_step3/（只用来读点云与标定），这里把 step2 的产物
    # 当成 "step4" 输入，并用软链造出 <clip>_step3 目录。
    step45_in = work_root / "step45_in"
    step45_in.mkdir(parents=True, exist_ok=True)
    shutil.copy(step2_json, step45_in / f"{clip.name}_step4.json")
    clip_root = work_root / "clip_root"
    clip_root.mkdir(parents=True, exist_ok=True)
    clip_link = clip_root / f"{clip.name}_step3"
    if clip_link.is_symlink() or clip_link.exists():
        clip_link.unlink()
    clip_link.symlink_to(Path(clip).resolve(), target_is_directory=True)

    step45_root = work_root / "step45"
    step45_root.mkdir(parents=True, exist_ok=True)
    out_json = step45_root / f"{clip.name}_step45.json"
    _run([python, STEP45_SCRIPT,
          "--step4-work-root", step45_in,
          "--step2-work-root", work_root,
          "--clip-root", clip_root,
          "--work-root", step45_root] + (["--overwrite"] if overwrite else []))
    if not out_json.is_file():
        raise RuntimeError(f"step4.5 没有产出: {out_json}")
    # main_chain 只负责 id：几何/yaw 还原成 detector 原值，交给本链 step3 定稿。
    report = _restore_geometry(out_json, Path(raw_json))
    diag_path = out_json.with_name(
        out_json.name.replace("_step45.json", "_step45_diagnostics.json"))
    if diag_path.is_file():
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        diag["geometry_restore"] = report
        diag_path.write_text(json.dumps(diag, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"[car-style-track] 几何还原: {report['restored']} 框 / 未匹配 "
          f"{report['unmatched']} / 其中 yaw 被 Car 栈改过 {report['yaw_restored']} 框",
          flush=True)
    return out_json


def main() -> int:                      # 方便单独调试
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--keep-classes", default="Truck")
    parser.add_argument("--score-threshold", type=float, default=0.2)
    args = parser.parse_args()
    out = run(args.raw_json, args.clip, args.work_root, args.python,
              keep_classes=args.keep_classes,
              score_threshold=args.score_threshold)
    frames = json.loads(Path(out).read_text())
    print("step45:", out, len(frames), "帧",
          sum(len(f.get("detections", [])) for f in frames), "框")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
