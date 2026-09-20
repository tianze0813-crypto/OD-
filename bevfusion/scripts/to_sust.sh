#!/bin/bash
# 把任意一份 label 目录写成 SUST 可直接打开的数据集（不改源数据）。
# 用法: bash bevfusion/scripts/to_sust.sh <源clip目录> <label目录> <SUST数据集名>
#   SUST 根目录可用 SUST_ROOT 覆盖（默认 ~/桌面/SUSTechPOINTS/data）
set -euo pipefail
SRC="$1"; LABELS="$2"; NAME="$3"
SUST="${SUST_ROOT:-$HOME/桌面/SUSTechPOINTS/data}"
DST="$SUST/$NAME"
[ -d "$SRC" ] || { echo "[skip] 源 clip 不存在: $SRC"; exit 1; }
[ -d "$LABELS" ] || { echo "[skip] label 目录不存在: $LABELS"; exit 1; }
rm -rf "$DST"; mkdir -p "$DST"
# 源数据盘与 SUST 目录若不在同一分区，只能软链（硬链会报 invalid cross-device link）
for sub in image lidar transforms readme.json; do
  [ -e "$SRC/$sub" ] && ln -s "$SRC/$sub" "$DST/$sub"
done
mkdir -p "$DST/label"
cp "$LABELS"/*.json "$DST/label/"
echo "$NAME: $(ls "$DST/label" | wc -l) 帧 label -> $DST"
