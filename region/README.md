# Parking / dynamic region prototype

## Agreed direction: high-speed dynamic regions

The reviewed direction is:

> an area is dynamic only when sustained high-speed driving is observed;
> everything else defaults to static / dense parking.

This avoids brittle parking-aisle width thresholds and keeps dense parking,
3 m aisles and road-side parking in the default static class.

High-speed evidence (agreed defaults):

- p90 speed >= `5.0 m/s`;
- robust path length >= `15.0 m`;
- at least `3` steps above the threshold;
- consecutive observations are connected unless the time gap exceeds
  `2.0 s`;
- the **swept vehicle box** between consecutive observations is rasterized
  (not just the centre line), then buffered by `1.0 m` on each side and
  connected into polygons.

ID-switch protection:

- a high-speed track whose observations repeatedly land on static slot
  centres (default `>= 50%` within `1.5 m`) is rejected;
- static slot footprints (plus `0.5 m`) are removed from the final dynamic
  mask.

Only dynamic polygons are written.  The complement is the default static
region.

CLI:

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python tools/build_dynamic_regions.py \
  --raw-json /path/to/raw.json \
  --clip /path/to/clip \
  --out-dir work/dynamic_regions/<name>
```

For an existing step2 result:

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python tools/build_dynamic_regions.py \
  --clip /path/to/clip \
  --out-dir work/dynamic_regions/<name> \
  --step2-json /path/to/step2.json \
  --step2-diagnostics /path/to/step2_diagnostics.json
```

For a SUSTechPOINTS `*_pre` clip without raw inference JSON:

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python tools/build_dynamic_regions.py \
  --clip /path/to/clip_pre \
  --prelabel-clip /path/to/clip_pre \
  --out-dir work/dynamic_regions/<name>
```

Outputs:

- `dynamic_regions.json`: high-speed dynamic polygons, track evidence and
  diagnostics;
- `dynamic_regions_overlay.png`: orange = dynamic region, blue = high-speed
  track, gray = other dynamic track, red x = static slot;
- `summary.json`.

## Parking-region prototype (earlier experiment)

The earlier `parking_region.py` / `build_parking_regions.py` prototype is kept
for comparison only.  Its logic is the opposite of the agreed direction: it
clusters static slots into parking rectangles and subtracts dynamic corridors.
Use `build_dynamic_regions.py` for the reviewed high-speed-dynamic-region
approach.


## Inputs

- static slot records from `step2_diagnostics.json`
  (`tracking.slot_details`);
- static detection points assigned to those slots (world frame);
- dynamic detection points from tracks with real motion evidence.

## Algorithm

1. Cluster static slots by the gap between their oriented vehicle rectangles
   plus heading compatibility.  This keeps separate rows / blocks separate and
   avoids one giant connected region.
2. Split oversized clusters recursively:
   - prefer the largest natural gap along the row / heading axes;
   - only accept a split when both sides keep at least
     `min_slots_per_region` slots, so no 1-3 slot edge fragments are created;
   - if there is no clear aisle, split the longer dimension near its median.
3. Each final cluster becomes an **oriented bounding rectangle** (4 corners)
   with a small margin.  Multiple rectangles are preferred over one large
   polygon.
4. Build a road seed from robust dynamic tracks:
   - a track must have enough net displacement or mean speed;
   - consecutive observations are rasterized as polyline segments;
   - segments across long occlusion gaps are skipped;
   - road seed cells are clipped against the parking rectangles so a
     parking-lot maneuver cannot delete a parking region.
5. Dilate the road seed; road polygons are written separately.
6. Keep only parking rectangles with enough slots and area.

## CLI

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python tools/build_parking_regions.py \
  --raw-json /home/moga/桌面/pandarset/test_090400/clip1_vn_waymo_e10.json \
  --clip /home/moga/桌面/new/scene_crossroad_my_record_20260803_090400_clip1 \
  --out-dir work/parking_regions/region_090400_clip1
```

`regions_overlay.png` 图例：

- 绿色填充多边形：停车区域（parking region）
- 橙色填充多边形：道路走廊（road corridor）
- 浅绿色小点：静态检测点
- 灰色小点：动态检测点
- 无颜色区域：证据不足，既不算停车区也不算道路

`regions.json` 里也写入了同样的 `legend` 字段。

Existing step2 outputs can be supplied with `--step2-json` and
`--step2-diagnostics` to skip the step2 run.

## Current defaults

- `resolution=1.0 m`
- `slot_footprint_extra=0.3 m`
- `cluster_connect_gap=1.0 m`
- `cluster_max_yaw_delta=0.8 rad`
- `rectangle_margin=0.5 m`
- `rectangle_max_slots=30`
- `rectangle_max_extent=30.0 m`
- `rectangle_min_split_gap=4.5 m`
- `min_slots_per_region=4`
- `road_min_track_net_displacement=5.0 m` (or mean speed >= 1.0 m/s)
- `road_dilate_radius=2.5 m`

The thresholds are deliberately exposed as CLI flags for tuning on the
090400 / 163412 clips.
