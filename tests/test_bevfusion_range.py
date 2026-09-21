"""BEVFusion 检测范围的守卫测试。

两个 police config 里的范围数字必须一致、且自洽（网格能被 8 整除、覆盖标注 ROI）。
不需要 mmengine：config 文件本身是纯 python，直接按模块执行读取变量即可。
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_CONFIGS = Path(__file__).parents[1] / "bevfusion" / "configs"
_NAMES = ("police_bevfusion_mmdet3d_lidaronly.py", "police_bevfusion_mmdet3d.py")

# 标注 ROI：前 80 / 后 20 / 左右 40（filtering/hard_filters.py 的范围过滤）
_FRONT_MIN, _REAR_MIN, _SIDE_MIN = 80.0, 20.0, 40.0


def _load(name):
    path = _CONFIGS / name
    spec = importlib.util.spec_from_file_location(f"_cfg_{name[:-3]}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BevfusionRangeTest(unittest.TestCase):
    def setUp(self):
        self.cfgs = {name: _load(name) for name in _NAMES}

    def test_configs_agree(self):
        first = self.cfgs[_NAMES[0]]
        for name, cfg in self.cfgs.items():
            self.assertEqual(cfg.point_cloud_range, first.point_cloud_range, name)
            self.assertEqual(cfg.BEV_GRID, first.BEV_GRID, name)
            self.assertEqual(cfg.VOXEL_SIZE, first.VOXEL_SIZE, name)

    def test_grid_matches_range(self):
        for name, cfg in self.cfgs.items():
            xmin, ymin, zmin, xmax, ymax, zmax = cfg.point_cloud_range
            vx, vy, vz = cfg.VOXEL_SIZE
            expected = [round((xmax - xmin) / vx), round((ymax - ymin) / vy),
                        round((zmax - zmin) / vz) + 1]
            self.assertEqual(cfg.BEV_GRID, expected, name)
            # SECOND 骨干 3 次 stride-2 -> 网格必须能被 8 整除
            self.assertEqual(cfg.BEV_GRID[0] % 8, 0, name)
            self.assertEqual(cfg.BEV_GRID[1] % 8, 0, name)

    def test_covers_annotation_roi(self):
        # 前进 = -y，侧向 = ±x（数据 lidar_top 系，见 config 注释）
        for name, cfg in self.cfgs.items():
            xmin, ymin, _, xmax, ymax, _ = cfg.point_cloud_range
            self.assertGreaterEqual(-ymin, _FRONT_MIN, name)
            self.assertGreaterEqual(ymax, _REAR_MIN, name)
            self.assertGreaterEqual(xmax, _SIDE_MIN, name)
            self.assertGreaterEqual(-xmin, _SIDE_MIN, name)

    def test_model_override_carries_the_range(self):
        for name, cfg in self.cfgs.items():
            rng = cfg.point_cloud_range
            grid = cfg.BEV_GRID
            model = cfg.model
            self.assertEqual(model["data_preprocessor"]["voxelize_cfg"]["point_cloud_range"], rng, name)
            self.assertEqual(model["pts_middle_encoder"]["sparse_shape"], grid, name)
            head = model["bbox_head"]
            self.assertEqual(head["test_cfg"]["grid_size"], grid, name)
            self.assertEqual(head["test_cfg"]["pc_range"], rng[:2], name)
            self.assertEqual(head["bbox_coder"]["pc_range"], rng[:2], name)
            self.assertEqual(head["train_cfg"]["point_cloud_range"], rng, name)
            # 图像分支的 view_transform 等既有覆盖不能被范围覆盖冲掉（C+L config）
            if "view_transform" in model:
                self.assertIn("image_size", model["view_transform"], name)

    def test_test_pipeline_filters_with_the_range(self):
        for name, cfg in self.cfgs.items():
            filters = [step for step in cfg.test_pipeline
                       if step.get("type") == "PointsRangeFilter"]
            self.assertEqual(len(filters), 1, name)
            self.assertEqual(filters[0]["point_cloud_range"], cfg.point_cloud_range, name)


if __name__ == "__main__":
    unittest.main()
