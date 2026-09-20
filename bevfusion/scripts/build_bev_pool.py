#!/usr/bin/env python3
"""编译 mmdet3d 版 BEVFusion 需要的 bev_pool CUDA 算子（JIT，缓存到 ~/.cache/torch_extensions）。

不动 mmdetection3d 仓库：编出来的模块会被注册成
projects.BEVFusion.bevfusion.ops.bev_pool.bev_pool_ext（相对 import 靠 sys.modules 注入）。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.4')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.6')     # RTX A4000 = sm_86
os.environ['PATH'] = os.path.join(os.environ['CUDA_HOME'], 'bin') + ':' + os.environ.get('PATH', '')

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

def _resolve_mmdet3d_root() -> str:
    candidates = [os.environ.get("MMDET3D_ROOT"),
                  str(Path.home() / "MMDetection" / "mmdetection3d"),
                  str(Path.home() / "桌面" / "MMDetection" / "mmdetection3d")]
    for cand in candidates:
        if cand and (Path(cand) / "projects" / "BEVFusion").is_dir():
            return str(cand)
    raise SystemExit("找不到 mmdetection3d 源码，请设 MMDET3D_ROOT")


MMDET = _resolve_mmdet3d_root()
OPS_ROOT = f'{MMDET}/projects/BEVFusion/bevfusion/ops'
OPS = f'{OPS_ROOT}/bev_pool'
VOX = f'{OPS_ROOT}/voxel'


_CFLAGS = ['-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__',
           '-D__CUDA_NO_HALF2_OPERATORS__']
_INC = [os.path.join(os.environ['CUDA_HOME'], 'include')]


def build_voxel_layer():
    return load(
        name='voxel_layer',
        sources=[f'{VOX}/src/voxelization.cpp', f'{VOX}/src/scatter_points_cpu.cpp',
                 f'{VOX}/src/scatter_points_cuda.cu', f'{VOX}/src/voxelization_cpu.cpp',
                 f'{VOX}/src/voxelization_cuda.cu'],
        # 仓库 setup.py 是靠 define_macros 加 WITH_CUDA 的，JIT load 没有这个参数 -> 直接塞编译宏
        extra_cflags=['-DWITH_CUDA'], extra_include_paths=_INC,
        extra_cuda_cflags=_CFLAGS, verbose=True)


def build():
    return load(
        name='bev_pool_ext',
        sources=[f'{OPS}/src/bev_pool.cpp', f'{OPS}/src/bev_pool_cuda.cu'],
        # 仓库 setup.py 是靠 define_macros 加 WITH_CUDA 的，JIT load 没有这个参数 -> 直接塞编译宏
        extra_cflags=['-DWITH_CUDA'], extra_include_paths=_INC,
        extra_cuda_cflags=_CFLAGS, verbose=True)


def register():
    """编好两个算子并注册到项目模块路径下（这样不用改 mmdetection3d 仓库）。"""
    exts = {}
    for mod_name, ext in (('bev_pool', build()), ('voxel', build_voxel_layer())):
        full = (f'projects.BEVFusion.bevfusion.ops.{mod_name}'
                f'.{"bev_pool_ext" if mod_name == "bev_pool" else "voxel_layer"}')
        sys.modules[full] = ext
        exts[mod_name] = ext
    return exts


if __name__ == '__main__':
    exts = register()
    print('算子就绪:', {k: v.__file__ for k, v in exts.items()})
    print(torch.cuda.get_device_name(0))
