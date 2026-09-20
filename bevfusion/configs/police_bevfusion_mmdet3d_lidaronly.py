# 上游基础配置已 vendor 到 ./upstream（见 bevfusion/README.md）。
# data_root 只是占位：infer_mmdet3d.py 会用 BEVFUSION_ROOT / BEVFUSION_DATA_ROOT 覆盖成绝对路径。
import os

# 消融用：mmdet3d 1.x 纯雷达 BEVFusion (TransFusion-L 同族)，nuScenes val NDS 69.6 / mAP 64.9
_base_ = [
    './upstream/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]
class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
               'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
metainfo = dict(classes=class_names)

data_root = os.environ.get('BEVFUSION_DATA_ROOT', 'bevfusion/data/police/')
data_prefix = dict(pts='', sweeps='')
input_modality = dict(use_lidar=True, use_camera=False)
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, backend_args=None),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points'],
         meta_keys=['box_type_3d', 'sample_idx', 'lidar_path', 'num_pts_feats']),
]

val_dataloader = dict(
    batch_size=1, num_workers=0, persistent_workers=False, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='NuScenesDataset', data_root=data_root,
                 ann_file='infos/police_mmdet3d_infos.pkl', pipeline=test_pipeline,
                 metainfo=metainfo, modality=input_modality, test_mode=True,
                 data_prefix=data_prefix, box_type_3d='LiDAR',
                 load_eval_anns=False, backend_args=None))
test_dataloader = val_dataloader
