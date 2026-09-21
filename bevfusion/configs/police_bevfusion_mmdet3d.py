# 上游基础配置已 vendor 到 ./upstream（见 bevfusion/README.md）。
# data_root 只是占位：infer_mmdet3d.py 会用 BEVFUSION_ROOT / BEVFUSION_DATA_ROOT 覆盖成绝对路径。
import os


# mmdet3d 1.x 版 BEVFusion 推理配置（交警 0914 clip）
# 权重: checkpoints/bevfusion_mmdet3d_lidarcam.pth
#       = bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth
#       nuScenes val NDS 71.4 / mAP 68.6（truck AP 0.643 / bus 0.764 / trailer 0.483）
_base_ = [
    './upstream/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# 类别与 base 一致（派生配置里 base 的变量名不可见，重新声明一遍）
class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
               'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
metainfo = dict(classes=class_names)

# ---- 数据：middle format infos（scripts/mmdet3d_prep.py 生成）----
data_root = os.environ.get('BEVFUSION_DATA_ROOT', 'bevfusion/data/police/')
data_prefix = dict(pts='', cam_front='', cam_left='', cam_right='', cam_rear='', sweeps='')
input_modality = dict(use_lidar=True, use_camera=True)
# ---- 检测范围（本项目约定；两个 police config 必须保持一致，有 tests/test_bevfusion_range.py 守着）----
# 坐标系：数据 lidar_top 系里 前进 = -y、侧向 = ±x（用相机外参视线方向 + pose_data.txt 运动方向双重验证）。
# 上游 nuScenes 权重是按 ±54 m 对称 range 训练的 -> 原来【前方只能看到 54 m】，比标注 ROI 的前 80 m 少 26 m。
# 现在：侧向 x 保持 ±54（比 ROI 的 ±40 宽，给检测留上下文，面积与原来基本持平，实测 91 ms/帧 vs 92）；
#      前进 y 扩到 -80.4、后退到 +20.4；z 不变。
# 网格顺序实测确认是 [x_cells, y_cells, z_cells]，且 cells = 长度/voxel 必须能被 8 整除（SECOND 骨干 3 次 stride-2）。
VOXEL_SIZE = [0.075, 0.075, 0.2]
point_cloud_range = [-54.0, -80.4, -5.0, 54.0, 20.4, 3.0]      # 108.0/0.075=1440  100.8/0.075=1344
BEV_GRID = [1440, 1344, 41]                                    # 1440/8=180  1344/8=168  8/0.2=40(+1)
POST_CENTER_RANGE = [-59.0, -85.4, -10.0, 59.0, 25.4, 10.0]

_range_override = dict(
    data_preprocessor=dict(voxelize_cfg=dict(
        point_cloud_range=point_cloud_range, voxel_size=VOXEL_SIZE)),
    pts_middle_encoder=dict(sparse_shape=BEV_GRID),
    bbox_head=dict(
        train_cfg=dict(point_cloud_range=point_cloud_range, grid_size=BEV_GRID,
                       voxel_size=VOXEL_SIZE),
        test_cfg=dict(grid_size=BEV_GRID, pc_range=point_cloud_range[:2],
                      voxel_size=VOXEL_SIZE[:2]),
        bbox_coder=dict(pc_range=point_cloud_range[:2],
                        post_center_range=POST_CENTER_RANGE,
                        voxel_size=VOXEL_SIZE[:2])),
)


# ---- 图像：1920x1536 去畸变针孔图 * 0.27 = 518x414 -> crop 成 384x512 ----
_final_dim = [384, 512]
_resize_lim = [0.27, 0.27]

# view_transform 的 image_size 必须等于最终图像尺寸，feature_size = /8
model = dict(_range_override,
    view_transform=dict(image_size=_final_dim, feature_size=[48, 64]),
    # base 里 img_backbone.init_cfg 指向 GitHub 的 swin_tiny 预训练（本机 SSL 不通）；
    # 反正整权重 ckpt 里就含 SwinT，直接在后面 load_checkpoint 覆盖 -> 关掉这次下载
    img_backbone=dict(init_cfg=None),
)

test_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True, color_type='color',
         backend_args=None),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5,
         backend_args=None),
    dict(type='ImageAug3D', final_dim=_final_dim, resize_lim=_resize_lim,
         bot_pct_lim=[0.0, 0.0], rot_lim=[0.0, 0.0], rand_flip=False, is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['img', 'points'],
         meta_keys=['cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
                    'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
                    'lidar_path', 'img_path', 'num_pts_feats'])
]

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesDataset',
        data_root=data_root,
        ann_file='infos/police_mmdet3d_infos.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        load_eval_anns=False,        # 我们只做推理+自有评测，不需要 nuScenes 格式 GT
        backend_args=None))
test_dataloader = val_dataloader
