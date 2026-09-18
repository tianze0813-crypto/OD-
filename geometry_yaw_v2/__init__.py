"""新版 yaw 管线（apply_motion_yaw=False：动态段保留 detector yaw）。

从 main_chain/geometry 隔离出来的副本，只服务 Truck 链，
不影响 geometry/ 下非车链在用的旧版 yaw。
"""
