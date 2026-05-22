"""
LiDAR depth-supervision target.

Lift-Splat's `DepthNet` otherwise learns depth only as a side-effect of the
detection loss, leaving the splat geometrically mushy. nuScenes ships a LiDAR
sweep with every keyframe; projecting those points into a camera gives sparse
ground-truth depth. This module produces, per camera frame, a depth-bin-index
map at the backbone's C4 feature resolution (the resolution `DepthNet` predicts
at), with -1 where no LiDAR point landed — those cells are ignored by the loss.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

FEAT_STRIDE = 16 # the BEV detector lifts the C4 feature map (stride 16)


def lidar_depth_bins(nusc, data_root: str | Path, cam_sd_token: str, image_size: Tuple[int, int], dbound: Tuple[float, float, float]) -> np.ndarray:
    """
    Project the keyframe's LiDAR sweep into a camera and bin the depths onto the
    C4 feature grid.
    Args:
      nusc — NuScenes instance; data_root — dataset root.
      cam_sd_token — camera sample_data token.
      image_size — (H, W) the model input is resized to.
      dbound — (depth_min, depth_max, depth_step) — the DepthNet depth bins.
    Returns: (Hf, Wf) int64 array of depth-bin indices; -1 = no LiDAR return.
    """
    cam_sd = nusc.get("sample_data", cam_sd_token)
    sample = nusc.get("sample", cam_sd["sample_token"])
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])

    pc = LidarPointCloud.from_file(str(Path(data_root) / lidar_sd["filename"]))
    # LiDAR sensor → ego (LiDAR timestamp)
    cs_l = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs_l["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_l["translation"]))
    # ego (LiDAR timestamp) → global
    ego_l = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    pc.rotate(Quaternion(ego_l["rotation"]).rotation_matrix)
    pc.translate(np.array(ego_l["translation"]))
    # global → ego (camera timestamp)
    ego_c = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    pc.translate(-np.array(ego_c["translation"]))
    pc.rotate(Quaternion(ego_c["rotation"]).rotation_matrix.T)
    # ego (camera timestamp) → camera
    cs_c = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    pc.translate(-np.array(cs_c["translation"]))
    pc.rotate(Quaternion(cs_c["rotation"]).rotation_matrix.T)

    depths = pc.points[2, :]
    K = np.array(cs_c["camera_intrinsic"])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        uv = view_points(pc.points[:3, :], K, normalize=True)[:2, :]

    H, W = image_size
    Hf, Wf = H // FEAT_STRIDE, W // FEAT_STRIDE
    dmin, dmax, dstep = dbound
    n_bins = int(round((dmax - dmin) / dstep))

    # native pixels → resized image → feature cells
    fu = uv[0] * (W / cam_sd["width"]) / FEAT_STRIDE
    fv = uv[1] * (H / cam_sd["height"]) / FEAT_STRIDE
    keep = ((depths > dmin) & (depths < dmax)
            & (fu >= 0) & (fu < Wf) & (fv >= 0) & (fv < Hf))
    fu = fu[keep].astype(np.int64)
    fv = fv[keep].astype(np.int64)
    dep = depths[keep]

    bins = np.full((Hf, Wf), -1, dtype=np.int64)
    best = np.full((Hf, Wf), np.inf)
    for u, v, d in zip(fu, fv, dep): # keep the nearest point per cell
        if d < best[v, u]:
            best[v, u] = d
            bins[v, u] = min(int((d - dmin) / dstep), n_bins - 1)
    return bins
