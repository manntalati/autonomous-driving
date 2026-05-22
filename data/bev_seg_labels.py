"""
BEV semantic-map label generator.

Phase 3's `seg_labels.py` projects nuScenes map-expansion polygons through a
camera into the *image* plane. For the bird's-eye-view stage there is no camera
projection: the polygons are already in the world frame, so they are simply
transformed into the ego frame and rasterized straight into the top-down BEV
grid. The result is a per-cell class map (background / drivable / lane /
ped_crossing / walkway) aligned with the BEV detection grid.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2
from pyquaternion import Quaternion

from data.seg_labels import load_map_for_log, sample_map_polygons, CLASS_MAP


def build_bev_seg_label(nusc, cam_sd_token: str, xbound: Tuple[float, float, float], ybound: Tuple[float, float, float], radius: float = 60.0) -> np.ndarray:
    """
    Rasterize the map around the ego vehicle into a top-down BEV class grid.
    Args:
      nusc — NuScenes instance.
      cam_sd_token — a camera sample_data token; its ego_pose defines the ego
        frame (use the same reference camera as the BEV boxes).
      xbound/ybound — BEV grid [lower, upper, cell_size] per axis.
      radius — metres of map to pull around the ego position.
    Returns: (X, Y) uint8 class grid, indexed [ix, iy] to match the BEV
      detection grid; higher class IDs paint over lower.
    Pipeline:
      1. resolve scene → log → NuScenesMap; read the camera's ego_pose.
      2. sample world-frame map polygons within `radius` of the ego.
      3. transform polygon vertices world → ego; map ego (x, y) → grid cells.
      4. cv2.fillPoly each class in ascending ID order.
    """
    sd = nusc.get("sample_data", cam_sd_token)
    sample = nusc.get("sample", sd["sample_token"])
    scene = nusc.get("scene", sample["scene_token"])
    nusc_map = load_map_for_log(nusc, scene["log_token"])

    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
    ego_t = np.array(ego_pose["translation"], dtype=np.float64)
    rot_inv = Quaternion(ego_pose["rotation"]).inverse.rotation_matrix  # world → ego

    nx = int(round((xbound[1] - xbound[0]) / xbound[2]))
    ny = int(round((ybound[1] - ybound[0]) / ybound[2]))
    mask = np.zeros((nx, ny), dtype=np.uint8)

    layer_polys = sample_map_polygons(nusc_map, (ego_t[0], ego_t[1]), radius)
    for class_id in sorted(CLASS_MAP):                  # ascending → higher paints over lower
        for layer in CLASS_MAP[class_id]:
            for poly_world in layer_polys.get(layer, []):
                n = poly_world.shape[0]
                p3d = np.concatenate([poly_world, np.zeros((n, 1))], axis=1)  # (n, 3), z=0
                # numpy's SIMD matmul raises spurious FP warnings on Apple Silicon
                # for large operands; inputs/outputs are finite — suppress the noise.
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    ego = (p3d - ego_t) @ rot_inv.T                          # world → ego
                ix = (ego[:, 0] - xbound[0]) / xbound[2]
                iy = (ego[:, 1] - ybound[0]) / ybound[2]
                # cv2 point is (col, row); we want mask[ix, iy] → point = (iy, ix)
                pts = np.stack([iy, ix], axis=1).round().astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(mask, [pts], color=int(class_id))
    return mask
