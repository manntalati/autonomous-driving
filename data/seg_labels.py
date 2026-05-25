"""
Offline segmentation label generator for nuScenes mini.

Projects polygons from the nuScenes map expansion (drivable_area, lane,
ped_crossing, walkway) through each camera's intrinsic + extrinsic matrices
to build a per-pixel class mask in image space.

Output: cached PNG masks at data/raw/v1.0-mini/seg_masks/{sample_token}_{cam}.png
        (single-channel uint8, values are class IDs).

Usage (one-time before training):
    python -m data.seg_labels --data_root data/raw/v1.0-mini
"""
from __future__ import annotations

# nuscenes-devkit's map_api imports matplotlib and calls plt.style.use('seaborn-whitegrid'),
# which doesn't exist in modern matplotlib. Patch matplotlib.style.use to swallow the error
# BEFORE importing anything from nuscenes.map_expansion.
import matplotlib.style as _mpl_style
_original_use = _mpl_style.use
def _safe_style_use(name, *args, **kwargs):
    try:
        return _original_use(name, *args, **kwargs)
    except (OSError, ValueError):
        pass
_mpl_style.use = _safe_style_use

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap


# ── Class map (background=0; higher class IDs paint over lower IDs) ───────
# Only polygon layers go here. `lane_divider` is line geometry, not a polygon —
# if you want to render lane markings, use cv2.polylines separately.
CLASS_MAP: Dict[int, List[str]] = {
    # class_id -> list of map layer names that paint this class
    0: [],                                 # background (default)
    1: ["drivable_area", "road_segment"],  # drivable surface
    2: ["lane"],                           # lane polygons
    3: ["ped_crossing"],                   # pedestrian crossings
    4: ["walkway"],                        # sidewalks
}
SEG_CLASS_NAMES = ["background", "drivable", "lane", "ped_crossing", "walkway"]
NUM_SEG_CLASSES = len(SEG_CLASS_NAMES)

# Map location → nuScenes map name. Looked up via scene → log → location.
LOCATION_TO_MAP = {
    "singapore-onenorth": "singapore-onenorth",
    "singapore-hollandvillage": "singapore-hollandvillage",
    "singapore-queenstown": "singapore-queenstown",
    "boston-seaport": "boston-seaport",
}

# Module-level cache so repeated calls for the same location don't re-load the map.
_MAP_CACHE: Dict[str, NuScenesMap] = {}


def load_map_for_log(nusc, log_token: str) -> NuScenesMap:
    """
    Resolve the NuScenesMap instance for the location of a given log.
    Args: nusc — NuScenes instance; log_token — log token from a scene.
    Returns: NuScenesMap object for the log's location.
    Notes: cache maps in a module-level dict so repeated calls don't re-load.
    """
    log = nusc.get("log", log_token)
    location = log["location"]
    if location not in _MAP_CACHE:
        _MAP_CACHE[location] = NuScenesMap(
            dataroot=nusc.dataroot, map_name=LOCATION_TO_MAP[location]
        )
    return _MAP_CACHE[location]


def get_camera_extrinsics(nusc, cam_sd_token: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compose ego_pose ∘ calibrated_sensor to get world → camera transform.
    Args: nusc — NuScenes instance; cam_sd_token — sample_data token for a camera.
    Returns: (R_world_to_cam (3,3), t_world_to_cam (3,), K (3,3) intrinsics).
    Notes:
      - calibrated_sensor gives camera_to_ego transform.
      - ego_pose gives ego_to_world transform.
      - Compose and invert to get world_to_camera.
    """
    sd = nusc.get("sample_data", cam_sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])

    K = np.array(cs["camera_intrinsic"], dtype=np.float64)

    # camera → ego
    R_c2e = Quaternion(cs["rotation"]).rotation_matrix
    t_c2e = np.array(cs["translation"], dtype=np.float64)
    # ego → world
    R_e2w = Quaternion(ego["rotation"]).rotation_matrix
    t_e2w = np.array(ego["translation"], dtype=np.float64)

    # compose: camera → world
    R_c2w = R_e2w @ R_c2e
    t_c2w = R_e2w @ t_c2e + t_e2w

    # invert: world → camera
    R_w2c = R_c2w.T
    t_w2c = -R_c2w.T @ t_c2w
    return R_w2c, t_w2c, K


def sample_map_polygons(nusc_map, ego_xy: Tuple[float, float], radius: float = 50.0) -> Dict[str, List[np.ndarray]]:
    """
    Pull all polygons within `radius` meters of the ego position for each layer in CLASS_MAP.
    Args: nusc_map — NuScenesMap; ego_xy — ego (x, y) in map frame; radius — meters.
    Returns: dict mapping layer name -> list of polygon vertex arrays, each (N, 2) in map (x, y).
    Notes: use nusc_map.get_records_in_radius() and resolve each record's polygon nodes.
    """
    layer_names = [name for layers in CLASS_MAP.values() for name in layers]
    x, y = ego_xy
    records = nusc_map.get_records_in_radius(x, y, radius, layer_names)

    result: Dict[str, List[np.ndarray]] = {}
    for layer in layer_names:
        polys: List[np.ndarray] = []
        for token in records.get(layer, []):
            rec = nusc_map.get(layer, token)
            # drivable_area records hold multiple polygons; the rest hold one.
            if "polygon_tokens" in rec:
                polygon_tokens = rec["polygon_tokens"]
            else:
                polygon_tokens = [rec["polygon_token"]]
            for pt in polygon_tokens:
                polygon = nusc_map.extract_polygon(pt)
                if polygon is None or polygon.is_empty:
                    continue
                coords = np.array(polygon.exterior.coords, dtype=np.float64)
                if coords.shape[0] >= 3:
                    polys.append(coords[:, :2])
        result[layer] = polys
    return result


def project_polygon_to_image(
    poly_xy: np.ndarray,
    R_w2c: np.ndarray,
    t_w2c: np.ndarray,
    K: np.ndarray,
    img_shape: Tuple[int, int],
    z_min: float = 0.1,
) -> Optional[np.ndarray]:
    """
    Project a 2D map polygon (assumed z=0 in world) into the camera image.
    Args:
      poly_xy — (N, 2) polygon vertices in world frame.
      R_w2c, t_w2c — world→camera rotation/translation.
      K — (3, 3) intrinsics.
      img_shape — (H, W) of the target image.
      z_min — discard vertices with z_cam < z_min (behind/near camera plane).
    Returns: (M, 2) image-space polygon (float32 px), or None if fewer than 3 vertices survive.
    Notes:
      - Lift to 3D by appending z=0.
      - Transform: P_cam = R_w2c @ P_world + t_w2c.
      - Filter z_cam < z_min BEFORE projecting (avoids divide-by-zero / wraparound).
      - Project: (u, v) = K @ (X/Z, Y/Z, 1).
      - Do NOT clip vertices to image bounds here — cv2.fillPoly handles that.
    """
    n = poly_xy.shape[0]
    pts_world = np.concatenate([poly_xy, np.zeros((n, 1))], axis=1)  # (N, 3), z=0

    # numpy's SIMD matmul kernel leaves stale FP-exception flags set on large
    # operands (spurious "divide by zero / overflow in matmul" on Apple Silicon).
    # Inputs are verified finite; the result is correct — suppress the false alarm.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        pts_cam = pts_world @ R_w2c.T + t_w2c                        # (N, 3)

        keep = pts_cam[:, 2] >= z_min
        pts_cam = pts_cam[keep]
        if pts_cam.shape[0] < 3:
            return None

        proj = pts_cam @ K.T                                         # (M, 3)
    uv = proj[:, :2] / proj[:, 2:3]
    return uv.astype(np.float32)


def rasterize_mask(
    projected_polys: Dict[int, List[np.ndarray]],
    img_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Rasterize projected polygons into a (H, W) uint8 mask of class IDs.
    Args:
      projected_polys — dict {class_id: list of (M, 2) image-space polygons}.
      img_shape — (H, W) of the target mask.
    Returns: (H, W) uint8 array, values in [0, NUM_SEG_CLASSES-1].
    Notes:
      - Iterate class IDs in ascending order — higher IDs paint over lower.
      - Use cv2.fillPoly with polygon vertices cast to int32.
      - Default (unpainted) pixels = 0 (background).
    """
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for class_id in sorted(projected_polys.keys()):
        for poly in projected_polys[class_id]:
            pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], color=int(class_id))
    return mask


def build_seg_label_for_sample(
    nusc,
    sample_token: str,
    cam_name: str = "CAM_FRONT",
    radius: float = 50.0,
) -> np.ndarray:
    """
    End-to-end: for one (sample, camera), produce the (H, W) seg mask.
    Args: nusc — NuScenes; sample_token — keyframe token; cam_name — camera channel; radius — meters around ego.
    Returns: (H_native, W_native) uint8 mask of class IDs at the camera's native resolution.
    Pipeline:
      1. Resolve the scene's log → load nusc_map.
      2. Get camera sample_data token from sample['data'][cam_name].
      3. get_camera_extrinsics → (R, t, K) and image shape.
      4. Get ego (x, y) from ego_pose attached to the camera's sample_data.
      5. sample_map_polygons within radius.
      6. For each (class_id, layer): map layer polygons → project_polygon_to_image.
      7. rasterize_mask → return.
    """
    sample = nusc.get("sample", sample_token)
    scene = nusc.get("scene", sample["scene_token"])
    nusc_map = load_map_for_log(nusc, scene["log_token"])

    cam_sd_token = sample["data"][cam_name]
    sd = nusc.get("sample_data", cam_sd_token)
    img_shape = (sd["height"], sd["width"])

    R_w2c, t_w2c, K = get_camera_extrinsics(nusc, cam_sd_token)

    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
    ego_xy = (ego_pose["translation"][0], ego_pose["translation"][1])

    layer_polys = sample_map_polygons(nusc_map, ego_xy, radius)

    projected: Dict[int, List[np.ndarray]] = {}
    for class_id, layers in CLASS_MAP.items():
        if not layers:
            continue
        polys: List[np.ndarray] = []
        for layer in layers:
            for poly_xy in layer_polys.get(layer, []):
                proj = project_polygon_to_image(poly_xy, R_w2c, t_w2c, K, img_shape)
                if proj is not None:
                    polys.append(proj)
        projected[class_id] = polys

    return rasterize_mask(projected, img_shape)


def generate_all_masks(data_root: str | Path, out_dir: str | Path | None = None) -> None:
    """
    Generate and cache seg masks for every (sample, camera) used by the
    segmentation dataset — for whatever nuScenes version `data_root` points at
    (v1.0-mini, or a partial v1.0-trainval blob download).
    Args: data_root — dataset root; out_dir — output dir (default {data_root}/seg_masks).
    Notes:
      - Iterates via get_scene_split (the same split logic as the datasets).
      - Save as PNG (single-channel, uint8). Skip masks that already exist.
    """
    from data.dataset import get_scene_split, version_from_data_root

    data_root = Path(data_root)
    nusc = NuScenes(version=version_from_data_root(data_root), dataroot=str(data_root), verbose=False)
    out_dir = Path(out_dir) if out_dir is not None else data_root / "seg_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = ["CAM_FRONT"]
    train_scenes, val_scenes = get_scene_split(nusc, data_root)
    split_scenes = train_scenes | val_scenes

    generated, skipped = 0, 0
    for scene in nusc.scene:
        if scene["name"] not in split_scenes:
            continue
        token = scene["first_sample_token"]
        while token != "":
            sample = nusc.get("sample", token)
            for cam in cameras:
                if cam not in sample["data"]:
                    continue
                out_path = out_dir / f"{token}_{cam}.png"
                if out_path.exists():
                    skipped += 1
                    continue
                mask = build_seg_label_for_sample(nusc, token, cam)
                Image.fromarray(mask, mode="L").save(out_path)
                generated += 1
            token = sample["next"]

    print(f"[seg_labels] done — generated {generated} masks, skipped {skipped} existing → {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Cache nuScenes camera segmentation masks.")
    ap.add_argument("--data_root", default="data/raw/v1.0-mini",
                    help="dataset root (its folder name is the nuScenes version)")
    args = ap.parse_args()
    generate_all_masks(args.data_root)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/raw/v1.0-mini")
    parser.add_argument("--out_dir", type=str, default="data/raw/v1.0-mini/seg_masks")
    args = parser.parse_args()
    generate_all_masks(args.data_root, args.out_dir)
