"""
CAM_FRONT BEV detection dataset for nuScenes mini (Phase 5).

Each item yields:
  - the front camera image (resized + normalised),
  - the camera calibration the Lift-Splat-Shoot transform needs
    (intrinsics K, and the camera→ego rotation/translation),
  - the GT objects as bird's-eye-view boxes in the EGO frame.

A BEV box is [x, y, length, width, yaw] — x forward, y left (ego frame),
yaw measured in the ego XY plane. Height is dropped (BEV is top-down).
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from torch.utils.data import Dataset

from data.dataset import TRAIN_SCENES, VAL_SCENES, LABEL_MAP
from data.transforms import MEAN, STD

CAM = "CAM_FRONT"
_MEAN = np.array(MEAN, dtype=np.float32)
_STD = np.array(STD, dtype=np.float32)


class NuScenesBEVDataset(Dataset):
    """
    BEV detection dataset. Same scene-level train/val split as detection.
    Image augmentation is intentionally minimal (resize + normalise only):
    flips/crops would desync the image from the fixed BEV geometry unless the
    intrinsics are also transformed.
    """

    def __init__(self, nusc: NuScenes, data_root: str | Path, split: str = "train", image_size: Tuple[int, int] = (448, 800), xbound: Tuple[float, float, float] = (0.0, 51.2, 0.8), ybound: Tuple[float, float, float] = (-25.6, 25.6, 0.8)) -> None:
        """
        Args:
          nusc — NuScenes instance.
          data_root — path to v1.0-mini.
          split — "train" or "val".
          image_size — (H, W) the image is resized to; intrinsics are scaled to match.
          xbound/ybound — BEV grid extent; GT boxes outside it are dropped.
        """
        self.nusc = nusc
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.xbound = xbound
        self.ybound = ybound
        self.index = self._build_index()

    def _build_index(self) -> List[str]:
        """
        Walk all scenes in the split, collect the CAM_FRONT-bearing sample tokens.
        Returns: list of sample_token. Mirrors NuScenesDetectionDataset._build_index.
        """
        tokens: List[str] = []
        scenes = TRAIN_SCENES if self.split == "train" else VAL_SCENES
        for scene in self.nusc.scene:
            if scene["name"] not in scenes:
                continue
            token = scene["first_sample_token"]
            while token != "":
                sample = self.nusc.get("sample", token)
                if CAM in sample["data"]:
                    tokens.append(token)
                token = sample["next"]
        return tokens

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        """
        Returns: (image, target) where
          image  — (3, H, W) float tensor, resized + ImageNet-normalised.
          target — dict with:
            'boxes'     (N, 5) float — [x, y, length, width, yaw] BEV boxes, ego frame.
            'labels'    (N,)   long  — class ids (car/ped/cyclist).
            'intrinsic' (3, 3) float — K, scaled to image_size.
            'cam_to_ego' (4, 4) float — homogeneous camera→ego transform.
        """
        sample = self.nusc.get("sample", self.index[idx])
        cam_sd_token = sample["data"][CAM]
        sd = self.nusc.get("sample_data", cam_sd_token)

        img = Image.open(self.data_root / sd["filename"]).convert("RGB")
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = (img - _MEAN) / _STD
        image = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()

        K, cam_to_ego = self._get_calibration(cam_sd_token)
        boxes, labels = self._get_bev_boxes(cam_sd_token)
        target = {
            "boxes": torch.from_numpy(boxes).float(),
            "labels": torch.from_numpy(labels).long(),
            "intrinsic": torch.from_numpy(K).float(),
            "cam_to_ego": torch.from_numpy(cam_to_ego).float(),
        }
        return image, target

    def _get_calibration(self, cam_sd_token: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args: cam_sd_token — CAM_FRONT sample_data token.
        Returns: (K (3,3), cam_to_ego (4,4)) —
          K is the camera intrinsic, scaled so it matches self.image_size.
          cam_to_ego is the homogeneous transform from calibrated_sensor
          (rotation quaternion + translation).
        """
        sd = self.nusc.get("sample_data", cam_sd_token)
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

        K = np.array(cs["camera_intrinsic"], dtype=np.float32)
        # scale intrinsics from native resolution to image_size
        scale_x = self.image_size[1] / sd["width"]
        scale_y = self.image_size[0] / sd["height"]
        K = K.copy()
        K[0, :] *= scale_x
        K[1, :] *= scale_y

        cam_to_ego = np.eye(4, dtype=np.float32)
        cam_to_ego[:3, :3] = Quaternion(cs["rotation"]).rotation_matrix
        cam_to_ego[:3, 3] = np.array(cs["translation"], dtype=np.float32)
        return K, cam_to_ego

    def _get_bev_boxes(self, cam_sd_token: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project the sample's 3D GT boxes into ego-frame BEV boxes.
        Args: cam_sd_token — CAM_FRONT sample_data token.
        Returns: (boxes (N,5)=[x,y,length,width,yaw], labels (N,)).
        """
        sd = self.nusc.get("sample_data", cam_sd_token)
        ego_pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
        ego_t = np.array(ego_pose["translation"])
        ego_rot_inv = Quaternion(ego_pose["rotation"]).inverse

        boxes_out: List[List[float]] = []
        labels_out: List[int] = []
        for box in self.nusc.get_boxes(cam_sd_token):
            if box.name not in LABEL_MAP:
                continue
            # global → ego frame
            center = ego_rot_inv.rotate(np.array(box.center) - ego_t)
            yaw = (ego_rot_inv * box.orientation).yaw_pitch_roll[0]
            width, length, _ = box.wlh
            x, y = float(center[0]), float(center[1])
            if not (self.xbound[0] <= x < self.xbound[1]):
                continue
            if not (self.ybound[0] <= y < self.ybound[1]):
                continue
            boxes_out.append([x, y, float(length), float(width), float(yaw)])
            labels_out.append(LABEL_MAP[box.name])

        boxes = np.array(boxes_out, dtype=np.float32).reshape(-1, 5)
        labels = np.array(labels_out, dtype=np.int64)
        return boxes, labels

def bev_collate_fn(batch):
    """
    Stack images + calibration into tensors; keep boxes/labels as lists
    (variable object counts per image).
    Returns: (images (B,3,H,W), targets) where targets is a list of dicts.
    """
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets
