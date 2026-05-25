from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from nuscenes.nuscenes import NuScenes
from torch.utils.data import Dataset

from data.dataset import get_scene_split
from data.transforms import get_seg_train_transforms, get_seg_val_transforms


class NuScenesSegmentationDataset(Dataset):
    """
    Loads (image, seg_mask) pairs. Masks come from data/seg_labels.py (pre-cached).
    Same scene-level train/val split as detection.
    """

    def __init__(self, nusc: NuScenes, data_root: str | Path, split: str = "train", cameras: Optional[List[str]] = None, mask_dir: str | Path | None = None, transform=None):
        """
        Args:
          nusc — NuScenes instance.
          data_root — dataset root.
          split — "train" or "val".
          cameras — list of camera channels (default ["CAM_FRONT"]).
          mask_dir — directory of cached {sample_token}_{cam}.png masks
                     (default {data_root}/seg_masks).
          transform — albumentations pipeline (defaults to seg train/val by split).
        """
        self.nusc = nusc
        self.data_root = Path(data_root)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else self.data_root / "seg_masks"
        self.split = split
        self.cameras = cameras if cameras is not None else ["CAM_FRONT"]
        if transform is None:
            self.transform = get_seg_train_transforms() if split == "train" else get_seg_val_transforms()
        else:
            self.transform = transform
        self.index = self._build_index()

    def _build_index(self) -> List[Tuple[str, str]]:
        """
        Walk scenes in split, collect (sample_token, camera) pairs for every keyframe
        that has a cached mask file.
        Returns: list of (sample_token, camera_name).
        Notes:
          - Skip pairs where mask file is missing (warn once).
          - Logic mirrors NuScenesDetectionDataset._build_index for parity.
        """
        keyframes: List[Tuple[str, str]] = []
        train_scenes, val_scenes = get_scene_split(self.nusc, self.data_root)
        scenes = train_scenes if self.split == "train" else val_scenes
        warned = False
        for scene in self.nusc.scene:
            if scene["name"] not in scenes:
                continue
            token = scene["first_sample_token"]
            while token != "":
                sample = self.nusc.get("sample", token)
                for camera in self.cameras:
                    if camera not in sample["data"]:
                        continue
                    mask_path = self.mask_dir / f"{token}_{camera}.png"
                    if not mask_path.exists():
                        if not warned:
                            print(
                                f"[seg_dataset] warning: missing mask {mask_path} "
                                f"— skipping pairs without cached masks "
                                f"(run `python -m data.seg_labels`). Further warnings suppressed."
                            )
                            warned = True
                        continue
                    keyframes.append((token, camera))
                token = sample["next"]
        return keyframes

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load image + cached mask, apply joint transforms.
        Returns: (image_tensor (3, H, W) float, mask_tensor (H, W) long).
        Pipeline:
          1. Resolve image path via nusc.get('sample_data', ...)['filename'].
          2. Load mask PNG from self.mask_dir.
          3. albumentations Compose(...)(image=..., mask=...) — applies same crops/flips to both.
          4. Cast mask to long; return.
        """
        sample_token, camera = self.index[idx]
        sample = self.nusc.get("sample", sample_token)
        sd_token = sample["data"][camera]
        sd = self.nusc.get("sample_data", sd_token)

        image = np.array(Image.open(self.data_root / sd["filename"]).convert("RGB"))
        mask_path = self.mask_dir / f"{sample_token}_{camera}.png"
        mask = np.array(Image.open(mask_path))

        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"].long()
        return image_tensor, mask_tensor


def seg_collate_fn(batch):
    """
    Stack images (B,3,H,W) and masks (B,H,W). All shapes equal post-transform so default stack works.
    """
    images = torch.stack([b[0] for b in batch])
    masks  = torch.stack([b[1] for b in batch])
    return images, masks
