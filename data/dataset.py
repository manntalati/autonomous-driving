from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
from torch.utils.data import Dataset
from data.transforms import get_train_transforms, get_val_transforms

LABEL_MAP: Dict[str, int] = {
    # ── Car (0) ───────────────────────────────────────────────────
    "vehicle.car":                    0,
    "vehicle.bus.bendy":              0,
    "vehicle.bus.rigid":              0,
    "vehicle.truck":                  0,
    "vehicle.trailer":                0,
    # ── Pedestrian (1) ────────────────────────────────────────────
    "human.pedestrian.adult":         1,
    "human.pedestrian.child":         1,
    "human.pedestrian.construction_worker": 1,
    "human.pedestrian.police_officer": 1,
    # ── Cyclist (2) ───────────────────────────────────────────────
    "vehicle.motorcycle":             2,
    "vehicle.bicycle":                2,
}

CLASS_NAMES = ["car", "pedestrian", "cyclist"]

# ── Train / val scene split ─────────────────────────────────────────────────
TRAIN_SCENES = {
    "scene-0061", "scene-0103", "scene-0553", "scene-0655",
    "scene-0757", "scene-0796", "scene-0916", "scene-1077",
}
VAL_SCENES = {
    "scene-1094", "scene-1100",
}

class NuScenesDetectionDataset(Dataset):

    def __init__(self, nusc: NuScenes, data_root: str | Path, split: str = "train", cameras: Optional[List[str]] = None, transform=None):
        self.nusc = nusc
        self.data_root = Path(data_root)
        self.split = split
        if cameras is None:
            self.cameras = ["CAM_FRONT"]
        else:
            self.cameras = cameras
        if transform is None:
            if self.split == "train":
                self.transform = get_train_transforms()
            else:
                self.transform = get_val_transforms()
        else:
            self.transform = transform
        self.index = self._build_index()


    def _build_index(self) -> List[Tuple[str, str]]:
        keyframes = []
        scenes = TRAIN_SCENES if self.split == 'train' else VAL_SCENES
        for scene in self.nusc.scene:
            if scene['name'] not in scenes: continue
            token = scene['first_sample_token']
            while token != '':
                sample = self.nusc.get('sample', token)
                for camera in self.cameras:
                    if camera in sample['data']:
                        keyframes.append((token, camera))
                token = sample['next']
        return keyframes

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        sample_token, camera = self.index[idx]
        sample = self.nusc.get('sample', sample_token)
        sample_data_token = sample['data'][camera]
        pil_image, img_w, img_h = self._load_image(sample_data_token)
        boxes, labels = self._get_2d_boxes(sample_data_token, img_w, img_h)
        pil_image = np.array(pil_image)
        transformed = self.transform(image=pil_image, bboxes=boxes, labels=labels)
        image_tensor, boxes, labels = transformed['image'], transformed['bboxes'], transformed['labels']
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0,4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        targets_dict = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'meta': {'sample_token': sample_token, 'camera': camera}
        }
        return (image_tensor, targets_dict)

    def _load_image(self, sd_token: str) -> Tuple[Image.Image, int, int]:
        sample_data = self.nusc.get('sample_data', sd_token)
        filename = self.data_root / sample_data['filename']
        pil_image = Image.open(filename).convert('RGB')
        return (pil_image, sample_data['width'], sample_data['height'])

    def _get_2d_boxes(self, sd_token: str, img_w: int, img_h: int) -> Tuple[List[List[float]], List[int]]:
        boxes_2d, labels = [], []
        sample_data = self.nusc.get('sample_data', sd_token)
        calibrated_sensor = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        K = np.array(calibrated_sensor['camera_intrinsic'])
        _, boxes_raw, _ = self.nusc.get_sample_data(sd_token, box_vis_level=BoxVisibility.ANY)
        for box in boxes_raw:
            if box.name not in LABEL_MAP: continue
            points = view_points(box.corners(), K, normalize=True)
            x1 = float(np.clip(np.min(points[0]), 0, img_w))
            y1 = float(np.clip(np.min(points[1]), 0, img_h))
            x2 = float(np.clip(np.max(points[0]), 0, img_w))
            y2 = float(np.clip(np.max(points[1]), 0, img_h))
            if (x2 - x1) < 2 or (y2 - y1) < 2: continue
            boxes_2d.append([x1, y1, x2, y2])
            labels.append(LABEL_MAP[box.name])
        return boxes_2d, labels


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

# ── Sanity check ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    DATA_ROOT = Path("data/raw/v1.0-mini")
    nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA_ROOT), verbose=False)

    train_ds = NuScenesDetectionDataset(nusc, DATA_ROOT, split="train")
    val_ds   = NuScenesDetectionDataset(nusc, DATA_ROOT, split="val")

    print(f"Train samples : {len(train_ds)}")
    print(f"Val samples   : {len(val_ds)}")

    img, targets = train_ds[0]
    print(f"Image shape   : {img.shape}")
    print(f"Boxes         : {targets['boxes']}")
    print(f"Labels        : {[CLASS_NAMES[l] for l in targets['labels'].tolist()]}")

    loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                        num_workers=0, collate_fn=collate_fn)
    images, batch_targets = next(iter(loader))
    print(f"Batch shape   : {images.shape}")
    print("Sanity check passed.")