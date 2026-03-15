from pathlib import Path
from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader
from data.dataset import NuScenesDetectionDataset, collate_fn, CLASS_NAMES
from utils.visualize import visualize_batch

DATA_ROOT = Path("data/raw/v1.0-mini")
BATCH_SIZE = 4
NUM_WORKERS = 2


def get_loaders(nusc: NuScenes, cameras=None):
    train_ds = NuScenesDetectionDataset(nusc, DATA_ROOT, split="train", cameras=cameras)
    val_ds   = NuScenesDetectionDataset(nusc, DATA_ROOT, split="val", cameras=cameras)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader