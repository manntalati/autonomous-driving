from pathlib import Path
from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader
from data.dataset import NuScenesDetectionDataset, collate_fn, CLASS_NAMES, version_from_data_root
from data.seg_dataset import NuScenesSegmentationDataset, seg_collate_fn
from utils.visualize import visualize_batch

NUM_WORKERS = 2


def get_loaders(
    nusc: NuScenes = None,
    cameras=None,
    data_root: str = "data/raw/v1.0-mini",
    batch_size: int = 4,
    train_transform=None,
):
    """
    Build train and val DataLoaders for nuScenes detection.
    Args: nusc — NuScenes instance (created internally if None); cameras — list of camera names;
          data_root — path to v1.0-mini; batch_size — samples per batch.
    Returns: (train_loader, val_loader) — shuffled train, ordered val, both with pin_memory=True.
    """
    if nusc is None:
        nusc = NuScenes(version=version_from_data_root(data_root), dataroot=str(data_root), verbose=False)
    data_root = Path(data_root)
    # train_transform lets a config swap in the P13 cross-camera augmentation
    # without touching the val pipeline — the val set must stay undegraded or the
    # comparison against every earlier phase breaks.
    train_ds = NuScenesDetectionDataset(nusc, data_root, split="train", cameras=cameras,
                                        transform=train_transform)
    val_ds   = NuScenesDetectionDataset(nusc, data_root, split="val", cameras=cameras)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_seg_loaders(
    nusc: NuScenes = None,
    cameras=None,
    data_root: str = "data/raw/v1.0-mini",
    mask_dir: str = None,
    batch_size: int = 4,
):
    """
    Build train and val DataLoaders for nuScenes segmentation.
    Args: same as get_loaders, plus mask_dir — directory of cached seg masks
          (default {data_root}/seg_masks).
    Returns: (train_loader, val_loader).
    Notes: uses NuScenesSegmentationDataset + seg_collate_fn (stacks masks into a single tensor).
    """
    if nusc is None:
        nusc = NuScenes(version=version_from_data_root(data_root), dataroot=str(data_root), verbose=False)
    data_root = Path(data_root)
    train_ds = NuScenesSegmentationDataset(nusc, data_root, split="train", cameras=cameras, mask_dir=mask_dir)
    val_ds   = NuScenesSegmentationDataset(nusc, data_root, split="val",   cameras=cameras, mask_dir=mask_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=seg_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=seg_collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader
