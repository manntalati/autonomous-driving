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


if __name__ == "__main__":
    print("Loading NuScenes...")
    nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA_ROOT), verbose=False)

    train_loader, val_loader = get_loaders(nusc)

    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")
    print(f"Train samples : {len(train_loader.dataset)}")
    print(f"Val samples   : {len(val_loader.dataset)}")

    # ── Visual sanity check — first train batch ──────────────────
    print("\nFetching first train batch...")
    images, targets = next(iter(train_loader))
    print(f"Batch image shape : {images.shape}")
    for i, t in enumerate(targets):
        label_names = [CLASS_NAMES[l] for l in t['labels'].tolist()]
        print(f"  [{i}] {t['boxes'].shape[0]} boxes — {label_names}")

    print("\nVisualizing batch (close window to continue)...")
    visualize_batch(images, targets, CLASS_NAMES)

    # ── Val batch check ──────────────────────────────────────────
    print("Fetching first val batch...")
    images, targets = next(iter(val_loader))
    print(f"Val batch image shape : {images.shape}")
    print("\nAll checks passed. P0-4 complete.")
