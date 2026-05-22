"""
Phase 5 training script — CAM_FRONT Lift-Splat-Shoot BEV detector.

Mirrors models/segmentation/train_seg.py: yaml config → build → epoch loop
with early stopping on validation loss.
"""
from __future__ import annotations
from pathlib import Path
from contextlib import nullcontext
import torch
import yaml
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes

from data.bev_dataset import NuScenesBEVDataset, bev_collate_fn
from models.bev.bev_detector import BEVDetector
from models.bev.losses import BEVDetectionLoss
from training.scheduler import build_scheduler, EarlyStopping

NUM_WORKERS = 2

def build_bev_detector(cfg: dict) -> BEVDetector:
    """Build the BEV detector from the config; load pretrained backbone if requested."""
    model = BEVDetector(
        num_classes=cfg["num_classes"],
        image_size=tuple(cfg.get("image_size", [448, 800])),
        xbound=tuple(cfg["xbound"]),
        ybound=tuple(cfg["ybound"]),
        zbound=tuple(cfg["zbound"]),
        dbound=tuple(cfg["dbound"]),
        bev_channels=cfg.get("bev_channels", 64),
    )
    if cfg.get("pretrained", False):
        model.load_pretrained()
    return model

def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _build_optimizer(model: BEVDetector, cfg: dict) -> torch.optim.Optimizer:
    """Differential LR: backbone gets backbone_lr_scale × lr; everything else full lr."""
    lr = cfg["lr"]
    backbone_scale = cfg.get("backbone_lr_scale", 0.1)
    wd = cfg.get("weight_decay", 1e-4)
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * backbone_scale},
            {"params": other_params, "lr": lr},
        ],
        weight_decay=wd,
    )

def get_bev_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders for the BEV dataset."""
    nusc = NuScenes(version="v1.0-mini", dataroot=cfg["data_root"], verbose=False)
    image_size = tuple(cfg.get("image_size", [448, 800]))
    train_ds = NuScenesBEVDataset(nusc, cfg["data_root"], split="train", image_size=image_size)
    val_ds = NuScenesBEVDataset(nusc, cfg["data_root"], split="val", image_size=image_size)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=bev_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=bev_collate_fn, pin_memory=True)
    return train_loader, val_loader

def _stack_calibration(targets: list, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pull the per-sample calibration out of the target dicts into batched tensors.
    Args: targets — list of B dicts with 'intrinsic' (3,3) and 'cam_to_ego' (4,4).
    Returns: (intrinsics (B,3,3), cam_to_ego (B,4,4)) on device.
    """
    intrinsics = torch.stack([t["intrinsic"] for t in targets]).to(device)
    cam_to_ego = torch.stack([t["cam_to_ego"] for t in targets]).to(device)
    return intrinsics, cam_to_ego

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, grad_clip: float = 1.0) -> dict:
    """
    One AMP training epoch. For each batch:
      1. images → device; _stack_calibration(targets) → intrinsics, cam_to_ego.
      2. heatmap, reg = model(images, intrinsics, cam_to_ego).
      3. loss, log = loss_fn(heatmap, reg, targets); backward; step (with grad clip).
    Returns: dict of averaged 'loss', 'hm_loss', 'reg_loss'.
    """
    model.train()
    total_loss = total_hm = total_reg = 0.0
    num_batches = 0
    amp_enabled = device.type == "cuda"
    for images, targets in loader:
        images = images.to(device)
        intrinsics, cam_to_ego = _stack_calibration(targets, device)
        optimizer.zero_grad()
        amp_ctx = torch.autocast(device_type="cuda") if amp_enabled else nullcontext()
        with amp_ctx:
            heatmap, reg = model(images, intrinsics, cam_to_ego)
            loss, log = loss_fn(heatmap, reg, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += log["loss"]
        total_hm += log["hm_loss"]
        total_reg += log["reg_loss"]
        num_batches += 1
    n = max(num_batches, 1)
    return {"loss": total_loss / n, "hm_loss": total_hm / n, "reg_loss": total_reg / n}

def val_one_epoch(model, loader, loss_fn, device) -> dict:
    """
    One eval epoch: model.eval(), no_grad, accumulate loss components.
    Returns: dict of averaged 'loss', 'hm_loss', 'reg_loss'.
    (BEV detection metric + visualisation is ticket P5-4, kept separate.)
    """
    model.eval()
    total_loss = total_hm = total_reg = 0.0
    num_batches = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            intrinsics, cam_to_ego = _stack_calibration(targets, device)
            heatmap, reg = model(images, intrinsics, cam_to_ego)
            _, log = loss_fn(heatmap, reg, targets)
            total_loss += log["loss"]
            total_hm += log["hm_loss"]
            total_reg += log["reg_loss"]
            num_batches += 1
    n = max(num_batches, 1)
    return {"loss": total_loss / n, "hm_loss": total_hm / n, "reg_loss": total_reg / n}

def main(cfg_path: str) -> None:
    """Load config → build model/optimizer/scheduler/loss → epoch loop with early stopping."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = _pick_device()
    train_loader, val_loader = get_bev_loaders(cfg)
    model = build_bev_detector(cfg).to(device)
    optimizer = _build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, scheduler_type=cfg.get("scheduler", "cosine"), epochs=cfg["epochs"])
    loss_fn = BEVDetectionLoss(
        num_classes=cfg["num_classes"],
        xbound=tuple(cfg["xbound"]),
        ybound=tuple(cfg["ybound"]),
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    grad_clip = cfg.get("grad_clip", 1.0)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    early_stop = EarlyStopping(
        patience=cfg.get("patience", 10),
        ckpt_path=cfg.get("ckpt_path", "checkpoints/bev_best.pt"),
        mode="min",
        min_delta=1e-3,
    )

    for epoch in range(cfg["epochs"]):
        train_log = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, grad_clip)
        val_log = val_one_epoch(model, val_loader, loss_fn, device)
        scheduler.step()
        print(
            f"Epoch {epoch+1}/{cfg['epochs']} | "
            f"train: {train_log['loss']:.4f} (hm {train_log['hm_loss']:.3f} reg {train_log['reg_loss']:.3f}) | "
            f"val: {val_log['loss']:.4f}"
        )
        early_stop(val_log["loss"], model)
        if early_stop.should_stop:
            print(f"Early stopping triggered. Best val loss: {early_stop.best:.4f}")
            break

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
