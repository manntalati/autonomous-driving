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
from data.dataset import version_from_data_root
from models.bev.bev_detector import BEVDetector
from models.bev.losses import BEVLoss
from evaluation.seg_metrics import ConfusionMatrixMeter
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
        num_seg_classes=cfg.get("num_seg_classes", 5),
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
    nusc = NuScenes(version=version_from_data_root(cfg["data_root"]), dataroot=cfg["data_root"], verbose=False)
    ds_kw = dict(
        image_size=tuple(cfg.get("image_size", [448, 800])),
        xbound=tuple(cfg["xbound"]),
        ybound=tuple(cfg["ybound"]),
        dbound=tuple(cfg["dbound"]),
        cameras=cfg.get("cameras"),   # None → dataset defaults to CAM_FRONT only
    )
    train_ds = NuScenesBEVDataset(nusc, cfg["data_root"], split="train", **ds_kw)
    val_ds = NuScenesBEVDataset(nusc, cfg["data_root"], split="val", **ds_kw)
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

_KEYS = ("loss", "hm_loss", "reg_loss", "seg_loss", "depth_loss")


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, grad_clip: float = 1.0) -> dict:
    """
    One AMP training epoch: forward the BEV detector, compute the combined
    detection + BEV-segmentation loss, backward, step (with grad clip).
    Returns: dict of averaged 'loss', 'hm_loss', 'reg_loss', 'seg_loss'.
    """
    model.train()
    total = {k: 0.0 for k in _KEYS}
    num_batches = 0
    amp_enabled = device.type == "cuda"
    for images, targets in loader:
        images = images.to(device)
        intrinsics, cam_to_ego = _stack_calibration(targets, device)
        optimizer.zero_grad()
        amp_ctx = torch.autocast(device_type="cuda") if amp_enabled else nullcontext()
        with amp_ctx:
            outputs = model(images, intrinsics, cam_to_ego)
            loss, log = loss_fn(outputs, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        for k in _KEYS:
            total[k] += log[k]
        num_batches += 1
    n = max(num_batches, 1)
    return {k: total[k] / n for k in _KEYS}

def val_one_epoch(model, loader, loss_fn, device, num_seg_classes: int) -> dict:
    """
    One eval epoch: accumulate loss components and the BEV-segmentation mIoU
    (argmax of the BEV seg head vs the rasterized map target).
    Returns: dict of averaged loss components plus 'mIoU'.
    """
    model.eval()
    total = {k: 0.0 for k in _KEYS}
    num_batches = 0
    meter = ConfusionMatrixMeter(num_seg_classes)
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            intrinsics, cam_to_ego = _stack_calibration(targets, device)
            outputs = model(images, intrinsics, cam_to_ego)
            _, log = loss_fn(outputs, targets)
            for k in _KEYS:
                total[k] += log[k]
            num_batches += 1
            preds = outputs["seg"].argmax(dim=1)
            seg_tgt = torch.stack([t["bev_seg"] for t in targets]).to(device)
            meter.update(preds, seg_tgt)
    n = max(num_batches, 1)
    out = {k: total[k] / n for k in _KEYS}
    out["mIoU"] = meter.miou()
    return out

def main(cfg_path: str) -> None:
    """Load config → build model/optimizer/scheduler/loss → epoch loop with early stopping."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = _pick_device()
    train_loader, val_loader = get_bev_loaders(cfg)
    model = build_bev_detector(cfg).to(device)
    optimizer = _build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, scheduler_type=cfg.get("scheduler", "cosine"), epochs=cfg["epochs"])
    num_seg_classes = cfg.get("num_seg_classes", 5)
    loss_fn = BEVLoss(
        num_classes=cfg["num_classes"],
        xbound=tuple(cfg["xbound"]),
        ybound=tuple(cfg["ybound"]),
        num_seg_classes=num_seg_classes,
        seg_weight=cfg.get("seg_weight", 1.0),
        depth_weight=cfg.get("depth_weight", 1.0),
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    grad_clip = cfg.get("grad_clip", 1.0)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    # early-stop on BEV mIoU (max): the combined val loss is dominated by the
    # CenterNet heatmap term, which overfits immediately — mIoU is the signal
    # that actually tracks BEV scene-understanding quality.
    early_stop = EarlyStopping(
        patience=cfg.get("patience", 10),
        ckpt_path=cfg.get("ckpt_path", "checkpoints/bev_best.pt"),
        mode="max",
        min_delta=1e-3,
    )

    for epoch in range(cfg["epochs"]):
        train_log = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, grad_clip)
        val_log = val_one_epoch(model, val_loader, loss_fn, device, num_seg_classes)
        scheduler.step()
        print(
            f"Epoch {epoch+1}/{cfg['epochs']} | "
            f"train {train_log['loss']:.3f} (hm {train_log['hm_loss']:.2f} "
            f"reg {train_log['reg_loss']:.2f} seg {train_log['seg_loss']:.3f} "
            f"depth {train_log['depth_loss']:.3f}) | "
            f"val {val_log['loss']:.3f} | BEV mIoU {val_log['mIoU']:.3f}"
        )
        early_stop(val_log["mIoU"], model)
        if early_stop.should_stop:
            print(f"Early stopping triggered. Best BEV mIoU: {early_stop.best:.4f}")
            break

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
