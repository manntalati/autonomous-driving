from __future__ import annotations
from pathlib import Path
import torch
import yaml
from contextlib import nullcontext
from torch.amp import GradScaler
from evaluation.seg_metrics import ConfusionMatrixMeter
from models.backbone.resnet import ResNetBackbone
from models.backbone.hybrid import HybridCNNViT
from models.segmentation.unet import UNet
from models.segmentation.losses import SegmentationLoss
from training.scheduler import build_scheduler, EarlyStopping
from data.dataloader import get_seg_loaders


def build_segmenter(cfg: dict) -> UNet:
    """
    Build the U-Net segmenter. The backbone is selected by cfg["backbone"]:
      "resnet" (default) — Phase 3 ResNet-18 backbone.
      "hybrid"           — Phase 4 CNN-ViT backbone.
    Both return (C3, C4, C5), so the U-Net decoder is identical either way.
    """
    backbone_type = cfg.get("backbone", "resnet")
    if backbone_type == "hybrid":
        backbone = HybridCNNViT(
            embed_dim=cfg.get("embed_dim", 512),
            depth=cfg.get("vit_depth", 4),
            num_heads=cfg.get("vit_heads", 8),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            image_size=tuple(cfg.get("image_size", [448, 800])),
        )
    else:
        backbone = ResNetBackbone()
    if cfg.get("pretrained", False):
        backbone.load_pretrained()
    return UNet(backbone, num_classes=cfg["num_classes"])


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_optimizer(model: UNet, cfg: dict) -> torch.optim.Optimizer:
    """
    Differential LR: backbone gets backbone_lr_scale × lr (default 0.1×); decoder + classifier get full lr.
    Same pattern as detection.
    """
    lr = cfg["lr"]
    backbone_scale = cfg.get("backbone_lr_scale", 0.1)
    wd = cfg.get("weight_decay", 1e-4)
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * backbone_scale},
            {"params": other_params,    "lr": lr},
        ],
        weight_decay=wd,
    )


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, grad_clip: float = 1.0) -> dict:
    """
    Standard AMP training loop. For each batch:
      1. Unpack (images, masks) → device.
      2. Forward → logits.
      3. Loss → backward → step (with grad clip).
    Returns: dict with avg 'loss', 'ce_loss', 'dice_loss' over batches.
    """
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    num_batches = 0
    amp_enabled = device.type == "cuda"
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        amp_ctx = torch.autocast(device_type="cuda") if amp_enabled else nullcontext()
        with amp_ctx:
            logits = model(images)
            loss, log = loss_fn(logits, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += log["loss"]
        total_ce += log["ce_loss"]
        total_dice += log["dice_loss"]
        num_batches += 1
    n = max(num_batches, 1)
    return {"loss": total_loss / n, "ce_loss": total_ce / n, "dice_loss": total_dice / n}


def val_one_epoch(model, loader, loss_fn, device, num_classes: int) -> dict:
    """
    Eval loop:
      1. model.eval(), no_grad.
      2. For each batch: forward → loss (for logging) → argmax preds → meter.update(preds, masks).
      3. Compute mIoU + per-class IoU after the loop.
    Returns: dict with 'loss', 'mIoU', 'IoU' (per-class array).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    meter = ConfusionMatrixMeter(num_classes, ignore_index=loss_fn.ignore_index)
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            _, log = loss_fn(logits, masks)
            total_loss += log["loss"]
            num_batches += 1

            preds = logits.argmax(dim=1)
            meter.update(preds, masks)

    return {
        "loss": total_loss / max(num_batches, 1),
        "mIoU": meter.miou(),
        "IoU": meter.iou_per_class(),
    }


def main(cfg_path: str) -> None:
    """
    Mirror of detection's main(): load yaml → build → train loop with mIoU eval each epoch.
    Checkpoints to checkpoints/segmenter_best.pt on mIoU improvement.
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = _pick_device()
    train_loader, val_loader = get_seg_loaders(
        data_root=cfg["data_root"],
        mask_dir=cfg.get("mask_dir"),   # None → {data_root}/seg_masks
        batch_size=cfg["batch_size"],
    )
    model = build_segmenter(cfg).to(device)
    optimizer = _build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, scheduler_type=cfg.get("scheduler", "cosine"), epochs=cfg["epochs"])
    loss_fn = SegmentationLoss(cfg["num_classes"])
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    grad_clip = cfg.get("grad_clip", 1.0)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    early_stop = EarlyStopping(
        patience=cfg.get("patience", 10),
        ckpt_path=cfg.get("ckpt_path", "checkpoints/segmenter_best.pt"),
        mode="max",
        min_delta=1e-3,
    )

    for epoch in range(cfg["epochs"]):
        train_log = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, grad_clip)
        val_log = val_one_epoch(model, val_loader, loss_fn, device, cfg["num_classes"])
        scheduler.step()
        per_cls = " ".join(f"{iou:.3f}" for iou in val_log["IoU"])
        print(
            f"Epoch {epoch+1}/{cfg['epochs']} | "
            f"train: {train_log['loss']:.4f} | "
            f"val: {val_log['loss']:.4f} | "
            f"mIoU: {val_log['mIoU']:.3f} [{per_cls}]"
        )
        early_stop(val_log["mIoU"], model)
        if early_stop.should_stop:
            print(f"Early stopping triggered. Best mIoU: {early_stop.best:.3f}")
            break


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
