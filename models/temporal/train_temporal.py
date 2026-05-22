"""
Phase 6 training script — 3-frame temporal-fusion detector.

Mirrors models/detection/train_detector.py: yaml config → build → epoch loop
with per-epoch mAP eval and early stopping.
"""
from __future__ import annotations
from pathlib import Path
from contextlib import nullcontext
import torch
import yaml
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes

from data.sequence_dataset import NuScenesSequenceDataset, sequence_collate_fn
from data.dataset import version_from_data_root
from models.backbone.resnet import ResNetBackbone
from models.detection.fpn import FPN
from models.detection.head import DetectionHead
from models.detection.anchors import AnchorGenerator
from models.detection.detector import FPNDetector
from models.detection.losses import DetectionLoss
from models.detection.map import compute_map
from models.temporal.temporal_attention import TemporalCrossAttention
from models.temporal.temporal_detector import TemporalDetector
from training.scheduler import build_scheduler, EarlyStopping

NUM_WORKERS = 2
_C5_CHANNELS = 512  # ResNet backbone C5 channel count — what temporal attention fuses


def build_temporal_detector(cfg: dict) -> TemporalDetector:
    """Build a Phase 2 FPNDetector + temporal cross-attention on C5."""
    resnet = ResNetBackbone()
    if cfg.get("pretrained", False):
        resnet.load_pretrained()
    fpn = FPN(in_channels=[128, 256, 512], out_channels=256)
    anchor = AnchorGenerator(cfg["scales"], cfg["aspect_ratios"], cfg["strides"])
    head = DetectionHead(256, cfg["num_anchors"], cfg["num_classes"])
    detector = FPNDetector(resnet, fpn, head, anchor, cfg["num_classes"])
    temporal_attn = TemporalCrossAttention(
        embed_dim=_C5_CHANNELS,
        num_heads=cfg.get("temporal_heads", 8),
        num_past_frames=cfg.get("seq_len", 3) - 1,
    )
    return TemporalDetector(detector, temporal_attn)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_optimizer(model: TemporalDetector, cfg: dict) -> torch.optim.Optimizer:
    """Differential LR: the shared backbone gets backbone_lr_scale × lr; everything
    else (temporal attention, FPN, head) gets the full lr."""
    lr = cfg["lr"]
    backbone_scale = cfg.get("backbone_lr_scale", 0.1)
    wd = cfg.get("weight_decay", 1e-4)
    backbone_params = list(model.detector.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * backbone_scale},
            {"params": other_params, "lr": lr},
        ],
        weight_decay=wd,
    )


def get_temporal_loaders(cfg: dict):
    """Build train/val DataLoaders of frame-window sequences."""
    nusc = NuScenes(version=version_from_data_root(cfg["data_root"]), dataroot=cfg["data_root"], verbose=False)
    seq_len = cfg.get("seq_len", 3)
    train_ds = NuScenesSequenceDataset(nusc, cfg["data_root"], split="train", seq_len=seq_len)
    val_ds = NuScenesSequenceDataset(nusc, cfg["data_root"], split="val", seq_len=seq_len)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=sequence_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=sequence_collate_fn, pin_memory=True)
    return train_loader, val_loader


def _unpack_batch(batch, device):
    """(frames, targets) → frames on device, gt_boxes list, gt_labels list."""
    frames, targets = batch
    frames = frames.to(device)
    gt_boxes = [t["boxes"].to(device) for t in targets]
    gt_labels = [t["labels"].to(device) for t in targets]
    return frames, gt_boxes, gt_labels


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, grad_clip: float = 1.0) -> dict:
    """
    One AMP training epoch. For each batch:
      1. _unpack_batch → frames (B,T,3,H,W), gt_boxes, gt_labels.
      2. cls_logits, bbox_deltas, anchors = model(frames).
      3. loss, log = loss_fn(cls_logits, bbox_deltas, anchors, gt_boxes, gt_labels);
         backward; step (with grad clip).
    Returns: dict with avg 'loss' (and 'pos_per_img').
    """
    model.train()
    total_loss = 0.0
    total_pos = 0
    num_batches = 0
    num_frames = 0
    amp_enabled = device.type == "cuda"
    for batch in loader:
        frames, gt_boxes, gt_labels = _unpack_batch(batch, device)
        optimizer.zero_grad()
        amp_ctx = torch.autocast(device_type="cuda") if amp_enabled else nullcontext()
        with amp_ctx:
            cls_logits, bbox_deltas, anchors = model(frames)
            loss, log = loss_fn(cls_logits, bbox_deltas, anchors, gt_boxes, gt_labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += log["loss"]
        total_pos += log["num_pos"]
        num_frames += frames.shape[0]
        num_batches += 1
    return {
        "loss": total_loss / max(num_batches, 1),
        "pos_per_img": total_pos / max(num_frames, 1),
    }


def val_one_epoch(model, loader, loss_fn, device, num_classes: int) -> dict:
    """
    Eval epoch: model.eval(); per batch get raw outputs (return_raw=True) for the
    val loss, AND postprocess the same outputs to collect predictions for mAP.
    Returns: dict with 'loss', 'mAP', 'AP' (per-class).
    """
    model.eval()
    total_loss = 0.0
    total_pos = 0
    num_images = 0
    num_batches = 0
    predictions: list = []
    ground_truths: list = []
    with torch.no_grad():
        for batch in loader:
            images, gt_boxes, gt_labels = _unpack_batch(batch, device)
            cls_logits, bbox_deltas, anchors = model(images, return_raw=True)
            _, log = loss_fn(cls_logits, bbox_deltas, anchors, gt_boxes, gt_labels)
            total_loss += log["loss"]
            total_pos += log["num_pos"]
            num_images += images.shape[0]
            num_batches += 1
            image_size = (images.shape[-2], images.shape[-1])
            pred_boxes, pred_scores, pred_labels = model.detector.postprocess(
                cls_logits, bbox_deltas, anchors, image_size
            )
            for b in range(images.shape[0]):
                predictions.append({
                    "boxes":  pred_boxes[b].detach().cpu().numpy().tolist(),
                    "scores": pred_scores[b].detach().cpu().numpy().tolist(),
                    "labels": pred_labels[b].detach().cpu().numpy().tolist(),
                })
                ground_truths.append({
                    "boxes":  gt_boxes[b].detach().cpu().numpy().tolist(),
                    "labels": gt_labels[b].detach().cpu().numpy().tolist(),
                })

    mean_ap, per_class_ap = compute_map(predictions, ground_truths, num_classes)
    return {
        "loss": total_loss / max(num_batches, 1),
        "pos_per_img": total_pos / max(num_images, 1),
        "mAP": mean_ap,
        "AP": per_class_ap,
    }


def main(cfg_path: str) -> None:
    """Load config → build → epoch loop with per-epoch mAP eval and early stopping."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = _pick_device()
    train_loader, val_loader = get_temporal_loaders(cfg)
    model = build_temporal_detector(cfg).to(device)
    optimizer = _build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, scheduler_type=cfg.get("scheduler", "cosine"), epochs=cfg["epochs"])
    loss_fn = DetectionLoss(cfg["num_classes"])
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    grad_clip = cfg.get("grad_clip", 1.0)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    early_stop = EarlyStopping(
        patience=cfg.get("patience", 10),
        ckpt_path=cfg.get("ckpt_path", "checkpoints/temporal_best.pt"),
        mode="max",
        min_delta=1e-3,
    )

    for epoch in range(cfg["epochs"]):
        train_log = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, grad_clip)
        val_log = val_one_epoch(model, val_loader, loss_fn, device, cfg["num_classes"])
        scheduler.step()
        per_cls = " ".join(f"{ap:.3f}" for ap in val_log["AP"])
        print(
            f"Epoch {epoch+1}/{cfg['epochs']} | "
            f"train: {train_log['loss']:.4f} | val: {val_log['loss']:.4f} | "
            f"mAP: {val_log['mAP']:.3f} [{per_cls}]"
        )
        early_stop(val_log["mAP"], model)
        if early_stop.should_stop:
            print(f"Early stopping triggered. Best mAP: {early_stop.best:.3f}")
            break


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
