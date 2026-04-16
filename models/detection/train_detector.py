from models.detection.detector import FPNDetector
from models.detection.fpn import FPN
from models.backbone.resnet import ResNetBackbone
from models.detection.anchors import AnchorGenerator
from models.detection.head import DetectionHead
from models.detection.losses import DetectionLoss
from training.scheduler import build_optimizer, build_scheduler, EarlyStopping
from data.dataloader import get_loaders
from torch.amp import GradScaler
from contextlib import nullcontext
from pathlib import Path
import torch
import yaml


def build_detector(cfg: dict) -> FPNDetector:
    """
    Load Phase 1 backbone (optionally pretrained), attach FPN + head, return full detector.
    Backbone produces C3/C4/C5 at 128/256/512 channels (see models/backbone/resnet.py).
    """
    resnet = ResNetBackbone()
    fpn = FPN(in_channels=[128, 256, 512], out_channels=256)
    anchor = AnchorGenerator(cfg["scales"], cfg["aspect_ratios"], cfg["strides"])
    detection = DetectionHead(256, cfg["num_anchors"], cfg["num_classes"])
    return FPNDetector(resnet, fpn, detection, anchor, cfg["num_classes"])


def _unpack_batch(batch, device):
    images, targets = batch
    images = images.to(device)
    gt_boxes = [t["boxes"].to(device) for t in targets]
    gt_labels = [t["labels"].to(device) for t in targets]
    return images, gt_boxes, gt_labels


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler) -> dict:
    model.train()
    total_loss = 0.0
    num_batches = 0
    amp_enabled = device.type == "cuda"
    for batch in loader:
        images, gt_boxes, gt_labels = _unpack_batch(batch, device)
        optimizer.zero_grad()
        amp_ctx = torch.autocast(device_type="cuda") if amp_enabled else nullcontext()
        with amp_ctx:
            cls_logits, bbox_deltas, anchors = model(images)
            loss, log = loss_fn(cls_logits, bbox_deltas, anchors, gt_boxes, gt_labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += log["loss"]
        num_batches += 1
    return {"loss": total_loss / max(num_batches, 1)}


def val_one_epoch(model, loader, loss_fn, device) -> dict:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            images, gt_boxes, gt_labels = _unpack_batch(batch, device)
            cls_logits, bbox_deltas, anchors = model(images, return_raw=True)
            loss, log = loss_fn(cls_logits, bbox_deltas, anchors, gt_boxes, gt_labels)
            total_loss += log["loss"]
            num_batches += 1
    return {"loss": total_loss / max(num_batches, 1)}


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(cfg_path: str) -> None:
    """
    Load config → build model → get_loaders() → train loop with mAP eval each epoch.
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = _pick_device()
    train_loader, val_loader = get_loaders(data_root=cfg["data_root"], batch_size=cfg["batch_size"])
    model = build_detector(cfg).to(device)
    optimizer = build_optimizer(model, optimizer_type=cfg.get("optimizer", "adamw"), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = build_scheduler(optimizer, scheduler_type=cfg.get("scheduler", "cosine"), epochs=cfg["epochs"])
    loss = DetectionLoss(cfg["num_classes"])
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    early_stop = EarlyStopping(patience=5, ckpt_path="checkpoints/detector_best.pt")
    for epoch in range(cfg["epochs"]):
        train_log = train_one_epoch(model, train_loader, optimizer, loss, device, scaler)
        val_log = val_one_epoch(model, val_loader, loss, device)
        scheduler.step()
        print(f"Epoch {epoch+1}/{cfg['epochs']} | train: {train_log['loss']:.4f} | val: {val_log['loss']:.4f}")
        early_stop(val_log["loss"], model)
        if early_stop.should_stop:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
