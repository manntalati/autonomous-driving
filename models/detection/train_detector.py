from models.detection.detector import FPNDetector
from models.detection.fpn import FPN
from models.backbone.resnet import ResNetBackbone
from models.detection.anchors import AnchorGenerator
from models.detection.head import DetectionHead
from models.detection.losses import DetectionLoss
from training.scheduler import build_optimizer, build_scheduler
from data.dataloader import get_loaders
from torch.cuda.amp import GradScaler
import torch
import numpy as np
import yaml
from training.scheduler import EarlyStopping

def build_detector(cfg: dict) -> FPNDetector:
    """
    Load Phase 1 backbone (optionally pretrained), attach FPN + head, return full detector.
    """
    resnet = ResNetBackbone(cfg["num_classes"])
    fpn = FPN(in_channels=[512, 1024, 2048], out_channels=256)
    anchor = AnchorGenerator(cfg["scales"], cfg["aspect_ratios"], cfg["strides"])
    detection = DetectionHead(256, cfg["num_anchors"], cfg["num_classes"])
    return FPNDetector(resnet, fpn, detection, anchor, cfg["num_classes"])


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler) -> dict:
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        images, gt_boxes, gt_labels = batch
        images = images.to(device)
        gt_boxes = [b.to(device) for b in gt_boxes]
        gt_labels = [l.to(device) for l in gt_labels]
        with torch.autocast(device_type="cuda"):
            cls_logits, bbox_deltas, anchors = model(images)
            loss, log = loss_fn(cls_logits, bbox_deltas, anchors, gt_boxes, gt_labels)
        total_loss += log["loss"]
        num_batches += 1    
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    return {"loss": total_loss / num_batches}

def val_one_epoch(model, loader, loss_fn, device) -> dict:
    model.train()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            images, gt_boxes, gt_labels = batch
            images = images.to(device)
            gt_boxes = [b.to(device) for b in gt_boxes]
            gt_labels = [l.to(device) for l in gt_labels]
            cls_logits, bbox_deltas, anchors = model(images)
            loss, log = loss_fn(cls_logits, bbox_deltas, anchors, gt_boxes, gt_labels)
            total_loss += log["loss"]
            num_batches += 1
    return {"loss": total_loss / max(num_batches, 1)}
            

def main(cfg_path: str) -> None:
    """
    Load config → build model → get_loaders() → train loop with mAP eval each epoch.
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loaders(data_root=cfg["data_root"], batch_size=cfg["batch_size"])
    model = build_detector(cfg)
    model = model.to(device)
    optimizer = build_optimizer(model, optimizer_type=cfg.get("optimizer", "adamw"), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = build_scheduler(optimizer, scheduler_type=cfg.get("scheduler", "cosine"), epochs=cfg["epochs"])
    loss = DetectionLoss(cfg["num_classes"])
    scaler = GradScaler()
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
