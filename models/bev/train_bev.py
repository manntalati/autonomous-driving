"""
Phase 5 training script — CAM_FRONT Lift-Splat-Shoot BEV detector.

Mirrors models/segmentation/train_seg.py: yaml config → build → epoch loop
with early stopping on validation loss.
"""
from __future__ import annotations
from pathlib import Path
from contextlib import nullcontext
import numpy as np
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
        use_radar=cfg.get("use_radar", False),
        radar_in_channels=cfg.get("radar_in_channels", 5),
        fusion_mode=cfg.get("fusion_mode", "gated"),
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
        use_radar=cfg.get("use_radar", False),
        radar_channels=cfg.get("radar_channels"),
        radar_dilate=cfg.get("radar_dilate", 1),
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


def _stack_radar(targets: list, device: torch.device):
    """Batched radar BEV grid (B, 5, X, Y), or None when radar is not in use."""
    if not targets or "radar_bev" not in targets[0]:
        return None
    return torch.stack([t["radar_bev"] for t in targets]).to(device)

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
        radar_bev = _stack_radar(targets, device)
        optimizer.zero_grad()
        amp_ctx = torch.autocast(device_type="cuda") if amp_enabled else nullcontext()
        with amp_ctx:
            outputs = model(images, intrinsics, cam_to_ego, radar_bev=radar_bev)
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
    gates: list[float] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            intrinsics, cam_to_ego = _stack_calibration(targets, device)
            radar_bev = _stack_radar(targets, device)
            outputs = model(images, intrinsics, cam_to_ego, radar_bev=radar_bev)
            _, log = loss_fn(outputs, targets)
            for k in _KEYS:
                total[k] += log[k]
            num_batches += 1
            if "gate" in outputs:
                gates.append(outputs["gate"])
            preds = outputs["seg"].argmax(dim=1)
            seg_tgt = torch.stack([t["bev_seg"] for t in targets]).to(device)
            meter.update(preds, seg_tgt)
    n = max(num_batches, 1)
    out = {k: total[k] / n for k in _KEYS}
    out["mIoU"] = meter.miou()
    if gates:
        # Mean fusion gate: ~1 means the model is leaning on the camera, ~0 on
        # radar. Compared across day and night splits this is the Phase 10 claim.
        out["gate"] = sum(gates) / len(gates)
    return out

def set_seed(seed: int) -> None:
    """
    Seed torch/numpy/python so a run is reproducible and, for the Phase 10
    ablation, so both arms see the SAME data order.

    This matters more than it looks. Two runs of the identical camera-only config
    produced epoch-1 mIoU of 0.124 and 0.185 — a 50% swing from data order and
    init alone. Unseeded, that variance sits on top of the radar effect the
    ablation is trying to measure.

    Caveat that seeding does NOT fix: building the radar branch consumes extra
    RNG draws, so the shared modules constructed after it (BEVEncoder, heads) get
    different initial weights in the radar arm than in the camera arm. That
    residual asymmetry is unavoidable without re-seeding mid-construction, and is
    one reason a single run per arm can only support a LARGE effect — see the
    n=1 caveat in evaluation/radar_ablation.py.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg_path: str) -> None:
    """Load config → build model/optimizer/scheduler/loss → epoch loop with early stopping."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = _pick_device()
    set_seed(cfg.get("seed", 0))
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

    # FIXED-EPOCH MODE (Phase 10 ablation) — see configs/bev_*_p10.yaml.
    #
    # Early stopping on BEV mIoU is unusable for a controlled ablation. Measured
    # over the camera-only arm, mIoU per epoch was:
    #     .124 .186 .120 .160 .142 .124 .136 .184 .139 .145 .154 .131
    # Range 0.12-0.19, no trend, epochs 2 and 8 tied within 0.002 — the metric is
    # noise-dominated at this dataset size, so "best epoch" is effectively a coin
    # flip. Two arms selected that way land on arbitrary and probably different
    # epochs, and their mAP difference then reflects training duration as much as
    # radar. A symmetric CRITERION does not give symmetric MATURITY.
    #
    # With early_stop: false, both arms train for exactly `epochs` and the ablation
    # compares the final checkpoints, which are matched by construction.
    use_early_stop = cfg.get("early_stop", True)
    last_path = cfg.get("last_ckpt_path")
    if not use_early_stop:
        print(f"Fixed-epoch mode: {cfg['epochs']} epochs, no early stopping.")

    for epoch in range(cfg["epochs"]):
        train_log = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, grad_clip)
        val_log = val_one_epoch(model, val_loader, loss_fn, device, num_seg_classes)
        scheduler.step()
        gate = f" | gate {val_log['gate']:.3f}" if "gate" in val_log else ""
        print(
            f"Epoch {epoch+1}/{cfg['epochs']} | "
            f"train {train_log['loss']:.3f} (hm {train_log['hm_loss']:.2f} "
            f"reg {train_log['reg_loss']:.2f} seg {train_log['seg_loss']:.3f} "
            f"depth {train_log['depth_loss']:.3f}) | "
            f"val {val_log['loss']:.3f} | BEV mIoU {val_log['mIoU']:.3f}{gate}"
        )
        # Always track the best-by-mIoU checkpoint (harmless, and other phases
        # rely on it); only let it TERMINATE the run when early stopping is on.
        early_stop(val_log["mIoU"], model)
        if last_path:
            torch.save(model.state_dict(), last_path)
        if use_early_stop and early_stop.should_stop:
            print(f"Early stopping triggered. Best BEV mIoU: {early_stop.best:.4f}")
            break

    if not use_early_stop and last_path:
        print(f"Final checkpoint (epoch {cfg['epochs']}) -> {last_path}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
